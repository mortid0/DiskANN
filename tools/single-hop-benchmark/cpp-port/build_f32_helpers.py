#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Link new Linux float32 benchmark helpers against the existing pinned C++ library.

Never rebuilds the core library or edits the historical checkout/build. Compiler
and linker settings come from one existing CMake target, or an explicit JSON
settings file. --plan-only validates discovery on any host; native execution is
restricted to Linux x86-64 and includes the real sparse 4-GiB extraction test.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import time

CPP_COMMIT = "78256bbab4685e1774e78d331e081a153be26823"
SOURCE_ARCHIVE_SHA256 = "ae0a76746560559e3850b9f6c5a03ed904999becaf0e51a06f8492d7cd7ac1cb"
TARGETS = {"build_stitched_f32": "bin/build_stitched_f32",
           "generate_pq_f32": "bin/utils/generate_pq_f32",
           "create_disk_layout": "bin/utils/create_disk_layout",
           "test_stitched_f32_large_offset": "bin/test_stitched_f32_large_offset"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    require(path.is_file(), f"Expected input file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def save(path, value):
    with Path(path).open("x") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")


def validate_tokens(tokens):
    require(isinstance(tokens, (list, tuple)), "Compiler arguments must be an array, not shell text")
    require(all(isinstance(x, str) and x and "\n" not in x and "\0" not in x for x in tokens), "Invalid compiler argument")
    require(not any(x in (";", "&&", "||", "|", ">", ">>", "<") for x in tokens), "Shell operations are unsupported; provide compiler arguments only")
    require(not any("$(" in x or "${" in x or x.startswith("@") for x in tokens), "Unresolved Make variables/response files require explicit JSON settings")
    return tokens


def parse_flags_make(path):
    text = Path(path).read_text().replace("\\\n", " ")
    result = []
    for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        found = re.findall(r"^" + name + r"\s*=\s*(.*)$", text, flags=re.MULTILINE)
        require(len(found) == 1, f"Expected one {name} assignment in {path}")
        result.extend(shlex.split(found[0]))
    return validate_tokens(result)


def parse_link_command(text):
    tokens = validate_tokens(shlex.split(text.strip()))
    require(tokens and not tokens[0].startswith("-"), "Missing C++ compiler")
    compiler, arguments, output, removed_objects = tokens[0], [], None, []
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "-o":
            require(output is None and i+1 < len(tokens), "Invalid/multiple link output")
            output = tokens[i+1]
            i += 2
        elif token.endswith((".o", ".obj")):
            removed_objects.append(token)
            i += 1
        else:
            require(not token.endswith((".cpp", ".cc", ".cxx", ".c")), "Unexpected source in link command")
            arguments.append(token)
            i += 1
    require(output is not None and removed_objects, "Link command must contain an object and output")
    return {"compiler": compiler, "link_flags": arguments, "reference_output": output,
            "discarded_objects": removed_objects}


def absolute_flags(tokens, working_directory):
    """Resolve paths while preserving the existing link order and math/OpenMP flags."""
    result, i = [], 0
    prefixes = ("-I", "-L")
    paired = ("-I", "-L", "-isystem", "-iquote", "-include", "-imacros", "-isysroot")
    while i < len(tokens):
        token = tokens[i]
        if token in paired:
            require(i+1 < len(tokens), f"Missing path after {token}")
            path = Path(tokens[i+1])
            result.extend((token, str((working_directory/path).resolve() if not path.is_absolute() else path)))
            i += 2
        elif token.startswith(prefixes) and len(token) > 2:
            path = Path(token[2:])
            result.append(token[:2] + str((working_directory/path).resolve() if not path.is_absolute() else path))
            i += 1
        elif not token.startswith("-") and re.search(r"\.(a|so)(\.\d+)*$", token):
            path = Path(token)
            result.append(str((working_directory/path).resolve() if not path.is_absolute() else path))
            i += 1
        else:
            result.append(token)
            i += 1
    return validate_tokens(result)


def discover_settings(native_build, target):
    require(Path(target).name == target, "Reference target must be a basename")
    matches = list(native_build.rglob(f"CMakeFiles/{target}.dir/link.txt"))
    require(len(matches) == 1,
            f"Expected one link.txt for {target}, found {len(matches)}; use --settings-json with explicit compiler/compile_flags/link_flags/working_directory")
    link = matches[0]
    flags = link.with_name("flags.make")
    require(flags.is_file(), "Missing flags.make; use --settings-json for Ninja/other generators")
    parsed = parse_link_command(link.read_text())
    working = link.parent.parent.parent
    return {**parsed, "compile_flags": parse_flags_make(flags), "working_directory": str(working),
            "discovery_inputs": [record(link), record(flags)]}


def libraries(flags):
    search_paths, direct, names = [], [], []
    for i, token in enumerate(flags):
        if token == "-L":
            search_paths.append(Path(flags[i+1]))
        elif token.startswith("-L"):
            search_paths.append(Path(token[2:]))
        elif token.startswith("-l") and len(token) > 2:
            names.append(token[2:])
        elif Path(token).is_absolute() and re.search(r"\.(a|so)(\.\d+)*$", token):
            direct.append(Path(token))
    unresolved = []
    for name in names:
        candidates = [directory/f"lib{name}{suffix}" for directory in search_paths for suffix in (".so", ".a")]
        match = next((path for path in candidates if path.is_file()), None)
        if match is None:
            unresolved.append(name)
        else:
            direct.append(match)
    return sorted(set(path.resolve() for path in direct)), search_paths, unresolved


def verify_upstream(source, archive, evidence, expected_archive):
    require(sha256(archive) == expected_archive, "Pinned C++ source archive SHA-256 mismatch")
    native = json.loads(evidence.read_text())
    require(native.get("source_commit") == CPP_COMMIT and native.get("source_archive_sha256") == expected_archive,
            "Native build evidence does not bind the pinned C++ source commit/archive")
    require(native.get("native_compiled") is True and native.get("machine") == "x86_64", "Evidence must describe an actual native x86-64 build")
    patched = native.get("patched_source_sha256", {})
    checked = {}
    for name, expected in patched.items():
        path = source / name
        require(path.is_file() and sha256(path) == expected, f"Historical native patched source changed: {name}")
        checked[name] = record(path)
    with tarfile.open(archive) as upstream:
        for member in upstream.getmembers():
            name = str(Path(member.name))
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe upstream archive member")
            if not member.isfile() or name in patched:
                continue
            if not (name.endswith((".h", ".hpp", ".cpp", ".cmake")) or Path(name).name == "CMakeLists.txt"):
                continue
            path = source / name
            require(path.is_file(), f"Pinned source file missing: {name}")
            data = upstream.extractfile(member).read()
            expected = hashlib.sha256(data).hexdigest()
            require(sha256(path) == expected, f"Pinned source differs from archive: {name}")
            checked[name] = record(path)
    require("apps/utils/create_disk_layout.cpp" in checked and any(name.startswith("include/") for name in checked),
            "Archive lacks the pinned upstream layout source/headers")
    return {"commit": CPP_COMMIT, "archive": record(archive), "native_evidence": record(evidence),
            "verified_source_files": checked}


def effective_ndebug(flags):
    defined, i = False, 0
    while i < len(flags):
        token = flags[i]
        if token in ("-D", "-U"):
            require(i+1 < len(flags), f"Missing macro after {token}")
            operation, macro = token, flags[i+1]
            i += 2
        elif token.startswith(("-D", "-U")):
            operation, macro = token[:2], token[2:]
            i += 1
        else:
            i += 1
            continue
        if macro.split("=", 1)[0] == "NDEBUG":
            defined = operation == "-D"
    return defined


def verify_release_configuration(native_build, compile_flags, link_flags, linked_libraries):
    require(effective_ndebug([*compile_flags, *link_flags]),
            "New helper compilation must effectively define NDEBUG; zero-neighbor release layouts cannot use Debug assertions")
    cache_path = native_build/"CMakeCache.txt"
    build_type = None
    if cache_path.is_file():
        values = re.findall(r"^CMAKE_BUILD_TYPE:[^=]+=(.*)$", cache_path.read_text(), re.MULTILINE)
        require(len(values) <= 1, "Ambiguous CMAKE_BUILD_TYPE")
        build_type = values[0] if values else None
    proofs = []
    for library in linked_libraries:
        if not library.name.startswith("libdiskann"):
            continue
        target = re.split(r"\.(?:a|so)(?:\.|$)", library.name[len("lib"):], maxsplit=1)[0]
        matches = list(native_build.rglob(f"CMakeFiles/{target}.dir/flags.make"))
        require(len(matches) <= 1, f"Ambiguous core-library flags for {target}")
        if matches:
            flags = parse_flags_make(matches[0])
            require(effective_ndebug(flags), f"Existing core library {target} lacks effective NDEBUG")
            proofs.append({"library": record(library), "target": target, "flags": record(matches[0]),
                           "effective_ndebug": True, "proof": "actual_core_target_flags"})
        else:
            require(build_type in ("Release", "RelWithDebInfo"),
                    f"Core-library flags unavailable and CMAKE_BUILD_TYPE is not Release/RelWithDebInfo for {target}; supply the existing native target flags before building")
            proofs.append({"library": record(library), "target": target, "flags": None,
                           "proof": "existing_cmake_cache_release_configuration"})
    require(proofs, "Missing existing-core release evidence")
    return {"helper_effective_ndebug": True, "core_libraries": proofs,
            "cmake_build_type": build_type, "cmake_cache": record(cache_path) if cache_path.is_file() else None}


def runtime_environment(link_flags, linked_libraries, extra):
    paths = [str(path.parent) for path in linked_libraries]
    for i, token in enumerate(link_flags):
        if token == "-L":
            paths.append(link_flags[i+1])
        elif token.startswith("-L"):
            paths.append(token[2:])
        elif token.startswith("-Wl,"):
            parts = token[4:].split(",")
            for offset, part in enumerate(parts):
                if part in ("-rpath", "--rpath") and offset+1 < len(parts):
                    paths.extend(parts[offset+1].split(":"))
                elif part.startswith(("-rpath=", "--rpath=")):
                    paths.extend(part.split("=", 1)[1].split(":"))
    paths = [*map(str, extra), *paths, *os.environ.get("LD_LIBRARY_PATH", "").split(":")]
    paths = list(dict.fromkeys(path for path in paths if path))
    environment = os.environ.copy()
    environment.update(LD_LIBRARY_PATH=":".join(paths), OMP_TOOL="disabled", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return environment


def execute(command, log, environment, working_directory):
    start = time.monotonic()
    with log.open("x") as output:
        result = subprocess.run(command, cwd=working_directory, env=environment, stdout=output, stderr=subprocess.STDOUT)
    item = {"command": command, "working_directory": str(working_directory), "exit_code": result.returncode,
            "elapsed_seconds": time.monotonic()-start, "log": record(log)}
    require(result.returncode == 0, f"Command failed; see {log}")
    return item


def build(args):
    require(args.plan_only or (platform.system() == "Linux" and platform.machine() in ("x86_64", "amd64")),
            "Native helper execution requires Linux x86-64; --plan-only is available on other hosts")
    source, native_build, out = args.source_dir.resolve(), args.native_build_dir.resolve(), args.out.resolve()
    require(not out.exists(), "Output exists; historical artifacts are never overwritten")
    require(not out.is_relative_to(source) and not out.is_relative_to(native_build), "New output must be outside the historical source/build")
    upstream = verify_upstream(source, args.source_archive.resolve(), args.native_evidence.resolve(), args.expected_source_sha256)
    if args.settings_json:
        settings = json.loads(args.settings_json.read_text())
        require(all(key in settings for key in ("compiler", "compile_flags", "link_flags", "working_directory")), "Explicit settings are incomplete")
        settings["discovery_inputs"] = [record(args.settings_json)]
    else:
        settings = discover_settings(native_build, args.reference_target)
    working = Path(settings["working_directory"]).resolve()
    require(working.is_dir(), "Reference link working directory does not exist")
    compiler_path = shutil.which(settings["compiler"])
    require(compiler_path is not None, "Reference compiler unavailable; provide settings matching the native toolchain")
    compiler = str(Path(compiler_path).resolve())
    compile_flags = absolute_flags(validate_tokens(settings["compile_flags"]), working)
    link_flags = absolute_flags(validate_tokens(settings["link_flags"]), working)
    linked, _, unresolved = libraries(link_flags)
    require(any(path.name.startswith("libdiskann") and "reference" not in path.name for path in linked),
            "No native pinned libdiskann library found in the reference link settings")
    linked_records = [record(path) for path in linked]
    release_configuration = verify_release_configuration(native_build, compile_flags, link_flags, linked)
    own = Path(__file__).resolve().parent
    inputs = {"build_stitched_f32": own/"build_stitched_f32.cpp",
              "generate_pq_f32": own/"generate_pq_f32.cpp",
              "create_disk_layout": source/"apps/utils/create_disk_layout.cpp",
              "test_stitched_f32_large_offset": own/"test_stitched_f32_large_offset.cpp"}
    input_records = {name: record(path) for name, path in inputs.items()}
    environment = runtime_environment(link_flags, linked, args.runtime_library_dir)
    out.mkdir(parents=True)
    (out/"src").mkdir()
    for name, path in inputs.items():
        copied = out/"src"/(name+".cpp")
        shutil.copyfile(path, copied)
        require(sha256(copied) == input_records[name]["sha256"], "Helper input changed while snapshotting")
    commands = {}
    for name, relative in TARGETS.items():
        binary = out/relative
        binary.parent.mkdir(parents=True, exist_ok=True)
        commands[name] = [compiler, *compile_flags, str(out/"src"/(name+".cpp")), *link_flags, "-o", str(binary)]
    plan = {"schema_version": 1, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tool": record(Path(__file__)), "plan_only": args.plan_only,
            "host": {"system": platform.system(), "machine": platform.machine(), "release": platform.release()},
            "upstream": upstream, "compiler": record(compiler), "settings": settings,
            "cmake_cache": record(native_build/"CMakeCache.txt") if (native_build/"CMakeCache.txt").is_file() else None,
            "compile_flags": compile_flags, "link_flags": link_flags,
            "release_configuration": release_configuration,
            "linked_libraries": linked_records, "system_link_names_resolved_by_loader_after_build": unresolved,
            "helper_inputs": input_records, "commands": commands,
            "runtime_environment": {key: environment[key] for key in ("LD_LIBRARY_PATH", "OMP_TOOL", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
            "historical_source_or_build_modified": False,
            "scope": "Compile helper translation units only; core C++ library and search implementation reused unchanged"}
    save(out/"build-plan.json", plan)
    if args.plan_only:
        print(json.dumps({"plan": str(out/"build-plan.json"), "compiled": False}))
        return plan
    compiler_version = subprocess.check_output([compiler, "--version"], text=True)
    jobs, binaries, runtime = {}, {}, {}
    for name, command in commands.items():
        jobs[name] = execute(command, out/(name+".compile.log"), environment, working)
        binary = out/TARGETS[name]
        binaries[name] = record(binary)
        result = subprocess.run(["ldd", str(binary)], env=environment, capture_output=True, text=True, check=True)
        require("not found" not in result.stdout, f"Unresolved runtime library for {name}")
        dependencies = {}
        for line in result.stdout.splitlines():
            match = re.search(r"(?:=>\s*)?(/\S+)\s+\(", line)
            if match:
                dependencies[match.group(1)] = record(match.group(1))
        runtime[name] = {"ldd": result.stdout, "resolved_dependencies": dependencies}
    test_directory = out/"sparse-offset-test"
    test_directory.mkdir()
    test = execute([str(out/TARGETS["test_stitched_f32_large_offset"]), str(test_directory)],
                   out/"sparse-offset-test.log", environment, out)
    require("PASS: exact extraction before and after 4 GiB; global IDs preserved" in (out/"sparse-offset-test.log").read_text(),
            "Sparse-offset test did not confirm its assertions")
    # A linker invocation must not change any pinned source/library input.
    release_records = [proof["flags"] for proof in release_configuration["core_libraries"] if proof["flags"] is not None]
    if release_configuration["cmake_cache"] is not None:
        release_records.append(release_configuration["cmake_cache"])
    for item in [*upstream["verified_source_files"].values(), *linked_records, *input_records.values(), *release_records]:
        require(record(item["path"]) == item, f"Historical/source input changed during helper build: {item['path']}")
    result = {"schema_version": 1, "plan": record(out/"build-plan.json"), "compiler_version": compiler_version,
              "jobs": jobs, "binaries": binaries, "runtime_dependencies": runtime,
              "sparse_offset_test": test, "sparse_offset_test_passed": True,
              "sparse_test_artifacts": [{"name": path.name, "logical_bytes": path.stat().st_size,
                  "allocated_bytes": path.stat().st_blocks*512,
                  "sha256": sha256(path) if path.stat().st_size < (1 << 20) else None,
                  "large_file_hash_omission": "Sparse fixture; real extraction assertion and input source are retained" if path.stat().st_size >= (1 << 20) else None}
                  for path in sorted(test_directory.iterdir()) if path.is_file()],
              "historical_source_or_build_modified": False}
    save(out/"build-result.json", result)
    print(json.dumps({"result": str(out/"build-result.json"), "bin_dir": str(out/"bin"), "sparse_offset_test_passed": True}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "native-build-dir", "source-archive", "native-evidence", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--expected-source-sha256", default=SOURCE_ARCHIVE_SHA256)
    parser.add_argument("--reference-target", default="build_disk_index")
    parser.add_argument("--settings-json", type=Path)
    parser.add_argument("--runtime-library-dir", action="append", type=Path, default=[])
    parser.add_argument("--plan-only", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
