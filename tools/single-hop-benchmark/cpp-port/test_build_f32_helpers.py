#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Native helper command discovery, pinning, preservation, and host guards."""

import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("helper_build", Path(__file__).with_name("build_f32_helpers.py"))
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class ParserTests(unittest.TestCase):
    def test_make_flags_preserve_defines_includes_and_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flags.make"
            path.write_text('CXX_DEFINES = -DMKL_ILP64 -DNAME=\\"native\\"\nCXX_INCLUDES = -I"/tmp/path with spaces"\nCXX_FLAGS = -O3 -DNDEBUG -mavx2 -fopenmp -std=gnu++17\n')
            flags = build.parse_flags_make(path)
            self.assertIn('-DNAME="native"', flags)
            self.assertIn('-I/tmp/path with spaces', flags)
            self.assertIn('-mavx2', flags)

    def test_link_parser_discards_only_old_object_and_output(self):
        parsed = build.parse_link_command('/usr/bin/g++ -O3 -Wl,--start-group CMakeFiles/tool.dir/old.cpp.o -o tool ../src/libdiskann.a -lmkl_core -Wl,--end-group -fopenmp')
        self.assertEqual(parsed["reference_output"], "tool")
        self.assertEqual(parsed["discarded_objects"], ["CMakeFiles/tool.dir/old.cpp.o"])
        self.assertEqual(parsed["link_flags"], ['-O3', '-Wl,--start-group', '../src/libdiskann.a', '-lmkl_core', '-Wl,--end-group', '-fopenmp'])

    def test_shell_operations_response_files_and_variables_rejected(self):
        for command in ('g++ @objects.rsp -o tool', 'g++ a.o -o tool && evil', 'g++ a.o -o tool $(bad)'):
            with self.assertRaises(ValueError):
                build.parse_link_command(command)
        with self.assertRaisesRegex(ValueError, "array"):
            build.validate_tokens('-O3 -Ifoo')

    def test_relative_paths_resolve_against_original_target_working_directory(self):
        flags = build.absolute_flags(['-I../include', '-L', '../lib', '../lib/libdiskann.a', '-Wl,-rpath,/opt/lib'], Path('/tmp/build/apps'))
        self.assertEqual(flags, ['-I'+str(Path('/tmp/build/include').resolve()), '-L', str(Path('/tmp/build/lib').resolve()), str(Path('/tmp/build/lib/libdiskann.a').resolve()), '-Wl,-rpath,/opt/lib'])


    def test_runtime_paths_preserve_cmake_combined_rpaths(self):
        with patch.dict(build.os.environ, {"LD_LIBRARY_PATH": ""}):
            environment = build.runtime_environment(["-Wl,-rpath,/private/deps,-rpath,/usr/lib/llvm-18/lib", "-L/private/extra"], [], [])
        self.assertEqual(environment["LD_LIBRARY_PATH"].split(":"), ["/private/deps", "/usr/lib/llvm-18/lib", "/private/extra"])


class PlanTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'cpp'
        native = source / 'build-native'
        target = native / 'apps/CMakeFiles/build_disk_index.dir'
        target.mkdir(parents=True)
        (native / 'src').mkdir()
        core_flags = native/'src/CMakeFiles/diskann.dir/flags.make'
        core_flags.parent.mkdir(parents=True)
        core_flags.write_text('CXX_DEFINES = -DMKL_ILP64\nCXX_INCLUDES =\nCXX_FLAGS = -O3 -DNDEBUG\n')
        (native/'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Release\n')
        library = native / 'src/libdiskann.a'
        library.write_bytes(b'unchanged native library')
        files = {'include/index.h': b'// pinned header\n', 'apps/utils/create_disk_layout.cpp': b'// pinned layout\n', 'src/pq.cpp': b'// original pq\n'}
        archive = root / 'source.tar'
        with tarfile.open(archive, 'w') as tar:
            for name, value in files.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
                info = tarfile.TarInfo(name)
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
        (source / 'src/pq.cpp').write_bytes(b'// pinned native metrics patch\n')
        evidence = root / 'native-evidence.json'
        evidence.write_text(json.dumps({'source_commit': build.CPP_COMMIT, 'source_archive_sha256': build.sha256(archive),
                                        'native_compiled': True, 'machine': 'x86_64',
                                        'patched_source_sha256': {'src/pq.cpp': build.sha256(source / 'src/pq.cpp')}}))
        compiler = shutil.which('c++')
        self.assertIsNotNone(compiler)
        (target / 'flags.make').write_text(f'CXX_DEFINES = -DMKL_ILP64\nCXX_INCLUDES = -I{source}/include\nCXX_FLAGS = -O3 -DNDEBUG -fopenmp -std=gnu++17\n')
        (target / 'link.txt').write_text(f'{compiler} -O3 CMakeFiles/build_disk_index.dir/old.cpp.o -o build_disk_index ../src/libdiskann.a -lm -ldl\n')
        return types.SimpleNamespace(source_dir=source, native_build_dir=native, out=root/'new-helpers',
                                     source_archive=archive, native_evidence=evidence,
                                     expected_source_sha256=build.sha256(archive), settings_json=None,
                                     reference_target='build_disk_index', runtime_library_dir=[], plan_only=True)

    def test_plan_discovers_existing_target_and_preserves_library_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            before = build.record(args.native_build_dir / 'src/libdiskann.a')
            with patch.object(build.subprocess, 'run', side_effect=AssertionError('plan must not compile')):
                plan = build.build(args)
            self.assertTrue(plan['plan_only'])
            self.assertFalse(plan['historical_source_or_build_modified'])
            self.assertEqual(before, build.record(args.native_build_dir / 'src/libdiskann.a'))
            self.assertEqual(len(plan['commands']), 4)
            self.assertIn(str((args.out / 'bin/utils/generate_pq_f32').resolve()), plan['commands']['generate_pq_f32'])
            self.assertEqual((args.out/'src/build_stitched_f32.cpp').read_bytes(), Path(__file__).with_name('build_stitched_f32.cpp').read_bytes())
            self.assertFalse((args.out/'build-result.json').exists())

    def test_source_archive_and_native_patch_changes_fail_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            args.expected_source_sha256 = '0'*64
            with self.assertRaisesRegex(ValueError, 'archive SHA-256'):
                build.build(args)
            self.assertFalse(args.out.exists())
            args.expected_source_sha256 = build.sha256(args.source_archive)
            (args.source_dir/'src/pq.cpp').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'patched source changed'):
                build.build(args)
            self.assertFalse(args.out.exists())

    def test_ambiguous_cmake_discovery_requires_explicit_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            second = args.native_build_dir/'other/CMakeFiles/build_disk_index.dir'
            second.mkdir(parents=True)
            (second/'link.txt').write_text('fake')
            with self.assertRaisesRegex(ValueError, 'found 2'):
                build.discover_settings(args.native_build_dir, 'build_disk_index')

    def test_native_execution_rejects_non_linux_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args.plan_only = False
            with patch.object(build.platform, 'system', return_value='Darwin'):
                with self.assertRaisesRegex(ValueError, 'Linux x86-64'):
                    build.build(args)
            self.assertFalse(args.out.exists())

    def test_debug_core_and_effective_undefine_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            flags = args.native_build_dir/'src/CMakeFiles/diskann.dir/flags.make'
            flags.write_text('CXX_DEFINES =\nCXX_INCLUDES =\nCXX_FLAGS = -O0 -g\n')
            with self.assertRaisesRegex(ValueError, 'core library diskann lacks effective NDEBUG'):
                build.build(args)
            self.assertFalse(args.out.exists())
        self.assertTrue(build.effective_ndebug(['-DNDEBUG=0']))
        self.assertFalse(build.effective_ndebug(['-DNDEBUG', '-UNDEBUG']))
        self.assertTrue(build.effective_ndebug(['-U', 'NDEBUG', '-D', 'NDEBUG']))

    def test_missing_core_flags_requires_release_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            (args.native_build_dir/'src/CMakeFiles/diskann.dir/flags.make').unlink()
            cache = args.native_build_dir/'CMakeCache.txt'
            cache.write_text('CMAKE_BUILD_TYPE:STRING=Debug\n')
            with self.assertRaisesRegex(ValueError, 'not Release/RelWithDebInfo'):
                build.build(args)
            self.assertFalse(args.out.exists())
            cache.write_text('CMAKE_BUILD_TYPE:STRING=RelWithDebInfo\n')
            result = build.build(args)
            proof = result['release_configuration']['core_libraries'][0]
            self.assertEqual(proof['proof'], 'existing_cmake_cache_release_configuration')
            self.assertIsNone(proof['flags'])

    def test_explicit_settings_fallback_still_requires_pinned_native_library(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            settings = {'compiler': shutil.which('c++'), 'compile_flags': ['-O3', '-DNDEBUG', '-std=c++17', '-I'+str(args.source_dir/'include')],
                        'link_flags': [str(args.native_build_dir/'src/libdiskann.a'), '-lm'], 'working_directory': str(args.native_build_dir)}
            args.settings_json = Path(directory)/'settings.json'
            args.settings_json.write_text(json.dumps(settings))
            result = build.build(args)
            self.assertEqual(result['settings']['discovery_inputs'], [build.record(args.settings_json)])


if __name__ == '__main__':
    unittest.main()
