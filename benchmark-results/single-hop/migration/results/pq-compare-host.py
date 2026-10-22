import json, struct, pathlib

root = pathlib.Path('/home/alex/diskann-single-hop-benchmark')
left = root / 'indexes/yfcc10m_stitched_R32_R128_L100_pq_compressed.bin'
right = root / 'indexes/rust-native-10m-frozen/generation-00000000000000000001/index_pq_compressed.bin'
counts = [0] * 32
samples = []
changed_rows = changed_bytes = offset = 0
with left.open('rb') as a, right.open('rb') as b:
    header_a, header_b = a.read(8), b.read(8)
    assert header_a == header_b
    rows, chunks = struct.unpack('<II', header_a)
    assert chunks == 32
    while True:
        x, y = a.read(1 << 20), b.read(1 << 20)
        assert len(x) == len(y)
        if not x:
            break
        row_ids = set()
        if x != y:
            for start in range(0, len(x), 4096):
                xx, yy = x[start:start+4096], y[start:start+4096]
                if xx == yy:
                    continue
                for position, (cpp, rust) in enumerate(zip(xx, yy)):
                    if cpp != rust:
                        byte = offset + start + position
                        row, chunk = divmod(byte, chunks)
                        counts[chunk] += 1
                        row_ids.add(row)
                        changed_bytes += 1
                        if len(samples) < 64:
                            samples.append({'row': row, 'chunk': chunk, 'cpp': cpp, 'rust': rust})
        changed_rows += len(row_ids)
        offset += len(x)
assert offset == rows * chunks
result = {'rows': rows, 'chunks': chunks, 'identical_headers': True,
          'changed_rows': changed_rows, 'changed_code_bytes': changed_bytes,
          'changed_row_fraction': changed_rows / rows,
          'changed_code_fraction': changed_bytes / (rows * chunks),
          'changed_codes_by_chunk': counts, 'first_differences': samples,
          'cpp_path': str(left), 'rust_path': str(right)}
out = root / 'results/migration/pq-code-differences.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'first_differences'}))
