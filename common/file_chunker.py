import glob
import math
import os
from pathlib import Path

CHUNK_SIZE = 19 * 1024 * 1024  # 49MB, under GitHub's 50MB limit

def get_chunk_name(name, idx, num_chunks):
  return f"{name}.chunk{idx+1:02d}of{num_chunks:02d}"

def get_chunk_paths(path, file_size):
  num_chunks = math.ceil(file_size / CHUNK_SIZE)
  return [get_chunk_name(path, i, num_chunks) for i in range(num_chunks)]

def chunk_file(path, num_chunks):
  for old in glob.glob(f"{path}.chunk*"):
    os.remove(old)
  with open(path, 'rb') as f:
    data = f.read()
  actual_num_chunks = max(1, math.ceil(len(data) / CHUNK_SIZE))
  assert num_chunks >= actual_num_chunks, f"Allowed {num_chunks} chunks but needs at least {actual_num_chunks}, for path {path}"
  for i in range(num_chunks):
    with open(get_chunk_name(path, i, num_chunks), 'wb') as f:
      f.write(data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE])
  os.remove(path)


def read_file_chunked(path):
  chunks = sorted(glob.glob(f"{path}.chunk*"))
  if chunks:
    expected = [get_chunk_name(path, i, len(chunks)) for i in range(len(chunks))]
    assert chunks == expected, f"Chunk mismatch: {chunks} != {expected}"
    return b''.join(Path(f).read_bytes() for f in chunks)
  if os.path.isfile(path):
    return Path(path).read_bytes()
  raise FileNotFoundError(path)
