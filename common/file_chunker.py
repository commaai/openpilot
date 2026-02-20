import math
import os
from pathlib import Path

CHUNK_SIZE = 49 * 1024 * 1024  # 49MB, under GitHub's 50MB limit

def get_chunk_name(name, idx, num_chunks):
  return f"{name}.chunk{idx+1:02d}of{num_chunks:02d}"

def get_chunk_paths(path, file_size):
  num_chunks = math.ceil(file_size / CHUNK_SIZE)
  return [get_chunk_name(path, i, num_chunks) for i in range(num_chunks)]

def chunk_file(path, num_chunks):
  with open(path, 'rb') as f:
    data = f.read()
  actual_num_chunks = max(1, math.ceil(len(data) / CHUNK_SIZE))
  assert num_chunks >= actual_num_chunks, f"Allowed {num_chunks} chunks but needs at least {actual_num_chunks}, for path {path}"
  for i in range(num_chunks):
    with open(get_chunk_name(path, i, num_chunks), 'wb') as f:
      f.write(data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE])


def read_file_chunked(path):
  for num_chunks in range(1, 100):
    if os.path.isfile(get_chunk_name(path, 0, num_chunks)):
      files = [get_chunk_name(path, i, num_chunks) for i in range(num_chunks)]
      return b''.join(Path(f).read_bytes() for f in files)
  if os.path.isfile(path):
    return Path(path).read_bytes()
  raise FileNotFoundError(path)
