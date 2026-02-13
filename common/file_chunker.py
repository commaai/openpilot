import glob
import math
import os
import sys
from pathlib import Path

CHUNK_SIZE = 49 * 1024 * 1024  # 49MB, under GitHub's 50MB limit


def chunk_file(path, num_chunks=None):
  with open(path, 'rb') as f:
    data = f.read()
  actual_num_chunks = max(1, math.ceil(len(data) / CHUNK_SIZE))
  if num_chunks is None:
    num_chunks = actual_num_chunks
  assert num_chunks >= actual_num_chunks, f"expected {num_chunks} chunks but data needs at least {actual_num_chunks}"
  for i in range(num_chunks):
    with open(f"{path}.chunk{i:02d}", 'wb') as f:
      f.write(data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE])
  os.remove(path)


def read_file_chunked(path):
  files = sorted(glob.glob(f"{path}.chunk*")) or ([path] if os.path.isfile(path) else [])
  if not files:
    raise FileNotFoundError(path)
  return b''.join(Path(f).read_bytes() for f in files)


if __name__ == "__main__":
  num_chunks = int(sys.argv[1])
  for path in sys.argv[2:]:
    chunk_file(path, num_chunks)
