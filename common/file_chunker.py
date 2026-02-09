import glob
import os
from pathlib import Path

CHUNK_SIZE = 49 * 1024 * 1024  # 49MB, under GitHub's 50MB limit


def rechunk_file(path):
  with open(path, 'rb') as f:
    data = f.read()
  for i in range(0, len(data), CHUNK_SIZE):
    with open(f"{path}.chunk{i // CHUNK_SIZE:02d}", 'wb') as f:
      f.write(data[i:i + CHUNK_SIZE])
  os.remove(path)


def read_file_chunked(path):
  files = sorted(glob.glob(f"{path}.chunk*")) or ([path] if os.path.isfile(path) else [])
  if not files:
    raise FileNotFoundError(path)
  return b''.join(Path(f).read_bytes() for f in files)
