import os

from openpilot.common.utils import atomic_write

CHUNK_SIZE = 49 * 1024 * 1024  # 49MB, under GitHub's 50MB limit


def write_file_chunked(path: str, data: bytes) -> None:
  if len(data) <= CHUNK_SIZE:
    with atomic_write(path, mode='wb', overwrite=True) as f:
      f.write(data)
  else:
    for i in range(0, len(data), CHUNK_SIZE):
      chunk_path = f"{path}.chunk{i // CHUNK_SIZE:02d}"
      with atomic_write(chunk_path, mode='wb', overwrite=True) as f:
        f.write(data[i:i + CHUNK_SIZE])


def read_file_chunked(path: str) -> bytes:
  if os.path.isfile(path):
    with open(path, 'rb') as f:
      return f.read()

  # look for chunk files
  chunks = []
  i = 0
  while True:
    chunk_path = f"{path}.chunk{i:02d}"
    if not os.path.isfile(chunk_path):
      break
    with open(chunk_path, 'rb') as f:
      chunks.append(f.read())
    i += 1

  if not chunks:
    raise FileNotFoundError(f"No such file or chunks: '{path}'")

  return b''.join(chunks)


def rechunk_file(path: str) -> None:
  """Read a file and resave it as chunks, removing the original."""
  with open(path, 'rb') as f:
    data = f.read()
  for i in range(0, len(data), CHUNK_SIZE):
    chunk_path = f"{path}.chunk{i // CHUNK_SIZE:02d}"
    with atomic_write(chunk_path, mode='wb', overwrite=True) as f:
      f.write(data[i:i + CHUNK_SIZE])
  os.remove(path)
