from tinygrad.tensor import Tensor
from tinygrad.helpers import CHUNK_SIZE
from tinygrad.nn.state import fs_load
import argparse, math, hashlib

def _python_hash_1mb(data:bytes|bytearray):
  chunks = [data[i:i+4096] for i in range(0, len(data), 4096)]
  chunk_hashes = [hashlib.shake_128(chunk).digest(16) for chunk in chunks]
  return hashlib.shake_128(b''.join(chunk_hashes)).digest(16)

def hash_file(data: bytes|bytearray):
  if len(data) % CHUNK_SIZE != 0: data += bytes(CHUNK_SIZE - len(data) % CHUNK_SIZE)
  base_chunks = math.ceil(len(data) / CHUNK_SIZE)
  tree_depth = math.ceil(math.log(base_chunks, CHUNK_SIZE // 16))

  for _ in range(tree_depth + 1):
    data_chunks = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
    data_chunk_hashes = [_python_hash_1mb(chunk) for chunk in data_chunks]
    data = b''.join(data_chunk_hashes)
    if len(data) % CHUNK_SIZE != 0: data += bytes(CHUNK_SIZE - len(data) % CHUNK_SIZE)

  return data[:16]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--hash", type=str, required=True, help="file hash to fetch")
  parser.add_argument("--len", type=int, required=True, help="file length to fetch")
  parser.add_argument("--dest", type=str, required=True, help="destination path to save the file")
  parser.add_argument("--check", action="store_true", help="verify the file hash after fetching")
  args = parser.parse_args()

  fs_load(Tensor(bytes.fromhex(args.hash), device="CPU"), args.len).to(f"disk:{args.dest}").realize()

  if args.check:
    with open(args.dest, "rb") as f:
      data = f.read()
      assert hash_file(data) == bytes.fromhex(args.hash), "Hash mismatch after fetching file"
    print("File hash verified successfully!")
