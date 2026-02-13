#!/usr/bin/env python3
import hashlib
import pickle
import sys
from pathlib import Path

def split_pickle(full_path: Path, out_prefix: Path, chunk_bytes: int) -> None:
  data = full_path.read_bytes()
  out_dir = out_prefix.parent

  for p in out_dir.glob(f"{out_prefix.name}.data-*"):
    p.unlink()

  total = (len(data) + chunk_bytes - 1) // chunk_bytes
  names = []
  for i in range(0, len(data), chunk_bytes):
    name = f"{out_prefix.name}.data-{(i // chunk_bytes) + 1:04d}-of-{total:04d}"
    (out_dir / name).write_bytes(data[i:i + chunk_bytes])
    names.append(name)

  manifest = hashlib.sha256(data).hexdigest() + "\n" + "\n".join(names) + "\n"
  (out_dir / (out_prefix.name + ".parts")).write_text(manifest)

def load_external_pickle(prefix: Path):
  parts = prefix.parent / (prefix.name + ".parts")
  lines = parts.read_text().splitlines()
  expected_hash, chunk_names = lines[0], lines[1:]

  data = bytearray()
  for name in chunk_names:
    data += (prefix.parent / name).read_bytes()

  if hashlib.sha256(data).hexdigest() != expected_hash:
    raise RuntimeError(f"hash mismatch loading {prefix}")
  return pickle.loads(data)

if __name__ == "__main__":
  split_pickle(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
