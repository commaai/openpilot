#!/usr/bin/env python3
import functools
import io
import lzma
import os
import struct
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, BinaryIO, Callable, Optional

import requests
from Crypto.Hash import SHA512

CA_FORMAT_INDEX = 0x96824d9c7b129ff9
CA_FORMAT_TABLE = 0xe75b9e112f17417d
CA_FORMAT_TABLE_TAIL_MARKER = 0xe75b9e112f17417
FLAGS = 0xb000000000000000

CA_HEADER_LEN = 48
CA_TABLE_HEADER_LEN = 16
CA_TABLE_ENTRY_LEN = 40
CA_TABLE_MIN_LEN = CA_TABLE_HEADER_LEN + CA_TABLE_ENTRY_LEN


def parse_caibx(caibx_path: str) -> List[Tuple[bytes, int, int]]:
  """Parses the chunks from a caibx file. Can handle both local and remote files.
  Returns a list of chunks with hash, offset and length"""
  if os.path.isfile(caibx_path):
    caibx = open(caibx_path, 'rb')
  else:
    resp = requests.get(caibx_path)
    resp.raise_for_status()
    caibx = io.BytesIO(resp.content)

  caibx.seek(0, os.SEEK_END)
  caibx_len = caibx.tell()
  caibx.seek(0, os.SEEK_SET)

  # Parse header
  length, magic, flags, min_size, _, max_size = struct.unpack("<QQQQQQ", caibx.read(CA_HEADER_LEN))
  assert flags == flags
  assert length == CA_HEADER_LEN
  assert magic == CA_FORMAT_INDEX

  # Parse table header
  length, magic = struct.unpack("<QQ", caibx.read(CA_TABLE_HEADER_LEN))
  assert magic == CA_FORMAT_TABLE

  # Parse chunks
  num_chunks = (caibx_len - CA_HEADER_LEN - CA_TABLE_MIN_LEN) // CA_TABLE_ENTRY_LEN
  chunks = []

  offset = 0
  for i in range(num_chunks):
    new_offset = struct.unpack("<Q", caibx.read(8))[0]

    sha = caibx.read(32)
    length = new_offset - offset

    assert length <= max_size

    # Last chunk can be smaller
    if i < num_chunks - 1:
      assert length >= min_size

    chunks.append((sha, offset, length))
    offset = new_offset

  return chunks


def build_chunk_dict(chunks: List[Tuple[bytes, int, int]]) -> Dict[bytes, Tuple[int, int]]:
  """Turn a list of chunks into a dict for faster lookups based on hash"""
  return {sha: (offset, length) for sha, offset, length in chunks}


def read_chunk_local_file(sha: bytes, chunk: Tuple[int, int], f: BinaryIO) -> bytes:
  """Read a chunk from an open file"""
  f.seek(chunk[0])
  return f.read(chunk[1])


def read_chunk_remote_store(sha: bytes, chunk: Tuple[int, int], store_path: str) -> bytes:
  """Read an lzma compressed chunk from a remote store"""
  sha_hex = sha.hex()
  url = os.path.join(store_path, sha_hex[:4], sha_hex + ".cacnk")

  resp = requests.get(url)
  resp.raise_for_status()

  decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
  return decompressor.decompress(resp.content)


def extract(target: List[Tuple[bytes, int, int]],
            sources: List[Tuple[str, functools.partial[bytes], Dict[bytes, Tuple[int, int]]]],
            out_path: str, progress: Optional[Callable[[int], None]] = None):
  stats: Dict[str, int] = defaultdict(int)

  with open(out_path, 'wb') as out:
    for sha, offset, length in target:

      # Find source for desired chunk
      for name, callback, store_chunks in sources:
        if sha in store_chunks:
          bts = callback(sha, store_chunks[sha])

          # Check length
          if len(bts) != length:
            continue

          # Check hash
          if SHA512.new(bts, truncate="256").digest() != sha:
            continue

          # Write to output
          out.seek(offset)
          out.write(bts)

          stats[name] += length

          if progress is not None:
            progress(sum(stats.values()))

          break
      else:
        raise RuntimeError("Desired chunk not found in provided stores")

  return stats


def print_stats(stats: Dict[str, int]):
  total_bytes = sum(stats.values())
  print(f"Total size: {total_bytes / 1024 / 1024:.2f} MB")
  for name, total in stats.items():
    print(f"  {name}: {total / 1024 / 1024:.2f} MB ({total / total_bytes * 100:.1f}%)")


def extract_simple(caibx_path, out_path, store_path):
  # (name, callback, chunks)
  target = parse_caibx(caibx_path)
  sources = [
    # (store_path, functools.partial(read_chunk_remote_store, store_path=store_path), build_chunk_dict(target)),
    (store_path, functools.partial(read_chunk_local_file, f=open(store_path, 'rb')), build_chunk_dict(target)),
  ]

  return extract(target, sources, out_path)


if __name__ == "__main__":
  caibx = sys.argv[1]
  out = sys.argv[2]
  store = sys.argv[3]

  stats = extract_simple(caibx, out, store)
  print_stats(stats)
