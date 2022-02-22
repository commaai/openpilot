#!/usr/bin/env python3
import sys
import struct
import os
import lzma
import functools
from Crypto.Hash import SHA512

CA_FORMAT_INDEX = 0x96824d9c7b129ff9
CA_FORMAT_TABLE = 0xe75b9e112f17417d
CA_FORMAT_TABLE_TAIL_MARKER = 0xe75b9e112f17417

CA_HEADER_LEN = 48
CA_TABLE_HEADER_LEN = 16
CA_TABLE_ENTRY_LEN = 40
CA_TABLE_MIN_LEN = CA_TABLE_HEADER_LEN + CA_TABLE_ENTRY_LEN


def parse_caibx(caibx_path):
  with open(caibx_path, 'rb') as caibx:
    caibx.seek(0, os.SEEK_END)
    caibx_len = caibx.tell()
    caibx.seek(0, os.SEEK_SET)

    # Parse header
    length, magic, flags, min_size, _, max_size = struct.unpack("<QQQQQQ", caibx.read(CA_HEADER_LEN))
    assert flags
    assert length == CA_HEADER_LEN
    assert magic == CA_FORMAT_INDEX

    # Parse table header
    length, magic = struct.unpack("<QQ", caibx.read(CA_TABLE_HEADER_LEN))
    assert magic == CA_FORMAT_TABLE

    # Parse chunks
    num_chunks = (caibx_len - CA_HEADER_LEN - CA_TABLE_MIN_LEN) // CA_TABLE_ENTRY_LEN
    chunks = {}

    offset = 0
    for i in range(num_chunks):
      new_offset = struct.unpack("<Q", caibx.read(8))[0]

      sha = caibx.read(32)
      length = new_offset - offset

      assert length <= max_size

      # Last chunk can be smaller
      if i != num_chunks - 1:
        assert length >= min_size

      chunks[sha] = (offset, length)
      offset = new_offset

  return chunks


def read_chunk_local_store(sha, store_path):
  sha_hex = sha.hex()
  path = os.path.join(store_path, sha_hex[:4], sha_hex + ".cacnk")
  return lzma.open(path).read()


def extract(target, sources, out_path):
  with open(out_path, 'wb') as out:
    for sha, (offset, length) in target.items():

      # Find source for desired chunk
      for callback, store_chunks in sources:
        if sha in store_chunks:
          bts = callback(sha)

          # Check length
          if len(bts) != length:
            continue

          # Check hash
          if SHA512.new(bts, truncate="256").digest() != sha:
            continue

          # Write to output
          out.seek(offset)
          out.write(bts)

          break
      else:
        raise RuntimeError("Desired chunk not found")


def extract_simple(caibx_path, out_path, store_path):
  # (callback, chunks)
  sources = [
    (functools.partial(read_chunk_local_store, store_path=store_path), parse_caibx(caibx_path)),
  ]
  extract(parse_caibx(caibx_path), sources, out_path)


if __name__ == "__main__":
  caibx = sys.argv[1]
  out = sys.argv[2]
  store = sys.argv[3]

  extract_simple(caibx, out, store)
