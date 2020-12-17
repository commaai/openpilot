#!/usr/bin/env python3
import json
import lzma
import hashlib
import requests
import struct
import subprocess
import os

from common.spinner import Spinner


class StreamingDecompressor:
  def __init__(self, url):
    self.buf = b""

    self.req = requests.get(url, stream=True, headers={'Accept-Encoding': None})
    self.it = self.req.iter_content(chunk_size=1024 * 1024)
    self.decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
    self.eof = False
    self.sha256 = hashlib.sha256()

  def read(self, length):
    while len(self.buf) < length:
      self.req.raise_for_status()

      try:
        compressed = next(self.it)
      except StopIteration:
        self.eof = True
        break
      out = self.decompressor.decompress(compressed)
      self.buf += out

    result = self.buf[:length]
    self.buf = self.buf[length:]

    self.sha256.update(result)
    return result


def unsparsify(f):
  magic = struct.unpack("I", f.read(4))[0]
  assert(magic == 0xed26ff3a)

  # Version
  major = struct.unpack("H", f.read(2))[0]
  minor = struct.unpack("H", f.read(2))[0]
  assert(major == 1 and minor == 0)

  # Header sizes
  _ = struct.unpack("H", f.read(2))[0]
  _ = struct.unpack("H", f.read(2))[0]

  block_sz = struct.unpack("I", f.read(4))[0]
  _ = struct.unpack("I", f.read(4))[0]
  num_chunks = struct.unpack("I", f.read(4))[0]
  _ = struct.unpack("I", f.read(4))[0]

  for _ in range(num_chunks):
    chunk_type = struct.unpack("H", f.read(2))[0]
    _ = struct.unpack("H", f.read(2))[0]
    out_blocks = struct.unpack("I", f.read(4))[0]
    _ = struct.unpack("I", f.read(4))[0]

    if chunk_type == 0xcac1:  # Raw
      # TODO: yield in smaller chunks. Yielding only block_sz is too slow. Largest observed data chunk is 252 MB.
      yield f.read(out_blocks * block_sz)
    elif chunk_type == 0xcac2:  # Fill
      filler = f.read(4) * (block_sz // 4)
      for _ in range(out_blocks):
        yield filler
    elif chunk_type == 0xcac3:  # Don't care
      yield b""
    else:
      raise Exception("Unhandled sparse chunk type")


def flash_agnos_update(manifest_path, cloudlog, spinner=None):
  update = json.load(open(manifest_path))

  current_slot = subprocess.check_output(["abctl", "--boot_slot"], encoding='utf-8').strip()
  target_slot = "_b" if current_slot == "_a" else "_a"
  target_slot_number = "0" if target_slot == "_a" else "1"

  cloudlog.info(f"Current slot {current_slot}, target slot {target_slot}")

  # set target slot as unbootable
  os.system(f"abctl --set_unbootable {target_slot_number}")

  for partition in update:
    cloudlog.info(f"Downloading and writing {partition['name']}")

    downloader = StreamingDecompressor(partition['url'])
    with open(f"/dev/disk/by-partlabel/{partition['name']}{target_slot}", 'wb') as out:
      partition_size = partition['size']
      # Clear hash before flashing
      out.seek(partition_size)
      out.write(b"\x00" * 64)
      out.seek(0)
      os.sync()

      # Flash partition
      if partition['sparse']:
        raw_hash = hashlib.sha256()
        for chunk in unsparsify(downloader):
          raw_hash.update(chunk)
          out.write(chunk)

          if spinner is not None:
            spinner.update_progress(out.tell(), partition_size)

        if raw_hash.hexdigest().lower() != partition['hash_raw'].lower():
          raise Exception(f"Unsparse hash mismatch '{raw_hash.hexdigest().lower()}'")
      else:
        while not downloader.eof:
          out.write(downloader.read(1024 * 1024))

          if spinner is not None:
            spinner.update_progress(out.tell(), partition_size)

      if downloader.sha256.hexdigest().lower() != partition['hash'].lower():
        raise Exception("Uncompressed hash mismatch")

      if out.tell() != partition['size']:
        raise Exception("Uncompressed size mismatch")

      # Write hash after successfull flash
      os.sync()
      out.write(partition['hash_raw'].lower().encode())

  cloudlog.info(f"AGNOS ready on slot {target_slot}")


if __name__ == "__main__":
  import logging
  import time
  import sys

  if len(sys.argv) != 2:
    print("Usage: ./agnos.py <manifest.json>")
    exit(1)

  spinner = Spinner()
  spinner.update("Updating AGNOS")
  time.sleep(5)

  logging.basicConfig(level=logging.INFO)
  flash_agnos_update(sys.argv[1], logging, spinner)
