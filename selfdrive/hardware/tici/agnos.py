#!/usr/bin/env python3
import json
import lzma
import hashlib
import requests
import struct
import subprocess
import os
from typing import Generator, Optional

from common.spinner import Spinner

SPARSE_CHUNK_FMT = struct.Struct('H2xI4x')


class StreamingDecompressor:
  def __init__(self, url: str) -> None:
    self.buf = b""

    self.req = requests.get(url, stream=True, headers={'Accept-Encoding': None})
    self.it = self.req.iter_content(chunk_size=1024 * 1024)
    self.decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
    self.eof = False
    self.sha256 = hashlib.sha256()

  def read(self, length: int) -> bytes:
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


def unsparsify(f: StreamingDecompressor) -> Generator[bytes, None, None]:
  # https://source.android.com/devices/bootloader/images#sparse-format
  magic = struct.unpack("I", f.read(4))[0]
  assert(magic == 0xed26ff3a)

  # Version
  major = struct.unpack("H", f.read(2))[0]
  minor = struct.unpack("H", f.read(2))[0]
  assert(major == 1 and minor == 0)

  f.read(2)  # file header size
  f.read(2)  # chunk header size

  block_sz = struct.unpack("I", f.read(4))[0]
  f.read(4)  # total blocks
  num_chunks = struct.unpack("I", f.read(4))[0]
  f.read(4)  # crc checksum

  for _ in range(num_chunks):
    chunk_type, out_blocks = SPARSE_CHUNK_FMT.unpack(f.read(12))

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


def get_target_slot_number() -> int:
  current_slot = subprocess.check_output(["abctl", "--boot_slot"], encoding='utf-8').strip()
  return 1 if current_slot == "_a" else 0


def slot_number_to_suffix(slot_number: int) -> str:
  assert slot_number in (0, 1)
  return '_a' if slot_number == 0 else '_b'


def get_partition_path(target_slot_number: int, partition: dict) -> str:
  path = f"/dev/disk/by-partlabel/{partition['name']}"

  if partition.get('has_ab', True):
    path += slot_number_to_suffix(target_slot_number)

  return path


def verify_partition(target_slot_number: int, partition: dict) -> bool:
  full_check = partition['full_check']
  path = get_partition_path(target_slot_number, partition)
  partition_size = partition['size']

  with open(path, 'rb+') as out:
    if full_check:
      raw_hash = hashlib.sha256()

      pos = 0
      chunk_size = 1024 * 1024
      while pos < partition_size:
        n = min(chunk_size, partition_size - pos)
        raw_hash.update(out.read(n))
        pos += n

      return raw_hash.hexdigest().lower() == partition['hash_raw'].lower()
    else:
      out.seek(partition_size)
      return out.read(64) == partition['hash_raw'].lower().encode()


def clear_partition_hash(target_slot_number: int, partition: dict) -> None:
  path = get_partition_path(target_slot_number, partition)
  with open(path, 'wb+') as out:
    partition_size = partition['size']

    out.seek(partition_size)
    out.write(b"\x00" * 64)
    os.sync()


def flash_partition(target_slot_number: int, partition: dict, cloudlog, spinner: Optional[Spinner] = None):
  cloudlog.info(f"Downloading and writing {partition['name']}")

  if verify_partition(target_slot_number, partition):
    cloudlog.info(f"Already flashed {partition['name']}")
    return

  downloader = StreamingDecompressor(partition['url'])

  # Clear hash before flashing in case we get interrupted
  full_check = partition['full_check']
  if not full_check:
    clear_partition_hash(target_slot_number, partition)

  path = get_partition_path(target_slot_number, partition)
  with open(path, 'wb+') as out:
    partition_size = partition['size']

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
    if not full_check:
      out.write(partition['hash_raw'].lower().encode())


def swap(manifest_path: str, target_slot_number: int, cloudlog) -> None:
  update = json.load(open(manifest_path))
  for partition in update:
    if not partition.get('full_check', False):
      clear_partition_hash(target_slot_number, partition)

  while True:
    out = subprocess.check_output(f"abctl --set_active {target_slot_number}", shell=True, stderr=subprocess.STDOUT, encoding='utf8')
    if ("No such file or directory" not in out) and ("lun as boot lun" in out):
      cloudlog.info(f"Swap successfull {out}")
      break
    else:
      cloudlog.error(f"Swap failed {out}")


def flash_agnos_update(manifest_path: str, target_slot_number: int, cloudlog, spinner: Optional[Spinner] = None) -> None:
  update = json.load(open(manifest_path))

  cloudlog.info(f"Target slot {target_slot_number}")

  # set target slot as unbootable
  os.system(f"abctl --set_unbootable {target_slot_number}")

  for partition in update:
    success = False

    for retries in range(10):
      try:
        flash_partition(target_slot_number, partition, cloudlog, spinner)
        success = True
        break

      except requests.exceptions.RequestException:
        cloudlog.exception("Failed")
        if spinner is not None:
          spinner.update("Waiting for internet...")
        cloudlog.info(f"Failed to download {partition['name']}, retrying ({retries})")
        time.sleep(10)

    if not success:
      cloudlog.info(f"Failed to flash {partition['name']}, aborting")
      raise Exception("Maximum retries exceeded")

  cloudlog.info(f"AGNOS ready on slot {target_slot_number}")


def verify_agnos_update(manifest_path: str, target_slot_number: int) -> bool:
  update = json.load(open(manifest_path))
  return all(verify_partition(target_slot_number, partition) for partition in update)


if __name__ == "__main__":
  import logging
  import time
  import argparse

  parser = argparse.ArgumentParser(description="Flash and verify AGNOS update",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--swap", action="store_true", help="Verify and perform swap, downloads if necessary")
  parser.add_argument("manifest", help="Manifest json")
  args = parser.parse_args()

  spinner = Spinner()
  spinner.update("Updating AGNOS")
  time.sleep(5)

  logging.basicConfig(level=logging.INFO)

  target_slot_number = get_target_slot_number()
  if args.swap:
    while not verify_agnos_update(args.manifest, target_slot_number):
      logging.error("Verification failed. Flashing AGNOS")
      flash_agnos_update(args.manifest, target_slot_number, logging, spinner)

    logging.warning(f"Verification succeeded. Swapping to slot {target_slot_number}")
    swap(args.manifest, target_slot_number, logging)
  else:
    flash_agnos_update(args.manifest, target_slot_number, logging, spinner)
