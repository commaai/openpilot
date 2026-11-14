#!/usr/bin/env python3
import contextlib
import hashlib
import json
import os
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Generator

import requests

SPARSE_CHUNK_FMT = struct.Struct('H2xI4x')
CHUNK_SIZE = 1024 * 1024

AGNOS_MANIFEST_FILE = "openpilot/system/hardware/comma/agnos.json"


class StreamingDecompressor:
  """Streams an xz image from url. The download runs in its own thread and xz decompresses with
  multiple threads, so the network, decompression, and the caller's hashing and writing all overlap."""

  def __init__(self, url: str) -> None:
    self.sha256 = hashlib.sha256()
    self.error: Exception | None = None
    self.finished = False

    self.req = requests.get(url, stream=True, headers={'Accept-Encoding': 'identity'}, timeout=60)
    self.req.raise_for_status()
    try:
      self.proc = subprocess.Popen(["xz", "--decompress", "--stdout", "--threads=4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except Exception:
      self.req.close()
      raise
    self.feeder = threading.Thread(target=self._feed, daemon=True)
    self.feeder.start()

  def _feed(self) -> None:
    assert self.proc.stdin is not None
    try:
      for chunk in self.req.iter_content(chunk_size=CHUNK_SIZE):
        self.proc.stdin.write(chunk)
    except requests.exceptions.RequestException as e:
      self.error = e  # retried by flash_agnos_update
    except OSError:
      pass  # xz exited, its exit code says why
    finally:
      try:
        self.proc.stdin.close()
      except OSError:
        pass

  def read(self, length: int) -> bytes:
    assert self.proc.stdout is not None
    result = self.proc.stdout.read(length)
    if len(result) < length and not self.finished:
      # end of stream, surface download errors (retried by the caller) before xz's own
      self.finished = True
      self.feeder.join()
      if self.error is not None:
        raise self.error
      if self.proc.wait() != 0:
        raise Exception(f"xz failed with exit code {self.proc.returncode}")

    self.sha256.update(result)
    return result

  def close(self) -> None:
    self.proc.kill()
    self.req.close()
    self.feeder.join()
    self.proc.wait()
    assert self.proc.stdout is not None
    self.proc.stdout.close()


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
      # yield in pieces, raw chunks can be hundreds of MB
      remaining = out_blocks * block_sz
      while remaining > 0:
        data = f.read(min(remaining, CHUNK_SIZE))
        if not data:
          raise Exception("Unexpected end of sparse image")
        remaining -= len(data)
        yield data
    elif chunk_type == 0xcac2:  # Fill
      filler = f.read(4) * (block_sz // 4)
      fill_blocks = CHUNK_SIZE // block_sz
      for i in range(0, out_blocks, fill_blocks):
        yield filler * min(fill_blocks, out_blocks - i)
    elif chunk_type == 0xcac3:  # Don't care
      yield b""
    else:
      raise Exception("Unhandled sparse chunk type")


# noop wrapper with same API as unsparsify() for non sparse images
def noop(f: StreamingDecompressor) -> Generator[bytes, None, None]:
  while len(chunk := f.read(CHUNK_SIZE)) > 0:
    yield chunk


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


def get_raw_hash(path: str, partition_size: int) -> str:
  raw_hash = hashlib.sha256()
  pos, chunk_size = 0, 1024 * 1024

  with open(path, 'rb+') as out:
    while pos < partition_size:
      n = min(chunk_size, partition_size - pos)
      raw_hash.update(out.read(n))
      pos += n

  return raw_hash.hexdigest().lower()


def verify_partition(target_slot_number: int, partition: dict[str, str | int], force_full_check: bool = False) -> bool:
  full_check = partition['full_check'] or force_full_check
  path = get_partition_path(target_slot_number, partition)

  if not isinstance(partition['size'], int):
    return False

  partition_size: int = partition['size']

  if not isinstance(partition['hash_raw'], str):
    return False

  partition_hash: str = partition['hash_raw']

  if full_check:
    return get_raw_hash(path, partition_size) == partition_hash.lower()
  else:
    with open(path, 'rb+') as out:
      out.seek(partition_size)
      return out.read(64) == partition_hash.lower().encode()


def clear_partition_hash(target_slot_number: int, partition: dict) -> None:
  path = get_partition_path(target_slot_number, partition)
  with open(path, 'wb+') as out:
    partition_size = partition['size']

    out.seek(partition_size)
    out.write(b"\x00" * 64)
    os.sync()


def extract_compressed_image(target_slot_number: int, partition: dict, cloudlog, progress: Callable[[int], None] | None = None):
  path = get_partition_path(target_slot_number, partition)
  with open(path, 'wb+') as out, contextlib.closing(StreamingDecompressor(partition['url'])) as downloader:
    # Flash partition
    last_p = 0
    raw_hash = hashlib.sha256()
    f = unsparsify if partition['sparse'] else noop
    for chunk in f(downloader):
      raw_hash.update(chunk)
      out.write(chunk)
      if progress is not None:
        progress(out.tell())
      p = int(out.tell() / partition['size'] * 100)
      if p != last_p:
        last_p = p
        print(f"Installing {partition['name']}: {p}", flush=True)

    if raw_hash.hexdigest().lower() != partition['hash_raw'].lower():
      raise Exception(f"Raw hash mismatch '{raw_hash.hexdigest().lower()}'")

    if downloader.sha256.hexdigest().lower() != partition['hash'].lower():
      raise Exception("Uncompressed hash mismatch")

    if out.tell() != partition['size']:
      raise Exception("Uncompressed size mismatch")

    os.sync()


def flash_partition(target_slot_number: int, partition: dict, cloudlog, standalone=False,
                    progress: Callable[[int], None] | None = None):
  cloudlog.info(f"Downloading and writing {partition['name']}")

  if verify_partition(target_slot_number, partition):
    cloudlog.info(f"Already flashed {partition['name']}")
    return

  # Clear hash before flashing in case we get interrupted
  full_check = partition['full_check']
  if not full_check:
    clear_partition_hash(target_slot_number, partition)

  path = get_partition_path(target_slot_number, partition)

  extract_compressed_image(target_slot_number, partition, cloudlog, progress)

  # Write hash after successful flash
  if not full_check:
    with open(path, 'wb+') as out:
      out.seek(partition['size'])
      out.write(partition['hash_raw'].lower().encode())


def swap(manifest_path: str, target_slot_number: int, cloudlog) -> None:
  update = json.load(open(manifest_path))
  for partition in update:
    if not partition.get('full_check', False):
      clear_partition_hash(target_slot_number, partition)

  while True:
    out = subprocess.check_output(f"abctl --set_active {target_slot_number}", shell=True, stderr=subprocess.STDOUT, encoding='utf8')
    if ("No such file or directory" not in out) and ("lun as boot lun" in out):
      cloudlog.info(f"Swap successful {out}")
      break
    else:
      cloudlog.error(f"Swap failed {out}")


def flash_agnos_update(manifest_path: str, target_slot_number: int, cloudlog, standalone=False,
                       progress: Callable[[int, int], None] | None = None) -> None:
  """progress is called with (bytes written, total bytes) across all partitions"""
  update = json.load(open(manifest_path))
  total_size = sum(partition['size'] for partition in update)
  done_size = 0

  cloudlog.info(f"Target slot {target_slot_number}")

  # set target slot as unbootable
  subprocess.run(f"abctl --set_unbootable {target_slot_number}", shell=True)

  for partition in update:
    success = False

    for retries in range(10):
      try:
        partition_progress = None if progress is None else lambda n, done=done_size: progress(done + n, total_size)
        flash_partition(target_slot_number, partition, cloudlog, standalone, partition_progress)
        success = True
        break

      except requests.exceptions.RequestException:
        cloudlog.exception("Failed")
        cloudlog.info(f"Failed to download {partition['name']}, retrying ({retries})")
        time.sleep(10)

    if not success:
      cloudlog.info(f"Failed to flash {partition['name']}, aborting")
      raise Exception("Maximum retries exceeded")

    done_size += partition['size']
    if progress is not None:
      progress(done_size, total_size)

  cloudlog.info(f"AGNOS ready on slot {target_slot_number}")


def verify_agnos_update(manifest_path: str, target_slot_number: int) -> bool:
  update = json.load(open(manifest_path))
  return all(verify_partition(target_slot_number, partition) for partition in update)


if __name__ == "__main__":
  import argparse
  import logging

  parser = argparse.ArgumentParser(description="Flash and verify AGNOS update",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--verify", action="store_true", help="Verify and perform swap if update ready")
  parser.add_argument("--swap", action="store_true", help="Verify and perform swap, downloads if necessary")
  parser.add_argument("manifest", help="Manifest json")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)

  target_slot_number = get_target_slot_number()
  if args.verify:
    if verify_agnos_update(args.manifest, target_slot_number):
      swap(args.manifest, target_slot_number, logging)
      exit(0)
    exit(1)
  elif args.swap:
    while not verify_agnos_update(args.manifest, target_slot_number):
      logging.error("Verification failed. Flashing AGNOS")
      flash_agnos_update(args.manifest, target_slot_number, logging, standalone=True)

    logging.warning(f"Verification succeeded. Swapping to slot {target_slot_number}")
    swap(args.manifest, target_slot_number, logging)
  else:
    flash_agnos_update(args.manifest, target_slot_number, logging, standalone=True)
