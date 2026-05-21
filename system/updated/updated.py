#!/usr/bin/env python3
import datetime
import fcntl
import hashlib
import json
import lzma
import os
import psutil
import shutil
import signal
import stat
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

import requests

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.time_helpers import system_time_valid
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.hardware import HARDWARE


LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/updater.lock")
MANIFEST_URL = os.getenv("UPDATER_MANIFEST_URL", "https://api.commadotai.com/v1/updater/manifest")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")

VERIFY_CURRENT_SLOT = os.getenv("UPDATER_VERIFY_CURRENT_SLOT", "1") != "0"
DOWNLOAD_TIMEOUT = 60
PARTITION_RETRIES = 10
CHUNK_SIZE = 1024 * 1024
SLOTS = ("a", "b")

SPARSE_HEADER = struct.Struct("<IHHHHIIII")
SPARSE_CHUNK_HEADER = struct.Struct("<HHII")
SPARSE_MAGIC = 0xED26FF3A
SPARSE_RAW = 0xCAC1
SPARSE_FILL = 0xCAC2
SPARSE_DONT_CARE = 0xCAC3
SPARSE_CRC32 = 0xCAC4


class UserRequest:
  NONE = 0
  CHECK = 1
  FETCH = 2


class WaitTimeHelper:
  def __init__(self) -> None:
    self.ready_event = threading.Event()
    self.user_request = UserRequest.NONE
    signal.signal(signal.SIGHUP, self.update_now)
    signal.signal(signal.SIGUSR1, self.check_now)

  def update_now(self, signum: int, frame) -> None:
    cloudlog.info("caught SIGHUP, downloading update")
    self.user_request = UserRequest.FETCH
    self.ready_event.set()

  def check_now(self, signum: int, frame) -> None:
    cloudlog.info("caught SIGUSR1, checking for updates")
    self.user_request = UserRequest.CHECK
    self.ready_event.set()

  def sleep(self, t: float) -> None:
    self.ready_event.wait(timeout=t)


def run(cmd: list[str], **kwargs) -> str:
  return subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf8", **kwargs)


def write_time_to_param(params: Params, key: str) -> None:
  params.put(key, datetime.datetime.now(datetime.UTC).replace(tzinfo=None), block=True)


def slot_number(slot: str) -> int:
  assert slot in SLOTS
  return 0 if slot == "a" else 1


def other_slot(slot: str) -> str:
  assert slot in SLOTS
  return "b" if slot == "a" else "a"


def partition_size(partition: dict[str, Any]) -> int:
  size = partition.get("size")
  if not isinstance(size, int):
    raise ValueError(f"invalid size for {partition.get('name')}: {size}")
  return size


def partition_hash(partition: dict[str, Any]) -> str:
  h = partition.get("hash_raw", partition.get("hash"))
  if not isinstance(h, str):
    raise ValueError(f"missing hash for {partition.get('name')}")
  return h.lower()


def hash_zeros(digest: "hashlib._Hash", size: int) -> None:
  zeros = b"\x00" * CHUNK_SIZE
  while size > 0:
    n = min(size, len(zeros))
    digest.update(zeros[:n])
    size -= n


def sha256_path(path: str, size: int) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    remaining = size
    while remaining > 0:
      chunk = f.read(min(CHUNK_SIZE, remaining))
      if len(chunk) == 0:
        return ""
      digest.update(chunk)
      remaining -= len(chunk)
  return digest.hexdigest().lower()


def ensure_regular_file_size(f: BinaryIO, size: int) -> None:
  mode = os.fstat(f.fileno()).st_mode
  if stat.S_ISREG(mode):
    f.truncate(size)


def url_is_xz(url: str, partition: dict[str, Any]) -> bool:
  compression = partition.get("compression")
  path = urlparse(url).path
  return compression == "xz" or (compression is None and path.endswith((".xz", ".lzma")))


class ImageReader:
  def __init__(self, url: str, compressed: bool):
    self.url = url
    self.compressed = compressed
    self.sha256 = hashlib.sha256()
    self.buf = b""
    self.eof = False
    self.response: requests.Response | None = None
    self.decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO) if compressed else None

    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
      self.source = open(parsed.path if parsed.scheme == "file" else url, "rb")
    elif parsed.scheme in ("http", "https"):
      self.response = requests.get(url, stream=True, headers={"Accept-Encoding": None}, timeout=DOWNLOAD_TIMEOUT)
      self.response.raise_for_status()
      self.source = self.response.raw
    else:
      raise ValueError(f"unsupported image URL scheme: {url}")

  def close(self) -> None:
    self.source.close()
    if self.response is not None:
      self.response.close()

  def __enter__(self) -> "ImageReader":
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()

  def read(self, size: int) -> bytes:
    if size <= 0:
      return b""

    if self.decompressor is None:
      data = self.source.read(size)
      self.sha256.update(data)
      return data

    while len(self.buf) < size and not self.eof:
      if self.decompressor.needs_input:
        compressed = self.source.read(CHUNK_SIZE)
        if len(compressed) == 0:
          self.eof = True
          break
      else:
        compressed = b""

      self.buf += self.decompressor.decompress(compressed, max_length=size - len(self.buf))
      if self.decompressor.eof:
        self.eof = True

    data, self.buf = self.buf[:size], self.buf[size:]
    self.sha256.update(data)
    return data


def read_exact(reader: ImageReader, size: int) -> bytes:
  data = reader.read(size)
  if len(data) != size:
    raise EOFError(f"short image read: wanted {size}, got {len(data)}")
  return data


def copy_exact(reader: ImageReader, out: BinaryIO, size: int, digest: "hashlib._Hash") -> None:
  remaining = size
  while remaining > 0:
    data = read_exact(reader, min(CHUNK_SIZE, remaining))
    digest.update(data)
    out.write(data)
    remaining -= len(data)


def write_pattern(out: BinaryIO, digest: "hashlib._Hash", pattern: bytes, size: int) -> None:
  block = (pattern * ((CHUNK_SIZE // len(pattern)) + 1))[:CHUNK_SIZE]
  remaining = size
  while remaining > 0:
    data = block[:min(len(block), remaining)]
    digest.update(data)
    out.write(data)
    remaining -= len(data)


def write_sparse_image(reader: ImageReader, out: BinaryIO, digest: "hashlib._Hash") -> None:
  header = read_exact(reader, SPARSE_HEADER.size)
  magic, major, minor, file_hdr_sz, chunk_hdr_sz, block_sz, total_blocks, total_chunks, _ = SPARSE_HEADER.unpack(header)
  if magic != SPARSE_MAGIC or major != 1 or minor != 0:
    raise ValueError("invalid Android sparse image")
  if file_hdr_sz < SPARSE_HEADER.size or chunk_hdr_sz < SPARSE_CHUNK_HEADER.size:
    raise ValueError("invalid Android sparse header size")

  read_exact(reader, file_hdr_sz - SPARSE_HEADER.size)
  for _ in range(total_chunks):
    chunk_header = read_exact(reader, chunk_hdr_sz)
    chunk_type, _, chunk_blocks, total_sz = SPARSE_CHUNK_HEADER.unpack(chunk_header[:SPARSE_CHUNK_HEADER.size])
    data_sz = total_sz - chunk_hdr_sz
    out_sz = chunk_blocks * block_sz

    if chunk_type == SPARSE_RAW:
      if data_sz != out_sz:
        raise ValueError("invalid sparse raw chunk size")
      copy_exact(reader, out, data_sz, digest)
    elif chunk_type == SPARSE_FILL:
      if data_sz != 4:
        raise ValueError("invalid sparse fill chunk size")
      write_pattern(out, digest, read_exact(reader, 4), out_sz)
    elif chunk_type == SPARSE_DONT_CARE:
      read_exact(reader, data_sz)
      hash_zeros(digest, out_sz)
      out.seek(out_sz, os.SEEK_CUR)
    elif chunk_type == SPARSE_CRC32:
      read_exact(reader, data_sz)
    else:
      raise ValueError(f"unknown sparse chunk type: {chunk_type}")

  expected_size = total_blocks * block_sz
  if out.tell() != expected_size:
    raise ValueError(f"sparse output size mismatch: {out.tell()} != {expected_size}")


@dataclass(frozen=True)
class Manifest:
  version: str
  description: str
  release_notes: bytes
  partitions: list[dict[str, Any]]

  @classmethod
  def parse(cls, data: Any) -> "Manifest":
    if isinstance(data, list):
      raise ValueError("manifest must include a top-level version")

    version = data.get("version") or data.get("os_version")
    partitions = data.get("partitions")
    if not isinstance(version, str) or len(version) == 0:
      raise ValueError("manifest missing version")
    if not isinstance(partitions, list) or len(partitions) == 0:
      raise ValueError("manifest missing partitions")

    for partition in partitions:
      if not isinstance(partition, dict) or not isinstance(partition.get("name"), str):
        raise ValueError("invalid partition manifest entry")
      partition_size(partition)
      partition_hash(partition)
      if not isinstance(partition.get("url"), str):
        raise ValueError(f"missing URL for {partition.get('name')}")

    notes = data.get("release_notes", b"")
    if isinstance(notes, str):
      notes = notes.encode("utf8")
    return cls(version, data.get("description") or version, notes, partitions)


class SlotBackend:
  def current_slot(self) -> str:
    raise NotImplementedError

  def partition_path(self, slot: str, partition: dict[str, Any]) -> str:
    raise NotImplementedError

  def set_unbootable(self, slot: str) -> None:
    raise NotImplementedError

  def set_active(self, slot: str) -> None:
    raise NotImplementedError

  def mark_successful(self) -> None:
    raise NotImplementedError

  def os_version(self) -> str:
    version = HARDWARE.get_os_version()
    return version or ""

  def target_slot(self) -> str:
    return other_slot(self.current_slot())


class RealSlotBackend(SlotBackend):
  def current_slot(self) -> str:
    current = run(["abctl", "--boot_slot"]).strip()
    if current not in ("_a", "_b"):
      raise RuntimeError(f"unknown boot slot: {current}")
    return current[-1]

  def partition_path(self, slot: str, partition: dict[str, Any]) -> str:
    path = f"/dev/disk/by-partlabel/{partition['name']}"
    if partition.get("has_ab", True):
      path += f"_{slot}"
    return path

  def set_unbootable(self, slot: str) -> None:
    run(["abctl", "--set_unbootable", str(slot_number(slot))])

  def set_active(self, slot: str) -> None:
    target = str(slot_number(slot))
    last_output = ""
    for _ in range(10):
      last_output = run(["abctl", "--set_active", target])
      if "No such file or directory" not in last_output and "lun as boot lun" in last_output:
        cloudlog.info(f"slot {slot} active for next boot: {last_output}")
        return
      cloudlog.error(f"failed to set active slot {slot}: {last_output}")
      time.sleep(1)
    raise RuntimeError(f"failed to set active slot {slot}: {last_output}")

  def mark_successful(self) -> None:
    run(["sudo", "abctl", "--set_success"])


class FakeSlotBackend(SlotBackend):
  def __init__(self, root: str):
    self.root = Path(root)
    self.root.mkdir(parents=True, exist_ok=True)
    if not self.state_path.is_file():
      self.write_state({"current_slot": "a", "active_slot": "a", "unbootable": [], "successful": []})
    for slot in SLOTS:
      (self.root / f"slot_{slot}").mkdir(exist_ok=True)

  @property
  def state_path(self) -> Path:
    return self.root / "state.json"

  def read_state(self) -> dict[str, Any]:
    return json.loads(self.state_path.read_text())

  def write_state(self, state: dict[str, Any]) -> None:
    self.state_path.write_text(json.dumps(state, sort_keys=True))

  def current_slot(self) -> str:
    return self.read_state()["current_slot"]

  def partition_path(self, slot: str, partition: dict[str, Any]) -> str:
    path = self.root / f"slot_{slot}" / partition["name"]
    path.parent.mkdir(exist_ok=True)
    return str(path)

  def set_unbootable(self, slot: str) -> None:
    state = self.read_state()
    state["unbootable"] = sorted(set(state.get("unbootable", [])) | {slot})
    self.write_state(state)

  def set_active(self, slot: str) -> None:
    state = self.read_state()
    state["active_slot"] = slot
    state["unbootable"] = [s for s in state.get("unbootable", []) if s != slot]
    self.write_state(state)

  def mark_successful(self) -> None:
    state = self.read_state()
    state["successful"] = sorted(set(state.get("successful", [])) | {state["current_slot"]})
    self.write_state(state)

  def os_version(self) -> str:
    version_path = self.root / f"slot_{self.current_slot()}" / "VERSION"
    return version_path.read_text().strip() if version_path.is_file() else ""

  def set_current_slot(self, slot: str) -> None:
    state = self.read_state()
    state["current_slot"] = slot
    state["active_slot"] = slot
    self.write_state(state)

  def set_slot_version(self, slot: str, version: str) -> None:
    (self.root / f"slot_{slot}" / "VERSION").write_text(version)


def make_backend() -> SlotBackend:
  fake_root = os.getenv("UPDATER_FAKE_ROOT")
  return FakeSlotBackend(fake_root) if fake_root else RealSlotBackend()


def open_partition(path: str) -> BinaryIO:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  return open(path, "r+b" if os.path.exists(path) else "w+b")


def clear_partition_hash(backend: SlotBackend, slot: str, partition: dict[str, Any]) -> None:
  path = backend.partition_path(slot, partition)
  with open_partition(path) as out:
    out.seek(partition_size(partition))
    out.write(b"\x00" * 64)
  os.sync()


def write_partition_hash(backend: SlotBackend, slot: str, partition: dict[str, Any]) -> None:
  path = backend.partition_path(slot, partition)
  with open_partition(path) as out:
    out.seek(partition_size(partition))
    out.write(partition_hash(partition).encode())
  os.sync()


def verify_partition(backend: SlotBackend, slot: str, partition: dict[str, Any], force_full_check: bool = False) -> bool:
  path = backend.partition_path(slot, partition)
  if not os.path.exists(path):
    return False

  size = partition_size(partition)
  expected = partition_hash(partition)
  if partition.get("full_check", True) or force_full_check:
    return sha256_path(path, size) == expected

  with open(path, "rb") as f:
    f.seek(size)
    return f.read(64) == expected.encode()


def verify_slot(backend: SlotBackend, slot: str, manifest: Manifest, force_full_check: bool = False) -> bool:
  return all(verify_partition(backend, slot, partition, force_full_check) for partition in manifest.partitions)


def flash_partition(backend: SlotBackend, slot: str, partition: dict[str, Any]) -> None:
  name = partition["name"]
  if verify_partition(backend, slot, partition):
    cloudlog.info(f"{name} already flashed on slot {slot}")
    return

  if not partition.get("full_check", True):
    clear_partition_hash(backend, slot, partition)

  path = backend.partition_path(slot, partition)
  url = partition["url"]
  expected_raw_hash = partition_hash(partition)
  raw_hash = hashlib.sha256()

  with ImageReader(url, url_is_xz(url, partition)) as image, open_partition(path) as out:
    out.seek(0)
    if partition.get("sparse", False):
      write_sparse_image(image, out, raw_hash)
    else:
      copy_exact(image, out, partition_size(partition), raw_hash)

    if len(image.read(1)) != 0:
      raise ValueError(f"{name} image is larger than manifest size")

    size = out.tell()
    if size != partition_size(partition):
      raise ValueError(f"{name} size mismatch: {size} != {partition_size(partition)}")
    ensure_regular_file_size(out, size)
    os.sync()

  if raw_hash.hexdigest().lower() != expected_raw_hash:
    raise ValueError(f"{name} raw hash mismatch")

  expected_image_hash = partition.get("hash")
  if isinstance(expected_image_hash, str) and image.sha256.hexdigest().lower() != expected_image_hash.lower():
    raise ValueError(f"{name} image hash mismatch")

  if not partition.get("full_check", True):
    write_partition_hash(backend, slot, partition)


def cleanup_old_overlay(basedir: str = BASEDIR, staging_root: str = STAGING_ROOT) -> None:
  staging = Path(staging_root)
  if str(staging) in ("", ".") or staging.resolve() in (Path("/"), Path("/data")):
    raise ValueError(f"refusing to remove unsafe staging root: {staging}")

  merged = staging / "merged"
  if os.path.ismount(merged):
    run(["sudo", "umount", "-l", str(merged)])
  shutil.rmtree(staging, ignore_errors=True)

  for name in (".overlay_init", ".overlay_consistent"):
    Path(basedir, name).unlink(missing_ok=True)


class Updater:
  def __init__(self, params: Params | None = None, backend: SlotBackend | None = None, manifest_url: str = MANIFEST_URL):
    self.params = params or Params()
    self.backend = backend or make_backend()
    self.manifest_url = manifest_url
    self.verified_current_versions: set[str] = set()

  def fetch_manifest(self) -> Manifest:
    parsed = urlparse(self.manifest_url)
    if parsed.scheme in ("", "file"):
      path = parsed.path if parsed.scheme == "file" else self.manifest_url
      return Manifest.parse(json.loads(Path(path).read_text()))

    query = {
      "dongle_id": self.params.get("DongleId") or "",
      "device": HARDWARE.get_device_type(),
      "version": self.backend.os_version(),
    }
    r = requests.get(self.manifest_url, params=query, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    return Manifest.parse(r.json())

  def set_descriptions(self, manifest: Manifest) -> None:
    self.params.put("UpdaterCurrentDescription", self.backend.os_version(), block=True)
    self.params.put("UpdaterCurrentReleaseNotes", b"", block=True)
    self.params.put("UpdaterNewDescription", manifest.description, block=True)
    self.params.put("UpdaterNewReleaseNotes", manifest.release_notes, block=True)
    self.params.put("UpdaterTargetBranch", "release", block=True)
    self.params.put("UpdaterAvailableBranches", "release", block=True)

  def current_slot_integrity_ok(self, manifest: Manifest) -> bool:
    if not VERIFY_CURRENT_SLOT or self.backend.os_version() != manifest.version:
      return True

    key = f"{self.backend.current_slot()}:{manifest.version}"
    if key in self.verified_current_versions:
      return True

    self.params.put("UpdaterState", "verifying current slot...", block=True)
    ok = verify_slot(self.backend, self.backend.current_slot(), manifest, force_full_check=True)
    self.verified_current_versions.add(key)
    if not ok:
      cloudlog.error("current slot failed manifest verification")
    return ok

  def mark_ready(self, manifest: Manifest) -> None:
    self.params.put_bool("UpdaterFetchAvailable", False, block=True)
    self.params.put_bool("UpdateAvailable", True, block=True)
    write_time_to_param(self.params, "LastUpdateTime")
    self.set_descriptions(manifest)

  def install_update(self, manifest: Manifest) -> None:
    target = self.backend.target_slot()
    cloudlog.info(f"installing {manifest.version} to slot {target}")

    self.backend.set_unbootable(target)
    self.params.put_bool("UpdateAvailable", False, block=True)

    for partition in manifest.partitions:
      self.params.put("UpdaterState", f"downloading {partition['name']}...", block=True)
      for retry in range(PARTITION_RETRIES):
        try:
          flash_partition(self.backend, target, partition)
          break
        except requests.RequestException:
          cloudlog.exception(f"failed to download {partition['name']}, retrying ({retry})")
          time.sleep(10)
      else:
        raise RuntimeError(f"failed to flash {partition['name']}")

    self.params.put("UpdaterState", "verifying update...", block=True)
    if not verify_slot(self.backend, target, manifest, force_full_check=True):
      raise RuntimeError(f"slot {target} failed verification")

    self.params.put("UpdaterState", "activating update...", block=True)
    self.backend.set_active(target)
    if isinstance(self.backend, FakeSlotBackend):
      self.backend.set_slot_version(target, manifest.version)
    self.mark_ready(manifest)

  def update_once(self, download: bool = True) -> None:
    self.params.put("UpdaterState", "checking...", block=True)
    manifest = self.fetch_manifest()
    self.set_descriptions(manifest)

    current_ok = self.current_slot_integrity_ok(manifest)
    update_needed = self.backend.os_version() != manifest.version or not current_ok
    self.params.put_bool("UpdaterFetchAvailable", update_needed, block=True)

    if not update_needed:
      self.params.put_bool("UpdateAvailable", False, block=True)
      write_time_to_param(self.params, "LastUpdateTime")
      cloudlog.info(f"up to date on {manifest.version}")
      return

    target = self.backend.target_slot()
    if not download:
      cloudlog.info(f"update available: {self.backend.os_version()} -> {manifest.version}")
      return

    if verify_slot(self.backend, target, manifest):
      cloudlog.info(f"slot {target} already contains {manifest.version}, activating")
      self.backend.set_active(target)
      self.mark_ready(manifest)
      return

    self.install_update(manifest)


def set_update_alert(failed_count: int, exception: str | None) -> None:
  set_offroad_alert("Offroad_UpdateFailed", False)
  if failed_count > 3 and exception is not None:
    set_offroad_alert("Offroad_UpdateFailed", True, extra_text=exception)


def should_download(params: Params, request: int) -> bool:
  if request == UserRequest.CHECK:
    return False
  if request == UserRequest.FETCH:
    return True

  last_fetch = params.get("UpdaterLastFetchTime")
  timed_out = last_fetch is None or (datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - last_fetch > datetime.timedelta(days=3))
  return not params.get_bool("NetworkMetered") or timed_out


def main() -> None:
  params = Params()
  if params.get_bool("DisableUpdates"):
    cloudlog.warning("updates are disabled by the DisableUpdates param")
    return

  with open(LOCK_FILE, "w") as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
      raise RuntimeError("couldn't get updater lock; is another instance running?") from e

    proc = psutil.Process()
    if psutil.LINUX:
      proc.ionice(psutil.IOPRIO_CLASS_BE, value=7)

    cleanup_old_overlay()
    updater = Updater(params)
    wait_helper = WaitTimeHelper()
    failed_count = 0

    try:
      updater.backend.mark_successful()
    except Exception:
      cloudlog.exception("failed to mark boot successful")

    params.put("UpdaterState", "idle", block=True)
    params.put_bool("UpdateAvailable", False, block=True)

    first_run = True
    while True:
      wait_helper.ready_event.clear()
      exception = None
      download = should_download(params, wait_helper.user_request)

      try:
        if not system_time_valid() or first_run:
          first_run = False
          wait_helper.sleep(60)
          continue

        updater.update_once(download)
        if download:
          write_time_to_param(params, "UpdaterLastFetchTime")
        failed_count = 0
        params.remove("LastUpdateException")
      except subprocess.CalledProcessError as e:
        failed_count += 1
        cloudlog.event("update process failed", cmd=e.cmd, output=e.output, returncode=e.returncode)
        exception = f"command failed: {e.cmd}\n{e.output}"
        params.put("LastUpdateException", exception, block=True)
      except Exception as e:
        failed_count += 1
        cloudlog.exception("uncaught updated exception")
        exception = str(e)
        params.put("LastUpdateException", exception, block=True)
      finally:
        params.put("UpdateFailedCount", failed_count, block=True)
        params.put("UpdaterState", "idle", block=True)
        set_update_alert(failed_count, exception)
        wait_helper.user_request = UserRequest.NONE

      wait_helper.sleep(5 * 60 if failed_count > 0 else 1.5 * 60 * 60)


if __name__ == "__main__":
  main()
