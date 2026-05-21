import datetime
import hashlib
import json
import lzma
from pathlib import Path

import pytest

from openpilot.system.updated.updated import (
  FakeSlotBackend,
  SPARSE_CHUNK_HEADER,
  SPARSE_HEADER,
  SPARSE_MAGIC,
  SPARSE_RAW,
  Updater,
  cleanup_old_overlay,
  should_download,
  verify_partition,
)


class MemoryParams:
  def __init__(self, values=None):
    self.values = dict(values or {})

  def get(self, key, return_default=False):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key))

  def put(self, key, value, block=False):
    self.values[key] = value

  def put_bool(self, key, value, block=False):
    self.values[key] = bool(value)

  def remove(self, key):
    self.values.pop(key, None)


def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest().lower()


def write_manifest(tmp_path: Path, version: str, partitions: list[dict]) -> str:
  manifest = {
    "version": version,
    "description": f"release {version}",
    "release_notes": f"notes for {version}",
    "partitions": partitions,
  }
  path = tmp_path / "manifest.json"
  path.write_text(json.dumps(manifest))
  return path.as_uri()


def make_partition(tmp_path: Path, name: str, data: bytes, *, full_check=True, sparse=False, compress=False, hash_raw=None) -> dict:
  image = make_sparse(data) if sparse else data
  path = tmp_path / f"{name}.img"
  path.write_bytes(lzma.compress(image) if compress else image)
  if compress:
    path = path.rename(path.with_suffix(".img.xz"))

  return {
    "name": name,
    "url": path.as_uri(),
    "hash": sha256(image),
    "hash_raw": hash_raw or sha256(data),
    "size": len(data),
    "sparse": sparse,
    "full_check": full_check,
    "has_ab": True,
  }


def make_sparse(data: bytes) -> bytes:
  block_size = 4096
  assert len(data) % block_size == 0
  blocks = len(data) // block_size
  header = SPARSE_HEADER.pack(SPARSE_MAGIC, 1, 0, SPARSE_HEADER.size, SPARSE_CHUNK_HEADER.size, block_size, blocks, 1, 0)
  chunk = SPARSE_CHUNK_HEADER.pack(SPARSE_RAW, 0, blocks, SPARSE_CHUNK_HEADER.size + len(data))
  return header + chunk + data


def write_current_slot(backend: FakeSlotBackend, version: str, partition: dict, data: bytes) -> None:
  backend.set_slot_version("a", version)
  path = Path(backend.partition_path("a", partition))
  path.write_bytes(data)
  if not partition.get("full_check", True):
    with open(path, "ab") as f:
      f.write(partition["hash_raw"].encode())


def test_successful_update_activates_verified_inactive_slot(tmp_path):
  data = b"new-system" * 100
  partition = make_partition(tmp_path, "system", data)
  manifest_url = write_manifest(tmp_path, "2.0", [partition])
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  backend.set_slot_version("a", "1.0")
  params = MemoryParams({"DongleId": "0000000000000000"})

  Updater(params, backend, manifest_url).update_once(download=True)

  state = backend.read_state()
  assert state["current_slot"] == "a"
  assert state["active_slot"] == "b"
  assert state["unbootable"] == []
  assert Path(backend.partition_path("b", partition)).read_bytes() == data
  assert params.get_bool("UpdateAvailable")
  assert not params.get_bool("UpdaterFetchAvailable")
  assert params.get("UpdaterNewDescription") == "release 2.0"


def test_check_only_does_not_flash_or_activate(tmp_path):
  data = b"new-system"
  partition = make_partition(tmp_path, "system", data)
  manifest_url = write_manifest(tmp_path, "2.0", [partition])
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  backend.set_slot_version("a", "1.0")
  params = MemoryParams()

  Updater(params, backend, manifest_url).update_once(download=False)

  state = backend.read_state()
  assert state["active_slot"] == "a"
  assert not Path(backend.partition_path("b", partition)).exists()
  assert params.get_bool("UpdaterFetchAvailable")
  assert not params.get_bool("UpdateAvailable")


def test_check_only_does_not_activate_ready_slot(tmp_path):
  data = b"ready-system"
  partition = make_partition(tmp_path, "system", data)
  manifest_url = write_manifest(tmp_path, "2.0", [partition])
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  backend.set_slot_version("a", "1.0")
  Path(backend.partition_path("b", partition)).write_bytes(data)
  params = MemoryParams()

  Updater(params, backend, manifest_url).update_once(download=False)

  assert backend.read_state()["active_slot"] == "a"
  assert params.get_bool("UpdaterFetchAvailable")
  assert not params.get_bool("UpdateAvailable")


def test_no_update_verifies_current_slot(tmp_path):
  data = b"current-system"
  partition = make_partition(tmp_path, "system", data)
  manifest_url = write_manifest(tmp_path, "2.0", [partition])
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  write_current_slot(backend, "2.0", partition, data)
  params = MemoryParams()

  Updater(params, backend, manifest_url).update_once(download=True)

  assert backend.read_state()["active_slot"] == "a"
  assert not params.get_bool("UpdateAvailable")
  assert not params.get_bool("UpdaterFetchAvailable")


def test_same_version_with_bad_current_slot_repairs_from_manifest(tmp_path):
  data = b"healthy-system"
  partition = make_partition(tmp_path, "system", data)
  manifest_url = write_manifest(tmp_path, "2.0", [partition])
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  write_current_slot(backend, "2.0", partition, b"corrupt-system")
  params = MemoryParams()

  Updater(params, backend, manifest_url).update_once(download=True)

  assert backend.read_state()["active_slot"] == "b"
  assert Path(backend.partition_path("b", partition)).read_bytes() == data
  assert params.get_bool("UpdateAvailable")


def test_hash_mismatch_leaves_target_unbootable(tmp_path):
  partition = make_partition(tmp_path, "system", b"downloaded", hash_raw=sha256(b"expected"))
  manifest_url = write_manifest(tmp_path, "2.0", [partition])
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  backend.set_slot_version("a", "1.0")
  params = MemoryParams()

  with pytest.raises(ValueError, match="raw hash mismatch"):
    Updater(params, backend, manifest_url).update_once(download=True)

  state = backend.read_state()
  assert state["active_slot"] == "a"
  assert state["unbootable"] == ["b"]
  assert not params.get_bool("UpdateAvailable")


def test_fast_verified_target_is_activated_without_redownload(tmp_path):
  data = b"system-with-sidecar-hash"
  partition = make_partition(tmp_path, "system", data, full_check=False)
  manifest_url = write_manifest(tmp_path, "2.0", [partition])
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  backend.set_slot_version("a", "1.0")
  params = MemoryParams()

  updater = Updater(params, backend, manifest_url)
  updater.update_once(download=True)
  assert verify_partition(backend, "b", partition)

  Path(tmp_path / "system.img").write_bytes(b"this would fail if redownloaded")
  updater.update_once(download=True)

  assert backend.read_state()["active_slot"] == "b"


def test_sparse_xz_partition_uses_same_flash_path(tmp_path):
  data = b"a" * 4096
  partition = make_partition(tmp_path, "system", data, sparse=True, compress=True)
  manifest_url = write_manifest(tmp_path, "2.0", [partition])
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  backend.set_slot_version("a", "1.0")

  Updater(MemoryParams(), backend, manifest_url).update_once(download=True)

  assert Path(backend.partition_path("b", partition)).read_bytes() == data
  assert backend.read_state()["active_slot"] == "b"


def test_cleanup_old_overlay_state(tmp_path):
  basedir = tmp_path / "openpilot"
  staging = tmp_path / "safe_staging"
  basedir.mkdir()
  (basedir / ".overlay_init").touch()
  (basedir / ".overlay_consistent").touch()
  (staging / "finalized").mkdir(parents=True)

  cleanup_old_overlay(str(basedir), str(staging))

  assert not staging.exists()
  assert not (basedir / ".overlay_init").exists()
  assert not (basedir / ".overlay_consistent").exists()


def test_cleanup_refuses_unsafe_staging_root(tmp_path):
  with pytest.raises(ValueError):
    cleanup_old_overlay(str(tmp_path), ".")


def test_manifest_api_passes_dongle_id(monkeypatch, tmp_path):
  calls = {}

  class Response:
    def raise_for_status(self):
      pass

    def json(self):
      return {
        "version": "2.0",
        "partitions": [{
          "name": "system",
          "url": "https://example.com/system.img",
          "hash": sha256(b"system"),
          "size": len(b"system"),
        }],
      }

  def get(url, params, timeout):
    calls["url"] = url
    calls["params"] = params
    calls["timeout"] = timeout
    return Response()

  monkeypatch.setattr("openpilot.system.updated.updated.requests.get", get)
  backend = FakeSlotBackend(str(tmp_path / "slots"))
  backend.set_slot_version("a", "1.0")

  manifest = Updater(MemoryParams({"DongleId": "abc123"}), backend, "https://example.com/manifest").fetch_manifest()

  assert manifest.version == "2.0"
  assert calls["url"] == "https://example.com/manifest"
  assert calls["params"]["dongle_id"] == "abc123"
  assert calls["params"]["version"] == "1.0"


def test_should_download_respects_metered_network():
  params = MemoryParams({"NetworkMetered": True, "UpdaterLastFetchTime": datetime.datetime.now(datetime.UTC).replace(tzinfo=None)})

  assert not should_download(params, 0)
  assert should_download(params, 2)
