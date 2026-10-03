import importlib

import openpilot.common.hardware as hardware
import openpilot.common.hardware.hw as hw


def test_default_download_cache_root_is_device_aware(monkeypatch):
  # Regression test for https://github.com/commaai/openpilot/issues/35583:
  # the download cache defaulted to /tmp/comma_download_cache
  # unconditionally, but /tmp is a small tmpfs (150M) on-device, too small
  # for multi-segment route downloads -- large downloads there silently
  # truncate once it fills. On-device (not PC) the cache root must be
  # under /data instead, matching the same PC-vs-device split
  # log_root()/persist_root() already use.
  original_pc = hardware.PC
  try:
    monkeypatch.setattr(hardware, "PC", True)
    importlib.reload(hw)
    assert hw.DEFAULT_DOWNLOAD_CACHE_ROOT.startswith("/tmp/"), \
      "PC should keep using /tmp, which isn't a constrained tmpfs there"

    monkeypatch.setattr(hardware, "PC", False)
    importlib.reload(hw)
    assert not hw.DEFAULT_DOWNLOAD_CACHE_ROOT.startswith("/tmp/"), \
      "on-device, the cache root must not default to the small /tmp tmpfs"
    assert hw.DEFAULT_DOWNLOAD_CACHE_ROOT.startswith("/data/"), \
      "on-device, the cache root should be under the large persistent /data partition"
  finally:
    monkeypatch.setattr(hardware, "PC", original_pc)
    importlib.reload(hw)


def test_download_cache_root_honors_comma_cache_override(monkeypatch):
  # COMMA_CACHE must still win regardless of platform -- this is the
  # existing, already-documented escape hatch (see the issue thread) and
  # must keep working unchanged by this fix.
  monkeypatch.setenv("COMMA_CACHE", "/some/other/path")
  monkeypatch.delenv("OPENPILOT_PREFIX", raising=False)
  assert hw.Paths.download_cache_root() == "/some/other/path/"
