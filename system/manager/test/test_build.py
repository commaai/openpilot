from pathlib import Path

import openpilot.system.manager.build as build


class SpinnerStub:
  def update_progress(self, cur: float, total: float) -> None:
    pass

  def close(self) -> None:
    pass


class FakeStderr:
  def __init__(self, lines: list[bytes]):
    self.lines = list(lines)

  def readline(self) -> bytes:
    return self.lines.pop(0) if self.lines else b""

  def read(self) -> bytes:
    remaining = b"\n".join(self.lines)
    self.lines.clear()
    return remaining


class FakePopen:
  def __init__(self, lines: list[bytes], returncode: int):
    self.stderr = FakeStderr(lines)
    self.returncode = None
    self._returncode = returncode

  def poll(self):
    if self.returncode is None and not self.stderr.lines:
      self.returncode = self._returncode
    return self.returncode


def test_clear_stale_scons_cache_lock(tmp_path: Path, monkeypatch):
  lock_path = tmp_path / "config.lock"
  lock_path.write_text("stale")
  monkeypatch.setattr(build, "get_scons_cache_lock_path", lambda: lock_path)

  removed = build.clear_stale_scons_cache_lock([
    f"SConsLockFailure: Timeout waiting for lock on '{lock_path}'".encode(),
  ])

  assert removed
  assert not lock_path.exists()


def test_build_retries_after_stale_scons_cache_lock(tmp_path: Path, monkeypatch):
  lock_path = tmp_path / "config.lock"
  lock_path.write_text("stale")

  popens = [
    FakePopen([f"SConsLockFailure: Timeout waiting for lock on '{lock_path}'".encode()], 1),
    FakePopen([], 0),
  ]

  def fake_popen(*args, **kwargs):
    return popens.pop(0)

  monkeypatch.setattr(build, "CACHE_DIR", tmp_path)
  monkeypatch.setattr(build, "get_scons_cache_lock_path", lambda: lock_path)
  monkeypatch.setattr(build.subprocess, "Popen", fake_popen)
  monkeypatch.setattr(build.os, "cpu_count", lambda: 2)

  build.build(SpinnerStub())

  assert not lock_path.exists()
  assert len(popens) == 0
