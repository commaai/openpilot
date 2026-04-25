"""
Unit tests for ``flash_panda`` in ``pandad.py``.

Maps: R2, R3.
"""

from __future__ import annotations

import pytest

from openpilot.selfdrive.pandad import pandad as pandad_mod


class FakePandaDevice:
  def __init__(self, *, bootstub=False, signature=b"\x01\x02", internal=False):
    self.bootstub = bootstub
    self._signature = signature
    self._internal = internal
    self.flash_calls = 0
    self.recover_calls = 0
    self.reset_args = []

  def is_internal(self):
    return self._internal

  def get_version(self):
    return "fake-version"

  def get_signature(self):
    return self._signature

  def flash(self):
    self.flash_calls += 1

  def recover(self, reset=True):
    self.recover_calls += 1
    self.reset_args.append(reset)


def _patch_panda_ctor(monkeypatch, fake_dev):
  monkeypatch.setattr(pandad_mod, "Panda", lambda serial: fake_dev)


def test_flash_panda_no_update_needed(monkeypatch):
  fake_dev = FakePandaDevice(bootstub=False, signature=b"\xaa\xbb")
  _patch_panda_ctor(monkeypatch, fake_dev)
  monkeypatch.setattr(pandad_mod, "get_expected_signature", lambda _panda: b"\xaa\xbb")

  out = pandad_mod.flash_panda("SERIAL")

  assert out is fake_dev
  assert fake_dev.flash_calls == 0
  assert fake_dev.recover_calls == 0


def test_flash_panda_updates_when_signature_mismatch(monkeypatch):
  fake_dev = FakePandaDevice(bootstub=False, signature=b"\x00\x00")
  _patch_panda_ctor(monkeypatch, fake_dev)
  monkeypatch.setattr(pandad_mod, "get_expected_signature", lambda _panda: b"\xaa\xbb")

  def fake_flash():
    fake_dev.flash_calls += 1
    fake_dev._signature = b"\xaa\xbb"

  fake_dev.flash = fake_flash

  out = pandad_mod.flash_panda("SERIAL")

  assert out is fake_dev
  assert fake_dev.flash_calls == 1
  assert fake_dev.recover_calls == 0
  assert fake_dev.get_signature() == b"\xaa\xbb"


def test_flash_panda_bootstub_recovery_failure_raises(monkeypatch):
  fake_dev = FakePandaDevice(bootstub=True, signature=b"\x00\x00", internal=False)
  _patch_panda_ctor(monkeypatch, fake_dev)
  monkeypatch.setattr(pandad_mod, "get_expected_signature", lambda _panda: b"\xaa\xbb")

  with pytest.raises(AssertionError):
    pandad_mod.flash_panda("SERIAL")

  # bootstub path should try flash + recover before failing.
  assert fake_dev.flash_calls == 1
  assert fake_dev.recover_calls == 1
  assert fake_dev.reset_args == [True]


def test_flash_panda_signature_mismatch_after_flash_raises(monkeypatch):
  fake_dev = FakePandaDevice(bootstub=False, signature=b"\x00\x00")
  _patch_panda_ctor(monkeypatch, fake_dev)
  monkeypatch.setattr(pandad_mod, "get_expected_signature", lambda _panda: b"\xaa\xbb")

  # Keep signature stale even after flashing to trigger post-flash assertion.
  def fake_flash():
    fake_dev.flash_calls += 1

  fake_dev.flash = fake_flash

  with pytest.raises(AssertionError):
    pandad_mod.flash_panda("SERIAL")

  assert fake_dev.flash_calls == 1
