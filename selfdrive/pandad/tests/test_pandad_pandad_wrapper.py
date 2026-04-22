"""
Unit tests for ``pandad.py`` (host wrapper: firmware path / signature helpers).

Aligns with stakeholder
``docs/testing/testing-plan/TESTING-PLAN.md`` §3.1 (unit / maintainability for
in-scope ``pandad.py``) and risk **R2** (firmware identity before safety-relevant
connect / flash paths).

Maps: R2.
"""

from __future__ import annotations

from types import SimpleNamespace

from openpilot.selfdrive.pandad import pandad as pandad_mod


def _fake_panda_stub(app_fn: str = "panda.bin"):
  cfg = SimpleNamespace(app_fn=app_fn)
  mcu = SimpleNamespace(config=cfg)
  return SimpleNamespace(get_mcu_type=lambda: mcu)


def test_get_expected_signature_returns_firmware_bytes(monkeypatch):
  app_fn = "panda.bin"

  class FakePanda:
    @classmethod
    def get_signature_from_firmware(cls, fn):
      assert app_fn in fn
      return b"\xde\xad\xbe\xef"

  monkeypatch.setattr(pandad_mod, "Panda", FakePanda)
  panda = _fake_panda_stub(app_fn)
  assert pandad_mod.get_expected_signature(panda) == b"\xde\xad\xbe\xef"


def test_get_expected_signature_returns_empty_on_exception(monkeypatch):
  class FakePanda:
    @classmethod
    def get_signature_from_firmware(cls, fn):
      raise OSError("missing firmware")

  monkeypatch.setattr(pandad_mod, "Panda", FakePanda)
  panda = _fake_panda_stub()
  assert pandad_mod.get_expected_signature(panda) == b""
