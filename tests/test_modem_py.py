from __future__ import annotations

import importlib.util
from pathlib import Path


def load_modem_module():
  module_path = Path(__file__).resolve().parents[1] / "system/hardware/tici/modem.py"
  spec = importlib.util.spec_from_file_location("modem_module_under_test", module_path)
  assert spec is not None
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_modem_module_file_exists():
  module_path = Path(__file__).resolve().parents[1] / "system/hardware/tici/modem.py"
  assert module_path.exists()


def test_read_modem_state_missing_returns_none(tmp_path):
  modem = load_modem_module()
  missing = tmp_path / "missing_state.txt"
  assert modem.read_modem_state(missing) is None


def test_write_and_read_modem_state_roundtrip(tmp_path):
  modem = load_modem_module()
  state_path = tmp_path / "modem_state.txt"
  modem.write_modem_state("CONNECTED", state_path)
  assert modem.read_modem_state(state_path) == "CONNECTED"


def test_read_modem_state_trims_whitespace(tmp_path):
  modem = load_modem_module()
  state_path = tmp_path / "modem_state.txt"
  state_path.write_text("  CONNECTING\n")
  assert modem.read_modem_state(state_path) == "CONNECTING"
