import json
import tempfile
import unittest
from pathlib import Path

from opendbc.car.toyota.values import CAR

from openpilot.selfdrive.car.toyota_tss3_oracle_kit import oracle_kit_compatibility, tss3_fingerprints


class TestToyotaTss3OracleKit(unittest.TestCase):
  def make_tool(self, root: Path, target: str) -> Path:
    tool = root / "tss3-request-signer"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o755)
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "request_signer.json").write_text(json.dumps({"target": {"name": target}}), encoding="utf-8")
    return tool

  def test_accepts_kit_for_any_tss3_vehicle(self):
    # The kit is a per-vehicle signed bundle; the F33 Camry kit is one choice,
    # not the only one. A Corolla-style kit must be equally usable.
    for target in ("camry-8965F3307000", "corolla-8965F1208000", "crown-8965F4102000"):
      with self.subTest(target=target):
        with tempfile.TemporaryDirectory() as td:
          tool = self.make_tool(Path(td), target)
          self.assertEqual(oracle_kit_compatibility(tool), (True, ""))

  def test_rejects_unreadable_metadata_without_crashing(self):
    with tempfile.TemporaryDirectory() as td:
      root = Path(td)
      tool = self.make_tool(root, "camry-8965F3307000")
      for data in (b"{", b'\xff{"target":{}}'):
        with self.subTest(data=data):
          (root / "bundle/request_signer.json").write_bytes(data)
          compatible, detail = oracle_kit_compatibility(tool)
          self.assertFalse(compatible)
          self.assertTrue(detail.startswith("oracle kit metadata invalid:"))

  def test_rejects_missing_metadata(self):
    with tempfile.TemporaryDirectory() as td:
      tool = Path(td) / "tss3-request-signer"
      tool.write_text("#!/bin/sh\n", encoding="utf-8")
      tool.chmod(0o755)
      compatible, detail = oracle_kit_compatibility(tool)
      self.assertFalse(compatible)
      self.assertTrue(detail.startswith("oracle kit metadata invalid:"))

  def test_rejects_unavailable_tool(self):
    with tempfile.TemporaryDirectory() as td:
      compatible, detail = oracle_kit_compatibility(Path(td) / "tss3-request-signer")
      self.assertFalse(compatible)
      self.assertTrue(detail.startswith("oracle tool unavailable:"))

  def test_tss3_fingerprints_follow_opendbc_platform_flags(self):
    # Derived from the platform flags, so the Camry is simply the first TSS3
    # platform and new ports join automatically.
    fingerprints = tss3_fingerprints()
    self.assertIn(CAR.TOYOTA_CAMRY_TSS3, fingerprints)
    self.assertNotIn(CAR.TOYOTA_COROLLA_TSS2, fingerprints)


if __name__ == "__main__":
  unittest.main()
