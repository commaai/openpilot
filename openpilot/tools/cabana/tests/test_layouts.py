import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from openpilot.common.test import OpenpilotTestCase

CABANA_DIR = Path(__file__).parent.parent
LAYOUTS_DIR = CABANA_DIR / "layouts"
MATH_EVAL_PATH = CABANA_DIR / "utils" / "math_eval.py"


class TestCabanaLayouts(OpenpilotTestCase):
  def test_presets_exist(self):
    required_presets = [
      "tuning.json",
      "longitudinal.json",
      "can-states.json",
      "locationd_debug.json",
      "controls_mismatch_debug.json",
      "thermal_debug.json",
      "camera-timings.json",
    ]
    for preset in required_presets:
      p = LAYOUTS_DIR / preset
      self.assertTrue(p.exists(), f"Missing preset {preset} in {LAYOUTS_DIR}")

  def test_preset_json_valid(self):
    preset_files = list(LAYOUTS_DIR.glob("*.json"))
    self.assertGreater(len(preset_files), 5)
    for p in preset_files:
      with open(p, encoding="utf-8") as f:
        data = json.load(f)
      self.assertIn("tabs", data, f"{p} has no tabs")
      self.assertIsInstance(data["tabs"], list)
      self.assertGreater(len(data["tabs"]), 0, f"{p} tabs list is empty")

  def test_custom_python_syntax(self):
    for p in LAYOUTS_DIR.glob("*.json"):
      with open(p, encoding="utf-8") as f:
        data = json.load(f)

      def check_curves(curves):
        for c in curves:
          if not isinstance(c, dict):
            continue
          custom = c.get("custom_python")
          if custom:
            globals_code = custom.get("globals_code", "")
            function_code = custom.get("function_code", "")
            if globals_code.strip():
              ast.parse(globals_code)
            if function_code.strip():
              # Wrap in function to parse return statements
              ast.parse("def __test():\n" + "    " + function_code.replace("\n", "\n    "))

      def walk_node(node):
        if not isinstance(node, dict):
          return
        if "curves" in node:
          check_curves(node["curves"])
        for child in node.get("children", []):
          walk_node(child)

      for tab in data.get("tabs", []):
        walk_node(tab.get("root", {}))

  def test_math_eval_runner(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = Path(tmpdir)
      t = np.linspace(0.0, 10.0, 100, dtype=np.float64)
      v = np.sin(t).astype(np.float64)

      t_path = tmp / "t.bin"
      v_path = tmp / "v.bin"
      t.tofile(t_path)
      v.tofile(v_path)

      manifest = {
        "linked_source": "/test/sin",
        "additional_sources": [],
        "series": [{"path": "/test/sin", "t": str(t_path), "v": str(v_path)}],
      }
      manifest_path = tmp / "manifest.json"
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

      globals_path = tmp / "globals.py"
      globals_path.write_text("scale = 2.0\n", encoding="utf-8")

      code_path = tmp / "code.py"
      code_path.write_text("return value * scale\n", encoding="utf-8")

      out_t_path = tmp / "out_t.bin"
      out_v_path = tmp / "out_v.bin"

      cmd = [
        "python3",
        str(MATH_EVAL_PATH),
        str(manifest_path),
        str(globals_path),
        str(code_path),
        str(out_t_path),
        str(out_v_path),
      ]
      res = subprocess.run(cmd, capture_output=True, text=True, check=False)
      self.assertEqual(res.returncode, 0, f"math_eval failed: {res.stderr}\n{res.stdout}")

      out_t = np.fromfile(out_t_path, dtype=np.float64)
      out_v = np.fromfile(out_v_path, dtype=np.float64)
      self.assertEqual(len(out_t), 100)
      self.assertEqual(len(out_v), 100)
      np.testing.assert_allclose(out_v, v * 2.0)


if __name__ == "__main__":
  unittest.main()
