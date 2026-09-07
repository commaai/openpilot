import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openpilot.tools.cabana.analysis.import_layout import import_layout

LAYOUTS = Path(__file__).resolve().parents[2] / "plotjuggler/layouts"


@pytest.mark.parametrize("path", sorted(LAYOUTS.glob("*.xml")), ids=lambda p: p.stem)
def test_bundled_layouts(path, tmp_path):
  original = ET.parse(path)
  converted = import_layout(path)
  native_layout = tmp_path / "converted.json"
  native_layout.write_text(json.dumps(converted))
  result = subprocess.run([str(Path(__file__).with_name("test_cabana")), "--check-layout", str(native_layout)], capture_output=True, text=True)
  assert result.returncode == 0, result.stdout + result.stderr
  assert converted["cabana_layout"] == 2
  assert len(converted["tabs"]) == len(original.findall(".//Tab"))
  assert sum(len(t) for t in converted["tabs"]) == len(original.findall(".//plot"))
  assert len(converted["equations"]) == len(original.findall("./customMathEquations/snippet"))
  assert [s["path"] for t in converted["tabs"] for c in t for s in c["signals"]] == [
    c.attrib["name"] for c in original.findall(".//plot/curve")]


def test_tuning_equations_and_tabs():
  layout = import_layout(LAYOUTS / "tuning.xml")
  assert layout["tab_names"] == ["Lateral", "Longitudinal", "Lateral Debug"]
  equations = {e["name"]: e for e in layout["equations"]}
  yaw = equations["engaged curvature yaw"]
  assert yaw["source"] == "/carControl/angularVelocity/2"
  assert yaw["additional"] == ["/carState/steeringPressed", "/carControl/enabled", "/carState/vEgo"]
  assert "last_bad_time = time" in yaw["function"]
  assert "engage_delay = 5" in yaw["globals"]
  assert "math.abs" in equations["steering rate limited"]["function"]


def test_colors_limits_and_scaling():
  layout = import_layout(LAYOUTS / "camera-timings.xml")
  chart = layout["tabs"][0][0]
  assert chart["y_min"] == 3.5e7
  assert chart["y_max"] == 6.5e7
  torque = import_layout(LAYOUTS / "max-torque-debug.xml")
  speed = next(s for t in torque["tabs"] for c in t for s in c["signals"] if s["path"] == "/carState/vEgo")
  assert speed["scale"] == 2.23694
  assert speed["color"] == "#f14cc1"


def test_unsupported_plot_is_not_silently_dropped(tmp_path):
  path = tmp_path / "unsupported.xml"
  path.write_text('<root><tabbed_widget><Tab><DockArea><plot mode="XY"/></DockArea></Tab></tabbed_widget></root>')
  with pytest.raises(ValueError, match="time-series"):
    import_layout(path)
