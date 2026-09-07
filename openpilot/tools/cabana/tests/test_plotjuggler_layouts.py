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
  assert converted["cabana_layout"] == 3
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
  assert yaw["language"] == "python"
  assert "global last_bad_time" in yaw["function"]
  assert "last_bad_time = time" in yaw["function"]
  assert "engage_delay = 5" in yaw["globals"]
  assert "abs" in equations["steering rate limited"]["function"]


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


def test_native_presets_match_imports(tmp_path):
  for xml in sorted(LAYOUTS.glob("*.xml")):
    path = Path(__file__).resolve().parents[1] / "layouts" / (xml.stem + ".json")
    native = json.loads(path.read_text())
    imported = import_layout(xml)
    for equation in native["equations"]:
      equation.pop("legacy_lua_hash", None)
      assert equation["language"] == "python"
    assert native == imported
    result = subprocess.run([str(Path(__file__).with_name("test_cabana")), "--check-layout", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_saved_equations_migrate(tmp_path):
  for xml in sorted(LAYOUTS.glob("*.xml")):
    layout = import_layout(xml)
    layout["cabana_layout"] = 2
    layout["equations"] = [dict(name=e.attrib["name"], source=e.findtext("linked_source", ""),
                               globals=e.findtext("global", ""), function=e.findtext("function", ""),
                               additional=[v.text for v in e.findall("./additional_sources/*")])
                           for e in ET.parse(xml).findall("./customMathEquations/snippet")]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(layout))
    result = subprocess.run([str(Path(__file__).with_name("test_cabana")), "--check-layout", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_lua_is_not_executed(tmp_path):
  path = tmp_path / "custom.xml"
  path.write_text('<root><tabbed_widget><Tab/></tabbed_widget><customMathEquations><snippet name="custom"><function>return 12345</function></snippet></customMathEquations></root>')
  with pytest.raises(ValueError, match="no Python port"):
    import_layout(path)


def test_python_ports_numeric_results():
  from openpilot.tools.cabana.analysis.cabana_equations import compile_equation

  cases = {
    "haversine distance [m]": (0, [0, 0, 90], 10018754.171394622),
    "roll compensated lateral acceleration": (5, [3, 0.2, 0, 1], 43.038),
    "Desired lateral accel (roll compensated)": (5, [3, 0.2], 43.038),
    "Actual lateral accel (roll compensated)": (5, [3, 0.2], 43.038),
    "carState.vEgo kmh": (5, [], 18),
    "carState.vEgo mph": (5, [], 11.1847),
    "opendbc default steering lag": (5, [], 5.2),
    "Zero": (5, [], 0),
    "engaged curvature yaw": (5, [0, 1, 10], 0.5),
    "engaged curvature vehicle model": (5, [0, 1], 5),
    "engaged curvature plan": (5, [0, 1], 5),
    "engaged_accel_actual": (5, [0, 0, 1], 5),
    "engaged_accel_plan": (5, [0, 0, 1], 5),
    "engaged_accel_actuator": (5, [0, 0, 1], 5),
    "steering rate limited": (5, [5, 1, 1], 0),
  }
  for path in (Path(__file__).resolve().parents[1] / "layouts").glob("*.json"):
    for e in json.loads(path.read_text())["equations"]:
      value, additional, expected = cases[e["name"]]
      calc = compile_equation(e["globals"], e["function"], len(additional))
      assert calc(100, value, *additional) == pytest.approx(expected), e["name"]


def test_python_tuning_gates_and_zero_speed():
  from openpilot.tools.cabana.analysis.cabana_equations import compile_equation

  equations = {e["name"]: e for e in import_layout(LAYOUTS / "tuning.xml")["equations"]}
  e = equations["engaged_accel_actual"]
  for brake, gas, enabled in [(1, 0, 1), (0, 1, 1), (0, 0, 0)]:
    calc = compile_equation(e["globals"], e["function"], 3)
    assert calc(100, 5, brake, gas, enabled) == 0
    assert calc(105, 5, 0, 0, 1) == 0
    assert calc(105.01, 5, 0, 0, 1) == 5
  e = equations["engaged curvature yaw"]
  calc = compile_equation(e["globals"], e["function"], 3)
  import math
  assert math.isnan(calc(100, 0.02, 0, 1, 0))
  assert calc(101, 0.02, 0, 1, 20) == pytest.approx(0.001)
