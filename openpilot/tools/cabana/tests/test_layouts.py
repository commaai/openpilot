import json
import subprocess
from pathlib import Path

import pytest

LAYOUTS = Path(__file__).resolve().parents[1] / "layouts"


def load_layout(name):
  return json.loads((LAYOUTS / name).read_text())


@pytest.mark.parametrize("path", sorted(LAYOUTS.glob("*.json")), ids=lambda p: p.stem)
def test_bundled_layouts(path):
  layout = json.loads(path.read_text())
  assert layout["cabana_layout"] == 3
  assert all(e["language"] == "python" for e in layout["equations"])
  result = subprocess.run([str(Path(__file__).with_name("test_cabana")), "--check-layout", str(path)], capture_output=True, text=True)
  assert result.returncode == 0, result.stdout + result.stderr


def test_tuning_equations_and_tabs():
  layout = load_layout("tuning.json")
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
  layout = load_layout("camera-timings.json")
  chart = layout["tabs"][0][0]
  assert chart["y_min"] == 3.5e7
  assert chart["y_max"] == 6.5e7
  torque = load_layout("max-torque-debug.json")
  speed = next(s for t in torque["tabs"] for c in t for s in c["signals"] if s["path"] == "/carState/vEgo")
  assert speed["scale"] == 2.23694
  assert speed["color"] == "#f14cc1"


@pytest.mark.parametrize("language", [None, "lua"])
def test_non_python_equations_rejected(language, tmp_path):
  layout = load_layout("tuning.json")
  layout["cabana_layout"] = 2
  for equation in layout["equations"]:
    if language is None:
      equation.pop("language")
    else:
      equation["language"] = language
  path = tmp_path / "unsupported.json"
  path.write_text(json.dumps(layout))
  result = subprocess.run([str(Path(__file__).with_name("test_cabana")), "--check-layout", str(path)], capture_output=True, text=True)
  assert result.returncode != 0


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

  equations = {e["name"]: e for e in load_layout("tuning.json")["equations"]}
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
