"""Import time-series curve groups from a PlotJuggler XML layout into Cabana.

Unsupported math and plot modes are rejected rather than producing different data.
Window geometry, tabs, curve colors, and axis limits are not transferred.
"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def convert_layout(xml: str) -> dict:
  root = ET.fromstring(xml)
  if root.tag != "root":
    raise ValueError("Expected a PlotJuggler <root> layout")
  equations = {item.get("name") for item in root.findall("./customMathEquations/*")}
  plots = []
  for plot in root.iter("plot"):
    if plot.get("mode", "TimeSeries") != "TimeSeries":
      raise ValueError("Only TimeSeries plots are supported; XY plots require manual migration")
    curves = []
    for element in plot.findall("curve"):
      name = element.get("name", "")
      if name in equations or not name.startswith("/"):
        raise ValueError(f"Custom or unrecognized curve requires manual migration: {name}")
      curve = {"signal": name.lstrip("/"), "scale": 1.0, "offset": 0.0, "derivative": False}
      if not curve["signal"]:
        raise ValueError("Empty signal name")
      transforms = element.findall("transform")
      if len(transforms) > 1:
        raise ValueError(f"Multiple transforms require manual migration: {name}")
      for transform in transforms:
        if transform.get("name") != "Scale/Offset":
          raise ValueError(f"Unsupported transform on {name}: {transform.get('name')}")
        options = transform.find("options")
        if options is None:
          raise ValueError(f"Missing Scale/Offset options: {name}")
        scale = float(options.get("value_scale", "1"))
        offset = float(options.get("value_offset", "0"))
        time_offset = float(options.get("time_offset", "0"))
        if not all(math.isfinite(v) for v in (scale, offset, time_offset)) or time_offset != 0:
          raise ValueError(f"Non-finite or time-shifted transform requires manual migration: {name}")
        curve.update(scale=scale, offset=offset)
      curves.append(curve)
    if curves:
      plots.append(curves)
  if not plots:
    raise ValueError("No time-series curves found")
  return {"version": 1, "plots": plots}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("layout", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  try:
    result = convert_layout(args.layout.read_text())
  except (ET.ParseError, ValueError, OSError) as error:
    parser.exit(1, f"Import failed: {error}\n")
  # Exclusive creation keeps an existing layout intact, including on rejected imports.
  try:
    with args.output.open("x") as output:
      json.dump(result, output, indent=2, allow_nan=False)
      output.write("\n")
  except OSError as error:
    parser.exit(1, f"Unable to create output: {error}\n")
  print(f"Imported {len(result['plots'])} plot groups. Review them in Cabana; window styling and tabs were not transferred.")


if __name__ == "__main__":
  main()
