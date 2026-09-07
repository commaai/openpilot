#!/usr/bin/env python3
"""Convert PlotJuggler time-series layouts to Cabana's chart workspace format."""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def number(value, default=0):
  result = float(value) if value is not None else default
  if not math.isfinite(result):
    raise ValueError("Layout contains a non-finite number")
  return result


def import_layout(path):
  root = ET.parse(path).getroot()
  if root.tag != "root":
    raise ValueError("Not a PlotJuggler layout")
  tabs, names = [], []
  for tab in root.findall("./tabbed_widget/Tab"):
    names.append(tab.get("tab_name", "Charts"))
    charts = []
    for area in tab.findall(".//DockArea"):
      plot = area.find("plot")
      if plot is None:
        raise ValueError("This layout contains a non-plot panel")
      if plot.get("mode", "TimeSeries") != "TimeSeries":
        raise ValueError("Only time-series plots are supported")
      signals = []
      for curve in plot.findall("curve"):
        signal = dict(path=curve.attrib["name"], signal=curve.attrib["name"], visible=True,
                      transform=0, scale=1, offset=0, window=10, color=curve.get("color", "#0072b2"))
        transform = curve.find("transform")
        if transform is not None:
          kind = transform.get("name")
          options = transform.find("options")
          if kind == "Scale/Offset" and options is not None:
            signal["scale"] = number(options.get("value_scale"), 1)
            signal["offset"] = number(options.get("value_offset"))
            if number(options.get("time_offset")) != 0:
              raise ValueError("Time-offset transforms are not supported")
          elif kind == "Derivative":
            signal["transform"] = 1
          else:
            raise ValueError(f"Unsupported curve transform: {kind}")
        signals.append(signal)
      style = plot.get("style", "Lines")
      if style not in ("Lines", "Steps", "Points", "Dots"):
        raise ValueError(f"Unsupported plot style: {style}")
      chart = dict(title=area.get("name", ""), type={"Lines": 0, "Steps": 1, "Points": 2, "Dots": 2}[style], signals=signals)
      limits = plot.find("limitY")
      if limits is not None:
        for key in ("min", "max"):
          if limits.get(key) is not None:
            chart[f"y_{key}"] = number(limits.get(key))
      charts.append(chart)
    tabs.append(charts)
  if not tabs:
    raise ValueError("Layout has no chart tabs")
  equations = []
  for snippet in root.findall("./customMathEquations/snippet"):
    additional = snippet.find("additional_sources")
    equations.append(dict(name=snippet.attrib["name"], source=snippet.findtext("linked_source", "").strip(),
                          globals=snippet.findtext("global", ""), function=snippet.findtext("function", ""),
                          additional=[] if additional is None else [(v.text or "").strip() for v in additional]))
  return dict(cabana_layout=2, columns=1, range=60, tabs=tabs, tab_names=names, equations=equations)


if __name__ == "__main__":
  try:
    print(json.dumps(import_layout(Path(sys.argv[1])), allow_nan=False))
  except (OSError, ValueError, KeyError, ET.ParseError) as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
