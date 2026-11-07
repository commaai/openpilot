"""Convert legacy PlotJuggler XML without translating or discarding Lua programs."""

import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def number(value):
  result = float(value)
  if not math.isfinite(result):
    raise ValueError("Layout numbers must be finite")
  return result


def convert(path):
  data = Path(path).read_bytes()
  if len(data) > 16 * 1024 * 1024:
    raise ValueError("Layout XML exceeds 16 MiB")
  root = ET.fromstring(data)
  snippets = {}
  for snippet in root.findall("./customMathEquations/snippet"):
    additional = snippet.find("additional_sources")
    snippets[snippet.attrib["name"]] = {
      "linked_source": snippet.findtext("linked_source", ""),
      "additional_sources": [child.text or "" for child in additional] if additional is not None else [],
      "globals_code": snippet.findtext("global", ""),
      "function_code": snippet.findtext("function", ""),
      "language": "lua",
    }

  def pane(element, depth=0):
    if depth > 64:
      raise ValueError("Layout nesting exceeds 64 levels")
    if element.tag == "DockSplitter":
      orientation = element.get("orientation", "-")
      if orientation not in ("-", "|"):
        raise ValueError("Unknown splitter orientation: " + orientation)
      children = [pane(child, depth + 1) for child in element if child.tag in ("DockArea", "DockSplitter")]
      if not children:
        raise ValueError("Empty XML splitter")
      sizes = [number(value) for value in element.get("sizes", "").split(";") if value]
      if not sizes:
        sizes = [1] * len(children)
      if len(sizes) != len(children) or any(value <= 0 for value in sizes):
        raise ValueError("Invalid XML splitter sizes")
      return {"split": "vertical" if orientation == "-" else "horizontal", "sizes": sizes, "children": children}
    plot = element.find("plot")
    if plot is None:
      raise ValueError("Unsupported XML dock pane: " + element.get("name", ""))
    if plot.get("mode", "TimeSeries") != "TimeSeries":
      raise ValueError("Unsupported legacy plot mode: " + plot.get("mode", ""))
    styles = {"Lines": 0, "Steps": 1, "Points": 2}
    style = plot.get("style", "Lines")
    if style not in styles:
      raise ValueError("Unsupported plot style: " + style)
    result = {
      "title": element.get("name", ""),
      "series_type": styles[style],
      "curves": [],
      "flip_x": plot.get("flip_x", "false").lower() == "true",
      "flip_y": plot.get("flip_y", "false").lower() == "true",
    }
    bounds = plot.find("range")
    if bounds is not None:
      result["range"] = {key: number(bounds.attrib[key]) for key in ("left", "right", "bottom", "top")}
    limits = plot.find("limitY")
    if limits is not None and limits.attrib:
      result["y_limits"] = {key: number(value) for key, value in limits.attrib.items() if key in ("min", "max")}
    for curve in plot.findall("curve"):
      name = curve.get("name", "")
      entry = {"name": name, "color": curve.get("color", "#0072b2"), "visible": curve.get("visible", "true").lower() != "false"}
      if name in snippets:
        entry["custom_python"] = snippets[name]
      transform = curve.find("transform")
      if transform is not None:
        kind = transform.get("name", "")
        options = transform.find("options")
        if kind.lower() == "derivative":
          entry["transform"] = "derivative"
          entry["derivative_dt"] = number(options.get("lineEdit", "0")) if options is not None and options.get("radioChecked") == "radioCustom" else 0
        elif kind.lower() == "scale/offset":
          entry["transform"] = "scale"
          entry["scale"] = number(options.get("value_scale", "1")) if options is not None else 1
          entry["offset"] = number(options.get("value_offset", "0")) if options is not None else 0
          entry["time_offset"] = number(options.get("time_offset", "0")) if options is not None else 0
        else:
          raise ValueError("Unsupported legacy transform: " + kind)
      result["curves"].append(entry)
    return result

  widgets = root.findall("tabbed_widget")
  if len(widgets) != 1:
    raise ValueError("Legacy layout must contain exactly one tabbed workspace")
  widget = widgets[0]
  tabs = []
  for tab in widget.findall("Tab"):
    containers = tab.findall("Container")
    if len(containers) != 1:
      raise ValueError("Legacy tabs with floating containers require explicit conversion")
    children = [child for child in containers[0] if child.tag in ("DockArea", "DockSplitter")]
    if len(children) != 1:
      raise ValueError("Legacy container has no unique layout root")
    tabs.append({"name": tab.get("tab_name", ""), "root": pane(children[0])})
  index = widget.find("currentTabIndex")
  return {"cabana_layout": 1, "calculations": snippets, "tabs": tabs, "current_tab_index": int(index.get("index", "0")) if index is not None else 0}


if __name__ == "__main__":
  try:
    print(json.dumps(convert(sys.argv[1]), allow_nan=False))
  except Exception as error:
    print(str(error), file=sys.stderr)
    raise SystemExit(1) from error
