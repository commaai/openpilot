#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PJ_LAYOUTS = ROOT / "tools" / "plotjuggler" / "layouts"
JOT_LAYOUTS = ROOT / "tools" / "jotpluggler" / "layouts"


def indent(text: str, level: int = 1) -> str:
  prefix = "    " * level
  return "\n".join(prefix + line if line else line for line in text.splitlines())


def convert_expr(expr: str) -> str:
  out = expr
  out = re.sub(r"--(.*)$", r"#\1", out)
  out = out.replace("^", "**")
  out = out.replace("~=", "!=")
  out = out.replace("math.pi", "np.pi")
  out = out.replace("math.sin", "np.sin")
  out = out.replace("math.cos", "np.cos")
  out = out.replace("math.sqrt", "np.sqrt")
  out = out.replace("math.abs", "np.abs")
  out = out.replace("math.atan", "np.arctan2")
  return out


def translate_lua_globals(lua_globals: str) -> tuple[str, list[str]]:
  translated_lines: list[str] = []
  assigned_names: list[str] = []
  for raw_line in lua_globals.splitlines():
    line = convert_expr(raw_line.rstrip())
    stripped = line.strip()
    if not stripped:
      continue
    match = re.match(r"^([A-Za-z_]\w*)\s*=", stripped)
    if match:
      assigned_names.append(match.group(1))
    translated_lines.append(stripped)
  return "\n".join(translated_lines), assigned_names


def translate_lua_function(lua_function: str, global_names: list[str], additional_count: int) -> str:
  translated: list[str] = []
  if global_names:
    translated.append("global " + ", ".join(global_names))

  indent_level = 0
  for raw_line in lua_function.splitlines():
    line = convert_expr(raw_line.rstrip())
    stripped = line.strip()
    if not stripped:
      continue
    if stripped.startswith("#"):
      translated.append("    " * indent_level + stripped)
      continue
    if stripped == "end":
      indent_level = max(0, indent_level - 1)
      continue
    if stripped == "else":
      indent_level = max(0, indent_level - 1)
      translated.append("    " * indent_level + "else:")
      indent_level += 1
      continue
    if stripped.startswith("elseif ") and stripped.endswith(" then"):
      indent_level = max(0, indent_level - 1)
      condition = stripped[len("elseif "):-len(" then")].strip()
      translated.append("    " * indent_level + f"elif {condition}:")
      indent_level += 1
      continue
    if stripped.startswith("if ") and stripped.endswith(" then"):
      condition = stripped[len("if "):-len(" then")].strip()
      translated.append("    " * indent_level + f"if {condition}:")
      indent_level += 1
      continue
    translated.append("    " * indent_level + stripped)

  args = ["time", "value"] + [f"v{i}" for i in range(1, additional_count + 1)]
  lines = [
    f"def __jotpluggler_eval_sample({', '.join(args)}):",
    indent("\n".join(translated) if translated else "return value"),
    "",
    "__jotpluggler_result = np.empty_like(value, dtype=np.float64)",
    "for __jotpluggler_i in range(len(value)):",
  ]
  call_args = ["time[__jotpluggler_i]", "value[__jotpluggler_i]"]
  for i in range(1, additional_count + 1):
    call_args.append(f"v{i}[__jotpluggler_i]")
  lines.append(indent(f"__jotpluggler_result[__jotpluggler_i] = __jotpluggler_eval_sample({', '.join(call_args)})"))
  lines.append("return __jotpluggler_result")
  return "\n".join(lines)


def parse_snippets(root: ET.Element) -> dict[str, dict]:
  snippets: dict[str, dict] = {}
  custom_math = root.find("customMathEquations")
  if custom_math is None:
    return snippets

  for snippet in custom_math.findall("snippet"):
    name = snippet.get("name", "").strip()
    if not name:
      continue
    linked_source = (snippet.findtext("linked_source") or "").strip()
    additional_sources: list[str] = []
    additional = snippet.find("additional_sources")
    if additional is not None:
      for child in additional:
        if child.text and child.text.strip():
          additional_sources.append(child.text.strip())
    globals_code, global_names = translate_lua_globals((snippet.findtext("global") or "").strip())
    function_code = translate_lua_function((snippet.findtext("function") or "").strip(), global_names, len(additional_sources))
    snippets[name] = {
      "linked_source": linked_source,
      "additional_sources": additional_sources,
      "globals_code": globals_code,
      "function_code": function_code,
    }
  return snippets


def parse_transform(curve_elem: ET.Element) -> dict:
  transform = curve_elem.find("transform")
  if transform is None:
    return {}
  name = transform.get("name", "")
  options = transform.find("options")
  if name == "Derivative":
    result = {"transform": "derivative"}
    if options is not None:
      dt = options.get("dTime")
      if dt is None and options.get("radioChecked") == "radioCustom":
        dt = options.get("lineEdit")
      if dt is not None:
        result["derivative_dt"] = float(dt)
    return result
  if name == "Scale/Offset" and options is not None:
    return {
      "transform": "scale",
      "scale": float(options.get("value_scale", "1")),
      "offset": float(options.get("value_offset", "0")),
    }
  return {}


def convert_curve(curve_elem: ET.Element, snippets: dict[str, dict]) -> dict:
  name = curve_elem.get("name", "")
  curve = {
    "name": name,
    "color": curve_elem.get("color", "#a0aab4"),
  }
  curve.update(parse_transform(curve_elem))
  if name in snippets:
    curve["custom_python"] = snippets[name]
  return curve


def parse_range(plot_elem: ET.Element) -> dict:
  range_elem = plot_elem.find("range")
  if range_elem is None:
    return {}
  return {
    "left": float(range_elem.get("left", "0")),
    "right": float(range_elem.get("right", "1")),
    "top": float(range_elem.get("top", "1")),
    "bottom": float(range_elem.get("bottom", "0")),
  }


def parse_y_limits(plot_elem: ET.Element) -> dict | None:
  limit_elem = plot_elem.find("limitY")
  if limit_elem is None:
    return None
  limits = {}
  if "min" in limit_elem.attrib:
    limits["min"] = float(limit_elem.get("min", "0"))
  if "max" in limit_elem.attrib:
    limits["max"] = float(limit_elem.get("max", "0"))
  return limits or None


def convert_dock_area(area_elem: ET.Element, snippets: dict[str, dict]) -> dict:
  plot_elem = area_elem.find("plot")
  if plot_elem is None:
    raise ValueError("DockArea missing plot")
  pane = {
    "title": area_elem.get("name", "..."),
    "range": parse_range(plot_elem),
    "curves": [convert_curve(curve, snippets) for curve in plot_elem.findall("curve")],
  }
  y_limits = parse_y_limits(plot_elem)
  if y_limits is not None:
    pane["y_limits"] = y_limits
  return pane


def convert_node(elem: ET.Element, snippets: dict[str, dict]) -> dict:
  if elem.tag == "DockArea":
    return convert_dock_area(elem, snippets)
  if elem.tag != "DockSplitter":
    raise ValueError(f"Unsupported layout node {elem.tag}")
  orientation = elem.get("orientation", "-")
  split = "vertical" if orientation == "-" else "horizontal"
  children = [convert_node(child, snippets) for child in elem if child.tag in {"DockArea", "DockSplitter"}]
  sizes_raw = elem.get("sizes", "")
  sizes = [float(part) for part in sizes_raw.split(";") if part]
  return {
    "split": split,
    "sizes": sizes if len(sizes) == len(children) else [1.0 / len(children)] * len(children),
    "children": children,
  }


def convert_xml_layout(path: Path) -> dict:
  root = ET.parse(path).getroot()
  snippets = parse_snippets(root)
  tabs = []
  tabbed = root.find("tabbed_widget")
  if tabbed is None:
    raise ValueError("Missing tabbed_widget")
  for tab_elem in tabbed.findall("Tab"):
    container = tab_elem.find("Container")
    if container is None:
      continue
    content = next((child for child in container if child.tag in {"DockArea", "DockSplitter"}), None)
    if content is None:
      continue
    tabs.append({
      "name": tab_elem.get("tab_name", "tab1"),
      "root": convert_node(content, snippets),
    })
  current_index = 0
  current = tabbed.find("currentTabIndex")
  if current is None:
    current = root.find(".//currentTabIndex")
  if current is not None:
    current_index = int(current.get("index", "0"))
  return {
    "current_tab_index": current_index,
    "tabs": tabs,
  }


def convert_all(layout_dir: Path, out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  for xml_path in sorted(layout_dir.glob("*.xml")):
    converted = convert_xml_layout(xml_path)
    out_path = out_dir / f"{xml_path.stem}.json"
    out_path.write_text(json.dumps(converted, separators=(",", ":")) + "\n", encoding="utf-8")
    print(out_path.relative_to(ROOT))


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--src", type=Path, default=PJ_LAYOUTS)
  parser.add_argument("--out", type=Path, default=JOT_LAYOUTS)
  args = parser.parse_args()
  convert_all(args.src, args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
