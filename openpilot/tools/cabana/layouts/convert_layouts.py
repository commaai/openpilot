#!/usr/bin/env python3
"""
Convert PlotJuggler XML layouts to cabana JSON layouts.

Rules (Option C):
  - Curves starting with '/' are cereal paths → keep as signals
  - Curves without '/' are math/derived signals → record in _skipped
  - All charts kept (even all-math); _skipped tells team what needs Lua support
  - Lua snippets extracted from <snippets> section → "snippets" array in JSON
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_snippets(root) -> list[dict]:
  snippets = []
  for s in root.iter("snippet"):
    name = s.get("name", "")
    if not name:
      continue
    global_init = (s.findtext("global") or "").strip()
    func_body   = (s.findtext("function") or "").strip()
    linked      = (s.findtext("linked_source") or "").strip()
    add_src     = s.find("additional_sources")
    extra = []
    if add_src is not None:
      for tag in ["v1", "v2", "v3", "v4"]:
        val = add_src.findtext(tag)
        if val and val.strip():
          extra.append(val.strip())
    entry = {"name": name, "linked_source": linked, "function_body": func_body}
    if global_init:
      entry["global_init"] = global_init
    if extra:
      entry["additional_sources"] = extra
    snippets.append(entry)
  return snippets


def convert(xml_path: Path) -> tuple[dict, dict]:
  """Returns (layout_dict, stats_dict)."""
  tree = ET.parse(xml_path)
  root = tree.getroot()

  stats = {"charts_kept": 0, "charts_dropped": 0, "signals_kept": 0, "signals_skipped": 0}
  tabs = []

  for tab in root.iter("Tab"):
    charts = []
    for dock in tab.iter("DockArea"):
      curves = dock.findall(".//curve")
      simple = [{"path": c.get("name"), "color": c.get("color", "#ffffff")}
                for c in curves if c.get("name", "").startswith("/")]
      skipped = [c.get("name") for c in curves if not c.get("name", "").startswith("/")]

      chart = {"title": dock.get("name", ""), "signals": simple}
      if skipped:
        chart["_skipped"] = skipped

      charts.append(chart)
      stats["charts_kept"] += 1
      stats["signals_kept"] += len(simple)
      stats["signals_skipped"] += len(skipped)
      if not simple:
        stats["charts_math_only"] = stats.get("charts_math_only", 0) + 1

    if charts:
      tabs.append({"name": tab.get("tab_name", ""), "charts": charts})

  snippets = parse_snippets(root)
  layout = {"name": xml_path.stem, "tabs": tabs}
  if snippets:
    layout["snippets"] = snippets
  stats["snippets"] = len(snippets)
  return layout, stats


def main():
  here = Path(__file__).parent
  xml_dir = here.parent.parent / "plotjuggler" / "layouts"
  out_dir = here

  xml_files = sorted(xml_dir.glob("*.xml"))
  if not xml_files:
    print(f"No XML files found in {xml_dir}")
    sys.exit(1)

  total_kept = total_sig_kept = total_sig_skipped = total_snippets = 0

  for xml_path in xml_files:
    layout, stats = convert(xml_path)
    out_path = out_dir / (xml_path.stem + ".json")
    out_path.write_text(json.dumps(layout, indent=2))

    total_kept += stats["charts_kept"]
    total_sig_kept += stats["signals_kept"]
    total_sig_skipped += stats["signals_skipped"]
    total_snippets += stats["snippets"]

    math_only = stats.get("charts_math_only", 0)
    notes = []
    if stats["signals_skipped"]:
      notes.append(f"{stats['signals_skipped']} math signals")
    if math_only:
      notes.append(f"{math_only} math-only charts")
    if stats["snippets"]:
      notes.append(f"{stats['snippets']} Lua snippets")
    suffix = f"  ({', '.join(notes)})" if notes else ""
    print(f"  {xml_path.stem}: {stats['charts_kept']} charts, {stats['signals_kept']} signals{suffix}")

  print(f"\nDone: {total_kept} charts, {total_sig_kept} signals, {total_snippets} Lua snippets")
  if total_sig_skipped:
    print(f"      {total_sig_skipped} math signals recovered via Lua snippets when loaded in cabana")


if __name__ == "__main__":
  main()
