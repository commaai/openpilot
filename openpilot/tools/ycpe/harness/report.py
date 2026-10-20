#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["pillow", "numpy"]
# ///
"""Builds report.html (for people) and report.md (for LLM critics) from out/latest/results.json; run alone to rebuild."""
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import triage

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
REFERENCE = HERE / "out" / "reference"
ERROR, WARN, INFO = "error", "warn", "info"
PERF_RULES = ("long-frame", "frame-work", "drops", "first-paint")  # budgets that only warn when the machine is too busy to measure
SEVERITY = {
  "blocked": ERROR, "page-error": ERROR, "console-error": ERROR, "long-frame": ERROR, "frame-work": ERROR, "drops": ERROR, "first-paint": ERROR,
  "step": ERROR, "assert": ERROR, "auth": ERROR, "native-select": ERROR, "native-dialog": ERROR,
  "text-ellipsis": INFO, "small-target": INFO, "network-flake": INFO, "load-frame": INFO,
}
ORDER = [ERROR, WARN, INFO]
DIFF_THRESHOLD = 0.1  # pixelmatch's default: fraction of the max YIQ distance
DIFF_REPORT = 0.001   # changed-pixel fraction worth a finding
MAX_WHERE = 8         # occurrences listed per finding in report.md


def bless(out):
  for shot in (out / "shots").glob("*/*.png"):
    if "." not in shot.stem:
      dst = GOLDEN / shot.parent.name / shot.name
      dst.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy(shot, dst)


def yiq(a):
  r, g, b = a[..., 0], a[..., 1], a[..., 2]
  return np.stack([r * 0.29889531 + g * 0.58662247 + b * 0.11448223,
                   r * 0.59597799 - g * 0.27417610 - b * 0.32180189,
                   r * 0.21147017 - g * 0.52261711 + b * 0.31114694], -1)


def diff(shot, golden, dst):
  """pixelmatch-style YIQ distance; a pixel only counts if nothing within 1px in the golden matches it (absorbs antialiasing)"""
  a, b = Image.open(shot).convert("RGB"), Image.open(golden).convert("RGB")
  if a.size != b.size:
    return 1.0
  ya, yb = yiq(np.asarray(a, np.float32)), yiq(np.pad(np.asarray(b, np.float32), ((1, 1), (1, 1), (0, 0)), mode="edge"))
  h, w = ya.shape[:2]
  best = np.full((h, w), np.inf, np.float32)
  for dy in range(3):
    for dx in range(3):
      d = ya - yb[dy:dy + h, dx:dx + w]
      best = np.minimum(best, 0.5053 * d[..., 0] ** 2 + 0.299 * d[..., 1] ** 2 + 0.1957 * d[..., 2] ** 2)
  changed = best > 35215 * DIFF_THRESHOLD ** 2
  gray = (np.asarray(a.convert("L"), np.float32) * 0.25 + 191).astype(np.uint8)
  img = np.stack([gray] * 3, -1)
  img[changed] = (230, 30, 30)
  Image.fromarray(img).save(dst)
  return float(changed.mean())


def annotate(src, dst, marks):
  img = Image.open(src).convert("RGB")
  d = ImageDraw.Draw(img)
  for n, (x, y, w, h) in marks:
    d.rectangle([x, y, x + max(w, 1), y + max(h, 1)], outline=(255, 40, 40), width=2)
    label = [x, max(0, y - 14), x + 7 * len(str(n)) + 6, max(14, y)]
    d.rectangle(label, fill=(255, 40, 40))
    d.text((label[0] + 3, label[1] + 1), str(n), fill=(255, 255, 255))
  img.save(dst)


def collect(out, scenarios, noisy):
  """every failure, event, lint issue and golden diff, grouped across scenarios and shots by what it is rather than where"""
  groups = {}

  def add(key, rule, detail, where, selector=None):
    f = groups.setdefault((rule, key), {"id": f"{rule}|{key}", "rule": rule, "key": key, "detail": detail, "selector": selector, "where": []})
    f["where"].append(where)

  for sc in scenarios:
    name, shots = sc["name"], [s["name"] for s in sc["shots"]]
    for f in sc["failures"]:
      add(f["key"], f["rule"], f["detail"], {"scenario": name, "note": f["detail"].split(" (")[0]})
    for e in sc["events"]:
      shot = shots[min(e["shot"], len(shots) - 1)] if shots else None
      key = e["text"].split()[0] if e["kind"] == "network-flake" else re.sub(r"[0-9a-f]{6,}|\d+", "#", e["text"])
      add(key, e["kind"], e["text"], {"scenario": name, "shot": shot})
    for shot in sc["shots"]:
      for i in shot["issues"]:
        add(re.sub(r":nth-of-type\(\d+\)", "", i["selector"]), i["rule"], i["detail"], {"scenario": name, "shot": shot["name"], "box": i["box"]}, i["selector"])
      golden = GOLDEN / name / f"{shot['name']}.png"
      if golden.exists():
        shot["diff"] = diff(out / shot["path"], golden, out / shot["path"].replace(".png", ".diff.png"))
        if shot["diff"] > DIFF_REPORT:
          add(f"{name}/{shot['name']}", "golden-diff", f"{shot['diff']:.2%} of pixels differ from the golden", {"scenario": name, "shot": shot["name"]})
  severity = lambda f: WARN if noisy and f["rule"] in PERF_RULES else SEVERITY.get(f["rule"], WARN)  # noqa: E731
  found = sorted(groups.values(), key=lambda f: (ORDER.index(severity(f)), -len(f["where"])))
  for n, f in enumerate(found, 1):
    f["n"], f["severity"] = n, severity(f)
  return found


def where_text(w, note=True):
  return (f"{w['scenario']}›{w['shot']}" if w.get("shot") else w["scenario"]) + (f" ({w['note']})" if note and w.get("note") else "")


def chart(perf, prev):
  """frame interval over time, this run against the previous one, with long animation frames as red bars along the floor"""
  W, H, P, YMAX = 640, 150, 30, 100
  t0 = perf["t"][0] if perf["t"] else 0
  span = max(1, perf["t"][-1] - t0 if perf["t"] else 1, prev["t"][-1] - prev["t"][0] if prev and prev["t"] else 1)
  x = lambda t: P + (W - P - 5) * min(max(t, 0), span) / span  # noqa: E731
  y = lambda ms: H - 15 - (H - 25) * min(ms, YMAX) / YMAX  # noqa: E731

  def steps(p, color, width):
    """each frame is a flat step over its own interval, so a long frame reads as a wide plateau"""
    if not p or not p["t"]:
      return ""
    d = "".join(f"M{x(t - v - p['t'][0]):.1f} {y(v):.1f}H{x(t - p['t'][0]):.1f}" for t, v in zip(p["t"], p["interval"], strict=True))
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"/>'

  grid = "".join(f'<line x1="{P}" x2="{W - 5}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#3a3f44" stroke-dasharray="3 3"/>' +
                 f'<text x="{P - 4}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>' for v in (16.7, 33, 50, 100))
  loaf = "".join(f'<rect x="{x(e["start"] - t0):.1f}" y="{H - 12}" width="{max(2, x(e["duration"]) - P):.1f}" height="8" fill="#e5484d">' +
                 f'<title>{e["duration"]:.0f} ms frame: {html.escape(", ".join(s["fn"] or s["invoker"] for s in e["scripts"]))}</title></rect>'
                 for e in perf["loaf"])
  paused = "".join(f'<rect x="{x(a - t0):.1f}" y="5" width="{x(b - a) - P:.1f}" height="{H - 20}" fill="#ffffff10"><title>harness, excluded</title></rect>'
                   for a, b in perf.get("paused", []))
  legend = (f'frame interval ms over {span / 1000:.1f} s · <tspan fill="#4f9cf9">this run</tspan> · previous run · ' +
            '<tspan fill="#e5484d">long frames</tspan> · shaded: harness, excluded')
  return (f'<svg width="{W}" height="{H}" class="chart">{paused}{grid}{steps(prev, "#8b9096", 1)}{steps(perf, "#4f9cf9", 1.5)}{loaf}' +
          f'<text x="{W - 5}" y="11" text-anchor="end">{legend}</text></svg>')


def num(v, prev=None, fmt="{:.0f}"):
  if v is None:
    return "-"
  return fmt.format(v) + (f" ({v - prev:+.0f})" if prev is not None and abs(v - prev) >= 1 else "")


def build(out, prev_dir=None):
  res = json.loads((out / "results.json").read_text())
  scenarios = res["scenarios"]
  prev = {}
  if prev_dir and (prev_dir / "results.json").exists():
    prev = {s["name"]: s for s in json.loads((prev_dir / "results.json").read_text())["scenarios"]}
  found = collect(out, scenarios, bool(res["noisy"]))
  triage.annotate(found)
  before = json.loads((prev_dir / "findings.json").read_text()) if prev_dir and (prev_dir / "findings.json").exists() else []
  for f in found:
    f["scenarios"] = sorted({w["scenario"] for w in f["where"]})
  changes = triage.diff(found, before, {sc["name"] for sc in scenarios})
  keep = ("id", "n", "rule", "severity", "detail", "selector", "file", "scenarios")
  (out / "findings.json").write_text(json.dumps([{**{k: f[k] for k in keep}, "count": len(f["where"])} for f in found], indent=1))
  for sc in scenarios:
    for film in sc.get("films", []):
      film["strip"] = filmstrip(out, sc["name"], film)

  marks = {}
  for f in found:
    for w in f["where"]:
      if f["severity"] != INFO and w.get("box") and w["box"][2] * w["box"][3] > 0:
        marks.setdefault((w["scenario"], w["shot"]), []).append((f["n"], w["box"]))
  for sc in scenarios:
    for shot in sc["shots"]:
      shot["annotated"] = shot["path"].replace(".png", ".annot.png")
      annotate(out / shot["path"], out / shot["annotated"], marks.get((sc["name"], shot["name"]), []))

  head = ["scenario", "result", "first paint ms (Δprev)", "frames", "worst frame ms (Δprev)", "p99 frame ms (Δprev)", "long frames (worst ms)",
          "perf.js work p99 ms", "drops", "findings", "time"]
  rows = []
  for sc in scenarios:
    s, p = sc["stats"], prev.get(sc["name"], {}).get("stats", {})
    mine = [f["n"] for f in found if any(w["scenario"] == sc["name"] for w in f["where"])]
    sc["findings"] = mine
    result = ("PASS" if sc["pass"] else "FAIL") + (" (retried)" if sc.get("retried") else "")
    rows.append([sc["name"], result, num(s.get("first_paint_ms"), p.get("first_paint_ms")), s["frames"],
                 num(s["worst_ms"], p.get("worst_ms")), num(s["p99_ms"], p.get("p99_ms")), f"{s['loaf']} ({s['loaf_worst_ms']:.0f})",
                 num(s["work_p99_ms"], fmt="{:.1f}"), num(s["drops"]), len(mine), f"{sc['duration']:.1f}s"])

  passed = sum(sc["pass"] for sc in scenarios)
  title = f"{passed}/{len(scenarios)} passed · {len(found)} findings · mode {res['mode']} · {res['duration']:.0f} s · peak load {res['load']:.0f}/{res['cpus']}"
  if res["noisy"]:
    title += f" · NOISY: {res['noisy']}, so perf budgets only warn"
  write_md(out, title, head, rows, found, scenarios, changes)
  write_html(out, title, head, rows, found, scenarios, prev, changes)
  return changes


def filmstrip(out, scenario, film):
  """one image: per frame, its time and the crops (or a thumbnail when there are none), top to bottom"""
  rows = []
  for f in film["frames"]:
    img = Image.open(out / f["path"]).convert("RGB")
    parts = [img.crop((b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"])) for b in f["boxes"]] or [img.resize((img.width // 3, img.height // 3))]
    row = Image.new("RGB", (60 + max(p.width for p in parts), sum(p.height for p in parts) + 2 * (len(parts) - 1)), (22, 25, 27))
    y = 0
    for p in parts:
      row.paste(p, (60, y))
      y += p.height + 2
    ImageDraw.Draw(row).text((4, 4), f"{f['t']:.2f}s", fill=(220, 220, 220))
    rows.append(row)
  strip = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 4 for r in rows)), (60, 60, 60))
  y = 0
  for r in rows:
    strip.paste(r, (0, y))
    y += r.height + 4
  path = f"shots/{scenario}/{film['name']}.strip.png"
  strip.save(out / path)
  return path


def by_file(items):
  """markdown lines for findings grouped by file"""
  lines = []
  for file in sorted({f["file"] or "?" for f in items}):
    lines.append(f"- {file}")
    lines += [f"  - #{f['n']} [{f['severity']}] {f['rule']}: {f['detail'][:160]}" for f in items if (f["file"] or "?") == file]
  return lines


def write_md(out, title, head, rows, found, scenarios, changes):
  md = [f"# YCPE harness: {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head),
        *("| " + " | ".join(map(str, r)) + " |" for r in rows), "",
        "## Findings", "",
        "Grouped by rule and selector across all scenarios. error fails the scenario, warn is a likely bug, info is probably intentional.",
        "Boxes are [x, y, w, h] in CSS px; the number is drawn on the annotated screenshots.",
        "Frame timings are noisy on a busy machine: when an empty page drops frames or the load average exceeds the cpu count,",
        "perf budgets are demoted to warnings.", ""]
  for f in found:
    sel = f" `{f['selector']}`" if f["selector"] else ""
    where = ", ".join(where_text(w, len(f["where"]) > 1) + (f" {w['box']}" if w.get("box") else "") for w in f["where"][:MAX_WHERE])
    more = f" and {len(f['where']) - MAX_WHERE} more" if len(f["where"]) > MAX_WHERE else ""
    md.append(f"{f['n']}. [{f['severity']}] {f['rule']} ×{len(f['where'])}: {f['detail']}{sel}\n   - in {where}{more}\n   - file: {f['file'] or '?'}")
  md += ["", f"## Since the previous run: {len(changes['new'])} new, {len(changes['fixed'])} fixed, {len(changes['still'])} still open", "",
         "### New", "", *by_file(changes["new"]), "", "### Fixed (numbers from the previous run)", "", *by_file(changes["fixed"]), "",
         "### Still open", "", *by_file(changes["still"])]
  md += ["", "## Screenshots", ""]
  for sc in scenarios:
    md.append(f"- {sc['name']} ({sc['url']}) findings {sc['findings']}")
    for s in sc["shots"]:
      d = f" · diff {s['diff']:.2%} {out / s['path'].replace('.png', '.diff.png')}" if "diff" in s else ""
      md.append(f"  - {s['name']}: {out / s['annotated']}{d}")
    md += [f"  - note: {n}" for n in sc.get("notes", [])]
    md += [f"  - filmstrip {film['name']} ({len(film['frames'])} frames): {out / film['strip']}" for film in sc.get("films", [])]
    md += [f"  - reference {r}: {REFERENCE / r}.png" for r in sc["reference"] if (REFERENCE / f"{r}.png").exists()]
  (out / "report.md").write_text("\n".join(md) + "\n")


def write_html(out, title, head, rows, found, scenarios, prev, changes):
  e = html.escape
  table = "".join(f'<tr class="{r[1].split()[0].lower()}">' + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in rows)
  new = {f["id"] for f in changes["new"]}
  items = "".join(
    f'<li value="{f["n"]}" class="{f["severity"]}">{"<i>new</i> " if f["id"] in new else ""}<b>{e(f["rule"])}</b> ×{len(f["where"])} {e(f["detail"])}'
    + (f' <code>{e(f["selector"])}</code>' if f["selector"] else "")
    + f'<div class="where">{e(", ".join(dict.fromkeys(where_text(w) for w in f["where"])))}'
    + f' · <span class="owner">{e(f["file"] or "?")}</span></div></li>' for f in found)
  fixed = "".join(f'<li>{e(f["rule"])}: {e(f["detail"][:160])} <span class="owner">{e(f["file"] or "?")}</span></li>'
                  for f in changes["fixed"])
  fig = lambda href, cap: f'<figure><a href="{href}"><img src="{href}" loading="lazy"></a><figcaption>{cap}</figcaption></figure>'  # noqa: E731
  sections = []
  for sc in scenarios:
    status = "pass" if sc["pass"] else "fail"
    diff = lambda s: f' · <a href="{s["path"].replace(".png", ".diff.png")}">diff {s["diff"]:.2%}</a>' if "diff" in s else ""  # noqa: E731
    shots = "".join(fig(s["annotated"], e(s["name"]) + diff(s)) for s in sc["shots"])
    refs = "".join(fig(f"../reference/{r}.png", f"reference: {e(r)}") for r in sc["reference"] if (REFERENCE / f"{r}.png").exists())
    films = "".join(fig(f["strip"], f"filmstrip {e(f['name'])}, {len(f['frames'])} frames") for f in sc.get("films", []))
    findings = ", ".join(map(str, sc["findings"])) or "none"
    heading = f'<h2>{e(sc["name"])} <span class="{status}">{status.upper()}</span> <small>{e(sc["url"])} · findings {findings}</small></h2>'
    svg = chart(sc["perf"], prev.get(sc["name"], {}).get("perf"))
    sections.append(f'<section id="{e(sc["name"])}">{heading}{svg}<div class="sheet">{shots}{refs}{films}</div></section>')
  (out / "report.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>YCPE harness</title>
<style>
body {{ font: 13px/1.45 system-ui, sans-serif; background: #16191b; color: #ddd; margin: 20px; }}
table {{ border-collapse: collapse; }} td, th {{ padding: 3px 10px; border-bottom: 1px solid #2c3135; text-align: left; white-space: nowrap; }}
tr.fail td:nth-child(2), span.fail {{ color: #ff6369; font-weight: 700; }} tr.pass td:nth-child(2), span.pass {{ color: #3dd68c; }}
h1 small, h2 small {{ color: #888; font-weight: 400; font-size: 13px; }} section {{ margin-top: 28px; border-top: 1px solid #2c3135; }}
ol li {{ margin-bottom: 4px; }} li.error b {{ color: #ff6369; }} li.warn b {{ color: #f5a623; }} li.info {{ color: #8b9096; }}
.where {{ color: #8ab4f8; font-size: 12px; }} code {{ color: #ccc; background: #262a2d; padding: 0 3px; }}
.sheet {{ display: flex; flex-wrap: wrap; gap: 12px; }} figure {{ margin: 0; }}
figure img {{ max-width: 460px; max-height: 340px; border: 1px solid #333; display: block; }} figcaption {{ color: #aaa; }}
a {{ color: #8ab4f8; }} i {{ color: #16191b; background: #f5a623; padding: 0 4px; border-radius: 3px; font-style: normal; }} .owner {{ color: #aaa; }}
.chart text {{ fill: #8b9096; font-size: 10px; }}
</style>
<h1>YCPE harness <small>{e(title)}</small></h1>
<table><tr>{"".join(f"<th>{e(h)}</th>" for h in head)}</tr>{table}</table>
<p><a href="report.md">report.md</a></p>
<h2>Findings <small>{len(changes["new"])} new, {len(changes["fixed"])} fixed, {len(changes["still"])} still open since the previous run</small></h2>
<ol>{items}</ol>
<details><summary>Fixed since the previous run ({len(changes["fixed"])})</summary><ul>{fixed}</ul></details>
{"".join(sections)}
""")


if __name__ == "__main__":
  build(HERE / "out" / "latest", HERE / "out" / "prev")
