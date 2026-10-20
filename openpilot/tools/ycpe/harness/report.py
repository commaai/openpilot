#!/usr/bin/env python3
"""Builds report.html (for people) and report.md (for LLM critics) from out/latest/results.json; run alone to rebuild."""
import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
REFERENCE = HERE / "out" / "reference"
SEVERITY = {
  "blocked": 0, "page-error": 0, "console-error": 0, "budget": 0, "step": 0,
  "console-warning": 1, "http-error": 1, "request-failed": 1, "broken-image": 1, "overflow-viewport": 1, "text-clipped": 1,
  "obscured": 1, "overlap": 1, "small-target": 1, "layout-shift": 1, "golden-diff": 1,
  "text-ellipsis": 2,
}
SEVERITY_NAME = ["error", "warn", "info"]
DIFF_THRESHOLD = 0.1  # pixelmatch's default: fraction of the max YIQ distance
DIFF_REPORT = 0.001   # changed-pixel fraction worth a finding


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
  """pixelmatch-style YIQ distance; a pixel only counts if no pixel within 1px in the golden matches it (absorbs antialiasing)"""
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
  out = (np.asarray(a.convert("L"), np.float32) * 0.25 + 191).astype(np.uint8)
  out = np.stack([out] * 3, -1)
  out[changed] = (230, 30, 30)
  Image.fromarray(out).save(dst)
  return float(changed.mean())


def annotate(src, dst, marks):
  img = Image.open(src).convert("RGB")
  d = ImageDraw.Draw(img)
  for n, (x, y, w, h) in marks:
    d.rectangle([x, y, x + max(w, 1), y + max(h, 1)], outline=(255, 40, 40), width=2)
    d.rectangle([x, max(0, y - 14), x + 8 * len(str(n)) + 6, max(14, y)], fill=(255, 40, 40))
    d.text((x + 3, max(0, y - 13)), str(n), fill=(255, 255, 255))
  img.save(dst)


def findings(out, sc):
  """events, budget failures and lint issues for one scenario; a rule+selector repeated across shots is one finding"""
  items = [{"rule": f.split(":")[0] if f.startswith(("budget", "auth")) else "step", "detail": f, "shot": None}
           for f in sc["failures"] if f.startswith(("budget", "step", "auth"))]
  shots = [s["name"] for s in sc["shots"]]
  for e in sc["events"]:
    items.append({"rule": e["kind"], "detail": e["text"], "shot": shots[min(e["shot"], len(shots) - 1)] if shots else None})
  seen = {}
  for shot in sc["shots"]:
    for i in shot["issues"]:
      key = (i["rule"], i["selector"])
      if key in seen:
        seen[key]["also"].append(shot["name"])
        continue
      seen[key] = {**i, "shot": shot["name"], "also": []}
      items.append(seen[key])
    golden = GOLDEN / sc["name"] / f"{shot['name']}.png"
    if golden.exists():
      shot["diff"] = diff(out / shot["path"], golden, out / shot["path"].replace(".png", ".diff.png"))
      if shot["diff"] > DIFF_REPORT:
        items.append({"rule": "golden-diff", "detail": f"{shot['diff']:.1%} of pixels differ from the golden", "shot": shot["name"]})
  items.sort(key=lambda i: SEVERITY.get(i["rule"], 1))
  return items


def chart(sc, prev):
  """frame interval over time, this run against the previous one, with long animation frames as ticks along the floor"""
  W, H, P, YMAX = 640, 150, 30, 100
  cur = sc["perf"]
  t0 = cur["t"][0] if cur["t"] else 0
  span = max([cur["t"][-1] - t0 if cur["t"] else 1] + ([prev["t"][-1] - prev["t"][0]] if prev and prev["t"] else []))
  x = lambda t: P + (W - P - 5) * t / span  # noqa: E731
  y = lambda ms: H - 15 - (H - 25) * min(ms, YMAX) / YMAX  # noqa: E731

  def line(perf, color, width):
    if not perf or not perf["t"]:
      return ""
    pts = " ".join(f"{x(t - perf['t'][0]):.1f},{y(v):.1f}" for t, v in zip(perf["t"], perf["interval"], strict=False))
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>'

  grid = "".join(f'<line x1="{P}" x2="{W - 5}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#444" stroke-dasharray="3 3"/>'
                 f'<text x="{P - 4}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>' for v in (16.7, 33, 50, 100))
  loaf = "".join(f'<rect x="{x(e["start"] - t0):.1f}" y="{H - 12}" width="{max(2, x(e["duration"]) - P):.1f}" height="8" fill="#e5484d">'
                 f'<title>LoAF {e["duration"]:.0f} ms: {html.escape(", ".join(s["fn"] or s["invoker"] for s in e["scripts"]))}</title></rect>'
                 for e in cur["loaf"])
  return (f'<svg width="{W}" height="{H}" class="chart">{grid}{line(prev, "#777", 1)}{line(cur, "#4f9cf9", 1.5)}{loaf}'
          f'<text x="{W - 5}" y="12" text-anchor="end">frame interval, ms (blue: this run, gray: previous; red: LoAF) over {span / 1000:.1f}s</text></svg>')


def fmt_ms(v, prev=None):
  if v is None:
    return "-"
  return f"{v:.0f}" + (f" ({prev:+.0f})" if prev is not None else "")


def build(out, prev_dir=None):
  res = json.loads((out / "results.json").read_text())
  prev = {}
  if prev_dir and (prev_dir / "results.json").exists():
    prev = {s["name"]: s for s in json.loads((prev_dir / "results.json").read_text())["scenarios"]}

  n = 0
  rows, md_rows, md_findings, sections = [], [], [], []
  for sc in res["scenarios"]:
    items = findings(out, sc)
    marks = {}
    for i in items:
      n += 1
      i["n"] = n
      if i.get("box") and i["shot"]:
        marks.setdefault(i["shot"], []).append((n, i["box"]))
        for also in i["also"]:
          marks.setdefault(also, []).append((n, i["box"]))
    for shot in sc["shots"]:
      shot["annotated"] = shot["path"].replace(".png", ".annot.png")
      annotate(out / shot["path"], out / shot["annotated"], marks.get(shot["name"], []))

    s, p = sc["stats"], prev.get(sc["name"], {}).get("stats", {})
    delta = lambda k: s[k] - p[k] if p.get(k) is not None and s[k] is not None else None  # noqa: E731
    lint = sum(1 for i in items if "box" in i)
    status = "PASS" if sc["pass"] else "FAIL"
    cells = [sc["name"], status, s["frames"], fmt_ms(s["worst_ms"], delta("worst_ms")), fmt_ms(s["p99_ms"], delta("p99_ms")),
             f"{s['loaf']} (worst {s['loaf_worst_ms']:.0f})", fmt_ms(s["work_p99_ms"]), lint, f"{sc['duration']:.1f}s"]
    rows.append(f'<tr class="{status.lower()}">' + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
    md_rows.append("| " + " | ".join(str(c) for c in cells) + " |")

    lis = []
    for i in items:
      where = f"{sc['name']} › {i['shot']}" if i["shot"] else sc["name"]
      also = f" (also in {', '.join(i['also'])})" if i.get("also") else ""
      sel = f" `{i['selector']}` box {i['box']}" if i.get("box") else ""
      sev = SEVERITY_NAME[SEVERITY.get(i["rule"], 1)]
      md_findings.append(f"{i['n']}. [{sev}] **{where}** {i['rule']}: {i['detail']}{sel}{also}")
      lis.append(f'<li value="{i["n"]}" class="{sev}"><b>{html.escape(i["rule"])}</b> <span class="where">{html.escape(i["shot"] or "")}</span> '
                 f'{html.escape(i["detail"])}{" <code>" + html.escape(i["selector"]) + "</code>" if i.get("selector") else ""}{html.escape(also)}</li>')

    shots_html = "".join(
      f'<figure><a href="{s_["annotated"]}"><img src="{s_["annotated"]}" loading="lazy"></a><figcaption>{html.escape(s_["name"])}'
      + (f' · <a href="{s_["path"].replace(".png", ".diff.png")}">diff {s_["diff"]:.2%}</a>' if "diff" in s_ else "") + "</figcaption></figure>"
      for s_ in sc["shots"])
    refs = [r for r in sc["reference"] if (REFERENCE / f"{r}.png").exists()]
    refs_html = "".join(f'<figure><a href="../reference/{r}.png"><img src="../reference/{r}.png" loading="lazy"></a>'
                        f'<figcaption>reference: {html.escape(r)}</figcaption></figure>' for r in refs)
    sections.append(f'<section id="{sc["name"]}"><h2>{html.escape(sc["name"])} <span class="{status.lower()}">{status}</span> '
                    f'<small>{html.escape(sc["url"])}</small></h2>{chart(sc["perf"] and sc, prev.get(sc["name"], {}).get("perf"))}'
                    f'<ol class="findings">{"".join(lis)}</ol><div class="sheet">{shots_html}{refs_html}</div></section>')

    md_findings_imgs = [f"   - {s_['name']}: {out / s_['annotated']}" + (f" (diff {s_['diff']:.2%}: {out / s_['path'].replace('.png', '.diff.png')})" if "diff" in s_ else "")
                        for s_ in sc["shots"]] + [f"   - reference {r}: {REFERENCE / (r + '.png')}" for r in refs]
    sc["md_images"] = md_findings_imgs

  head = ["scenario", "result", "frames", "worst frame ms (Δprev)", "p99 frame ms (Δprev)", "LoAF", "perf.js work p99", "lint", "time"]
  passed = sum(sc["pass"] for sc in res["scenarios"])
  title = f"{passed}/{len(res['scenarios'])} passed · mode {res['mode']} · {res['duration']:.1f}s"
  md = [f"# YCPE harness: {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head), *md_rows, "",
        "## Findings", "", "Severity: error fails the scenario; warn is a likely bug; info is probably intentional.", "", *md_findings, "",
        "## Images", ""]
  for sc in res["scenarios"]:
    md += [f"- {sc['name']} ({sc['url']})", *sc.pop("md_images")]
  (out / "report.md").write_text("\n".join(md) + "\n")

  (out / "report.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>YCPE harness</title>
<style>
body {{ font: 13px/1.4 system-ui, sans-serif; background: #16191b; color: #ddd; margin: 20px; }}
table {{ border-collapse: collapse; }} td, th {{ padding: 4px 10px; border-bottom: 1px solid #333; text-align: left; }}
tr.fail td:nth-child(2), span.fail {{ color: #ff6369; font-weight: 700; }} tr.pass td:nth-child(2), span.pass {{ color: #3dd68c; }}
h2 small {{ color: #888; font-weight: 400; font-size: 13px; }} section {{ margin-top: 28px; border-top: 1px solid #333; }}
.findings li.error {{ color: #ff6369; }} .findings li.info {{ color: #999; }} .where {{ color: #8ab4f8; }}
code {{ color: #ccc; background: #262a2d; padding: 0 3px; }}
.sheet {{ display: flex; flex-wrap: wrap; gap: 12px; }} figure {{ margin: 0; }} figure img {{ max-width: 460px; max-height: 340px; border: 1px solid #333; display: block; }}
figcaption {{ color: #aaa; }} a {{ color: #8ab4f8; }} .chart text {{ fill: #999; font-size: 10px; }}
</style>
<h1>YCPE harness <small>{html.escape(title)}</small></h1>
<table><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>{"".join(rows)}</table>
<p><a href="report.md">report.md</a></p>
{"".join(sections)}
""")


if __name__ == "__main__":
  build(HERE / "out" / "latest", HERE / "out" / "prev")
