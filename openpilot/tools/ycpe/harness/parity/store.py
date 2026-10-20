#!/usr/bin/env python3
"""Store parity: finds a log the way the app does (data/sources.js), loads it through the app's Store in Chrome (loader,
workers, migration, pack, unpack) and compares every value and CAN frame with pycapnp after migrate_all. The local file
is always dropped (sources.localRoute); --url and --source add the same log from elsewhere. Needs ./serve.py <port>
running; on a daemon's port, sources resolve through the daemon.

usage: store.py <port> <log> [--url URL] [--source "<route>/<seg>[:auto|name,name]"] [--rtol x]
"""
import argparse
import json
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.tools.ycpe.harness.parity.diff import by_time, compare
from openpilot.tools.ycpe.harness.parity.dump import dump

SKIP = {"can", "sendcan"}  # CAN frames take their own path (data/can.js), compared by message


def relative(py):
  """route seconds, as the store has them: the time since initData, or since the first event without one"""
  t0 = py["initData"]["t"][0] if "initData" in py else min(svc["t"][0] for svc in py.values())
  return {name: svc | {"t": [t - t0 for t in svc["t"]]} for name, svc in py.items() if name not in SKIP} | {"@t0": t0}


def can_frames(events, t0):
  """{"bus:address": [[t, hex data], ...]} of the can service"""
  out = {}
  for e in events:
    if e.which() == "can":
      for f in e.can:
        out.setdefault(f"{f.src}:{f.address}", []).append([e.logMonoTime / 1e9 - t0, f.dat.hex()])
  return out


def compare_can(py, js):
  """{"bus:address": (problem, py, js)}; times match to the f32 the CAN worker keeps"""
  bad = {}
  for k in py.keys() | js.keys():
    a, b = py.get(k, []), js.get(k, [])
    if len(a) != len(b):
      bad[k] = ("frames", len(a), len(b))
    elif (i := next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x[1] != y[1] or abs(x[0] - y[0]) > 1e-4), None)) is not None:
      bad[k] = (f"frame {i}", a[i], b[i])
  return bad


def browser_dump(page, name, spec):
  """(store entry, dump); spec as harness/parity/store.js find() takes it, with drop: true for the page's file"""
  return page.evaluate("""async ([name, spec]) => {
    const {find, dumpStore} = await import('/harness/parity/store.js');
    const entry = await find(spec.drop ? {drop: document.getElementById('log').files[0]} : spec);
    return [{source: entry.source, url: String(entry.url).slice(0, 100)}, JSON.parse(await dumpStore(name, entry))];
  }""", [name, spec])


def source_spec(arg):
  """"<route>/<seg>[:sources]" → {route, n, sources}"""
  where, _, sources = arg.partition(":")
  route, _, n = where.rpartition("/")
  return {"route": route, "n": int(n), "sources": "auto" if sources in ("", "auto") else sources.split(",")}


def run(port, log, urls, sources, rtol):
  events = list(LogReader(log))
  py = relative(dump(migrate_all(events)))
  py_can = can_frames(events, py.pop("@t0"))
  results = {}
  with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=False, args=["--headless=new"])
    page = browser.new_page()
    page.goto(f"http://localhost:{port}/")  # index.html carries the daemon session token
    page.evaluate("document.body.append(Object.assign(document.createElement('input'), {type: 'file', id: 'log'}))")
    page.set_input_files("#log", log)
    specs = {"drop": {"drop": True}} | {url: {"url": url, "file": Path(log).name} for url in urls} | {s: source_spec(s) for s in sources}
    for label, spec in specs.items():
      # a fresh route name each run, so the OPFS cache never answers instead of the source
      entry, js = browser_dump(page, f"0000000000000000|{uuid.uuid4().hex[:8]}--{uuid.uuid4().hex[:10]}", spec)
      services = {name: by_time(svc) for name, svc in js["services"].items() if name not in SKIP}
      results[f"{label} ({entry['source']}: {entry['url']})"] = (js["failed"], compare(py, services, rtol), compare_can(py_can, js["can"]))
    browser.close()
  return py, sum(map(len, py_can.values())), results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("port")
  parser.add_argument("log")
  parser.add_argument("--url", action="append", default=[])
  parser.add_argument("--source", action="append", default=[])
  parser.add_argument("--rtol", type=float, default=1e-6)
  parser.add_argument("--max", type=int, default=40)
  args = parser.parse_args()

  py, frames, results = run(args.port, args.log, args.url, args.source, args.rtol)
  failed = False
  for label, (failed_segs, (bad, n), bad_can) in results.items():
    print(f"{args.log} via {label}:\n  {len(py)} services, {n} values, {len(bad)} mismatching paths, "
          f"{frames} CAN frames, {len(bad_can)} mismatching CAN messages, {failed_segs} failed loads")
    for path, (count, i, a, b) in list(bad.items())[:args.max]:
      print(f"  {path}: {count} (first at {i}: py={a!r} js={b!r})")
    for k, (problem, a, b) in list(bad_can.items())[:args.max]:
      print(f"  can {k}: {problem}: py={a!r} js={b!r}")
    failed |= bool(bad) or bool(bad_can) or bool(failed_segs)
  raise SystemExit(failed)
