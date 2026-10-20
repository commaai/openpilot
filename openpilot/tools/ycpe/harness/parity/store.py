#!/usr/bin/env python3
"""Store parity: loads a log through the app's Store in Chrome (loader, workers, migration, pack, unpack) and compares
every value with pycapnp after migrate_all. Needs ./serve.py <port> running. A log is a local file, sent to the page
as a File, or a URL the page fetches itself; --url adds one and is compared against the same local file.

usage: store.py <port> <log> [--url URL] [--rtol x]
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

SKIP = {"can", "sendcan"}  # CAN frames take their own path (data/can.js)


def relative(py):
  """route seconds, as the store has them: the time since initData"""
  t0 = py["initData"]["t"][0]
  return {name: svc | {"t": [t - t0 for t in svc["t"]]} for name, svc in py.items() if name not in SKIP}


def browser_dump(page, name, entry):
  return json.loads(page.evaluate("""async ([name, entry]) => {
    const {dumpStore} = await import('/harness/parity/store.js');
    if (entry.input) entry = {file: entry.file, url: URL.createObjectURL(document.getElementById(entry.input).files[0])};
    return dumpStore(name, [entry]);
  }""", [name, entry]))


def run(port, log, urls, rtol):
  py = relative(dump(migrate_all(list(LogReader(log)))))
  results = {}
  with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=False, args=["--headless=new"])
    page = browser.new_page()
    page.goto(f"http://localhost:{port}/harness/parity/store.js")
    page.evaluate("document.body.append(Object.assign(document.createElement('input'), {type: 'file', id: 'log'}))")
    page.set_input_files("#log", log)
    file = Path(log).name
    # a fresh route name each run, so the OPFS cache never answers instead of the source
    sources = {"file": {"input": "log", "file": file}} | {url: {"url": url, "file": file} for url in urls}
    for label, entry in sources.items():
      js = browser_dump(page, f"0000000000000000|{uuid.uuid4().hex[:8]}--{uuid.uuid4().hex[:10]}", entry)
      services = {name: by_time(svc) for name, svc in js["services"].items() if name not in SKIP}
      results[label] = (js["failed"], compare(py, services, rtol))
    browser.close()
  return py, results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("port")
  parser.add_argument("log")
  parser.add_argument("--url", action="append", default=[])
  parser.add_argument("--rtol", type=float, default=1e-6)
  parser.add_argument("--max", type=int, default=40)
  args = parser.parse_args()

  py, results = run(args.port, args.log, args.url, args.rtol)
  failed = False
  for label, (failed_segs, (bad, n)) in results.items():
    print(f"{args.log} via {label[:80]}: {len(py)} services, {n} values, {len(bad)} mismatching paths, {failed_segs} failed loads")
    for path, (count, i, a, b) in list(bad.items())[:args.max]:
      print(f"  {path}: {count} (first at {i}: py={a!r} js={b!r})")
    failed |= bool(bad) or bool(failed_segs)
  raise SystemExit(failed)
