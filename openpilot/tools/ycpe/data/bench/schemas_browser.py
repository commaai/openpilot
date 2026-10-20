#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["playwright"]
# ///
"""Log schemas in Chrome. The demo route reads with the bundled schema (auto), then toggles to its own commit's.
A fork log (fork_log.py) stands in for the demo's segment 0, and the route's git remote is rewritten to the fork: auto
mode switches to the fork's schema by itself, and the fork's fields show up in the series browser.
usage: schemas_browser.py <base url> <fork rlog.zst> <fork commit>"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROUTE = "5beb9b58bd12b691/0000010a--a51155e496"
RAW = "https://raw.githubusercontent.com/"
FLAGS = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]


def schema_row(page):
  return page.locator(".schema-row").inner_text().replace("\n", " ")


def wait(page, what, cond, timeout=60):
  t0 = time.monotonic()
  while time.monotonic() - t0 < timeout:
    if cond():
      return time.monotonic() - t0
    page.wait_for_timeout(100)
  raise TimeoutError(f"{what}: {schema_row(page)}")


def paths(page, text):
  """paths the series browser matches with a filter"""
  page.keyboard.press("2")
  f = page.locator(".browser .filter").first
  f.fill(text)
  page.wait_for_timeout(400)
  n = int(page.locator(".browser .count").inner_text())
  f.fill("")
  f.blur()  # view keys don't work in inputs
  page.keyboard.press("1")
  return n


def open_route(ctx, base):
  page = ctx.new_page()
  errors, api = [], []
  # a 404 is how GitHub says a path doesn't exist at a commit: schema lookups probe for them
  page.on("console", lambda m: m.type == "error" and not m.text.startswith("Failed to load resource") and errors.append(m.text))
  ctx.on("response", lambda r: r.status >= 400 and not r.url.startswith(RAW) and errors.append(f"{r.status} {r.url}"))
  page.on("pageerror", lambda e: errors.append(str(e)))
  ctx.on("request", lambda r: r.url.startswith("https://api.github.com/") and api.append(r.url))
  page.goto(f"{base}/{ROUTE}/drive")
  page.wait_for_selector(".info-pane")
  page.keyboard.press("2")  # the plot view loads the logs
  page.wait_for_selector(".browser .row", timeout=60000)
  page.keyboard.press("1")
  page.wait_for_selector(".schema-row select")
  return page, errors, api


def drops(page):
  return page.evaluate("window.__perf.drops")


if __name__ == "__main__":
  base, fork_log, fork_commit = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
  with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", args=FLAGS)
    ok = True

    # the demo: auto keeps the bundled schema, the toggle reads it with its own
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    page, errors, api = open_route(ctx, base)
    wait(page, "auto settles", lambda: "fetching" not in schema_row(page))
    print(f"demo, auto: {schema_row(page)}")
    ok &= "stock" in schema_row(page).split(" auto")[0]
    stock_only, log_only = paths(page, "narrowRoadCameraState/frameId"), paths(page, "roadCameraState/integLines")
    local = page.locator(".schema-row option[value^='local:']")
    if local.count():  # the daemon's checkout of this branch
      page.select_option(".schema-row select", local.first.get_attribute("value"))
      took = wait(page, "local schema", lambda: not schema_row(page).startswith(("stock", "commaai")) and "fetching" not in schema_row(page))
      print(f"demo, local: {schema_row(page)} (switched in {took:.1f} s)")
      ok &= "error" not in schema_row(page)
    d0 = drops(page)
    page.select_option(".schema-row select", "log")
    took = wait(page, "log schema", lambda: "commaai/openpilot@" in schema_row(page) and "fetching" not in schema_row(page))
    page.wait_for_timeout(3000)  # the re-read of every loaded segment
    print(f"demo, log: {schema_row(page)} (switched in {took:.1f} s, {drops(page) - d0} dropped frames over switch + re-read, {len(api)} API calls)")
    print(f"  paths: narrowRoadCameraState/frameId {stock_only} → {paths(page, 'narrowRoadCameraState/frameId')},",
          f"roadCameraState/integLines {log_only} → {paths(page, 'roadCameraState/integLines')}")
    ok &= paths(page, "roadCameraState/integLines") > 0 and paths(page, "narrowRoadCameraState/frameId") == 0
    page.select_option(".schema-row select", "auto")
    page.wait_for_timeout(500)
    print(f"demo, back to auto: {schema_row(page)}; errors: {errors or 'none'}")
    ok &= not errors
    ctx.close()

    # the fork: segment 0 replaced, the route's git remote rewritten
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    def as_fork(route):
      body = route.fetch().json()
      for r in body if isinstance(body, list) else [body]:
        r.update(git_remote="github.com/sunnypilot/sunnypilot", git_commit=fork_commit, git_branch="master")
      route.fulfill(json=body)
    ctx.route("**/v1/devices/*/routes_segments*", as_fork)
    ctx.route("**/v1/route/*/", as_fork)
    ctx.route("**/0/rlog.zst*", lambda route: route.fulfill(body=fork_log.read_bytes()))
    page, errors, api = open_route(ctx, base)
    took = wait(page, "fork schema", lambda: "sunnypilot/sunnypilot@" in schema_row(page) and "fetching" not in schema_row(page))
    page.wait_for_timeout(3000)
    sp = paths(page, "selfdriveStateSP/mads")
    print(f"fork, auto: {schema_row(page)} ({took:.1f} s after the route opened); selfdriveStateSP/mads paths: {sp},",
          f"carStateSP: {paths(page, 'carStateSP/speedLimit')}; dropped frames {drops(page)}; GitHub API calls {len(api)}; errors: {errors or 'none'}")
    ok &= sp >= 4 and not errors
    print(json.dumps({"ok": bool(ok)}))
    browser.close()
    sys.exit(0 if ok else 1)
