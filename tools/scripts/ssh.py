#!/usr/bin/env python3
import os
import sys
import argparse
import queue
import re
import socket
import subprocess
import threading
import time

from openpilot.common.basedir import BASEDIR
from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.lib.api import CommaApi

LOCAL_TIMEOUT = 1.5
ATHENA_TIMEOUT = 8.0


def read_banner(f) -> None:
  banner = f.readline()
  if not banner.startswith(b"SSH-"):
    raise ConnectionError("no ssh banner")


def probe_local(host: str) -> None:
  with socket.create_connection((host, 22), timeout=LOCAL_TIMEOUT) as s, s.makefile("rb") as f:
    read_banner(f)


def probe_athena(proxy_cmd: list[str], procs: list[subprocess.Popen]) -> None:
  p = subprocess.Popen(proxy_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
  procs.append(p)
  try:
    read_banner(p.stdout)
  finally:
    p.kill()


def run_probe(name: str, fn, results: queue.Queue) -> None:
  t = time.monotonic()
  try:
    fn()
    results.put((name, (time.monotonic() - t) * 1000, None))
  except socket.gaierror:
    results.put((name, None, "not found"))
  except Exception as e:
    results.put((name, None, type(e).__name__ if not str(e) else str(e)))


def probe_routes(routes: dict, athena_proxy: list[str]) -> tuple[str | None, dict]:
  # probe all routes in parallel, prefer local since it's direct
  results: queue.Queue = queue.Queue()
  procs: list[subprocess.Popen] = []
  fns = {"local": lambda: probe_local(routes["local"]), "athena": lambda: probe_athena(athena_proxy, procs)}
  for name in routes:
    threading.Thread(target=run_probe, args=(name, fns[name], results), daemon=True).start()

  status: dict = {}
  deadline = time.monotonic() + ATHENA_TIMEOUT
  choice = None
  while len(status) < len(routes) and choice is None:
    try:
      name, ms, err = results.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty:
      break
    status[name] = (ms, err)
    if err is None and (name == "local" or status.get("local", (None, "x"))[1] is not None or "local" not in routes):
      choice = name

  for p in procs:
    p.kill()
  return choice, status


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper for connecting to devices, either directly over the local network or through the \
      comma prime SSH proxy. Adding your SSH key to your SSH config is recommended for more convenient use; see https://docs.comma.ai/how-to/connect-to-comma/.")
  parser.add_argument("device", help="device name or dongle id")
  parser.add_argument("--route", help="connection route", choices=["auto", "local", "athena"], default="auto")
  parser.add_argument("--host", help="ssh jump server host", default="ssh.comma.ai")
  parser.add_argument("--port", help="ssh jump server port", default=22, type=int)
  parser.add_argument("--key", help="ssh key", default=os.path.join(BASEDIR, "openpilot/common/hardware/comma/id_rsa"))
  parser.add_argument("--debug", help="enable debug output", action="store_true")
  args = parser.parse_args()

  r = CommaApi(get_token()).get("v1/me/devices")
  devices = {x['dongle_id']: x for x in r}

  if not re.match("[0-9a-zA-Z]{16}", args.device):
    user_input = args.device.replace(" ", "").lower()
    matches = { k: v['alias'] for k, v in devices.items() if isinstance(v['alias'], str) and user_input in v['alias'].replace(" ", "").lower() }
    if len(matches) == 1:
      dongle_id = list(matches.keys())[0]
    else:
      print(f"failed to look up dongle id for \"{args.device}\"", file=sys.stderr)
      if len(matches) > 1:
        print("found multiple matches:", file=sys.stderr)
        for k, v in matches.items():
          print(f"  \"{v}\" ({k})", file=sys.stderr)
      exit(1)
  else:
    dongle_id = args.device

  device = devices.get(dongle_id, {})
  name = f"{device['alias']} ({dongle_id})" if device else dongle_id

  athena_host = f"comma-{dongle_id}"
  athena_proxy = ["ssh", "-i", args.key, "-o", "BatchMode=yes", "-W", f"{athena_host}:{args.port}", "-p", str(args.port), f"{athena_host}@{args.host}"]
  routes = {}
  if device.get('serial') and args.route in ("auto", "local"):
    routes["local"] = f"comma-{device['serial']}"
  if args.route in ("auto", "athena"):
    routes["athena"] = args.host

  if not routes:
    print(f"no serial known for {name}, can't use local route", file=sys.stderr)
    exit(1)

  print(f"connecting to {name}")
  if len(routes) > 1:
    route, status = probe_routes(routes, athena_proxy)
    for r_name, r_host in routes.items():
      ms, err = status.get(r_name, (None, "not needed" if route else "timeout"))
      result = f"{ms:.0f} ms" if err is None else err
      print(f"  {'*' if r_name == route else ' '} {r_name:<7}{r_host:<24}{result}")
    if route is None:
      print("no route to device", file=sys.stderr)
      exit(1)
  else:
    route = next(iter(routes))
    print(f"  * {route:<7}{routes[route]:<24}{'forced' if args.route != 'auto' else 'only route (serial unknown)'}")

  command = ["ssh", "-i", args.key]
  if route == "athena":
    command += ["-o", f"ProxyCommand=ssh -i {args.key} -W %h:%p -p %p %h@{args.host}", "-p", str(args.port)]
  if args.debug:
    command += ["-v"]
  command += [f"comma@{athena_host if route == 'athena' else routes['local']}"]
  if args.debug:
    print(" ".join([f"'{c}'" if " " in c else c for c in command]))
  sys.stdout.flush()
  os.execvp(command[0], command)
