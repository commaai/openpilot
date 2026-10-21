"""Logs straight off a comma device on the LAN, over ssh, into the shared cache (ycpe/device/...), where the "device"
source serves them. Strictly read-only on the device: the only remote commands are cat and find."""
import os
import posixpath
import re
import shlex
import subprocess
import threading
import time

from openpilot.tools.lib.helpers import RE
from openpilot.tools.ycpe.daemon import cache, jobs
from openpilot.tools.ycpe.daemon.cache import Key
from openpilot.tools.ycpe.daemon.core import HTTPError, config, module, route
from openpilot.tools.ycpe.daemon.sources import FILES, KINDS, parse_route

HOST = re.compile(r"^(?:[a-z_][a-z0-9_-]*@)?[A-Za-z0-9][A-Za-z0-9.:-]*\Z")  # [user@]host, never an ssh option
REALDATA = "/data/media/0/realdata"
DONGLE_FILE = "../../../params/d/DongleId"  # from realdata: /data/params/d/DongleId
SEG_DIR = re.compile(rf"^{RE.LOG_ID}--(?P<seg>\d+)\Z")  # LOG_ID is the log_id group

module("device", "1")
settings = {"host": None}  # a host set from the page wins over serve.py's


def realdata() -> str:
  return config.device_root or REALDATA


def host() -> str:
  h = settings["host"] or config.device
  if not h:
    raise HTTPError(409, "no device configured: serve.py --device <host>, or POST /local/device {host}")
  if not HOST.match(h):
    raise HTTPError(400, "bad device host")
  return h if "@" in h else f"comma@{h}"


def ssh(remote: str) -> list[str]:
  """argv running remote on the device. One master connection per host is reused for 60 s."""
  return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new",
          "-o", "ControlMaster=auto", "-o", f"ControlPath=/tmp/ycpe-ssh-{os.getuid()}-%C", "-o", "ControlPersist=60",
          *(["-i", config.device_key] if config.device_key else []), "-p", "22", host(), remote]


def run(remote: str, timeout: float = 20) -> str:
  try:
    p = subprocess.run(ssh(remote), capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired as e:
    raise HTTPError(504, "device didn't answer") from e
  if p.returncode == 255:
    raise HTTPError(502, f"ssh: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else 'failed'}")
  return p.stdout


def listing() -> dict:
  """{dongle, routes: [{route, log_id, segments: {n: {file: size}}, mtime, active}]}, newest first. A segment with a
  .lock file is still being written (the drive in progress), so it's marked active and not offered for pull."""
  root, dongle_file = shlex.quote(realdata()), shlex.quote(posixpath.normpath(posixpath.join(realdata(), DONGLE_FILE)))
  out = run(f"cat {dongle_file} 2>/dev/null; echo; find {root} -mindepth 2 -maxdepth 2 -type f -printf '%P\\t%s\\t%T@\\n' 2>/dev/null")
  first, _, rest = out.partition("\n")
  dongle = first.strip() if re.fullmatch(RE.DONGLE_ID, first.strip()) else None
  routes: dict[str, dict] = {}
  for line in rest.splitlines():
    try:
      path, size, mtime = line.split("\t")
      seg_dir, name = path.split("/")
    except ValueError:
      continue
    m = SEG_DIR.match(seg_dir)
    if not m:
      continue
    r = routes.setdefault(m["log_id"], {"route": f"{dongle}|{m['log_id']}" if dongle else None, "log_id": m["log_id"],
                                        "segments": {}, "mtime": 0.0, "active": []})
    seg = r["segments"].setdefault(int(m["seg"]), {})
    r["mtime"] = max(r["mtime"], float(mtime))
    if name.endswith(".lock"):
      r["active"].append(int(m["seg"]))
    elif name in FILES:
      seg[name] = int(size)
  for r in routes.values():
    r["segments"] = dict(sorted(r["segments"].items()))
  return {"dongle": dongle, "host": host(), "routes": sorted(routes.values(), key=lambda r: -r["mtime"])}


@route("GET", "/local/device")
def device_info(req):
  return {"host": settings["host"] or config.device, "root": realdata()}


@route("POST", "/local/device")
def device_config(req):
  """Body {host}: the device to use until the daemon restarts."""
  h = str((req.json() or {}).get("host") or "")
  if not HOST.match(h):
    raise HTTPError(400, "bad device host")
  settings["host"] = h
  return device_info(req)


@route("GET", "/local/device/routes")
def device_routes(req):
  return listing()


def pull_one(key: Key, size: int, job, progress) -> int:
  """Copies one file off the device into the shared cache with `ssh cat`; returns bytes."""
  dest = cache.device_path(key)
  dest.parent.mkdir(parents=True, exist_ok=True)
  if dest.is_file() and dest.stat().st_size == size:
    return 0
  tmp = dest.with_name(f"{cache.TMP}{dest.name}.{threading.get_ident()}")
  remote = f"{realdata()}/{key.log_id}--{key.seg}/{key.file}"
  p = subprocess.Popen(ssh(f"cat {shlex.quote(remote)}"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  n = 0
  try:
    with open(tmp, "wb") as f:
      while (data := p.stdout.read(1 << 20)):
        f.write(data)
        n += len(data)
        progress(len(data))
        job.check()
    if p.wait(timeout=30) != 0 or n != size:
      raise OSError(f"{key.file} of segment {key.seg}: got {n} of {size} bytes ({p.stderr.read().decode().strip()[-200:]})")
    os.replace(tmp, dest)
    cache.account(n, 1)
    return n
  finally:
    if p.poll() is None:
      p.kill()
    p.stdout.close()
    p.stderr.close()
    if tmp.exists():
      tmp.unlink()


@route("POST", "/local/device/pull")
def device_pull(req):
  """Body {route ("dongle|log_id" or a bare log_id, with an optional slice), segs?, kinds? (default ["rlog", "qlog"])}.
  Copies those files into the shared cache as a job; segments still being recorded are skipped."""
  body = req.json() or {}
  text = str(body.get("route", ""))
  info = listing()
  if "|" not in text and "/" not in text and info["dongle"]:
    text = f"{info['dongle']}|{text}"
  dongle, log_id, sl, _ = parse_route(text)
  r = next((r for r in info["routes"] if r["log_id"] == log_id), None)
  if r is None:
    raise HTTPError(404, "route not on the device")
  kinds = body.get("kinds") or ["rlog", "qlog"]
  names = {n for k in kinds if k in KINDS for n in KINDS[k]}
  segs = body.get("segs")
  if segs is None:
    from openpilot.tools.ycpe.daemon.sources import pick
    segs = pick(max(r["segments"], default=-1) + 1, sl)
  todo = [(Key(dongle, log_id, n, name), size) for n in segs if n not in r["active"]
          for name, size in r["segments"].get(n, {}).items() if name in names]
  total = sum(size for _, size in todo)
  states = {str(n): "queued" for n in {k.seg for k, _ in todo}}

  def work(job):
    done, t0 = 0, time.monotonic()
    job.progress(0, total, unit="bytes", segs=dict(states))

    def progress(nbytes):
      nonlocal done
      done += nbytes
      dt = time.monotonic() - t0
      job.progress(done, text=f"{done / dt / 1e6:.0f} MB/s" if dt > 0.5 else "", segs=dict(states))
    for key, size in todo:
      states[str(key.seg)] = "loading"
      try:
        if not pull_one(key, size, job, progress):
          progress(size)  # already here
      except jobs.Cancelled:
        states[str(key.seg)] = "queued"
        raise
      except Exception:
        states[str(key.seg)] = "error"
        raise
      if all(k.seg != key.seg or k == key or cache.device_path(k).is_file() for k, _ in todo):
        states[str(key.seg)] = "disk"
    job.progress(total, total, segs=dict(states))
    return {"files": len(todo), "bytes": total, "seconds": round(time.monotonic() - t0, 2), "skipped": r["active"]}

  return jobs.start("pull", work, title=f"pull {log_id} from {host().split('@')[-1]}", route=f"{dongle}|{log_id}", segs=dict(states)).json()
