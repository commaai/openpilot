"""Re-runs openpilot processes from a local checkout over a route's segments: "what would my branch have done here".

A replay is a daemon job running replay_runner.py in the checkout's environment. It writes one rlog per segment,
holding the segment's original initData plus everything the processes published, so the page loads it like any
route log on the same clock, plus a diff summary against the original log. A replay starting mid-route runs over
the segment before too, without writing it, so engagement, calibration and learners are warm when the range starts.

GET  /local/replay/procs?repo=<path>     the replayable processes and what each is missing in that checkout
POST /local/replay                       {route, segs: [first, last], procs, repo} → the replay's manifest
GET  /local/replay?route=<dongle|log_id> replays of a route, newest first
GET  /local/replay/<id>                  manifest: state, segments written, diff summary when done
GET  /local/replay/<id>/<seg>/rlog.zst   a replayed segment
"""
import glob
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

RUNNER = Path(__file__).with_name("replay_runner.py")
DEFAULT_PROCS = ["controlsd", "plannerd", "radard"]
SUBMODULES = ["msgq_repo", "opendbc_repo", "rednose_repo", "tinygrad_repo", "teleoprtc_repo"]
ERROR_LINES = 30
MAX_SEGS = 100  # segments in one replay: about 15 minutes of work for the default processes

# what each process needs built, beyond plain Python: (description, globs relative to the checkout, any one will do)
COMMON = [("msgq", ["msgq_repo/msgq/ipc_pyx*.so"]),
          ("visionipc", ["msgq_repo/msgq/visionipc/visionipc_pyx*.so"]),
          ("params", ["{pkg}/common/libparams_c.so", "{pkg}/common/params_pyx*.so"])]
NATIVE = {
  "plannerd": [("longitudinal MPC (acados)", ["{pkg}/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx*.so"])],
  "locationd": [("pose EKF", ["{pkg}/selfdrive/locationd/models/generated/libpose.so"]),
                ("rednose", ["rednose_repo/rednose/helpers/ekf_sym_pyx*.so"])],
  "paramsd": [("car EKF", ["{pkg}/selfdrive/locationd/models/generated/libcar.so"]),
              ("rednose", ["rednose_repo/rednose/helpers/ekf_sym_pyx*.so"])],
  "modeld": [("driving model", ["{pkg}/selfdrive/modeld/models/driving*_tinygrad.pkl"])],
  "dmonitoringmodeld": [("driver monitoring model", ["{pkg}/selfdrive/modeld/models/dmonitoring_model_tinygrad.pkl"])],
}
# processes whose inputs include camera frames: not supported here yet
VISION = {"modeld", "dmonitoringmodeld"}
PROCS = ["selfdrived", "controlsd", "card", "radard", "plannerd", "calibrationd", "dmonitoringd", "locationd", "paramsd", "lagd",
         "torqued", "ubloxd"]


def package(root: Path) -> Path:
  """The directory that is the openpilot package: nested (root/openpilot) or the checkout itself (older layout)."""
  return root / "openpilot" if (root / "openpilot" / "selfdrive").is_dir() else root


def missing(root: Path, procs: list[str]) -> dict[str, list[str]]:
  """For each process (and "all"), the native parts not built in the checkout, as "what (path)"."""
  pkg = os.path.relpath(package(root), root)
  def check(reqs):
    out = []
    for what, globs in reqs:
      pats = [g.format(pkg=pkg) for g in globs]
      if not any(glob.glob(str(root / p)) for p in pats):
        out.append(f"{what} ({pats[0].replace('*', '')})")
    return out
  found = {"all": check(COMMON)}
  for p in procs:
    found[p] = check(NATIVE.get(p, [])) + (["camera frames aren't supported by the daemon yet"] if p in VISION else [])
  return {k: v for k, v in found.items() if v}


def environment(root: Path, shims: Path) -> tuple[str, dict]:
  """The checkout's Python and an environment that imports openpilot, msgq, opendbc, ... from it."""
  pkg = package(root)
  if pkg == root:  # an older checkout is the package itself: make it importable as "openpilot"
    shim = shims / hashlib.sha1(str(root).encode()).hexdigest()[:12]
    shim.mkdir(parents=True, exist_ok=True)
    if not (shim / "openpilot").exists():
      (shim / "openpilot").symlink_to(root)
    paths = [shim]
  else:
    paths = [root]
  paths += [root / s for s in SUBMODULES if (root / s).is_dir()]
  env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "OPENPILOT_PREFIX")}
  env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
  python = root / ".venv" / "bin" / "python"
  return str(python if python.exists() else sys.executable), env


def commit(root: Path) -> str | None:
  """HEAD of the checkout without running git: worktrees and submodules have a .git file pointing at the git dir."""
  try:
    git = root / ".git"
    if git.is_file():
      git = (root / git.read_text().split("gitdir:")[1].strip()).resolve()
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
      return head
    ref = head[5:]
    for d in (git, (git / "commondir").exists() and (git / (git / "commondir").read_text().strip()).resolve()):
      if d and (d / ref).exists():
        return (d / ref).read_text().strip()
      if d and (d / "packed-refs").exists():
        for line in (d / "packed-refs").read_text().splitlines():
          if line.endswith(" " + ref):
            return line.split()[0]
  except (OSError, IndexError):
    pass
  return None


class Replay:
  """One replay's directory and manifest; run() does the work and reports through report(done, total, text, **meta)."""
  def __init__(self, root: Path, route: str, segs: list[int], procs: list[str], repo: Path, rid: str | None = None):
    self.id = rid or time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    self.dir = root / self.id
    self.route, self.segs, self.procs, self.repo = route, segs, procs, repo
    self.written: list[int] = []
    self.warm: int | None = None  # the segment replayed only to warm up
    self.state, self.error, self.diff = "running", "", None
    self.started, self.seconds = time.time(), None  # noqa: TID251  a wall-clock date for the list of replays
    self.t0 = time.monotonic()

  def manifest(self) -> dict:
    return {"id": self.id, "route": self.route, "segs": self.segs, "warm": self.warm, "procs": self.procs, "repo": str(self.repo), "commit": commit(self.repo),
            "state": self.state, "error": self.error, "written": self.written, "started": self.started, "seconds": self.seconds,
            "diff": self.diff}

  def save(self) -> None:
    self.dir.mkdir(parents=True, exist_ok=True)
    tmp = self.dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(self.manifest()))
    os.replace(tmp, self.dir / "manifest.json")

  def run(self, fetch, report, cancelled=lambda: False, shims: Path | None = None) -> dict:
    """fetch(route, seg) → local rlog path. Raises with the exact cause on failure."""
    self.save()
    try:
      gaps = missing(self.repo, self.procs)
      if gaps:
        raise RuntimeError(f"{self.repo} is missing native parts; build them with scons there first: " +
                           "; ".join(f"{k}: {', '.join(v)}" for k, v in gaps.items()))
      rlogs = []
      if self.segs[0] > 0:
        report(0, len(self.segs), f"downloading segment {self.segs[0] - 1} to warm up on", phase="download")
        try:
          rlogs.append([self.segs[0] - 1, str(fetch(self.route, self.segs[0] - 1))])
          self.warm = self.segs[0] - 1
        except FileNotFoundError:  # never uploaded: start cold
          pass
      for i, seg in enumerate(self.segs):
        report(i, len(self.segs), f"downloading segment {seg}", phase="download")
        try:
          rlogs.append([seg, str(fetch(self.route, seg))])
        except FileNotFoundError:  # past the end of the route, or never uploaded
          if len(rlogs) == (self.warm is not None):
            raise
          break
        if cancelled():
          raise InterruptedError("cancelled")
      self.segs = [seg for seg, _ in rlogs if seg != self.warm]
      self._spawn(rlogs, report, cancelled, shims or self.dir.parent / "shims")
      self.diff = json.loads((self.dir / "diff.json").read_text())
      self.state, self.seconds = "done", time.monotonic() - self.t0
      return self.manifest()
    except InterruptedError:
      self.state = "cancelled"
      raise
    except Exception as e:
      self.state, self.error = ("cancelled", "") if cancelled() else ("error", str(e))
      raise
    finally:
      self.seconds = time.monotonic() - self.t0
      self.save()

  def _spawn(self, rlogs, report, cancelled, shims: Path) -> None:
    python, env = environment(self.repo, shims)
    args = json.dumps({"procs": self.procs, "segs": rlogs, "out": str(self.dir), "warm": self.warm is not None})
    cmd = ["nice", "-n", "10", python, str(RUNNER), args]
    p = subprocess.Popen(cmd, cwd=self.dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    tail: deque[str] = deque(maxlen=ERROR_LINES)
    drain = threading.Thread(target=lambda: tail.extend(p.stderr), daemon=True)  # tqdm, prints, the traceback
    drain.start()
    start = time.monotonic()
    stop = threading.Event()
    def watch():
      while not stop.wait(0.2):
        if cancelled():
          os.killpg(p.pid, signal.SIGKILL)
          return
    threading.Thread(target=watch, daemon=True).start()
    try:
      for line in p.stdout:
        try:
          ev = json.loads(line)
        except ValueError:
          tail.append(line)
          continue
        kind, segs = ev.pop("event"), len(self.segs)
        if kind == "load":
          report(ev["i"], len(rlogs), f"reading segment {ev['seg']}", phase="load")
        elif kind == "replay":
          start = time.monotonic()
          report(0, ev["messages"], f"replaying {', '.join(ev['procs'])}", phase="replay")
        elif kind == "progress":
          done, total = ev["done"], ev["total"]
          eta = (time.monotonic() - start) * (total - done) / done if done * 50 >= total else None  # too early to tell
          where = f"warming up on segment {ev['seg']}" if ev.get("warm") else f"segment {ev['seg']} · {len(self.written)}/{segs} written"
          report(done, total, where + ("" if eta is None else f" · {eta:.0f} s left"), phase="replay", eta=eta, seg=ev["seg"])
        elif kind == "segment":
          self.written.append(ev["seg"])
          self.save()
          report(None, None, None, written=self.written)
        elif kind == "diffing":
          report(None, None, "comparing with the original", phase="diff")
      code = p.wait()
      drain.join(5)
    finally:
      stop.set()
      if p.poll() is None:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    if cancelled():
      raise InterruptedError("cancelled")
    if code != 0:
      raise RuntimeError(f"replay exited with {code}:\n" + "".join(tail).strip())


def load_manifest(path: Path) -> dict | None:
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return None


# the daemon side

try:
  from openpilot.tools.ycpe.daemon import File, HTTPError, cache, config, confine, jobs, module, route
except ImportError:  # used as a library without the daemon
  route = None

if route is not None:
  ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
  ROUTE = re.compile(r"^[0-9a-f]{16}\|[0-9a-zA-Z_-]+$")
  _running: dict[str, Replay] = {}

  module("replay", "1", procs=PROCS, default=DEFAULT_PROCS)

  def root() -> Path:
    return cache.root() / "ycpe" / "replay"

  def checkouts() -> list[str]:
    try:
      from openpilot.tools.ycpe.daemon import repos
      return [c["path"] for c in repos.checkouts()]
    except ImportError:
      return list(config.repo_roots)

  def repo_arg(path: str | None) -> Path:
    roots = checkouts()
    if not path:
      if not roots:
        raise HTTPError(400, "no checkouts configured")
      path = roots[0]
    if os.path.realpath(path) not in {os.path.realpath(r) for r in roots}:
      raise HTTPError(403, "not a known checkout")
    return confine(path, roots)

  @route("GET", "/local/replay/procs")
  def replay_procs(req):
    repo = repo_arg(req.query.get("repo"))
    gaps = missing(repo, PROCS + sorted(VISION))
    return {"repo": str(repo), "commit": commit(repo), "default": DEFAULT_PROCS, "missing": gaps.get("all", []),
            "procs": [{"name": p, "missing": gaps.get(p, [])} for p in PROCS + sorted(VISION)]}

  @route("POST", "/local/replay")
  def replay_start(req):
    body = req.json() or {}
    name, segs, procs = body.get("route", ""), body.get("segs"), body.get("procs") or DEFAULT_PROCS
    if not ROUTE.match(name):
      raise HTTPError(400, "bad route")
    if not (isinstance(segs, list) and len(segs) == 2 and all(isinstance(s, int) and 0 <= s < 10000 for s in segs) and segs[0] <= segs[1]):
      raise HTTPError(400, "segs must be [first, last]")
    if segs[1] - segs[0] >= MAX_SEGS:
      raise HTTPError(400, f"at most {MAX_SEGS} segments at once")
    if not set(procs) <= set(PROCS) | VISION:
      raise HTTPError(400, f"unknown process in {procs}")
    replay = Replay(root(), name, list(range(segs[0], segs[1] + 1)), procs, repo_arg(body.get("repo")))
    _running[replay.id] = replay

    def body_fn(job):
      def report(done, total, text, **meta):
        job.progress(done, total, text=text, unit="segments" if meta.get("phase") in ("download", "load") else "messages", **meta)
      def fetch(name, seg):
        try:
          return cache.fetch(*name.split("|"), seg, "rlog.zst", job=job)
        except FileNotFoundError:
          return cache.fetch(*name.split("|"), seg, "rlog.bz2", job=job)  # older routes
      try:
        return replay.run(fetch, report, lambda: job.cancelled)
      except InterruptedError as e:
        raise jobs.Cancelled from e
      finally:
        _running.pop(replay.id, None)
        cache.prune_soon()

    job = jobs.start("replay", body_fn, title=f"replay {', '.join(procs)} on {name} {segs[0]}–{segs[1]}", replay=replay.id, route=name,
                     segs=replay.segs, procs=procs, written=[])
    return {**replay.manifest(), "job": job.id}

  @route("GET", "/local/replay")
  def replay_list(req):
    want = req.query.get("route")
    out = [m for p in root().glob("*/manifest.json") if (m := load_manifest(p)) and (not want or m["route"] == want)]
    for m in out:
      if m["state"] == "running" and m["id"] not in _running:
        m["state"] = "interrupted"  # the daemon restarted under it
    return sorted(out, key=lambda m: m["started"], reverse=True)

  def _replay_dir(rid: str) -> Path:
    if not ID.match(rid) or not (root() / rid / "manifest.json").exists():
      raise HTTPError(404, "no such replay")
    return root() / rid

  @route("GET", "/local/replay/{id}")
  def replay_get(req):
    rid = req.params["id"]
    return _running[rid].manifest() if rid in _running else load_manifest(_replay_dir(rid) / "manifest.json")

  @route("GET", "/local/replay/{id}/{seg}/rlog.zst")
  def replay_log(req):
    d, seg = _replay_dir(req.params["id"]), req.params["seg"]
    if not seg.isdigit() or not (d / seg / "rlog.zst").exists():
      raise HTTPError(404, "segment not replayed (yet)")
    return File(str(d / seg / "rlog.zst"), "application/octet-stream")
