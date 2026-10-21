"""Long tasks (prefetch, device pull, transcode) as jobs in threads, with progress, cancel and an SSE feed."""
import itertools
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait

from openpilot.tools.ycpe.daemon.core import Feed, HTTPError, route

KEEP = 200  # finished jobs remembered
STATES = ["error", "loading", "queued", "disk"]  # a segment's state is its files' first in this order


class Cancelled(Exception):
  pass


class Job:
  def __init__(self, kind: str, title: str, meta: dict):
    self.id = f"{kind}-{next(_ids)}"
    self.kind, self.title, self.meta = kind, title, meta
    self.state = "running"  # running | done | error | cancelled
    self.done, self.total, self.unit, self.text = 0, 0, "", ""
    self.result = self.error = None
    self.started, self.ended = time.time(), None  # noqa: TID251  wall clock, for display
    self.cancel_event = threading.Event()
    self.finished = threading.Event()  # set once it's done, failed or cancelled
    self._sent = 0.0

  @property
  def cancelled(self) -> bool:
    return self.cancel_event.is_set()

  def check(self) -> None:
    """Raises Cancelled once cancel was asked for; call it between steps."""
    if self.cancelled:
      raise Cancelled

  def progress(self, done=None, total=None, text=None, unit=None, **meta) -> None:
    """Updates progress; listeners hear about it at most every 100 ms (and always on state changes)."""
    if done is not None:
      self.done = done
    if total is not None:
      self.total = total
    if text is not None:
      self.text = text
    if unit is not None:
      self.unit = unit
    self.meta.update(meta)
    now = time.monotonic()
    if now - self._sent >= 0.1:
      self._sent = now
      _publish(self)

  def json(self) -> dict:
    return {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state, "done": self.done, "total": self.total,
            "unit": self.unit, "text": self.text, "result": self.result, "error": self.error, "started": self.started,
            "ended": self.ended, "meta": self.meta}


_ids = itertools.count(1)
_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_feed = Feed()


def _publish(job: Job) -> None:
  _feed.publish(job.json())


def start(kind: str, fn, title: str = "", **meta) -> Job:
  """Runs fn(job) in a thread. fn reports with job.progress(), calls job.check() to stop when cancelled, and
  returns a JSON-able result."""
  job = Job(kind, title or kind, meta)
  with _lock:
    _jobs[job.id] = job
    for old in [j for j in _jobs.values() if j.state != "running"][:-KEEP]:
      del _jobs[old.id]

  def run():
    try:
      job.result = fn(job)
      job.state = "done"
    except Cancelled:
      job.state = "cancelled"
    except Exception as e:
      job.state, job.error = "error", f"{type(e).__name__}: {e}"
      traceback.print_exc()
    job.ended = time.time()  # noqa: TID251
    _publish(job)
    job.finished.set()

  _publish(job)
  threading.Thread(target=run, name=job.id, daemon=True).start()
  return job


def run_files(job: Job, todo: list, one, workers: int = 1, have=()) -> dict:
  """Runs one(key, moved) for each (key, size) in todo, workers at a time, as job's work. job.meta["segs"] has each
  segment's state (queued, loading, disk or error; the keys in have are on disk already) and the progress counts
  bytes, moved(n) saying n of key's are in so far. → {files, bytes, seconds}; if any failed, raises once the rest are
  through."""
  total, t0 = sum(size for _, size in todo), time.monotonic()
  state = dict.fromkeys(have, "disk") | {key: "queued" for key, _ in todo}
  moving: dict = {}  # key → bytes in, while it loads

  def segs() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, s in state.items():
      out[str(key.seg)] = min(out.get(str(key.seg), "disk"), s, key=STATES.index)
    return out

  def report() -> None:
    done = sum(size for key, size in todo if state[key] == "disk") + sum(list(moving.values()))  # list(): workers add and drop keys meanwhile
    dt = time.monotonic() - t0
    job.progress(done, total, text=f"{done / dt / 1e6:.0f} MB/s" if dt > 0.5 else "", segs=segs())

  def run(key) -> None:
    def moved(n: int) -> None:
      moving[key] = n
    state[key] = "loading"
    try:
      one(key, moved)
      state[key] = "disk"
    except Cancelled:
      state[key] = "queued"
      raise
    except Exception:
      state[key] = "error"
      raise
    finally:
      moving.pop(key, None)

  job.progress(0, total, unit="bytes", segs=segs())
  with ThreadPoolExecutor(workers) as ex:
    futs = [ex.submit(run, key) for key, _ in todo]
    while wait(futs, timeout=0.2).not_done:
      if job.cancelled:
        ex.shutdown(cancel_futures=True)
        job.check()
      report()
  report()
  if (errors := [e for f in futs if (e := f.exception())]):
    raise OSError(f"{len(errors)} of {len(todo)} files failed, first: {errors[0]}")
  return {"files": len(todo), "bytes": total, "seconds": round(time.monotonic() - t0, 2)}


def get(job_id: str) -> Job:
  if (job := _jobs.get(job_id)) is None:
    raise HTTPError(404, "no such job")
  return job


@route("GET", "/local/jobs/events")
def job_events(req):
  """SSE: one "job" event per running or recent job on connect, then one per update."""
  def current():
    with _lock:  # start() drops old jobs from another thread
      return [j.json() for j in _jobs.values()]
  return _feed.stream("job", current)


@route("POST", "/local/jobs/{id}/cancel")
def job_cancel(req):
  job = get(req.params["id"])
  job.cancel_event.set()
  return job.json()
