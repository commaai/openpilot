"""Long tasks (prefetch, device pull, transcode, replay) as jobs in threads, with progress, cancel and an SSE feed."""
import itertools
import threading
import time
import traceback

from openpilot.tools.ycpe.daemon.core import Feed, HTTPError, module, route

KEEP = 200  # finished jobs remembered


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

  _publish(job)
  threading.Thread(target=run, name=job.id, daemon=True).start()
  return job


def get(job_id: str) -> Job:
  if (job := _jobs.get(job_id)) is None:
    raise HTTPError(404, "no such job")
  return job


module("jobs")


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
