"""HEVC camera segments → H.264 the browser decodes in hardware, encoded on the GPU when there is one.

One encoder pass writes Annex B (.h264), what YCPE's WebCodecs decoder reads. Frame i of the output is frame i of the
source (no B-frames, no drops), so the encodeIdx frame mapping stays exact, and every GOP-th frame is an IDR with its
parameter sets. The file can be read while it grows; the .json written next to it at the end marks it complete.

GET /local/video/<dongle>/<log_id>/<seg>/<cam>.h264, and the next AHEAD segments are encoded ahead of it
"""
import functools
import heapq
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from openpilot.tools.lib.helpers import RE
from openpilot.tools.ycpe.daemon import File, HTTPError, cache, config, confine, jobs, module, route, sources

FPS = 20
CAMERAS = ("fcamera", "ecamera", "dcamera")
OUTPUTS = (".h264", ".json")  # the stream, and its summary once it's complete
GOP = 20                     # frames per keyframe: a seek decodes at most a second
CONCURRENT = 2               # encodes at once; the GPU and the CPUs are shared
AHEAD = 2                    # segments encoded ahead of the one asked for
STALE = 60                   # seconds without a request for a route's camera after which its queued prefetches are dropped
QUALITY = "28"               # constant quality: nvenc -cq, vaapi -qp, x264 -crf
FFMPEG = shutil.which("ffmpeg", path="/usr/bin:/usr/local/bin") or shutil.which("ffmpeg") or "ffmpeg"


@dataclass(frozen=True)
class Backend:
  name: str
  decode: tuple[str, ...]  # input options: decode on the same device, straight into the encoder's memory
  encode: tuple[str, ...]


BACKENDS = [
  Backend("h264_nvenc", ("-hwaccel", "cuda", "-hwaccel_output_format", "cuda"),
          ("-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-rc", "vbr", "-forced-idr", "1", "-cq", QUALITY)),
  Backend("h264_vaapi", ("-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128", "-hwaccel_output_format", "vaapi"),
          ("-c:v", "h264_vaapi", "-qp", QUALITY)),
  Backend("libx264", (), ("-c:v", "libx264", "-preset", "ultrafast", "-x264-params", f"scenecut=0:min-keyint={GOP}", "-crf", QUALITY)),
]


def ffmpeg(*args: str) -> list[str]:
  return [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]


def encoder_args(b: Backend, src: str) -> list[str]:
  return ffmpeg(*b.decode, "-f", "hevc", "-r", str(FPS), "-i", src, "-map", "0:v:0", *b.encode,
                "-g", str(GOP), "-bf", "0", "-fps_mode", "passthrough", "-an", "-f", "h264", "-")


def niced(cmd: list[str]) -> list[str]:
  return (["ionice", "-c", "3"] if shutil.which("ionice") else []) + ["nice", "-n", "10", *cmd]


_backends: list[Backend] | None = None


def backends() -> list[Backend]:
  """The backends that get a test clip through on this machine, best first; libx264 comes last."""
  global _backends
  if _backends is None:
    clip = subprocess.run(ffmpeg("-f", "lavfi", "-i", "testsrc2=s=256x144:r=20:d=0.2", "-c:v", "libx265", "-x265-params", "log-level=none",
                                 "-f", "hevc", "-"), capture_output=True, timeout=30).stdout
    ok = [b for b in BACKENDS if (r := subprocess.run(encoder_args(b, "pipe:"), input=clip, capture_output=True, timeout=30)).returncode == 0
          and r.stdout]
    _backends = ok or [BACKENDS[-1]]
  return _backends


class Pictures:
  """Counts H.264 pictures in an Annex B stream going by: each starts with a slice whose first_mb_in_slice is 0."""
  def __init__(self):
    self.count = self.keys = 0
    self._tail = b""

  def feed(self, chunk: bytes) -> None:
    d = self._tail + chunk
    i = d.find(b"\x00\x00\x01")
    while 0 <= i < len(d) - 4:
      t = d[i + 3] & 31
      if t in (1, 5) and d[i + 4] & 0x80:
        self.count += 1
        self.keys += t == 5
      i = d.find(b"\x00\x00\x01", i + 3)
    self._tail = d[i:] if i >= 0 else d[-3:]


@dataclass
class Encode:
  """One segment's encode, queued, running or finished."""
  key: str                 # <dongle>/<log_id>/<seg>/<cam>
  base: Path               # output path without the extension
  priority: float
  src: str = ""            # local path or URL of the .hevc, resolved when the encode starts
  state: str = "queued"    # queued | running | done | error | missing (no such source file)
  frames: int = 0          # complete pictures written so far
  total: int = 60 * FPS    # pictures expected, for progress
  backend: str = ""
  error: str = ""
  info: dict = field(default_factory=dict)
  writing: threading.Event = field(default_factory=threading.Event)  # the output file exists
  finished: threading.Event = field(default_factory=threading.Event)

  @property
  def running(self) -> bool:
    return self.state in ("queued", "running")

  def json(self) -> dict:
    return {"key": self.key, "state": self.state, "frames": self.frames, "total": self.total, "backend": self.backend,
            "error": self.error, **self.info}


def encode(job: Encode, backend: Backend, cancelled=lambda: False, progress=lambda job: None) -> dict:
  """Encodes job.src into job.base.h264 and writes the summary to job.base.json."""
  h264 = job.base.with_suffix(".h264")
  job.base.parent.mkdir(parents=True, exist_ok=True)
  pics = Pictures()
  # stderr goes to a file: a corrupt input can log more than a pipe holds
  with tempfile.TemporaryFile() as err, open(h264, "wb") as out:
    enc = subprocess.Popen(niced(encoder_args(backend, job.src)), stdout=subprocess.PIPE, stderr=err)
    job.writing.set()
    try:
      while (chunk := enc.stdout.read1(1 << 16)):
        if cancelled():
          raise InterruptedError("cancelled")
        out.write(chunk)
        out.flush()
        pics.feed(chunk)
        job.frames = max(0, pics.count - 1)  # a picture is complete once the next one starts
        progress(job)
      if enc.wait() != 0 or not pics.count:
        err.seek(0)
        raise RuntimeError(f"{backend.name}: {err.read().decode(errors='replace').strip()[-500:] or 'no output'}")
    except BaseException:
      enc.kill()
      enc.wait()
      h264.unlink(missing_ok=True)
      raise
  job.frames = pics.count
  info = {"frames": pics.count, "keyframes": pics.keys, "backend": backend.name}
  tmp = job.base.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(info))
  os.replace(tmp, job.base.with_suffix(".json"))
  return info


class Transcoder:
  """Encodes on request and ahead of it into root, CONCURRENT at a time, nearest first. The shared cache prunes
  root by file mtime, so outputs are touched on use and an encode counts as cached only while both files exist.
  resolve(dongle, log_id, seg, cam) gives the source .hevc as a local path or URL (ffmpeg streams URLs, so the
  first frames come out before the download finishes). run(job, fn) runs fn() for job, e.g. as a daemon job;
  written() is called after each finished encode, e.g. to prune the cache."""
  def __init__(self, root: str | os.PathLike, resolve, workers: int = CONCURRENT, run=None, written=lambda: None):
    self.root = Path(root)
    self.resolve = resolve
    self.run = run or (lambda job, fn: fn(lambda: False, lambda job: None))
    self.written = written
    self.jobs: dict[str, Encode] = {}
    self.queue: list[tuple[float, int, str]] = []
    self.lock = threading.Condition()
    self.order = itertools.count()
    self.asked: dict[tuple[str, str, str], float] = {}  # (dongle, log_id, cam) → last direct request
    self.stopped = False
    self._clean()
    for i in range(workers):
      threading.Thread(target=self._work, name=f"transcode-{i}", daemon=True).start()

  def get(self, dongle: str, log_id: str, seg: int, cam: str, priority: float = 0) -> Encode:
    """The encode of a segment: finished in the cache, running, or queued now. Lower priorities go first."""
    if cam not in CAMERAS:
      raise ValueError(f"bad camera: {cam}")
    key = f"{dongle}/{log_id}/{seg}/{cam}"
    base = self.root / key
    with self.lock:
      if priority == 0:
        self.asked[(dongle, log_id, cam)] = time.monotonic()
      job = self.jobs.get(key)
      outputs = [base.with_suffix(s) for s in OUTPUTS]
      cached = all(f.exists() for f in outputs)
      if job is not None and job.state == "done" and not cached:
        job = None  # pruned
      if job is None and cached:
        job = self.jobs[key] = Encode(key, base, priority, state="done", info=json.loads(base.with_suffix(".json").read_text()))
        job.frames = job.total = job.info["frames"]
        job.writing.set()
        job.finished.set()
      if job is None or job.state == "error":
        job = self.jobs[key] = Encode(key, base, priority)
        job.src = (dongle, log_id, seg, cam)
        self._push(job)
      elif job.state == "queued" and priority < job.priority:
        job.priority = priority
        self._push(job)
      if job.state == "done":
        for f in outputs:  # recently used
          os.utime(f)
      return job

  def prefetch(self, dongle: str, log_id: str, segs, cam: str) -> None:
    """Queues segments behind whatever was asked for directly, in the given order."""
    for i, seg in enumerate(segs):
      self.get(dongle, log_id, seg, cam, priority=1 + i)

  def _push(self, job: Encode) -> None:
    heapq.heappush(self.queue, (job.priority, next(self.order), job.key))
    self.lock.notify()

  def _work(self) -> None:
    while True:
      with self.lock:
        while not self.queue and not self.stopped:
          self.lock.wait()
        if self.stopped:
          return
        _, _, key = heapq.heappop(self.queue)
        job = self.jobs.get(key)
        if job is None or job.state != "queued":
          continue  # left over from a priority bump
        dongle, log_id, _, cam = key.split("/")
        if job.priority > 0 and time.monotonic() - self.asked.get((dongle, log_id, cam), 0) > STALE:
          del self.jobs[key]  # a prefetch for a view nobody's watching anymore
          continue
        job.state = "running"
      try:
        if isinstance(job.src, tuple):
          job.src = self.resolve(*job.src)
        self.run(job, functools.partial(self._encode, job))
      except FileNotFoundError as e:
        job.state, job.error = "missing", str(e)
      except Exception as e:
        job.state, job.error = "error", job.error or str(e)
      job.finished.set()
      if job.state == "done":
        self.written()

  def _encode(self, job: Encode, cancelled, progress) -> dict:
    errors = []
    for b in backends():  # e.g. every NVENC session taken: the next backend
      job.backend = b.name
      try:
        job.info = encode(job, b, lambda: cancelled() or self.stopped, progress)
        job.state = "done"
        return job.json()
      except InterruptedError:
        raise
      except Exception as e:
        errors.append(str(e))
    job.error = "; ".join(errors)
    raise RuntimeError(job.error)

  def _clean(self) -> None:
    """Drops what's left of encodes cut short by a restart or partly pruned."""
    for f in self.root.glob("*/*/*/*.h264"):
      outputs = [f.with_suffix(s) for s in OUTPUTS]
      if not all(o.exists() for o in outputs):
        for o in outputs:
          o.unlink(missing_ok=True)

  def close(self) -> None:
    with self.lock:
      self.stopped = True
      self.lock.notify_all()


# the daemon side

_transcoder: Transcoder | None = None
_transcoder_lock = threading.Lock()

module("transcode", "1")

def resolve_source(dongle: str, log_id: str, seg: int, cam: str) -> str:
  """The .hevc from the first source that has it: a copy in the shared cache or a local log directory, else a URL
  ffmpeg streams (the API's signed with the ~/.comma sign-in, for private routes)."""
  key = cache.Key(dongle, log_id, seg, f"{cam}.hevc")
  if (hit := cache.lookup(key)):
    return str(hit[1])
  r = sources.Resolver(dongle, log_id, sources.SOURCES, None, False)
  for source in r.sources:
    if (found := r.find(source, key)):
      return str(confine(found[0], config.log_roots)) if source == "local" else found[0]
  raise FileNotFoundError(f"no {cam} for segment {seg}")


def run_as_job(enc: Encode, fn) -> None:
  """Runs the encode as a daemon job, so the page sees its progress and can cancel it."""
  ended = threading.Event()
  def body(job):
    job.progress(0, enc.total, text="starting", unit="frames")  # the total is known before the first frame
    try:
      return fn(lambda: job.cancelled, lambda e: job.progress(e.frames, e.total, text=e.backend, unit="frames"))
    except InterruptedError as e:
      enc.error = "cancelled"
      raise jobs.Cancelled from e
    finally:
      ended.set()
  dongle, log_id, seg, camera = enc.key.split("/")
  jobs.start("transcode", body, title=f"{camera} {seg} → h264", route=f"{dongle}|{log_id}", seg=int(seg), camera=camera)
  ended.wait()
  if enc.state != "done":
    enc.state = "error"


def transcoder() -> Transcoder:
  global _transcoder
  with _transcoder_lock:
    if _transcoder is None:
      _transcoder = Transcoder(cache.ycpe_dir("transcode"), resolve_source, run=run_as_job, written=cache.prune_soon)
    return _transcoder


@route("GET", "/local/video/{dongle}/{log_id}/{seg}/{file}")
def video(req):
  """The transcoded segment, followed as it's written while the encode runs."""
  p = req.params
  cam, _, ext = p["file"].rpartition(".")
  if not (re.fullmatch(RE.DONGLE_ID, p["dongle"]) and re.fullmatch(RE.LOG_ID, p["log_id"]) and p["seg"].isdigit() and ext == "h264"):
    raise HTTPError(404)
  seg = int(p["seg"])
  try:
    t = transcoder()
    job = t.get(p["dongle"], p["log_id"], seg, cam)
    t.prefetch(p["dongle"], p["log_id"], range(seg + 1, seg + 1 + AHEAD), cam)
  except ValueError as e:
    raise HTTPError(404, str(e)) from e
  while not job.writing.wait(0.25):
    if job.state in ("error", "missing"):
      raise HTTPError(404 if job.state == "missing" else 502, job.error)
  if job.state == "error":
    raise HTTPError(502, job.error)
  return File(str(job.base.with_suffix(".h264")), "video/h264", growing=lambda: job.state in ("queued", "running"))
