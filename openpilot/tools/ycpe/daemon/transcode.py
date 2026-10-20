"""HEVC camera segments → H.264 the browser decodes in hardware, encoded on the GPU when there is one.

One encoder pass writes two files: Annex B (.h264, what YCPE's WebCodecs decoder reads) and fragmented MP4
(.mp4, for <video> and downloads). Frame i of the output is frame i of the source (no B-frames, no drops, 20 fps
timestamps), so the encodeIdx frame mapping stays exact, and every GOP-th frame is an IDR with its parameter sets.
Both files can be read while they grow; the .json written next to them at the end marks them complete.

GET /local/video/<dongle>/<log_id>/<seg>/<cam>.h264|.mp4|.json?preset=medium&ahead=2
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

FPS = 20
CAMERAS = ("fcamera", "ecamera", "dcamera")
GOP = 20                     # frames per keyframe: a seek decodes at most a second
CONCURRENT = 2               # encodes at once; the GPU and the CPUs are shared
AHEAD = 2                    # segments encoded ahead of the one asked for
BUDGET = 10 * 1000 ** 3      # bytes of cached output
FFMPEG = shutil.which("ffmpeg", path="/usr/bin:/usr/local/bin") or shutil.which("ffmpeg") or "ffmpeg"

# constant quality per backend family (nvenc -cq, vaapi -qp, x264 -crf)
PRESETS = {
  "low": {"nvenc": 33, "vaapi": 33, "x264": 30},
  "medium": {"nvenc": 28, "vaapi": 28, "x264": 26},
  "high": {"nvenc": 23, "vaapi": 23, "x264": 21},
}


@dataclass(frozen=True)
class Backend:
  name: str
  family: str
  decode: tuple[str, ...]  # input options: decode on the same device, straight into the encoder's memory
  encode: tuple[str, ...]
  quality: str


BACKENDS = [
  Backend("h264_nvenc", "nvenc", ("-hwaccel", "cuda", "-hwaccel_output_format", "cuda"),
          ("-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-rc", "vbr", "-forced-idr", "1"), "-cq"),
  Backend("h264_vaapi", "vaapi", ("-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128", "-hwaccel_output_format", "vaapi"),
          ("-c:v", "h264_vaapi"), "-qp"),
  Backend("libx264", "x264", (), ("-c:v", "libx264", "-preset", "ultrafast", "-x264-params", f"scenecut=0:min-keyint={GOP}"), "-crf"),
]


def ffmpeg(*args: str) -> list[str]:
  return [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]


def encoder_args(b: Backend, preset: str, src: str) -> list[str]:
  return ffmpeg(*b.decode, "-f", "hevc", "-r", str(FPS), "-i", src, "-map", "0:v:0", *b.encode, b.quality, str(PRESETS[preset][b.family]),
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
    ok = [b for b in BACKENDS if (r := subprocess.run(encoder_args(b, "medium", "pipe:"), input=clip, capture_output=True, timeout=30)).returncode == 0
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
  key: str                 # <dongle>/<log_id>/<seg>/<cam>-<preset>
  base: Path               # output path without the extension
  preset: str
  priority: float
  src: str = ""            # local path or URL of the .hevc, resolved when the encode starts
  state: str = "queued"    # queued | running | done | error | missing (no such source file)
  frames: int = 0          # complete pictures written so far
  total: int = 60 * FPS    # pictures expected, for progress
  backend: str = ""
  error: str = ""
  info: dict = field(default_factory=dict)
  first_frame_ms: int | None = None
  writing: threading.Event = field(default_factory=threading.Event)  # the output files exist
  finished: threading.Event = field(default_factory=threading.Event)

  @property
  def running(self) -> bool:
    return self.state in ("queued", "running")

  def json(self) -> dict:
    return {"key": self.key, "state": self.state, "frames": self.frames, "total": self.total, "backend": self.backend,
            "preset": self.preset, "error": self.error, "firstFrameMs": self.first_frame_ms, **self.info}


def encode(job: Encode, backend: Backend, cancelled=lambda: False, progress=lambda job: None) -> dict:
  """Encodes job.src into job.base.{h264,mp4} and writes the summary to job.base.json."""
  h264, mp4 = job.base.with_suffix(".h264"), job.base.with_suffix(".mp4")
  job.base.parent.mkdir(parents=True, exist_ok=True)
  start = time.monotonic()
  pics, size = Pictures(), 0
  # stderr goes to files: a corrupt input can log more than a pipe holds
  with tempfile.TemporaryFile() as enc_err, tempfile.TemporaryFile() as mux_err, open(h264, "wb") as out:
    enc = subprocess.Popen(niced(encoder_args(backend, job.preset, job.src)), stdout=subprocess.PIPE, stderr=enc_err)
    mux = subprocess.Popen(niced(ffmpeg("-f", "h264", "-r", str(FPS), "-i", "pipe:", "-c", "copy", "-flush_packets", "1",
                                        "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", str(mp4))),
                           stdin=subprocess.PIPE, stderr=mux_err)
    job.writing.set()
    try:
      while (chunk := enc.stdout.read1(1 << 16)):
        if cancelled():
          raise InterruptedError("cancelled")
        out.write(chunk)
        out.flush()
        mux.stdin.write(chunk)
        size += len(chunk)
        pics.feed(chunk)
        job.frames = max(0, pics.count - 1)  # a picture is complete once the next one starts
        if job.frames and job.first_frame_ms is None:
          job.first_frame_ms = round((time.monotonic() - start) * 1000)
        progress(job)
      mux.stdin.close()
      if enc.wait() != 0 or not pics.count:
        enc_err.seek(0)
        raise RuntimeError(f"{backend.name}: {enc_err.read().decode(errors='replace').strip()[-500:] or 'no output'}")
      if mux.wait() != 0:
        mux_err.seek(0)
        raise RuntimeError(f"mp4 mux: {mux_err.read().decode(errors='replace').strip()[-500:]}")
    except BaseException:
      for p in (enc, mux):
        p.kill()
        p.wait()
      h264.unlink(missing_ok=True)
      mp4.unlink(missing_ok=True)
      raise
  ms = (time.monotonic() - start) * 1000
  job.frames = pics.count
  info = {"frames": pics.count, "keyframes": pics.keys, "backend": backend.name, "preset": job.preset, "gop": GOP, "fps": FPS,
          "encodeMs": round(ms), "encodeFps": round(pics.count * 1000 / ms), "firstFrameMs": job.first_frame_ms,
          "h264Bytes": size, "mp4Bytes": mp4.stat().st_size, "srcBytes": _size(job.src)}
  tmp = job.base.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(info))
  os.replace(tmp, job.base.with_suffix(".json"))
  return info


def _size(src: str) -> int | None:
  return os.path.getsize(src) if os.path.isfile(src) else None


class Transcoder:
  """Encodes on request and ahead of it into root under a byte budget, CONCURRENT at a time, nearest first.
  resolve(dongle, log_id, seg, cam) gives the source .hevc as a local path or URL (ffmpeg streams URLs, so the
  first frames come out before the download finishes). run(job, fn) runs fn() for job, e.g. as a daemon job."""
  def __init__(self, root: str | os.PathLike, resolve, budget: int = BUDGET, workers: int = CONCURRENT, run=None):
    self.root = Path(root)
    self.resolve = resolve
    self.budget = budget
    self.run = run or (lambda job, fn: fn(lambda: False, lambda job: None))
    self.jobs: dict[str, Encode] = {}
    self.queue: list[tuple[float, int, str]] = []
    self.lock = threading.Condition()
    self.order = itertools.count()
    self.stopped = False
    self._clean()
    for i in range(workers):
      threading.Thread(target=self._work, name=f"transcode-{i}", daemon=True).start()

  def get(self, dongle: str, log_id: str, seg: int, cam: str, preset: str = "medium", priority: float = 0) -> Encode:
    """The encode of a segment: finished in the cache, running, or queued now. Lower priorities go first."""
    if cam not in CAMERAS or preset not in PRESETS:
      raise ValueError(f"bad camera or preset: {cam} {preset}")
    key = f"{dongle}/{log_id}/{seg}/{cam}-{preset}"
    base = self.root / key
    with self.lock:
      job = self.jobs.get(key)
      if job is not None and job.state == "done" and not base.with_suffix(".json").exists():
        job = None  # evicted
      if job is None and base.with_suffix(".json").exists():
        job = self.jobs[key] = Encode(key, base, preset, priority, state="done", info=json.loads(base.with_suffix(".json").read_text()))
        job.frames = job.total = job.info["frames"]
        job.writing.set()
        job.finished.set()
      if job is None or job.state == "error":
        job = self.jobs[key] = Encode(key, base, preset, priority)
        job.src = (dongle, log_id, seg, cam)
        self._push(job)
      elif job.state == "queued" and priority < job.priority:
        job.priority = priority
        self._push(job)
      if job.state == "done":
        os.utime(base.with_suffix(".json"))  # recently used
      return job

  def prefetch(self, dongle: str, log_id: str, segs, cam: str, preset: str = "medium") -> None:
    """Queues segments behind whatever was asked for directly, in the given order."""
    for i, seg in enumerate(segs):
      self.get(dongle, log_id, seg, cam, preset, priority=1 + i)

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
        job.state = "running"
      try:
        if isinstance(job.src, tuple):
          job.src = self.resolve(*job.src)
        self.run(job, functools.partial(self._encode, job))
      except FileNotFoundError as e:
        job.state, job.error = "missing", str(e)
      except Exception as e:
        job.state, job.error = "error", job.error or str(e)
      if job.state == "done":
        self._evict(keep=job.base)
      job.finished.set()

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
    """Drops outputs without a .json: encodes cut short by a restart."""
    for f in self.root.glob("*/*/*/*.h264"):
      if not f.with_suffix(".json").exists():
        f.unlink(missing_ok=True)
        f.with_suffix(".mp4").unlink(missing_ok=True)

  def _evict(self, keep: Path) -> None:
    """Deletes the least recently used finished encodes but keep until the cache fits the budget."""
    done = []
    for meta in self.root.glob("*/*/*/*.json"):
      if meta.with_suffix("") == keep:
        continue
      files = [meta.with_suffix(s) for s in (".h264", ".mp4", ".json")]
      done.append((meta.stat().st_mtime, sum(f.stat().st_size for f in files if f.exists()), files))
    total = sum(size for _, size, _ in done)
    for _, size, files in sorted(done, key=lambda d: d[0]):
      if total <= self.budget:
        break
      for f in files:
        f.unlink(missing_ok=True)
      total -= size

  def close(self) -> None:
    with self.lock:
      self.stopped = True
      self.lock.notify_all()


# the daemon side

try:
  from openpilot.tools.ycpe.daemon import File, HTTPError, config, jobs, module, route
except ImportError:  # used as a library without the daemon
  route = None

if route is not None:
  NAME = re.compile(r"^[0-9a-zA-Z_-]+$")  # dongle ids and log ids
  _transcoder: Transcoder | None = None
  _transcoder_lock = threading.Lock()

  module("transcode", "1", presets=list(PRESETS), default="medium", gop=GOP, fps=FPS, formats=["h264", "mp4"])

  _files: dict[str, tuple[float, dict]] = {}

  def resolve_api(dongle: str, log_id: str, seg: int, cam: str) -> str:
    """The camera file's URL from the comma API, with the ~/.comma sign-in for private routes. The signed URLs
    last an hour; the listing is reused for ten minutes."""
    from openpilot.tools.lib.api import CommaApi
    from openpilot.tools.lib.auth_config import get_token
    name = f"{dongle}|{log_id}"
    when, files = _files.get(name, (0.0, {}))
    if time.monotonic() - when > 600:
      files = CommaApi(get_token()).get(f"v1/route/{name}/files")
      _files[name] = (time.monotonic(), files)
    kind = {"fcamera": "cameras", "ecamera": "ecameras", "dcamera": "dcameras"}[cam]
    for url in files.get(kind, []):
      if url.split("?")[0].split("/")[-2] == str(seg):
        return url
    raise FileNotFoundError(f"no {cam} for segment {seg}")

  def run_as_job(enc: Encode, fn) -> None:
    """Runs the encode as a daemon job, so the page sees its progress and can cancel it."""
    ended = threading.Event()
    def body(job):
      try:
        return fn(lambda: job.cancelled, lambda e: job.progress(e.frames, e.total, text=e.backend, unit="frames"))
      except InterruptedError as e:
        enc.error = "cancelled"
        raise jobs.Cancelled from e
      finally:
        ended.set()
    dongle, log_id, seg, cam = enc.key.split("/")
    camera = cam.split("-")[0]
    jobs.start("transcode", body, title=f"{camera} {seg} → h264", route=f"{dongle}|{log_id}", seg=int(seg), camera=camera, preset=enc.preset)
    ended.wait()
    if enc.state != "done":
      enc.state = "error"

  def transcoder() -> Transcoder:
    global _transcoder
    with _transcoder_lock:
      if _transcoder is None:
        root = Path(config.cache_root or tempfile.gettempdir()) / "ycpe" / "transcode"
        _transcoder = Transcoder(root, resolve_api, run=run_as_job)
      return _transcoder

  def _job(req) -> tuple[Encode, str]:
    p = req.params
    stem, _, ext = p["file"].rpartition(".")
    if not (NAME.match(p["dongle"]) and NAME.match(p["log_id"]) and p["seg"].isdigit() and ext in ("h264", "mp4", "json")):
      raise HTTPError(404)
    preset, seg = req.query.get("preset", "medium"), int(p["seg"])
    try:
      t = transcoder()
      job = t.get(p["dongle"], p["log_id"], seg, stem, preset)
      t.prefetch(p["dongle"], p["log_id"], range(seg + 1, seg + 1 + req.arg("ahead", AHEAD, int)), stem, preset)
    except ValueError as e:
      raise HTTPError(404, str(e)) from e
    return job, ext

  @route("GET", "/local/video/{dongle}/{log_id}/{seg}/{file}")
  def video(req):
    """The transcoded segment, followed as it's written while the encode runs. .json is its state or summary."""
    job, ext = _job(req)
    if ext == "json":
      return job.json()
    while not job.writing.wait(0.25):
      if job.state in ("error", "missing"):
        raise HTTPError(404 if job.state == "missing" else 502, job.error)
    path = job.base.with_suffix(f".{ext}")
    for _ in range(100):  # the mp4 muxer creates its file a moment after starting
      if path.exists() or job.state == "error":
        break
      time.sleep(0.02)
    if job.state == "error":
      raise HTTPError(502, job.error)
    return File(str(path), "video/mp4" if ext == "mp4" else "video/h264", {"X-Frames": str(job.total)},
                growing=lambda: job.state in ("queued", "running"))

  @route("GET", "/local/video/backends")
  def video_backends(req):
    return {"backends": [b.name for b in backends()], "presets": PRESETS, "gop": GOP, "concurrent": CONCURRENT}
