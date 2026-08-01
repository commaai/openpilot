#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import queue
import random
import re
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import partial, total_ordering
from queue import Queue
from typing import cast
from collections.abc import Callable

import requests
from requests.adapters import HTTPAdapter, DEFAULT_POOLBLOCK
from websocket import (ABNF, WebSocket, WebSocketException, WebSocketTimeoutException,
                       create_connection)

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.api import Api, get_key_pair
from openpilot.common.utils import CallbackReader, get_upload_stream
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.common.hardware import HARDWARE, PC
from openpilot.system.loggerd.xattr_cache import getxattr, setxattr
from openpilot.common.swaglog import cloudlog
from openpilot.common.version import get_build_metadata
from openpilot.common.hardware.hw import Paths
from openpilot.system.athena.rpc import dispatcher, dumps_call, handle, is_call, is_response, loads


ATHENA_HOST = os.getenv('ATHENA_HOST', 'wss://athena.comma.ai')
HANDLER_THREADS = int(os.getenv('HANDLER_THREADS', "4"))
LOCAL_PORT_WHITELIST = {22, }  # SSH

LOG_ATTR_NAME = 'user.upload'
LOG_ATTR_VALUE_MAX_UNIX_TIME = int.to_bytes(2147483647, 4, sys.byteorder)
RECONNECT_TIMEOUT_S = 70

RETRY_DELAY = 10  # seconds
MAX_RETRY_COUNT = 30  # Try for at most 5 minutes if upload fails immediately
MAX_AGE = 31 * 24 * 3600  # seconds
WS_FRAME_SIZE = 4096
DEVICE_STATE_UPDATE_INTERVAL = 1.0  # in seconds
DEFAULT_UPLOAD_PRIORITY = 99  # higher number = lower priority
VIDEO_CLIP_FPS = 20
VIDEO_CLIP_MAX_DURATION = 30 * 60
VIDEO_CLIP_BITRATES = (5, 8, 12)
VIDEO_CLIP_SPEEDUPS = (1, 2, 5, 10)
VIDEO_CLIP_CACHE_DIR = "clips"
VIDEO_CLIP_MANIFEST_VERSION = 1
VIDEO_CLIP_CAMERAS = {
  "fcamera": "fcamera.hevc",
  "ecamera": "ecamera.hevc",
  "dcamera": "dcamera.hevc",
}
VIDEO_CLIP_ROUTE_RE = re.compile(r"(?:[0-9]{4}(?:-[0-9]{2}){2}--(?:[0-9]{2}-){2}[0-9]{2}|[a-f0-9]{8}--[a-z0-9]{10})")
VIDEO_CLIP_DONGLE_ID_RE = re.compile(r"[a-f0-9]{16}")

SEND_PRIORITY_HIGH = 0
SEND_PRIORITY_LOW = 1

# https://bytesolutions.com/dscp-tos-cos-precedence-conversion-chart,
# https://en.wikipedia.org/wiki/Differentiated_services
UPLOAD_TOS = 0x20  # CS1, low priority background traffic
SSH_TOS = 0x90  # AF42, DSCP of 36/HDD_LINUX_AC_VI with the minimum delay flag

NetworkType = log.DeviceState.NetworkType

UploadFileDict = dict[str, str | int | float | bool]
UploadItemDict = dict[str, str | bool | int | float | dict[str, str]]

UploadFilesToUrlResponse = dict[str, int | list[UploadItemDict] | list[str]]


class UploadTOSAdapter(HTTPAdapter):
  def init_poolmanager(self, connections, maxsize, block=DEFAULT_POOLBLOCK, **pool_kwargs):
    pool_kwargs["socket_options"] = [(socket.IPPROTO_IP, socket.IP_TOS, UPLOAD_TOS)]
    super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


UPLOAD_SESS = requests.Session()
UPLOAD_SESS.mount("http://", UploadTOSAdapter())
UPLOAD_SESS.mount("https://", UploadTOSAdapter())


@dataclass
class UploadFile:
  fn: str
  url: str
  headers: dict[str, str]
  allow_cellular: bool
  priority: int = DEFAULT_UPLOAD_PRIORITY

  @classmethod
  def from_dict(cls, d: dict) -> UploadFile:
    return cls(d.get("fn", ""), d.get("url", ""), d.get("headers", {}), d.get("allow_cellular", False), d.get("priority", DEFAULT_UPLOAD_PRIORITY))


@dataclass
@total_ordering
class UploadItem:
  path: str
  url: str
  headers: dict[str, str]
  created_at: int
  id: str | None
  retry_count: int = 0
  current: bool = False
  progress: float = 0
  allow_cellular: bool = False
  priority: int = DEFAULT_UPLOAD_PRIORITY

  @classmethod
  def from_dict(cls, d: dict) -> UploadItem:
    return cls(d["path"], d["url"], d["headers"], d["created_at"], d["id"], d["retry_count"], d["current"],
               d["progress"], d["allow_cellular"], d["priority"])

  def __lt__(self, other):
    if not isinstance(other, UploadItem):
      return NotImplemented
    return self.priority < other.priority

  def __eq__(self, other):
    if not isinstance(other, UploadItem):
      return NotImplemented
    return self.priority == other.priority


@dataclass
class VideoClipJob:
  id: str
  request_id: str
  route: str
  camera: str
  source_start_time: float
  source_end_time: float
  bitrate: int
  speedup: int
  filename: str
  status: str
  progress: float
  fn: str | None = None
  size: int | None = None
  created_at: float = 0
  error: str | None = None


dispatcher["echo"] = lambda s: s
recv_queue: Queue[str] = queue.Queue()
send_queue: Queue[tuple[int, int, str]] = queue.PriorityQueue()
upload_queue: Queue[UploadItem] = queue.PriorityQueue()
log_recv_queue: Queue[str] = queue.Queue()
cancelled_uploads: set[str] = set()

cur_upload_items: dict[int, UploadItem | None] = {}
video_clip_lock = threading.Lock()
video_clip_jobs: dict[str, VideoClipJob] = {}
video_clip_requests: dict[str, list[str]] = {}
video_clip_queue: Queue[str] = queue.Queue()
video_clip_worker_running = False
cancelled_video_clips: set[str] = set()

send_seq = itertools.count()
def send_queue_push(data: str, priority: int) -> None:
  assert priority is not None, "send queue priority must be specified"
  send_queue.put_nowait((priority, next(send_seq), data)) # tie-break with a monotonic counter


def strip_zst_extension(fn: str) -> str:
  if fn.endswith('.zst'):
    return fn[:-4]
  return fn


class AbortTransferException(Exception):
  pass


class UploadQueueCache:

  @staticmethod
  def initialize(upload_queue: Queue[UploadItem]) -> None:
    try:
      upload_queue_json = Params().get("AthenadUploadQueue")
      if upload_queue_json is not None:
        for item in upload_queue_json:
          upload_queue.put(UploadItem.from_dict(item))
    except Exception:
      cloudlog.exception("athena.UploadQueueCache.initialize.exception")

  @staticmethod
  def cache(upload_queue: Queue[UploadItem]) -> None:
    try:
      queue: list[UploadItem | None] = list(upload_queue.queue)
      items = [asdict(i) for i in queue if i is not None and (i.id not in cancelled_uploads)]
      Params().put("AthenadUploadQueue", items, block=True)
    except Exception:
      cloudlog.exception("athena.UploadQueueCache.cache.exception")


def handle_long_poll(ws: WebSocket, exit_event: threading.Event | None) -> None:
  end_event = threading.Event()
  _resume_video_clip_jobs()

  threads = [
    threading.Thread(target=ws_manage, args=(ws, end_event), name='ws_manage'),
    threading.Thread(target=ws_recv, args=(ws, end_event), name='ws_recv'),
    threading.Thread(target=ws_send, args=(ws, end_event), name='ws_send'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler2'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler3'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler4'),
    threading.Thread(target=log_handler, args=(end_event,), name='log_handler'),
  ] + [
    threading.Thread(target=jsonrpc_handler, args=(end_event,), name=f'worker_{x}')
    for x in range(HANDLER_THREADS)
  ]

  for thread in threads:
    thread.start()
  try:
    while not end_event.wait(0.1):
      if exit_event is not None and exit_event.is_set():
        end_event.set()
  except (KeyboardInterrupt, SystemExit):
    end_event.set()
    raise
  finally:
    for thread in threads:
      cloudlog.debug(f"athena.joining {thread.name}")
      thread.join()


def jsonrpc_handler(end_event: threading.Event) -> None:
  dispatcher["startLocalProxy"] = partial(startLocalProxy, end_event)
  while not end_event.is_set():
    try:
      data = recv_queue.get(timeout=1)
      msg = loads(data)
      if is_call(msg):
        cloudlog.event("athena.jsonrpc_handler.call_method", data=data)
        send_queue_push(handle(msg, dispatcher), SEND_PRIORITY_HIGH)
      elif is_response(msg):
        log_recv_queue.put_nowait(data)
      else:
        raise Exception("not a valid request or response")
    except queue.Empty:
      pass
    except Exception as e:
      cloudlog.exception("athena jsonrpc handler failed")
      send_queue_push(json.dumps({"error": str(e)}), SEND_PRIORITY_HIGH)


def retry_upload(tid: int, end_event: threading.Event, increase_count: bool = True) -> None:
  item = cur_upload_items[tid]
  if item is not None and item.retry_count < MAX_RETRY_COUNT:
    new_retry_count = item.retry_count + 1 if increase_count else item.retry_count

    item = replace(
      item,
      retry_count=new_retry_count,
      progress=0,
      current=False
    )
    upload_queue.put_nowait(item)
    UploadQueueCache.cache(upload_queue)

    cur_upload_items[tid] = None

    for _ in range(RETRY_DELAY):
      time.sleep(1)
      if end_event.is_set():
        break


def cb(sm, item, tid, end_event: threading.Event, sz: int, cur: int) -> None:
  # Abort transfer if connection changed to metered after starting upload
  # or if athenad is shutting down to re-connect the websocket
  if not item.allow_cellular:
    if (time.monotonic() - sm.recv_time['deviceState']) > DEVICE_STATE_UPDATE_INTERVAL:
      sm.update(0)
      if sm['deviceState'].networkMetered:
        raise AbortTransferException

  if end_event.is_set():
    raise AbortTransferException

  cur_upload_items[tid] = replace(item, progress=cur / sz if sz else 1)


def upload_handler(end_event: threading.Event) -> None:
  sm = messaging.SubMaster(['deviceState'])
  tid = threading.get_ident()

  while not end_event.is_set():
    cur_upload_items[tid] = None

    try:
      cur_upload_items[tid] = item = replace(upload_queue.get(timeout=1), current=True)

      if item.id in cancelled_uploads:
        cancelled_uploads.remove(item.id)
        continue

      # Remove item if too old
      age = datetime.now() - datetime.fromtimestamp(item.created_at / 1000)
      if age.total_seconds() > MAX_AGE:
        cloudlog.event("athena.upload_handler.expired", item=item, error=True)
        continue

      # Check if uploading over metered connection is allowed
      sm.update(0)
      metered = sm['deviceState'].networkMetered
      network_type = sm['deviceState'].networkType.raw
      if metered and (not item.allow_cellular):
        retry_upload(tid, end_event, False)
        continue

      try:
        fn = item.path
        try:
          sz = os.path.getsize(fn)
        except OSError:
          sz = -1

        cloudlog.event("athena.upload_handler.upload_start", fn=fn, sz=sz, network_type=network_type, metered=metered, retry_count=item.retry_count)

        with _do_upload(item, partial(cb, sm, item, tid, end_event)) as response:
          if response.status_code not in (200, 201, 401, 403, 412):
            cloudlog.event("athena.upload_handler.retry", status_code=response.status_code, fn=fn, sz=sz, network_type=network_type, metered=metered)
            retry_upload(tid, end_event)
          else:
            cloudlog.event("athena.upload_handler.success", fn=fn, sz=sz, network_type=network_type, metered=metered)

        UploadQueueCache.cache(upload_queue)
      except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError):
        cloudlog.event("athena.upload_handler.timeout", fn=fn, sz=sz, network_type=network_type, metered=metered)
        retry_upload(tid, end_event)
      except AbortTransferException:
        cloudlog.event("athena.upload_handler.abort", fn=fn, sz=sz, network_type=network_type, metered=metered)
        retry_upload(tid, end_event, False)

    except queue.Empty:
      pass
    except Exception:
      cloudlog.exception("athena.upload_handler.exception")


def _do_upload(upload_item: UploadItem, callback: Callable | None = None) -> requests.Response:
  path = upload_item.path
  compress = False

  # If file does not exist, but does exist without the .zst extension we will compress on the fly
  if not os.path.exists(path) and os.path.exists(strip_zst_extension(path)):
    path = strip_zst_extension(path)
    compress = True

  stream = None
  try:
    stream, content_length = get_upload_stream(path, compress)
    response = UPLOAD_SESS.put(upload_item.url,
                               data=CallbackReader(stream, callback, content_length) if callback else stream,
                               headers={**upload_item.headers, 'Content-Length': str(content_length)},
                               timeout=30)
    return response
  finally:
    if stream:
      stream.close()


# security: user should be able to request any message from their car
@dispatcher.add_method
def getMessage(service: str, timeout: int = 1000) -> dict:
  if service is None or service not in SERVICE_LIST:
    raise Exception("invalid service")

  socket = messaging.sub_sock(service, timeout=timeout)
  try:
    ret = messaging.recv_one(socket)

    if ret is None:
      raise TimeoutError

    # this is because capnp._DynamicStructReader doesn't have typing information
    return cast(dict, ret.to_dict())
  finally:
    del socket


@dispatcher.add_method
def getVersion() -> dict[str, str]:
  build_metadata = get_build_metadata()
  return {
    "version": build_metadata.openpilot.version,
    "remote": build_metadata.openpilot.git_normalized_origin,
    "branch": build_metadata.channel,
    "commit": build_metadata.openpilot.git_commit,
  }


def scan_dir(path: str, prefix: str) -> list[str]:
  files = []
  # only walk directories that match the prefix
  # (glob and friends traverse entire dir tree)
  with os.scandir(path) as i:
    for e in i:
      rel_path = os.path.relpath(e.path, Paths.log_root())
      if e.is_dir(follow_symlinks=False):
        # add trailing slash
        rel_path = os.path.join(rel_path, '')
        # if prefix is a partial dir name, current dir will start with prefix
        # if prefix is a partial file name, prefix with start with dir name
        if rel_path.startswith(prefix) or prefix.startswith(rel_path):
          files.extend(scan_dir(e.path, prefix))
      else:
        if rel_path.startswith(prefix):
          files.append(rel_path)
  return files

@dispatcher.add_method
def listDataDirectory(prefix='') -> list[str]:
  return scan_dir(Paths.log_root(), prefix)


def _video_clip_route_name(route: str) -> str:
  parts = re.split(r"[|/]", route)
  if len(parts) == 1:
    route_name = parts[0]
  elif len(parts) == 2 and VIDEO_CLIP_DONGLE_ID_RE.fullmatch(parts[0]):
    route_name = parts[1]
  else:
    raise ValueError(f"invalid route: {route}")

  if not VIDEO_CLIP_ROUTE_RE.fullmatch(route_name):
    raise ValueError(f"invalid route: {route}")
  return route_name


def _video_clip_inputs(route: str, camera: str, start_time: float, end_time: float) -> tuple[list[str], float, float]:
  if camera not in VIDEO_CLIP_CAMERAS:
    raise ValueError(f"invalid camera: {camera}")
  _validate_video_clip_range(start_time, end_time)

  route_name = _video_clip_route_name(route)
  first_segment = math.floor(start_time / 60)
  last_segment = math.ceil(end_time / 60) - 1
  filename = VIDEO_CLIP_CAMERAS[camera]
  inputs = [
    os.path.join(Paths.log_root(), f"{route_name}--{segment}", filename)
    for segment in range(first_segment, last_segment + 1)
  ]
  missing = [os.path.relpath(path, Paths.log_root()) for path in inputs if not os.path.isfile(path)]
  if missing:
    raise FileNotFoundError(f"missing camera file(s): {', '.join(missing)}")

  return inputs, start_time - first_segment * 60, end_time - start_time


def _validate_video_clip_range(start_time: float, end_time: float) -> None:
  if isinstance(start_time, bool) or not isinstance(start_time, (int, float)) or not math.isfinite(start_time) or start_time < 0:
    raise ValueError("start_time must be a finite, non-negative number")
  if isinstance(end_time, bool) or not isinstance(end_time, (int, float)) or not math.isfinite(end_time) or end_time <= start_time:
    raise ValueError("end_time must be a finite number greater than start_time")


class VideoClipCancelled(Exception):
  pass


def _encode_video_clip(
  inputs: list[str],
  output_path: str,
  start_time: float,
  duration: float,
  bitrate: int,
  speedup: int,
  metadata: str,
  progress_callback: Callable[[float], None],
  cancel_callback: Callable[[], bool],
) -> None:
  with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as manifest:
    manifest.write("ffconcat version 1.0\n")
    for path in inputs:
      escaped_path = path.replace("'", "'\\''")
      manifest.write(f"file '{escaped_path}'\n")
      manifest.write(f"option framerate {VIDEO_CLIP_FPS}\n")
      manifest.write("duration 60\n")
    manifest_path = manifest.name

  try:
    command = [
      "ffmpeg",
      "-hide_banner",
      "-loglevel", "error",
      "-nostdin",
      "-y",
      "-f", "concat",
      "-safe", "0",
      "-c:v", "hevc",
      "-itsscale", str(1 / speedup),
      "-i", manifest_path,
      "-ss", str(start_time / speedup),
      "-t", str(duration / speedup),
      "-map", "0:v:0",
      "-an",
      "-r", str(VIDEO_CLIP_FPS),
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-b:v", f"{bitrate}M",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart+use_metadata_tags",
      "-metadata", f"com.comma.clip.settings={metadata}",
      "-progress", "pipe:1",
      "-nostats",
      output_path,
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
      assert process.stdout is not None
      output_tail: list[str] = []
      for raw_line in process.stdout:
        if cancel_callback():
          raise VideoClipCancelled
        line = raw_line.strip()
        if line.startswith("out_time_us="):
          try:
            encoded_time = int(line.split("=", 1)[1]) / 1e6
            progress_callback(min(encoded_time * speedup / duration, 0.99))
          except ValueError:
            pass
        elif line:
          output_tail.append(line)
          output_tail = output_tail[-20:]

      if cancel_callback():
        raise VideoClipCancelled
      returncode = process.wait()
      if returncode != 0:
        detail = "\n".join(output_tail)
        raise RuntimeError(f"ffmpeg exited with code {returncode}" + (f":\n{detail}" if detail else ""))
    finally:
      if process.poll() is None:
        process.terminate()
        try:
          process.wait(timeout=5)
        except subprocess.TimeoutExpired:
          process.kill()
          process.wait()
  finally:
    os.unlink(manifest_path)


def _video_clip_cache_path() -> str:
  return os.path.join(Paths.log_root(), VIDEO_CLIP_CACHE_DIR)


def _video_clip_manifest_path() -> str:
  return os.path.join(_video_clip_cache_path(), "manifest.json")


def _write_video_clip_manifest() -> None:
  cache_path = _video_clip_cache_path()
  os.makedirs(cache_path, exist_ok=True)
  fd, temporary_path = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=cache_path)
  try:
    with os.fdopen(fd, "w") as f:
      json.dump({"version": VIDEO_CLIP_MANIFEST_VERSION, "clips": [asdict(j) for j in video_clip_jobs.values()]}, f)
      f.flush()
      os.fsync(f.fileno())
    os.replace(temporary_path, _video_clip_manifest_path())
  finally:
    try:
      os.unlink(temporary_path)
    except FileNotFoundError:
      pass


def _load_video_clip_manifest() -> None:
  if video_clip_jobs or not os.path.isfile(_video_clip_manifest_path()):
    return
  try:
    for filename in os.listdir(_video_clip_cache_path()):
      if filename.startswith((".clip-", ".manifest-")):
        try:
          os.unlink(os.path.join(_video_clip_cache_path(), filename))
        except FileNotFoundError:
          pass
    with open(_video_clip_manifest_path()) as f:
      manifest = json.load(f)
    if manifest.get("version") != VIDEO_CLIP_MANIFEST_VERSION:
      return
    resumed = False
    for item in manifest.get("clips", []):
      job = VideoClipJob(**item)
      if job.status in ("queued", "encoding"):
        job = replace(job, status="queued", progress=0.0, error=None)
        video_clip_queue.put_nowait(job.id)
        resumed = True
      elif job.status == "ready" and (job.fn is None or not os.path.isfile(os.path.join(Paths.log_root(), job.fn))):
        continue
      video_clip_jobs[job.id] = job
      video_clip_requests.setdefault(job.request_id, []).append(job.id)
    if resumed:
      _write_video_clip_manifest()
  except Exception:
    cloudlog.exception("athena.video_clip.load_manifest_failed")


def _set_video_clip_job(job_id: str, **changes) -> None:
  with video_clip_lock:
    if job_id in video_clip_jobs:
      old_job = video_clip_jobs[job_id]
      new_job = replace(old_job, **changes)
      video_clip_jobs[job_id] = new_job
      if (old_job.status != new_job.status or old_job.fn != new_job.fn or old_job.error != new_job.error
          or new_job.progress - old_job.progress >= 0.01):
        _write_video_clip_manifest()


def _video_clip_is_cancelled(job_id: str) -> bool:
  with video_clip_lock:
    return job_id in cancelled_video_clips


def _video_clip_worker() -> None:
  global video_clip_worker_running
  while True:
    try:
      job_id = video_clip_queue.get_nowait()
    except queue.Empty:
      with video_clip_lock:
        if video_clip_queue.empty():
          video_clip_worker_running = False
          return
      continue

    temporary_path = ""
    try:
      with video_clip_lock:
        job = video_clip_jobs.get(job_id)
        if job is None:
          continue
        video_clip_jobs[job_id] = replace(job, status="encoding")
        _write_video_clip_manifest()

      inputs, relative_start, duration = _video_clip_inputs(
        job.route,
        job.camera,
        job.source_start_time,
        job.source_end_time,
      )
      cache_path = _video_clip_cache_path()
      os.makedirs(cache_path, exist_ok=True)
      fd, temporary_path = tempfile.mkstemp(prefix=".clip-", suffix=".mp4", dir=cache_path)
      os.close(fd)
      output_path = os.path.join(cache_path, f"{job.id}.mp4")
      metadata = json.dumps(
        {
          "version": 1,
          "clip_id": job.id,
          "route": job.route,
          "camera": job.camera,
          "source_start_time": job.source_start_time,
          "source_end_time": job.source_end_time,
          "bitrate": job.bitrate,
          "speedup": job.speedup,
        },
        separators=(",", ":"),
      )
      _encode_video_clip(
        inputs,
        temporary_path,
        relative_start,
        duration,
        job.bitrate,
        job.speedup,
        metadata,
        lambda progress, clip_id=job.id: _set_video_clip_job(clip_id, progress=progress),
        lambda clip_id=job.id: _video_clip_is_cancelled(clip_id),
      )
      with video_clip_lock:
        if job.id in cancelled_video_clips or job.id not in video_clip_jobs:
          raise VideoClipCancelled
        os.replace(temporary_path, output_path)
        fn = os.path.relpath(output_path, Paths.log_root())
        video_clip_jobs[job.id] = replace(video_clip_jobs[job.id], status="ready", progress=1.0, fn=fn, size=os.path.getsize(output_path))
        _write_video_clip_manifest()
    except VideoClipCancelled:
      pass
    except Exception as e:
      cloudlog.exception("athena.video_clip.failed")
      _set_video_clip_job(job_id, status="failed", error=str(e))
    finally:
      if temporary_path:
        try:
          os.unlink(temporary_path)
        except FileNotFoundError:
          pass
      with video_clip_lock:
        cancelled_video_clips.discard(job_id)
      video_clip_queue.task_done()


def _start_video_clip_worker() -> None:
  global video_clip_worker_running
  if not PC and not Params().get_bool("IsOffroad"):
    return
  with video_clip_lock:
    if video_clip_worker_running:
      return
    video_clip_worker_running = True
  threading.Thread(target=_video_clip_worker, name="video_clip", daemon=True).start()


def _resume_video_clip_jobs() -> None:
  with video_clip_lock:
    _load_video_clip_manifest()
    has_queued_jobs = not video_clip_queue.empty()
  if has_queued_jobs:
    _start_video_clip_worker()


def _video_clip_available_ranges(route: str, camera: str) -> list[list[int]]:
  route_name = _video_clip_route_name(route)
  filename = VIDEO_CLIP_CAMERAS[camera]
  segments = []
  try:
    for entry in os.scandir(Paths.log_root()):
      prefix = f"{route_name}--"
      if entry.is_dir() and entry.name.startswith(prefix) and os.path.isfile(os.path.join(entry.path, filename)):
        try:
          segments.append(int(entry.name[len(prefix):]))
        except ValueError:
          pass
  except FileNotFoundError:
    pass

  ranges: list[list[int]] = []
  for segment in sorted(set(segments)):
    if ranges and ranges[-1][1] == segment * 60:
      ranges[-1][1] += 60
    else:
      ranges.append([segment * 60, (segment + 1) * 60])
  return ranges


def _validate_video_clip_filename(filename: str, clip_id: str) -> str:
  if not filename:
    return f"clip-{clip_id}.mp4"
  if filename != os.path.basename(filename) or filename in (".", "..") or "\x00" in filename:
    raise ValueError("filename must be a plain file name")
  return filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"


@dispatcher.add_method
def createClips(request_id: str, route: str, source_start_time: float, source_end_time: float, clips: list[dict]) -> dict:
  if not PC and not Params().get_bool("IsOffroad"):
    raise RuntimeError("video clips can only be created while offroad")
  if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
    raise ValueError("request_id must be a non-empty string")
  if not clips:
    raise ValueError("clips must contain at least one clip")
  existing_response = None
  has_queued_jobs = False
  with video_clip_lock:
    _load_video_clip_manifest()
    if request_id in video_clip_requests:
      existing_response = {
        "request_id": request_id,
        "clips": [asdict(video_clip_jobs[i]) for i in video_clip_requests[request_id]],
      }
      has_queued_jobs = not video_clip_queue.empty()
  if existing_response is not None:
    if has_queued_jobs:
      _start_video_clip_worker()
    return existing_response

  route_name = _video_clip_route_name(route)
  _validate_video_clip_range(source_start_time, source_end_time)
  if source_end_time - source_start_time > VIDEO_CLIP_MAX_DURATION:
    raise ValueError(f"clip duration cannot exceed {VIDEO_CLIP_MAX_DURATION} seconds")

  pending: list[VideoClipJob] = []
  for settings in clips:
    camera = settings.get("camera")
    bitrate = settings.get("bitrate")
    speedup = settings.get("speedup", 1)
    if not isinstance(camera, str) or camera not in VIDEO_CLIP_CAMERAS:
      raise ValueError(f"invalid camera: {camera}")
    if isinstance(bitrate, bool) or not isinstance(bitrate, int) or bitrate not in VIDEO_CLIP_BITRATES:
      raise ValueError(f"bitrate must be one of {VIDEO_CLIP_BITRATES}")
    if isinstance(speedup, bool) or not isinstance(speedup, int) or speedup not in VIDEO_CLIP_SPEEDUPS:
      raise ValueError(f"speedup must be one of {VIDEO_CLIP_SPEEDUPS}")
    _video_clip_inputs(route_name, camera, source_start_time, source_end_time)
    clip_id = uuid.uuid4().hex
    requested_filename = settings.get("filename", "")
    if not isinstance(requested_filename, str):
      raise ValueError("filename must be a string")
    filename = _validate_video_clip_filename(requested_filename, clip_id)
    pending.append(VideoClipJob(
      clip_id, request_id, route_name, camera, source_start_time, source_end_time, bitrate, speedup, filename,
      "queued", 0.0, created_at=datetime.now().timestamp(),
    ))

  with video_clip_lock:
    _load_video_clip_manifest()
    if request_id in video_clip_requests:
      return {"request_id": request_id, "clips": [asdict(video_clip_jobs[i]) for i in video_clip_requests[request_id]]}
    video_clip_requests[request_id] = [job.id for job in pending]
    video_clip_jobs.update((job.id, job) for job in pending)
    _write_video_clip_manifest()
    for job in pending:
      video_clip_queue.put_nowait(job.id)

  _start_video_clip_worker()
  return {"request_id": request_id, "clips": [asdict(job) for job in pending]}


@dispatcher.add_method
def getClipsState(routes: list[str]) -> dict:
  if not isinstance(routes, list) or len(routes) > 100:
    raise ValueError("routes must be a list of at most 100 routes")
  route_names = [_video_clip_route_name(route) for route in routes]
  with video_clip_lock:
    _load_video_clip_manifest()
    jobs = [asdict(job) for job in video_clip_jobs.values()]
    has_queued_jobs = not video_clip_queue.empty()
  if has_queued_jobs:
    _start_video_clip_worker()
  route_state = {
    route: {
      "cameras": {camera: {"available_ranges": _video_clip_available_ranges(route, camera)} for camera in VIDEO_CLIP_CAMERAS},
    }
    for route in route_names
  }
  return {
    "version": VIDEO_CLIP_MANIFEST_VERSION,
    "capabilities": {
      "cameras": list(VIDEO_CLIP_CAMERAS), "bitrates": list(VIDEO_CLIP_BITRATES),
      "speedups": list(VIDEO_CLIP_SPEEDUPS), "max_duration": VIDEO_CLIP_MAX_DURATION,
    },
    "clips": sorted(jobs, key=lambda job: job["created_at"], reverse=True),
    "routes": route_state,
  }


@dispatcher.add_method
def deleteClips(clip_ids: list[str]) -> dict:
  if not isinstance(clip_ids, list) or len(clip_ids) > 100 or not all(isinstance(clip_id, str) for clip_id in clip_ids):
    raise ValueError("clip_ids must be a list of at most 100 strings")

  deleted = []
  failed = []
  with video_clip_lock:
    _load_video_clip_manifest()
    for clip_id in dict.fromkeys(clip_ids):
      job = video_clip_jobs.get(clip_id)
      if job is None:
        failed.append(clip_id)
        continue

      if job.fn is not None:
        try:
          os.unlink(os.path.join(_video_clip_cache_path(), f"{job.id}.mp4"))
        except FileNotFoundError:
          pass
        except OSError:
          cloudlog.exception("athena.video_clip.delete_failed")
          failed.append(clip_id)
          continue
      if job.status in ("queued", "encoding"):
        cancelled_video_clips.add(clip_id)

      del video_clip_jobs[clip_id]
      request_clips = video_clip_requests[job.request_id]
      request_clips.remove(clip_id)
      if not request_clips:
        del video_clip_requests[job.request_id]
      deleted.append(clip_id)

    if deleted:
      _write_video_clip_manifest()
  return {"deleted": deleted, "failed": failed}


@dispatcher.add_method
def uploadFileToUrl(fn: str, url: str, headers: dict[str, str]) -> UploadFilesToUrlResponse:
  # this is because mypy doesn't understand that the decorator doesn't change the return type
  response: UploadFilesToUrlResponse = uploadFilesToUrls([{
    "fn": fn,
    "url": url,
    "headers": headers,
  }])
  return response


@dispatcher.add_method
def uploadFilesToUrls(files_data: list[UploadFileDict]) -> UploadFilesToUrlResponse:
  files = map(UploadFile.from_dict, files_data)

  items: list[UploadItemDict] = []
  failed: list[str] = []
  for file in files:
    if len(file.fn) == 0 or file.fn[0] == '/' or '..' in file.fn or len(file.url) == 0:
      failed.append(file.fn)
      continue

    path = os.path.join(Paths.log_root(), file.fn)
    if not os.path.exists(path) and not os.path.exists(strip_zst_extension(path)):
      failed.append(file.fn)
      continue

    # Skip item if already in queue
    url = file.url.split('?')[0]
    if any(url == item['url'].split('?')[0] for item in listUploadQueue()):
      continue

    item = UploadItem(
      path=path,
      url=file.url,
      headers=file.headers,
      created_at=int(time.time() * 1000),  # noqa: TID251
      id=None,
      allow_cellular=file.allow_cellular,
      priority=file.priority,
    )
    upload_id = hashlib.sha1(str(item).encode()).hexdigest()
    item = replace(item, id=upload_id)
    upload_queue.put_nowait(item)
    items.append(asdict(item))

  UploadQueueCache.cache(upload_queue)

  resp: UploadFilesToUrlResponse = {"enqueued": len(items), "items": items}
  if failed:
    cloudlog.event("athena.uploadFilesToUrls.failed", failed=failed, error=True)
    resp["failed"] = failed

  return resp


@dispatcher.add_method
def listUploadQueue() -> list[UploadItemDict]:
  items = list(upload_queue.queue) + list(cur_upload_items.values())
  return [asdict(i) for i in items if (i is not None) and (i.id not in cancelled_uploads)]


@dispatcher.add_method
def cancelUpload(upload_id: str | list[str]) -> dict[str, int | str]:
  if not isinstance(upload_id, list):
    upload_id = [upload_id]

  uploading_ids = {item.id for item in list(upload_queue.queue)}
  cancelled_ids = uploading_ids.intersection(upload_id)
  if len(cancelled_ids) == 0:
    return {"success": 0, "error": "not found"}

  cancelled_uploads.update(cancelled_ids)
  return {"success": 1}

@dispatcher.add_method
def setRouteViewed(route: str) -> dict[str, int | str]:
  # maintain a list of the last 10 routes viewed in connect
  params = Params()

  r = params.get("AthenadRecentlyViewedRoutes")
  routes = [] if r is None else r.split(",")
  routes.append(route)

  # remove duplicates
  routes = list(dict.fromkeys(routes))

  params.put("AthenadRecentlyViewedRoutes", ",".join(routes[-10:]), block=True)
  return {"success": 1}


def startLocalProxy(global_end_event: threading.Event, remote_ws_uri: str, local_port: int) -> dict[str, int]:
  try:
    # migration, can be removed once 0.9.8 is out for a while
    if local_port == 8022:
      local_port = 22

    if local_port not in LOCAL_PORT_WHITELIST:
      raise Exception("Requested local port not whitelisted")

    cloudlog.debug("athena.startLocalProxy.starting")

    dongle_id = Params().get("DongleId")
    identity_token = Api(dongle_id).get_token()
    ws = create_connection(remote_ws_uri,
                           cookie="jwt=" + identity_token,
                           enable_multithread=True)

    # Set TOS to keep connection responsive while under load.
    ws.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, SSH_TOS)

    ssock, csock = socket.socketpair()
    local_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_sock.connect(('127.0.0.1', local_port))
    local_sock.setblocking(False)

    proxy_end_event = threading.Event()
    threads = [
      threading.Thread(target=ws_proxy_recv, args=(ws, local_sock, ssock, proxy_end_event, global_end_event)),
      threading.Thread(target=ws_proxy_send, args=(ws, local_sock, csock, proxy_end_event))
    ]
    for thread in threads:
      thread.start()

    cloudlog.debug("athena.startLocalProxy.started")
    return {"success": 1}
  except Exception as e:
    cloudlog.exception("athenad.startLocalProxy.exception")
    raise e


@dispatcher.add_method
def getPublicKey() -> str | None:
  _, _, public_key = get_key_pair()
  return public_key


@dispatcher.add_method
def getSshAuthorizedKeys() -> str:
  return cast(str, Params().get("GithubSshKeys") or "")


@dispatcher.add_method
def getGithubUsername() -> str:
  return cast(str, Params().get("GithubUsername") or "")


@dispatcher.add_method
def getNotCar() -> bool:
  cp_bytes = Params().get("CarParamsPersistent")
  if cp_bytes is not None:
    with car.CarParams.from_bytes(cp_bytes) as CP:
      return CP.notCar
  return False


@dispatcher.add_method
def getSimInfo():
  return HARDWARE.get_sim_info()


@dispatcher.add_method
def getNetworkType():
  return HARDWARE.get_network_type()


@dispatcher.add_method
def getNetworkMetered() -> bool:
  network_type = HARDWARE.get_network_type()
  return HARDWARE.get_network_metered(network_type)


@dispatcher.add_method
def startStream(sdp: str, enabled: bool) -> dict:
  from openpilot.system.webrtc.helpers import StreamRequestBody, post_stream_request, wait_for_webrtcd
  params = Params()
  bridge_services_in = []

  # stale car params case taken care of by webrtcd being shut off on ignition
  cp_bytes = Params().get("CarParamsPersistent")
  if cp_bytes is not None:
    with car.CarParams.from_bytes(cp_bytes) as CP:
      if CP.notCar:
        bridge_services_in.append("testJoystick")
  else:
    raise Exception("failed to get CarParamsPersistent")

  if params.get_bool("IsOffroad"):
    # manager owns camerad/stream_encoderd/webrtcd; flip the param and let it bring them up.
    # webrtcd clears IsLiveStreaming when the session ends
    params.put_bool("IsLiveStreaming", True)
    # wait for webrtcd end points to wake up
    wait_for_webrtcd()

  return post_stream_request(StreamRequestBody(sdp, ["wideRoad"], enabled, bridge_services_in, ["carState", "deviceState"]))


def get_logs_to_send_sorted() -> list[str]:
  # TODO: scan once then use inotify to detect file creation/deletion
  curr_time = int(time.time())  # noqa: TID251
  logs = []
  for log_entry in os.listdir(Paths.swaglog_root()):
    log_path = os.path.join(Paths.swaglog_root(), log_entry)
    time_sent = 0
    try:
      value = getxattr(log_path, LOG_ATTR_NAME)
      if value is not None:
        time_sent = int.from_bytes(value, sys.byteorder)
    except (ValueError, TypeError):
      pass
    # assume send failed and we lost the response if sent more than one hour ago
    if not time_sent or curr_time - time_sent > 3600:
      logs.append(log_entry)
  # excluding most recent (active) log file
  return sorted(logs)[:-1]


def log_handler(end_event: threading.Event) -> None:
  if PC:
    return

  log_files = []
  last_scan = 0.
  while not end_event.is_set():
    try:
      curr_scan = time.monotonic()
      if curr_scan - last_scan > 10:
        log_files = get_logs_to_send_sorted()
        last_scan = curr_scan

      # send one log
      curr_log = None
      if len(log_files) > 0:
        log_entry = log_files.pop() # newest log file
        cloudlog.debug(f"athena.log_handler.forward_request {log_entry}")
        try:
          curr_time = int(time.time())  # noqa: TID251
          log_path = os.path.join(Paths.swaglog_root(), log_entry)
          setxattr(log_path, LOG_ATTR_NAME, int.to_bytes(curr_time, 4, sys.byteorder))
          with open(log_path) as f:
            send_queue_push(dumps_call("forwardLogs", {"logs": f.read()}, request_id=log_entry), SEND_PRIORITY_LOW)
            curr_log = log_entry
        except OSError:
          pass  # file could be deleted by log rotation

      # wait for response up to ~100 seconds
      # always read queue at least once to process any old responses that arrive
      for _ in range(100):
        if end_event.is_set():
          break
        try:
          log_resp = json.loads(log_recv_queue.get(timeout=1))
          log_entry = log_resp.get("id")
          log_success = "result" in log_resp and log_resp["result"].get("success")
          cloudlog.debug(f"athena.log_handler.forward_response {log_entry} {log_success}")
          if log_entry and log_success:
            log_path = os.path.join(Paths.swaglog_root(), log_entry)
            try:
              setxattr(log_path, LOG_ATTR_NAME, LOG_ATTR_VALUE_MAX_UNIX_TIME)
            except OSError:
              pass  # file could be deleted by log rotation
          if curr_log == log_entry:
            break
        except queue.Empty:
          if curr_log is None:
            break

    except Exception:
      cloudlog.exception("athena.log_handler.exception")


def ws_proxy_recv(ws: WebSocket, local_sock: socket.socket, ssock: socket.socket, end_event: threading.Event, global_end_event: threading.Event) -> None:
  while not (end_event.is_set() or global_end_event.is_set()):
    try:
      sock = ws.sock
      if sock is None:
        return
      r = select.select((sock,), (), (), 30)
      if r[0]:
        data = ws.recv()
        if isinstance(data, str):
          data = data.encode("utf-8")
        local_sock.sendall(data)
    except WebSocketTimeoutException:
      pass
    except Exception:
      cloudlog.exception("athenad.ws_proxy_recv.exception")
      break

  cloudlog.debug("athena.ws_proxy_recv closing sockets")
  ssock.close()
  local_sock.close()
  ws.close()
  cloudlog.debug("athena.ws_proxy_recv done closing sockets")

  end_event.set()


def ws_proxy_send(ws: WebSocket, local_sock: socket.socket, signal_sock: socket.socket, end_event: threading.Event) -> None:
  while not end_event.is_set():
    try:
      r, _, _ = select.select((local_sock, signal_sock), (), ())
      if r:
        if r[0].fileno() == signal_sock.fileno():
          # got end signal from ws_proxy_recv
          end_event.set()
          break
        data = local_sock.recv(4096)
        if not data:
          # local_sock is dead
          end_event.set()
          break

        ws.send(data, ABNF.OPCODE_BINARY)
    except Exception:
      cloudlog.exception("athenad.ws_proxy_send.exception")
      end_event.set()

  cloudlog.debug("athena.ws_proxy_send closing sockets")
  signal_sock.close()
  cloudlog.debug("athena.ws_proxy_send done closing sockets")


def ws_recv(ws: WebSocket, end_event: threading.Event) -> None:
  last_ping = int(time.monotonic() * 1e9)
  while not end_event.is_set():
    try:
      opcode, data = ws.recv_data(control_frame=True)
      if opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
        if opcode == ABNF.OPCODE_TEXT:
          data = data.decode("utf-8")
        recv_queue.put_nowait(data)
      elif opcode == ABNF.OPCODE_PING:
        last_ping = int(time.monotonic() * 1e9)
        Params().put("LastAthenaPingTime", last_ping, block=True)
    except WebSocketTimeoutException:
      ns_since_last_ping = int(time.monotonic() * 1e9) - last_ping
      if ns_since_last_ping > RECONNECT_TIMEOUT_S * 1e9:
        cloudlog.exception("athenad.ws_recv.timeout")
        end_event.set()
    except Exception:
      cloudlog.exception("athenad.ws_recv.exception")
      end_event.set()


def ws_send(ws: WebSocket, end_event: threading.Event) -> None:
  while not end_event.is_set():
    try:
      _, _, data = send_queue.get(timeout=1)
      for i in range(0, len(data), WS_FRAME_SIZE):
        frame = data[i:i+WS_FRAME_SIZE]
        last = i + WS_FRAME_SIZE >= len(data)
        opcode = ABNF.OPCODE_TEXT if i == 0 else ABNF.OPCODE_CONT
        ws.send_frame(ABNF.create_frame(frame, opcode, last))
    except queue.Empty:
      pass
    except Exception:
      cloudlog.exception("athenad.ws_send.exception")
      end_event.set()


def ws_manage(ws: WebSocket, end_event: threading.Event) -> None:
  params = Params()
  onroad_prev = None
  sock = ws.sock

  while True:
    onroad = not params.get_bool("IsOffroad")
    if onroad != onroad_prev:
      onroad_prev = onroad

      if sock is not None:
        # While not sending data, onroad, we can expect to time out in 7 + (7 * 2) = 21s
        #                         offroad, we can expect to time out in 30 + (10 * 3) = 60s
        # FIXME: TCP_USER_TIMEOUT is effectively 2x for some reason (32s), so it's mostly unused
        if sys.platform == 'linux':
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 16000 if onroad else 0)
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 7 if onroad else 30)
        elif sys.platform == 'darwin':
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 7 if onroad else 30)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 7 if onroad else 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 2 if onroad else 3)

    if end_event.wait(5):
      break


def backoff(retries: int) -> int:
  return random.randrange(0, min(128, int(2 ** retries)))


def main(exit_event: threading.Event | None = None):
  try:
    set_core_affinity([0, 1, 2, 3])
  except Exception:
    cloudlog.exception("failed to set core affinity")

  params = Params()
  dongle_id = params.get("DongleId")
  UploadQueueCache.initialize(upload_queue)

  ws_uri = ATHENA_HOST + "/ws/v2/" + dongle_id
  api = Api(dongle_id)

  conn_start = None
  conn_retries = 0
  while exit_event is None or not exit_event.is_set():
    try:
      if conn_start is None:
        conn_start = time.monotonic()

      cloudlog.event("athenad.main.connecting_ws", ws_uri=ws_uri, retries=conn_retries)
      ws = create_connection(ws_uri,
                             cookie="jwt=" + api.get_token(),
                             enable_multithread=True,
                             timeout=30.0)
      cloudlog.event("athenad.main.connected_ws", ws_uri=ws_uri, retries=conn_retries,
                     duration=time.monotonic() - conn_start)
      conn_start = None

      conn_retries = 0
      cur_upload_items.clear()

      handle_long_poll(ws, exit_event)

      ws.close()
    except (KeyboardInterrupt, SystemExit):
      break
    except (ConnectionError, TimeoutError, WebSocketException):
      conn_retries += 1
      params.remove("LastAthenaPingTime")
    except Exception:
      cloudlog.exception("athenad.main.exception")

      conn_retries += 1
      params.remove("LastAthenaPingTime")

    time.sleep(backoff(conn_retries))


if __name__ == "__main__":
  main()
