#!/usr/bin/env python3
from __future__ import annotations

import base64
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
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import partial, total_ordering
from queue import Queue
from typing import cast
from collections.abc import Callable, Iterable

import requests
from requests.adapters import HTTPAdapter, DEFAULT_POOLBLOCK
from websocket import (ABNF, WebSocket, WebSocketException, WebSocketTimeoutException,
                       create_connection)

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.api import Api, get_key_pair
from openpilot.common.basedir import BASEDIR
from openpilot.common.utils import CallbackReader, get_upload_stream
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.common.hardware import HARDWARE, PC
from openpilot.system.loggerd.config import CAMERA_FPS, SEGMENT_LENGTH
from openpilot.system.loggerd.xattr_cache import getxattr, setxattr
from openpilot.tools.lib.helpers import RE
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
CLIP_CHUNK_SIZE = 512 * 1024

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


dispatcher["echo"] = lambda s: s
recv_queue: Queue[str] = queue.Queue()
send_queue: Queue[tuple[int, int, str]] = queue.PriorityQueue()
upload_queue: Queue[UploadItem] = queue.PriorityQueue()
log_recv_queue: Queue[str] = queue.Queue()
cancelled_uploads: set[str] = set()

cur_upload_items: dict[int, UploadItem | None] = {}

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
    "commit_date": build_metadata.openpilot.git_commit_date.strip("'").split()[0],
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


class VideoClips:
  @dataclass
  class Clip:
    route: str
    camera: str
    source_start_time: float
    source_end_time: float
    bitrate: int
    speedup: int
    filename: str
    requested_at: float

  def __init__(self):
    self.clip_path = os.path.join(Paths.log_root(), "clips")
    self.lock = threading.Condition()
    self.clips: dict[str, VideoClips.Clip] = {}
    self.transcode_proc: tuple[str, subprocess.Popen] | None = None
    threading.Thread(target=self._worker, name="video_clip", daemon=True).start()

  def _encode(self, clip: Clip, inputs: Iterable[str], output_path: str, start_time: float, duration: float) -> None:
    inputs = list(inputs)
    metadata = json.dumps(asdict(clip), separators=(',', ':'))
    if PC:
      command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-r", str(CAMERA_FPS * clip.speedup), "-f", "concat", "-safe", "0", "-protocol_whitelist", "file,pipe", "-c:v", "hevc",
        "-i", "pipe:0", "-ss", str(start_time / clip.speedup), "-t", str(duration / clip.speedup),
        "-map", "0:v:0", "-an", "-r", str(CAMERA_FPS), "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", f"{clip.bitrate}M", "-pix_fmt", "yuv420p", "-movflags", "+faststart+use_metadata_tags",
        "-metadata", f"ai.comma.clip.settings={metadata}", output_path,
      ]
    else:
      command = [os.path.join(BASEDIR, "openpilot/system/loggerd/encoderd"), "--clip", output_path,
                 str(start_time), str(duration), "--bitrate", str(clip.bitrate * 1_000_000),
                 "--speedup", str(clip.speedup), "--metadata", metadata, "--", *inputs]

    with self.lock:
      if self.clips.get(clip.filename) is not clip:
        return
      process = subprocess.Popen(command, stdin=subprocess.PIPE if PC else subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
      self.transcode_proc = (clip.filename, process)
    try:
      if PC:
        if process.stdin is None:
          raise RuntimeError("ffmpeg stdin is unavailable")
        process.stdin.write("ffconcat version 1.0\n")
        for path in inputs:
          escaped_path = path.replace("'", "'\\''")
          process.stdin.write(f"file 'file:{escaped_path}'\noption framerate {CAMERA_FPS}\nduration {SEGMENT_LENGTH}\n")
        process.stdin.close()
      process.wait()
      if process.returncode != 0:
        raise RuntimeError(f"clip encoder exited with code {process.returncode}")
    finally:
      with suppress(OSError):
        if process.stdin is not None:
          process.stdin.close()
      if process.poll() is None:
        process.terminate()
        process.wait()
      with self.lock:
        if self.transcode_proc is not None and self.transcode_proc[0] == clip.filename:
          self.transcode_proc = None

  def _worker(self) -> None:
    while True:
      with self.lock:
        while not self.clips:
          self.lock.wait()
        clip = next(iter(self.clips.values()))
      temporary_path = ""
      try:
        with self.lock:
          if self.clips.get(clip.filename) is not clip:
            continue
        first_segment = math.floor(clip.source_start_time / SEGMENT_LENGTH)
        inputs = (
          os.path.join(Paths.log_root(), f"{clip.route}--{segment}", clip.camera)
          for segment in range(first_segment, math.ceil(clip.source_end_time / SEGMENT_LENGTH))
        )
        os.makedirs(self.clip_path, exist_ok=True)
        temporary_path = os.path.join(self.clip_path, f".{clip.filename}")
        output_path = os.path.join(self.clip_path, clip.filename)
        self._encode(clip, inputs, temporary_path, clip.source_start_time - first_segment * SEGMENT_LENGTH,
                     clip.source_end_time - clip.source_start_time)
        with self.lock:
          if self.clips.get(clip.filename) is clip:
            os.replace(temporary_path, output_path)
            del self.clips[clip.filename]
      except Exception:
        with self.lock:
          failed = self.clips.get(clip.filename) is clip
          if failed:
            del self.clips[clip.filename]
        if failed:
          cloudlog.exception("athena.video_clip.failed")
      finally:
        with suppress(OSError):
          if temporary_path:
            os.unlink(temporary_path)

  def _on_disk(self) -> dict[str, dict]:
    clips = {}
    try:
      entries = os.scandir(self.clip_path)
    except FileNotFoundError:
      return clips
    with entries:
      for entry in entries:
        if entry.name.startswith(".") or not entry.is_file():
          continue
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=ai.comma.clip.settings",
                                "-of", "json", entry.path], capture_output=True, text=True)
        if probe.returncode != 0:
          continue
        try:
          metadata = json.loads(json.loads(probe.stdout)["format"]["tags"]["ai.comma.clip.settings"])
          size = entry.stat().st_size
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
          continue
        if not isinstance(metadata, dict) or not isinstance(metadata.get("requested_at"), (int, float)):
          continue
        clips[entry.name] = {**metadata, "filename": entry.name, "status": "ready",
                             "fn": os.path.relpath(entry.path, Paths.log_root()), "size": size}
    return clips

  def _available_ranges(self, route: str) -> dict:
    cameras: dict[str, list[int]] = {}
    try:
      with os.scandir(Paths.log_root()) as entries:
        for entry in entries:
          entry_route, _, segment = entry.name.rpartition("--")
          if entry_route != route or not segment.isdigit() or not entry.is_dir():
            continue
          with os.scandir(entry.path) as files:
            for camera in files:
              if camera.is_file() and camera.name.endswith("camera.hevc"):
                cameras.setdefault(camera.name, []).append(int(segment))
    except OSError:
      return {}

    available = {}
    for camera, camera_segments in cameras.items():
      ranges: list[list[int]] = []
      for segment in sorted(camera_segments):
        if ranges and ranges[-1][1] == segment * SEGMENT_LENGTH:
          ranges[-1][1] += SEGMENT_LENGTH
        else:
          ranges.append([segment * SEGMENT_LENGTH, (segment + 1) * SEGMENT_LENGTH])
      available[camera] = {"available_ranges": ranges}
    return available

  def createClip(self, route: str, source_start_time: float, source_end_time: float, clip: dict):
    if not PC and not Params().get_bool("IsOffroad"):
      raise RuntimeError("video clips can only be created while offroad")
    route_match = re.fullmatch(RE.ROUTE_NAME, route)
    assert route_match is not None, "invalid route"
    route_name = route_match.group("log_id")
    camera = clip["camera"]
    filename = clip["filename"]
    assert camera == os.path.basename(camera) and camera.endswith("camera.hevc"), "invalid camera filename"
    assert filename == os.path.basename(filename), "invalid filename"
    with self.lock:
      self.clips[filename] = self.Clip(route_name, camera, source_start_time, source_end_time, clip["bitrate"], clip["speedup"],
                                        filename, datetime.now().timestamp())
      self.lock.notify()

  def getClipState(self, route: str | None = None) -> dict:
    route_match = re.search(RE.ROUTE_NAME, route or "")
    with self.lock:
      transcode_filename = self.transcode_proc[0] if self.transcode_proc is not None else None
      active_clips = {clip.filename: {**asdict(clip), "status": "encoding" if clip.filename == transcode_filename else "queued"}
                      for clip in self.clips.values()}
    clips = self._on_disk()
    clips.update(active_clips)
    state = {"clips": sorted(clips.values(), key=lambda clip: clip["requested_at"], reverse=True)}
    if route_match is not None:
      route_name = route_match.group("log_id")
      state.update({"route": route_name, "cameras": self._available_ranges(route_name)})
    return state

  def deleteClip(self, filename: str) -> None:
    assert filename == os.path.basename(filename), "invalid filename"
    with self.lock:
      self.clips.pop(filename, None)
      output_path = os.path.join(self.clip_path, filename)
      if self.transcode_proc is not None and self.transcode_proc[0] == filename:
        self.transcode_proc[1].terminate()
      if os.path.exists(output_path):
        os.unlink(output_path)

  def getClipChunk(self, filename: str, offset: int) -> dict:
    assert filename == os.path.basename(filename) and not filename.startswith("."), "invalid filename"
    assert isinstance(offset, int) and offset >= 0, "invalid offset"
    path = os.path.join(self.clip_path, filename)
    size = os.path.getsize(path)
    assert offset <= size, "offset past end of file"
    with open(path, "rb") as f:
      f.seek(offset)
      data = f.read(CLIP_CHUNK_SIZE)
    return {"data": base64.b64encode(data).decode(), "offset": offset, "size": size}


video_clips = VideoClips()
dispatcher.add_method(video_clips.createClip)
dispatcher.add_method(video_clips.getClipState)
dispatcher.add_method(video_clips.deleteClip)
dispatcher.add_method(video_clips.getClipChunk)


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
