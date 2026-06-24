#!/usr/bin/env python3

from abc import abstractmethod
import os
import socket
import time
import argparse
import asyncio
import contextlib
import json
import uuid
import logging
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Any, TYPE_CHECKING

# aiortc and its dependencies have lots of internal warnings :(
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # TODO: remove this when google-crc32c publish a python3.12 wheel

import capnp
if TYPE_CHECKING:
  from aiortc.rtcdatachannel import RTCDataChannel
import aioice.ice

from openpilot.system.webrtc.helpers import StreamRequestBody
from openpilot.system.webrtc.schema import generate_field
from openpilot.common.params import Params
from openpilot.cereal import messaging, log


# socket trick: route lookup for 8.8.8.8 (nothing is sent or actually connected to)
# return the source interfaces IP which is the default interface of the device
def _default_route_ip() -> str | None:
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 53))  # selects a route, sends nothing
    return s.getsockname()[0]
  except OSError:
    return None
  finally:
    s.close()

# aioice patch: gather ICE candidates only on the default-route interface
_get_host_addresses = aioice.ice.get_host_addresses
def _primary_host_addresses(use_ipv4: bool, use_ipv6: bool) -> list[str]:
  addresses = _get_host_addresses(use_ipv4, use_ipv6)
  primary = _default_route_ip()
  if primary not in addresses:
    return addresses
  return [primary, ]
aioice.ice.get_host_addresses = _primary_host_addresses


class AsyncTaskRunner:
  def __init__(self):
    self.is_running = False
    self.task = None
    self.logger = logging.getLogger("webrtcd")

  def start(self):
    assert self.task is None
    self.task = asyncio.create_task(self.run())

  async def stop(self):
    if self.task is None:
      return
    task = self.task
    self.task = None
    if task.done():
      return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await task

  @abstractmethod
  async def run(self):
    pass


class CerealOutgoingMessageProxy(AsyncTaskRunner):
  def __init__(self, services: list[str], enabled: bool = True):
    super().__init__()
    self.services = list(services)
    self.sm = messaging.SubMaster(self.services)
    self.channels: list[RTCDataChannel] = []
    self._enabled = enabled

  def add_channel(self, channel: 'RTCDataChannel'):
    self.channels.append(channel)

  def enable(self, enable: bool):
    self._enabled = enable

  def to_json(self, msg_content: Any):
    if isinstance(msg_content, capnp._DynamicStructReader):
      msg_dict = msg_content.to_dict()
    elif isinstance(msg_content, capnp._DynamicListReader):
      msg_dict = [self.to_json(msg) for msg in msg_content]
    elif isinstance(msg_content, bytes):
      msg_dict = msg_content.decode()
    else:
      msg_dict = msg_content

    return msg_dict

  def update(self):
    # this is blocking in async context...
    self.sm.update(0)
    for service, updated in self.sm.updated.items():
      if not updated:
        continue
      msg_dict = self.to_json(self.sm[service])
      mono_time, valid = self.sm.logMonoTime[service], self.sm.valid[service]
      outgoing_msg = {"type": service, "logMonoTime": mono_time, "valid": valid, "data": msg_dict}
      encoded_msg = json.dumps(outgoing_msg).encode()
      for channel in self.channels:
        channel.send(encoded_msg)

  async def run(self):
    from aiortc.exceptions import InvalidStateError

    while True:
      if not self._enabled:
        await asyncio.sleep(0.01)
        continue
      try:
        self.update()
      except InvalidStateError:
        self.logger.warning("Cereal outgoing proxy invalid state (connection closed)")
        break
      except Exception:
        self.logger.exception("Cereal outgoing proxy failure")
      await asyncio.sleep(0.01)


class CerealIncomingMessageProxy:
  def __init__(self, pm: messaging.PubMaster):
    self.pm = pm

  def send(self, message: bytes):
    msg_json = json.loads(message)
    msg_type, msg_data = msg_json["type"], msg_json["data"]
    size = None
    if not isinstance(msg_data, dict):
      size = len(msg_data)

    msg = messaging.new_message(msg_type, size=size)
    setattr(msg, msg_type, msg_data)
    self.pm.send(msg_type, msg)


class DynamicPubMaster(messaging.PubMaster):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.lock = asyncio.Lock()

  async def add_services_if_needed(self, services):
    async with self.lock:
      for service in services:
        if service not in self.sock:
          self.sock[service] = messaging.pub_sock(service)


class LivestreamBitrateController(AsyncTaskRunner):
  bitrates = [500_000, 1_500_000, int(os.environ.get("STREAM_BITRATE", 5_000_000))]
  label_to_bitrate = { "high": bitrates[2], "med": bitrates[1], "low": bitrates[0]}
  sample_interval = 0.2
  high_level = 0.1 # drop immediately
  med_level = 0.05 # drop after # of samples
  low_level = 0 # raise after # of samples
  down_samples = 5 # 1s
  param_name = "LivestreamEncoderBitrate"

  def __init__(self, peer_connection: Any, params: Params, enabled: bool = True):
    super().__init__()
    self.pc = peer_connection
    self.params = params

    self.level = 2
    self._publish(self.bitrates[self.level])
    self.prev_lost, self.prev_sent = None, None
    self.counter = 0
    self.up_samples = 5 # 1s
    self._auto = True
    self._enabled = enabled

  def enable(self, enable: bool):
    self._enabled = enable

  async def run(self):
    while True:
      await asyncio.sleep(self.sample_interval)
      if not self._enabled:
        continue
      if not self._auto:
        continue

      loss_rate = await self._sample()
      if loss_rate is None:
        continue
      if loss_rate >= self.med_level and self.level > 0:
        self.counter += 1
        if self.counter >= self.down_samples or loss_rate >= self.high_level:
          self.level -= 1
          self.up_samples *= 2 # exponential backoff before raising again
          self.counter = 0
          self._publish(self.bitrates[self.level])
      elif loss_rate <= self.low_level and self.level < len(self.bitrates) - 1:
        self.counter -= 1
        if -self.counter >= self.up_samples:
          self.level += 1
          self.counter = 0
          self._publish(self.bitrates[self.level])

  async def _sample(self) -> float | None:
    report = await self.pc.getStats()
    packets_lost = packets_sent = 0
    for s in report.values():
      if s.type == "remote-inbound-rtp":
        packets_lost += s.packetsLost
      elif s.type == "outbound-rtp":
        packets_sent += s.packetsSent

    if self.prev_lost is None:
      self.prev_lost, self.prev_sent = packets_lost, packets_sent
      return None
    lost_delta = max(0, packets_lost - self.prev_lost)
    sent_delta = max(0, packets_sent - self.prev_sent)
    self.prev_lost, self.prev_sent = packets_lost, packets_sent
    return lost_delta / sent_delta if sent_delta else 0.0

  def _publish(self, bitrate: float):
    self.params.put(self.param_name, bitrate)

  def set_quality(self, quality):
    if quality in self.label_to_bitrate:
      self._publish(self.label_to_bitrate[quality])
      self._auto = False
    elif quality == "auto":
      self._auto = True


class StreamSession:
  shared_pub_master = DynamicPubMaster([])

  def __init__(self, body: StreamRequestBody, debug_mode: bool = False):
    if debug_mode:
      from aiortc.mediastreams import VideoStreamTrack
    from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
    from teleoprtc.builder import WebRTCAnswerBuilder

    self.identifier = str(uuid.uuid4())
    self.params = Params()
    builder = WebRTCAnswerBuilder(body.sdp)

    self.enabled = body.enabled
    self.video_track = LiveStreamVideoStreamTrack(body.init_camera, self.enabled) if not debug_mode else VideoStreamTrack()
    builder.add_video_stream(body.init_camera, self.video_track)
    self.stream = builder.stream()

    self.incoming_bridge: CerealIncomingMessageProxy | None = None
    self.incoming_bridge_services = body.bridge_services_in
    self.outgoing_bridge: CerealOutgoingMessageProxy | None = None
    self.bitrate_controller: LivestreamBitrateController | None = None
    if len(body.bridge_services_in) > 0:
      self.incoming_bridge = CerealIncomingMessageProxy(self.shared_pub_master)
    if len(body.bridge_services_out) > 0:
      self.outgoing_bridge = CerealOutgoingMessageProxy(body.bridge_services_out, self.enabled)
    self.bitrate_controller = LivestreamBitrateController(self.stream.peer_connection, self.params, self.enabled)

    self.run_task: asyncio.Task | None = None
    self._cleanup_lock = asyncio.Lock()
    self._cleanup_done = False
    self.logger = logging.getLogger("webrtcd")
    self.logger.info(
      "New stream session (%s), init camera %s, video enabled %s, incoming services %s, outgoing services %s",
      self.identifier, body.init_camera, body.enabled, body.bridge_services_in, body.bridge_services_out,
    )

  def start(self):
    self.run_task = asyncio.create_task(self.run())

  async def stop(self):
    if self.run_task is not None and not self.run_task.done() and self.run_task is not asyncio.current_task():
      self.run_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self.run_task
    self.run_task = None
    await self.post_run_cleanup()

  async def get_answer(self):
    return await self.stream.start()

  def message_handler(self, message: bytes):
    try:
      payload = json.loads(message) if isinstance(message, (bytes, str)) else None
      if isinstance(payload, dict):
        msg_type = payload.get("type")

        match msg_type:
          case "livestreamCameraSwitch":
            self.video_track.switch_camera(payload["data"]["camera"])
          case "livestreamSettings":
            self.bitrate_controller.set_quality(payload["data"]["quality"])
          case "livestreamVideoEnable":
            enabled = payload["data"]["enabled"]
            self.enabled = enabled
            self.video_track.enable(enabled)
            self.outgoing_bridge.enable(enabled)
            self.bitrate_controller.enable(enabled)
            if not enabled:
              self.params.put("LivestreamRequestKeyframe", True)
          case "clockSync":
            pong = json.dumps({"type": "clockSync", "data": {
              "action": "pong", "browserSendTime": payload["data"]["browserSendTime"], "deviceTime": time.time() * 1000, # noqa: TID251
            }})
            self.stream.get_messaging_channel().send(pong)
          case "enableTimingSei":
            if hasattr(self.video_track, 'timing_sei_enabled'):
              self.video_track.timing_sei_enabled = bool(payload["data"]["enabled"])
          case _:
            if payload.get("type") not in self.incoming_bridge_services:
              return
            self.incoming_bridge.send(message)
    except Exception:
      self.logger.exception("Cereal incoming proxy failure")

  async def run(self):
    try:
      self.params.put("LivestreamRequestKeyframe", True)
      await asyncio.wait_for(self.stream.wait_for_connection(), timeout=15)
      if self.stream.has_messaging_channel():
        self.stream.set_message_handler(self.message_handler)
        if self.incoming_bridge is not None:
          await self.shared_pub_master.add_services_if_needed(self.incoming_bridge_services)
        if self.outgoing_bridge is not None:
          channel = self.stream.get_messaging_channel()
          self.outgoing_bridge.add_channel(channel)
          self.outgoing_bridge.start()
      self.bitrate_controller.start()

      self.logger.info("Stream session (%s) connected", self.identifier)
      await self.stream.wait_for_disconnection()
      self.logger.info("Stream session (%s) ended", self.identifier)
    except Exception:
      self.logger.exception("Stream session failure")
    finally:
      await self.post_run_cleanup()

  async def post_run_cleanup(self):
    async with self._cleanup_lock:
      if self._cleanup_done:
        return
      self._cleanup_done = True
      self.params.put("LivestreamRequestKeyframe", False)
      await self.bitrate_controller.stop()
      if self.outgoing_bridge is not None:
        await self.outgoing_bridge.stop()
      if self.video_track is not None:
        self.video_track.stop()
        self.video_track = None
      await self.stream.stop()


class ServerState:
  def __init__(self, debug: bool):
    self.streams: dict[str, StreamSession] = {}
    self.stream_lock = asyncio.Lock()
    self.debug = debug
    self.teardown: asyncio.TimerHandle | None = None


# if nothing connects for 5 seconds, tear down livestreaming processes
def schedule_teardown(state: ServerState):
  if state.teardown is not None:
    state.teardown.cancel()

  def clear():
    if not state.streams:
      Params().put_bool("IsLiveStreaming", False)

  state.teardown = asyncio.get_running_loop().call_later(5.0, clear)


def _json_response(obj: Any, status: int = 200) -> tuple[int, bytes, str]:
  return (status, json.dumps(obj).encode(), "application/json; charset=utf-8")


def _text_response(text: str, status: int = 200) -> tuple[int, bytes, str]:
  return (status, text.encode(), "text/plain; charset=utf-8")


async def handle_get_stream(state: ServerState, raw_body: bytes) -> tuple[int, bytes, str]:
  stream_dict, debug_mode = state.streams, state.debug
  body = StreamRequestBody(**json.loads(raw_body))

  async with state.stream_lock:
    # don't remove existing connection on prewarm request
    enabled = any(s.run_task and not s.run_task.done() and s.enabled for s in stream_dict.values())
    if enabled and not body.enabled:
      return _json_response({"error": "busy", "message": "someone else is connected."})

    for sid, s in list(stream_dict.items()):
      if s.run_task and not s.run_task.done():
        try:
          ch = s.stream.get_messaging_channel()
          ch.send(json.dumps({"type": "disconnect", "data": "Another device has connected, closing this session."}))
        except Exception:
          pass
      await s.stop()
      stream_dict.pop(sid, None)

    session = StreamSession(body, debug_mode)
    stream_dict[session.identifier] = session
    try:
      answer = await session.get_answer()
    except Exception:
      await session.stop()
      stream_dict.pop(session.identifier, None)
      logging.getLogger("webrtcd").exception("Failed to create stream answer")
      raise
    session.start()

    def remove_finished_session(_: asyncio.Task) -> None:
      stream_dict.pop(session.identifier, None)
      schedule_teardown(state)

    session.run_task.add_done_callback(remove_finished_session)

  return _json_response({"sdp": answer.sdp, "type": answer.type})


async def handle_get_schema(state: ServerState, services_param: str) -> tuple[int, bytes, str]:
  services = services_param.split(",")
  services = [s for s in services if s]
  assert all(s in log.Event.schema.fields and not s.endswith("DEPRECATED") for s in services), "Invalid service name"
  schema_dict = {s: generate_field(log.Event.schema.fields[s]) for s in services}
  return _json_response(schema_dict)


async def handle_post_notify(state: ServerState, payload: Any) -> tuple[int, bytes, str]:
  for session in list(state.streams.values()):
    try:
      ch = session.stream.get_messaging_channel()
      ch.send(json.dumps(payload))
    except Exception:
      continue

  return _text_response("OK")


async def on_shutdown(state: ServerState):
  for session in list(state.streams.values()):
    try:
      ch = session.stream.get_messaging_channel()
      ch.send(json.dumps({"type": "disconnect", "data": "device streaming has been stopped."}))
    except Exception:
      pass
    await session.stop()
  state.streams.clear()


class WebrtcdHandler(BaseHTTPRequestHandler):
  protocol_version = "HTTP/1.1"

  # path -> allowed methods (aiohttp registered POST /stream, POST /notify, GET /schema + its auto HEAD)
  _routes = {
    "/schema": ("GET", "HEAD"),
    "/stream": ("POST",),
    "/notify": ("POST",),
  }

  def _send(self, status: int, body: bytes, content_type: str) -> None:
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    if self.command != "HEAD":
      self.wfile.write(body)

  def _read_body(self) -> bytes:
    length = int(self.headers.get("Content-Length", 0))
    return self.rfile.read(length) if length else b""

  def _run(self, coro) -> tuple[int, bytes, str]:
    return asyncio.run_coroutine_threadsafe(coro, self.server.loop).result()

  def _dispatch_request(self) -> None:
    parsed = urlparse(self.path)
    allowed = self._routes.get(parsed.path)

    try:
      if allowed is None:
        result = _json_response({"error": "not found"}, status=404)
      elif self.command not in allowed:
        result = _json_response({"error": "method not allowed"}, status=405)
      elif parsed.path == "/schema":
        services = parse_qs(parsed.query).get("services", [""])[0]
        result = self._run(handle_get_schema(self.server.state, services))
      elif parsed.path == "/stream":
        result = self._run(handle_get_stream(self.server.state, self._read_body()))
      else:  # /notify
        try:
          payload = json.loads(self._read_body())
        except Exception:
          result = _json_response({"error": "bad request"}, status=400)
        else:
          result = self._run(handle_post_notify(self.server.state, payload))
    except Exception as e:
      logging.getLogger("webrtcd").exception("Unhandled error handling %s", self.path)
      result = _json_response({"error": "exception", "message": f"{type(e).__name__}: {e}"}, status=500)

    self._send(*result)

  def do_GET(self) -> None:
    self._dispatch_request()

  def do_HEAD(self) -> None:
    self._dispatch_request()

  def do_POST(self) -> None:
    self._dispatch_request()

  def do_PUT(self) -> None:
    self._dispatch_request()

  def do_DELETE(self) -> None:
    self._dispatch_request()

  def do_PATCH(self) -> None:
    self._dispatch_request()

  def do_OPTIONS(self) -> None:
    self._dispatch_request()

  def log_message(self, fmt, *args) -> None:
    # silence default access logging; errors are logged explicitly in _dispatch_request
    pass


class WebrtcdHTTPServer(ThreadingHTTPServer):
  daemon_threads = True
  allow_reuse_address = True
  state: ServerState
  loop: asyncio.AbstractEventLoop


async def _shutdown(server: WebrtcdHTTPServer, state: ServerState, loop: asyncio.AbstractEventLoop) -> None:
  # stop accepting new HTTP connections (blocks until serve_forever returns, so
  # run it off the loop) then tear down active stream sessions.
  await loop.run_in_executor(None, server.shutdown)
  await on_shutdown(state)
  loop.stop()


def prewarm_stream_session_imports(debug_mode: bool = False) -> None:
  if debug_mode:
    from aiortc.mediastreams import VideoStreamTrack
    assert VideoStreamTrack
  from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
  from teleoprtc.builder import WebRTCAnswerBuilder
  assert LiveStreamVideoStreamTrack
  assert WebRTCAnswerBuilder


def webrtcd_thread(host: str, port: int, debug: bool):
  logging.basicConfig(level=logging.CRITICAL, handlers=[logging.StreamHandler()])
  prewarm_start = time.monotonic()
  prewarm_stream_session_imports(debug)
  prewarm_end = time.monotonic()
  logging.getLogger("webrtcd").info(f"webrtc prewarm finished in {(prewarm_end - prewarm_start) * 1000} ms")

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  state = ServerState(debug)

  server = WebrtcdHTTPServer((host, port), WebrtcdHandler)
  server.state = state
  server.loop = loop

  # serve HTTP on a daemon thread so the asyncio loop can own the main thread
  http_thread = threading.Thread(target=server.serve_forever, name="webrtcd-http", daemon=True)
  http_thread.start()

  shutting_down = False

  def request_shutdown() -> None:
    nonlocal shutting_down
    if shutting_down:
      return
    shutting_down = True
    loop.create_task(_shutdown(server, state, loop))

  for sig in (signal.SIGINT, signal.SIGTERM):
    loop.add_signal_handler(sig, request_shutdown)

  try:
    loop.run_forever()
  finally:
    server.server_close()
    loop.close()


def main():
  parser = argparse.ArgumentParser(description="WebRTC daemon")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
  parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  args = parser.parse_args()

  webrtcd_thread(args.host, args.port, args.debug)


if __name__=="__main__":
  main()
