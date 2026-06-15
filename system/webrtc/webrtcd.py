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
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

# aiortc and its dependencies have lots of internal warnings :(
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # TODO: remove this when google-crc32c publish a python3.12 wheel

import capnp
from aiohttp import web
if TYPE_CHECKING:
  from aiortc.rtcdatachannel import RTCDataChannel

from openpilot.system.webrtc.schema import generate_field
from openpilot.common.params import Params
from cereal import messaging, log

import aioice.ice


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
    if self.task is not None and not self.task.done():
      return
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

  def __init__(self, peer_connection: Any, params, enabled):
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

  def __init__(
    self,
    sdp: str,
    init_camera: str,
    incoming_services: list[str],
    outgoing_services: list[str],
    session_id: str,
    enabled: bool | None = None,
    debug_mode: bool = False,
  ):
    from aiortc.mediastreams import VideoStreamTrack
    from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
    from teleoprtc import WebRTCAnswerBuilder

    self.identifier = session_id or str(uuid.uuid4())
    self.params = Params()
    builder = WebRTCAnswerBuilder(sdp)

    self.enabled = enabled if enabled else True # default to enabled
    self.video_track = LiveStreamVideoStreamTrack(init_camera, self.enabled) if not debug_mode else VideoStreamTrack()
    builder.add_video_stream(init_camera, self.video_track)
    self.stream = builder.stream()

    self.incoming_bridge: CerealIncomingMessageProxy | None = None
    self.incoming_bridge_services = incoming_services
    self.outgoing_bridge: CerealOutgoingMessageProxy | None = None
    self.bitrate_controller: LivestreamBitrateController | None = None
    if len(incoming_services) > 0:
      self.incoming_bridge = CerealIncomingMessageProxy(self.shared_pub_master)
    if len(outgoing_services) > 0:
      self.outgoing_bridge = CerealOutgoingMessageProxy(outgoing_services, self.enabled)
    self.bitrate_controller = LivestreamBitrateController(self.stream.peer_connection, self.params, self.enabled)

    self.run_task: asyncio.Task | None = None
    self._cleanup_lock = asyncio.Lock()
    self._cleanup_done = False
    self.logger = logging.getLogger("webrtcd")
    self.logger.info(
      "New stream session (%s), init camera %s, video enabled %s, incoming services %s, outgoing services %s",
      self.identifier, init_camera, enabled, incoming_services, outgoing_services,
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

  async def add_ice_candidate(self, candidate_init: dict | None):
    from aiortc.sdp import candidate_from_sdp

    pc = self.stream.peer_connection
    if pc.iceConnectionState not in ("new", "checking"):
      return

    # a null/empty candidate signals end-of-candidates per the WebRTC convention
    if not candidate_init or not candidate_init.get("candidate"):
      await pc.addIceCandidate(None)
      return

    candidate = candidate_from_sdp(candidate_init["candidate"].split(":", 1)[1])
    candidate.sdpMid = candidate_init.get("sdpMid")
    candidate.sdpMLineIndex = candidate_init.get("sdpMLineIndex")
    await pc.addIceCandidate(candidate)

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
      await self.stream.wait_for_connection()
      if self.stream.has_messaging_channel():
        if self.incoming_bridge is not None:
          await self.shared_pub_master.add_services_if_needed(self.incoming_bridge_services)
          self.stream.set_message_handler(self.message_handler)
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


@dataclass
class StreamRequestBody:
  sdp: str
  initCamera: str
  session_id: str | None = None
  video_enabled: bool = True
  bridge_services_in: list[str] = field(default_factory=list)
  bridge_services_out: list[str] = field(default_factory=list)


async def get_stream(request: 'web.Request'):
  stream_dict, debug_mode = request.app['streams'], request.app['debug']
  raw_body = await request.json()
  body = StreamRequestBody(**raw_body)

  async with request.app['stream_lock']:
    # Fully disconnect any other active stream before starting the replacement.
    for sid, s in list(stream_dict.items()):
      if s.run_task and not s.run_task.done():
        try:
          ch = s.stream.get_messaging_channel()
          ch.send(json.dumps({"type": "connectionReplaced", "data": "Another device has connected, closing this session."}))
        except Exception:
          pass
      await s.stop()
      stream_dict.pop(sid, None)

    # create new stream
    session = StreamSession(body.sdp, body.initCamera, body.bridge_services_in, body.bridge_services_out, debug_mode, body.video_enabled, body.session_id)
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
    session.run_task.add_done_callback(remove_finished_session)

  return web.json_response({"sdp": answer.sdp, "type": answer.type, "session_id": session.identifier})


async def post_candidate(request: 'web.Request'):
  body = await request.json()
  session = request.app.get('streams', {}).get(body.get("session_id"))
  if session is None:
    raise Exception("stream session not found")

  await session.add_ice_candidate(body.get("candidate"))
  return web.json_response({"success": True})


async def get_schema(request: 'web.Request'):
  services = request.query["services"].split(",")
  services = [s for s in services if s]
  assert all(s in log.Event.schema.fields and not s.endswith("DEPRECATED") for s in services), "Invalid service name"
  schema_dict = {s: generate_field(log.Event.schema.fields[s]) for s in services}
  return web.json_response(schema_dict)


async def post_notify(request: 'web.Request'):
  try:
    payload = await request.json()
  except Exception as e:
    raise web.HTTPBadRequest(text="Invalid JSON") from e

  for session in list(request.app.get('streams', {}).values()):
    try:
      ch = session.stream.get_messaging_channel()
      ch.send(json.dumps(payload))
    except Exception:
      continue

  return web.Response(status=200, text="OK")


async def on_shutdown(app: 'web.Application'):
  for session in app['streams'].values():
    await session.stop()
  del app['streams']


@web.middleware
async def error_middleware(request: 'web.Request', handler):
  try:
    return await handler(request)
  except web.HTTPException:
    raise  # intentional responses (400/404/etc.) pass through untouched
  except Exception as e:
    logging.getLogger("webrtcd").exception("Unhandled error handling %s", request.path)
    return web.json_response({"error": "exception", "message": f"{type(e).__name__}: {e}"}, status=500)


def webrtcd_thread(host: str, port: int, debug: bool):
  logging.basicConfig(level=logging.CRITICAL, handlers=[logging.StreamHandler()])
  logging_level = logging.DEBUG if debug else logging.INFO
  logging.getLogger("WebRTCStream").setLevel(logging_level)
  logging.getLogger("webrtcd").setLevel(logging_level)

  app = web.Application(middlewares=[error_middleware])

  app['streams'] = dict()
  app['stream_lock'] = asyncio.Lock()
  app['debug'] = debug
  app.on_shutdown.append(on_shutdown)
  app.router.add_post("/stream", get_stream)
  app.router.add_post("/candidate", post_candidate)
  app.router.add_post("/notify", post_notify)
  app.router.add_get("/schema", get_schema)

  web.run_app(app, host=host, port=port)


def main():
  parser = argparse.ArgumentParser(description="WebRTC daemon")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
  parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  args = parser.parse_args()

  webrtcd_thread(args.host, args.port, args.debug)


if __name__=="__main__":
  main()
