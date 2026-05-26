#!/usr/bin/env python3

from abc import abstractmethod
import time
import argparse
import os
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
from cereal import messaging, log


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
  def __init__(self, sm: messaging.SubMaster):
    super().__init__()
    self.sm = sm
    self.channels: list[RTCDataChannel] = []

  def add_channel(self, channel: 'RTCDataChannel'):
    self.channels.append(channel)

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


LIVESTREAM_ENCODER_CONTROL = "livestreamEncoderControl"
STREAM_BITRATE_DEFAULT = 4_000_000
STREAM_BITRATE_MIN_DEFAULT = 500_000
BITRATE_ROUNDING = 50_000
BITRATE_MIN_PUBLISH_DELTA = 100_000


def _env_int(name: str, default: int) -> int:
  try:
    return int(os.getenv(name, default))
  except (TypeError, ValueError):
    return default


@dataclass
class LivestreamStatsSample:
  packet_loss_fraction: float | None = None
  packets_lost_delta: int | None = None
  rtt: float | None = None
  available_outgoing_bitrate: float | None = None
  send_bitrate: float | None = None


class LivestreamBitrateController:
  sample_interval = 1.0
  downshift_congested_samples = 2
  upshift_clean_samples = 8
  downshift_factor = 0.70
  upshift_factor = 1.15
  loss_fraction_threshold = 0.05
  rtt_threshold = 0.50
  available_bitrate_margin = 0.85
  send_rate_drop_ratio = 0.50
  send_rate_previous_ratio = 0.75
  _sequence = 0

  def __init__(self, peer_connection: Any, pub_master: DynamicPubMaster,
               max_bitrate: int | None = None, min_bitrate: int | None = None):
    self.peer_connection = peer_connection
    self.pub_master = pub_master
    configured_max = max_bitrate if max_bitrate is not None else _env_int("STREAM_BITRATE", STREAM_BITRATE_DEFAULT)
    configured_min = min_bitrate if min_bitrate is not None else _env_int("STREAM_BITRATE_MIN", STREAM_BITRATE_MIN_DEFAULT)
    self.max_bitrate = self._round_step(max(BITRATE_ROUNDING, configured_max))
    self.min_bitrate = min(self.max_bitrate, self._round_step(max(BITRATE_ROUNDING, configured_min)))
    self.target_bitrate = self.max_bitrate
    self.task: asyncio.Task | None = None
    self.clean_samples = 0
    self.congested_samples = 0
    self.last_published_bitrate: int | None = None
    self.prev_packets_lost: dict[str, int] = {}
    self.prev_outbound_bytes: int | None = None
    self.prev_outbound_time: float | None = None
    self.prev_send_bitrate: float | None = None
    self.logger = logging.getLogger("webrtcd")

  @staticmethod
  def _round_step(bitrate: float) -> int:
    return int(round(bitrate / BITRATE_ROUNDING) * BITRATE_ROUNDING)

  def _clamp_and_round(self, bitrate: float) -> int:
    rounded = self._round_step(bitrate)
    return max(self.min_bitrate, min(self.max_bitrate, rounded))

  @classmethod
  def _next_sequence(cls) -> int:
    cls._sequence = (cls._sequence + 1) & 0xffffffff
    return cls._sequence

  async def start(self):
    assert self.task is None
    await self.pub_master.add_services_if_needed([LIVESTREAM_ENCODER_CONTROL])
    self._set_target(self.target_bitrate, force=True)
    self.task = asyncio.create_task(self.run())

  async def stop(self, reset: bool = True):
    if self.task is not None:
      task = self.task
      self.task = None
      if not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
          await task
    if reset:
      await self.pub_master.add_services_if_needed([LIVESTREAM_ENCODER_CONTROL])
      self._set_target(self.max_bitrate, force=True)

  async def run(self):
    while True:
      await asyncio.sleep(self.sample_interval)
      try:
        sample = await self.collect_stats()
        self.update_target(sample)
      except asyncio.CancelledError:
        raise
      except Exception:
        self.logger.exception("livestream bitrate controller failure")

  async def collect_stats(self) -> LivestreamStatsSample:
    report = await self.peer_connection.getStats()
    return self.sample_from_stats(report)

  @staticmethod
  def _stat_get(stat: Any, key: str, default: Any = None) -> Any:
    if isinstance(stat, dict):
      return stat.get(key, default)
    return getattr(stat, key, default)

  def sample_from_stats(self, report: Any) -> LivestreamStatsSample:
    loss_fractions: list[float] = []
    packets_lost_delta = 0
    saw_packets_lost_delta = False
    rtts: list[float] = []
    available_bitrates: list[float] = []
    outbound_bytes = 0
    saw_outbound_bytes = False

    items = report.items() if hasattr(report, "items") else enumerate(report)
    for stat_id, stat in items:
      stat_type = self._stat_get(stat, "type")
      kind = self._stat_get(stat, "kind", self._stat_get(stat, "mediaType"))

      if stat_type in ("remote-inbound-rtp", "remote-outbound-rtp", "inbound-rtp"):
        fraction_lost = self._stat_get(stat, "fractionLost")
        if fraction_lost is not None:
          loss = max(0.0, float(fraction_lost))
          if loss > 1.0:
            loss /= 256.0
          loss_fractions.append(loss)

        packets_lost = self._stat_get(stat, "packetsLost")
        if packets_lost is not None:
          lost = max(0, int(packets_lost))
          stat_key = str(stat_id)
          prev_lost = self.prev_packets_lost.get(stat_key)
          if prev_lost is not None:
            packets_lost_delta += max(0, lost - prev_lost)
            saw_packets_lost_delta = True
          self.prev_packets_lost[stat_key] = lost

      for rtt_field in ("roundTripTime", "currentRoundTripTime"):
        rtt = self._stat_get(stat, rtt_field)
        if rtt is not None:
          rtts.append(float(rtt))

      if stat_type == "candidate-pair":
        state = self._stat_get(stat, "state")
        selected = bool(self._stat_get(stat, "selected", False) or self._stat_get(stat, "nominated", False))
        if state in (None, "succeeded") or selected:
          available = self._stat_get(stat, "availableOutgoingBitrate")
          if available is not None and float(available) > 0:
            available_bitrates.append(float(available))

      if stat_type == "outbound-rtp" and kind in (None, "video"):
        bytes_sent = self._stat_get(stat, "bytesSent")
        if bytes_sent is not None:
          outbound_bytes += max(0, int(bytes_sent))
          saw_outbound_bytes = True

    send_bitrate = None
    now = time.monotonic()
    if saw_outbound_bytes:
      if self.prev_outbound_bytes is not None and self.prev_outbound_time is not None and now > self.prev_outbound_time:
        byte_delta = max(0, outbound_bytes - self.prev_outbound_bytes)
        send_bitrate = byte_delta * 8.0 / (now - self.prev_outbound_time)
      self.prev_outbound_bytes = outbound_bytes
      self.prev_outbound_time = now

    return LivestreamStatsSample(
      packet_loss_fraction=max(loss_fractions) if loss_fractions else None,
      packets_lost_delta=packets_lost_delta if saw_packets_lost_delta else None,
      rtt=max(rtts) if rtts else None,
      available_outgoing_bitrate=min(available_bitrates) if available_bitrates else None,
      send_bitrate=send_bitrate,
    )

  def congestion_reasons(self, sample: LivestreamStatsSample) -> list[str]:
    reasons = []
    if sample.packet_loss_fraction is not None and sample.packet_loss_fraction >= self.loss_fraction_threshold:
      reasons.append("packet_loss")
    if sample.packets_lost_delta is not None and sample.packets_lost_delta > 0:
      reasons.append("packet_loss_delta")
    if sample.rtt is not None and sample.rtt >= self.rtt_threshold:
      reasons.append("rtt")
    if sample.available_outgoing_bitrate is not None and sample.available_outgoing_bitrate < self.target_bitrate * self.available_bitrate_margin:
      reasons.append("available_bitrate")
    if sample.send_bitrate is not None and self.prev_send_bitrate is not None:
      if sample.send_bitrate < self.target_bitrate * self.send_rate_drop_ratio and \
         self.prev_send_bitrate > self.target_bitrate * self.send_rate_previous_ratio:
        reasons.append("send_rate_drop")
    return reasons

  def update_target(self, sample: LivestreamStatsSample) -> bool:
    reasons = self.congestion_reasons(sample)
    changed = False

    if reasons:
      self.congested_samples += 1
      self.clean_samples = 0
      if self.congested_samples >= self.downshift_congested_samples:
        new_target = self.target_bitrate * self.downshift_factor
        if sample.available_outgoing_bitrate is not None:
          new_target = min(new_target, sample.available_outgoing_bitrate * self.available_bitrate_margin)
        changed = self._set_target(new_target)
        if changed:
          self.logger.debug("livestream bitrate downshift to %d after %s", self.target_bitrate, reasons)
          self.congested_samples = 0
    else:
      self.clean_samples += 1
      self.congested_samples = 0
      if self.clean_samples >= self.upshift_clean_samples:
        changed = self._set_target(max(self.target_bitrate + BITRATE_MIN_PUBLISH_DELTA,
                                       self.target_bitrate * self.upshift_factor))
        if changed:
          self.logger.debug("livestream bitrate upshift to %d", self.target_bitrate)
          self.clean_samples = 0

    if sample.send_bitrate is not None:
      self.prev_send_bitrate = sample.send_bitrate
    return changed

  def _set_target(self, bitrate: float, force: bool = False) -> bool:
    target = self._clamp_and_round(bitrate)
    if not force and abs(target - self.target_bitrate) < BITRATE_MIN_PUBLISH_DELTA:
      return False
    if not force and self.last_published_bitrate is not None and \
       abs(target - self.last_published_bitrate) < BITRATE_MIN_PUBLISH_DELTA:
      return False

    self.target_bitrate = target
    msg = messaging.new_message(LIVESTREAM_ENCODER_CONTROL)
    msg.livestreamEncoderControl.bitrate = target
    msg.livestreamEncoderControl.sequence = self._next_sequence()
    self.pub_master.send(LIVESTREAM_ENCODER_CONTROL, msg)
    self.last_published_bitrate = target
    return True


class StreamSession:
  shared_pub_master = DynamicPubMaster([])

  def __init__(self, sdp: str, init_camera: str, incoming_services: list[str], outgoing_services: list[str], debug_mode: bool = False):
    from aiortc.mediastreams import VideoStreamTrack
    from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
    from teleoprtc import WebRTCAnswerBuilder

    builder = WebRTCAnswerBuilder(sdp)

    self.video_track = LiveStreamVideoStreamTrack(init_camera) if not debug_mode else VideoStreamTrack()
    builder.add_video_stream(init_camera, self.video_track)

    self.stream = builder.stream()
    self.identifier = str(uuid.uuid4())

    self.incoming_bridge: CerealIncomingMessageProxy | None = None
    self.incoming_bridge_services = incoming_services
    self.outgoing_bridge: CerealOutgoingMessageProxy | None = None
    self.bitrate_controller: LivestreamBitrateController | None = None
    if len(incoming_services) > 0:
      self.incoming_bridge = CerealIncomingMessageProxy(self.shared_pub_master)
    if len(outgoing_services) > 0:
      self.outgoing_bridge = CerealOutgoingMessageProxy(messaging.SubMaster(outgoing_services))

    self.run_task: asyncio.Task | None = None
    self._cleanup_lock = asyncio.Lock()
    self._cleanup_done = False
    self.logger = logging.getLogger("webrtcd")
    self.logger.info(
      "New stream session (%s), init camera %s, incoming services %s, outgoing services %s",
      self.identifier, init_camera, incoming_services, outgoing_services,
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
    assert self.incoming_bridge is not None
    try:
      payload = json.loads(message) if isinstance(message, (bytes, str)) else None
      if isinstance(payload, dict):
        msg_type = payload.get("type")

        if msg_type == "livestreamCameraSwitch":
          self.video_track.switch_camera(payload["data"]["camera"])
          return

        if msg_type == "clockSync":
          data = payload.get("data", {})
          pong = json.dumps({"type": "clockSync", "data": {
            "action": "pong", "browserSendTime": data.get("browserSendTime"), "deviceTime": time.time() * 1000, # noqa: TID251
          }})
          self.stream.get_messaging_channel().send(pong)
          return

        if msg_type == "enableTimingSei":
          enabled = bool(payload.get("data", {}).get("enabled"))
          if hasattr(self.video_track, 'timing_sei_enabled'):
            self.video_track.timing_sei_enabled = enabled
          return

      if payload.get("type") not in self.incoming_bridge_services:
        return
      self.incoming_bridge.send(message)
    except Exception:
      self.logger.exception("Cereal incoming proxy failure")

  async def run(self):
    try:
      await self.stream.wait_for_connection()
      if self.stream.has_messaging_channel():
        if self.incoming_bridge is not None:
          await self.shared_pub_master.add_services_if_needed(self.incoming_bridge_services)
          self.stream.set_message_handler(self.message_handler)
        if self.outgoing_bridge is not None:
          channel = self.stream.get_messaging_channel()
          self.outgoing_bridge.add_channel(channel)
          self.outgoing_bridge.start()
      self.bitrate_controller = LivestreamBitrateController(self.stream.peer_connection, self.shared_pub_master)
      await self.bitrate_controller.start()

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
      if self.bitrate_controller is not None:
        await self.bitrate_controller.stop(reset=True)
        self.bitrate_controller = None
      if self.outgoing_bridge is not None:
        await self.outgoing_bridge.stop()
      await self.stream.stop()


@dataclass
class StreamRequestBody:
  sdp: str
  initCamera: str
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
      del stream_dict[sid]

    session = StreamSession(body.sdp, body.initCamera, body.bridge_services_in, body.bridge_services_out, debug_mode)
    try:
      answer = await session.get_answer()
    except ValueError as e:
      await session.stop()
      raise web.HTTPBadRequest(
        text=json.dumps({"error": "invalid_sdp", "message": str(e)}),
        content_type="application/json",
      ) from e
    except Exception:
      await session.stop()
      raise
    session.start()

    stream_dict[session.identifier] = session

  return web.json_response({"sdp": answer.sdp, "type": answer.type})


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


def webrtcd_thread(host: str, port: int, debug: bool):
  logging.basicConfig(level=logging.CRITICAL, handlers=[logging.StreamHandler()])
  logging_level = logging.DEBUG if debug else logging.INFO
  logging.getLogger("WebRTCStream").setLevel(logging_level)
  logging.getLogger("webrtcd").setLevel(logging_level)

  app = web.Application()

  app['streams'] = dict()
  app['stream_lock'] = asyncio.Lock()
  app['debug'] = debug
  app.on_shutdown.append(on_shutdown)
  app.router.add_post("/stream", get_stream)
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
