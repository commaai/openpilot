#!/usr/bin/env python3

import argparse
import asyncio
import dataclasses
import json
import uuid
import logging
from typing import List, Optional

import aiortc
from aiohttp import web

from openpilot.tools.bodyteleop.webrtc import WebRTCAnswerBuilder
from openpilot.tools.bodyteleop.webrtc.info import parse_info_from_offer
from openpilot.tools.bodyteleop.webrtc.tracks import DummyVideoStreamTrack
from openpilot.tools.bodyteleop.webrtc.device.video import LiveStreamVideoStreamTrack
from openpilot.tools.bodyteleop.webrtc.device.audio import AudioInputStreamTrack, AudioOutputSpeaker

from cereal import messaging


class CerealOutgoingMessageProxy:
  def __init__(self, services: List[str]):
    self.sm = messaging.SubMaster(services)
    self.channels: List[aiortc.RTCDataChannel] = []
    self.is_running = False
    self.task = None
    self.logger = logging.getLogger("webrtcd")

  def add_channel(self, channel: aiortc.RTCDataChannel):
    self.channels.append(channel)

  def start(self):
    assert self.task is None
    self.task = asyncio.create_task(self.run())

  def stop(self):
    if self.task.done():
      return
    self.task.cancel()
    self.task = None

  async def run(self):
    while True:
      try:
        # this is blocking in async context...
        self.sm.update(0)
        for service, updated in self.sm.updated.items():
          if not updated:
            continue
          msg_dict, mono_time, valid = self.sm[service].to_dict(), self.sm.logMonoTime[service], self.sm.valid[service]
          outgoing_msg = {"type": service, "logMonoTime": mono_time, "valid": valid, "data": msg_dict}
          encoded_msg = json.dumps(outgoing_msg).encode()
          for channel in self.channels:
            channel.send(encoded_msg)
      except Exception as ex:
        self.logger.error("Cereal outgoing proxy failure: %s", ex)
      await asyncio.sleep(0.01)


class StreamSession:
  def __init__(self, sdp: str, cameras: List[str], incoming_services: List[str], outgoing_services: List[str], debug_mode: bool = False):
    config = parse_info_from_offer(sdp)
    builder = WebRTCAnswerBuilder(sdp)

    assert len(cameras) == config.n_expected_camera_tracks, "Incoming stream has misconfigured number of video tracks"
    for cam in cameras:
      if debug_mode:
        track = DummyVideoStreamTrack(cam)
      else:
        track = LiveStreamVideoStreamTrack(cam)
      builder.add_video_stream(cam, track)
    if config.expected_audio_track and not debug_mode:
      track = AudioInputStreamTrack()
      builder.add_audio_stream(track)
    if config.incoming_audio_track and not debug_mode:
      builder.offer_to_receive_audio_stream()

    self.stream = builder.stream()
    self.identifier = str(uuid.uuid4())
    self.outgoing_bridge = CerealOutgoingMessageProxy(outgoing_services)
    self.pub_master = messaging.PubMaster(incoming_services)
    self.audio_output: Optional[AudioOutputSpeaker] = None
    self.run_task: Optional[asyncio.Task] = None
    self.logger = logging.getLogger("webrtcd")
    self.logger.info("New stream session (%s), cameras %s, audio in %s out %s, incoming services %s, outgoing services %s",
                      self.identifier, cameras, config.incoming_audio_track, config.expected_audio_track, incoming_services, outgoing_services)

  def start(self):
    self.run_task = asyncio.create_task(self.run())

  def stop(self):
    if self.run_task.done():
      return
    self.run_task.cancel()
    self.run_task = None
    asyncio.run(self.post_run_cleanup())

  async def get_answer(self):
    return await self.stream.start()

  async def message_handler(self, message: bytes):
    msg_json = json.loads(message)
    try:
      msg_type, msg_data = msg_json["type"], msg_json["data"]
      msg = messaging.new_message(msg_type)
      setattr(msg, msg_type, msg_data)
      self.pub_master.send(msg_type, msg)
    except Exception as ex:
      self.logger.error("Cereal incoming proxy failure: %s", ex)

  async def run(self):
    try:
      await self.stream.wait_for_connection()
      if self.stream.has_messaging_channel():
        self.stream.set_message_handler(self.message_handler)
        channel = self.stream.get_messaging_channel()
        self.outgoing_bridge.add_channel(channel)
        self.outgoing_bridge.start()
      if self.stream.has_incoming_audio_track():
        track = self.stream.get_incoming_audio_track(False)
        self.audio_output = AudioOutputSpeaker()
        self.audio_output.add_track(track)
        self.audio_output.start()
      self.logger.info("Stream session (%s) connected", self.identifier)

      await self.stream.wait_for_disconnection()
      await self.post_run_cleanup()

      self.logger.info("Stream session (%s) ended", self.identifier)
    except Exception as ex:
      self.logger.error("Stream session failure: %s", ex)

  async def post_run_cleanup(self):
    await self.stream.stop()
    self.outgoing_bridge.stop()
    if self.audio_output:
      self.audio_output.stop()


@dataclasses.dataclass
class StreamRequestBody:
  sdp: str
  cameras: List[str]
  bridge_services_in: List[str]
  bridge_services_out: List[str]


async def get_stream(request: web.Request):
  stream_dict, debug_mode = request.app['streams'], request.app['debug']
  raw_body = await request.json()
  body = StreamRequestBody(**raw_body)

  session = StreamSession(body.sdp, body.cameras, body.bridge_services_in, body.bridge_services_out, debug_mode)
  answer = await session.get_answer()
  session.start()

  stream_dict[session.identifier] = session

  return web.json_response({"sdp": answer.sdp, "type": answer.type})


async def on_shutdown(app: web.Application):
  for session in app['streams'].values():
    session.stop()
  del app['streams']


def webrtcd_thread(host: str, port: int, debug: bool):
  logging.basicConfig(level=logging.CRITICAL, handlers=[logging.StreamHandler()])
  logging_level = logging.DEBUG if debug else logging.INFO
  logging.getLogger("WebRTCStream").setLevel(logging_level)
  logging.getLogger("webrtcd").setLevel(logging_level)

  app = web.Application()

  app['streams'] = dict()
  app['debug'] = debug
  app.on_shutdown.append(on_shutdown)
  app.router.add_post("/stream", get_stream)

  web.run_app(app, host=host, port=port)


if __name__=="__main__":
  parser = argparse.ArgumentParser(description="WebRTC daemon")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
  parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  args = parser.parse_args()

  webrtcd_thread(args.host, args.port, args.debug)
