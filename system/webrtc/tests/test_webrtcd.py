#!/usr/bin/env python
import asyncio
import unittest
import multiprocessing
# for aiortc and its dependencies
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openpilot.system.webrtc.webrtcd import webrtcd_thread

import aiortc
import aiohttp
from teleoprtc import WebRTCOfferBuilder


class TestWebrtcdProc(unittest.IsolatedAsyncioTestCase):
  HOST = "0.0.0.0"
  PORT = 48888

  def setUp(self):
    # run webrtcd in debug mode
    self.proc = multiprocessing.Process(target=webrtcd_thread, args=(self.HOST, self.PORT, True))
    self.proc.start()

  def tearDown(self) -> None:
    self.proc.kill()

  async def assertCompletesWithTimeout(self, awaitable, timeout=1):
    try:
      async with asyncio.timeout(timeout):
        await awaitable
    except asyncio.TimeoutError:
      self.fail("Timeout while waiting for awaitable to complete")

  async def test_webrtcd(self):
    self.assertTrue(self.proc.is_alive())

    url = f"http://{self.HOST}:{self.PORT}/stream"
    async def connect(offer):
      async with aiohttp.ClientSession() as session:
        body = {'sdp': offer.sdp, 'cameras': offer.video, 'bridge_services_in': [], 'bridge_services_out': []}
        async with session.post(url, json=body) as resp:
          payload = await resp.json()
          answer = aiortc.RTCSessionDescription(**payload)
          return answer

    builder = WebRTCOfferBuilder(connect)
    builder.offer_to_receive_video_stream("road")
    builder.offer_to_receive_audio_stream()
    builder.add_messaging()

    stream = builder.stream()

    await self.assertCompletesWithTimeout(stream.start())
    await self.assertCompletesWithTimeout(stream.wait_for_connection())

    self.assertTrue(stream.has_incoming_video_track("road"))
    self.assertTrue(stream.has_incoming_audio_track())
    self.assertTrue(stream.has_messaging_channel())

    video_track, audio_track = stream.get_incoming_video_track("road"), stream.get_incoming_audio_track()
    await self.assertCompletesWithTimeout(video_track.recv())
    await self.assertCompletesWithTimeout(audio_track.recv())

    await self.assertCompletesWithTimeout(stream.stop())


if __name__ == "__main__":
  unittest.main()
