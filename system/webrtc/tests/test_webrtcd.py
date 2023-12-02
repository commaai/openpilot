#!/usr/bin/env python
import asyncio
from contextlib import closing
import multiprocessing
import socket
import unittest
# for aiortc and its dependencies
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openpilot.system.webrtc.webrtcd import webrtcd_thread

import aiortc
import aiohttp
from teleoprtc import WebRTCOfferBuilder


def get_free_port():
  with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return s.getsockname()[1]


class TestWebrtcdProc(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.host = "0.0.0.0"
    self.proc = None

  def tearDown(self):
    if self.proc is not None and self.proc.is_alive():
      self.proc.kill()

  def start_proc(self):
    # run webrtcd in debug mode
    self.port = get_free_port()
    self.proc = multiprocessing.Process(target=webrtcd_thread, args=(self.host, self.port, True))
    self.proc.start()

  async def assertCompletesWithTimeout(self, awaitable, timeout=1):
    try:
      async with asyncio.timeout(timeout):
        await awaitable
    except asyncio.TimeoutError:
      self.fail("Timeout while waiting for awaitable to complete")

  async def test_webrtcd(self):
    import random
    asyncio.sleep(random.random() * 5)


    self.start_proc()

    url = f"http://{self.host}:{self.port}/stream"
    async def connect(offer):
      async with aiohttp.ClientSession(raise_for_status=True) as session:
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
