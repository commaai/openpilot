#!/usr/bin/env python3

import argparse
import asyncio
import json
import urllib.error
import urllib.request

import av
import cv2
import pygame
from libdatachannel import H264RtpDepacketizer, NalUnit, RtcpReceivingSession, Track

from teleoprtc import StreamingOffer, WebRTCOfferBuilder
from teleoprtc.stream import RTCSessionDescription


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


def pygame_should_quit() -> bool:
  return any(event.type == pygame.QUIT for event in pygame.event.get())


class WebrtcdConnectionProvider:
  """Connection provider reaching the webrtcd server on a comma three."""
  def __init__(self, host: str, port: int = 5001, timeout: float = 10.0):
    self.url = f"http://{host}:{port}/stream"
    self.timeout = timeout

  async def __call__(self, offer: StreamingOffer) -> RTCSessionDescription:
    return await asyncio.to_thread(self._post_offer, offer)

  def _post_offer(self, offer: StreamingOffer) -> RTCSessionDescription:
    body = {"sdp": offer.sdp, "init_camera": offer.video[0], "enabled": True}
    data = json.dumps(body).encode()
    request = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    try:
      with urllib.request.urlopen(request, timeout=self.timeout) as response:
        payload = json.loads(response.read().decode())
    except urllib.error.URLError as e:
      raise RuntimeError(f"failed to connect to webrtcd at {self.url}") from e

    if "error" in payload:
      raise RuntimeError(f"webrtcd rejected stream request: {payload.get('message', payload['error'])}")

    return RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])


class H264FrameReceiver:
  """
  Decode an incoming libdatachannel H264 video track into av.VideoFrame objects.

  Construct with a track from WebRTCBaseStream.get_incoming_video_track() while an
  asyncio loop is running, then `await recv()` for the most recent decoded frame.
  A small bounded queue drops stale frames so recv() stays close to live.
  """
  def __init__(self, track: Track, max_pending_frames: int = 2):
    self._loop = asyncio.get_running_loop()
    self._track = track
    self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_pending_frames)
    self._decoder = av.CodecContext.create("h264", "r")

    depacketizer = H264RtpDepacketizer(NalUnit.Separator.StartSequence)
    self._rtcp = RtcpReceivingSession()
    self._media_handlers = [depacketizer, self._rtcp]  # keep refs alive for the track's lifetime
    track.set_media_handler(depacketizer)
    track.chain_media_handler(self._rtcp)
    track.on_frame(self._on_frame)
    track.request_keyframe()

  def request_keyframe(self) -> None:
    self._track.request_keyframe()

  def _on_frame(self, data, info) -> None:
    self._loop.call_soon_threadsafe(self._enqueue, bytes(data))

  def _enqueue(self, data: bytes) -> None:
    if self._queue.full():
      self._queue.get_nowait()
    self._queue.put_nowait(data)

  async def recv(self) -> av.VideoFrame:
    while True:
      data = await self._queue.get()
      try:
        for packet in self._decoder.parse(data):
          frames = self._decoder.decode(packet)
          if frames:
            return frames[-1]
      except av.FFmpegError:
        self._track.request_keyframe()


class FaceDetector:
  """Simple face detector using OpenCV."""
  def __init__(self):
    self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if self.classifier.empty():
      raise RuntimeError("failed to load OpenCV haarcascade_frontalface_default.xml")

  def detect(self, array):
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    return self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  def draw(self, array, faces) -> None:
    for (x, y, w, h) in faces:
      cv2.rectangle(array, (x, y), (x + w, y + h), (0, 255, 0), 2)


async def run_face_detection(stream, camera: str):
  pygame.init()
  screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
  pygame.display.set_caption("Face detection demo")

  receiver = H264FrameReceiver(stream.get_incoming_video_track(camera))
  detector = FaceDetector()

  try:
    while stream.is_connected_and_ready and not pygame_should_quit():
      try:
        frame = await asyncio.wait_for(receiver.recv(), timeout=0.5)
      except TimeoutError:
        receiver.request_keyframe()  # the initial keyframe request can be dropped, so keep nudging until frames arrive
        continue

      array = cv2.resize(frame.to_ndarray(format="rgb24"), (SCREEN_WIDTH, SCREEN_HEIGHT))
      detector.draw(array, detector.detect(array))

      pygame.surfarray.blit_array(screen, array.swapaxes(0, 1))
      pygame.display.flip()
  finally:
    pygame.quit()
    await stream.stop()


async def run(args):
  builder = WebRTCOfferBuilder(WebrtcdConnectionProvider(args.host, args.port))
  builder.offer_to_receive_video_stream(args.camera)
  builder.add_messaging()

  stream = builder.stream()
  print(f"Connecting to webrtcd at {args.host}:{args.port} ...")
  await stream.start()
  await stream.wait_for_connection()
  print(f"Connected. Opening window for {args.camera} camera.")

  assert stream.has_incoming_video_track(args.camera)

  await run_face_detection(stream, args.camera)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="localhost", help="Host for webrtcd server")
  parser.add_argument("--port", type=int, default=5001, help="Port for webrtcd server")
  parser.add_argument("--camera", choices=("driver", "wideRoad", "road"), default="driver", help="Camera to stream")

  args = parser.parse_args()
  asyncio.run(run(args))


if __name__ == "__main__":
  main()
