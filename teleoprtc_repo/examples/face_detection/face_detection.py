#!/usr/bin/env python3

import argparse
import asyncio

import aiortc
import aiohttp
import cv2
import pygame

from teleoprtc import WebRTCOfferBuilder, StreamingOffer


def pygame_should_quit():
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      return True
  return False


class WebrtcdConnectionProvider:
  """
  Connection provider reaching webrtcd server on comma three
  """
  def __init__(self, host, port=5001):
    self.url = f"http://{host}:{port}/stream"

  async def __call__(self, offer: StreamingOffer) -> aiortc.RTCSessionDescription:
    async with aiohttp.ClientSession() as session:
      body = {'sdp': offer.sdp, 'cameras': offer.video, 'bridge_services_in': [], 'bridge_services_out': []}
      async with session.post(self.url, json=body) as resp:
        payload = await resp.json()
        answer = aiortc.RTCSessionDescription(**payload)
        return answer


class FaceDetector:
  """
  Simple face detector using opencv
  """
  def __init__(self):
    self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  def detect(self, array):
    gray_array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    faces = self.classifier.detectMultiScale(gray_array, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

  def draw(self, array, faces):
    for (x, y, w, h) in faces:
      cv2.rectangle(array, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return array


async def run_face_detection(stream):
  # setup pygame window
  pygame.init()
  screen_width, screen_height = 1280, 720
  screen = pygame.display.set_mode((screen_width, screen_height))
  pygame.display.set_caption("Face detection demo")
  surface = pygame.Surface((screen_width, screen_height))

  # get the driver camera video track from the stream
  # generally its better to reuse the track object instead of getting it every time
  track = stream.get_incoming_video_track("driver", buffered=False)
  # cv2 face detector
  detector = FaceDetector()
  while stream.is_connected_and_ready and not pygame_should_quit():
    try:
      # receive frame as pyAV VideoFrame, convert to rgb24 numpy array
      frame = await track.recv()
      array = frame.to_ndarray(format="rgb24")

      # detect faces and draw rects around them
      resized_array = cv2.resize(array, (screen_width, screen_height))
      faces = detector.detect(resized_array)
      detector.draw(resized_array, faces)

      # display the image
      pygame.surfarray.blit_array(surface, resized_array.swapaxes(0, 1))
      screen.blit(surface, (0, 0))
      pygame.display.flip()

      print("Received frame from", "driver", frame.time)
    except aiortc.mediastreams.MediaStreamError:
      break

  pygame.quit()
  await stream.stop()


async def run(args):
  # build your own the offer stream
  builder = WebRTCOfferBuilder(WebrtcdConnectionProvider(args.host))
  # request video stream from drivers camera
  builder.offer_to_receive_video_stream("driver")
  # add cereal messaging streaming support
  builder.add_messaging()

  stream = builder.stream()

  # start the stream then wait for connection
  # server will receive the offer and attempt to fulfill it
  await stream.start()
  await stream.wait_for_connection()
  # all the tracks and channel are ready to be used at this point

  assert stream.has_incoming_video_track("driver") and stream.has_messaging_channel()

  # run face detection loop on the drivers camera
  await run_face_detection(stream)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="localhost", help="Host for webrtcd server")

  args = parser.parse_args()
  asyncio.run(run(args))
