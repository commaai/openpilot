import argparse
import asyncio
import aiortc
import dataclasses
import json
import logging

from openpilot.tools.bodyteleop.webrtc import WebRTCStreamBuilder
from openpilot.tools.bodyteleop.webrtc.stream import StreamingOffer
from openpilot.tools.bodyteleop.webrtc.tracks import DummyVideoStreamTrack
from openpilot.tools.bodyteleop.webrtc.device.tracks import LiveStreamVideoStreamTrack, FrameReaderVideoStreamTrack


async def async_input():
  return await asyncio.to_thread(input)


async def StdioConnectionProvider(offer: StreamingOffer) -> aiortc.RTCSessionDescription:
  print("-- Please send this JSON to server --")
  print(json.dumps(dataclasses.asdict(offer)))
  print("-- Press enter when the answer is ready --")
  raw_payload = await async_input()
  payload = json.loads(raw_payload)
  answer = aiortc.RTCSessionDescription(**payload)

  return answer


async def run_answer(args):
  streams = []
  while True:
    print("-- Please enter a JSON from client --")
    raw_payload = await async_input()

    payload = json.loads(raw_payload)
    offer = StreamingOffer(**payload)
    video_tracks = []
    for cam in offer.video:
      if args.dummy_video:
        track = DummyVideoStreamTrack(camera_type=cam)
      elif args.input_video:
        track = FrameReaderVideoStreamTrack(args.input_video, camera_type=cam)
      else:
        track = LiveStreamVideoStreamTrack(cam)
      video_tracks.append(track)
    audio_tracks = []

    stream_builder = WebRTCStreamBuilder.answer(offer.sdp)
    for cam, track in zip(offer.video, video_tracks):
      stream_builder.add_video_stream(cam, track)
    for track in audio_tracks:
      stream_builder.add_audio_stream(track)
    stream = stream_builder.stream()
    answer = await stream.start()
    streams.append(stream)

    print("-- Please send this JSON to client --")
    print(json.dumps({"sdp": answer.sdp, "type": answer.type}))

    await stream.wait_for_connection()


async def run_offer(args):
  stream_builder = WebRTCStreamBuilder.offer(StdioConnectionProvider)
  for cam in args.cameras:
    stream_builder.request_video_stream(cam)
  stream_builder.add_messaging()
  stream = stream_builder.stream()
  _ = await stream.start()
  await stream.wait_for_connection()
  print("Connection established and all tracks are ready")

  tracks = [stream.get_incoming_video_track(cam, False) for cam in args.cameras]
  while True:
    try:
      frames = await asyncio.gather(*[track.recv() for track in tracks])
      for key, frame in zip(args.cameras, frames):
        print("Received frame from", key, frame.time)
    except aiortc.mediastreams.MediaStreamError:
      return
    print("=====================================")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)

  offer_parser = subparsers.add_parser("offer")
  offer_parser.add_argument("cameras", metavar="CAMERA", type=str, nargs="+", default=["driver"], help="Camera types to stream")

  answer_parser = subparsers.add_parser("answer")
  answer_parser.add_argument("--dummy-video", action="store_true", help="Stream dummy frames")
  answer_parser.add_argument("--input-video", type=str, required=False, help="Stream from video file instead")

  args = parser.parse_args()

  loop = asyncio.get_event_loop()
  if args.command == "offer":
    loop.run_until_complete(run_offer(args))
  elif args.command == "answer":
    loop.run_until_complete(run_answer(args))
