#!/usr/bin/env python

import argparse
import asyncio
import dataclasses
import json
import logging

import aiortc
from aiortc.mediastreams import VideoStreamTrack, AudioStreamTrack

from teleoprtc import WebRTCOfferBuilder, WebRTCAnswerBuilder
from teleoprtc.stream import StreamingOffer
from teleoprtc.info import parse_info_from_offer


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
    info = parse_info_from_offer(offer.sdp)
    assert len(offer.video) == info.n_expected_camera_tracks
    video_tracks = [VideoStreamTrack() for _ in offer.video]
    audio_tracks = [AudioStreamTrack()] if info.expected_audio_track else []

    stream_builder = WebRTCAnswerBuilder(offer.sdp)
    for cam, track in zip(offer.video, video_tracks, strict=True):
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
  stream_builder = WebRTCOfferBuilder(StdioConnectionProvider)
  for cam in args.cameras:
    stream_builder.offer_to_receive_video_stream(cam)
  if args.audio:
    stream_builder.offer_to_receive_audio_stream()
  if args.messaging:
    stream_builder.add_messaging()
  stream = stream_builder.stream()
  _ = await stream.start()
  await stream.wait_for_connection()
  print("Connection established and all tracks are ready")

  video_tracks = [stream.get_incoming_video_track(cam, False) for cam in args.cameras]
  audio_track = None
  if stream.has_incoming_audio_track():
    audio_track = stream.get_incoming_audio_track(False)
  while True:
    try:
      frames = await asyncio.gather(*[track.recv() for track in video_tracks])
      for key, frame in zip(args.cameras, frames, strict=True):
        print("Received frame from", key, frame.time)
      if audio_track:
        frame = await audio_track.recv()
        print("Received frame from audio", frame.time)
    except aiortc.mediastreams.MediaStreamError:
      return
    print("=====================================")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)

  offer_parser = subparsers.add_parser("offer", description="Create offer stream")
  offer_parser.add_argument("--audio", action="store_true", help="Offer to receive audio")
  offer_parser.add_argument("--messaging", action="store_true", help="Add messaging support")
  offer_parser.add_argument("cameras", metavar="CAMERA", type=str, nargs="+", default=[], help="Camera types to stream")

  answer_parser = subparsers.add_parser("answer", description="Create answer stream")

  args = parser.parse_args()

  logging.basicConfig(level=logging.CRITICAL, handlers=[logging.StreamHandler()])
  logger = logging.getLogger("WebRTCStream")
  logger.setLevel(logging.DEBUG)

  loop = asyncio.get_event_loop()
  if args.command == "offer":
    loop.run_until_complete(run_offer(args))
  elif args.command == "answer":
    loop.run_until_complete(run_answer(args))
