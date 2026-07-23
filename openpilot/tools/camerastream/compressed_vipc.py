#!/usr/bin/env python3
import os
import argparse
import multiprocessing
import time
import signal
from collections import deque


import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.tools.camerastream.ffmpeg_decoder import Decoder, FFmpegError

V4L2_BUF_FLAG_KEYFRAME = 8

# start encoderd
# also start cereal messaging bridge
# then run this "./compressed_vipc.py <ip>"

ENCODE_SOCKETS = {
  VisionStreamType.VISION_STREAM_ROAD: "roadEncodeData",
  VisionStreamType.VISION_STREAM_DRIVER: "driverEncodeData",
  VisionStreamType.VISION_STREAM_WIDE_ROAD: "wideRoadEncodeData",
}

def decoder(addr, vipc_server, vst, W, H, debug=False):
  sock_name = ENCODE_SOCKETS[vst]
  if debug:
    print(f"start decoder for {sock_name}, {W}x{H}")

  codec = Decoder("hevc")

  os.environ["ZMQ"] = "1"
  messaging.reset_context()
  sock = messaging.sub_sock(sock_name, None, addr=addr, conflate=False)
  cnt = 0
  last_idx = -1
  seen_iframe = False

  time_q = deque()

  def resync():
    nonlocal seen_iframe
    codec.reset()
    seen_iframe = False
    time_q.clear()

  while 1:
    msgs = messaging.drain_sock(sock, wait_for_one=True)
    for evt in msgs:
      evta = getattr(evt, evt.which())
      if last_idx != -1 and evta.idx.encodeId != (last_idx + 1):
        if debug:
          print("DROP PACKET!")
        resync()
      last_idx = evta.idx.encodeId
      if not seen_iframe and not (evta.idx.flags & V4L2_BUF_FLAG_KEYFRAME):
        if debug:
          print("waiting for iframe")
        continue
      time_q.append(time.monotonic())
      network_latency = (int(time.time()*1e9) - evta.unixTimestampNanos)/1e6  # noqa: TID251
      frame_latency = ((evta.idx.timestampEof/1e9) - (evta.idx.timestampSof/1e9))*1000
      process_latency = ((evt.logMonoTime/1e9) - (evta.idx.timestampEof/1e9))*1000

      # put in header (first) — VPS/SPS/PPS only, no frame expected
      if not seen_iframe:
        try:
          codec.decode(evta.header)
        except FFmpegError as e:
          if debug:
            print(f"HEADER ERROR: {e}")
          resync()
          continue
        seen_iframe = True

      try:
        img_yuv = codec.decode(evta.data)
      except FFmpegError as e:
        if debug:
          print(f"DECODE ERROR: {e}")
        resync()
        continue

      if img_yuv is None:
        if debug:
          print("DROP SURFACE")
        continue

      if codec.width != W or codec.height != H:
        if debug:
          print(f"DECODE ERROR: decoded frame is {codec.width}x{codec.height}, expected {W}x{H}")
        resync()
        continue

      frame_start_time = time_q.popleft()
      vipc_server.send(vst, img_yuv.data, cnt, int(frame_start_time*1e9), int(time.monotonic()*1e9))
      cnt += 1

      pc_latency = (time.monotonic()-frame_start_time)*1000
      if debug:
        print(f"{len(msgs):2d} {evta.idx.encodeId:4d} {evt.logMonoTime/1e9:.3f} {evta.idx.timestampEof/1e6:.3f} \
            roll {frame_latency:6.2f} ms latency {process_latency:6.2f} ms + {network_latency:6.2f} ms + {pc_latency:6.2f} ms \
            = {process_latency+network_latency+pc_latency:6.2f} ms", len(evta.data), sock_name)

class CompressedVipc:
  def __init__(self, addr, vision_streams, server_name, debug=False):
    print("getting frame sizes")
    os.environ["ZMQ"] = "1"
    messaging.reset_context()
    sm = messaging.SubMaster([ENCODE_SOCKETS[s] for s in vision_streams], addr=addr)
    while min(sm.recv_frame.values()) == 0:
      sm.update(100)
    os.environ.pop("ZMQ")
    messaging.reset_context()

    self.vipc_server = VisionIpcServer(server_name)
    for vst in vision_streams:
      ed = sm[ENCODE_SOCKETS[vst]]
      self.vipc_server.create_buffers(vst, 4, ed.width, ed.height)
    self.vipc_server.start_listener()

    self.procs = []
    for vst in vision_streams:
      ed = sm[ENCODE_SOCKETS[vst]]
      p = multiprocessing.Process(target=decoder, args=(addr, self.vipc_server, vst, ed.width, ed.height, debug))
      p.start()
      self.procs.append(p)

  def join(self):
    for p in self.procs:
      p.join()

  def kill(self):
    for p in self.procs:
      p.terminate()
    self.join()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Decode video streams and broadcast on VisionIPC")
  parser.add_argument("addr", help="Address of comma three")
  parser.add_argument("--cams", default="0,1,2", help="Cameras to decode")
  parser.add_argument("--server", default="camerad", help="choose vipc server name")
  parser.add_argument("--silent", action="store_true", help="Suppress debug output")
  args = parser.parse_args()

  vision_streams = [
    VisionStreamType.VISION_STREAM_ROAD,
    VisionStreamType.VISION_STREAM_DRIVER,
    VisionStreamType.VISION_STREAM_WIDE_ROAD,
  ]

  vsts = [vision_streams[int(x)] for x in args.cams.split(",")]
  cvipc = CompressedVipc(args.addr, vsts, args.server, debug=(not args.silent))

  # register exit handler
  signal.signal(signal.SIGINT, lambda sig, frame: cvipc.kill())

  cvipc.join()
