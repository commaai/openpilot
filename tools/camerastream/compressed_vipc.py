#!/usr/bin/env python3
import os
import sys
import numpy as np
import multiprocessing

from cereal.visionipc.visionipc_pyx import VisionIpcServer, VisionStreamType  # pylint: disable=no-name-in-module, import-error
W, H = 1928, 1208

def writer(fn, addr, sock_name):
  import cereal.messaging as messaging
  HEADER = b"\x00\x00\x00\x01\x40\x01\x0c\x01\xff\xff\x01\x60\x00\x00\x03\x00\xb0\x00\x00\x03\x00\x00\x03\x00\x96\xac\x09\x00\x00\x00\x01\x42\x01\x01\x01\x60\x00\x00\x03\x00\xb0\x00\x00\x03\x00\x00\x03\x00\x96\xa0\x03\xd0\x80\x13\x07\x1b\x2e\x5a\xee\x4c\x92\xea\x00\xbb\x42\x84\xa0\x00\x00\x00\x01\x44\x01\xc0\xe2\x4f\x09\xc1\x80\xc6\x08\x40\x00"
  fifo_file = open(fn, "wb")
  fifo_file.write(HEADER)
  fifo_file.flush()

  os.environ["ZMQ"] = "1"
  messaging.context = messaging.Context()

  sock = messaging.sub_sock(sock_name, None, addr=addr, conflate=False)
  last_idx = -1
  seen_iframe = False
  while 1:
    msgs = messaging.drain_sock(sock, wait_for_one=True)
    for evt in msgs:
      evta = getattr(evt, evt.which())
      lat = ((evt.logMonoTime/1e9) - (evta.timestampEof/1e6))*1000
      print("%2d %4d %.3f %.3f latency %.2f ms" % (len(msgs), evta.idx, evt.logMonoTime/1e9, evta.timestampEof/1e6, lat), len(evta.data), sock_name)
      if evta.idx != 0 and evta.idx != (last_idx+1):
        print("DROP!")
      last_idx = evta.idx
      if len(evta.data) > 4 and evta.data[4] == 0x26:
        seen_iframe = True
      if not seen_iframe:
        print("waiting for iframe")
        continue
      fifo_file.write(evta.data)
      fifo_file.flush()

def decoder_nvidia(fn, vipc_server, vst):
  sys.path.append("/raid.dell2/PyNvCodec")
  import PyNvCodec as nvc # pylint: disable=import-error
  decoder = nvc.PyNvDecoder(fn, 0, {"probesize": "32"})
  conv = nvc.PySurfaceConverter(W, H, nvc.PixelFormat.NV12, nvc.PixelFormat.BGR, 0)
  cc1 = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709, nvc.ColorRange.JPEG)
  nvDwn = nvc.PySurfaceDownloader(W, H, nvc.PixelFormat.BGR, 0)

  img = np.ndarray((H,W,3), dtype=np.uint8)
  cnt = 0
  while 1:
    rawSurface = decoder.DecodeSingleSurface()
    if rawSurface.Empty():
      continue
    convSurface = conv.Execute(rawSurface, cc1)
    nvDwn.DownloadSingleSurface(convSurface, img)
    vipc_server.send(vst, img.flatten().data, cnt, 0, 0)
    cnt += 1

def decoder_ffmpeg(fn, vipc_server, vst):
  import av # pylint: disable=import-error
  container = av.open(fn, options={"probesize": "32"})
  cnt = 0
  for frame in container.decode(video=0):
    img = frame.to_ndarray(format=av.video.format.VideoFormat('bgr24'))
    vipc_server.send(vst, img.flatten().data, cnt, 0, 0)
    cnt += 1

import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Decode video streams and broacast on VisionIPC')
  parser.add_argument("addr", help="Address of comma 3")
  parser.add_argument('--nvidia', action='store_true', help='Use nvidia instead of ffmpeg')
  parser.add_argument("--cams", default="0,1,2", help="Cameras to decode")
  args = parser.parse_args()

  all_cams = [
    ("roadEncodeData", VisionStreamType.VISION_STREAM_RGB_ROAD),
    ("wideRoadEncodeData", VisionStreamType.VISION_STREAM_RGB_WIDE_ROAD),
    ("driverEncodeData", VisionStreamType.VISION_STREAM_RGB_DRIVER),
  ]
  cams = dict([all_cams[int(x)] for x in args.cams.split(",")])

  vipc_server = VisionIpcServer("camerad")
  for vst in cams.values():
    vipc_server.create_buffers(vst, 4, True, W, H)
  vipc_server.start_listener()

  for k,v in cams.items():
    FIFO_NAME = "/tmp/decodepipe_"+k
    if os.path.exists(FIFO_NAME):
      os.unlink(FIFO_NAME)
    os.mkfifo(FIFO_NAME)
    multiprocessing.Process(target=writer, args=(FIFO_NAME, sys.argv[1], k)).start()
    if args.nvidia:
      multiprocessing.Process(target=decoder_nvidia, args=(FIFO_NAME, vipc_server, v)).start()
    else:
      multiprocessing.Process(target=decoder_ffmpeg, args=(FIFO_NAME, vipc_server, v)).start()
