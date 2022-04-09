#!/usr/bin/env python3
import os
import sys
import cereal.messaging as messaging
from common.window import Window
import av

def get_codec():
  # init from stock hevc
  fdat = open("/home/batman/Downloads/fcamera.hevc", "rb").read(243446)
  codec = av.codec.Codec('hevc').create()
  buf = av.packet.Packet(fdat)
  codec.decode(buf)
  return codec

if __name__ == "__main__":
  win = Window(1928, 1208)
  codec = get_codec()

  # connect to socket
  os.environ["ZMQ"] = "1"
  messaging.context = messaging.Context()

  sock = messaging.sub_sock("wideRoadEncodeData", None, addr=sys.argv[1], conflate=False)
  #sock = messaging.sub_sock("driverEncodeData", None, addr=sys.argv[1], conflate=False)

  while 1:
    msgs = messaging.drain_sock(sock, wait_for_one=True)
    if len(msgs) > 15:
      print("FRAME DROP")
      continue

    for i,evt in enumerate(msgs):
      dat = evt.wideRoadEncodeData.data
      #dat = evt.driverEncodeData.data
      buf = av.packet.Packet(dat)
      #print(buf)
      try:
        frame = codec.decode(buf)
      except av.error.EOFError:
        print("EOFError")
        codec = get_codec()
        continue
      print(len(msgs), "%.2f" % (evt.logMonoTime/1e9), len(dat), frame)
      if len(frame) == 0:
        continue
      # only draw last frame
      if i == len(msgs) - 1:
        img = frame[0].to_ndarray(format=av.video.format.VideoFormat('rgb24'))
        win.draw(img)


