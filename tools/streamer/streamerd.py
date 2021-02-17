#!/usr/bin/env python
# pylint: skip-file

import os
import sys
import zmq
import cv2
import numpy as np
import struct

# sudo pip install git+git://github.com/mikeboers/PyAV.git
import av

import cereal.messaging as messaging
from cereal.services import service_list

PYGAME = os.getenv("PYGAME") is not None
if PYGAME:
  import pygame
  imgff = np.zeros((874, 1164, 3), dtype=np.uint8)

# first 74 bytes in any stream
start = "0000000140010c01ffff016000000300b0000003000003005dac5900000001420101016000000300b0000003000003005da0025080381c5c665aee4c92ec80000000014401c0f1800420"

def receiver_thread():
  if PYGAME:
    pygame.init()
    pygame.display.set_caption("vnet debug UI")
    screen = pygame.display.set_mode((1164, 874), pygame.DOUBLEBUF)
    camera_surface = pygame.surface.Surface((1164, 874), 0, 24).convert()

  addr = "192.168.5.11"
  if len(sys.argv) >= 2:
    addr = sys.argv[1]

  context = zmq.Context()
  s = messaging.sub_sock(context, 9002, addr=addr)
  frame_sock = messaging.pub_sock(context, service_list['roadCameraState'].port)

  ctx = av.codec.codec.Codec('hevc', 'r').create()
  ctx.decode(av.packet.Packet(start.decode("hex")))

  # import time
  while 1:
    # t1 = time.time()
    ts, raw = s.recv_multipart()
    ts = struct.unpack('q', ts)[0] * 1000
    # t1, t2 = time.time(), t1
    #print 'ms to get frame:', (t1-t2)*1000

    pkt = av.packet.Packet(raw)
    f = ctx.decode(pkt)
    if not f:
      continue
    f = f[0]
    # t1, t2 = time.time(), t1
    #print 'ms to decode:', (t1-t2)*1000

    y_plane = np.frombuffer(f.planes[0], np.uint8).reshape((874, 1216))[:, 0:1164]
    u_plane = np.frombuffer(f.planes[1], np.uint8).reshape((437, 608))[:, 0:582]
    v_plane = np.frombuffer(f.planes[2], np.uint8).reshape((437, 608))[:, 0:582]
    yuv_img = y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()
    # t1, t2 = time.time(), t1
    #print 'ms to make yuv:', (t1-t2)*1000
    #print 'tsEof:', ts

    dat = messaging.new_message('roadCameraState')
    dat.roadCameraState.image = yuv_img
    dat.roadCameraState.timestampEof = ts
    dat.roadCameraState.transform = map(float, list(np.eye(3).flatten()))
    frame_sock.send(dat.to_bytes())

    if PYGAME:
      yuv_np = np.frombuffer(yuv_img, dtype=np.uint8).reshape(874 * 3 // 2, -1)
      cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420, dst=imgff)
      #print yuv_np.shape, imgff.shape

      #scipy.misc.imsave("tmp.png", imgff)

      pygame.surfarray.blit_array(camera_surface, imgff.swapaxes(0, 1))
      screen.blit(camera_surface, (0, 0))
      pygame.display.flip()


def main(gctx=None):
  receiver_thread()

if __name__ == "__main__":
  main()
