#!/usr/bin/env python3

import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient
from openpilot.common.realtime import DT_MDL
from openpilot.system.camerad.cameras.nv12_info import nv12_planes


VISION_STREAMS = {
  "narrowRoadCameraState": VisionStreamType.VISION_STREAM_NARROW_ROAD,
  "cabinCameraState": VisionStreamType.VISION_STREAM_CABIN,
  "wideRoadCameraState": VisionStreamType.VISION_STREAM_WIDE_ROAD,
}


def yuv_to_rgb(y, u, v):
  ul = np.repeat(np.repeat(u, 2).reshape(u.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
  vl = np.repeat(np.repeat(v, 2).reshape(v.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)

  yuv = np.dstack((y, ul, vl)).astype(np.int16)
  yuv[:, :, 1:] -= 128

  m = np.array([
    [1.00000,  1.00000, 1.00000],
    [0.00000, -0.39465, 2.03211],
    [1.13983, -0.58060, 0.00000],
  ])
  rgb = np.dot(yuv, m).clip(0, 255)
  return rgb.astype(np.uint8)


def extract_image(buf):
  return yuv_to_rgb(*nv12_planes(buf))


def get_snapshots(frame="narrowRoadCameraState", front_frame="cabinCameraState"):
  sockets = [s for s in (frame, front_frame) if s is not None]
  sm = messaging.SubMaster(sockets)
  vipc_clients = {s: VisionIpcClient("camerad", VISION_STREAMS[s], True) for s in sockets}

  # wait 4 sec from camerad startup for focus and exposure
  while sm[sockets[0]].frameId < int(4. / DT_MDL):
    sm.update()

  for client in vipc_clients.values():
    client.connect(True)

  # grab images
  rear, front = None, None
  if frame is not None:
    c = vipc_clients[frame]
    rear = extract_image(c.recv())
  if front_frame is not None:
    c = vipc_clients[front_frame]
    front = extract_image(c.recv())
  return rear, front
