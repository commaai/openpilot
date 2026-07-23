#!/usr/bin/env python3

import numpy as np

import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.realtime import DT_MDL


VISION_STREAMS = {
  "roadCameraState": VisionStreamType.VISION_STREAM_ROAD,
  "driverCameraState": VisionStreamType.VISION_STREAM_DRIVER,
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
  # NV12 format: Y plane followed by interleaved UV plane
  # UV plane size is stride * uv_height, where uv_height = align(height/2, 16)
  uv_height = ((buf.height // 2) + 15) // 16 * 16
  uv_plane_size = buf.stride * uv_height

  y = np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
  uv_data = buf.data[buf.uv_offset:buf.uv_offset + uv_plane_size]
  u = np.array(uv_data[::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]
  v = np.array(uv_data[1::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]

  return yuv_to_rgb(y, u, v)


def recv_image(client, retries=3):
  for _ in range(retries):
    if (buf := client.recv()) is not None:
      return extract_image(buf)
  raise RuntimeError(f"failed to receive camera frame after {retries} attempts")


def get_snapshots(frame="roadCameraState", front_frame="driverCameraState"):
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
    rear = recv_image(vipc_clients[frame])
  if front_frame is not None:
    front = recv_image(vipc_clients[front_frame])
  return rear, front
