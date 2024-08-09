import os
import time
import random
import numpy as np
from msgq.visionipc import VisionIpcServer, VisionIpcClient, VisionStreamType

def zmq_sleep(t=1):
  if "ZMQ" in os.environ:
    time.sleep(t)


class TestVisionIpc:

  def setup_vipc(self, name, *stream_types, num_buffers=1, rgb=False, width=100, height=100, conflate=False):
    self.server = VisionIpcServer(name)
    for stream_type in stream_types:
      self.server.create_buffers(stream_type, num_buffers, rgb, width, height)
    self.server.start_listener()

    if len(stream_types):
      self.client = VisionIpcClient(name, stream_types[0], conflate)
      assert self.client.connect(True)
    else:
      self.client = None

    zmq_sleep()
    return self.server, self.client

  def test_connect(self):
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD)
    assert self.client.is_connected
    del self.client
    del self.server

  def test_available_streams(self):
    for k in range(4):
      stream_types = set(random.choices([x.value for x in VisionStreamType], k=k))
      self.setup_vipc("camerad", *stream_types)
      available_streams = VisionIpcClient.available_streams("camerad", True)
      assert available_streams == stream_types
      del self.client
      del self.server

  def test_buffers(self):
    width, height, num_buffers = 100, 200, 5
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD, num_buffers=num_buffers, width=width, height=height)
    assert self.client.width == width
    assert self.client.height == height
    assert self.client.buffer_len > 0
    assert self.client.num_buffers == num_buffers
    del self.client
    del self.server

  def test_yuv_rgb(self):
    _, client_yuv = self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD, rgb=False)
    _, client_rgb = self.setup_vipc("navd", VisionStreamType.VISION_STREAM_MAP, rgb=True)
    assert client_rgb.rgb
    assert not client_yuv.rgb
    del client_yuv
    del client_rgb
    del self.server

  def test_send_single_buffer(self):
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD)

    buf = np.zeros(self.client.buffer_len, dtype=np.uint8)
    buf.view('<i4')[0] = 1234
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=1337)

    recv_buf = self.client.recv()
    assert recv_buf is not None
    assert recv_buf.data.view('<i4')[0] == 1234
    assert self.client.frame_id == 1337
    del self.client
    del self.server

  def test_no_conflate(self):
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD)

    buf = np.zeros(self.client.buffer_len, dtype=np.uint8)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=1)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=2)

    recv_buf = self.client.recv()
    assert recv_buf is not None
    assert self.client.frame_id == 1

    recv_buf = self.client.recv()
    assert recv_buf is not None
    assert self.client.frame_id == 2
    del self.client
    del self.server

  def test_conflate(self):
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD, conflate=True)

    buf = np.zeros(self.client.buffer_len, dtype=np.uint8)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=1)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=2)

    recv_buf = self.client.recv()
    assert recv_buf is not None
    assert self.client.frame_id == 2

    recv_buf = self.client.recv()
    assert recv_buf is None
    del self.client
    del self.server
