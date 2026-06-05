import struct
import unittest
from typing import Optional
from msgq.visionipc import VisionIpcServer, VisionIpcClient, VisionStreamType


class TestVisionIpc(unittest.TestCase):
  server: Optional[VisionIpcServer]
  client: Optional[VisionIpcClient]

  def setUp(self):
    self.server = None
    self.client = None

  def tearDown(self):
    self.client = None
    self.server = None

  def setup_vipc(self, name, *stream_types, num_buffers=1, width=100, height=100, conflate=False):
    self.server = VisionIpcServer(name)
    for stream_type in stream_types:
      self.server.create_buffers(stream_type, num_buffers, width, height)
    self.server.start_listener()

    if len(stream_types):
      self.client = VisionIpcClient(name, stream_types[0], conflate)
      assert self.client.connect(True)
    else:
      self.client = None

    return self.server, self.client

  def test_connect(self):
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD)
    assert self.client is not None
    assert self.client.is_connected()

  def test_available_streams(self):
    stream_types = (VisionStreamType.VISION_STREAM_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.setup_vipc("camerad", *stream_types)
    available_streams = VisionIpcClient.available_streams("camerad", True)
    assert available_streams == {stream.value for stream in stream_types}

  def test_buffers(self):
    width, height, num_buffers = 100, 200, 5
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD, num_buffers=num_buffers, width=width, height=height)
    assert self.client is not None
    assert self.client.width == width
    assert self.client.height == height
    assert self.client.buffer_len is not None and self.client.buffer_len > 0
    assert self.client.num_buffers == num_buffers

  def test_send_single_buffer(self):
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD)
    assert self.server is not None
    assert self.client is not None
    assert self.client.buffer_len is not None
    buf = bytearray(self.client.buffer_len)
    struct.pack_into("<Q", buf, 0, 1234)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=1337)

    recv_buf = self.client.recv()
    assert recv_buf is not None
    data = recv_buf.data
    assert isinstance(data, memoryview)
    assert struct.unpack_from("<Q", data, 0)[0] == 1234
    assert len(data) == self.client.buffer_len
    assert data[8:].nbytes == self.client.buffer_len - 8
    assert self.client.frame_id == 1337
    assert recv_buf.frame_id == 1337

  def test_no_conflate(self):
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD)
    assert self.server is not None
    assert self.client is not None
    assert self.client.buffer_len is not None
    buf = bytearray(self.client.buffer_len)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=1)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=2)

    recv_buf = self.client.recv()
    assert recv_buf is not None
    assert self.client.frame_id == 1

    recv_buf = self.client.recv()
    assert recv_buf is not None
    assert self.client.frame_id == 2

  def test_conflate(self):
    self.setup_vipc("camerad", VisionStreamType.VISION_STREAM_ROAD, conflate=True)
    assert self.server is not None
    assert self.client is not None
    assert self.client.buffer_len is not None
    buf = bytearray(self.client.buffer_len)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=1)
    self.server.send(VisionStreamType.VISION_STREAM_ROAD, buf, frame_id=2)

    recv_buf = self.client.recv()
    assert recv_buf is not None
    assert self.client.frame_id == 2

    recv_buf = self.client.recv(timeout_ms=5)
    assert recv_buf is None
