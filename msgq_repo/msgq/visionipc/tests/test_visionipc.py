import gc
import struct
import unittest
from typing import Optional
from msgq.visionipc import VisionIpcServer, VisionIpcClient


class TestVisionIpc(unittest.TestCase):
  server: Optional[VisionIpcServer]
  client: Optional[VisionIpcClient]

  def setUp(self):
    self.server = None
    self.client = None

  def tearDown(self):
    self.client = None
    self.server = None

  def setup_vipc(
    self, name: str, stream_type: int, *other_stream_types: int,
    num_buffers: int = 1, width: int = 100, height: int = 100, conflate: bool = False,
  ) -> tuple[VisionIpcServer, VisionIpcClient, int]:
    self.server = server = VisionIpcServer(name)
    for stream in (stream_type, *other_stream_types):
      server.create_buffers(stream, num_buffers, width, height)
    server.start_listener()

    self.client = client = VisionIpcClient(name, stream_type, conflate)
    assert client.connect(True)
    buffer_len = client.buffer_len
    assert buffer_len is not None
    return server, client, buffer_len

  def test_connect(self):
    _, client, _ = self.setup_vipc("camerad", 0)
    assert client.is_connected()

  def test_available_streams(self):
    stream_types = (0, 2)
    self.setup_vipc("camerad", *stream_types)
    available_streams = VisionIpcClient.available_streams("camerad", True)
    assert available_streams == set(stream_types)

  def test_buffers(self):
    width, height, num_buffers = 100, 200, 5
    _, client, buffer_len = self.setup_vipc("camerad", 0, num_buffers=num_buffers, width=width, height=height)
    assert client.width == width
    assert client.height == height
    assert buffer_len > 0
    assert client.num_buffers == num_buffers

  def test_send_single_buffer(self):
    server, client, buffer_len = self.setup_vipc("camerad", 0)
    buf = bytearray(buffer_len)
    struct.pack_into("<Q", buf, 0, 1234)
    server.send(0, buf, frame_id=1337)

    recv_buf = client.recv()
    assert recv_buf is not None
    data = recv_buf.data
    assert isinstance(data, memoryview)
    assert struct.unpack_from("<Q", data, 0)[0] == 1234
    assert len(data) == buffer_len
    assert data[8:].nbytes == buffer_len - 8
    assert client.frame_id == 1337
    assert recv_buf.frame_id == 1337

  def test_no_conflate(self):
    server, client, buffer_len = self.setup_vipc("camerad", 0)
    buf = bytearray(buffer_len)
    server.send(0, buf, frame_id=1)
    server.send(0, buf, frame_id=2)

    recv_buf = client.recv()
    assert recv_buf is not None
    assert client.frame_id == 1

    recv_buf = client.recv()
    assert recv_buf is not None
    assert client.frame_id == 2

  def test_conflate(self):
    server, client, buffer_len = self.setup_vipc("camerad", 0, conflate=True)
    buf = bytearray(buffer_len)
    server.send(0, buf, frame_id=1)
    server.send(0, buf, frame_id=2)

    recv_buf = client.recv()
    assert recv_buf is not None
    assert client.frame_id == 2

    recv_buf = client.recv(timeout_ms=5)
    assert recv_buf is None

  def test_send_buffer_types(self):
    server, client, buffer_len = self.setup_vipc("camerad", 0)
    data = bytes(i % 256 for i in range(buffer_len))
    strided = bytearray(len(data) * 2)
    strided[::2] = data
    for payload in (data, bytearray(data), memoryview(data), memoryview(strided)[::2], memoryview(data[::-1])[::-1]):
      with self.subTest(type=type(payload)):
        server.send(0, payload, frame_id=42, timestamp_sof=123, timestamp_eof=456)
        buf = client.recv()
        assert buf is not None
        self.assertEqual(bytes(buf.data), data)
        self.assertEqual(buf.frame_id, 42)
        self.assertEqual(client.timestamp_sof, 123)
        self.assertEqual(client.timestamp_eof, 456)
        self.assertFalse(client.valid)

  def test_send_wrong_size(self):
    server, _, _ = self.setup_vipc("camerad", 0)
    with self.assertRaises(AssertionError):
      server.send(0, b"short")

  def test_buffer_keeps_client_alive(self):
    server, client, buffer_len = self.setup_vipc("camerad", 0)
    data = b"a" * buffer_len
    server.send(0, data, frame_id=42)
    buf = client.recv()
    assert buf is not None
    self.client = None
    del client
    gc.collect()
    self.assertEqual(buf.frame_id, 42)
    view = buf.data
    del buf
    gc.collect()
    self.assertEqual(bytes(view), data)
    self.assertFalse(view.readonly)
    view[0] = ord("b")
    self.assertEqual(view[0], ord("b"))

  def test_unconnected_properties(self):
    client = VisionIpcClient("unconnected", 0, False)
    for value in (client.width, client.height, client.stride, client.uv_offset, client.buffer_len):
      self.assertIsNone(value)
    self.assertEqual(client.num_buffers, 0)
    self.assertFalse(client.is_connected())
    self.assertFalse(client.connect(False))
    self.assertIsNone(client.recv(timeout_ms=0))
