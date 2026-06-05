import json, math, os, socketserver, threading, unittest
import numpy as np
from tinygrad import Tensor, dtypes
from extra.tinyfs.fetch_file import hash_file, _python_hash_1mb

_chunks: dict[bytes, bytes] = {}

class _Handler(socketserver.StreamRequestHandler):
  def handle(self):
    while line := self.rfile.readline():
      cmd = line.decode().strip()
      if cmd == "INFO":
        self.wfile.write(json.dumps({"node0": ["node0", f"127.0.0.1:{self.server.server_address[1]}"]}).encode() + b"\r\n")
      elif cmd.startswith("STORE_IN"):
        data = self.rfile.read(int(cmd.split()[1]))
        hashes = bytearray()
        for i in range(math.ceil(len(data) / Tensor.CHUNK_SIZE)):
          chunk = data[i*Tensor.CHUNK_SIZE:(i+1)*Tensor.CHUNK_SIZE].ljust(Tensor.CHUNK_SIZE, b'\0')
          h = _python_hash_1mb(chunk)
          _chunks[h] = chunk
          hashes.extend(h)
        self.wfile.write(hashes)
      elif cmd.startswith("LOAD_IN"):
        hashes = self.rfile.read(int(cmd.split()[1]))
        self.wfile.write(json.dumps(["node0"] * (len(hashes) // 16)).encode() + b"\r\n")
      elif cmd.startswith("CHUNK_OUT"):
        size = int(cmd.split()[1])
        self.wfile.write(_chunks.get(self.rfile.read(16), bytes(size))[:size])
      self.wfile.flush()

# regressed in 55d3a5def "preallocate all realized buffers"
class TestTinyFS(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    _chunks.clear()
    cls._server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), _Handler)
    cls._server.daemon_threads = True
    threading.Thread(target=cls._server.serve_forever, daemon=True).start()
    os.environ["TINYFS_ENDPOINT"] = f"127.0.0.1:{cls._server.server_address[1]}"

  @classmethod
  def tearDownClass(cls):
    _chunks.clear()
    os.environ.pop("TINYFS_ENDPOINT", None)
    cls._server.shutdown()
    cls._server.server_close()

  def test_store(self):
    h = Tensor([1.0, 2.0, 3.0, 4.0]).fs_store().realize()
    self.assertEqual(h.shape, (16,))
    self.assertEqual(h.dtype, dtypes.uint8)

  def test_store_deterministic(self):
    a = Tensor([1.0, 2.0, 3.0, 4.0]).fs_store().realize()
    b = Tensor([1.0, 2.0, 3.0, 4.0]).fs_store().realize()
    np.testing.assert_array_equal(a.numpy(), b.numpy())

  def test_store_different_data(self):
    a = Tensor([1.0, 2.0, 3.0, 4.0]).fs_store().realize()
    b = Tensor([5.0, 6.0, 7.0, 8.0]).fs_store().realize()
    self.assertNotEqual(a.tolist(), b.tolist())

  def test_roundtrip_uint8(self):
    arr = np.arange(256, dtype=np.uint8)
    loaded = Tensor(arr).fs_store().realize().fs_load(len(arr)).to("CPU")
    np.testing.assert_array_equal(loaded.numpy(), arr)

  def test_roundtrip_multichunk_uint8(self):
    arr = np.random.default_rng(42).integers(0, 256, size=Tensor.CHUNK_SIZE + 1024, dtype=np.uint8)
    loaded = Tensor(arr).fs_store().realize().fs_load(len(arr)).to("CPU")
    np.testing.assert_array_equal(loaded.numpy(), arr)

  def test_hash_matches_python_impl(self):
    arr = np.arange(256, dtype=np.uint8)
    h = Tensor(arr).fs_store().realize()
    # the hash from fs_store should match the pure-Python hash_file reference
    padded = arr.tobytes().ljust(Tensor.CHUNK_SIZE, b'\0')
    self.assertEqual(h.data().tobytes(), hash_file(padded))

if __name__ == "__main__":
  unittest.main()
