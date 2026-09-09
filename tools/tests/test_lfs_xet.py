import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import lfs_xet


class TestLfsXet(unittest.TestCase):
  def tearDown(self):
    lfs_xet.xet_connection.cache_clear()

  def test_native_download_and_progress(self):
    oid, xet_hash = "a" * 64, "b" * 64
    action = {"href": f"https://cdn.hf.co/xet-bridge-us/repo/{xet_hash}?signed=secret"}
    token = {"casUrl": "https://storage.example", "accessToken": "first", "exp": 100}

    def native(files, *, endpoint, token_info, token_refresher, progress_updater):
      self.assertEqual(files, [(str(path), xet_hash, 5)])
      self.assertEqual(endpoint, token["casUrl"])
      self.assertEqual(token_info, ("first", 100))
      self.assertEqual(token_refresher(), ("refreshed", 200))
      path.write_bytes(b"hello")
      progress_updater[0](2)
      progress_updater[0](3)

    module = Mock(PyXetDownloadInfo=lambda *args: args, download_files=native)
    responses = [{"objects": [{"actions": {"download": action}}]}, token, token | {"accessToken": "refreshed", "exp": 200}]
    with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", hf_xet=module), \
         patch.object(lfs_xet, "request_json", side_effect=responses), patch.object(lfs_xet, "send") as send:
      path = Path(directory) / oid
      lfs_xet.download(oid, 5, path)
      self.assertEqual(path.read_bytes(), b"hello")
      self.assertEqual([call.args[0]["bytesSoFar"] for call in send.call_args_list], [2, 5])
      self.assertEqual([call.args[0]["bytesSinceLast"] for call in send.call_args_list], [2, 3])

  def test_lfs_object_not_yet_converted_to_xet(self):
    action = {"href": "https://cdn.example/object", "header": {"X-Test": "value"}}
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(lfs_xet, "request_json", return_value={"objects": [{"actions": {"download": action}}]}), \
         patch.object(lfs_xet.urllib.request, "urlopen", return_value=io.BytesIO(b"hello")) as request, \
         patch.object(lfs_xet, "send"):
      path = Path(directory) / "object"
      lfs_xet.download("a" * 64, 5, path)
      self.assertEqual(path.read_bytes(), b"hello")
      self.assertEqual(request.call_args.args[0].get_header("X-test"), "value")

  def test_protocol_continues_after_failure_and_cleans_up(self):
    first, second = "a" * 64, "b" * 64
    paths, messages = [], []

    def download(oid, size, path):
      paths.append(path)
      path.write_bytes(b"hello")
      if oid == first:
        raise RuntimeError("https://cdn.example/?token=secret")

    def send(message):
      messages.append(message)
      if message.get("path"):
        self.assertEqual(Path(message["path"]).read_bytes(), b"hello")
        self.assertFalse(paths[0].exists())

    requests = [{"event": "init", "operation": "download"},
                *({"event": "download", "oid": oid, "size": 5} for oid in (first, second)), {"event": "terminate"}]
    with patch.object(lfs_xet.sys, "stdin", io.StringIO("\n".join(map(json.dumps, requests)))), \
         patch.object(lfs_xet, "download", side_effect=download), patch.object(lfs_xet, "send", side_effect=send):
      lfs_xet.main()
    self.assertEqual(messages[0], {})
    self.assertTrue("error" in messages[1])
    self.assertEqual(messages[2]["oid"], second)
    self.assertNotIn("secret", json.dumps(messages))
    self.assertTrue(all(not path.exists() for path in paths))


if __name__ == "__main__":
  unittest.main()
