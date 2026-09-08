#!/usr/bin/env python3
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import xet

DATA = b"openpilot Xet test object"
OID = hashlib.sha256(DATA).hexdigest()
ROOT = Path(__file__).resolve().parents[4]


class TestXet(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.path = Path(self.tmp.name) / "model.onnx"
    self.path.write_bytes(DATA)
    self.api = Mock()
    self.download = Mock(return_value=str(self.path))
    self.available = Mock(return_value=True)
    self.addCleanup(patch.stopall)
    patch.dict(sys.modules, {
      "huggingface_hub": SimpleNamespace(HfApi=Mock(return_value=self.api), hf_hub_download=self.download),
      "huggingface_hub.utils._runtime": SimpleNamespace(is_xet_available=self.available),
    }).start()
    patch.object(xet.subprocess, "run", return_value=SimpleNamespace(stdout="")).start()
    patch.object(xet, "git", return_value=str(self.path.parent / "lfs/tmp")).start()

  def test_download_verifies_and_preserves_hub_cache(self):
    downloaded = Path(xet.XetTransfer().download(OID, len(DATA)))
    try:
      self.assertNotEqual(downloaded, self.path)
      self.assertEqual(downloaded.read_bytes(), DATA)
      downloaded.unlink()
      self.assertEqual(self.path.read_bytes(), DATA)
      self.download.assert_called_once_with(repo_id=xet.REPO_ID, filename=f"sha256/{OID}")
    finally:
      downloaded.unlink(missing_ok=True)

  def test_download_rejects_wrong_hash_and_size(self):
    client = xet.XetTransfer()
    for oid, size in [(OID, len(DATA) + 1), ("0" * 64, len(DATA))]:
      with self.subTest(oid=oid, size=size), self.assertRaisesRegex(ValueError, "mismatch"):
        client.download(oid, size)

  def test_upload_verifies_before_sending_and_skips_existing_objects(self):
    client = xet.XetTransfer()
    with self.assertRaisesRegex(ValueError, "mismatch"):
      client.upload("0" * 64, len(DATA), self.path)
    self.api.get_paths_info.assert_not_called()
    self.api.get_paths_info.return_value = []
    client.upload(OID, len(DATA), self.path)
    self.api.upload_file.assert_called_once_with(
      repo_id=xet.REPO_ID, path_in_repo=f"sha256/{OID}", path_or_fileobj=self.path, commit_message=f"Add {OID}",
    )
    self.api.upload_file.reset_mock()
    self.api.get_paths_info.return_value = [SimpleNamespace(size=len(DATA), lfs=SimpleNamespace(sha256=OID))]
    client.upload(OID, len(DATA), self.path)
    self.api.upload_file.assert_not_called()
    self.api.get_paths_info.return_value[0].lfs.sha256 = "0" * 64
    with self.assertRaisesRegex(ValueError, "remote object"):
      client.upload(OID, len(DATA), self.path)

  def test_disabled_xet_fails_initialization(self):
    self.available.return_value = False
    output = io.StringIO()
    xet.transfer(io.StringIO('{"event":"init","operation":"download"}\n'), output)
    self.assertIn("HF_HUB_DISABLE_XET", json.loads(output.getvalue())["error"]["message"])

  def test_protocol_continues_after_failure_and_keeps_stdout_clean(self):
    client = Mock()
    def upload(*args):
      print("library diagnostic")
    client.upload.side_effect = upload
    requests = [
      {"event": "init", "operation": "upload"},
      {"event": "upload", "oid": "../bad", "size": len(DATA), "path": str(self.path)},
      {"event": "upload", "oid": OID, "size": len(DATA), "path": str(self.path)},
      {"event": "terminate"},
    ]
    output = io.StringIO()
    with patch.object(xet, "XetTransfer", return_value=client), patch("sys.stderr", new=io.StringIO()):
      xet.transfer(io.StringIO("".join(json.dumps(r) + "\n" for r in requests)), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    self.assertEqual(responses[0], {})
    self.assertIn("error", responses[1])
    self.assertEqual(responses[2], {"event": "complete", "oid": OID})
    client.upload.assert_called_once_with(OID, len(DATA), self.path)

  def test_fork_repository_override(self):
    xet.subprocess.run.return_value.stdout = "example/openpilot-lfs\n"
    self.assertEqual(xet.XetTransfer().repo_id, "example/openpilot-lfs")


@unittest.skipUnless(shutil.which("git-lfs"), "git-lfs required")
class TestGitLfsXet(unittest.TestCase):
  def test_push_and_pull_with_real_git_lfs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      repo = root / "repo"
      repo.mkdir()
      remote = root / "remote"
      remote.mkdir()
      stub = root / "stub" / "huggingface_hub"
      stub.mkdir(parents=True)
      (stub / "__init__.py").write_text("""
import hashlib
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
STORE = Path(os.environ["TEST_XET_STORE"])
class HfApi:
  def get_paths_info(self, repo_id, paths):
    path = STORE / paths[0]
    if not path.exists():
      return []
    return [SimpleNamespace(size=path.stat().st_size, lfs=SimpleNamespace(sha256=hashlib.sha256(path.read_bytes()).hexdigest()))]
  def upload_file(self, *, repo_id, path_in_repo, path_or_fileobj, commit_message):
    path = STORE / path_in_repo
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path_or_fileobj, path)
def hf_hub_download(*, repo_id, filename):
  return str(STORE / filename)
""")
      (stub / "utils").mkdir()
      (stub / "utils/__init__.py").touch()
      (stub / "utils/_runtime.py").write_text("def is_xet_available():\n  return True\n")
      env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
      env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0",
                 PYTHONPATH=str(stub.parent), TEST_XET_STORE=str(remote), GIT_LFS_SKIP_SMUDGE="1")
      def run(*args):
        return subprocess.check_output(args, cwd=repo, env=env, stderr=subprocess.STDOUT, text=True)
      run("git", "init", "-q")
      run("git", "config", "user.name", "Xet test")
      run("git", "config", "user.email", "xet@example.invalid")
      run("git", "lfs", "install", "--local")
      # No HTTP server exists: these commands must use the standalone agent.
      run("git", "remote", "add", "origin", "https://example.invalid/openpilot.git")
      (repo / "tools").mkdir()
      shutil.copyfile(ROOT / "tools/xet.py", repo / "tools/xet.py")
      run(sys.executable, "tools/xet.py", "configure")
      # Older commits do not contain the adapter script.
      (repo / "tools/xet.py").unlink()
      (repo / ".gitattributes").write_text("*.onnx filter=lfs diff=lfs merge=lfs -text\n")
      model = repo / "model.onnx"
      excluded = repo / "excluded.onnx"
      model.write_bytes(DATA)
      excluded.write_bytes(b"excluded model")
      run("git", "add", ".")
      run("git", "commit", "-qm", "test models")
      run("git", "lfs", "push", "--all", "origin")
      self.assertEqual((remote / f"sha256/{OID}").read_bytes(), DATA)
      # Simulate a fresh clone's empty object cache and pointer worktree.
      shutil.rmtree(repo / ".git/lfs/objects")
      model.unlink()
      excluded.unlink()
      run("git", "checkout", "--", "model.onnx", "excluded.onnx")
      run("git", "lfs", "pull", "--exclude=excluded.onnx")
      self.assertEqual(model.read_bytes(), DATA)
      self.assertTrue(excluded.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"))
      self.assertEqual((remote / f"sha256/{OID}").read_bytes(), DATA)
      self.assertEqual(run("git", "status", "--porcelain"), "")


if __name__ == "__main__":
  unittest.main()
