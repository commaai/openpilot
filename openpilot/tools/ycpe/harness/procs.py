"""What the harness and the dev tests need to start servers: the openpilot venv, free ports, waiting for a listener."""
import socket
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def openpilot_venv(root=REPO):
  """<root>/.venv, else the main checkout's (worktrees share it)"""
  common = subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True)
  return next((v for v in (root / ".venv", Path(common.stdout.strip()).parent / ".venv") if v.exists()), None)


def listening(port):
  with socket.socket() as s:
    return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(ports, offsets=(0,)):
  """the first of ports with nothing listening on port + each offset (a server and its sidecars)"""
  for port in ports:
    if not any(listening(port + o) for o in offsets):
      return port
  raise RuntimeError(f"no free port in {ports}")


def wait_port(port, proc, what, timeout=15):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if proc.poll() is not None:
      raise RuntimeError(f"{what} exited")
    if listening(port):
      return
    time.sleep(0.05)
  raise RuntimeError(f"{what} isn't listening on {port}")
