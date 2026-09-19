#!/usr/bin/env python3
"""Run the narrow-camera 720p60 bench and its browser receiver; Ctrl-C stops it."""
import argparse
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--duration", type=float, default=60, help="maximum run time in seconds")
  args = parser.parse_args()
  from openpilot.common.params import Params
  from openpilot.common.hardware import HARDWARE
  from openpilot.cereal import messaging

  if HARDWARE.get_device_type() != "mici" or not Params().get_bool("IsOffroad"):
    parser.error("requires a parked, offroad comma four")
  if args.duration <= 0:
    parser.error("duration must be positive")
  # Never compete with the normal manager or camera/encoder daemons.
  for proc in Path("/proc").iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
      continue
    try:
      cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
      comm = (proc / "comm").read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
      continue
    if comm in ("camerad", "encoderd", "manager", "webrtcd") or any(x in cmd for x in (
      "openpilot.system.manager.manager", "system/manager/manager.py", "openpilot.system.webrtc.webrtcd", "system/webrtc/webrtcd.py"
    )):
      parser.error(f"stop the normal manager/camera/stream processes first (PID {proc.name}: {comm})")

  class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      data = (ROOT / "scripts/camera120.html").read_bytes()
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)

    def do_POST(self):
      if self.path != "/stream":
        self.send_error(404)
        return
      conn = http.client.HTTPConnection("127.0.0.1", 5001, timeout=35)
      try:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        conn.request("POST", "/stream", body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        self.send_response(response.status)
        self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
      finally:
        conn.close()

  server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
  server.daemon_threads = True
  env = dict(os.environ, CAMERA_720P60="1", PYTHONPATH=str(ROOT))
  env.pop("DISABLE_ROAD", None)
  children = []
  old_bitrate = Params().get("LivestreamEncoderBitrate")
  old_keyframe = Params().get("LivestreamRequestKeyframe")
  sockets = {name: messaging.sub_sock(name, conflate=False) for name in ("narrowRoadCameraState", "livestreamNarrowRoadEncodeData")}
  samples = {name: [] for name in sockets}
  commands = [
    [str(ROOT / "openpilot/system/camerad/camerad")],
    [str(ROOT / "openpilot/system/loggerd/encoderd"), "--stream"],
    [sys.executable, "-m", "openpilot.system.webrtc.webrtcd", "--host", "0.0.0.0"],
  ]
  def stop(signum, frame):
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, stop)
  try:
    for command in commands:
      children.append(subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Open http://<comma-IP>:{args.port} on your laptop. Bench stops after {args.duration:g}s or Ctrl-C.", flush=True)
    deadline = time.monotonic() + args.duration
    started = report_time = time.monotonic()
    last_received = dict.fromkeys(sockets, started)
    while time.monotonic() < deadline:
      if not Params().get_bool("IsOffroad"):
        raise RuntimeError("device left offroad state; stopping bench")
      for child in children:
        if child.poll() is not None:
          raise RuntimeError(f"bench process exited: {child.args}, status {child.returncode}")
      now = time.monotonic()
      for name, sock in sockets.items():
        for msg in messaging.drain_sock(sock):
          data = getattr(msg, name)
          idx = data.idx if name.endswith("EncodeData") else data
          samples[name].append((idx.frameId, idx.timestampSof))
          last_received[name] = now
        if now - last_received[name] > 10:
          raise RuntimeError(f"no frames from {name} for 10 seconds")
      if now - report_time >= 5:
        for name, rows in samples.items():
          if len(rows) >= 2 and rows[-1][1] > rows[0][1]:
            fps = (len(rows) - 1) * 1e9 / (rows[-1][1] - rows[0][1])
            missing = sum(max(0, b[0] - a[0] - 1) for a, b in zip(rows, rows[1:], strict=False))
            print(f"{name}: {fps:.2f} FPS, {missing} missing frames", flush=True)
            if now - started > 10 and (not 57 <= fps <= 63 or missing):
              raise RuntimeError(f"{name} did not sustain 60 FPS without frame loss")
          samples[name] = []
        report_time = now
      # Abort when any reported CPU/GPU/memory temperature reaches 85 C.
      for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
          kind = (zone / "type").read_text().lower()
          if any(name in kind for name in ("cpu", "gpu", "ddr")) and int((zone / "temp").read_text()) >= 85000:
            raise RuntimeError(f"thermal limit reached: {kind.strip()}")
        except (FileNotFoundError, ValueError):
          pass
      time.sleep(0.25)
  except KeyboardInterrupt:
    pass
  finally:
    for child in reversed(children):
      if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    for child in children:
      try:
        child.wait(timeout=5)
      except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    server.server_close()
    if old_bitrate is None:
      Params().remove("LivestreamEncoderBitrate")
    else:
      Params().put("LivestreamEncoderBitrate", old_bitrate)
    if old_keyframe is None:
      Params().remove("LivestreamRequestKeyframe")
    else:
      Params().put("LivestreamRequestKeyframe", old_keyframe)


if __name__ == "__main__":
  main()
