#!/usr/bin/env python3
import io
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

sys.path.insert(0, "/data/arm_teleop_deps")
from servos import Arm, IDS, JOINTS, ports
from msgq.visionipc import VisionIpcClient
from openpilot.cereal.visionipc import VisionStreamType

FOLLOWER = "5CBC431994"
WATCHDOG = 0.4
TICKS_PER_DEGREE = 4096 / 360


class Camera:
  def __init__(self):
    self.condition = threading.Condition()
    self.jpeg = b""
    self.frame = 0
    self.updated = 0.
    self.fps = 0.
    self.error = "Waiting for wide camera"
    threading.Thread(target=self.run, daemon=True).start()

  def run(self):
    while True:
      try:
        client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)
        client.connect(True)
        last = time.monotonic()
        while True:
          buf = client.recv()
          if buf is None:
            if time.monotonic() - last > 2:
              raise TimeoutError("Wide camera stopped sending frames")
            continue
          data = np.asarray(buf.data, dtype=np.uint8)
          y = data[:buf.uv_offset].reshape(-1, buf.stride)[:buf.height, :buf.width]
          uv = data[buf.uv_offset:buf.uv_offset + buf.stride * (buf.height // 2)].reshape(-1, buf.stride)[:, :buf.width]
          yuv = np.empty((buf.height, buf.width, 3), dtype=np.uint8)
          yuv[:, :, 0] = np.clip((y.astype(np.float32) - 16) * (255 / 219), 0, 255).astype(np.uint8)
          yuv[:, :, 1] = uv[:, ::2].repeat(2, axis=0).repeat(2, axis=1)
          yuv[:, :, 2] = uv[:, 1::2].repeat(2, axis=0).repeat(2, axis=1)
          image = Image.frombytes("YCbCr", (buf.width, buf.height), yuv.tobytes()).convert("RGB")
          image.thumbnail((1008, 570))
          output = io.BytesIO()
          image.save(output, "JPEG", quality=75)
          now = time.monotonic()
          with self.condition:
            self.jpeg = output.getvalue()
            self.frame += 1
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 0.001)
            self.updated = now
            self.error = ""
            self.condition.notify_all()
          last = now
      except Exception as e:
        self.error = str(e)
        time.sleep(1)


class Robot:
  def __init__(self):
    self.lock = threading.RLock()
    self.arm = None
    self.state = {"serial": FOLLOWER, "connected": False, "positions": [], "error": "Connecting to left arm"}
    self.active = False
    self.holding = False
    self.session = None
    self.sequence = -1
    self.last_command = 0.
    self.goal = []
    self.commanded = []
    self.speed = 30.
    self.reason = "Control is off"
    self.running = True
    threading.Thread(target=self.run, daemon=True).start()

  def stop(self, reason):
    self.active = False
    self.session = None
    self.reason = reason
    if self.arm is not None and self.holding:
      self.arm.targets(self.arm.positions())

  def command(self, body):
    with self.lock:
      op = body["op"]
      if op == "stop":
        self.stop(body.get("reason", "Stopped by operator"))
        return {"ok": True}
      if not self.arm:
        raise ValueError("Left arm is not connected")
      if op == "release":
        self.stop("Torque released")
        self.arm.torque(False)
        self.holding = False
        return {"ok": True}
      if op == "start":
        if self.active:
          raise ValueError("Control is already active")
        if any(m != 777 for m in self.arm.models) or any(self.arm.modes):
          raise ValueError("Expected six STS3215 servos in position mode")
        positions = self.arm.positions()
        if any(not lo <= p <= hi for p, (lo, hi) in zip(positions, self.arm.limits, strict=True)):
          raise ValueError("Joint position is outside the servo limits")
        self.speed = float(body.get("speed", 30))
        if not math.isfinite(self.speed) or not 5 <= self.speed <= 60:
          raise ValueError("Speed must be between 5 and 60 degrees/second")
        self.arm.targets(positions)
        self.holding = True
        try:
          for i in IDS:
            self.arm.write(i, 42, positions[i - 1], 2)
            self.arm.write(i, 41, 20)
            self.arm.write(i, 44, 0, 2)
            self.arm.write(i, 46, round(self.speed * TICKS_PER_DEGREE), 2)
          self.arm.torque(True)
        except Exception:
          self.stop("Could not enable all servos")
          raise
        self.goal = positions[:]
        self.commanded = positions[:]
        self.session = secrets.token_hex(16)
        self.sequence = -1
        self.last_command = time.monotonic()
        self.active = True
        self.reason = "Following leader"
        return {"session": self.session, "positions": positions, "limits": self.arm.limits}
      if op == "target":
        if not self.active or body.get("session") != self.session:
          raise ValueError("Control session has stopped")
        sequence = int(body["sequence"])
        if sequence <= self.sequence:
          raise ValueError("Stale command")
        values = body["positions"]
        if len(values) != 6 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
          raise ValueError("Expected six finite joint positions")
        self.goal = [max(lo, min(hi, p)) for p, (lo, hi) in zip(values, self.arm.limits, strict=True)]
        self.sequence = sequence
        self.last_command = time.monotonic()
        return {"ok": True, "positions": self.state["positions"]}
      raise ValueError("Unknown command")

  def run(self):
    last = time.monotonic()
    retry_at = 0.
    while self.running:
      now = time.monotonic()
      with self.lock:
        try:
          if self.arm is None and now >= retry_at:
            retry_at = now + 2
            self.arm = Arm(FOLLOWER)
            self.state.update(connected=True, port=self.arm.path, limits=self.arm.limits, models=self.arm.models, error="")
          if self.arm is not None:
            positions = self.arm.positions()
            self.state["positions"] = positions
            if self.active:
              if now - self.last_command > WATCHDOG:
                self.stop("Command connection lost; holding pose")
              else:
                step = self.speed * TICKS_PER_DEGREE * min(now - last, 0.05)
                self.commanded = [max(p - 100, min(p + 100, c + max(-step, min(step, g - c))))
                                  for p, c, g in zip(positions, self.commanded, self.goal, strict=True)]
                self.arm.targets(self.commanded)
        except Exception as e:
          self.active = False
          self.session = None
          self.reason = "Arm communication lost"
          self.state.update(connected=False, error=str(e) or "Left arm not found")
          if self.arm:
            try:
              if self.holding:
                self.arm.targets(self.arm.positions())
            except Exception:
              pass
            self.arm.close()
            self.arm = None
          self.holding = False
        last = now
      time.sleep(max(0, 1 / 30 - (time.monotonic() - now)))

  def status(self):
    with self.lock:
      return {**self.state, "active": self.active, "holding": self.holding, "reason": self.reason,
              "target": self.commanded, "command_age": time.monotonic() - self.last_command if self.active else None}


class Handler(BaseHTTPRequestHandler):
  protocol_version = "HTTP/1.1"

  def setup(self):
    super().setup()
    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

  def log_message(self, *args):
    pass

  def json(self, body, status=200):
    data = json.dumps(body).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(data)

  def do_GET(self):
    if self.path == "/status":
      self.json({"arm": robot.status(), "joints": JOINTS, "camera": {"fps": round(camera.fps, 1), "frames": camera.frame,
                 "age": time.monotonic() - camera.updated if camera.updated else None, "error": camera.error},
                 "os": Path("/VERSION").read_text().strip(), "performance": os.getenv("ARM_TELEOP") == "1"})
    elif self.path == "/video":
      self.connection.settimeout(2)
      self.send_response(200)
      self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
      self.send_header("Cache-Control", "no-store")
      self.send_header("Connection", "close")
      self.end_headers()
      frame = -1
      try:
        while True:
          with camera.condition:
            camera.condition.wait_for(lambda: camera.frame != frame and camera.jpeg, timeout=3)
            if not camera.jpeg or time.monotonic() - camera.updated > 2:
              break
            frame, jpeg = camera.frame, camera.jpeg
          self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
          self.wfile.flush()
      except (BrokenPipeError, ConnectionResetError, TimeoutError):
        pass
      self.close_connection = True
    else:
      self.json({"error": "Not found"}, 404)

  def do_POST(self):
    if self.path != "/command" or self.headers.get("Origin") or self.headers.get_content_type() != "application/json":
      self.json({"error": "Unsupported request"}, 400)
      self.close_connection = True
      return
    try:
      size = int(self.headers.get("Content-Length", "0"))
      if not 0 < size < 8192:
        raise ValueError("Invalid command length")
      self.json(robot.command(json.loads(self.rfile.read(size))))
    except (ValueError, KeyError, TypeError, OSError) as e:
      self.json({"error": str(e)}, 400)


if __name__ == "__main__":
  camera = Camera()
  robot = Robot()
  server = ThreadingHTTPServer(("127.0.0.1", 8022), Handler)
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
  try:
    server.serve_forever()
  finally:
    with robot.lock:
      robot.running = False
      robot.stop("Device service stopped")
    server.server_close()
