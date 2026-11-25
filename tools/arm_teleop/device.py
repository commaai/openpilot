#!/usr/bin/env python3
from dataclasses import dataclass
from ipaddress import ip_address
import json
import math
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
import time
from http.server import ThreadingHTTPServer

sys.path.insert(0, "/data/arm_teleop_deps")
from servos import Arm, IDS, JOINTS
from openpilot.cereal import messaging
from openpilot.common.http_server import HTTPRequestHandler
from openpilot.common.realtime import Ratekeeper
from openpilot.system.webrtc.helpers import StreamRequestBody, post_stream_request

FOLLOWERS = {"left": "5CBC431994", "middle": "5CBC431998", "right": "5CBC453422"}
DISPLAY_PATH = Path("/tmp/balltoss_display.json")
WATCHDOG = 0.4
TICKS_PER_DEGREE = 4096 / 360
MAX_SERVO_SPEED = 0x7fff  # STS register 46 uses bit 15 for direction.
TORQUE_LIMITS = (450, 700, 650, 500, 450, 333)  # STS reg 48 (0-1000) per joint.
IDLE_TORQUE_LIMITS = (200, 350, 300, 250, 200, 150)
IDLE_RELEASE_SECONDS = 60.0
OVERLOAD_TORQUES = (38, 60, 55, 42, 38, 25)  # STS reg 36 (% threshold, matched below TORQUE_LIMITS).
PROTECTION_TIME = 60  # STS reg 35 (10ms units -> 0.6s).
PROTECTIVE_TORQUE = 20  # STS reg 34 (% fallback torque).


class Camera:
  def __init__(self):
    self.sample = (0, 0., 0.)
    threading.Thread(target=self.monitor, daemon=True).start()

  def monitor(self):
    sm = messaging.SubMaster(["wideRoadCameraState"])
    while True:
      sm.update(1000)
      if sm.updated["wideRoadCameraState"]:
        msg = sm["wideRoadCameraState"]
        sof = msg.timestampSof / 1e9
        frame, last, fps = self.sample
        if last and sof > last and msg.frameId > frame:
          fps = .9 * fps + .1 * (msg.frameId - frame) / (sof - last)
        self.sample = (msg.frameId, sof, fps)

  def status(self):
    frame, sof, fps = self.sample
    age = time.monotonic() - sof if sof else None
    return {"fps": round(fps, 1), "frames": frame, "age": age, "captured_at": time.time() - age if age is not None else 0.,  # noqa: TID251
            "error": "" if age is not None and age < 1 else "Waiting for wide camera"}


@dataclass
class GripperContact:
  position: int
  request: float
  direction: int
  torque_off: bool = False

  def retreat_requested(self, goal, position):
    # Clear the contact only on deliberate input away from the obstruction, beyond encoder jitter.
    retreat = 2 * TICKS_PER_DEGREE
    direction = self.direction or (1 if goal < self.position else -1)
    return (direction * (self.position - goal) >= retreat and direction * (self.request - goal) >= retreat
            and direction * (position - goal) >= 0)


class Robot:
  def __init__(self, name):
    self.name = name
    self.serial = FOLLOWERS[name]
    self.lock = threading.RLock()
    self.arm = None
    self.state = {"serial": self.serial, "connected": False, "positions": [], "error": f"Connecting to {name} arm"}
    self.active = False
    self.holding = False
    self.session = None
    self.lease_deadline = float("inf")
    self.sequence = -1
    self.last_command = 0.
    self.idle_since = time.monotonic()
    self.gripper_contact = None
    self.goal = []
    self.commanded = []
    self.speed = None
    self.reason = "Control is off"
    self.running = True
    threading.Thread(target=self.run, daemon=True).start()

  def configure_protection(self):
    assert self.arm is not None
    for i, overload in zip(IDS, OVERLOAD_TORQUES, strict=True):
      desired = ((34, PROTECTIVE_TORQUE), (35, PROTECTION_TIME), (36, overload))
      changes = [(addr, val) for addr, val in desired if self.arm.read(i, addr, 1) != val]
      if changes:
        was_on = self.arm.read(i, 40, 1)
        if was_on:
          self.arm.write(i, 40, 0)
        self.arm.write(i, 55, 0)
        time.sleep(0.02)
        for addr, val in changes:
          self.arm.write(i, addr, val)
          time.sleep(0.02)
        self.arm.write(i, 55, 1)
        time.sleep(0.02)
        if was_on:
          self.arm.write(i, 40, 1)
      elif self.arm.read(i, 55, 1) != 1:
        self.arm.write(i, 55, 1)
        time.sleep(0.02)

  def stop(self, reason):
    self.active = False
    self.session = None
    self.lease_deadline = float("inf")
    self.reason = reason
    self.gripper_contact = None
    self.idle_since = time.monotonic()
    if self.arm is not None and self.holding:
      self.arm.targets(self.arm.positions())
      self.arm.torque_limits(IDLE_TORQUE_LIMITS)

  def command(self, body):
    with self.lock:
      op = body["op"]
      if op == "stop":
        self.stop(body.get("reason", "Stopped by operator"))
        return {"ok": True}
      if not self.arm:
        raise ValueError(f"{self.name.capitalize()} arm is not connected")
      if op == "release":
        self.stop("Torque released")
        self.arm.torque(False)
        self.holding = False
        return {"ok": True}
      if op == "start":
        if self.active:
          raise ValueError("Control is already active")
        remaining = float(body.get("lease_expires_at", 0)) - time.time()  # noqa: TID251 - Convert UTC once, then enforce with monotonic time.
        if not 0 < remaining <= 300:
          raise ValueError("A current reservation lease of at most five minutes is required")
        lease_deadline = time.monotonic() + remaining
        if any(m != 777 for m in self.arm.models) or any(self.arm.modes):
          raise ValueError("Expected six STS3215 servos in position mode")
        positions = self.arm.positions()
        if self.arm.errors:
          raise ValueError("; ".join(f"{name}: {error}" for name, error in self.arm.errors.items()))
        if any(not lo <= p <= hi for p, (lo, hi) in zip(positions, self.arm.limits, strict=True)):
          raise ValueError("Joint position is outside the servo limits")
        requested = body.get("speed")
        self.speed = None if requested is None else float(requested)
        if self.speed is not None and (isinstance(requested, bool) or not math.isfinite(self.speed) or self.speed <= 0):
          raise ValueError("Speed must be positive and finite, or omitted for full speed")
        servo_speed = MAX_SERVO_SPEED
        if self.speed is not None:
          servo_speed = max(1, round(min(MAX_SERVO_SPEED, self.speed * TICKS_PER_DEGREE)))
        if time.monotonic() >= lease_deadline:
          raise ValueError("Reservation ended before control could start")
        self.arm.targets(positions)
        self.holding = True
        try:
          for i in IDS:
            self.arm.write(i, 42, positions[i - 1], 2)
            self.arm.write(i, 41, 20)
            self.arm.write(i, 44, 0, 2)
            self.arm.write(i, 46, servo_speed, 2)
          self.arm.torque(True)
          self.arm.torque_limits(TORQUE_LIMITS)
        except Exception:
          self.stop("Could not enable all servos")
          raise
        self.goal = positions[:]
        self.commanded = positions[:]
        self.session = secrets.token_hex(16)
        self.sequence = -1
        self.last_command = time.monotonic()
        self.lease_deadline = lease_deadline
        self.active = True
        self.reason = "Control active"
        return {"session": self.session, "positions": positions, "limits": self.arm.limits}
      if op == "target":
        if time.monotonic() >= self.lease_deadline:
          self.stop("Reservation ended; holding pose")
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

  def gripper_goal(self, position, overload):
    goal = self.goal[-1]
    contact = self.gripper_contact
    if contact is None:
      if not overload:
        return goal
      drive = self.commanded[-1] - position or goal - position
      self.gripper_contact = GripperContact(position, goal, (drive > 0) - (drive < 0))
    elif overload and not contact.torque_off:
      self.arm.write(IDS[-1], 40, 0)
      contact.torque_off = True
    elif not overload and contact.retreat_requested(goal, position):
      if contact.torque_off:
        self.arm.write(IDS[-1], 42, position, 2)
        self.arm.write(IDS[-1], 40, 1)
        self.arm.write(IDS[-1], 48, TORQUE_LIMITS[-1], 2)
      self.gripper_contact = None
      self.commanded[-1] = position
      return goal
    self.commanded[-1] = position
    return position

  def run(self):
    rk = Ratekeeper(30, print_delay_threshold=None)
    last = time.monotonic()
    retry_at = 0.
    while self.running:
      now = time.monotonic()
      with self.lock:
        try:
          if self.arm is None and now >= retry_at:
            retry_at = now + 2
            self.arm = Arm(self.serial)
            self.configure_protection()
            self.holding = any(self.arm.read(i, 40, 1) for i in IDS)
            self.idle_since = now
            if self.holding:
              self.arm.torque_limits(IDLE_TORQUE_LIMITS)
            self.state.update(connected=True, port=self.arm.path, limits=self.arm.limits, models=self.arm.models, error="")
          if self.arm is not None:
            positions = self.arm.positions()
            self.state["positions"] = positions
            errors = self.arm.errors
            overload = self.active and self.arm.gripper_overload_only
            fault = "" if overload else "; ".join(f"{name}: {error}" for name, error in errors.items())
            self.state.update(error=fault, faults=errors)
            if fault and self.reason != fault:
              self.stop(fault)
            if self.active:
              if now >= self.lease_deadline:
                self.stop("Reservation ended; holding pose")
              elif now - self.last_command > WATCHDOG:
                self.stop("Command connection lost; holding pose")
              else:
                goals = [*self.goal[:-1], self.gripper_goal(positions[-1], overload)]
                if self.speed is not None:
                  step = self.speed * TICKS_PER_DEGREE * min(now - last, 0.05)
                  goals = [c + max(-step, min(step, g - c)) for c, g in zip(self.commanded, goals, strict=True)]
                self.commanded = [max(p - 100, min(p + 100, g)) for p, g in zip(positions, goals, strict=True)]
                self.arm.targets(self.commanded, gripper=not (self.gripper_contact and self.gripper_contact.torque_off))
            elif self.holding and now - self.idle_since >= IDLE_RELEASE_SECONDS:
              self.arm.torque(False)
              self.holding = False
              if not fault:
                self.reason = "Idle torque released"
        except Exception as e:
          self.active = False
          self.session = None
          self.gripper_contact = None
          self.reason = "Arm communication lost"
          self.state.update(connected=False, error=str(e) or f"{self.name.capitalize()} arm not found")
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
      rk.keep_time()

  def status(self):
    with self.lock:
      return {**self.state, "active": self.active, "holding": self.holding, "reason": self.reason,
              "managed": True, "lease_remaining": max(0, self.lease_deadline - time.monotonic()) if self.active else None,
              "gripper_limited": self.gripper_contact is not None, "target": self.commanded,
              "command_age": time.monotonic() - self.last_command if self.active else None}


def stop_all(reason):
  errors = []
  for robot in robots.values():
    with robot.lock:
      try:
        robot.stop(reason)
      except OSError as e:
        errors.append(str(e))
  if errors:
    raise OSError("; ".join(errors))


def command(body):
  name = body.get("arm", "all" if body["op"] == "stop" else "left")
  if name == "all" and body["op"] == "stop":
    stop_all(body.get("reason", "Stopped by operator"))
    return {"ok": True}
  if not isinstance(name, str) or name not in robots:
    raise ValueError("Choose left, middle, or right; all is only supported for stop")
  return robots[name].command(body)


class Handler(HTTPRequestHandler):
  disable_nagle_algorithm = True

  def end_headers(self):
    if self.close_connection:
      self.send_header("Connection", "close")
    self.send_header("Cache-Control", "no-store")
    super().end_headers()

  def json(self, body, status=200):
    self.send_body(status, json.dumps(body).encode(), "application/json")

  def do_GET(self):
    if not self.authorized():
      return
    if self.path == "/status":
      arms = {name: robot.status() for name, robot in robots.items()}
      self.json({"arm": arms["left"], "arms": arms, "joints": JOINTS, "camera": camera.status(),
                 "device_time": time.time(),  # noqa: TID251
                 "os": Path("/VERSION").read_text().strip(), "performance": os.getenv("ARM_TELEOP") == "1"})
    else:
      self.json({"error": "Not found"}, 404)

  def do_POST(self):
    if not self.authorized():
      return
    if self.path not in ("/command", "/camera/offer", "/display") or self.headers.get("Origin") or self.headers.get_content_type() != "application/json":
      self.close_connection = True
      self.json({"error": "Unsupported request"}, 400)
      return
    try:
      size = int(self.headers.get("Content-Length", "0"))
      offer = self.path == "/camera/offer"
      if not 0 < size < (65536 if offer else 8192):
        raise ValueError("Invalid request length")
      body = json.loads(self.rfile.read(size))
      if self.path == "/display":
        tmp = DISPLAY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({**body, "updated_at": time.time()}))  # noqa: TID251
        tmp.replace(DISPLAY_PATH)
        self.json({"ok": True})
        return
      if offer:
        sdp = body["sdp"]
        media = [line for line in sdp.splitlines() if line.startswith("m=")] if isinstance(sdp, str) else []
        if len(media) != 1 or not media[0].startswith("m=video "):
          raise ValueError("Expected a single video track")
        address = str(ip_address(body["device_ip"])) if "device_ip" in body else None
        request = StreamRequestBody(sdp, ["wideRoad"], enabled=True, continuous_camera=True, camera_bind_address=address)
        try:
          sdp = post_stream_request(request)["sdp"]
        except Exception:
          self.json({"error": "Camera signaling unavailable"}, 503)
          return
        self.json({"type": "answer", "sdp": sdp})
      else:
        self.json(command(body))
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as e:
      self.close_connection = True
      self.json({"error": str(e)}, 400)

  def authorized(self):
    if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + gateway_key):
      self.close_connection = True
      self.json({"error": "Use the reservation gateway"}, 401)
      return False
    return True


if __name__ == "__main__":
  gateway_key = Path("/data/arm_teleop_gateway_key").read_text().strip()
  if len(gateway_key) < 32:
    raise ValueError("Invalid reservation gateway key")
  camera = Camera()
  robots = {name: Robot(name) for name in FOLLOWERS}
  server = ThreadingHTTPServer(("127.0.0.1", 8022), Handler)
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
  try:
    server.serve_forever()
  finally:
    for robot in robots.values():
      robot.running = False
    try:
      stop_all("Device service stopped")
    finally:
      server.server_close()
