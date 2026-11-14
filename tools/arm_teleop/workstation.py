#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import json
import math
from pathlib import Path
import time

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

from servos import Arm, JOINTS


class Station:
  def __init__(self, args):
    self.args = args
    self.leader = None
    self.leader_positions = []
    self.leader_error = "Connecting to leader"
    self.device = {}
    self.device_error = "Connecting to comma four"
    self.active = False
    self.owner = None
    self.heartbeat = 0.
    self.session = None
    self.sequence = 0
    self.scale = 0.5
    self.signs = [1] * 6
    self.note = "Ready when both arms and camera are connected"
    self.rate = 0.
    self.latency = 0.
    self.clients = set()
    self.lock = asyncio.Lock()
    self.base = f"http://127.0.0.1:{args.tunnel_port}"

  async def command(self, body):
    async with self.http.post(self.base + "/command", json=body, timeout=ClientTimeout(total=0.7)) as response:
      result = await response.json()
      if response.status != 200:
        raise ValueError(result.get("error", "Device command failed"))
      return result

  async def stop(self, reason):
    self.active = False
    self.session = None
    self.owner = None
    self.note = reason
    try:
      await self.command({"op": "stop", "reason": reason})
    except Exception:
      pass  # The device's command watchdog also stops motion.

  async def action(self, ws, body):
    async with self.lock:
      op = body["op"]
      if op == "stop":
        await self.stop("Paused; restart to recenter the leader")
      elif op == "release":
        await self.stop("Motors released")
        await self.command({"op": "release"})
      elif op == "start":
        if self.active:
          raise ValueError("Already following; pause before changing settings")
        if not self.leader or self.leader_error:
          raise ValueError("Leader is not connected")
        camera = self.device.get("camera", {})
        if self.device_error or camera.get("age") is None or camera["age"] > 1:
          raise ValueError("Wait for a live camera connection")
        scale = float(body.get("scale", 0.5))
        signs = body.get("signs", [1] * 6)
        if not math.isfinite(scale) or not 0.1 <= scale <= 1 or len(signs) != 6 or any(s not in (-1, 1) for s in signs):
          raise ValueError("Invalid motion settings")
        await asyncio.to_thread(self.leader.torque, False)
        positions = await asyncio.to_thread(self.leader.positions)
        result = await self.command({"op": "start", "speed": body.get("speed", 30)})
        self.leader_base = positions[:]
        self.previous_leader = positions[:]
        self.follower_base = result["positions"]
        self.limits = result["limits"]
        self.offsets = [0.] * 6
        self.scale = scale
        self.signs = signs
        self.session = result["session"]
        self.sequence = 0
        self.owner = ws
        self.heartbeat = time.monotonic()
        self.active = True
        self.note = "Following your leader arm"
      else:
        raise ValueError("Unknown command")

  async def controls(self):
    retry_at = 0.
    last_command = time.monotonic()
    while True:
      started = time.monotonic()
      async with self.lock:
        try:
          if self.leader is None and started >= retry_at:
            retry_at = started + 2
            self.leader = await asyncio.to_thread(Arm, self.args.leader)
          if self.leader:
            self.leader_positions = await asyncio.to_thread(self.leader.positions)
            self.leader_error = ""
          if self.active:
            if started - self.heartbeat > 0.6:
              await self.stop("Browser connection lost; holding pose")
            elif self.device_error or (self.device.get("camera", {}).get("age") or 0) > 1:
              await self.stop("Camera connection lost; holding pose")
            else:
              deltas = [(p - prev + 2048) % 4096 - 2048 for p, prev in zip(self.leader_positions, self.previous_leader, strict=True)]
              if any(abs(d) > 256 for d in deltas):
                raise ValueError("Leader moved too quickly; pause and restart")
              self.offsets = [old + delta for old, delta in zip(self.offsets, deltas, strict=True)]
              self.previous_leader = self.leader_positions[:]
              target = [max(lo, min(hi, p + delta * self.scale * sign))
                        for p, delta, sign, (lo, hi) in zip(self.follower_base, self.offsets, self.signs, self.limits, strict=True)]
              before = time.monotonic()
              await self.command({"op": "target", "session": self.session, "sequence": self.sequence, "positions": target})
              now = time.monotonic()
              self.latency = 0.8 * self.latency + 0.2 * (now - before) * 1000
              self.rate = 0.8 * self.rate + 0.2 / max(now - last_command, 0.001)
              last_command = now
              self.sequence += 1
        except Exception as e:
          if self.active:
            await self.stop(str(e))
          self.leader_error = str(e) or "Leader not found"
          if self.leader:
            self.leader.close()
            self.leader = None
        await asyncio.sleep(max(0, 1 / 30 - (time.monotonic() - started)))

  def status(self):
    return {"leader": {"serial": self.args.leader, "connected": self.leader is not None and not self.leader_error,
                       "positions": self.leader_positions, "error": self.leader_error},
            "device": self.device, "device_error": self.device_error, "active": self.active, "note": self.note,
            "rate": round(self.rate, 1) if self.active else 0, "latency": round(self.latency, 1), "joints": JOINTS}

  async def monitor(self):
    while True:
      try:
        async with self.http.get(self.base + "/status", timeout=ClientTimeout(total=0.7)) as response:
          response.raise_for_status()
          self.device = await response.json()
          self.device_error = ""
      except Exception:
        self.device_error = "Reconnecting to comma four"
      state = {"type": "state", **self.status()}
      for ws in tuple(self.clients):
        with contextlib.suppress(ConnectionError):
          await ws.send_json(state)
      await asyncio.sleep(0.1)

  async def tunnel(self):
    while True:
      proc = await asyncio.create_subprocess_exec(
        "ssh", "-N", "-T", "-i", self.args.key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=5",
        "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2",
        "-L", f"127.0.0.1:{self.args.tunnel_port}:127.0.0.1:8022", f"comma@{self.args.device}")
      try:
        await proc.wait()
      finally:
        if proc.returncode is None:
          proc.terminate()
          await proc.wait()
      await asyncio.sleep(2)

  async def lifecycle(self, app):
    self.http = ClientSession()
    tasks = [asyncio.create_task(fn()) for fn in (self.tunnel, self.controls, self.monitor)]
    yield
    async with self.lock:
      await self.stop("Workstation service stopped")
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if self.leader:
      self.leader.close()
    await self.http.close()

  async def websocket(self, request):
    if request.headers.get("Origin") != f"http://{request.host}":
      raise web.HTTPForbidden()
    ws = web.WebSocketResponse(heartbeat=5, max_msg_size=8192)
    await ws.prepare(request)
    self.clients.add(ws)
    try:
      async for msg in ws:
        if msg.type == WSMsgType.TEXT:
          try:
            body = json.loads(msg.data)
            if body.get("op") == "heartbeat":
              if ws is self.owner:
                self.heartbeat = time.monotonic()
            else:
              await self.action(ws, body)
          except (ValueError, KeyError, TypeError, OSError) as e:
            await ws.send_json({"type": "error", "message": str(e)})
    finally:
      self.clients.discard(ws)
      if ws is self.owner:
        async with self.lock:
          await self.stop("Browser closed; holding pose")
    return ws

  async def video(self, request):
    try:
      upstream = await self.http.get(self.base + "/video", timeout=ClientTimeout(total=None, connect=2, sock_read=4))
    except Exception:
      raise web.HTTPServiceUnavailable(text="Waiting for wide camera")
    async with upstream:
      response = web.StreamResponse(headers={"Content-Type": upstream.headers["Content-Type"], "Cache-Control": "no-store"})
      await response.prepare(request)
      try:
        async for chunk in upstream.content.iter_chunked(65536):
          await response.write(chunk)
      except (ConnectionError, asyncio.TimeoutError):
        pass
    return response


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="192.168.63.237")
  parser.add_argument("--key", required=True)
  parser.add_argument("--leader", default="5CBC431223")
  parser.add_argument("--port", type=int, default=8070)
  parser.add_argument("--tunnel-port", type=int, default=18022)
  args = parser.parse_args()
  args.key = str(Path(args.key).expanduser().resolve())
  station = Station(args)
  app = web.Application()
  app.cleanup_ctx.append(station.lifecycle)

  async def index(request):
    return web.FileResponse(Path(__file__).with_name("index.html"), headers={"Cache-Control": "no-store"})

  async def status(request):
    return web.json_response(station.status())

  app.router.add_get("/", index)
  app.router.add_get("/status", status)
  app.router.add_get("/ws", station.websocket)
  app.router.add_get("/video", station.video)
  web.run_app(app, host="127.0.0.1", port=args.port, shutdown_timeout=2)


if __name__ == "__main__":
  main()
