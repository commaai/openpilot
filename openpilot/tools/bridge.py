#!/usr/bin/env python3
import argparse
import asyncio
import atexit
import logging
import os
import re
import signal
import subprocess
import sys
from dataclasses import asdict

import requests

from openpilot.cereal.services import SERVICE_LIST
from openpilot.system.webrtc.cereal_bridge import CerealBridge
from openpilot.system.webrtc.helpers import WEBRTCD_PORT, BridgeRequestBody
from openpilot.tools.lib.auth_config import get_token

ATHENA_API_HOST = os.getenv("ATHENA_API_HOST", "https://athena.comma.ai")
DONGLE_ID_RE = re.compile(r"^[0-9a-f]{16}$")

logger = logging.getLogger("bridge")


class BridgeRefused(Exception):
  pass


def start_bridge(device: str, services: list[str], send: list[str] | None = None) -> subprocess.Popen:
  """Republish services from a remote device into the local msgq, for as long as this process lives."""
  proc = subprocess.Popen([sys.executable, __file__, device, *services, *(["--send", *send] if send else [])])
  atexit.register(proc.terminate)
  return proc


def request_bridge(device: str, body: BridgeRequestBody) -> dict:
  if DONGLE_ID_RE.match(device):
    token = get_token()
    if token is None:
      raise RuntimeError("not authenticated, run openpilot/tools/lib/auth.py")
    payload = {"method": "startBridge", "params": asdict(body), "jsonrpc": "2.0", "id": 0}
    resp = requests.post(f"{ATHENA_API_HOST}/{device}", json=payload, headers={"Authorization": f"JWT {token}"}, timeout=30).json()
    if "error" in resp:
      raise RuntimeError(f"athena: {resp['error']}")
    resp = resp["result"]
  else:
    resp = requests.post(f"http://{device}:{WEBRTCD_PORT}/bridge", json=asdict(body), timeout=30).json()

  if "error" in resp:
    # anything but an unhandled exception means webrtcd turned down the request, so retrying won't help
    raise (RuntimeError if resp["error"] == "exception" else BridgeRefused)(f"webrtcd: {resp}")
  return resp


async def run_session(device: str, services_in: list[str], services_out: list[str]) -> None:
  from teleoprtc.builder import WebRTCOfferBuilder
  from teleoprtc.stream import RTCSessionDescription, StreamingOffer

  # device's perspective: it sends what we receive, and receives what we send
  async def connect(offer: StreamingOffer) -> RTCSessionDescription:
    body = BridgeRequestBody(offer.sdp, services_out=services_in, services_in=services_out)
    answer = await asyncio.to_thread(request_bridge, device, body)
    return RTCSessionDescription(answer["sdp"], answer["type"])

  builder = WebRTCOfferBuilder(connect)
  builder.add_messaging()
  stream = builder.stream()

  bridge = CerealBridge(services_out, services_in)
  stream.set_message_handler(bridge.on_message)
  try:
    await stream.start()
    await asyncio.wait_for(stream.wait_for_connection(), timeout=15)
    bridge.start(stream.get_messaging_channel())
    logger.warning("connected to %s", device)
    await stream.wait_for_disconnection()
    logger.warning("disconnected from %s", device)
  finally:
    await asyncio.to_thread(bridge.stop)
    await stream.stop()


async def run(device: str, services_in: list[str], services_out: list[str]) -> None:
  loop = asyncio.get_running_loop()
  task = asyncio.current_task()
  # close the connection on SIGTERM so the device notices right away
  loop.add_signal_handler(signal.SIGTERM, task.cancel)

  # tools spawn the bridge as a child, don't outlive them if they die without cleaning up
  ppid = os.getppid()
  def check_parent():
    if os.getppid() != ppid:
      task.cancel()
    else:
      loop.call_later(1, check_parent)
  check_parent()

  while True:
    try:
      await run_session(device, services_in, services_out)
    except BridgeRefused:
      raise
    except Exception as e:
      logger.warning("bridge to %s failed: %s", device, e)
    await asyncio.sleep(1)


def main():
  parser = argparse.ArgumentParser(description="Bridge cereal services between a comma device and this machine's msgq over WebRTC.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("device", help="device IP/hostname on the LAN (notCar devices, or webrtcd started with --lan), or a dongle id to connect through athena")
  parser.add_argument("services", nargs="*", help="services to receive from the device (default: all, unless --send is given)")
  parser.add_argument("--send", nargs="+", default=[], help="services to send to the device (e.g. testJoystick), needs athena or webrtcd started with --lan")
  args = parser.parse_args()

  # with only --send, don't receive anything
  services_in = [s for s in args.services or ([] if args.send else SERVICE_LIST) if s not in args.send]
  invalid = [s for s in services_in + args.send if s not in SERVICE_LIST]
  if invalid:
    parser.error(f"unknown services: {invalid}")

  logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
  try:
    asyncio.run(run(args.device, services_in, args.send))
  except (KeyboardInterrupt, asyncio.CancelledError):
    pass
  except BridgeRefused as e:
    sys.exit(f"bridge to {args.device} refused: {e}")


if __name__ == "__main__":
  main()
