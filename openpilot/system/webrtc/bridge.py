#!/usr/bin/env python3
import argparse
import asyncio

import requests
from teleoprtc.builder import WebRTCOfferBuilder
from teleoprtc.stream import RTCSessionDescription, StreamingOffer

from openpilot.cereal import messaging
from openpilot.cereal.services import SERVICE_LIST
from openpilot.system.webrtc.helpers import WEBRTCD_PORT


async def bridge(ip: str, services_in: list[str], services_out: list[str]) -> None:
  async def connect(offer: StreamingOffer) -> RTCSessionDescription:
    body = {"sdp": offer.sdp, "cameras": [], "enabled": True, "raw": True,
            "bridge_services_in": services_out, "bridge_services_out": services_in}
    resp = await asyncio.to_thread(requests.post, f"http://{ip}:{WEBRTCD_PORT}/stream", json=body, timeout=30)
    answer = resp.json()
    if "sdp" not in answer:
      raise Exception(f"webrtcd error: {answer}")
    return RTCSessionDescription(answer["sdp"], answer["type"])

  pm = messaging.PubMaster(services_in)
  socks = [messaging.sub_sock(s) for s in services_out]

  def on_message(message: bytes | str) -> None:
    if isinstance(message, str):
      print("webrtcd:", message)
    else:
      pm.send(messaging.log_from_bytes(message).which(), message)

  builder = WebRTCOfferBuilder(connect)
  builder.add_messaging()
  stream = builder.stream()
  stream.set_message_handler(on_message)
  try:
    await stream.start()
    await stream.wait_for_connection()
    print(f"connected to {ip}")

    channel = stream.get_messaging_channel()
    while not stream.connection_stopped_event.is_set():
      for sock in socks:
        for dat in messaging.drain_sock_raw(sock):
          channel.send(dat)
      await asyncio.sleep(0.01)
  finally:
    await stream.stop()


async def main() -> None:
  parser = argparse.ArgumentParser(description="Bridge cereal services between this machine and a device running webrtcd. " +
                                               "On the device, run: openpilot/system/webrtc/webrtcd.py --host 0.0.0.0")
  parser.add_argument("ip", help="device IP address")
  parser.add_argument("services", nargs="?", default=None,
                      help="comma-separated services to receive from the device (default: all except video)")
  parser.add_argument("--send", default="", help="comma-separated services to send to the device")
  args = parser.parse_args()

  services_out = [s for s in args.send.split(",") if s]
  services_in = args.services.split(",") if args.services else [s for s in SERVICE_LIST if not s.endswith("EncodeData") and s not in services_out]

  while True:
    try:
      await bridge(args.ip, services_in, services_out)
      print("disconnected")
    except Exception as e:
      print(f"failed to connect to {args.ip}: {e}")
    await asyncio.sleep(1)


if __name__ == "__main__":
  asyncio.run(main())
