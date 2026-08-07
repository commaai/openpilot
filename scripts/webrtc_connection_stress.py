#!/usr/bin/env python3
"""Repeatedly connect to webrtcd to exercise connection setup and teardown.

Run from the openpilot repository root:
  ./scripts/webrtc_connection_stress.py --host <device-ip> --count 100
"""

import argparse
import asyncio
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from teleoprtc import StreamingOffer, WebRTCOfferBuilder
from teleoprtc.stream import RTCSessionDescription


@dataclass
class Attempt:
  number: int
  elapsed: float
  error: str | None = None


class WebrtcdConnectionProvider:
  def __init__(self, host: str, port: int, camera: str, timeout: float):
    self.url = f"http://{host}:{port}/stream"
    self.camera = camera
    self.timeout = timeout

  async def __call__(self, offer: StreamingOffer) -> RTCSessionDescription:
    return await asyncio.to_thread(self._post_offer, offer)

  def _post_offer(self, offer: StreamingOffer) -> RTCSessionDescription:
    body = {
      "sdp": offer.sdp,
      "cameras": [self.camera],
      "enabled": True,
      "bridge_services_in": [],
      "bridge_services_out": [],
    }
    request = urllib.request.Request(
      self.url,
      data=json.dumps(body).encode(),
      headers={"Content-Type": "application/json"},
      method="POST",
    )

    try:
      with urllib.request.urlopen(request, timeout=self.timeout) as response:
        payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as e:
      raise RuntimeError(f"request to {self.url} failed: {e}") from e

    if "error" in payload:
      raise RuntimeError(f"webrtcd rejected the connection: {payload.get('message', payload['error'])}")
    return RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])


async def connect_once(args: argparse.Namespace) -> float:
  provider = WebrtcdConnectionProvider(args.host, args.port, args.camera, args.timeout)
  builder = WebRTCOfferBuilder(provider)
  builder.offer_to_receive_video_stream(args.camera)
  if args.messaging:
    builder.add_messaging()
  stream = builder.stream()

  started = time.monotonic()
  try:
    await asyncio.wait_for(stream.start(), timeout=args.timeout)
    await asyncio.wait_for(stream.wait_for_connection(), timeout=args.timeout)
    elapsed = time.monotonic() - started
    if args.hold:
      await asyncio.sleep(args.hold)
    return elapsed
  finally:
    await stream.stop()


def percentile(values: list[float], fraction: float) -> float:
  ordered = sorted(values)
  index = round((len(ordered) - 1) * fraction)
  return ordered[index]


async def run(args: argparse.Namespace) -> int:
  attempts: list[Attempt] = []
  for number in range(1, args.count + 1):
    started = time.monotonic()
    try:
      elapsed = await connect_once(args)
      attempts.append(Attempt(number, elapsed))
      print(f"[{number:>{len(str(args.count))}}/{args.count}] connected in {elapsed * 1000:.0f} ms")
    except Exception as e:
      elapsed = time.monotonic() - started
      attempts.append(Attempt(number, elapsed, f"{type(e).__name__}: {e}"))
      print(f"[{number:>{len(str(args.count))}}/{args.count}] FAILED after {elapsed:.2f} s: {e}")

    if number != args.count and args.delay:
      await asyncio.sleep(args.delay)

  successful = [attempt.elapsed for attempt in attempts if attempt.error is None]
  failures = [attempt for attempt in attempts if attempt.error is not None]
  print(f"\nResult: {len(successful)}/{args.count} connections succeeded")
  if successful:
    stats = [
      f"min {min(successful) * 1000:.0f} ms",
      f"mean {statistics.fmean(successful) * 1000:.0f} ms",
      f"p50 {percentile(successful, 0.50) * 1000:.0f} ms",
      f"p95 {percentile(successful, 0.95) * 1000:.0f} ms",
      f"max {max(successful) * 1000:.0f} ms",
    ]
    print(f"Connection time: {', '.join(stats)}")
  if failures:
    print("Failures:")
    for attempt in failures:
      print(f"  #{attempt.number}: {attempt.error}")
  return int(bool(failures))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="localhost", help="Host running webrtcd (default: localhost)")
  parser.add_argument("--port", type=int, default=5001, help="webrtcd port (default: 5001)")
  parser.add_argument("--count", type=int, default=100, help="Number of connection attempts (default: 100)")
  parser.add_argument("--camera", choices=("driver", "wideRoad", "road"), default="driver")
  parser.add_argument("--timeout", type=float, default=15, help="Timeout for each setup phase in seconds (default: 15)")
  parser.add_argument("--hold", type=float, default=0.1, help="Seconds to hold each connection (default: 0.1)")
  parser.add_argument("--delay", type=float, default=0.1, help="Seconds between attempts (default: 0.1)")
  parser.add_argument("--messaging", action="store_true", help="Also negotiate the cereal messaging data channel")
  args = parser.parse_args()
  if args.count < 1:
    parser.error("--count must be at least 1")
  if args.timeout <= 0 or args.hold < 0 or args.delay < 0:
    parser.error("--timeout must be positive; --hold and --delay cannot be negative")
  return args


def main() -> None:
  try:
    raise SystemExit(asyncio.run(run(parse_args())))
  except KeyboardInterrupt:
    print("\nInterrupted")
    raise SystemExit(130) from None


if __name__ == "__main__":
  main()
