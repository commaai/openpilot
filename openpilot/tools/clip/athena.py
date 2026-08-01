#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time
import uuid
from typing import Any

import requests

from openpilot.common.hardware.hw import Paths
from openpilot.tools.lib.auth_config import get_token


DEMO_ROUTE = "5beb9b58bd12b691/0000010a--a51155e496"
DEMO_START = 59.25
DEMO_END = 61.75
DEFAULT_ATHENA_HOST = "https://athena.comma.ai"


class AthenaError(RuntimeError):
  pass


class AthenaClient:
  def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
    raise NotImplementedError


class RemoteAthenaClient(AthenaClient):
  def __init__(self, dongle_id: str, token: str, host: str) -> None:
    self.url = f"{host.rstrip('/')}/{dongle_id}"
    self.session = requests.Session()
    self.session.headers.update(
      {
        "Authorization": f"JWT {token}",
        "Content-Type": "application/json",
        "User-Agent": "openpilot-clip",
      }
    )
    self.request_ids = itertools.count()

  def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
    payload: dict[str, Any] = {
      "jsonrpc": "2.0",
      "id": next(self.request_ids),
      "method": method,
    }
    if params is not None:
      payload["params"] = params

    response = self.session.post(self.url, json=payload, timeout=30)
    response.raise_for_status()
    return _jsonrpc_result(response.json())


class LocalAthenaClient(AthenaClient):
  def __init__(self) -> None:
    from openpilot.system.athena.athenad import dispatcher
    from openpilot.system.athena.rpc import handle

    self.dispatcher = dispatcher
    self.handle = handle
    self.request_ids = itertools.count()

  def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
    payload: dict[str, Any] = {
      "jsonrpc": "2.0",
      "id": next(self.request_ids),
      "method": method,
    }
    if params is not None:
      payload["params"] = params
    return _jsonrpc_result(json.loads(self.handle(payload, self.dispatcher)))


def _jsonrpc_result(response: dict[str, Any]) -> Any:
  if "error" in response:
    error = response["error"]
    if isinstance(error, dict):
      data = error.get("data")
      if isinstance(data, dict) and data.get("message"):
        raise AthenaError(str(data["message"]))
      raise AthenaError(str(error.get("message", error)))
    raise AthenaError(str(error))
  if "result" not in response:
    raise AthenaError(f"invalid Athena response: {response}")
  return response["result"]


def _dongle_id(route: str) -> str:
  parts = route.replace("|", "/").split("/")
  if len(parts) != 2 or len(parts[0]) != 16:
    raise AthenaError("remote routes must use dongle_id/route_id or dongle_id|route_id")
  return parts[0]


def _parse_headers(values: list[str]) -> dict[str, str]:
  headers = {}
  for value in values:
    if "=" not in value:
      raise AthenaError(f"upload header must be NAME=VALUE: {value}")
    name, header_value = value.split("=", 1)
    if not name:
      raise AthenaError(f"upload header name is empty: {value}")
    headers[name] = header_value
  return headers


def _print_progress(label: str, progress: float, last_percent: int) -> int:
  percent = round(progress * 100)
  if percent != last_percent:
    print(f"\r{label}: {percent:3d}%", end="", flush=True)
  return percent


def wait_for_clip(client: AthenaClient, clip_id: str, poll_interval: float) -> dict[str, Any]:
  last_percent = -1
  while True:
    state = client.call("getClipsState", {"routes": []})
    job = next((clip for clip in state["clips"] if clip["id"] == clip_id), None)
    if job is None:
      raise AthenaError("clip disappeared from the device")
    if job["status"] in ("queued", "encoding"):
      last_percent = _print_progress("Encoding", job["progress"], last_percent)
      time.sleep(poll_interval)
      continue

    if job["status"] == "failed":
      if last_percent >= 0:
        print()
      raise AthenaError(job["error"] or "video encoding failed")
    if job["status"] != "ready" or not job.get("fn"):
      if last_percent >= 0:
        print()
      raise AthenaError(f"invalid video clip status: {job}")
    _print_progress("Encoding", 1.0, last_percent)
    print()
    return job


def wait_for_upload(client: AthenaClient, upload_id: str, poll_interval: float) -> None:
  last_percent = -1
  while True:
    queue = client.call("listUploadQueue")
    item = next((item for item in queue if item["id"] == upload_id), None)
    if item is None:
      if last_percent >= 0:
        print()
      return
    last_percent = _print_progress("Uploading", item["progress"], last_percent)
    time.sleep(poll_interval)


def create_clip(
  client: AthenaClient, route: str, camera: str, start_time: float, end_time: float, bitrate: int, speedup: int, filename: str, poll_interval: float
) -> dict[str, Any]:
  response = client.call(
    "createClips",
    {
      "request_id": uuid.uuid4().hex,
      "route": route,
      "source_start_time": start_time,
      "source_end_time": end_time,
      "clips": [{"camera": camera, "bitrate": bitrate, "speedup": speedup, "filename": filename}],
    },
  )
  return wait_for_clip(client, response["clips"][0]["id"], poll_interval)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Create a camera clip on a device through Athena")
  parser.add_argument("route", nargs="?", help="dongle_id/route_id")
  parser.add_argument("start", nargs="?", type=float, help="start time in seconds from the beginning of the route")
  parser.add_argument("end", nargs="?", type=float, help="end time in seconds from the beginning of the route")
  parser.add_argument("--camera", choices=("fcamera", "ecamera", "dcamera"), default="fcamera")
  parser.add_argument("--bitrate", choices=(5, 8, 12), type=int, default=5, help="output bitrate in Mbps")
  parser.add_argument("--speedup", choices=(1, 2, 5, 10), type=int, default=1)
  parser.add_argument("--filename", default="")
  parser.add_argument("--demo", action="store_true", help=f"use {DEMO_ROUTE}, {DEMO_START}s-{DEMO_END}s")
  parser.add_argument("--local", action="store_true", help="call the local Athena dispatcher using this computer's realdata")
  parser.add_argument("-o", "--output", help="copy the completed clip here (local mode only)")
  parser.add_argument("--upload-url", help="enqueue the completed clip to this HTTP PUT URL")
  parser.add_argument("--upload-header", action="append", default=[], metavar="NAME=VALUE")
  parser.add_argument("--athena-host", default=os.getenv("ATHENA_URL_ROOT", DEFAULT_ATHENA_HOST))
  parser.add_argument("--token", default=get_token(), help=argparse.SUPPRESS)
  parser.add_argument("--poll-interval", type=float, default=1.0)
  args = parser.parse_args()

  if args.demo:
    args.route = args.route or DEMO_ROUTE
    args.start = DEMO_START if args.start is None else args.start
    args.end = DEMO_END if args.end is None else args.end
  if args.route is None or args.start is None or args.end is None:
    parser.error("route, start, and end are required (or use --demo)")
  if args.end <= args.start:
    parser.error("end must be greater than start")
  if args.poll_interval <= 0:
    parser.error("--poll-interval must be greater than zero")
  if args.output and not args.local:
    parser.error("--output is only available with --local")
  if args.upload_url and args.local:
    parser.error("--upload-url requires a running Athena upload handler; omit --local")
  if not args.local and not args.token:
    parser.error("not authenticated; run openpilot/tools/lib/auth.py")
  return args


def main() -> int:
  args = parse_args()
  try:
    client: AthenaClient
    if args.local:
      client = LocalAthenaClient()
    else:
      client = RemoteAthenaClient(_dongle_id(args.route), args.token, args.athena_host)

    job = create_clip(client, args.route, args.camera, args.start, args.end, args.bitrate, args.speedup, args.filename, args.poll_interval)
    fn = job["fn"]
    print(f"Created {fn}")

    if args.output:
      source = os.path.join(Paths.log_root(), fn)
      shutil.copy2(source, args.output)
      print(f"Copied to {os.path.abspath(args.output)}")

    if args.upload_url:
      response = client.call(
        "uploadFileToUrl",
        {
          "fn": fn,
          "url": args.upload_url,
          "headers": _parse_headers(args.upload_header),
        },
      )
      items = response.get("items", [])
      if not items:
        raise AthenaError(f"upload was not enqueued: {response}")
      wait_for_upload(client, items[0]["id"], args.poll_interval)
      print("Upload left the Athena queue")
    return 0
  except (AthenaError, OSError, requests.RequestException) as e:
    print(f"error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
