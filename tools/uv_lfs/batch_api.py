"""LFS batch API HTTP client for download and upload."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from openpilot.tools.uv_lfs.config import lfs_push_url, lfs_url
from openpilot.tools.uv_lfs.storage import has_object, read_object, store_object

BATCH_SIZE = 100
DOWNLOAD_WORKERS = 8


def _batch_request(oids_sizes: list[tuple[str, int]], operation: str, url: str | None = None, headers: dict | None = None) -> dict:
  """Send a batch API request to the LFS server."""
  if url is None:
    url = lfs_url()
  batch_url = f"{url}/objects/batch"

  req_headers = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
  }
  if headers:
    req_headers.update(headers)

  data = {
    "operation": operation,
    "transfers": ["basic"],
    "objects": [{"oid": oid, "size": size} for oid, size in oids_sizes],
    "hash_algo": "sha256",
  }

  resp = requests.post(batch_url, json=data, headers=req_headers, timeout=30)
  resp.raise_for_status()
  return resp.json()


def _download_one(oid: str, href: str, dl_headers: dict) -> str | None:
  """Download a single object. Returns oid on success, None on failure."""
  try:
    resp = requests.get(href, headers=dl_headers, timeout=120)
    resp.raise_for_status()
    store_object(oid, resp.content)
    return oid
  except Exception as e:
    print(f"  error downloading {oid[:12]}: {e}", file=sys.stderr)
    return None


def download_objects(oids_sizes: list[tuple[str, int]], progress: bool = True) -> int:
  """Download LFS objects that aren't in the local cache. Returns count downloaded."""
  # filter out already-cached objects
  needed = [(oid, size) for oid, size in oids_sizes if not has_object(oid)]
  if not needed:
    return 0

  downloaded = 0
  total = len(needed)

  # collect all download URLs via batch API, then fetch in parallel
  to_fetch: list[tuple[str, str, dict]] = []  # (oid, href, headers)
  for i in range(0, len(needed), BATCH_SIZE):
    batch = needed[i:i + BATCH_SIZE]
    result = _batch_request(batch, "download")

    for obj in result.get("objects", []):
      oid = obj["oid"]
      if "error" in obj:
        print(f"  error for {oid}: {obj['error']}", file=sys.stderr)
        continue
      actions = obj.get("actions", {})
      dl = actions.get("download")
      if dl is None:
        continue
      to_fetch.append((oid, dl["href"], dl.get("header", {})))

  # parallel download
  with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
    futures = {pool.submit(_download_one, oid, href, hdrs): oid for oid, href, hdrs in to_fetch}
    for future in as_completed(futures):
      if future.result() is not None:
        downloaded += 1
        if progress:
          print(f"\r  downloading: {downloaded}/{total}", end="", flush=True)

  if progress and downloaded > 0:
    print()
  return downloaded


def upload_objects(oids_sizes: list[tuple[str, int]]) -> int:
  """Upload LFS objects to the server via SSH-authenticated batch. Returns count uploaded."""
  if not oids_sizes:
    return 0

  push_url = lfs_push_url()
  auth_header = _ssh_auth(push_url)

  uploaded = 0
  for i in range(0, len(oids_sizes), BATCH_SIZE):
    batch = oids_sizes[i:i + BATCH_SIZE]
    result = _batch_request(batch, "upload", url=lfs_url(), headers=auth_header)

    for obj in result.get("objects", []):
      oid = obj["oid"]
      actions = obj.get("actions", {})
      ul = actions.get("upload")
      if ul is None:
        # server already has it
        continue

      data = read_object(oid)
      href = ul["href"]
      ul_headers = ul.get("header", {})
      ul_headers["Content-Type"] = "application/octet-stream"
      resp = requests.put(href, data=data, headers=ul_headers, timeout=120)
      resp.raise_for_status()

      # verify action if present
      verify = actions.get("verify")
      if verify:
        v_headers = verify.get("header", {})
        v_headers["Content-Type"] = "application/vnd.git-lfs+json"
        v_data = json.dumps({"oid": oid, "size": len(data)})
        resp = requests.post(verify["href"], data=v_data, headers=v_headers, timeout=30)
        resp.raise_for_status()

      uploaded += 1
      print(f"  uploaded {uploaded}/{len(oids_sizes)}")

  return uploaded


def _ssh_auth(push_url: str) -> dict:
  """Get auth headers via SSH for push operations."""
  # parse ssh://git@gitlab.com/commaai/openpilot-lfs.git
  if not push_url.startswith("ssh://"):
    return {}

  # extract host and path
  url_part = push_url[len("ssh://"):]
  host, _, path = url_part.partition("/")

  try:
    result = subprocess.check_output(
      ["ssh", "-o", "StrictHostKeyChecking=accept-new", host,
       "git-lfs-authenticate", f"/{path}", "upload"],
      text=True, timeout=30,
    )
    auth = json.loads(result)
    return auth.get("header", {})
  except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired):
    return {}
