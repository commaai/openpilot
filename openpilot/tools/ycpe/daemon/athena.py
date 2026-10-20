"""Relays the page's JSON-RPC calls to athena, which only answers CORS for connect's origins. One fixed upstream, the
few methods the app calls, and the page's own Authorization: never an open relay, and it adds no credentials."""
import json
import re

from urllib3 import Timeout
from urllib3.exceptions import HTTPError as UpstreamError

from openpilot.tools.lib.url_file import URLFile
from openpilot.tools.ycpe.daemon.core import HTTPError, Response, module, route

ATHENA = "https://athena.comma.ai"
METHODS = {"listUploadQueue", "cancelUpload", "uploadFilesToUrls", "getNetworkMetered"}
DONGLE_RE = re.compile(r"[0-9a-f]{16}")
MAX_BODY = 1 << 20  # a few hundred files_data entries
MAX_RESPONSE = 4 << 20
TIMEOUT = Timeout(connect=5.0, read=30.0)

module("athena", "1", methods=sorted(METHODS))


def rpc_body(body: bytes) -> bytes:
  """the call, rebuilt from only the JSON-RPC fields, if it's a single call to an allowed method"""
  try:
    call = json.loads(body)
  except ValueError as e:
    raise HTTPError(400, "bad json") from e
  if not isinstance(call, dict) or call.get("method") not in METHODS:
    raise HTTPError(403, "method not allowed")
  out = {"jsonrpc": "2.0", "id": call.get("id", 0), "method": call["method"]}
  for k in ("params", "expiry"):
    if call.get(k) is not None:
      out[k] = call[k]
  return json.dumps(out).encode()


@route("POST", "/local/athena/{dongle}", max_body=MAX_BODY)
def athena(req):
  dongle = req.params["dongle"]
  if not DONGLE_RE.fullmatch(dongle):
    raise HTTPError(400, "bad dongle id")
  body = rpc_body(req.body)
  authorization = req.headers.get("authorization", "")
  if not re.fullmatch(r"JWT [\w.-]+", authorization):
    raise HTTPError(401, "needs the page's JWT")
  try:
    r = URLFile.pool_manager().request("POST", f"{ATHENA}/{dongle}", body=body, preload_content=False, redirect=False, retries=False,
                                       timeout=TIMEOUT, headers={"Authorization": authorization, "Content-Type": "application/json"})
    try:
      data = r.read(MAX_RESPONSE + 1)
    finally:
      r.release_conn()
  except UpstreamError as e:
    raise HTTPError(502, f"athena: {type(e).__name__}") from e
  if len(data) > MAX_RESPONSE:
    raise HTTPError(502, "athena answer too large")
  return Response(data, r.status, "application/json")
