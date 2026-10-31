"""Relays the page's JSON-RPC calls to athena, which only answers CORS for connect's origins. One fixed upstream, the
few methods the app calls, and the page's own Authorization: never an open relay, and it adds no credentials."""
import json
import re

from urllib3 import Timeout
from urllib3.exceptions import HTTPError as UpstreamError

from openpilot.tools.lib.helpers import RE
from openpilot.tools.lib.url_file import URLFile
from openpilot.tools.ycpe.daemon.core import HTTPError, Response, route

ATHENA = "https://athena.comma.ai"
METHODS = {"listUploadQueue", "cancelUpload", "uploadFilesToUrls", "getNetworkMetered", "startStream", "getMessage", "takeSnapshot",
           "setRouteViewed", "getVersion", "getClipState", "createClip", "deleteClip", "getClipChunk"}
MAX_BODY = 1 << 20  # a few hundred files_data entries
MAX_RESPONSE = 4 << 20
TIMEOUT = Timeout(connect=5.0, read=30.0)


def rpc_body(call) -> bytes:
  """the call, rebuilt from only the JSON-RPC fields, if it's a single call to an allowed method"""
  if not isinstance(call, dict) or call.get("method") not in METHODS:
    raise HTTPError(403, "method not allowed")
  out = {"jsonrpc": "2.0", "id": call.get("id", 0), "method": call["method"]}
  for k in ("params", "expiry"):
    if call.get(k) is not None:
      out[k] = call[k]
  return json.dumps(out).encode()


def relay(method: str, url: str, name: str, body: bytes | None = None, headers=None, timeout=TIMEOUT, limit=MAX_RESPONSE) -> Response:
  """the call to a fixed url with only the given headers (and a JSON content type with a body), its JSON answer
  size-limited"""
  headers = (headers or {}) | ({"Content-Type": "application/json"} if body is not None else {})
  try:
    r = URLFile.pool_manager().request(method, url, body=body, preload_content=False, redirect=False, retries=False, timeout=timeout, headers=headers)
    try:
      data = r.read(limit + 1)
    finally:
      r.release_conn()
  except UpstreamError as e:
    raise HTTPError(502, f"{name}: {type(e).__name__}") from e
  if len(data) > limit:
    raise HTTPError(502, f"{name} answer too large")
  return Response(data, r.status, "application/json")


def forward(req, method: str, url: str, body: bytes | None = None, name: str = "athena") -> Response:
  """the call with the page's own JWT and nothing else of the page's"""
  authorization = req.headers.get("authorization", "")
  if not re.fullmatch(r"JWT [\w.-]+", authorization):
    raise HTTPError(401, "needs the page's JWT")
  return relay(method, url, name, body, {"Authorization": authorization})


@route("POST", "/local/athena/{dongle}", max_body=MAX_BODY)
def athena(req):
  dongle = req.params["dongle"]
  if not re.fullmatch(RE.DONGLE_ID, dongle):
    raise HTTPError(400, "bad dongle id")
  return forward(req, "POST", f"{ATHENA}/{dongle}", rpc_body(req.json()))
