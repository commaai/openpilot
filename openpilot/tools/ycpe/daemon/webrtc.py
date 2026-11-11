"""Signaling for /live/local: forwards the page's offer to the webrtcd on this PC (openpilot, a sim, tools/replay),
which listens on 127.0.0.1 only. One fixed upstream and one endpoint: never an open relay."""
import json

from urllib3 import Timeout
from urllib3.exceptions import HTTPError as UpstreamError

from openpilot.tools.lib.url_file import URLFile
from openpilot.tools.ycpe.daemon.core import HTTPError, Response, config, route

MAX_BODY = 256 << 10  # an offer is a few KB
MAX_RESPONSE = 256 << 10
TIMEOUT = Timeout(connect=5.0, read=40.0)  # webrtcd gives up building an answer after 30 s


@route("POST", "/local/webrtc/stream", max_body=MAX_BODY)
def stream(req):
  if not isinstance(offer := req.json(), dict) or not isinstance(offer.get("sdp"), str):
    raise HTTPError(400, "needs an offer")
  name = f"webrtcd on 127.0.0.1:{config.webrtcd_port}"
  try:
    r = URLFile.pool_manager().request("POST", f"http://127.0.0.1:{config.webrtcd_port}/stream", body=json.dumps(offer).encode(),
                                       headers={"Content-Type": "application/json"}, preload_content=False, redirect=False,
                                       retries=False, timeout=TIMEOUT)
    try:
      data = r.read(MAX_RESPONSE + 1)
    finally:
      r.release_conn()
  except UpstreamError as e:
    raise HTTPError(502, f"{name}: {type(e).__name__}") from e
  if len(data) > MAX_RESPONSE:
    raise HTTPError(502, f"{name} answer too large")
  return Response(data, r.status, "application/json")
