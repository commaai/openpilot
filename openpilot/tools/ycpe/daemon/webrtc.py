"""Signaling for /live/local: forwards the page's offer to the webrtcd on this PC (openpilot, a sim, tools/replay),
which listens on 127.0.0.1 only. One fixed upstream and one endpoint: never an open relay."""
import json

from urllib3 import Timeout

from openpilot.tools.ycpe.daemon.athena import relay
from openpilot.tools.ycpe.daemon.core import HTTPError, config, route

MAX_BODY = 256 << 10  # an offer is a few KB
MAX_RESPONSE = 256 << 10
TIMEOUT = Timeout(connect=5.0, read=40.0)  # webrtcd gives up building an answer after 30 s


@route("POST", "/local/webrtc/stream", max_body=MAX_BODY)
def stream(req):
  if not isinstance(offer := req.json(), dict) or not isinstance(offer.get("sdp"), str):
    raise HTTPError(400, "needs an offer")
  host = f"127.0.0.1:{config.webrtcd_port}"
  return relay("POST", f"http://{host}/stream", f"webrtcd on {host}", json.dumps(offer).encode(), timeout=TIMEOUT, limit=MAX_RESPONSE)
