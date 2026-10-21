"""Signaling for /live/local: forwards the page's offer to the webrtcd on this PC (openpilot, a sim, tools/replay),
which listens on 127.0.0.1 only. One fixed upstream and one endpoint: never an open relay."""
import http.client
import json

from openpilot.tools.ycpe.daemon.core import HTTPError, Response, config, route

MAX_BODY = 256 << 10  # an offer is a few KB
MAX_RESPONSE = 256 << 10
TIMEOUT = 40  # webrtcd gives up building an answer after 30 s


@route("POST", "/local/webrtc/stream", max_body=MAX_BODY)
def stream(req):
  try:
    body = json.loads(req.body)
  except ValueError as e:
    raise HTTPError(400, "bad json") from e
  if not isinstance(body, dict) or not isinstance(body.get("sdp"), str):
    raise HTTPError(400, "needs an offer")
  c = http.client.HTTPConnection("127.0.0.1", config.webrtcd_port, timeout=TIMEOUT)
  try:
    c.request("POST", "/stream", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read(MAX_RESPONSE + 1)
  except OSError as e:
    raise HTTPError(502, f"no webrtcd on 127.0.0.1:{config.webrtcd_port} ({type(e).__name__})") from e
  finally:
    c.close()
  if len(data) > MAX_RESPONSE:
    raise HTTPError(502, "webrtcd answer too large")
  return Response(data, r.status, "application/json")
