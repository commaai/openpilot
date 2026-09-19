import time
import requests
from dataclasses import asdict, dataclass, field


WEBRTCD_PORT = 5001

@dataclass
class StreamRequestBody:
  sdp: str
  cameras: list[str]
  enabled: bool
  bridge_services_in: list[str] = field(default_factory=list)
  bridge_services_out: list[str] = field(default_factory=list)


def post_stream_request(body: StreamRequestBody) -> dict:
  t_start = time.monotonic()
  try:
    resp = requests.post(f"http://localhost:{WEBRTCD_PORT}/stream", json=asdict(body), timeout=10)
    t_end = time.monotonic()
    ret = resp.json()
    if resp.status_code >= 400 or ret.get("error"):
      raise RuntimeError(ret.get("message") or ret.get("error") or f"Livestream request failed: HTTP {resp.status_code}")
    if not isinstance(ret.get("sdp"), str) or not ret["sdp"].startswith("v=0"):
      raise RuntimeError("Livestream service returned an invalid SDP answer")
    ret["time"] = (t_end - t_start) * 1000
    return ret
  except requests.ConnectTimeout as e:
    raise Exception("device took too long to respond.") from e
  except requests.ConnectionError as e:
    raise Exception("turn car ignition off to use livestreaming.") from e


def wait_for_webrtcd(max_retries: float = 10) -> None:
  attempts = 0
  while attempts < max_retries:
    try:
      if requests.get(f"http://localhost:{WEBRTCD_PORT}/schema", timeout=1).ok:
        return
    except requests.ConnectionError:
      attempts += 1
      time.sleep(0.5)
  raise TimeoutError("livestreaming service did not initialize in time.")
