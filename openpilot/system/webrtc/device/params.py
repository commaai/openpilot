import base64
import datetime
import json
from typing import Any

from openpilot.common.params import Params, UnknownKeyName

# secrets that should never leave the device over a livestream session
BLOCKED_PARAMS = {
  "AccessToken",
  "AssistNowToken",
  "GithubSshKeys",
  "SecOCKey",
}


class ParamsProxy:
  """Read-only access to device params over the webrtc datachannel.

  request:  {"type": "paramsRequest", "data": {"keys": ["IsMetric", ...]}}
  response: {"type": "paramsResponse", "data": {"params": {"IsMetric": true, ...}, "errors": {...}}}

  Missing params are returned as null. BYTES params are base64 encoded, TIME params are ISO 8601 strings.
  """

  def __init__(self, params: Params):
    self.params = params

  @staticmethod
  def _serialize(value: Any) -> Any:
    if isinstance(value, bytes):
      return base64.b64encode(value).decode()
    if isinstance(value, datetime.datetime):
      return value.isoformat()
    return value

  def get(self, keys: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key in keys:
      if not isinstance(key, str):
        continue
      if key in BLOCKED_PARAMS:
        errors[key] = "blocked"
        continue
      try:
        values[key] = self._serialize(self.params.get(key))
      except UnknownKeyName:
        errors[key] = "unknown key"

    response: dict[str, Any] = {"params": values}
    if errors:
      response["errors"] = errors
    return response

  def handle_request(self, data: dict[str, Any]) -> str:
    keys = data.get("keys", [])
    if isinstance(keys, str):
      keys = [keys]
    return json.dumps({"type": "paramsResponse", "data": self.get(list(keys))})
