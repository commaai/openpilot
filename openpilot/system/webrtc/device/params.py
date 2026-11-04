import asyncio
import base64
import datetime
import json
from collections.abc import Callable
from typing import Any

from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.swaglog import cloudlog

# secrets that should never leave the device over a livestream session
BLOCKED_PARAMS = {
  "AccessToken",
  "AssistNowToken",
  "GithubSshKeys",
  "SecOCKey",
}


class ParamsProxy:
  """Read-only params over the webrtc datachannel.

  -> {"action": "get" | "subscribe" | "unsubscribe", "keys": [...], "id": 7}
  <- {"id": 7, "params": {...}, "errors": {...}}  reply
  <- {"params": {...}}                           subscribed param changed
  """
  update_interval = 0.5  # 2Hz

  def __init__(self, params: Params, send: Callable[[str], None]):
    self.params = params
    self.send = send
    # subscribed key -> last value sent to the client
    self.subscriptions: dict[str, Any] = {}

  @staticmethod
  def _serialize(value: Any) -> Any:
    if isinstance(value, bytes):
      return base64.b64encode(value).decode()
    if isinstance(value, datetime.datetime):
      return value.isoformat()
    return value

  @staticmethod
  def _keys(data: dict[str, Any]) -> list[str]:
    keys = data.get("keys", [])
    if isinstance(keys, str):
      keys = [keys]
    return [k for k in keys if isinstance(k, str)]

  def get(self, keys: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key in keys:
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

  def _send(self, data: dict[str, Any]):
    self.send(json.dumps({"type": "params", "data": data}))

  def handle(self, data: dict[str, Any]):
    action = data.get("action")
    keys = self._keys(data)
    if action == "get":
      response = self.get(keys)
    elif action == "subscribe":
      response = self.get(keys)
      self.subscriptions.update(response["params"])
    elif action == "unsubscribe":
      if "keys" not in data:
        self.subscriptions.clear()
      for key in keys:
        self.subscriptions.pop(key, None)
      response = {"params": {}}
    else:
      response = {"params": {}, "error": f"unknown action: {action}"}

    if "id" in data:
      response = {"id": data["id"], **response}
    self._send(response)

  def update(self):
    changed = {}
    for key, last in self.subscriptions.items():
      value = self._serialize(self.params.get(key))
      if value != last:
        changed[key] = value
    if changed:
      self.subscriptions.update(changed)
      self._send({"params": changed})

  async def run(self):
    while True:
      await asyncio.sleep(self.update_interval)
      try:
        self.update()
      except Exception:
        cloudlog.exception("webrtcd.params_proxy.update_failed")
