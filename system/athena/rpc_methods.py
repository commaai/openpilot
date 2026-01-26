"""
Shared RPC methods for Athena (WebSocket) and BLE server.
"""
from __future__ import annotations

import base64
import io
import os
from typing import cast

from jsonrpc import dispatcher

from openpilot.common.api import get_key_pair
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.system.version import get_build_metadata
from openpilot.system.hardware.hw import Paths

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST

# Parameters that should never be remotely modified
BLOCKED_PARAMS = {
  "GithubUsername",  # Could grant SSH access
  "GithubSshKeys",   # Direct SSH key injection
}

dispatcher["echo"] = lambda s: s


@dispatcher.add_method
def getMessage(service: str, timeout: int = 1000) -> dict:
  if service is None or service not in SERVICE_LIST:
    raise Exception("invalid service")
  socket = messaging.sub_sock(service, timeout=timeout)
  try:
    ret = messaging.recv_one(socket)
    if ret is None:
      raise TimeoutError
    return cast(dict, ret.to_dict())
  finally:
    del socket


@dispatcher.add_method
def getVersion() -> dict[str, str]:
  build_metadata = get_build_metadata()
  return {
    "version": build_metadata.openpilot.version,
    "remote": build_metadata.openpilot.git_normalized_origin,
    "branch": build_metadata.channel,
    "commit": build_metadata.openpilot.git_commit,
  }


def scan_dir(path: str, prefix: str) -> list[str]:
  files = []
  with os.scandir(path) as i:
    for e in i:
      rel_path = os.path.relpath(e.path, Paths.log_root())
      if e.is_dir(follow_symlinks=False):
        rel_path = os.path.join(rel_path, '')
        if rel_path.startswith(prefix) or prefix.startswith(rel_path):
          files.extend(scan_dir(e.path, prefix))
      else:
        if rel_path.startswith(prefix):
          files.append(rel_path)
  return files


@dispatcher.add_method
def listDataDirectory(prefix='') -> list[str]:
  return scan_dir(Paths.log_root(), prefix)


@dispatcher.add_method
def setRouteViewed(route: str) -> dict[str, int | str]:
  params = Params()
  r = params.get("AthenadRecentlyViewedRoutes")
  routes = [] if r is None else r.split(",")
  routes.append(route)
  routes = list(dict.fromkeys(routes))
  params.put("AthenadRecentlyViewedRoutes", ",".join(routes[-10:]))
  return {"success": 1}


@dispatcher.add_method
def getPublicKey() -> str | None:
  _, _, public_key = get_key_pair()
  return public_key


@dispatcher.add_method
def getSshAuthorizedKeys() -> str:
  return cast(str, Params().get("GithubSshKeys") or "")


@dispatcher.add_method
def getGithubUsername() -> str:
  return cast(str, Params().get("GithubUsername") or "")


@dispatcher.add_method
def getSimInfo():
  return HARDWARE.get_sim_info()


@dispatcher.add_method
def getNetworkType():
  return HARDWARE.get_network_type()


@dispatcher.add_method
def getNetworkMetered() -> bool:
  network_type = HARDWARE.get_network_type()
  return HARDWARE.get_network_metered(network_type)


@dispatcher.add_method
def getNetworks():
  return HARDWARE.get_networks()


@dispatcher.add_method
def takeSnapshot() -> str | dict[str, str] | None:
  from openpilot.system.camerad.snapshot import jpeg_write, snapshot
  ret = snapshot()
  if ret is not None:
    def b64jpeg(x):
      if x is not None:
        f = io.BytesIO()
        jpeg_write(f, x)
        return base64.b64encode(f.getvalue()).decode("utf-8")
      return None
    return {'jpegBack': b64jpeg(ret[0]), 'jpegFront': b64jpeg(ret[1])}
  raise Exception("not available while camerad is started")


@dispatcher.add_method
def getAllParams() -> list[dict[str, str | bool | int | object | dict | None]]:
  if not Params().get_bool("EnableRemoteParams"):
    raise Exception("EnableRemoteParams is disabled")

  available_keys: list[str] = [k.decode('utf-8') for k in Params().all_keys()]
  params_list: list[dict[str, str | bool | int | object | dict | None]] = []
  params = Params()

  for key in available_keys:
    value = params.get(key)
    if value is not None and not isinstance(value, bytes):
      if isinstance(value, bool):
        value = b"1" if value else b"0"
      else:
        value = str(value).encode('utf-8')

    entry = {
      "key": key,
      "type": int(params.get_type(key).value),
      "value": base64.b64encode(value).decode('utf-8') if value else None,
    }
    params_list.append(entry)

  return params_list


@dispatcher.add_method
def saveParams(params_to_update: dict[str, str | None]) -> dict[str, str]:
  if not Params().get_bool("EnableRemoteParams"):
    raise Exception("EnableRemoteParams is disabled")

  from openpilot.common.params_pyx import ParamKeyType
  from openpilot.common.swaglog import cloudlog

  params = Params()
  results = {}

  for key, value in params_to_update.items():
    if key in BLOCKED_PARAMS:
      cloudlog.warning(f"athenad.saveParams.blocked: Attempted to modify blocked parameter '{key}'")
      results[key] = "error: blocked"
      continue

    try:
      if value is None or value == "":
        params.remove(key)
        results[key] = "ok: removed"
        continue

      decoded_value = base64.b64decode(value)
      decoded_str = decoded_value.decode('utf-8')

      key_type = params.get_type(key)
      if key_type == ParamKeyType.BOOL:
        typed_value = decoded_str in ('1', 'true', 'True')
      elif key_type == ParamKeyType.INT:
        typed_value = int(decoded_str)
      elif key_type == ParamKeyType.FLOAT:
        typed_value = float(decoded_str)
      else:
        typed_value = decoded_str

      params.put(key, typed_value)
      results[key] = f"ok: {decoded_str}"
    except Exception as e:
      results[key] = f"error: {e}"

  return results
