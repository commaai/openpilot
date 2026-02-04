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

# Transport context — set before dispatch by ble.py / athenad.py
_current_transport: str = "websocket"


def set_transport(transport: str) -> None:
  global _current_transport
  _current_transport = transport


def get_transport() -> str:
  return _current_transport


# Params in saveParams() that require BLE to modify
BLE_ONLY_PARAMS: set[str] = {
  "GithubUsername",
  "GithubSshKeys",
  "DoReboot",
  "DoShutdown",
}


def ble_only(fn):
  def wrapper(*args, **kwargs):
    if _current_transport != "ble":
      raise Exception(f"{fn.__name__} requires bluetooth")
    return fn(*args, **kwargs)
  wrapper.__name__ = fn.__name__
  dispatcher.add_method(wrapper)
  return wrapper


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
def getAllParams() -> dict[str, str | bool | int | float | None]:
  from openpilot.common.params_pyx import ParamKeyType
  import json

  available_keys: list[str] = [k.decode('utf-8') for k in Params().all_keys()]
  result: dict[str, str | bool | int | float | None] = {}
  params = Params()

  for key in available_keys:
    value = params.get(key)
    if value is None:
      result[key] = None
      continue

    key_type = params.get_type(key)
    if key_type == ParamKeyType.BYTES:
      continue
    elif key_type == ParamKeyType.BOOL:
      result[key] = bool(value) if isinstance(value, bool) else value in (b'1', b'true', b'True', '1', 'true', 'True')
    elif key_type == ParamKeyType.INT:
      result[key] = int(value) if isinstance(value, int) else int(value.decode('utf-8') if isinstance(value, bytes) else value)
    elif key_type == ParamKeyType.FLOAT:
      result[key] = float(value) if isinstance(value, float) else float(value.decode('utf-8') if isinstance(value, bytes) else value)
    elif key_type == ParamKeyType.TIME:
      result[key] = value.timestamp()
    elif key_type == ParamKeyType.JSON:
      if isinstance(value, (dict, list)):
        result[key] = value
      else:
        result[key] = json.loads(value.decode('utf-8') if isinstance(value, bytes) else value)
    else:
      result[key] = value.decode('utf-8') if isinstance(value, bytes) else str(value)

  return result


@dispatcher.add_method
def saveParams(params_to_update: dict[str, str | bool | int | float | dict | list | None]) -> dict[str, str]:
  import json
  from openpilot.common.params_pyx import ParamKeyType
  from openpilot.common.swaglog import cloudlog

  params = Params()
  results = {}

  for key, value in params_to_update.items():
    if key in BLE_ONLY_PARAMS and get_transport() != "ble":
      cloudlog.warning(f"saveParams.blocked: '{key}' requires bluetooth")
      results[key] = "error: requires bluetooth"
      continue

    try:
      if value is None:
        params.remove(key)
        results[key] = "ok: removed"
        continue

      key_type = params.get_type(key)
      if key_type == ParamKeyType.BYTES:
        results[key] = "error: bytes not supported"
        continue
      elif key_type == ParamKeyType.BOOL:
        params.put(key, bool(value))
      elif key_type == ParamKeyType.INT:
        params.put(key, int(value))
      elif key_type == ParamKeyType.FLOAT or key_type == ParamKeyType.TIME:
        params.put(key, float(value))
      elif key_type == ParamKeyType.JSON:
        params.put(key, json.dumps(value) if isinstance(value, (dict, list)) else str(value))
      else:
        params.put(key, str(value))

      results[key] = "ok"
    except Exception as e:
      results[key] = f"error: {e}"

  return results


_wifi_manager = None


def _get_wifi_manager():
  global _wifi_manager
  if _wifi_manager is None:
    from openpilot.system.ui.lib.wifi_manager import WifiManager

    _wifi_manager = WifiManager()
  return _wifi_manager


@ble_only
def getWifiNetworks() -> list[dict]:
  wm = _get_wifi_manager()
  return [
    {
      "ssid": n.ssid,
      "strength": n.strength,
      "security": n.security_type.name.lower(),
      "connected": n.is_connected,
      "saved": n.is_saved,
    }
    for n in wm._networks
  ]


@ble_only
def connectWifi(ssid: str, password: str = "") -> dict[str, str]:
  wm = _get_wifi_manager()
  wm.connect_to_network(ssid, password)
  return {"status": "connecting"}


@ble_only
def forgetWifi(ssid: str) -> dict[str, str]:
  wm = _get_wifi_manager()
  wm.forget_connection(ssid, block=True)
  return {"status": "ok"}


@ble_only
def setTethering(enabled: bool) -> dict[str, str]:
  wm = _get_wifi_manager()
  wm.set_tethering_active(enabled)
  return {"status": "ok"}


@ble_only
def setTetheringPassword(password: str) -> dict[str, str]:
  if len(password) < 8:
    raise Exception("Password must be at least 8 characters")
  wm = _get_wifi_manager()
  wm.set_tethering_password(password)
  return {"status": "ok"}


@ble_only
def getNetworkStatus() -> dict:
  wm = _get_wifi_manager()
  return {
    "ip_address": wm.ipv4_address,
    "tethering_active": wm.is_tethering_active(),
    "tethering_password": wm.tethering_password,
    "metered": int(wm.current_network_metered),
  }


@ble_only
def blePair(code: str, dongleId: str) -> dict[str, str]:
  """Pair a BLE client using pairing code and return access token"""
  from openpilot.system.athena.ble import set_ble_token

  pairing_code = Params().get("BlePairingCode")
  if not pairing_code:
    raise Exception("Pairing mode not active")

  if code != pairing_code:
    raise Exception("Invalid pairing code")

  # Verify dongleId matches
  device_dongle_id = Params().get("DongleId")
  if dongleId != device_dongle_id:
    raise Exception("Wrong device")

  token = set_ble_token()
  return {"token": token}


@ble_only
def bleRevoke() -> dict[str, str]:
  """Revoke the BLE token, disconnecting any paired device."""
  from openpilot.system.athena.ble import clear_ble_token

  clear_ble_token()
  return {"status": "ok"}


@dispatcher.add_method
def webrtc(sdp: str, cameras: list[str], bridge_services_in: list[str], bridge_services_out: list[str]):
  from openpilot.common.swaglog import cloudlog

  if not Params().get_bool("EnableWebRTC"):
    raise Exception("EnableWebRTC is disabled")
  try:
    from openpilot.system.webrtc.session_manager import create_session

    return create_session(sdp, cameras, bridge_services_in, bridge_services_out)
  except Exception as e:
    cloudlog.exception("athena.webrtc.exception")
    return {"error": str(e)}
