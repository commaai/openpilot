"""
Shared RPC methods for Athena

These methods are shared between athenad (WebSocket) and ble_server (BLE).
Methods here should have minimal dependencies to work with system python.
"""
from __future__ import annotations

import base64
import os
import subprocess
from typing import cast

from jsonrpc import dispatcher

# Parameters that should never be remotely modified
BLOCKED_PARAMS = {
  "GithubUsername",  # Could grant SSH access
  "GithubSshKeys",   # Direct SSH key injection
}


def get_params():
  """Lazy import of Params to avoid circular imports"""
  try:
    from openpilot.common.params import Params
    return Params()
  except ImportError:
    return None


def get_hardware():
  """Lazy import of HARDWARE"""
  try:
    from openpilot.system.hardware import HARDWARE
    return HARDWARE
  except ImportError:
    return None


@dispatcher.add_method
def echo(**kwargs):
  """Echo back params - test method"""
  return kwargs


@dispatcher.add_method
def getDongleId() -> str:
  """Get device dongle ID"""
  try:
    with open("/data/params/d/DongleId", "r") as f:
      return f.read().strip()
  except:
    return "unknown"


@dispatcher.add_method
def getVersion() -> dict[str, str]:
  """Get openpilot version info"""
  try:
    from openpilot.system.version import get_build_metadata
    build_metadata = get_build_metadata()
    return {
      "version": build_metadata.openpilot.version,
      "remote": build_metadata.openpilot.git_normalized_origin,
      "branch": build_metadata.channel,
      "commit": build_metadata.openpilot.git_commit,
    }
  except ImportError:
    # Fallback for system python without full openpilot
    version = "unknown"
    git_commit = "unknown"
    git_branch = "unknown"
    git_remote = "unknown"

    try:
      git_commit = subprocess.check_output(
        ["git", "-C", "/data/openpilot", "rev-parse", "HEAD"],
        timeout=5
      ).decode().strip()[:12]
      git_branch = subprocess.check_output(
        ["git", "-C", "/data/openpilot", "rev-parse", "--abbrev-ref", "HEAD"],
        timeout=5
      ).decode().strip()
      git_remote = subprocess.check_output(
        ["git", "-C", "/data/openpilot", "config", "--get", "remote.origin.url"],
        timeout=5
      ).decode().strip()
    except:
      pass

    return {
      "version": version,
      "commit": git_commit,
      "branch": git_branch,
      "remote": git_remote,
    }


@dispatcher.add_method
def getGithubUsername() -> str:
  """Get GitHub username from params"""
  try:
    with open("/data/params/d/GithubUsername", "r") as f:
      return f.read().strip()
  except FileNotFoundError:
    return ""
  except:
    return ""


@dispatcher.add_method
def getSshAuthorizedKeys() -> str:
  """Get SSH authorized keys"""
  try:
    with open("/data/params/d/GithubSshKeys", "r") as f:
      return f.read()
  except:
    return ""


@dispatcher.add_method
def getNetworkType() -> str:
  """Get network connection type"""
  hw = get_hardware()
  if hw:
    try:
      network_type = hw.get_network_type()
      # Convert enum to string
      return str(network_type).split('.')[-1]
    except:
      pass

  # Fallback
  try:
    result = subprocess.run(
      ["nmcli", "-t", "-f", "TYPE,STATE", "connection", "show", "--active"],
      capture_output=True, text=True, timeout=5
    )
    for line in result.stdout.strip().split("\n"):
      if "wifi" in line.lower() and "activated" in line.lower():
        return "wifi"
      if "ethernet" in line.lower():
        return "ethernet"
      if "gsm" in line.lower():
        return "cell"
    return "none"
  except:
    return "unknown"


@dispatcher.add_method
def getSimInfo() -> dict:
  """Get SIM card info"""
  hw = get_hardware()
  if hw:
    try:
      return hw.get_sim_info()
    except:
      pass
  return {"error": "not available"}


@dispatcher.add_method
def getNetworkMetered() -> bool:
  """Check if network is metered"""
  hw = get_hardware()
  if hw:
    try:
      network_type = hw.get_network_type()
      return hw.get_network_metered(network_type)
    except:
      pass
  return False


@dispatcher.add_method
def getNetworks() -> dict:
  """Get available networks"""
  hw = get_hardware()
  if hw:
    try:
      return hw.get_networks()
    except:
      pass
  return {}


@dispatcher.add_method
def reboot() -> dict:
  """Reboot the device"""
  try:
    subprocess.Popen(["sudo", "reboot"], start_new_session=True)
    return {"success": True}
  except Exception as e:
    return {"error": str(e)}


@dispatcher.add_method
def shutdown() -> dict:
  """Shutdown the device"""
  try:
    subprocess.Popen(["sudo", "shutdown", "now"], start_new_session=True)
    return {"success": True}
  except Exception as e:
    return {"error": str(e)}


@dispatcher.add_method
def getParam(key: str) -> dict:
  """Get a specific param value"""
  path = f"/data/params/d/{key}"
  try:
    with open(path, "rb") as f:
      value = f.read()
      try:
        return {"value": value.decode("utf-8")}
      except:
        return {"value_base64": base64.b64encode(value).decode()}
  except FileNotFoundError:
    return {"error": "not found"}
  except Exception as e:
    return {"error": str(e)}


@dispatcher.add_method
def setParam(key: str, value: str) -> dict:
  """Set a param value"""
  if key in BLOCKED_PARAMS:
    return {"error": "blocked"}

  path = f"/data/params/d/{key}"
  try:
    with open(path, "w") as f:
      f.write(value)
    return {"success": True}
  except Exception as e:
    return {"error": str(e)}


@dispatcher.add_method
def listDataDirectory(prefix: str = "") -> list[str]:
  """List files in the data directory"""
  log_root = "/data/media/realdata"
  try:
    result = []
    for root, dirs, files in os.walk(log_root):
      for f in files:
        rel_path = os.path.relpath(os.path.join(root, f), log_root)
        if rel_path.startswith(prefix):
          result.append(rel_path)
      if len(result) > 1000:
        break
    return result[:1000]
  except Exception as e:
    return []


@dispatcher.add_method
def getAllParams() -> list[dict]:
  """Get all params with their values"""
  params = get_params()
  if not params:
    # Fallback - read directly from filesystem
    result = []
    params_dir = "/data/params/d"
    try:
      for name in os.listdir(params_dir):
        path = os.path.join(params_dir, name)
        if os.path.isfile(path):
          try:
            with open(path, "rb") as f:
              value = f.read()
              result.append({
                "key": name,
                "value": base64.b64encode(value).decode('utf-8')
              })
          except:
            result.append({"key": name, "value": None})
    except:
      pass
    return result

  # Use Params class if available
  try:
    if not params.get_bool("EnableRemoteParams"):
      raise Exception("EnableRemoteParams is disabled")

    available_keys = [k.decode('utf-8') for k in params.all_keys()]
    params_list = []

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
  except Exception as e:
    return [{"error": str(e)}]


@dispatcher.add_method
def saveParams(params_to_update: dict[str, str | None]) -> dict[str, str]:
  """Save multiple params at once"""
  p = get_params()
  if not p:
    return {"error": "Params not available"}

  try:
    if not p.get_bool("EnableRemoteParams"):
      raise Exception("EnableRemoteParams is disabled")
  except:
    return {"error": "EnableRemoteParams is disabled"}

  results = {}
  for key, value in params_to_update.items():
    if key in BLOCKED_PARAMS:
      results[key] = "error: blocked"
      continue

    try:
      if value is None or value == "":
        p.remove(key)
        results[key] = "ok: removed"
        continue

      decoded_value = base64.b64decode(value)
      decoded_str = decoded_value.decode('utf-8')

      try:
        from openpilot.common.params_pyx import ParamKeyType
        key_type = p.get_type(key)
        if key_type == ParamKeyType.BOOL:
          typed_value = decoded_str in ('1', 'true', 'True')
        elif key_type == ParamKeyType.INT:
          typed_value = int(decoded_str)
        elif key_type == ParamKeyType.FLOAT:
          typed_value = float(decoded_str)
        else:
          typed_value = decoded_str
      except:
        typed_value = decoded_str

      p.put(key, typed_value)
      results[key] = f"ok: {decoded_str}"
    except Exception as e:
      results[key] = f"error: {e}"

  return results
