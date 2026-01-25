#!/usr/bin/env python3
"""
BLE GATT Server for Athena

Exposes athenad JSON-RPC methods over Bluetooth Low Energy.
This allows controlling the device without network connectivity.

Architecture:
- GATT Service with custom UUID for Athena
- RPC Request characteristic (Write) - client sends JSON-RPC requests
- RPC Response characteristic (Notify) - server sends responses
- Handles message chunking for BLE MTU limits

Usage:
  python3 ble_server.py

This script is standalone and doesn't require the full openpilot environment.
It communicates with the running athenad via internal dispatch.
"""
from __future__ import annotations

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
import json
import subprocess
import threading
import os
import sys
from gi.repository import GLib

# BLE GATT UUIDs
# Custom UUIDs for Asius/Athena service
ATHENA_SERVICE_UUID = "a51a5a10-0001-4c0d-b8e6-a51a5a100001"
RPC_REQUEST_CHAR_UUID = "a51a5a10-0002-4c0d-b8e6-a51a5a100001"
RPC_RESPONSE_CHAR_UUID = "a51a5a10-0003-4c0d-b8e6-a51a5a100001"

# BlueZ DBus constants
BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"

# BLE MTU - conservative value
BLE_MTU = 512


def log(msg: str):
  print(f"[BLE] {msg}", flush=True)


def get_dongle_id() -> str:
  """Get device dongle ID from params"""
  try:
    with open("/data/params/d/DongleId", "r") as f:
      return f.read().strip()
  except:
    return "unknown"


def get_device_name() -> str:
  """Get BLE device name"""
  dongle_id = get_dongle_id()
  return f"comma-{dongle_id[:8]}" if dongle_id else "comma-device"


# ============ Simple JSON-RPC handlers (standalone) ============

def handle_echo(params):
  return params


def handle_get_version():
  """Get openpilot version info"""
  try:
    version = "unknown"
    git_commit = "unknown"
    git_branch = "unknown"
    git_remote = "unknown"

    version_file = "/data/openpilot/common/version.h"
    if os.path.exists(version_file):
      with open(version_file) as f:
        for line in f:
          if "OPENPILOT_VERSION" in line:
            version = line.split('"')[1]

    # Try git info
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
  except Exception as e:
    return {"error": str(e)}


def handle_get_params():
  """Get all params"""
  params = {}
  params_dir = "/data/params/d"
  try:
    for name in os.listdir(params_dir):
      path = os.path.join(params_dir, name)
      if os.path.isfile(path):
        try:
          with open(path, "rb") as f:
            value = f.read()
            # Try to decode as string
            try:
              params[name] = value.decode("utf-8")
            except:
              # Return as base64 for binary data
              import base64
              params[name] = {"_base64": base64.b64encode(value).decode()}
        except:
          params[name] = None
  except Exception as e:
    return {"error": str(e)}
  return params


def handle_get_param(params):
  """Get a specific param"""
  key = params.get("key", "")
  path = f"/data/params/d/{key}"
  try:
    with open(path, "rb") as f:
      value = f.read()
      try:
        return {"value": value.decode("utf-8")}
      except:
        import base64
        return {"value_base64": base64.b64encode(value).decode()}
  except FileNotFoundError:
    return {"error": "not found"}
  except Exception as e:
    return {"error": str(e)}


def handle_set_param(params):
  """Set a param value"""
  key = params.get("key", "")
  value = params.get("value", "")

  # Security: block certain params
  blocked = {"GithubUsername", "GithubSshKeys"}
  if key in blocked:
    return {"error": "blocked"}

  path = f"/data/params/d/{key}"
  try:
    with open(path, "w") as f:
      f.write(value)
    return {"success": True}
  except Exception as e:
    return {"error": str(e)}


def handle_list_data_directory(params):
  """List files in data directory"""
  prefix = params.get("prefix", "")
  log_root = "/data/media/realdata"
  try:
    result = []
    for root, dirs, files in os.walk(log_root):
      for f in files:
        rel_path = os.path.relpath(os.path.join(root, f), log_root)
        if rel_path.startswith(prefix):
          result.append(rel_path)
      # Limit results
      if len(result) > 1000:
        break
    return result[:1000]
  except Exception as e:
    return {"error": str(e)}


def handle_get_network_type():
  """Get network type"""
  try:
    # Check if connected to wifi
    result = subprocess.run(
      ["nmcli", "-t", "-f", "TYPE,STATE", "connection", "show", "--active"],
      capture_output=True, text=True, timeout=5
    )
    for line in result.stdout.strip().split("\n"):
      if "wifi" in line.lower() and "activated" in line.lower():
        return "wifi"
      if "ethernet" in line.lower() and "activated" in line.lower():
        return "ethernet"
      if "gsm" in line.lower() or "cellular" in line.lower():
        return "cell"
    return "none"
  except:
    return "unknown"


def handle_reboot():
  """Reboot the device"""
  try:
    subprocess.Popen(["sudo", "reboot"], start_new_session=True)
    return {"success": True}
  except Exception as e:
    return {"error": str(e)}


def handle_shutdown():
  """Shutdown the device"""
  try:
    subprocess.Popen(["sudo", "shutdown", "now"], start_new_session=True)
    return {"success": True}
  except Exception as e:
    return {"error": str(e)}


def handle_get_github_username():
  """Get GitHub username param"""
  path = "/data/params/d/GithubUsername"
  try:
    with open(path, "r") as f:
      return {"username": f.read().strip()}
  except FileNotFoundError:
    return {"username": None}
  except Exception as e:
    return {"error": str(e)}


def handle_set_github_username(params):
  """Set GitHub username param"""
  username = params.get("username", "")
  path = "/data/params/d/GithubUsername"
  try:
    with open(path, "w") as f:
      f.write(username)
    return {"success": True}
  except Exception as e:
    return {"error": str(e)}


def handle_get_dongle_id():
  """Get dongle ID"""
  return {"dongleId": get_dongle_id()}


# RPC method dispatcher
RPC_METHODS = {
  "echo": lambda p: handle_echo(p),
  "getVersion": lambda p: handle_get_version(),
  "getAllParams": lambda p: handle_get_params(),
  "getParam": lambda p: handle_get_param(p),
  "setParam": lambda p: handle_set_param(p),
  "listDataDirectory": lambda p: handle_list_data_directory(p),
  "getNetworkType": lambda p: handle_get_network_type(),
  "getGithubUsername": lambda p: handle_get_github_username(),
  "setGithubUsername": lambda p: handle_set_github_username(p),
  "getDongleId": lambda p: handle_get_dongle_id(),
  "reboot": lambda p: handle_reboot(),
  "shutdown": lambda p: handle_shutdown(),
}


def handle_jsonrpc(message: str) -> str:
  """Handle a JSON-RPC request and return response"""
  try:
    request = json.loads(message)
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method not in RPC_METHODS:
      return json.dumps({
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": f"Method not found: {method}"},
        "id": req_id
      })

    result = RPC_METHODS[method](params)
    return json.dumps({
      "jsonrpc": "2.0",
      "result": result,
      "id": req_id
    })

  except json.JSONDecodeError as e:
    return json.dumps({
      "jsonrpc": "2.0",
      "error": {"code": -32700, "message": f"Parse error: {e}"},
      "id": None
    })
  except Exception as e:
    return json.dumps({
      "jsonrpc": "2.0",
      "error": {"code": -32603, "message": f"Internal error: {e}"},
      "id": None
    })


# ============ BLE GATT Implementation ============

class InvalidArgsException(dbus.exceptions.DBusException):
  _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


class NotSupportedException(dbus.exceptions.DBusException):
  _dbus_error_name = "org.bluez.Error.NotSupported"


class Application(dbus.service.Object):
  """GATT Application - container for services"""

  def __init__(self, bus):
    self.path = "/"
    self.services = []
    dbus.service.Object.__init__(self, bus, self.path)
    self.add_service(AthenaService(bus, 0))

  def get_path(self):
    return dbus.ObjectPath(self.path)

  def add_service(self, service):
    self.services.append(service)

  @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
  def GetManagedObjects(self):
    response = {}
    for service in self.services:
      response[service.get_path()] = service.get_properties()
      for chrc in service.characteristics:
        response[chrc.get_path()] = chrc.get_properties()
    return response


class Service(dbus.service.Object):
  """GATT Service base class"""

  PATH_BASE = "/org/bluez/athena"

  def __init__(self, bus, index, uuid, primary):
    self.path = f"{self.PATH_BASE}/service{index}"
    self.bus = bus
    self.uuid = uuid
    self.primary = primary
    self.characteristics = []
    dbus.service.Object.__init__(self, bus, self.path)

  def get_properties(self):
    return {
      GATT_SERVICE_IFACE: {
        "UUID": self.uuid,
        "Primary": self.primary,
        "Characteristics": dbus.Array(
          [c.get_path() for c in self.characteristics],
          signature="o"
        )
      }
    }

  def get_path(self):
    return dbus.ObjectPath(self.path)

  def add_characteristic(self, characteristic):
    self.characteristics.append(characteristic)


class Characteristic(dbus.service.Object):
  """GATT Characteristic base class"""

  def __init__(self, bus, index, uuid, flags, service):
    self.path = f"{service.path}/char{index}"
    self.bus = bus
    self.uuid = uuid
    self.service = service
    self.flags = flags
    dbus.service.Object.__init__(self, bus, self.path)

  def get_properties(self):
    return {
      GATT_CHRC_IFACE: {
        "Service": self.service.get_path(),
        "UUID": self.uuid,
        "Flags": self.flags,
      }
    }

  def get_path(self):
    return dbus.ObjectPath(self.path)

  @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
  def ReadValue(self, options):
    raise NotSupportedException()

  @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
  def WriteValue(self, value, options):
    raise NotSupportedException()

  @dbus.service.method(GATT_CHRC_IFACE)
  def StartNotify(self):
    raise NotSupportedException()

  @dbus.service.method(GATT_CHRC_IFACE)
  def StopNotify(self):
    raise NotSupportedException()

  @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
  def PropertiesChanged(self, interface, changed, invalidated):
    pass


class AthenaService(Service):
  """Athena GATT Service"""

  def __init__(self, bus, index):
    super().__init__(bus, index, ATHENA_SERVICE_UUID, True)
    self.response_char = RpcResponseCharacteristic(bus, 1, self)
    self.add_characteristic(RpcRequestCharacteristic(bus, 0, self, self.response_char))
    self.add_characteristic(self.response_char)


class RpcRequestCharacteristic(Characteristic):
  """
  RPC Request Characteristic - receives JSON-RPC requests from client
  Supports chunked writes for messages larger than MTU
  """

  def __init__(self, bus, index, service, response_char):
    super().__init__(bus, index, RPC_REQUEST_CHAR_UUID, ["write", "write-without-response"], service)
    self.response_char = response_char
    self.buffer = bytearray()
    self.lock = threading.Lock()

  def WriteValue(self, value, options):
    data = bytes(value)
    log(f"Received {len(data)} bytes")

    with self.lock:
      self.buffer.extend(data)

      # Try to parse as complete JSON
      try:
        message = self.buffer.decode("utf-8")
        # Check if it's complete JSON by trying to parse
        json.loads(message)

        # Process the complete message
        log(f"Processing request: {message[:100]}...")
        self.buffer = bytearray()

        # Handle in thread to not block BLE
        threading.Thread(
          target=self._handle_request,
          args=(message,),
          daemon=True
        ).start()

      except (json.JSONDecodeError, UnicodeDecodeError):
        # Incomplete message, wait for more chunks
        if len(self.buffer) > 65536:  # 64KB max message size
          log("Message too large, clearing buffer")
          self.buffer = bytearray()

  def _handle_request(self, message: str):
    try:
      response = handle_jsonrpc(message)
      self.response_char.send_response(response)
    except Exception as e:
      log(f"RPC handler error: {e}")
      error_response = json.dumps({"error": str(e), "jsonrpc": "2.0", "id": None})
      self.response_char.send_response(error_response)


class RpcResponseCharacteristic(Characteristic):
  """
  RPC Response Characteristic - sends JSON-RPC responses via notifications
  Handles chunking for responses larger than MTU
  """

  def __init__(self, bus, index, service):
    super().__init__(bus, index, RPC_RESPONSE_CHAR_UUID, ["notify"], service)
    self.notifying = False

  def send_response(self, response: str):
    if not self.notifying:
      log("Client not subscribed to notifications, can't send response")
      return

    data = response.encode("utf-8")
    log(f"Sending response: {len(data)} bytes")

    # Send in chunks
    for i in range(0, len(data), BLE_MTU):
      chunk = data[i:i + BLE_MTU]
      self._send_notification(chunk)

  def _send_notification(self, data: bytes):
    value = dbus.Array([dbus.Byte(b) for b in data], signature="y")
    self.PropertiesChanged(
      GATT_CHRC_IFACE,
      {"Value": value},
      []
    )

  def StartNotify(self):
    if self.notifying:
      return
    self.notifying = True
    log("Client subscribed to notifications")

  def StopNotify(self):
    if not self.notifying:
      return
    self.notifying = False
    log("Client unsubscribed from notifications")


class Advertisement(dbus.service.Object):
  """BLE Advertisement for discoverability"""

  PATH_BASE = "/org/bluez/athena"

  def __init__(self, bus, index):
    self.path = f"{self.PATH_BASE}/advertisement{index}"
    self.bus = bus
    self.ad_type = "peripheral"
    self.service_uuids = [ATHENA_SERVICE_UUID]
    self.local_name = get_device_name()
    self.includes = ["tx-power"]
    dbus.service.Object.__init__(self, bus, self.path)

  def get_properties(self):
    return {
      LE_ADVERTISEMENT_IFACE: {
        "Type": self.ad_type,
        "ServiceUUIDs": dbus.Array(self.service_uuids, signature="s"),
        "LocalName": dbus.String(self.local_name),
        "Includes": dbus.Array(self.includes, signature="s"),
      }
    }

  def get_path(self):
    return dbus.ObjectPath(self.path)

  @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
  def GetAll(self, interface):
    if interface != LE_ADVERTISEMENT_IFACE:
      raise InvalidArgsException()
    return self.get_properties()[LE_ADVERTISEMENT_IFACE]

  @dbus.service.method(LE_ADVERTISEMENT_IFACE, in_signature="", out_signature="")
  def Release(self):
    log("Advertisement released")


def find_adapter(bus):
  """Find the BlueZ adapter"""
  remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
  objects = remote_om.GetManagedObjects()

  for path, interfaces in objects.items():
    if GATT_MANAGER_IFACE in interfaces:
      return path

  return None


def init_bluetooth():
  """Initialize Bluetooth hardware if not already done"""
  import time

  # Check if hci0 already exists
  try:
    result = subprocess.run(["hciconfig", "hci0"], capture_output=True, timeout=5)
    if result.returncode == 0 and b"UP RUNNING" in result.stdout:
      log("Bluetooth already initialized")
      return True
  except:
    pass

  # Check if ttyHS1 exists (BT UART)
  if not os.path.exists("/dev/ttyHS1"):
    log("ERROR: /dev/ttyHS1 not found - kernel may not have BT support")
    return False

  # Kill any existing btattach
  subprocess.run(["pkill", "-f", "btattach"], capture_output=True)
  time.sleep(1)

  # Start btattach
  log("Starting btattach...")
  subprocess.Popen(
    ["btattach", "-B", "/dev/ttyHS1", "-S", "115200"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
  )

  # Wait for hci0 to come up
  for i in range(10):
    time.sleep(1)
    try:
      result = subprocess.run(["hciconfig", "hci0"], capture_output=True, timeout=5)
      if result.returncode == 0 and b"UP RUNNING" in result.stdout:
        log("Bluetooth initialized successfully")
        return True
    except:
      pass
    log(f"Waiting for Bluetooth... ({i+1}/10)")

  log("ERROR: Failed to initialize Bluetooth")
  return False


def main():
  log("Starting BLE GATT Server...")
  log(f"Device name: {get_device_name()}")
  log(f"Service UUID: {ATHENA_SERVICE_UUID}")

  # Initialize Bluetooth hardware
  if not init_bluetooth():
    return 1

  # Initialize DBus
  dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
  bus = dbus.SystemBus()

  # Find adapter
  adapter_path = find_adapter(bus)
  if not adapter_path:
    log("ERROR: No GATT adapter found. Is Bluetooth initialized?")
    return 1

  log(f"Using adapter: {adapter_path}")

  # Get manager interfaces
  service_manager = dbus.Interface(
    bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
    GATT_MANAGER_IFACE
  )

  ad_manager = dbus.Interface(
    bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
    LE_ADVERTISING_MANAGER_IFACE
  )

  # Create and register application
  app = Application(bus)

  def register_app_cb():
    log("GATT application registered successfully")

  def register_app_error_cb(error):
    log(f"Failed to register application: {error}")
    mainloop.quit()

  service_manager.RegisterApplication(
    app.get_path(),
    {},
    reply_handler=register_app_cb,
    error_handler=register_app_error_cb
  )

  # Create and register advertisement
  advertisement = Advertisement(bus, 0)

  def register_ad_cb():
    log("Advertisement registered - device is now discoverable")

  def register_ad_error_cb(error):
    log(f"Failed to register advertisement: {error}")

  ad_manager.RegisterAdvertisement(
    advertisement.get_path(),
    {},
    reply_handler=register_ad_cb,
    error_handler=register_ad_error_cb
  )

  log("Server running. Press Ctrl+C to stop.")

  # Run main loop
  mainloop = GLib.MainLoop()
  try:
    mainloop.run()
  except KeyboardInterrupt:
    log("Shutting down...")

  return 0


if __name__ == "__main__":
  sys.exit(main())
