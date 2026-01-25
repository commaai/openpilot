#!/usr/bin/env python3
"""
BLE GATT Server for Athena

Exposes athenad JSON-RPC methods over Bluetooth Low Energy.
Uses the same dispatcher as athenad - no method duplication.

Architecture:
- GATT Service with custom UUID for Athena
- RPC Request characteristic (Write) - client sends JSON-RPC requests
- RPC Response characteristic (Notify) - server sends responses
- Handles message chunking for BLE MTU limits

Usage:
  python3 ble_server.py

Requires system python with gi module (not openpilot venv).
"""
from __future__ import annotations

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
import json
import subprocess
import os
import sys
import time
from gi.repository import GLib

# Add openpilot to path for importing athenad dispatcher
sys.path.insert(0, "/data/openpilot")

# Import the shared dispatcher from athenad
# This is the key - we use the SAME dispatcher, so all @dispatcher.add_method
# functions from athenad are automatically available over BLE
from jsonrpc import JSONRPCResponseManager

# BLE GATT UUIDs - Custom UUIDs for Asius/Athena service
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

# BLE MTU - conservative value for chunking
BLE_MTU = 512

# Methods that should be blocked over BLE (require network, too large, security risk)
BLE_BLOCKED_METHODS = {
  "uploadFileToUrl",      # Requires network
  "uploadFilesToUrls",    # Requires network
  "startLocalProxy",      # SSH tunneling - security risk over BLE
  "installStandby",       # Large file transfer
  "takeSnapshot",         # Response too large for BLE
  "webrtc",               # Requires network
}


def log(msg: str):
  print(f"[BLE] {msg}", flush=True)


def get_dongle_id() -> str:
  try:
    with open("/data/params/d/DongleId", "r") as f:
      return f.read().strip()
  except:
    return "unknown"


def get_device_name() -> str:
  dongle_id = get_dongle_id()
  return f"comma-{dongle_id[:8]}" if dongle_id else "comma-device"


def get_dispatcher():
  """
  Import and return the shared RPC dispatcher.
  Uses rpc_methods module which has minimal dependencies.
  """
  try:
    # Import the shared RPC methods module - this registers methods with the dispatcher
    from openpilot.system.athena import rpc_methods
    from jsonrpc import dispatcher
    log(f"Loaded {len(dispatcher.keys())} RPC methods")
    return dispatcher
  except ImportError as e:
    log(f"Warning: Could not import rpc_methods: {e}")
    log("Falling back to minimal dispatcher")
    # Fallback: create minimal dispatcher with basic methods
    from jsonrpc import dispatcher

    @dispatcher.add_method
    def echo(**kwargs):
      return kwargs

    @dispatcher.add_method
    def getVersion():
      return {"version": "unknown", "error": "rpc_methods not available"}

    @dispatcher.add_method
    def getDongleId():
      return get_dongle_id()

    return dispatcher


# Global dispatcher - initialized in main()
_dispatcher = None


class InvalidArgsException(dbus.exceptions.DBusException):
  _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


class Application(dbus.service.Object):
  """GATT Application for BLE server"""

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
      for chrc in service.get_characteristics():
        response[chrc.get_path()] = chrc.get_properties()
    return response


class Service(dbus.service.Object):
  """GATT Service base class"""
  PATH_BASE = "/org/bluez/app/service"

  def __init__(self, bus, index, uuid, primary):
    self.path = self.PATH_BASE + str(index)
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
          self.get_characteristic_paths(), signature="o"
        ),
      }
    }

  def get_path(self):
    return dbus.ObjectPath(self.path)

  def add_characteristic(self, characteristic):
    self.characteristics.append(characteristic)

  def get_characteristic_paths(self):
    return [dbus.ObjectPath(c.get_path()) for c in self.characteristics]

  def get_characteristics(self):
    return self.characteristics


class Characteristic(dbus.service.Object):
  """GATT Characteristic base class"""

  def __init__(self, bus, index, uuid, flags, service):
    self.path = service.path + "/char" + str(index)
    self.bus = bus
    self.uuid = uuid
    self.service = service
    self.flags = flags
    self.value = []
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

  @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
  def GetAll(self, interface):
    if interface != GATT_CHRC_IFACE:
      raise InvalidArgsException()
    return self.get_properties()[GATT_CHRC_IFACE]

  @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
  def ReadValue(self, options):
    return self.value

  @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
  def WriteValue(self, value, options):
    self.value = value

  @dbus.service.method(GATT_CHRC_IFACE)
  def StartNotify(self):
    pass

  @dbus.service.method(GATT_CHRC_IFACE)
  def StopNotify(self):
    pass

  @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
  def PropertiesChanged(self, interface, changed, invalidated):
    pass


class AthenaService(Service):
  """Athena GATT Service"""

  def __init__(self, bus, index):
    Service.__init__(self, bus, index, ATHENA_SERVICE_UUID, True)
    self.response_char = RPCResponseCharacteristic(bus, 1, self)
    self.add_characteristic(RPCRequestCharacteristic(bus, 0, self, self.response_char))
    self.add_characteristic(self.response_char)


class RPCRequestCharacteristic(Characteristic):
  """Write characteristic for RPC requests"""

  def __init__(self, bus, index, service, response_char):
    Characteristic.__init__(
      self, bus, index, RPC_REQUEST_CHAR_UUID, ["write", "write-without-response"], service
    )
    self.response_char = response_char
    self.buffer = b""

  def WriteValue(self, value, options):
    data = bytes(value)
    self.buffer += data

    # Check if we have a complete JSON message
    try:
      text = self.buffer.decode("utf-8")
      # Try to parse as JSON to see if complete
      json.loads(text)
      # If we get here, it's valid JSON
      self.buffer = b""
      self.process_request(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
      # Incomplete message, wait for more data
      pass
    except Exception as e:
      log(f"Error processing request: {e}")
      self.buffer = b""

  def process_request(self, request_text: str):
    global _dispatcher

    log(f"Received {len(request_text)} bytes")
    log(f"Processing request: {request_text[:100]}...")

    try:
      # Check if method is blocked for BLE
      try:
        req = json.loads(request_text)
        method = req.get("method", "")
        if method in BLE_BLOCKED_METHODS:
          error_response = json.dumps({
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method '{method}' not available over BLE"},
            "id": req.get("id")
          })
          self.response_char.send_response(error_response)
          return
      except:
        pass

      # Use athenad's dispatcher to handle the request
      response = JSONRPCResponseManager.handle(request_text, _dispatcher)
      response_text = response.json

      log(f"Sending response: {len(response_text)} bytes")
      self.response_char.send_response(response_text)

    except Exception as e:
      log(f"Error: {e}")
      error_response = json.dumps({
        "jsonrpc": "2.0",
        "error": {"code": -32603, "message": str(e)},
        "id": None
      })
      self.response_char.send_response(error_response)


class RPCResponseCharacteristic(Characteristic):
  """Notify characteristic for RPC responses"""

  def __init__(self, bus, index, service):
    Characteristic.__init__(
      self, bus, index, RPC_RESPONSE_CHAR_UUID, ["notify"], service
    )
    self.notifying = False

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

  def send_response(self, response: str):
    if not self.notifying:
      log("Warning: No client subscribed, cannot send response")
      return

    data = response.encode("utf-8")

    # Send in chunks if needed
    for i in range(0, len(data), BLE_MTU):
      chunk = data[i : i + BLE_MTU]
      self.value = dbus.Array(chunk, signature="y")
      self.PropertiesChanged(
        GATT_CHRC_IFACE, {"Value": self.value}, []
      )


class Advertisement(dbus.service.Object):
  """BLE Advertisement"""
  PATH_BASE = "/org/bluez/app/advertisement"

  def __init__(self, bus, index):
    self.path = self.PATH_BASE + str(index)
    self.bus = bus
    self.ad_type = "peripheral"
    self.service_uuids = [ATHENA_SERVICE_UUID]
    self.local_name = get_device_name()
    self.include_tx_power = True
    dbus.service.Object.__init__(self, bus, self.path)

  def get_properties(self):
    properties = {
      "Type": self.ad_type,
      "ServiceUUIDs": dbus.Array(self.service_uuids, signature="s"),
      "LocalName": dbus.String(self.local_name),
      "IncludeTxPower": dbus.Boolean(self.include_tx_power),
    }
    return {LE_ADVERTISEMENT_IFACE: properties}

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
  """Find the first available Bluetooth adapter"""
  remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
  objects = remote_om.GetManagedObjects()

  for o, props in objects.items():
    if GATT_MANAGER_IFACE in props:
      return o

  return None


def register_ad_cb():
  log("Advertisement registered - device is now discoverable")


def register_ad_error_cb(error):
  log(f"Failed to register advertisement: {error}")


def register_app_cb():
  log("GATT application registered successfully")


def register_app_error_cb(error):
  log(f"Failed to register GATT application: {error}")
  mainloop.quit()


def init_bluetooth():
  """Initialize Bluetooth hardware if not already done"""
  # Check if already running
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

  # Kill any existing btattach and reset hci0 if it exists but is down
  subprocess.run(["sudo", "pkill", "-f", "btattach"], capture_output=True)
  subprocess.run(["sudo", "hciconfig", "hci0", "down"], capture_output=True)
  time.sleep(1)

  # Start btattach
  log("Starting btattach...")
  subprocess.Popen(
    ["sudo", "btattach", "-B", "/dev/ttyHS1", "-S", "115200"],
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
      # Try to bring it up if it exists but is down
      if result.returncode == 0 and b"DOWN" in result.stdout:
        subprocess.run(["sudo", "hciconfig", "hci0", "up"], capture_output=True)
    except:
      pass
    log(f"Waiting for Bluetooth... ({i+1}/10)")

  log("ERROR: Failed to initialize Bluetooth")
  return False


def main():
  global _dispatcher, mainloop

  log("Starting BLE GATT Server...")
  log(f"Device name: {get_device_name()}")
  log(f"Service UUID: {ATHENA_SERVICE_UUID}")

  # Initialize Bluetooth hardware
  if not init_bluetooth():
    return 1

  # Initialize DBus
  dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
  bus = dbus.SystemBus()

  # Initialize dispatcher (imports athenad methods)
  _dispatcher = get_dispatcher()
  log(f"Dispatcher loaded with {len(_dispatcher.keys())} methods")

  # Find adapter
  adapter_path = find_adapter(bus)
  if not adapter_path:
    log("ERROR: No Bluetooth adapter found")
    return 1

  log(f"Using adapter: {adapter_path}")

  # Create GATT application
  app = Application(bus)

  # Create advertisement
  advertisement = Advertisement(bus, 0)

  # Get managers
  service_manager = dbus.Interface(
    bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), GATT_MANAGER_IFACE
  )
  ad_manager = dbus.Interface(
    bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), LE_ADVERTISING_MANAGER_IFACE
  )

  # Register GATT application
  service_manager.RegisterApplication(
    app.get_path(),
    {},
    reply_handler=register_app_cb,
    error_handler=register_app_error_cb
  )

  # Register advertisement
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
