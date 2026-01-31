#!/usr/bin/env python3
"""
BLE GATT Server for Athena RPC.

NOTE: DBus imports happen inside main() to avoid fork issues with the manager.
"""
from __future__ import annotations

import json
import subprocess
import os
import sys
import time
import random

sys.path.insert(0, "/data/openpilot")

from jsonrpc import JSONRPCResponseManager, dispatcher
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE

ATHENA_SERVICE_UUID = "a51a5a10-0001-4c0d-b8e6-a51a5a100001"
RPC_REQUEST_CHAR_UUID = "a51a5a10-0002-4c0d-b8e6-a51a5a100001"
RPC_RESPONSE_CHAR_UUID = "a51a5a10-0003-4c0d-b8e6-a51a5a100001"

BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"

BLE_MTU = 512


def log(msg: str):
  print(f"[BLE] {msg}", flush=True)


def get_dongle_id() -> str:
  try:
    with open("/data/params/d/DongleId") as f:
      return f.read().strip()
  except Exception:
    return "unknown"


def get_device_name() -> str:
  dongle_id = get_dongle_id()
  return f"comma-{dongle_id[:8]}" if dongle_id else "comma-device"


def get_pairing_code() -> str | None:
  params = Params()
  return params.get("BlePairingCode")


def start_pairing() -> str:
  params = Params()
  code = f"{random.randint(0, 999999):06d}"
  params.put("BlePairingCode", code)
  log(f"Pairing mode started with code: {code}")
  return code


def stop_pairing():
  params = Params()
  params.remove("BlePairingCode")
  log("Pairing mode stopped")


def generate_token() -> str:
  import secrets
  return secrets.token_urlsafe(32)


def get_authorized_tokens() -> set[str]:
  params = Params()
  tokens = params.get("BleAuthorizedTokens")
  if not tokens:
    return set()
  return set(tokens)


def add_authorized_token() -> str:
  params = Params()
  token = generate_token()
  tokens = get_authorized_tokens()
  tokens.add(token)
  params.put("BleAuthorizedTokens", list(tokens))
  stop_pairing()  # Stop pairing mode after successful pairing
  log(f"Token {token[:8]}... issued and authorized")
  return token


def init_bluetooth():
  try:
    result = subprocess.run(["hciconfig", "hci0"], capture_output=True, timeout=5)
    if result.returncode == 0 and b"UP RUNNING" in result.stdout:
      return True
  except Exception:
    pass

  if not os.path.exists("/dev/ttyHS1"):
    log("ERROR: /dev/ttyHS1 not found")
    return False

  subprocess.run(["sudo", "pkill", "-f", "btattach"], capture_output=True)
  subprocess.run(["sudo", "hciconfig", "hci0", "down"], capture_output=True)
  time.sleep(1)

  subprocess.Popen(["sudo", "btattach", "-B", "/dev/ttyHS1", "-S", "115200"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  for i in range(10):
    time.sleep(1)
    try:
      result = subprocess.run(["hciconfig", "hci0"], capture_output=True, timeout=5)
      if result.returncode == 0 and b"UP RUNNING" in result.stdout:
        log("Bluetooth initialized")

        # Set unique MAC address based on hardware serial
        serial = HARDWARE.get_serial()
        if serial and len(serial) >= 10:
          # Use last 10 hex chars of serial to create MAC address (5 bytes)
          # Format: C0:xx:xx:xx:xx:xx (C0 is locally administered unicast)
          mac_suffix = serial[-10:].lower()
          mac_address = f"C0:{mac_suffix[0:2]}:{mac_suffix[2:4]}:{mac_suffix[4:6]}:{mac_suffix[6:8]}:{mac_suffix[8:10]}"
          subprocess.run(["sudo", "hciconfig", "hci0", "down"], capture_output=True)
          subprocess.run(["sudo", "btmgmt", "--index", "0", "public-addr", mac_address], capture_output=True)
          subprocess.run(["sudo", "hciconfig", "hci0", "up"], capture_output=True)
          log(f"Set Bluetooth MAC address to {mac_address} (from serial {serial})")

        return True
      if result.returncode == 0 and b"DOWN" in result.stdout:
        subprocess.run(["sudo", "hciconfig", "hci0", "up"], capture_output=True)
    except Exception:
      pass
    log(f"Waiting for Bluetooth... ({i+1}/10)")

  log("ERROR: Failed to initialize Bluetooth")
  return False


def main():
  import dbus
  import dbus.exceptions
  import dbus.mainloop.glib
  import dbus.service
  from gi.repository import GLib

  # Import shared RPC methods - registers them with dispatcher
  from openpilot.system.athena import rpc_methods  # noqa: F401

  log(f"Starting BLE server: {get_device_name()}")
  log(f"Loaded {len(dispatcher.keys())} RPC methods")

  if not init_bluetooth():
    return 1

  mainloop = dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
  bus = dbus.SystemBus(mainloop=mainloop)

  class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"

  class Characteristic(dbus.service.Object):
    def __init__(self, bus, index, uuid, flags, service):
      self.path = service.path + "/char" + str(index)
      self.uuid = uuid
      self.service = service
      self.flags = flags
      self.value = []
      dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
      return {GATT_CHRC_IFACE: {"Service": self.service.get_path(), "UUID": self.uuid, "Flags": self.flags}}

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

  class Service(dbus.service.Object):
    PATH_BASE = "/org/bluez/app/service"

    def __init__(self, bus, index, uuid, primary):
      self.path = self.PATH_BASE + str(index)
      self.uuid = uuid
      self.primary = primary
      self.characteristics = []
      dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
      return {GATT_SERVICE_IFACE: {
        "UUID": self.uuid,
        "Primary": self.primary,
        "Characteristics": dbus.Array([dbus.ObjectPath(c.get_path()) for c in self.characteristics], signature="o"),
      }}

    def get_path(self):
      return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic):
      self.characteristics.append(characteristic)

  class RPCResponseCharacteristic(Characteristic):
    def __init__(self, bus, index, service):
      Characteristic.__init__(self, bus, index, RPC_RESPONSE_CHAR_UUID, ["notify"], service)
      self.notifying = False

    def StartNotify(self):
      self.notifying = True

    def StopNotify(self):
      self.notifying = False

    def send_response(self, response: str):
      if not self.notifying:
        return
      data = response.encode("utf-8")
      for i in range(0, len(data), BLE_MTU):
        self.value = dbus.Array(data[i:i + BLE_MTU], signature="y")
        self.PropertiesChanged(GATT_CHRC_IFACE, {"Value": self.value}, [])

  class RPCRequestCharacteristic(Characteristic):
    def __init__(self, bus, index, service, response_char):
      Characteristic.__init__(self, bus, index, RPC_REQUEST_CHAR_UUID, ["write", "write-without-response"], service)
      self.response_char = response_char
      self.buffer = b""
      self.current_token = None

    def WriteValue(self, value, options):
      self.buffer += bytes(value)
      try:
        text = self.buffer.decode("utf-8")
        json.loads(text)
        self.buffer = b""
        self.process_request(text)
      except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    def process_request(self, request_text: str):
      log(f"RPC: {request_text[:80]}...")
      try:
        req = json.loads(request_text)
        method = req.get("method", "")
        params = req.get("params", {})
        request_id = req.get("id")

        # Track and remove token from params if provided
        if isinstance(params, dict) and "token" in params:
          self.current_token = params["token"]
          # Remove token from params before passing to RPC methods
          params = {k: v for k, v in params.items() if k != "token"}
          req["params"] = params
          request_text = json.dumps(req)

        # Allow pairing methods and echo without authorization
        if method not in ["blePair", "echo"]:
          authorized_tokens = get_authorized_tokens()
          if not self.current_token or self.current_token not in authorized_tokens:
            error_response = json.dumps({
              "jsonrpc": "2.0",
              "error": {"code": -32001, "message": "Unauthorized: pair with device first"},
              "id": request_id
            })
            self.response_char.send_response(error_response)
            return

        response = JSONRPCResponseManager.handle(request_text, dispatcher)
        self.response_char.send_response(response.json)
      except Exception as e:
        self.response_char.send_response(json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": None}))

  class AthenaService(Service):
    def __init__(self, bus, index):
      Service.__init__(self, bus, index, ATHENA_SERVICE_UUID, True)
      self.response_char = RPCResponseCharacteristic(bus, 1, self)
      self.add_characteristic(RPCRequestCharacteristic(bus, 0, self, self.response_char))
      self.add_characteristic(self.response_char)

  class Application(dbus.service.Object):
    def __init__(self, bus):
      self.path = "/"
      self.services = []
      dbus.service.Object.__init__(self, bus, self.path)
      self.services.append(AthenaService(bus, 0))

    def get_path(self):
      return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self):
      response = {}
      for service in self.services:
        response[service.get_path()] = service.get_properties()
        for chrc in service.characteristics:
          response[chrc.get_path()] = chrc.get_properties()
      return response

  class Advertisement(dbus.service.Object):
    PATH_BASE = "/org/bluez/app/advertisement"

    def __init__(self, bus, index):
      self.path = self.PATH_BASE + str(index)
      dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
      return {LE_ADVERTISEMENT_IFACE: {
        "Type": "peripheral",
        "ServiceUUIDs": dbus.Array([ATHENA_SERVICE_UUID], signature="s"),
        "LocalName": dbus.String(get_device_name()),
        "IncludeTxPower": dbus.Boolean(True),
      }}

    def get_path(self):
      return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
      if interface != LE_ADVERTISEMENT_IFACE:
        raise InvalidArgsException()
      return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE)
    def Release(self):
      pass

  def find_adapter(bus):
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
    for o, props in remote_om.GetManagedObjects().items():
      if GATT_MANAGER_IFACE in props:
        return o
    return None

  adapter_path = find_adapter(bus)
  if not adapter_path:
    log("ERROR: No Bluetooth adapter found")
    return 1

  app = Application(bus)
  advertisement = Advertisement(bus, 0)

  service_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), GATT_MANAGER_IFACE)
  ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), LE_ADVERTISING_MANAGER_IFACE)

  service_manager.RegisterApplication(app.get_path(), {},
    reply_handler=lambda: log("GATT registered"),
    error_handler=lambda e: log(f"GATT failed: {e}"))

  ad_manager.RegisterAdvertisement(advertisement.get_path(), {},
    reply_handler=lambda: log("Advertising"),
    error_handler=lambda e: log(f"Ad failed: {e}"))

  log("Server running")
  mainloop = GLib.MainLoop()
  try:
    mainloop.run()
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == "__main__":
  sys.exit(main())
