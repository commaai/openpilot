#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import os
import sys
import time
import secrets

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib
from jsonrpc import JSONRPCResponseManager, dispatcher
from openpilot.common.params import Params

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
ADAPTER_PATH = "/org/bluez/hci0"

BLE_MTU = 512
MAX_REQUEST_SIZE = 128 * 1024


def log(msg: str):
  print(f"[BLE] {msg}", flush=True)


def get_device_name() -> str:
  dongle_id = Params().get("DongleId")
  return f"comma-{dongle_id}"


def start_pairing() -> str:
  code = f"{secrets.randbelow(1_000_000):06d}"
  Params().put("BlePairingCode", code)
  log(f"Pairing mode started with code: {code}")
  return code


def stop_pairing():
  Params().remove("BlePairingCode")
  log("Pairing mode stopped")


def get_ble_token() -> str | None:
  return Params().get("BleToken") or None


def set_ble_token() -> str:
  token = secrets.token_urlsafe(32)
  Params().put("BleToken", token)
  stop_pairing()
  log(f"Token {token[:8]}... issued")
  return token


def clear_ble_token():
  Params().remove("BleToken")
  log("BLE token revoked")


def get_mac_addr() -> str:
  mac_suffix = Params().get("DongleId").lower()
  return f"C0:{mac_suffix[0:2]}:{mac_suffix[2:4]}:{mac_suffix[4:6]}:{mac_suffix[6:8]}:{mac_suffix[8:10]}"


def set_static_addr():
  mac_address = get_mac_addr()
  # Must be set while controller is powered off — btmgmt power off/on deregisters
  # the adapter from BlueZ DBus, so we set addr+privacy on the DOWN adapter
  # before powering it on with hciconfig
  subprocess.run(["sudo", "btmgmt", "--index", "0", "static-addr", mac_address], capture_output=True)
  subprocess.run(["sudo", "btmgmt", "--index", "0", "privacy", "on"], capture_output=True)
  subprocess.run(["sudo", "hciconfig", "hci0", "up"], capture_output=True)
  log(f"Set BLE static address to {mac_address}")


def init_bluetooth():
  if not os.path.exists("/dev/ttyHS1"):
    log("ERROR: /dev/ttyHS1 not found")
    return False

  subprocess.run(["sudo", "pkill", "-f", "btattach"], capture_output=True)
  subprocess.run(["sudo", "hciconfig", "hci0", "down"], capture_output=True)
  time.sleep(1)

  subprocess.Popen(["sudo", "btattach", "-B", "/dev/ttyHS1", "-S", "115200"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  for i in range(10):
    time.sleep(1)
    result = subprocess.run(["hciconfig", "hci0"], capture_output=True, timeout=5)
    if result.returncode == 0:
      log("Bluetooth adapter found")
      subprocess.run(["sudo", "hciconfig", "hci0", "down"], capture_output=True)
      set_static_addr()
      return True
    log(f"Waiting for Bluetooth... ({i + 1}/10)")

  log("ERROR: Failed to initialize Bluetooth")
  return False


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
    return {
      GATT_SERVICE_IFACE: {
        "UUID": self.uuid,
        "Primary": self.primary,
        "Characteristics": dbus.Array([dbus.ObjectPath(c.get_path()) for c in self.characteristics], signature="o"),
      }
    }

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
      self.value = dbus.Array(data[i : i + BLE_MTU], signature="y")
      self.PropertiesChanged(GATT_CHRC_IFACE, {"Value": self.value}, [])


class RPCRequestCharacteristic(Characteristic):
  def __init__(self, bus, index, service, response_char):
    Characteristic.__init__(self, bus, index, RPC_REQUEST_CHAR_UUID, ["write", "write-without-response"], service)
    self.response_char = response_char
    self.buffer = b""
    self.current_token = None

  def WriteValue(self, value, options):
    self.buffer += bytes(value)
    if len(self.buffer) > MAX_REQUEST_SIZE:
      log(f"Buffer exceeded {MAX_REQUEST_SIZE} bytes, dropping")
      self.buffer = b""
      return
    try:
      text = self.buffer.decode("utf-8")
      json.loads(text)
      self.buffer = b""
      self.process_request(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
      pass

  def process_request(self, request_text: str):
    log(f"RPC: {request_text[:80]}...")
    request_id = None
    try:
      req = json.loads(request_text)
      method = req.get("method", "")
      params = req.get("params", {})
      request_id = req.get("id")

      if isinstance(params, dict) and "token" in params:
        self.current_token = params.pop("token")
        request_text = json.dumps(req)

      if method not in ("blePair", "echo"):
        authorized = get_ble_token()
        if not self.current_token or not authorized or self.current_token != authorized:
          err = {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Unauthorized: pair with device first"}, "id": request_id}
          self.response_char.send_response(json.dumps(err))
          return

      from openpilot.system.athena.rpc_methods import set_transport

      set_transport("ble")
      response = JSONRPCResponseManager.handle(request_text, dispatcher)
      self.response_char.send_response(response.json)
    except Exception as e:
      log(f"RPC error: {e}")
      self.response_char.send_response(json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": request_id}))


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

  def __init__(self, bus, index, ad_manager):
    self.path = self.PATH_BASE + str(index)
    self.ad_manager = ad_manager
    self.registered = False
    dbus.service.Object.__init__(self, bus, self.path)

  def get_properties(self):
    return {
      LE_ADVERTISEMENT_IFACE: {
        "Type": "peripheral",
        "ServiceUUIDs": dbus.Array([ATHENA_SERVICE_UUID], signature="s"),
        "LocalName": dbus.String(get_device_name()),
        "IncludeTxPower": dbus.Boolean(True),
      }
    }

  def get_path(self):
    return dbus.ObjectPath(self.path)

  @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
  def GetAll(self, interface):
    if interface != LE_ADVERTISEMENT_IFACE:
      raise InvalidArgsException()
    return self.get_properties()[LE_ADVERTISEMENT_IFACE]

  @dbus.service.method(LE_ADVERTISEMENT_IFACE)
  def Release(self):
    self.registered = False
    if Params().get_bool("EnableBLE"):
      log("Advertisement released, re-registering...")
      GLib.timeout_add(100, self._re_register)
    else:
      log("Advertisement released, EnableBLE is off — not re-registering")

  def register(self):
    if self.registered:
      return
    self.ad_manager.RegisterAdvertisement(
      self.get_path(),
      {},
      reply_handler=self._on_registered,
      error_handler=lambda e: log(f"Ad register failed: {e}"),
    )

  def unregister(self):
    if not self.registered:
      return
    try:
      self.ad_manager.UnregisterAdvertisement(self.get_path())
      log("Advertisement unregistered")
    except Exception as e:
      log(f"Ad unregister failed: {e}")
    self.registered = False

  def _on_registered(self):
    self.registered = True
    log("Advertising")

  def _re_register(self):
    self.register()
    return False


def _poll_enable_ble(advertisement):
  enabled = Params().get_bool("EnableBLE")
  if enabled and not advertisement.registered:
    log("EnableBLE turned on — registering advertisement")
    advertisement.register()
  elif not enabled and advertisement.registered:
    log("EnableBLE turned off — unregistering advertisement")
    advertisement.unregister()
  return True


def main():
  from openpilot.system.athena import rpc_methods  # noqa: F401

  log(f"Starting BLE server: {get_device_name()}")
  log(f"Loaded {len(dispatcher.keys())} RPC methods")
  time.sleep(10)

  if not init_bluetooth():
    return 1
  time.sleep(10)

  dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
  bus = dbus.SystemBus()

  app = Application(bus)
  service_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, ADAPTER_PATH), GATT_MANAGER_IFACE)
  ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, ADAPTER_PATH), LE_ADVERTISING_MANAGER_IFACE)
  advertisement = Advertisement(bus, 0, ad_manager)

  service_manager.RegisterApplication(app.get_path(), {}, reply_handler=lambda: log("GATT registered"), error_handler=lambda e: log(f"GATT failed: {e}"))

  # Only start advertising if EnableBLE is on; poll every 5s to react to toggle changes
  if Params().get_bool("EnableBLE"):
    advertisement.register()
  GLib.timeout_add_seconds(5, _poll_enable_ble, advertisement)

  log("Server running")
  mainloop = GLib.MainLoop()
  try:
    mainloop.run()
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == "__main__":
  sys.exit(main())
