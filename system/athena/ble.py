#!/usr/bin/env python3
"""
BLE GATT Server for Athena RPC.

NOTE: DBus imports happen inside main() to avoid fork issues with the manager.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import os
import sys
import time

sys.path.insert(0, "/data/openpilot")

from jsonrpc import JSONRPCResponseManager, dispatcher
from openpilot.common.swaglog import cloudlog

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
BLE_BLOCKED_METHODS = {"uploadFileToUrl", "uploadFilesToUrls", "startLocalProxy", "takeSnapshot", "webrtc"}


def get_serial() -> str:
  try:
    with open("/proc/cmdline") as f:
      for part in f.read().split():
        if part.startswith("androidboot.serialno="):
          return part.split("=", 1)[1]
  except Exception:
    pass
  return "unknown"


def get_device_name() -> str:
  return f"comma-{get_serial()}"


def get_bd_address() -> str:
  h = hashlib.sha256(get_serial().encode()).digest()
  # BLE static random address: top 2 bits of first byte must be 11
  first_byte = h[0] | 0xC0
  return f"{first_byte:02X}:{h[1]:02X}:{h[2]:02X}:{h[3]:02X}:{h[4]:02X}:{h[5]:02X}"


def power_on_wcn3990():
  for rfkill_dir in sorted(os.listdir("/sys/class/rfkill/")):
    rfkill_path = f"/sys/class/rfkill/{rfkill_dir}"
    try:
      with open(f"{rfkill_path}/type") as f:
        if f.read().strip() != "bluetooth":
          continue
      with open(f"{rfkill_path}/name") as f:
        if f.read().strip() != "bt_power":
          continue
      cloudlog.info("Powering on WCN3990 via rfkill")
      subprocess.run(["sudo", "sh", "-c", f"echo 0 > {rfkill_path}/soft"], capture_output=True)
      time.sleep(2)
      return True
    except Exception:
      continue
  return False


def init_bluetooth():
  try:
    result = subprocess.run(["hciconfig", "hci0"], capture_output=True, timeout=5)
    if result.returncode == 0 and b"UP RUNNING" in result.stdout:
      return True
  except Exception:
    pass

  if not os.path.exists("/dev/ttyHS1"):
    cloudlog.info("ERROR: /dev/ttyHS1 not found")
    return False

  subprocess.run(["sudo", "pkill", "-f", "btattach"], capture_output=True)
  subprocess.run(["sudo", "hciconfig", "hci0", "down"], capture_output=True)
  time.sleep(1)

  # WCN3990: power on via rfkill, attach at 3Mbaud with H4 protocol
  power_on_wcn3990()
  subprocess.Popen(["sudo", "btattach", "-B", "/dev/ttyHS1", "-S", "3000000"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  device_name = get_device_name()
  bd_addr = get_bd_address()
  addr_set = False

  for i in range(20):
    time.sleep(1)
    try:
      result = subprocess.run(["hciconfig", "hci0"], capture_output=True, timeout=5)
      if result.returncode != 0:
        continue

      # Set static address while adapter is down (must be done before power on)
      if not addr_set and b"UP RUNNING" not in result.stdout:
        subprocess.run(["sudo", "btmgmt", "static-addr", bd_addr], capture_output=True)
        subprocess.run(["sudo", "hciconfig", "hci0", "up"], capture_output=True)
        addr_set = True
        continue

      if b"UP RUNNING" in result.stdout:
        logcloudlog.info("Bluetooth initialized")
        subprocess.run(["sudo", "hciconfig", "hci0", "name", device_name], capture_output=True)
        cloudlog.info(f"BD address: {bd_addr}, name: {device_name}")
        # Enable LE peripheral mode for GATT server
        subprocess.run(["sudo", "btmgmt", "le", "on"], capture_output=True)
        subprocess.run(["sudo", "btmgmt", "connectable", "on"], capture_output=True)
        subprocess.run(["sudo", "btmgmt", "advertising", "on"], capture_output=True)
        return True

      if b"DOWN" in result.stdout:
        subprocess.run(["sudo", "hciconfig", "hci0", "up"], capture_output=True)
    except Exception:
      pass
    cloudlog.info(f"Waiting for Bluetooth... ({i+1}/20)")

  cloudlog.info("ERROR: Failed to initialize Bluetooth")
  return False


def main():
  import dbus
  import dbus.exceptions
  import dbus.mainloop.glib
  import dbus.service
  from gi.repository import GLib

  # Import shared RPC methods - registers them with dispatcher
  from openpilot.system.athena import rpc_methods  # noqa: F401

  cloudlog.info(f"Starting BLE server: {get_device_name()}")
  cloudlog.info(f"Loaded {len(dispatcher.keys())} RPC methods")

  if not init_bluetooth():
    return 1

  dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
  bus = dbus.SystemBus()

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
      cloudlog.info(f"RPC: {request_text[:80]}...")
      try:
        req = json.loads(request_text)
        method = req.get("method", "")
        if method in BLE_BLOCKED_METHODS:
          self.response_char.send_response(json.dumps({
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method '{method}' not available over BLE"},
            "id": req.get("id")
          }))
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
    cloudlog.info("ERROR: No Bluetooth adapter found")
    return 1

  app = Application(bus)
  advertisement = Advertisement(bus, 0)

  service_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), GATT_MANAGER_IFACE)
  ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), LE_ADVERTISING_MANAGER_IFACE)

  service_manager.RegisterApplication(app.get_path(), {},
    reply_handler=lambda: cloudlog.info("GATT registered"),
    error_handler=lambda e: cloudlog.info(f"GATT failed: {e}"))

  ad_manager.RegisterAdvertisement(advertisement.get_path(), {},
    reply_handler=lambda: cloudlog.info("Advertising"),
    error_handler=lambda e: cloudlog.info(f"Ad failed: {e}"))

  cloudlog.info("Server running")
  mainloop = GLib.MainLoop()
  try:
    mainloop.run()
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == "__main__":
  sys.exit(main())
