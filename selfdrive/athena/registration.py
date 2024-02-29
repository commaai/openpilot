#!/usr/bin/env python3
import os
import time
import json
import jwt
import subprocess
from pathlib import Path

from datetime import datetime, timedelta
from openpilot.common.api import api_get
from openpilot.common.params import Params
from openpilot.common.spinner import Spinner
from openpilot.selfdrive.controls.lib.alertmanager import set_offroad_alert
from openpilot.system.hardware import HARDWARE, PC, AGNOS
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"

DONGLE_PATH = Paths.persist_root() + "/comma/dongle_id"

def is_registered_device() -> bool:
  dongle = Params().get("DongleId", encoding='utf-8')
  return dongle not in (None, UNREGISTERED_DONGLE_ID)


def write_dongle_id(dongle):
  # factory provisioning now writes out the dongle id to persist;
  # this is for devices that didn't ship with it written out

  if not AGNOS:
    return

  cloudlog.warning("writing dongle id to persist")
  try:
    subprocess.run([
      ["sudo", "mount", "-o", "rw,remount", "/persist"],
      ["echo", "-n", f"'{dongle}'", ">", DONGLE_PATH],
      ["sudo", "mount", "-o", "ro,remount", "/persist"],
    ])
    cloudlog.warning("successfully wrote dongle id to persist")
  except subprocess.CalledProcessError:
    cloudlog.exception("failed to write dongle id to persist")

def network_register(show_spinner=False) -> str:
  # this is only used for older devices that didn't have their dongle id
  # written out to /persist from the factory

  if show_spinner:
    spinner = Spinner()
    spinner.update("registering device")

  # Create registration token, in the future, this key will make JWTs directly
  with open(Paths.persist_root()+"/comma/id_rsa.pub") as f1, open(Paths.persist_root()+"/comma/id_rsa") as f2:
    public_key = f1.read()
    private_key = f2.read()

  # Block until we get the imei
  serial = HARDWARE.get_serial()
  start_time = time.monotonic()
  imei1: str | None = None
  imei2: str | None = None
  while imei1 is None and imei2 is None:
    try:
      imei1, imei2 = HARDWARE.get_imei(0), HARDWARE.get_imei(1)
    except Exception:
      cloudlog.exception("Error getting imei, trying again...")
      time.sleep(1)

    if time.monotonic() - start_time > 60 and show_spinner:
      spinner.update(f"registering device - serial: {serial}, IMEI: ({imei1}, {imei2})")

  backoff = 0
  dongle_id: str
  start_time = time.monotonic()
  while True:
    try:
      register_token = jwt.encode({'register': True, 'exp': datetime.utcnow() + timedelta(hours=1)}, private_key, algorithm='RS256')
      cloudlog.info("getting pilotauth")
      resp = api_get("v2/pilotauth/", method='POST', timeout=15,
                     imei=imei1, imei2=imei2, serial=serial, public_key=public_key, register_token=register_token)

      if resp.status_code in (402, 403):
        cloudlog.info(f"Unable to register device, got {resp.status_code}")
        dongle_id = UNREGISTERED_DONGLE_ID
      else:
        dongleauth = json.loads(resp.text)
        dongle_id = dongleauth["dongle_id"]
      break
    except Exception:
      cloudlog.exception("failed to authenticate")
      backoff = min(backoff + 1, 15)
      time.sleep(backoff)

    if time.monotonic() - start_time > 60 and show_spinner:
      spinner.update(f"registering device - serial: {serial}, IMEI: ({imei1}, {imei2})")

  if show_spinner:
    spinner.close()

  return dongle_id


def register(show_spinner=False) -> str | None:
  params = Params()
  dongle_id = params.get("DongleId", encoding='utf8')

  pubkey = Path(Paths.persist_root()+"/comma/id_rsa.pub")
  if not pubkey.is_file():
    dongle_id = UNREGISTERED_DONGLE_ID
    cloudlog.warning(f"missing public key: {pubkey}")
  elif dongle_id is None:
    if os.path.exists(DONGLE_PATH):
      with open(DONGLE_PATH) as f:
        dongle_id = f.read().strip()
    else:
      dongle_id = network_register(show_spinner)

  if dongle_id:
    params.put("DongleId", dongle_id)
    set_offroad_alert("Offroad_UnofficialHardware", (dongle_id == UNREGISTERED_DONGLE_ID) and AGNOS)
    if AGNOS and not os.path.exists(DONGLE_PATH):
      write_dongle_id(dongle_id)

  return dongle_id


if __name__ == "__main__":
  print(register())
