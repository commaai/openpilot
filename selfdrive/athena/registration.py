#!/usr/bin/env python3
import os
import time
import json
import jwt

from datetime import datetime, timedelta
from common.api import api_get
from common.params import Params
from common.spinner import Spinner
from common.file_helpers import mkdirs_exists_ok
from common.basedir import PERSIST
from selfdrive.controls.lib.alertmanager import set_offroad_alert
from selfdrive.hardware import HARDWARE
from selfdrive.swaglog import cloudlog


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"


def register(show_spinner=False) -> str:
  params = Params()
  params.put("SubscriberInfo", HARDWARE.get_subscriber_info())

  IMEI = params.get("IMEI", encoding='utf8')
  HardwareSerial = params.get("HardwareSerial", encoding='utf8')
  dongle_id = params.get("DongleId", encoding='utf8')
  needs_registration = None in (IMEI, HardwareSerial, dongle_id)

  # create a key for auth
  # your private key is kept on your device persist partition and never sent to our servers
  # do not erase your persist partition
  if not os.path.isfile(f"{PERSIST}/comma/id_rsa.pub"):
    needs_registration = True
    cloudlog.warning("generating your personal RSA key")
    mkdirs_exists_ok(f"{PERSIST}/comma")
    assert os.system(f"openssl genrsa -out {PERSIST}/comma/id_rsa.tmp 2048") == 0
    assert os.system(f"openssl rsa -in {PERSIST}/comma/id_rsa.tmp -pubout -out {PERSIST}/comma/id_rsa.tmp.pub") == 0
    os.rename(f"{PERSIST}/comma/id_rsa.tmp", f"{PERSIST}/comma/id_rsa")
    os.rename(f"{PERSIST}/comma/id_rsa.tmp.pub", f"{PERSIST}/comma/id_rsa.pub")

  if needs_registration:
    if show_spinner:
      spinner = Spinner()
      spinner.update("registering device")

    # Create registration token, in the future, this key will make JWTs directly
    with open(f"{PERSIST}/comma/id_rsa.pub") as f1, open(f"{PERSIST}/comma/id_rsa") as f2:
      public_key = f1.read()
      private_key = f2.read()

    # Block until we get the imei
    serial = HARDWARE.get_serial()
    start_time = time.monotonic()
    imei1, imei2 = None, None
    while imei1 is None and imei2 is None:
      try:
        imei1, imei2 = HARDWARE.get_imei(0), HARDWARE.get_imei(1)
      except Exception:
        cloudlog.exception("Error getting imei, trying again...")
        time.sleep(1)

      if time.monotonic() - start_time > 60 and show_spinner:
        spinner.update(f"registering device - serial: {serial}, IMEI: ({imei1}, {imei2})")

    params.put("IMEI", imei1)
    params.put("HardwareSerial", serial)

    backoff = 0
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

  if dongle_id:
    params.put("DongleId", dongle_id)
    set_offroad_alert("Offroad_UnofficialHardware", dongle_id == UNREGISTERED_DONGLE_ID)
  return dongle_id


if __name__ == "__main__":
  print(register())
