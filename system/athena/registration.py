#!/usr/bin/env python3
import time
import json
import jwt
import pyray as rl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from datetime import datetime, timedelta, UTC
from openpilot.common.api import api_get
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.hardware import HARDWARE, PC
from openpilot.system.hardware.hw import Paths
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.spinner import Spinner
from openpilot.common.swaglog import cloudlog


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"

def is_registered_device() -> bool:
  dongle = Params().get("DongleId", encoding='utf-8')
  return dongle not in (None, UNREGISTERED_DONGLE_ID)

def _get_dongle_id() -> str | None:
  dongle_id: str | None = Params().get("DongleId", encoding='utf8')
  if dongle_id is None and Path(Paths.persist_root()+"/comma/dongle_id").is_file():
    # not all devices will have this; added early in comma 3X production (2/28/24)
    with open(Paths.persist_root()+"/comma/dongle_id") as f:
      dongle_id = f.read().strip()

  pubkey = Path(Paths.persist_root()+"/comma/id_rsa.pub")
  if not pubkey.is_file():
    dongle_id = UNREGISTERED_DONGLE_ID
    cloudlog.warning(f"missing public key: {pubkey}")

  return dongle_id

def do_register(spinner = None, done_event=None) -> str | None:
    """
    All devices built since March 2024 come with all
    info stored in /persist/. This is kept around
    only for devices built before then.

    With a backend update to take serial number instead
    of dongle ID to some endpoints, this can be removed
    entirely.
    """
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

      if time.monotonic() - start_time > 60 and spinner:
        spinner.set_text(f"registering device - serial: {serial}, IMEI: ({imei1}, {imei2})")

    dongle_id = None
    backoff = 0
    start_time = time.monotonic()
    while True:
      try:
        register_token = jwt.encode({'register': True, 'exp': datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)}, private_key, algorithm='RS256')
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

      if time.monotonic() - start_time > 60 and spinner:
        spinner.set_text(f"registering device - serial: {serial}, IMEI: ({imei1}, {imei2})")

    return dongle_id

def register(show_spinner=False) -> str | None:
  dongle_id = _get_dongle_id()
  if not dongle_id:
    if show_spinner:
      with ThreadPoolExecutor(max_workers=1) as executor:
        gui_app.init_window("Register")
        spinner = Spinner()
        spinner.set_text("registering device")
        future = executor.submit(do_register, spinner)
        while not future.done():
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)
          spinner.render()
          rl.end_drawing()
        gui_app.close()
        dongle_id = future.result()
    else:
      dongle_id = do_register()

  if dongle_id:
    Params().put("DongleId", dongle_id)
    set_offroad_alert("Offroad_UnofficialHardware", (dongle_id == UNREGISTERED_DONGLE_ID) and not PC)
  return dongle_id


if __name__ == "__main__":
  print(register())
