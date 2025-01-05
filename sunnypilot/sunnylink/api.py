import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import jwt
from openpilot.common.api.base import BaseApi
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.system.hardware.hw import Paths

API_HOST = os.getenv('SUNNYLINK_API_HOST', 'https://stg.api.sunnypilot.ai')
UNREGISTERED_SUNNYLINK_DONGLE_ID = "UnregisteredDevice"
MAX_RETRIES = 6
CRASH_LOG_DIR = '/data/community/crashes'


class SunnylinkApi(BaseApi):
  def __init__(self, dongle_id):
    super().__init__(dongle_id, API_HOST)
    self.user_agent = "sunnypilot-"
    self.spinner = None
    self.params = Params()

  def api_get(self, endpoint, method='GET', timeout=10, access_token=None, **kwargs):
    if not self.params.get_bool("SunnylinkEnabled"):
      return None

    return super().api_get(endpoint, method, timeout, access_token, **kwargs)

  def resume_queued(self, timeout=10, **kwargs):
    sunnylinkId, commaId = self._resolve_dongle_ids()
    return self.api_get(f"ws/{sunnylinkId}/resume_queued", "POST", timeout, access_token=self.get_token(), **kwargs)

  def get_token(self, expiry_hours=1):
    # Add your additional data here
    additional_data = {}
    return super()._get_token(expiry_hours, **additional_data)

  def _status_update(self, message):
    print(message)
    if self.spinner:
      self.spinner.update(message)
      time.sleep(0.5)

  def _resolve_dongle_ids(self):
    sunnylink_dongle_id = self.params.get("SunnylinkDongleId", encoding='utf-8')
    comma_dongle_id = self.dongle_id or self.params.get("DongleId", encoding='utf-8')
    return sunnylink_dongle_id, comma_dongle_id

  def _resolve_imeis(self):
    imei1, imei2 = None, None
    imei_try = 0
    while imei1 is None and imei2 is None and imei_try < MAX_RETRIES:
      try:
        imei1, imei2 = self.params.get("IMEI", encoding='utf8') or HARDWARE.get_imei(0), HARDWARE.get_imei(1)
      except Exception:
        self._status_update(f"Error getting imei, trying again... [{imei_try + 1}/{MAX_RETRIES}]")
        time.sleep(1)
      imei_try += 1
    return imei1, imei2

  def _resolve_serial(self):
    return (self.params.get("HardwareSerial", encoding='utf8')
            or HARDWARE.get_serial())

  def register_device(self, spinner=None, timeout=60, verbose=False):
    self.spinner = spinner

    sunnylink_dongle_id, comma_dongle_id = self._resolve_dongle_ids()

    if comma_dongle_id is None:
      self._status_update("Comma dongle ID not found, deferring sunnylink's registration to comma's registration process.")
      return None

    imei1, imei2 = self._resolve_imeis()
    serial = self._resolve_serial()

    if sunnylink_dongle_id not in (None, UNREGISTERED_SUNNYLINK_DONGLE_ID):
      return sunnylink_dongle_id

    privkey_path = Path(f"{Paths.persist_root()}/comma/id_rsa")
    pubkey_path = Path(f"{Paths.persist_root()}/comma/id_rsa.pub")

    start_time = time.monotonic()
    successful_registration = False
    if not pubkey_path.is_file():
      sunnylink_dongle_id = UNREGISTERED_SUNNYLINK_DONGLE_ID
      self._status_update("Public key not found, setting dongle ID to unregistered.")
    else:
      Params().put("LastSunnylinkPingTime", "0")  # Reset the last ping time to 0 if we are trying to register
      with pubkey_path.open() as f1, privkey_path.open() as f2:
        public_key = f1.read()
        private_key = f2.read()

      backoff = 1
      while True:
        register_token = jwt.encode({'register': True, 'exp': datetime.utcnow() + timedelta(hours=1)}, private_key, algorithm='RS256')
        try:
          if verbose or time.monotonic() - start_time < timeout / 2:
            self._status_update("Registering device to sunnylink...")
          elif time.monotonic() - start_time >= timeout / 2:
            self._status_update("Still registering device to sunnylink...")

          resp = self.api_get("v2/pilotauth/", method='POST', timeout=15, imei=imei1, imei2=imei2, serial=serial,
                              comma_dongle_id=comma_dongle_id, public_key=public_key, register_token=register_token)

          if resp is None:
            raise Exception("Unable to register device, request was None")

          if resp.status_code in (409, 412):
            timeout = time.monotonic() - start_time  # Don't retry if the public key is already in use
            key_in_use = "Public key is already in use, is your key unique? Contact your vendor for a new key."
            unsafe_key = "Public key is known to not be unique and it's unsafe. Contact your vendor for a new key."
            error_message = key_in_use if resp.status_code == 409 else unsafe_key
            raise Exception(error_message)

          if resp.status_code != 200:
            raise Exception(f"Failed to register with sunnylink. Status code: {resp.status_code}\nData\n:{resp.text}")

          dongleauth = json.loads(resp.text)
          sunnylink_dongle_id = dongleauth["device_id"]
          if sunnylink_dongle_id:
            self._status_update("Device registered successfully.")
            successful_registration = True
            break
        except Exception as e:
          if verbose:
            self._status_update(f"Waiting {backoff}s before retry, Exception occurred during registration: [{str(e)}]")

          if not os.path.exists(CRASH_LOG_DIR):
            os.makedirs(CRASH_LOG_DIR)

          with open(f'{CRASH_LOG_DIR}/error.txt', 'a') as f:
            f.write(f"[{datetime.now()}] sunnylink: {str(e)}\n")

          backoff = min(backoff * 2 * (0.5 + random.random()), 60)
          time.sleep(backoff)

        if time.monotonic() - start_time > timeout:
          self._status_update(f"Giving up on sunnylink's registration after {timeout}s. Will retry on next boot.")
          time.sleep(3)
          break

    self.params.put("SunnylinkDongleId", sunnylink_dongle_id or UNREGISTERED_SUNNYLINK_DONGLE_ID)

    # Set the last ping time to the current time since we were just talking to the API
    last_ping = int(time.monotonic() * 1e9) if successful_registration else start_time
    Params().put("LastSunnylinkPingTime", str(last_ping))

    # Disable sunnylink if registration was not successful
    if not successful_registration:
      Params().put_bool("SunnylinkEnabled", False)

    self.spinner = None
    return sunnylink_dongle_id
