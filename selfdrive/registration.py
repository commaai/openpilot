import os
import json
import subprocess

from selfdrive.swaglog import cloudlog
from common.api import api_get
from common.params import Params

def get_imei():
  # Telephony.getDeviceId()
  result = subprocess.check_output(["service", "call", "phone", "130"]).strip().split("\n")
  hex_data = ''.join(l[14:49] for l in result[1:]).replace(" ", "")
  data = hex_data.decode("hex")

  imei_str = data[8:-4].replace("\x00", "")
  return imei_str

def get_serial():
  return subprocess.check_output(["getprop", "ro.serialno"]).strip()

def register():
  params = Params()
  try:
    dongle_id, dongle_secret = params.get("DongleId"), params.get("DongleSecret")
    if dongle_id is None or dongle_secret is None:
      resp = api_get("pilot_auth", method='POST', imei=get_imei(), serial=get_serial())
      dongleauth = json.loads(resp.text)
      dongle_id, dongle_secret = dongleauth["dongle_id"].encode('ascii'), dongleauth["dongle_secret"].encode('ascii')
      params.put("DongleId", dongle_id)
      params.put("DongleSecret", dongle_secret)
    return dongle_id, dongle_secret
  except Exception:
    cloudlog.exception("failed to authenticate")
    return None

if __name__ == "__main__":
  print api_get("").text
  print register()
