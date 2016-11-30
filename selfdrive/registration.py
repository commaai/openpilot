import os
import json
import subprocess

from selfdrive.swaglog import cloudlog
from common.api import api_get

DONGLEAUTH_PATH = "/sdcard/dongleauth"

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
  try:
    if os.path.exists(DONGLEAUTH_PATH):
      dongleauth = json.load(open(DONGLEAUTH_PATH))
    else:
      resp = api_get("pilot_auth", method='POST', imei=get_imei(), serial=get_serial())
      resp = resp.text
      dongleauth = json.loads(resp)
      open(DONGLEAUTH_PATH, "w").write(resp)
    return dongleauth["dongle_id"], dongleauth["dongle_secret"]
  except Exception:
    cloudlog.exception("failed to authenticate")
    return None

if __name__ == "__main__":
  print api_get("").text
  print register()
