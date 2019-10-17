import os
import json
import binascii
import subprocess
import itertools

from datetime import datetime, timedelta
from selfdrive.swaglog import cloudlog
from selfdrive.version import version, terms_version, training_version, get_git_commit, get_git_branch, get_git_remote
from common.api import api_get
from common.params import Params


def get_imei():
  ret = subprocess.check_output(["getprop", "oem.device.imeicache"], encoding='utf8').strip()  # pylint: disable=unexpected-keyword-arg
  if ret == "":
    ret = "000000000000000"
  return ret


def get_serial():
  return subprocess.check_output(["getprop", "ro.serialno"], encoding='utf8').strip()  # pylint: disable=unexpected-keyword-arg


# TODO: move this to a library
def parse_service_call(call):
  ret = subprocess.check_output(call, encoding='utf8').strip()   # pylint: disable=unexpected-keyword-arg
  if 'Parcel' not in ret:
    return None

  try:
    r = b""
    for line in ret.split("\n")[1:]: # Skip 'Parcel('
      line_hex = line[14:49].replace(' ', '')
      r += binascii.unhexlify(line_hex)

    r = r[8:] # Cut off length field
    r = r.decode('utf_16_be')

    # All pairs of two characters seem to be swapped. Not sure why
    result = ""
    for a, b, in itertools.zip_longest(r[::2], r[1::2], fillvalue='\x00'):
        result += b + a

    result = result.replace('\x00', '')

    return result
  except Exception:
    return None


def get_subscriber_info():
  ret = parse_service_call(["service", "call", "iphonesubinfo", "7"])
  if ret is None or len(ret) < 8:
    return ""
  return ret


def register():
  params = Params()
  params.put("Version", version)
  params.put("TermsVersion", terms_version)
  params.put("TrainingVersion", training_version)
  params.put("GitCommit", get_git_commit())
  params.put("GitBranch", get_git_branch())
  params.put("GitRemote", get_git_remote())
  params.put("SubscriberInfo", get_subscriber_info())

  # make key readable by app users (ai.comma.plus.offroad)
  os.chmod('/persist/comma/', 0o755)
  os.chmod('/persist/comma/id_rsa', 0o744)

  dongle_id, access_token = params.get("DongleId", encoding='utf8'), params.get("AccessToken", encoding='utf8')
  public_key = open("/persist/comma/id_rsa.pub").read()

  # create registration token
  # in the future, this key will make JWTs directly
  private_key = open("/persist/comma/id_rsa").read()

  # late import
  import jwt
  register_token = jwt.encode({'register':True, 'exp': datetime.utcnow() + timedelta(hours=1)}, private_key, algorithm='RS256')

  try:
    cloudlog.info("getting pilotauth")
    resp = api_get("v2/pilotauth/", method='POST', timeout=15,
                   imei=get_imei(), serial=get_serial(), public_key=public_key, register_token=register_token)
    dongleauth = json.loads(resp.text)
    dongle_id, access_token = dongleauth["dongle_id"], dongleauth["access_token"]

    params.put("DongleId", dongle_id)
    params.put("AccessToken", access_token)
    return dongle_id, access_token
  except Exception:
    cloudlog.exception("failed to authenticate")
    if dongle_id is not None and access_token is not None:
      return dongle_id, access_token
    else:
      return None

if __name__ == "__main__":
  print(api_get("").text)
  print(register())
