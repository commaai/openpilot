import os
import json
import subprocess
import struct

from datetime import datetime, timedelta
from selfdrive.swaglog import cloudlog
from selfdrive.version import version, terms_version, training_version, get_git_commit, get_git_branch, get_git_remote
from common.api import api_get
from common.params import Params
from common.file_helpers import mkdirs_exists_ok


def get_imei():
  ret = subprocess.check_output(["getprop", "oem.device.imeicache"]).strip()
  if ret == "":
    ret = "000000000000000"
  return ret


def get_serial():
  return subprocess.check_output(["getprop", "ro.serialno"]).strip()


# TODO: move this to a library
def parse_service_call(call):
  ret = subprocess.check_output(call).strip()
  if 'Parcel' not in ret:
    return None
  try:
    def fh(x):
      if len(x) != 8:
        return []
      return [x[6:8], x[4:6], x[2:4], x[0:2]]
    hd = []
    for x in ret.split("\n")[1:]:
      for k in map(fh, x.split(": ")[1].split(" '")[0].split(" ")):
        hd.extend(k)
    return ''.join([chr(int(x, 16)) for x in hd])
  except Exception:
    return None


def get_subscriber_info():
  ret = parse_service_call(["service", "call", "iphonesubinfo", "7"])
  if ret is None or len(ret) < 8:
    return ""
  if struct.unpack("I", ret[4:8]) == -1:
    return ""
  return ret[8:-2:2]


def register():
  params = Params()
  params.put("Version", version)
  params.put("TermsVersion", terms_version)
  params.put("TrainingVersion", training_version)
  params.put("GitCommit", get_git_commit())
  params.put("GitBranch", get_git_branch())
  params.put("GitRemote", get_git_remote())
  params.put("SubscriberInfo", get_subscriber_info())

  # create a key for auth
  # your private key is kept on your device persist partition and never sent to our servers
  # do not erase your persist partition
  if not os.path.isfile("/persist/comma/id_rsa.pub"):
    cloudlog.warning("generating your personal RSA key")
    mkdirs_exists_ok("/persist/comma")
    assert os.system("openssl genrsa -out /persist/comma/id_rsa.tmp 2048") == 0
    assert os.system("openssl rsa -in /persist/comma/id_rsa.tmp -pubout -out /persist/comma/id_rsa.tmp.pub") == 0
    os.rename("/persist/comma/id_rsa.tmp", "/persist/comma/id_rsa")
    os.rename("/persist/comma/id_rsa.tmp.pub", "/persist/comma/id_rsa.pub")

  # make key readable by app users (ai.comma.plus.offroad)
  os.chmod('/persist/comma/', 0o755)
  os.chmod('/persist/comma/id_rsa', 0o744)

  dongle_id, access_token = params.get("DongleId"), params.get("AccessToken")
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
    dongle_id, access_token = dongleauth["dongle_id"].encode('ascii'), dongleauth["access_token"].encode('ascii')

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
