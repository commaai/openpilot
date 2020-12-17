import os
import time
import json

import jwt

from datetime import datetime, timedelta
from common.api import api_get
from common.params import Params
from common.file_helpers import mkdirs_exists_ok
from common.basedir import PERSIST
from selfdrive.hardware import HARDWARE
from selfdrive.swaglog import cloudlog
from selfdrive.version import version, terms_version, training_version, get_git_commit, \
                              get_git_branch, get_git_remote


def register(spinner=None):
  params = Params()
  params.put("Version", version)
  params.put("TermsVersion", terms_version)
  params.put("TrainingVersion", training_version)

  params.put("GitCommit", get_git_commit(default=""))
  params.put("GitBranch", get_git_branch(default=""))
  params.put("GitRemote", get_git_remote(default=""))
  params.put("SubscriberInfo", HARDWARE.get_subscriber_info())

  needs_registration = False

  # create a key for auth
  # your private key is kept on your device persist partition and never sent to our servers
  # do not erase your persist partition
  if not os.path.isfile(PERSIST+"/comma/id_rsa.pub"):
    needs_registration = True
    cloudlog.warning("generating your personal RSA key")
    mkdirs_exists_ok(PERSIST+"/comma")
    assert os.system("openssl genrsa -out "+PERSIST+"/comma/id_rsa.tmp 2048") == 0
    assert os.system("openssl rsa -in "+PERSIST+"/comma/id_rsa.tmp -pubout -out "+PERSIST+"/comma/id_rsa.tmp.pub") == 0
    os.rename(PERSIST+"/comma/id_rsa.tmp", PERSIST+"/comma/id_rsa")
    os.rename(PERSIST+"/comma/id_rsa.tmp.pub", PERSIST+"/comma/id_rsa.pub")

  # make key readable by app users (ai.comma.plus.offroad)
  os.chmod(PERSIST+'/comma/', 0o755)
  os.chmod(PERSIST+'/comma/id_rsa', 0o744)

  dongle_id = params.get("DongleId", encoding='utf8')
  needs_registration = needs_registration or dongle_id is None

  if needs_registration:
    if spinner is not None:
      spinner.update("registering device")

    # Create registration token, in the future, this key will make JWTs directly
    private_key = open(PERSIST+"/comma/id_rsa").read()
    public_key = open(PERSIST+"/comma/id_rsa.pub").read()
    register_token = jwt.encode({'register': True, 'exp': datetime.utcnow() + timedelta(hours=1)}, private_key, algorithm='RS256')

    # Block until we get the imei
    imei1, imei2 = None, None
    while imei1 is None and imei2 is None:
      try:
        imei1, imei2 = HARDWARE.get_imei(0), HARDWARE.get_imei(1)
      except Exception:
        cloudlog.exception("Error getting imei, trying again...")
        time.sleep(1)

    while True:
      try:
        cloudlog.info("getting pilotauth")
        resp = api_get("v2/pilotauth/", method='POST', timeout=15,
                       imei=imei1, imei2=imei2, serial=HARDWARE.get_serial(), public_key=public_key, register_token=register_token)
        dongleauth = json.loads(resp.text)
        dongle_id = dongleauth["dongle_id"]
        params.put("DongleId", dongle_id)
        break
      except Exception:
        cloudlog.exception("failed to authenticate")
        time.sleep(1)

  return dongle_id

if __name__ == "__main__":
  print(register())
