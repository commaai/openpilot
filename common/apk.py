import os
import subprocess
import glob
import hashlib
import shutil
import json
from common.basedir import BASEDIR
from common.params import Params
from selfdrive.swaglog import cloudlog

android_packages = ("ai.comma.plus.offroad",)

def get_installed_apks():
  dat = subprocess.check_output(["pm", "list", "packages", "-f"], encoding='utf8').strip().split("\n")
  ret = {}
  for x in dat:
    if x.startswith("package:"):
      v,k = x.split("package:")[1].split("=")
      ret[k] = v
  return ret

def install_apk(path):
  # can only install from world readable path
  install_path = "/sdcard/%s" % os.path.basename(path)
  shutil.copyfile(path, install_path)

  ret = subprocess.call(["pm", "install", "-r", install_path])
  os.remove(install_path)
  return ret == 0

def start_offroad():
  set_package_permissions()
  system("am start -n ai.comma.plus.offroad/.MainActivity")

def set_package_permissions():
  pm_grant("ai.comma.plus.offroad", "android.permission.ACCESS_FINE_LOCATION")
  pm_grant("ai.comma.plus.offroad", "android.permission.READ_PHONE_STATE")
  pm_grant("ai.comma.plus.offroad", "android.permission.READ_EXTERNAL_STORAGE")
  appops_set("ai.comma.plus.offroad", "SU", "allow")
  appops_set("ai.comma.plus.offroad", "WIFI_SCAN", "allow")

def appops_set(package, op, mode):
  system(f"LD_LIBRARY_PATH= appops set {package} {op} {mode}")

def pm_grant(package, permission):
  system(f"pm grant {package} {permission}")

def system(cmd):
  try:
    cloudlog.info("running %s" % cmd)
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as e:
    cloudlog.event("running failed",
      cmd=e.cmd,
      output=e.output[-1024:],
      returncode=e.returncode)

# *** external functions ***

def update_apks():
  # install apks
  params = Params()
  dirty = False
  updateApks = {}
  
  try:
    updateApks = json.loads(params.get("UpdateApks"))
  except Exception as ex:
    pass
  
  installed = get_installed_apks()

  install_apks = glob.glob(os.path.join(BASEDIR, "apk/*.apk"))
  for apk in install_apks:
    app = os.path.basename(apk)[:-4]
    if app not in installed:
      installed[app] = None

  cloudlog.info("installed apks %s" % (str(installed),))

  for app in installed.keys():
    apk_path = os.path.join(BASEDIR, "apk/"+app+".apk")
    if not os.path.exists(apk_path):
      continue
    
    fi = updateApks.get(app)
    if fi == None:
      fi = {"install": {"mtime": None, "checksum": None},
            "installed": {"mtime": None, "checksum": None}}
      updateApks[app] = fi
    
    h1 = None
    mtime1 = os.path.getmtime(apk_path)
    if fi["install"]["mtime"] == mtime1:
      h1 = fi["install"]["checksum"]
    else:
      dirty = True
      h1 = hashlib.sha1(open(apk_path, 'rb').read()).hexdigest()
      fi["install"]["checksum"] = h1
      fi["install"]["mtime"] = mtime1
      
    
    h2 = None
    if installed[app] is not None:
      mtime2 = os.path.getmtime(installed[app])
      if fi["installed"]["mtime"] == mtime2:
        h2 = fi["installed"]["checksum"]
      else:
        dirty = True
        h2 = hashlib.sha1(open(installed[app], 'rb').read()).hexdigest()
        fi["installed"]["checksum"] = h2
        fi["installed"]["mtime"] = mtime2
      
      cloudlog.info("comparing version of %s  %s vs %s" % (app, h1, h2))
    else:
      dirty = True
      fi["installed"]["checksum"] = None
      fi["installed"]["mtime"] = None
      
    if h2 is None or h1 != h2:
      cloudlog.info("installing %s" % app)

      success = install_apk(apk_path)
      if not success:
        cloudlog.info("needing to uninstall %s" % app)
        system("pm uninstall %s" % app)
        success = install_apk(apk_path)

      assert success
  if dirty:
    params.put("UpdateApks", json.dumps(updateApks))

def pm_apply_packages(cmd):
  for p in android_packages:
    system("pm %s %s" % (cmd, p))

if __name__ == "__main__":
  update_apks()
