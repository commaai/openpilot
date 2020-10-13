import os
import subprocess
import glob
import hashlib
import shutil
import threading
import time
from common.basedir import BASEDIR
from selfdrive.swaglog import cloudlog

android_packages = ("ai.comma.plus.offroad",)

def get_installed_apks():
  dat = subprocess.check_output(["pm", "list", "packages", "-f"], encoding='utf8').strip().split("\n")
  ret = {}
  for x in dat:
    if x.startswith("package:"):
      v, k = x.split("package:")[1].split("=")
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
  def f():
    set_package_permissions()
    system("am start -n ai.comma.plus.offroad/.MainActivity")
  threading.Thread(target=f).start()

def extract_current_permissions(dump): #  Doesn't look nice, but the output from dumpsys is also not very nice so blame them
  perms = dump.split("runtime permissions")[1]
  perms2 = perms.split("\\n")
  perms3 = [p.replace(" ","") for p in perms2]
  permsfiltered = [p.split(":")[0].split("android.permission.")[1] for p in perms3 if len(p) > 20]
  return permsfiltered

def set_package_permissions():
  init = time.time()
  out = subprocess.Popen(['dumpsys', 'package', 'ai.comma.plus.offroad'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  stdout, _ = out.communicate()
  given = extract_current_permissions(str(stdout))
  wanted_permissions = ["ACCESS_FINE_LOCATION", "READ_PHONE_STATE", "READ_EXTERNAL_STORAGE"]
  for permission in wanted_permissions:
    if permission not in given:
      pm_grant("ai.comma.plus.offroad", "android.permission."+permission)

  appops_set("ai.comma.plus.offroad", "SU", "allow")
  appops_set("ai.comma.plus.offroad", "WIFI_SCAN", "allow")
  print("Time spent with android:", str(time.time()-init))
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
  installed = get_installed_apks()

  install_apks = glob.glob(os.path.join(BASEDIR, "apk/*.apk"))
  for apk in install_apks:
    app = os.path.basename(apk)[:-4]
    if app not in installed:
      installed[app] = None

  cloudlog.info("installed apks %s" % (str(installed), ))

  for app in installed.keys():
    apk_path = os.path.join(BASEDIR, "apk/"+app+".apk")
    if not os.path.exists(apk_path):
      continue

    h1 = hashlib.sha1(open(apk_path, 'rb').read()).hexdigest()
    h2 = None
    if installed[app] is not None:
      h2 = hashlib.sha1(open(installed[app], 'rb').read()).hexdigest()
      cloudlog.info("comparing version of %s  %s vs %s" % (app, h1, h2))

    if h2 is None or h1 != h2:
      cloudlog.info("installing %s" % app)

      success = install_apk(apk_path)
      if not success:
        cloudlog.info("needing to uninstall %s" % app)
        system("pm uninstall %s" % app)
        success = install_apk(apk_path)

      assert success

def pm_apply_packages(cmd):
  def f():
    for p in android_packages:
      system("pm %s %s" % (cmd, p))
  threading.Thread(target=f).start()

if __name__ == "__main__":
  update_apks()
