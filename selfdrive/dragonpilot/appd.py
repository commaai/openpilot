#!/usr/bin/env python3.7
import time
import subprocess
from cereal import log
import cereal.messaging as messaging
ThermalStatus = log.DeviceState.ThermalStatus
from selfdrive.swaglog import cloudlog
from common.params import Params, put_nonblocking
params = Params()
from math import floor
import re
import os
from common.dp_common import is_online
from common.dp_conf import get_struct_name
from common.realtime import sec_since_boot
import psutil

is_online = is_online()
# auto_update = params.get("dp_app_auto_update", encoding='utf8') == "1"

class App():

  # app type
  TYPE_GPS = 0
  TYPE_SERVICE = 1
  TYPE_FULLSCREEN = 2
  TYPE_UTIL = 3
  TYPE_ANDROID_AUTO = 4

  # manual switch stats
  MANUAL_OFF = -1
  MANUAL_IDLE = 0
  MANUAL_ON = 1

  def appops_set(self, package, op, mode):
    self.system(f"LD_LIBRARY_PATH= appops set {package} {op} {mode}")

  def pm_grant(self, package, permission):
    self.system(f"pm grant {package} {permission}")

  def set_package_permissions(self):
    if self.permissions is not None:
      for permission in self.permissions:
        self.pm_grant(self.app, permission)
    if self.opts is not None:
      for opt in self.opts:
        self.appops_set(self.app, opt, "allow")

  def __init__(self, app, start_cmd, enable_param, auto_run_param, manual_ctrl_param, app_type, check_crash, permissions, opts):
    self.app = app
    # main activity
    self.start_cmd = start_cmd
    # read enable param
    self.enable_param = enable_param
    self.enable_struct = get_struct_name(enable_param) if enable_param is not None else None
    # read auto run param
    self.auto_run_struct = get_struct_name(auto_run_param) if auto_run_param is not None else None
    # read manual run param
    self.manual_ctrl_param = manual_ctrl_param if manual_ctrl_param is not None else None
    self.manual_ctrl_struct = get_struct_name(manual_ctrl_param) if manual_ctrl_param is not None else None
    # if it's a service app, we do not kill if device is too hot
    self.app_type = app_type
    # app permissions
    self.permissions = permissions
    # app options
    self.opts = opts

    self.own_apk = "/sdcard/apks/" + self.app + ".apk"
    self.has_own_apk = os.path.exists(self.own_apk)
    self.is_installed = False
    self.is_enabled = False
    self.last_is_enabled = False
    self.is_auto_runnable = False
    self.is_running = False
    self.manual_ctrl_status = self.MANUAL_IDLE
    self.manually_ctrled = False
    self.init = False
    self.check_crash = check_crash

  def is_crashed(self):
    return getattr(self, self.enable_param + "_is_crashed")()

  def dp_app_hr_is_crashed(self):
    try:
      result = subprocess.check_output(["dumpsys", "activity", "gb.xxy.hr"], encoding='utf8')
      print("is_crash = %s" % "ACTIVITY" in result)
      return "ACTIVITY" not in result
    except (subprocess.CalledProcessError, IndexError) as e:
      return False

  def get_remote_version(self):
    apk = self.app + ".apk"
    try:
      url = "https://raw.githubusercontent.com/dragonpilot-community/apps/%s/VERSION" % apk
      return subprocess.check_output(["curl", "-H", "'Cache-Control: no-cache'", "-s", url]).decode('utf8').rstrip()
    except subprocess.CalledProcessError as e:
      pass
    return None

  def uninstall_app(self):
    try:
      local_version = self.get_local_version()
      if local_version is not None:
        subprocess.check_output(["pm","uninstall", self.app])
        self.is_installed = False
    except subprocess.CalledProcessError as e:
      pass

  def update_app(self):
    if self.has_own_apk:
      try:
        subprocess.check_output(["pm","install","-r",self.own_apk])
        self.is_installed = True
      except subprocess.CalledProcessError as e:
        self.is_installed = False
    else:
      apk = self.app + ".apk"
      apk_path = "/sdcard/" + apk
      try:
        os.remove(apk_path)
      except (OSError, FileNotFoundError) as e:
        pass

      self.uninstall_app()
      # if local_version is not None:
      #   try:
      #     subprocess.check_output(["pm","uninstall", self.app], stderr=subprocess.STDOUT, shell=True)
      #   except subprocess.CalledProcessError as e:
      #     pass
      try:
        url = "https://raw.githubusercontent.com/dragonpilot-community/apps/%s/%s" % (apk, apk)
        subprocess.check_output(["curl","-o", apk_path,"-LJO", url])
        subprocess.check_output(["pm","install","-r",apk_path])
        self.is_installed = True
      except subprocess.CalledProcessError as e:
        self.is_installed = False
      try:
        os.remove(apk_path)
      except (OSError, FileNotFoundError) as e:
        pass

  def get_local_version(self):
    try:
      result = subprocess.check_output(["dumpsys", "package", self.app, "|", "grep", "versionName"], encoding='utf8')
      if len(result) > 12:
        return re.findall(r"versionName=(.*)", result)[0]
    except (subprocess.CalledProcessError, IndexError) as e:
      pass
    return None

  def init_vars(self, dragonconf):
    self.is_enabled = getattr(dragonconf, self.enable_struct)

    if self.is_enabled:
      local_version = self.get_local_version()
      if local_version is not None:
        self.is_installed = True

      if self.has_own_apk and not self.is_installed:
        self.update_app()

      elif is_online:
        if local_version is None:
          self.update_app()
        else:
          remote_version = self.get_remote_version() if not self.own_apk else local_version
          if remote_version is not None and local_version != remote_version:
            self.update_app()
      if self.is_installed:
        self.set_package_permissions()
    else:
      self.uninstall_app()

    if self.manual_ctrl_param is not None and getattr(dragonconf, self.manual_ctrl_struct) != self.MANUAL_IDLE:
      put_nonblocking(self.manual_ctrl_param, str(self.MANUAL_IDLE))
    self.init = True

  def read_params(self, dragonconf):
    if not self.init:
      self.init_vars(dragonconf)

    self.last_is_enabled = self.is_enabled
    self.is_enabled = False if self.enable_struct is None else getattr(dragonconf, self.enable_struct)

    if self.is_installed:
      if self.is_enabled:
        # a service app should run automatically and not manual controllable.
        if self.app_type in [App.TYPE_SERVICE]:
          self.is_auto_runnable = True
          self.manual_ctrl_status = self.MANUAL_IDLE
        else:
          self.manual_ctrl_status = self.MANUAL_IDLE if self.manual_ctrl_param is None else getattr(dragonconf, self.manual_ctrl_struct)
          if self.manual_ctrl_status == self.MANUAL_IDLE:
            self.is_auto_runnable = False if self.auto_run_struct is None else getattr(dragonconf, self.auto_run_struct)
      else:
        if self.last_is_enabled:
          self.uninstall_app()
        self.is_auto_runnable = False
        self.manual_ctrl_status = self.MANUAL_IDLE
        self.manually_ctrled = False
    else:
      if not self.last_is_enabled and self.is_enabled:
        self.update_app()

  def run(self, force = False):
    if self.is_installed and (force or self.is_enabled):
      # app is manually ctrl, we record that
      if self.manual_ctrl_param is not None and self.manual_ctrl_status == self.MANUAL_ON:
        put_nonblocking(self.manual_ctrl_param, '0')
        put_nonblocking('dp_last_modified', str(floor(time.time())))
        self.manually_ctrled = True
        self.is_running = False

      # only run app if it's not running
      if force or not self.is_running:
        self.system("pm enable %s" % self.app)

        if self.app_type == self.TYPE_SERVICE:
          self.appops_set(self.app, "android:mock_location", "allow")
        self.system(self.start_cmd)
    self.is_running = True

  def kill(self, force = False):
    if self.is_installed and (force or self.is_enabled):
      # app is manually ctrl, we record that
      if self.manual_ctrl_param is not None and self.manual_ctrl_status == self.MANUAL_OFF:
        put_nonblocking(self.manual_ctrl_param, '0')
        self.manually_ctrled = True
        self.is_running = True

      # only kill app if it's running
      if force or self.is_running:
        if self.app_type == self.TYPE_SERVICE:
          self.appops_set(self.app, "android:mock_location", "deny")

        self.system("pkill %s" % self.app)
        self.is_running = False

  def system(self, cmd):
    try:
      subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
      cloudlog.event("running failed",
                     cmd=e.cmd,
                     output=e.output[-1024:],
                     returncode=e.returncode)

def init_apps(apps):
  apps.append(App(
    "cn.dragonpilot.gpsservice",
    "am startservice cn.dragonpilot.gpsservice/cn.dragonpilot.gpsservice.MainService",
    "dp_app_ext_gps",
    None,
    None,
    App.TYPE_SERVICE,
    False,
    [],
    [],
  ))
  apps.append(App(
    "com.mixplorer",
    "am start -n com.mixplorer/com.mixplorer.activities.BrowseActivity",
    "dp_app_mixplorer",
    None,
    "dp_app_mixplorer_manual",
    App.TYPE_UTIL,
    False,
    [
      "android.permission.READ_EXTERNAL_STORAGE",
      "android.permission.WRITE_EXTERNAL_STORAGE",
    ],
    [],
  ))
  apps.append(App(
    "com.tomtom.speedcams.android.map",
    "am start -n com.tomtom.speedcams.android.map/com.tomtom.speedcams.android.activities.SpeedCamActivity",
    "dp_app_tomtom",
    "dp_app_tomtom_auto",
    "dp_app_tomtom_manual",
    App.TYPE_GPS,
    False,
    [
      "android.permission.ACCESS_FINE_LOCATION",
      "android.permission.ACCESS_COARSE_LOCATION",
      "android.permission.READ_EXTERNAL_STORAGE",
      "android.permission.WRITE_EXTERNAL_STORAGE",
    ],
    [
      "SYSTEM_ALERT_WINDOW",
    ]
  ))
  # apps.append(App(
  #   "tw.com.ainvest.outpack",
  #   "am start -n tw.com.ainvest.outpack/tw.com.ainvest.outpack.ui.MainActivity",
  #   "dp_app_aegis",
  #   "dp_app_aegis_auto",
  #   "dp_app_aegis_manual",
  #   App.TYPE_GPS,
  #   False,
  #   [
  #     "android.permission.ACCESS_FINE_LOCATION",
  #     "android.permission.READ_EXTERNAL_STORAGE",
  #     "android.permission.WRITE_EXTERNAL_STORAGE",
  #   ],
  #   [
  #     "SYSTEM_ALERT_WINDOW",
  #   ]
  # ))
  # apps.append(App(
  #   "com.autonavi.amapauto",
  #   "am start -n com.autonavi.amapauto/com.autonavi.amapauto.MainMapActivity",
  #   "dp_app_autonavi",
  #   "dp_app_autonavi_auto",
  #   "dp_app_autonavi_manual",
  #   App.TYPE_GPS,
  #   False,
  #   [
  #     "android.permission.ACCESS_FINE_LOCATION",
  #     "android.permission.ACCESS_COARSE_LOCATION",
  #     "android.permission.READ_EXTERNAL_STORAGE",
  #     "android.permission.WRITE_EXTERNAL_STORAGE",
  #   ],
  #   [
  #     "SYSTEM_ALERT_WINDOW",
  #   ]
  # ))
  # # pm disable gb.xxy.hr && pm enable gb.xxy.hr && am broadcast -a "gb.xxy.hr.WIFI_START"
  # apps.append(App(
  #   "gb.xxy.hr",
  #   "am start -n gb.xxy.hr/.MainActivity && pm disable gb.xxy.hr && pm enable gb.xxy.hr && am broadcast -a gb.xxy.hr.WIFI_START",
  #   "dp_app_hr",
  #   None,
  #   "dp_app_hr_manual",
  #   App.TYPE_ANDROID_AUTO,
  #   True,
  #   [
  #     "android.permission.ACCESS_FINE_LOCATION",
  #     "android.permission.ACCESS_COARSE_LOCATION",
  #     "android.permission.READ_EXTERNAL_STORAGE",
  #     "android.permission.WRITE_EXTERNAL_STORAGE",
  #     "android.permission.RECORD_AUDIO",
  #   ],
  #   [],
  # ))

def main():
  apps = []

  last_started = False
  sm = messaging.SubMaster(['dragonConf'])

  frame = 0
  start_delay = None
  stop_delay = None
  allow_auto_run = True

  last_overheat = False
  init_done = False
  dragon_conf_msg = None

  next_check_process_frame = 0

  while 1: #has_enabled_apps:
    start_sec = sec_since_boot()
    if not init_done:
      if frame >= 10:
        init_apps(apps)
        sm.update()
        dragon_conf_msg = sm['dragonConf']
        init_done = True
    else:
      sm.update(1000)
      if sm.updated['dragonConf']:
        dragon_conf_msg = sm['dragonConf']
      else:
        continue
      enabled_apps = []
      has_fullscreen_apps = False
      has_check_crash = False
      for app in apps:
        # read params loop
        app.read_params(dragon_conf_msg)
        if app.last_is_enabled and not app.is_enabled and app.is_running:
          app.kill(True)

        if app.is_enabled:
          if not has_fullscreen_apps and app.app_type in [App.TYPE_FULLSCREEN, App.TYPE_ANDROID_AUTO]:
            has_fullscreen_apps = True
          if not has_check_crash and app.check_crash:
            has_check_crash = True

          # process manual ctrl apps
          if app.manual_ctrl_status != App.MANUAL_IDLE:
            app.run(True) if app.manual_ctrl_status == App.MANUAL_ON else app.kill(True)

          enabled_apps.append(app)

      started = dragon_conf_msg.dpThermalStarted
      # when car is running
      if started:
        # we run service apps and kill all util apps
        # only run once
        if last_started != started:
          for app in enabled_apps:
            if app.app_type in [App.TYPE_SERVICE]:
              app.run()
            elif app.app_type == App.TYPE_UTIL:
              app.kill()

        stop_delay = None
        # apps start 5 secs later
        if start_delay is None:
          start_delay = frame + 5

        if not dragon_conf_msg.dpThermalOverheat:
          allow_auto_run = True
          # when temp reduce from red to yellow, we add start up delay as well
          # so apps will not start up immediately
          if last_overheat:
            start_delay = frame + 60
        else:
          allow_auto_run = False
        last_overheat = dragon_conf_msg.dpThermalOverheat

        # only run apps that's not manually ctrled
        if has_check_crash and frame >= next_check_process_frame:
          for app in enabled_apps:
            if app.is_running and app.check_crash and app.is_crashed():
              app.kill()
          next_check_process_frame = frame + 15

        for app in enabled_apps:
          if not app.manually_ctrled:
            if has_fullscreen_apps:
              if app.app_type in [App.TYPE_FULLSCREEN, App.TYPE_ANDROID_AUTO]:
                app.run()
              elif app.app_type in [App.TYPE_GPS, App.TYPE_UTIL]:
                app.kill()
            else:
              if not allow_auto_run:
                app.kill()
              else:
                if frame >= start_delay and app.is_auto_runnable and app.app_type == App.TYPE_GPS:
                  app.run()
      # when car is stopped
      else:
        start_delay = None
        # set delay to 30 seconds
        if stop_delay is None:
          stop_delay = frame + 30

        for app in enabled_apps:
          if app.is_running and not app.manually_ctrled:
            if has_fullscreen_apps or frame >= stop_delay:
              app.kill()

      if last_started != started:
        for app in enabled_apps:
          app.manually_ctrled = False

      last_started = started
    frame += 1
    sleep = 1 - (sec_since_boot() - start_sec)
    if sleep > 0:
      time.sleep(sleep)

def system(cmd):
  try:
    cloudlog.info("running %s" % cmd)
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as e:
    cloudlog.event("running failed",
                   cmd=e.cmd,
                   output=e.output[-1024:],
                   returncode=e.returncode)

if __name__ == "__main__":
  main()
