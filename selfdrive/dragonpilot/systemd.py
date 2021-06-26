#!/usr/bin/env python3
'''
This is a service that broadcast dp config values to openpilot's messaging queues
'''
import cereal.messaging as messaging
import time

from common.dp_conf import confs, get_struct_name, to_struct_val
from common.params import Params, put_nonblocking
import subprocess
import re
import os
from selfdrive.hardware import HARDWARE
params = Params()
from common.realtime import sec_since_boot
from common.i18n import get_locale
from common.dp_common import param_get, get_last_modified
from common.dp_time import LAST_MODIFIED_SYSTEMD
from selfdrive.dragonpilot.dashcamd import Dashcamd
from selfdrive.hardware import EON

PARAM_PATH = params.get_params_path() + '/d/'

DELAY = 0.5 # 2hz
HERTZ = 1/DELAY

last_modified_confs = {}

def confd_thread():
  sm = messaging.SubMaster(['deviceState'])
  pm = messaging.PubMaster(['dragonConf'])

  last_dp_msg = None
  frame = 0
  update_params = False
  modified = None
  last_modified = None
  last_modified_check = None
  started = False
  free_space = 1
  battery_percent = 0
  overheat = False
  last_charging_ctrl = False
  last_started = False
  dashcamd = Dashcamd()
  dashcam_recorded = False
  last_dashcam_recorded = False

  while True:
    start_sec = sec_since_boot()
    msg = messaging.new_message('dragonConf')
    if last_dp_msg is not None:
      msg.dragonConf = last_dp_msg

    '''
    ===================================================
    load thermald data every 3 seconds
    ===================================================
    '''
    if frame % (HERTZ * 3) == 0:
      started, free_space, battery_percent, overheat = pull_thermald(frame, sm, started, free_space, battery_percent, overheat)
    setattr(msg.dragonConf, get_struct_name('dp_thermal_started'), started)
    setattr(msg.dragonConf, get_struct_name('dp_thermal_overheat'), overheat)
    '''
    ===================================================
    hotspot on boot
    we do it after 30 secs just in case
    ===================================================
    '''
    if frame == (HERTZ * 30) and param_get("dp_hotspot_on_boot", "bool", False):
      os.system("service call wifi 37 i32 0 i32 1 &")
    '''
    ===================================================
    check dp_last_modified every second
    ===================================================
    '''
    if not update_params:
      last_modified_check, modified = get_last_modified(LAST_MODIFIED_SYSTEMD, last_modified_check, modified)
      if last_modified != modified:
        update_params = True
        last_modified = modified
    '''
    ===================================================
    conditionally set update_params to true 
    ===================================================
    '''
    # force updating param when `started` changed
    if last_started != started:
      update_params = True
      last_started = started

    if frame == 0:
      update_params = True
    '''
    ===================================================
    conditionally update dp param base on stock param 
    ===================================================
    '''
    # if update_params and params.get("LaneChangeEnabled") == b"1":
    #   params.put("dp_steering_on_signal", "0")
    '''
    ===================================================
    push param vals to message
    ===================================================
    '''
    if update_params:
      msg = update_conf_all(confs, msg, frame == 0)
      update_params = False
    '''
    ===================================================
    push once
    ===================================================
    '''
    if frame == 0:
      setattr(msg.dragonConf, get_struct_name('dp_locale'), get_locale())
      if (not os.path.isfile("/data/params/d/GithubSshKeys") or params.get("GithubSshKeys") == '') and \
          os.path.isfile("/data/data/com.termux/files/home/setup_keys"):
        os.system("cp /data/data/com.termux/files/home/setup_keys /data/params/d/GithubSshKeys")

    '''
    ===================================================
    push ip addr every 10 secs
    ===================================================
    '''
    if frame % (HERTZ * 10) == 0:
      msg = update_ip(msg)
    '''
    ===================================================
    update msg based on some custom logic
    ===================================================
    '''
    msg = update_custom_logic(msg)
    '''
    ===================================================
    battery ctrl every 30 secs
    PowerMonitor in thermald turns back on every mins
    so lets turn it off more frequent
    ===================================================
    '''
    # if frame % (HERTZ * 30) == 0:
    #   last_charging_ctrl = process_charging_ctrl(msg, last_charging_ctrl, battery_percent)
    '''
    ===================================================
    dashcam
    ===================================================
    '''
    if msg.dragonConf.dpDashcamd:
      if frame % HERTZ == 0:
        dashcamd.run(started, free_space)
        dashcam_recorded = True
      if not started:
        dashcam_recorded = False
    else:
      dashcam_recorded = False

    if not dashcam_recorded and last_dashcam_recorded:
      dashcamd.stop()

    last_dashcam_recorded = dashcam_recorded
    '''
    ===================================================
    finalise
    ===================================================
    '''
    last_dp_msg = msg.dragonConf
    pm.send('dragonConf', msg)
    frame += 1
    sleep = DELAY-(sec_since_boot() - start_sec)
    if sleep > 0:
      time.sleep(sleep)

def update_conf(msg, conf, first_run = False):
  conf_type = conf.get('conf_type')

  # skip checking since modified date time hasn't been changed.
  if (last_modified_confs.get(conf['name'])) is not None and last_modified_confs.get(conf['name']) == os.stat(PARAM_PATH + conf['name']).st_mtime:
    return msg

  if 'param' in conf_type and 'struct' in conf_type:
    update_this_conf = True

    if not first_run:
      update_once = conf.get('update_once')
      if update_once is not None and update_once is True:
        return msg
      if update_this_conf:
        update_this_conf = check_dependencies(msg, conf)

    if update_this_conf:
      msg = set_message(msg, conf)
      if os.path.isfile(PARAM_PATH + conf['name']):
        last_modified_confs[conf['name']] = os.stat(PARAM_PATH + conf['name']).st_mtime
  return msg

def update_conf_all(confs, msg, first_run = False):
  for conf in confs:
    msg = update_conf(msg, conf, first_run)
  return msg

def process_charging_ctrl(msg, last_charging_ctrl, battery_percent):
  charging_ctrl = msg.dragonConf.dpChargingCtrl
  if last_charging_ctrl != charging_ctrl:
    HARDWARE.set_battery_charging(True)
  if charging_ctrl:
    if battery_percent >= msg.dragonConf.dpDischargingAt and HARDWARE.get_battery_charging():
      HARDWARE.set_battery_charging(False)
    elif battery_percent <= msg.dragonConf.dpChargingAt and not HARDWARE.get_battery_charging():
      HARDWARE.set_battery_charging(True)
  return charging_ctrl


def pull_thermald(frame, sm, started, free_space, battery_percent, overheat):
  sm.update(0)
  if sm.updated['deviceState']:
    started = sm['deviceState'].started
    free_space = sm['deviceState'].freeSpacePercent
    battery_percent = sm['deviceState'].batteryPercent
    overheat = sm['deviceState'].thermalStatus >= 2
  return started, free_space, battery_percent, overheat

def update_custom_logic(msg):
  if msg.dragonConf.dpAtl:
    msg.dragonConf.dpAllowGas = True
    msg.dragonConf.dpFollowingProfileCtrl = False
    msg.dragonConf.dpAccelProfileCtrl = False
    msg.dragonConf.dpGearCheck = False
  if msg.dragonConf.dpLcMinMph > msg.dragonConf.dpLcAutoMinMph:
    put_nonblocking('dp_lc_auto_min_mph', str(msg.dragonConf.dpLcMinMph))
    msg.dragonConf.dpLcAutoMinMph = msg.dragonConf.dpLcMinMph
  # if msg.dragonConf.dpSrCustom <= 4.99 and msg.dragonConf.dpSrStock > 0:
  #   put_nonblocking('dp_sr_custom', str(msg.dragonConf.dpSrStock))
  #   msg.dragonConf.dpSrCustom = msg.dragonConf.dpSrStock
  # if msg.dragonConf.dpAppWaze or msg.dragonConf.dpAppHr:
  #   msg.dragonConf.dpDrivingUi = False
  # if not msg.dragonConf.dpDriverMonitor:
  #   msg.dragonConf.dpUiFace = False
  return msg


def update_ip(msg):
  val = 'N/A'
  if EON:
    try:
      result = subprocess.check_output(["ifconfig", "wlan0"], encoding='utf8')
      val = re.findall(r"inet addr:((\d+\.){3}\d+)", result)[0][0]
    except:
      pass
  setattr(msg.dragonConf, get_struct_name('dp_ip_addr'), val)
  return msg


def set_message(msg, conf):
  val = params.get(conf['name'], encoding='utf8')
  if val is not None:
    val = val.rstrip('\x00')
  else:
    val = conf.get('default')
    params.put(conf['name'], str(val))
  struct_val = to_struct_val(conf['name'], val)
  orig_val = struct_val
  if struct_val is not None:
    if conf.get('min') is not None:
      struct_val = max(struct_val, conf.get('min'))
    if conf.get('max') is not None:
      struct_val = min(struct_val, conf.get('max'))
  if orig_val != struct_val:
    params.put(conf['name'], str(struct_val))
  setattr(msg.dragonConf, get_struct_name(conf['name']), struct_val)
  return msg

def check_dependencies(msg, conf):
  passed = True
  # if has dependency and the depend param val is not in depend_vals, we dont update that conf val
  # this should reduce chance of reading unnecessary params
  dependencies = conf.get('depends')
  if dependencies is not None:
    for dependency in dependencies:
      if getattr(msg.dragonConf, get_struct_name(dependency['name'])) not in dependency['vals']:
        passed = False
        break
  return passed

def main():
  confd_thread()

if __name__ == "__main__":
  main()
