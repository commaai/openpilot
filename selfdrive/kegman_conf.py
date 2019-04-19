import json
import copy
import os
import threading
import time
from selfdrive.swaglog import cloudlog
from common.basedir import BASEDIR

def read_config():
  default_config = {"cameraOffset": "0.06", "lastTrMode": "1", "battChargeMin": "90", "battChargeMax": "95",
                    "wheelTouchSeconds": "1800", "battPercOff": "25", "carVoltageMinEonShutdown": "11200",
                    "brakeStoppingTarget": "0.25", "angle_steers_offset": "0", "brake_distance_extra": "1",
                    "lastALCAMode": "1", "brakefactor": "1.2", "lastGasMode": "0", "lastSloMode": "1",
                    "leadDistance": "5"}

  if os.path.isfile(kegman_file):
    with open(kegman_file, "r") as f:
      try:
        config = json.load(f)
      except:
        cloudlog.exception("reading kegman.json error")
        config = default_config
    if "battPercOff" not in config:
      config.update({"battPercOff": "25"})
    if "carVoltageMinEonShutdown" not in config:
      config.update({"carVoltageMinEonShutdown": "11200"})
    if "brakeStoppingTarget" not in config:
      config.update({"brakeStoppingTarget": "0.25"})
    if "angle_steers_offset" not in config:
      config.update({"angle_steers_offset": "0"})
    if "brake_distance_extra" not in config:  # extra braking distance in m
      config.update({"brake_distance_extra": "1"})
    if "lastALCAMode" not in config:
      config.update({"lastALCAMode": "1"})
    if "brakefactor" not in config:  # brake at 20% higher speeds than what I like
      config.update({"brakefactor": "1.2"})
    if "lastGasMode" not in config:
      config.update({"lastGasMode": "0"})
    if "lastSloMode" not in config:
      config.update({"lastSloMode": "1"})
    if "leadDistance" not in config:  # leadDistance only works for Accord and Insight, have not tested other honda vehicles
      config.update({"leadDistance": "5.0"})

    # force update
    if config["carVoltageMinEonShutdown"] == "11800":
      config.update({"carVoltageMinEonShutdown": "11200"})
    if int(config["wheelTouchSeconds"]) < 200:
      config.update({"wheelTouchSeconds": "1800"})
    if int(config["battChargeMin"]) == 85:
      config.update({"battChargeMin": "90"})
    if int(config["battChargeMax"]) == 90:
      config.update({"battChargeMax": "95"})
  else:
    config = default_config
  return config

def kegman_thread():  # read and write thread; now merges changes from file and variable
  global conf
  global variables_written
  global thread_started
  last_conf = copy.deepcopy(conf)
  try:
    while True:
      time.sleep(30)  # every n seconds check for conf change
      with open(kegman_file, "r") as f:
        conf_tmp = json.load(f)
      if conf != last_conf or conf != conf_tmp:  # if either variable or file has changed
        if conf_tmp != conf:  # if change in file
          changed_keys = []
          for i in conf_tmp:
            try:
              if conf_tmp[i] != conf[i]:
                changed_keys.append(i)
            except:  # if new param from file not existing in variable
              changed_keys.append(i)
          for i in changed_keys:
            if i not in variables_written:
              conf.update({i: conf_tmp[i]})
        if conf != conf_tmp:
          write_config(conf)
        last_conf = copy.deepcopy(conf)
      variables_written = []
  except:
    print("Error in kegman thread!")
    cloudlog.exception("error in kegman thread")
    thread_started = False

def write_config(conf):  # never to be called outside kegman_conf
  try:
    with open(kegman_file, "w") as f:
      json.dump(conf, f, indent=2, sort_keys=True)
      os.chmod(kegman_file, 0o764)
  except IOError:
    os.mkdir("/data")
    with open(kegman_file, "w") as f:
      json.dump(conf, f, indent=2, sort_keys=True)
      os.chmod(kegman_file, 0o764)

def save(data):  # allows for writing multiple key/value pairs
  global conf
  global thread_started
  global variables_written
  if not thread_started and BASEDIR == "/data/openpilot":
    threading.Thread(target=kegman_thread).start()  # automatically start write thread if file needs it
    thread_started = True
    print("Starting thread!")
  for key in data:
    variables_written.append(key)
  conf.update(data)

def get(key_s):  # can get multiple keys from a list
  if type(key_s) == list:
    return [conf[i] if i in conf else None for i in key_s]
  if key_s in conf:
    return conf[key_s]
  else:
    return None

thread_started = False
kegman_file = "/data/kegman.json"
variables_written = []
conf = read_config()