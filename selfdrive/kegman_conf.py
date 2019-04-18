import json
import copy
import os
import threading
import time
from selfdrive.swaglog import cloudlog
from common.basedir import BASEDIR
lock = threading.Lock()

class kegman_conf():
  def __init__(self, read_only=False):  # start thread by default
    self.conf = self.read_config()
    # when you import kegman_conf and only use it to read data, you can specify read_only in your import as to not start the write_thread
    if not read_only and BASEDIR == "/data/openpilot":  # if not travis test nor read only
      threading.Thread(target=self.kegman_thread).start()

  def read_config(self):
    default_config = {"cameraOffset":"0.06", "lastTrMode":"1", "battChargeMin":"90", "battChargeMax":"95", "wheelTouchSeconds":"1800", "battPercOff":"25", "carVoltageMinEonShutdown":"11200", "brakeStoppingTarget":"0.25", "angle_steers_offset":"0" , "brake_distance_extra":"1" , "lastALCAMode":"1" , "brakefactor":"1.2", "lastGasMode":"0" , "lastSloMode":"1", "leadDistance":"5"}

    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        try:
          config = json.load(f)
        except:
          cloudlog.exception("reading kegman.json error")
          config = default_config
      self.last_conf = copy.deepcopy(config)
      if "battPercOff" not in config:
        config.update({"battPercOff":"25"})
      if "carVoltageMinEonShutdown" not in config:
        config.update({"carVoltageMinEonShutdown":"11200"})
      if "brakeStoppingTarget" not in config:
        config.update({"brakeStoppingTarget":"0.25"})
      if "angle_steers_offset" not in config:
        config.update({"angle_steers_offset":"0"})
      if "brake_distance_extra" not in config: # extra braking distance in m
        config.update({"brake_distance_extra":"1"})
      if "lastALCAMode" not in config:
        config.update({"lastALCAMode":"1"})
      if "brakefactor" not in config: # brake at 20% higher speeds than what I like
        config.update({"brakefactor":"1.2"})
      if "lastGasMode" not in config:
        config.update({"lastGasMode":"0"})
      if "lastSloMode" not in config:
        config.update({"lastSloMode":"1"})
      if "leadDistance" not in config: # leadDistance only works for Accord and Insight, have not tested other honda vehicles
        config.update({"leadDistance":"5.0"})

      # force update
      if config['carVoltageMinEonShutdown'] == "11800":
        config.update({"carVoltageMinEonShutdown":"11200"})
      if int(config['wheelTouchSeconds']) < 200:
        config.update({"wheelTouchSeconds":"1800"})
      if int(config['battChargeMin']) == 85:
        config.update({"battChargeMin":"90"})
      if int(config['battChargeMax']) == 90:
        config.update({"battChargeMax":"95"})
    else:
      config = default_config
    return config

  def kegman_thread(self):  # do reading and writing in one thread
    last_conf = copy.deepcopy(self.conf)
    while True:
      if self.conf != last_conf:
        self.write_config()
        last_conf = copy.deepcopy(self.conf)  # cache the current config
      time.sleep(15)  # every n seconds check for conf change
      with open('/data/kegman.json', 'r') as f:
        conf_tmp = json.load(f)
        if conf_tmp != self.conf:
          self.conf = conf_tmp
          last_conf = copy.deepcopy(self.conf)

  def write_config(self):  # never to be called outside kegman_conf
    try:
      with open('/data/kegman.json', 'w') as f:
        json.dump(self.conf, f, indent=2, sort_keys=True)
        os.chmod("/data/kegman.json", 0o764)
    except IOError:
      os.mkdir('/data')
      with open('/data/kegman.json', 'w') as f:
        json.dump(self.conf, f, indent=2, sort_keys=True)
        os.chmod("/data/kegman.json", 0o764)