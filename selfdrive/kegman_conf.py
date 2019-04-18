import json
import copy
import os, stat
import threading
import time

class kegman_conf():
  def __init__(self):
    self.conf = self.read_config()
    self.last_conf = copy.deepcopy(self.conf)

  def read_config(self):
    self.element_updated = False
    self.default_config = {"cameraOffset":"0.06", "lastTrMode":"1", "battChargeMin":"90", "battChargeMax":"95", "wheelTouchSeconds":"1800", "battPercOff":"25", "carVoltageMinEonShutdown":"11200", "brakeStoppingTarget":"0.25", "angle_steers_offset":"0" , "brake_distance_extra":"1" , "lastALCAMode":"1" , "brakefactor":"1.2", "lastGasMode":"0" , "lastSloMode":"1", "leadDistance":"5"}

    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        try:
          self.config = json.load(f)
        except:
          self.config = self.default_config
      self.last_conf = copy.deepcopy(self.config)
      if "battPercOff" not in self.config:
        self.config.update({"battPercOff":"25"})
        self.element_updated = True
      if "carVoltageMinEonShutdown" not in self.config:
        self.config.update({"carVoltageMinEonShutdown":"11200"})
        self.element_updated = True
      if "brakeStoppingTarget" not in self.config:
        self.config.update({"brakeStoppingTarget":"0.25"})
        self.element_updated = True
      if "angle_steers_offset" not in self.config:
        self.config.update({"angle_steers_offset":"0"})
        self.element_updated = True
      if "brake_distance_extra" not in self.config: # extra braking distance in m
        self.config.update({"brake_distance_extra":"1"})
        self.element_updated = True
      if "lastALCAMode" not in self.config:
        self.config.update({"lastALCAMode":"1"})
        self.element_updated = True
      if "brakefactor" not in self.config: # brake at 20% higher speeds than what I like
        self.config.update({"brakefactor":"1.2"}) 
        self.element_updated = True
      if "lastGasMode" not in self.config:
        self.config.update({"lastGasMode":"0"}) 
        self.element_updated = True
      if "lastSloMode" not in self.config:
        self.config.update({"lastSloMode":"1"}) 
        self.element_updated = True
      if "leadDistance" not in self.config: # leadDistance only works for Accord and Insight, have not tested other honda vehicles
        self.config.update({"leadDistance":"5.0"})
        self.element_updated = True

      # force update
      if self.config['carVoltageMinEonShutdown'] == "11800":
        self.config.update({"carVoltageMinEonShutdown":"11200"})
        self.element_updated = True
      if int(self.config['wheelTouchSeconds']) < 200:
        self.config.update({"wheelTouchSeconds":"1800"})
        self.element_updated = True
      if int(self.config['battChargeMin']) == 85:
        self.config.update({"battChargeMin":"90"})
        self.element_updated = True
      if int(self.config['battChargeMax']) == 90:
        self.config.update({"battChargeMax":"95"})
        self.element_updated = True
      
      if self.element_updated:      
        self.write_config(self.config)

    else:
      self.config = self.default_config
      self.write_config(self.config)
    return self.config

  def write_config(self, config):
    try:
      # Only write if data has changed
      if (self.last_conf != config):
        #print "Config changed, writing file"
        self.last_conf = copy.deepcopy(config) # cache the current config
        with open('/data/kegman.json', 'w') as f:
          json.dump(config, f, indent=2, sort_keys=True)
          os.chmod("/data/kegman.json", 0o764)
    except IOError:
      os.mkdir('/data')
      with open('/data/kegman.json', 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)
        os.chmod("/data/kegman.json", 0o764)
