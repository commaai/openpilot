import json
import os

class kegman_conf():
  def __init__(self):
    self.conf = self.read_config()

  def read_config(self):
    self.element_updated = False

    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        self.config = json.load(f)
      if "battPercOff" not in self.config:
        self.config.update({"battPercOff":"25"})
        self.element_updated = True
      if "carVoltageMinEonShutdown" not in self.config:
        self.config.update({"carVoltageMinEonShutdown":"11800"})
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
      if "lastALCAMode" not in self.config: # extra braking distance in m
        self.config.update({"lastALCAMode":"1"})
        self.element_updated = True

      # Force update battery charge limits to higher values for Big Model
      #if self.config['battChargeMin'] != "75":
      #  self.config.update({"battChargeMin":"75"})
      #  self.config.update({"battChargeMax":"80"})
      #  self.element_updated = True
      
      if self.element_updated:      
        self.write_config(self.config)

    else:
      self.config = {"cameraOffset":"0.06", "lastTrMode":"1", "battChargeMin":"85", "battChargeMax":"90", "wheelTouchSeconds":"1800", "battPercOff":"25", "carVoltageMinEonShutdown":"11800", "brakeStoppingTarget":"0.25", "angle_steers_offset":"0" , "brake_distance_extra":"1" , "lastALCAMode":"1"}
      self.write_config(self.config)
    return self.config

  def write_config(self, config):
    with open('/data/kegman.json', 'w') as f:
      json.dump(self.config, f, indent=2, sort_keys=True)
      os.chmod("/data/kegman.json", 0o764)
