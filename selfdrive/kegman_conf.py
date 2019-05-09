import json
import os

class kegman_conf():
  def __init__(self, CP=None):
    self.conf = self.read_config()
    if CP is not None:
      self.init_config(CP)

  def init_config(self, CP):
    write_conf = False
    if self.conf['tuneGernby'] != "1":
      self.conf['tuneGernby'] = str(1)
      write_conf = True
    if self.conf['Kp'] == "-1":
      self.conf['Kp'] = str(round(CP.steerKpV[0],3))
      write_conf = True
    if self.conf['Ki'] == "-1":
      self.conf['Ki'] = str(round(CP.steerKiV[0],3))
      write_conf = True

    if write_conf:
      self.write_config(self.config)

  def read_config(self):
    self.element_updated = False

    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        self.config = json.load(f)

      if "battPercOff" not in self.config:
        self.config.update({"battPercOff":"25"})
        self.config.update({"carVoltageMinEonShutdown":"11800"})
        self.config.update({"brakeStoppingTarget":"0.25"})
        self.element_updated = True

      if "tuneGernby" not in self.config:
        self.config.update({"tuneGernby":"1"})
        self.config.update({"Kp":"-1"})
        self.config.update({"Ki":"-1"})
	self.element_updated = True

      if "liveParams" not in self.config:
        self.config.update({"liveParams":"1"})
        self.element_updated = True


      # Force update battery charge limits to higher values for Big Model
      #if self.config['battChargeMin'] != "75":
      #  self.config.update({"battChargeMin":"75"})
      #  self.config.update({"battChargeMax":"80"})
      #  self.element_updated = True

      if self.element_updated:
        print("updated")
        self.write_config(self.config)

    else:
      self.config = {"cameraOffset":"0.06", "lastTrMode":"1", "battChargeMin":"60", "battChargeMax":"70", \
                     "wheelTouchSeconds":"180", "battPercOff":"25", "carVoltageMinEonShutdown":"11800", \
                     "brakeStoppingTarget":"0.25", "tuneGernby":"1", \
                     "Kp":"-1", "Ki":"-1", "liveParams":"1"}

      self.write_config(self.config)
    return self.config

  def write_config(self, config):
    try:
      with open('/data/kegman.json', 'w') as f:
        json.dump(self.config, f, indent=2, sort_keys=True)
        os.chmod("/data/kegman.json", 0o764)
    except IOError:
      os.mkdir('/data')
      with open('/data/kegman.json', 'w') as f:
        json.dump(self.config, f, indent=2, sort_keys=True)
        os.chmod("/data/kegman.json", 0o764)
