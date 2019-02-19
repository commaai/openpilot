import json
import os

class kegman_conf():
  def __init__(self):
    self.conf = self.read_config()

  def read_config(self):
    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        self.config = json.load(f)
    else:
      self.config = {"cameraOffset":"0.06", "lastTrMode":"1", "battChargeMin":"60", "battChargeMax":"70", "wheelTouchSeconds":"180" }
      self.write_config(self.config)
    return self.config

  def write_config(self, config):
    with open('/data/kegman.json', 'w') as f:
      json.dump(self.config, f, indent=2, sort_keys=True)
      os.chmod("/data/kegman.json", 0o764)

