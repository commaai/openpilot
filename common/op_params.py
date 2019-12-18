import os
import json
import time
import string
import random
from common.travis_checker import travis


def write_params(params, params_file):
  if not travis:
    with open(params_file, "w") as f:
      json.dump(params, f, indent=2, sort_keys=True)
    os.chmod(params_file, 0o764)


def read_params(params_file, default_params):
  try:
    with open(params_file, "r") as f:
      params = json.load(f)
    return params, True
  except Exception as e:
    print(e)
    params = default_params
    return params, False


class opParams:
  def __init__(self):
    self.default_params = {'camera_offset': {'default': 0.06, 'allowed_types': [float, int], 'description': 'Your camera offset to use in lane_planner.py'},
                           'awareness_factor': {'default': 2.0, 'allowed_types': [float, int], 'description': 'Multiplier for the awareness times'},
                           'lane_hug_direction': {'default': None, 'allowed_types': [type(None), str], 'description': "(NoneType, 'left', 'right'): Direction of your lane hugging, if present. None will disable this modification"},
                           'lane_hug_angle_offset': {'default': 0.0, 'allowed_types': [float, int], 'description': ('This is the angle your wheel reads when driving straight at highway speeds. '
                                                                                                                    'Used to offset desired angle_steers in latcontrol to help fix lane hugging. '
                                                                                                                    'Enter absolute value here, direction is determined by parameter \'lane_hug_direction\'')},
                           'use_car_caching': {'default': True, 'allowed_types': [bool], 'description': 'Whether to use fingerprint caching'},
                           'force_pedal': {'default': False, 'allowed_types': [bool], 'description': "If openpilot isn't recognizing your comma pedal, set this to True"},
                           'following_distance': {'default': None, 'allowed_types': [type(None), float], 'description': 'None has no effect, while setting this to a float will let you change the TR'},
                           'alca_nudge_required': {'default': True, 'allowed_types': [bool], 'description': ('Whether to wait for applied torque to the wheel (nudge) before making lane changes. '
                                                                                                             'If False, lane change will occur IMMEDIATELY after signaling')},
                           'alca_min_speed': {'default': 30.0, 'allowed_types': [float, int], 'description': 'The minimum speed allowed for an automatic lane change (in MPH)'},
                           'min_model_speed': {'default': 20.0, 'allowed_types': [float, int], 'description': 'The minimum speed the model will be allowed to slow down for curves (in MPH)'}}

    self.params = {}
    self.params_file = "/data/op_params.json"
    self.kegman_file = "/data/kegman.json"
    self.last_read_time = time.time()
    self.read_timeout = 1.0  # max frequency to read with self.get(...) (sec)
    self.force_update = False  # replaces values with default params if True, not just add add missing key/value pairs
    self.run_init()  # restores, reads, and updates params

  def create_id(self):  # creates unique identifier to send with sentry errors. please update uniqueID with op_edit.py to your username!
    need_id = False
    if "uniqueID" not in self.params:
      need_id = True
    if "uniqueID" in self.params and self.params["uniqueID"] is None:
      need_id = True
    if need_id:
      random_id = ''.join([random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for i in range(15)])
      self.params["uniqueID"] = random_id

  def add_default_params(self):
    prev_params = dict(self.params)
    if not travis:
      self.create_id()
      for key in self.default_params:
        if self.force_update:
          self.params[key] = self.default_params[key]['default']
        elif key not in self.params:
          self.params[key] = self.default_params[key]['default']
    return prev_params == self.params

  def format_default_params(self):
    return {key: self.default_params[key]['default'] for key in self.default_params}

  def run_init(self):  # does first time initializing of default params, and/or restoring from kegman.json
    if travis:
      self.params = self.format_default_params()
      return
    self.params = self.format_default_params()  # in case any file is corrupted
    to_write = False
    no_params = False
    if os.path.isfile(self.params_file):
      self.params, read_status = read_params(self.params_file, self.format_default_params())
      if read_status:
        to_write = not self.add_default_params()  # if new default data has been added
      else:  # don't overwrite corrupted params, just print to screen
        print("ERROR: Can't read op_params.json file")
    elif os.path.isfile(self.kegman_file):
      to_write = True  # write no matter what
      try:
        with open(self.kegman_file, "r") as f:  # restore params from kegman
          self.params = json.load(f)
          self.add_default_params()
      except:
        print("ERROR: Can't read kegman.json file")
    else:
      no_params = True  # user's first time running a fork with kegman_conf or op_params
    if to_write or no_params:
      write_params(self.params, self.params_file)

  def put(self, key, value):
    self.params.update({key: value})
    write_params(self.params, self.params_file)

  def get(self, key=None, default=None):  # can specify a default value if key doesn't exist
    if (time.time() - self.last_read_time) >= self.read_timeout and not travis:  # make sure we aren't reading file too often
      self.params, read_status = read_params(self.params_file, self.format_default_params())
      self.last_read_time = time.time()
    if key is None:  # get all
      return self.params
    else:
      return self.params[key] if key in self.params else default

  def delete(self, key):
    if key in self.params:
      del self.params[key]
      write_params(self.params, self.params_file)
