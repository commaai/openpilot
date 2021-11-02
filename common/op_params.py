#!/usr/bin/env python3
import os
import json
from atomicwrites import atomic_write
from common.colors import COLORS
from common.travis_checker import BASEDIR
from selfdrive.hardware import TICI
try:
  from common.realtime import sec_since_boot
except ImportError:
  import time
  sec_since_boot = time.time

warning = lambda msg: print('{}opParams WARNING: {}{}'.format(COLORS.WARNING, msg, COLORS.ENDC))
error = lambda msg: print('{}opParams ERROR: {}{}'.format(COLORS.FAIL, msg, COLORS.ENDC))

NUMBER = [float, int]  # value types
NONE_OR_NUMBER = [type(None), float, int]

BASEDIR = os.path.dirname(BASEDIR)
PARAMS_DIR = os.path.join(BASEDIR, 'community', 'params')
IMPORTED_PATH = os.path.join(PARAMS_DIR, '.imported')
OLD_PARAMS_FILE = os.path.join(BASEDIR, 'op_params.json')


class Param:
  def __init__(self, default, allowed_types=[], description=None, *, static=False, live=False, hidden=False):  # pylint: disable=dangerous-default-value
    self.default_value = default  # value first saved and returned if actual value isn't a valid type
    if not isinstance(allowed_types, list):
      allowed_types = [allowed_types]
    self.allowed_types = allowed_types  # allowed python value types for opEdit
    self.description = description  # description to be shown in opEdit
    self.hidden = hidden  # hide this param to user in opEdit
    self.live = live  # show under the live menu in opEdit
    self.static = static  # use cached value, never reads to update
    self._create_attrs()

  def is_valid(self, value):
    if not self.has_allowed_types:  # always valid if no allowed types, otherwise checks to make sure
      return True
    return type(value) in self.allowed_types

  def _create_attrs(self):  # Create attributes and check Param is valid
    self.has_allowed_types = isinstance(self.allowed_types, list) and len(self.allowed_types) > 0
    self.has_description = self.description is not None
    self.is_list = list in self.allowed_types
    self.read_frequency = None if self.static else (1 if self.live else 10)  # how often to read param file (sec)
    self.last_read = -1
    if self.has_allowed_types:
      assert type(self.default_value) in self.allowed_types, 'Default value type must be in specified allowed_types!'
    if self.is_list:
      self.allowed_types.remove(list)


def _read_param(key):  # Returns None, False if a json error occurs
  try:
    with open(os.path.join(PARAMS_DIR, key), 'r') as f:
      value = json.loads(f.read())
    return value, True
  except json.decoder.JSONDecodeError:
    return None, False


def _write_param(key, value):
  param_path = os.path.join(PARAMS_DIR, key)
  with atomic_write(param_path, overwrite=True) as f:
    f.write(json.dumps(value))


def _import_params():
  if os.path.exists(OLD_PARAMS_FILE) and not os.path.exists(IMPORTED_PATH):  # if opParams needs to import from old params file
    try:
      with open(OLD_PARAMS_FILE, 'r') as f:
        old_params = json.loads(f.read())
      for key in old_params:
        _write_param(key, old_params[key])
      open(IMPORTED_PATH, 'w').close()
    except:  # pylint: disable=bare-except
      pass


class opParams:
  def __init__(self):
    """
      To add your own parameter to opParams in your fork, simply add a new entry in self.fork_params, instancing a new Param class with at minimum a default value.
      The allowed_types and description args are not required but highly recommended to help users edit their parameters with opEdit safely.
        - The description value will be shown to users when they use opEdit to change the value of the parameter.
        - The allowed_types arg is used to restrict what kinds of values can be entered with opEdit so that users can't crash openpilot with unintended behavior.
          (setting a param intended to be a number with a boolean, or viceversa for example)
          Limiting the range of floats or integers is still recommended when `.get`ting the parameter.
          When a None value is allowed, use `type(None)` instead of None, as opEdit checks the type against the values in the arg with `isinstance()`.
        - If you want your param to update within a second, specify live=True. If your param is designed to be read once, specify static=True.
          Specifying neither will have the param update every 10 seconds if constantly .get()
          If the param is not static, call the .get() function on it in the update function of the file you're reading from to use live updating

      Here's an example of a good fork_param entry:
      self.fork_params = {'camera_offset': Param(0.06, allowed_types=NUMBER), live=True}  # NUMBER allows both floats and ints
    """

    self.fork_params = {'camera_offset': Param(-0.04 if TICI else 0.06, NUMBER, 'Your camera offset to use in lane_planner.py\n'
                                                                                'If you have a comma three, note that the default camera offset is -0.04!', live=True),
                        'dynamic_follow': Param('stock', str, static=True, hidden=True),
                        'global_df_mod': Param(1.0, NUMBER, 'The multiplier for the current distance used by dynamic follow. The range is limited from 0.85 to 2.5\n'
                                                            'Smaller values will get you closer, larger will get you farther\n'
                                                            'This is multiplied by any profile that\'s active. Set to 1. to disable', live=True),
                        'min_TR': Param(0.9, NUMBER, 'The minimum allowed following distance in seconds. Default is 0.9 seconds\n'
                                                     'The range is limited from 0.85 to 2.7', live=True),
                        'alca_no_nudge_speed': Param(90., NUMBER, 'Above this speed (mph), lane changes initiate IMMEDIATELY. Behavior is stock under'),
                        'steer_ratio': Param(None, NONE_OR_NUMBER, '(Can be: None, or a float) If you enter None, openpilot will use the learned sR.\n'
                                                                   'If you use a float/int, openpilot will use that steer ratio instead', live=True),
                        # 'lane_speed_alerts': Param('silent', str, 'Can be: (\'off\', \'silent\', \'audible\')\n'
                        #                                           'Whether you want openpilot to alert you of faster-traveling adjacent lanes'),
                        'upload_on_hotspot': Param(False, bool, 'If False, openpilot will not upload driving data while connected to your phone\'s hotspot'),
                        'disengage_on_gas': Param(False, bool, 'Whether you want openpilot to disengage on gas input or not'),
                        'update_behavior': Param('alert', str, 'Can be: (\'off\', \'alert\', \'auto\') without quotes\n'
                                                               'off will never update, alert shows an alert on-screen\n'
                                                               'auto will reboot the device when an update is seen', static=True),
                        'dynamic_gas': Param(False, bool, 'Whether to use dynamic gas if your car is supported'),
                        'hide_auto_df_alerts': Param(False, bool, 'Hides the alert that shows what profile the model has chosen'),
                        'df_button_alerts': Param('audible', str, 'Can be: (\'off\', \'silent\', \'audible\')\n'
                                                                  'How you want to be alerted when you change your dynamic following profile'),
                        'log_auto_df': Param(False, bool, 'Logs dynamic follow data for auto-df', static=True),
                        # 'dynamic_camera_offset': Param(False, bool, 'Whether to automatically keep away from oncoming traffic.\n'
                        #                                             'Works from 35 to ~60 mph (requires radar)'),
                        # 'dynamic_camera_offset_time': Param(3.5, NUMBER, 'How long to keep away from oncoming traffic in seconds after losing lead'),
                        'support_white_panda': Param(False, bool, 'Enable this to allow engagement with the deprecated white panda.\n'
                                                                  'localizer might not work correctly', static=True),
                        'disable_charging': Param(30, NUMBER, 'How many hours until charging is disabled while idle', static=True),
                        'hide_model_long': Param(False, bool, 'Enable this to hide the Model Long button on the screen', static=True),
                        'prius_use_pid': Param(False, bool, 'This enables the PID lateral controller with new a experimental derivative tune\n'
                                                            'False: stock INDI, True: TSS2-tuned PID', static=True),
                        'use_lqr': Param(False, bool, 'Enable this to use LQR as your lateral controller over default with any car', static=True),
                        'use_steering_model': Param(False, bool, 'Enable this to use an experimental ML-based lateral controller trained on the TSSP Corolla\n'
                                                                 'This overrides all other tuning parameters\n'
                                                                 'Warning: the model may behave unexpectedly at any time, so always pay attention', static=True),
                        'corollaTSS2_use_indi': Param(False, bool, 'Enable this to use INDI for lat with your TSS2 Corolla', static=True),
                        'rav4TSS2_use_indi': Param(False, bool, 'Enable this to use INDI for lat with your TSS2 RAV4', static=True),
                        'standstill_hack': Param(False, bool, 'Some cars support stop and go, you just need to enable this', static=True),
                        'toyota_distance_btn': Param(False, bool, 'Set to True to use the steering wheel distance button to control the dynamic follow profile.\n'
                                                                  'Works on TSS2 vehicles and on TSS1 vehicles with an sDSU with a Sep. 2020 firmware or newer.', static=True)}

    self._to_delete = ['enable_long_derivative']  # a list of unused params you want to delete from users' params file
    self._to_reset = []  # a list of params you want reset to their default values
    self._run_init()  # restores, reads, and updates params

  def _run_init(self):  # does first time initializing of default params
    # Two required parameters for opEdit
    self.fork_params['username'] = Param(None, [type(None), str, bool], 'Your identifier provided with any crash logs sent to Sentry.\nHelps the developer reach out to you if anything goes wrong')
    self.fork_params['op_edit_live_mode'] = Param(False, bool, 'This parameter controls which mode opEdit starts in', hidden=True)

    self.params = self._load_params(can_import=True)
    self._add_default_params()  # adds missing params and resets values with invalid types to self.params
    self._delete_and_reset()  # removes old params

  def get(self, key=None, *, force_update=False):  # key=None returns dict of all params
    if key is None:
      return self._get_all_params(to_update=force_update)
    self._check_key_exists(key, 'get')
    param_info = self.fork_params[key]
    rate = param_info.read_frequency  # will be None if param is static, so check below

    if (not param_info.static and sec_since_boot() - self.fork_params[key].last_read >= rate) or force_update:
      value, success = _read_param(key)
      self.fork_params[key].last_read = sec_since_boot()
      if not success:  # in case of read error, use default and overwrite param
        value = param_info.default_value
        _write_param(key, value)
      self.params[key] = value

    if param_info.is_valid(value := self.params[key]):
      return value  # all good, returning user's value
    print(warning('User\'s value type is not valid! Returning default'))  # somehow... it should always be valid
    return param_info.default_value  # return default value because user's value of key is not in allowed_types to avoid crashing openpilot

  def put(self, key, value):
    self._check_key_exists(key, 'put')
    if not self.fork_params[key].is_valid(value):
      raise Exception('opParams: Tried to put a value of invalid type!')
    self.params.update({key: value})
    _write_param(key, value)

  def _load_params(self, can_import=False):
    if not os.path.exists(PARAMS_DIR):
      os.makedirs(PARAMS_DIR)
      if can_import:
        _import_params()  # just imports old params. below we read them in

    params = {}
    for key in os.listdir(PARAMS_DIR):  # PARAMS_DIR is guaranteed to exist
      if key.startswith('.') or key not in self.fork_params:
        continue
      value, success = _read_param(key)
      if not success:
        value = self.fork_params[key].default_value
        _write_param(key, value)
      params[key] = value
    return params

  def _get_all_params(self, to_update=False):
    if to_update:
      self.params = self._load_params()
    return {k: self.params[k] for k, p in self.fork_params.items() if k in self.params and not p.hidden}

  def _check_key_exists(self, key, met):
    if key not in self.fork_params:
      raise Exception('opParams: Tried to {} an unknown parameter! Key not in fork_params: {}'.format(met, key))

  def _add_default_params(self):
    for key, param in self.fork_params.items():
      if key not in self.params:
        self.params[key] = param.default_value
        _write_param(key, self.params[key])
      elif not param.is_valid(self.params[key]):
        print(warning('Value type of user\'s {} param not in allowed types, replacing with default!'.format(key)))
        self.params[key] = param.default_value
        _write_param(key, self.params[key])

  def _delete_and_reset(self):
    for key in list(self.params):
      if key in self._to_delete:
        del self.params[key]
        os.remove(os.path.join(PARAMS_DIR, key))
      elif key in self._to_reset and key in self.fork_params:
        self.params[key] = self.fork_params[key].default_value
        _write_param(key, self.params[key])
