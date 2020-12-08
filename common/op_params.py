#!/usr/bin/env python3
import os
import json
from common.colors import opParams_error as error
from common.colors import opParams_warning as warning
try:
  from common.realtime import sec_since_boot
except ImportError:
  import time
  sec_since_boot = time.time
  warning("Using python time.time() instead of faster sec_since_boot")

travis = False  # replace with travis_checker if you use travis or GitHub Actions


class ValueTypes:
  number = [float, int]
  none_or_number = [type(None), float, int]


class Param:
  def __init__(self, default=None, allowed_types=[], description=None, live=False, hidden=False):
    self.default = default
    if not isinstance(allowed_types, list):
      allowed_types = [allowed_types]
    self.allowed_types = allowed_types
    self.description = description
    self.hidden = hidden
    self.live = live
    self._create_attrs()

  def is_valid(self, value):
    if not self.has_allowed_types:
      return True
    return type(value) in self.allowed_types

  def _create_attrs(self):  # Create attributes and check Param is valid
    self.has_allowed_types = isinstance(self.allowed_types, list) and len(self.allowed_types) > 0
    self.has_description = self.description is not None
    self.is_list = list in self.allowed_types
    if self.has_allowed_types:
      assert type(self.default) in self.allowed_types, 'Default value type must be in specified allowed_types!'
    if self.is_list:
      self.allowed_types.remove(list)


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
        - Finally, the live arg tells both opParams and opEdit that it's a live parameter that will change. Therefore, you must place the `op_params.get()` call in the update function so that it can update.

      Here's an example of a good fork_param entry:
      self.fork_params = {'camera_offset': Param(default=0.06, allowed_types=VT.number), live=True}  # VT.number allows both floats and ints
    """

    VT = ValueTypes()
    self.fork_params = {'camera_offset': Param(0.06, VT.number, 'Your camera offset to use in lane_planner.py', live=True),
                        'dynamic_follow': Param('auto', str, 'Can be: (\'traffic\', \'relaxed\', \'roadtrip\'): Left to right increases in following distance.\n'
                                                             'All profiles support dynamic follow so you\'ll get your preferred distance while\n'
                                                             'retaining the smoothness and safety of dynamic follow!'),
                        'global_df_mod': Param(1.0, VT.number, 'The multiplier for the current distance used by dynamic follow. The range is limited from 0.85 to 1.5\n'
                                                               'Smaller values will get you closer, larger will get you farther\n'
                                                               'This is multiplied by any profile that\'s active. Set to 1. to disable', live=True),
                        'min_TR': Param(0.9, VT.number, 'The minimum allowed following distance in seconds. Default is 0.9 seconds.\n'
                                                        'The range is limited from 0.85 to 1.6.', live=True),
                        'alca_nudge_required': Param(True, bool, 'Whether to wait for applied torque to the wheel (nudge) before making lane changes. '
                                                                 'If False, lane change will occur IMMEDIATELY after signaling'),
                        'alca_min_speed': Param(25.0, VT.number, 'The minimum speed allowed for an automatic lane change (in MPH)'),
                        'steer_ratio': Param(None, VT.none_or_number, '(Can be: None, or a float) If you enter None, openpilot will use the learned sR.\n'
                                                                      'If you use a float/int, openpilot will use that steer ratio instead', live=True),
                        'lane_speed_alerts': Param('silent', str, 'Can be: (\'off\', \'silent\', \'audible\')\n'
                                                                  'Whether you want openpilot to alert you of faster-traveling adjacent lanes'),
                        'upload_on_hotspot': Param(False, bool, 'If False, openpilot will not upload driving data while connected to your phone\'s hotspot'),
                        'enable_long_derivative': Param(False, bool, 'If you have longitudinal overshooting, enable this! This enables derivative-based\n'
                                                                     'integral wind-down to help reduce overshooting within the long PID loop'),
                        'disengage_on_gas': Param(True, bool, 'Whether you want openpilot to disengage on gas input or not'),
                        'update_behavior': Param('auto', str, 'Can be: (\'off\', \'alert\', \'auto\') without quotes\n'
                                                              'off will never update, alert shows an alert on-screen\n'
                                                              'auto will reboot the device when an update is seen'),
                        'dynamic_gas': Param(True, bool, 'Whether to use dynamic gas if your car is supported'),
                        'hide_auto_df_alerts': Param(False, bool, 'Hides the alert that shows what profile the model has chosen'),
                        'log_auto_df': Param(False, bool, 'Logs dynamic follow data for auto-df'),
                        'dynamic_camera_offset': Param(True, bool, 'Whether to automatically keep away from oncoming traffic.\n'
                                                                   'Works from 35 to ~60 mph (requires radar)'),
                        'dynamic_camera_offset_time': Param(3.5, VT.number, 'How long to keep away from oncoming traffic in seconds after losing lead'),
                        'support_white_panda': Param(False, bool, 'Enable this to allow engagement with the deprecated white panda.\n'
                                                                  'localizer might not work correctly'),
                        'slowdown_for_curves': Param(True, bool, 'Whether your car will slow down for curves using the old planner code from 0.5/0.6'),

                        'prius_use_pid': Param(False, bool, 'This enables the PID lateral controller with new a experimental derivative tune\nFalse: stock INDI, True: TSS2-tuned PID'),
                        'use_lqr': Param(False, bool, 'Enable this to use LQR as your lateral controller over default with any car'),
                        'corollaTSS2_use_indi': Param(False, bool, 'Enable this to use INDI for lat with your TSS2 Corolla'),
                        'rav4TSS2_use_indi': Param(False, bool, 'Enable this to use INDI for lat with your TSS2 RAV4'),
                        'standstill_hack': Param(False, bool, 'Some cars support stop and go, you just need to enable this')}

    self._params_file = '/data/op_params.json'
    self._backup_file = '/data/op_params_corrupt.json'
    self._last_read_time = sec_since_boot()
    self.read_frequency = 2.5  # max frequency to read with self.get(...) (sec)
    self._to_delete = ['no_ota_updates', 'auto_update']  # a list of unused params you want to delete
    self._run_init()  # restores, reads, and updates params

  def _run_init(self):  # does first time initializing of default params
    # Two required parameters for opEdit
    self.fork_params['username'] = Param(None, [type(None), str, bool], 'Your identifier provided with any crash logs sent to Sentry.\nHelps the developer reach out to you if anything goes wrong')
    self.fork_params['op_edit_live_mode'] = Param(False, bool, 'This parameter controls which mode opEdit starts in', hidden=True)
    self.params = self._get_all_params(default=True)  # in case file is corrupted

    if travis:
      return

    if os.path.isfile(self._params_file):
      if self._read():
        to_write = self._add_default_params()  # if new default data has been added
        to_write |= self._delete_old()  # or if old params have been deleted
      else:  # backup and re-create params file
        error("Can't read op_params.json file, backing up to /data/op_params_corrupt.json and re-creating file!")
        to_write = True
        if os.path.isfile(self._backup_file):
          os.remove(self._backup_file)
        os.rename(self._params_file, self._backup_file)
    else:
      to_write = True  # user's first time running a fork with op_params, write default params

    if to_write:
      self._write()
      os.chmod(self._params_file, 0o764)

  def get(self, key=None, force_live=False):  # any params you try to get MUST be in fork_params
    param_info = self.param_info(key)
    self._update_params(param_info, force_live)

    if key is None:
      return self._get_all_params()

    self._check_key_exists(key, 'get')
    value = self.params[key]
    if param_info.is_valid(value):  # always valid if no allowed types, otherwise checks to make sure
      return value  # all good, returning user's value

    warning('User\'s value type is not valid! Returning default')  # somehow... it should always be valid
    return param_info.default  # return default value because user's value of key is not in allowed_types to avoid crashing openpilot

  def put(self, key, value):
    self._check_key_exists(key, 'put')
    if not self.param_info(key).is_valid(value):
      raise Exception('opParams: Tried to put a value of invalid type!')
    self.params.update({key: value})
    self._write()

  def __getitem__(self, s):  # can also do op_params['param_name']
    return self.get(s)

  def delete(self, key):  # todo: might be obsolete. remove?
    if key in self.params:
      del self.params[key]
      self._write()

  def param_info(self, key):
    if key in self.fork_params:
      return self.fork_params[key]
    return Param()

  def _check_key_exists(self, key, met):
    if key not in self.fork_params or key not in self.params:
      raise Exception('opParams: Tried to {} an unknown parameter! Key not in fork_params: {}'.format(met, key))

  def _add_default_params(self):
    added = False
    for key, param in self.fork_params.items():
      if key not in self.params:
        self.params[key] = param.default
        added = True
      elif not param.is_valid(self.params[key]):
        warning('Value type of user\'s {} param not in allowed types, replacing with default!'.format(key))
        self.params[key] = param.default
        added = True
    return added

  def _delete_old(self):
    deleted = False
    for param in self._to_delete:
      if param in self.params:
        del self.params[param]
        deleted = True
    return deleted

  def _get_all_params(self, default=False, return_hidden=False):
    if default:
      return {k: p.default for k, p in self.fork_params.items()}
    return {k: self.params[k] for k, p in self.fork_params.items() if k in self.params and (not p.hidden or return_hidden)}

  def _update_params(self, param_info, force_live):
    if force_live or param_info.live:  # if is a live param, we want to get updates while openpilot is running
      if not travis and sec_since_boot() - self._last_read_time >= self.read_frequency:  # make sure we aren't reading file too often
        if self._read():
          self._last_read_time = sec_since_boot()

  def _read(self):
    try:
      with open(self._params_file, "r") as f:
        self.params = json.loads(f.read())
      return True
    except Exception as e:
      error(e)
      return False

  def _write(self):
    if not travis:
      with open(self._params_file, "w") as f:
        f.write(json.dumps(self.params, indent=2))  # can further speed it up by remove indentation but makes file hard to read
