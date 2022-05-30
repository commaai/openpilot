#!/usr/bin/env python3
import os
import json
import math
import time
from common.numpy_fast import find_nearest_index, interp, is_multi_iter
from common.colors import opParams_error as error
from common.colors import opParams_warning as warning
from selfdrive.hardware import PC
from selfdrive.car.chrysler.values import STEER_MAX_LOOKUP, STEER_DELTA_UP, STEER_DELTA_DOWN
try:
  from common.realtime import sec_since_boot
except ImportError:
  sec_since_boot = time.time
  warning("Using python time.time() instead of faster sec_since_boot")

travis = True if PC else False  # replace with travis_checker if you use travis or GitHub Actions


def parse_param_modifiers(src, value):
  if src:
    if ParamModifierKeys.ABS in src:
      return parse_param_modifiers(src.replace(ParamModifierKeys.ABS, ''), abs(value))
    elif ParamModifierKeys.DEGREES in src:
      return parse_param_modifiers(src.replace(ParamModifierKeys.DEGREES, ''), math.degrees(value))
    elif ParamModifierKeys.RADIANS in src:
      return parse_param_modifiers(src.replace(ParamModifierKeys.RADIANS, ''), math.radians(value))
  else:
    return value

def eval_breakpoint_source(sources, CS, controls_state):
  '''
  Maps a breakpoint source array to actual values
  '''

  def eval_source(src):
    if BreakPointSourceKeys.VEGO in src:
      return parse_param_modifiers(src.replace(BreakPointSourceKeys.VEGO, ''), CS.vEgo)
    elif BreakPointSourceKeys.AEGO in src:
      return parse_param_modifiers(src.replace(BreakPointSourceKeys.AEGO, ''), CS.aEgo)
    elif BreakPointSourceKeys.DESIRED_STEER in src:
      return parse_param_modifiers(src.replace(BreakPointSourceKeys.DESIRED_STEER, ''), controls_state.desiredSteerDeg)
    else:
      raise ValueError(f'Unknown value option: {src}')

  return [eval_source(source) for source in sources]

def interp_multi_bp(x, bp, v):
  def correct_multi_bp(idx):
    if not is_multi_iter(bp[idx]):
      bp[idx] = [bp[idx], bp[idx]]

      if len(bp) <= 1:
        bp.insert(0, bp[idx][0])

  is_bp_multi_iter = is_multi_iter(bp)
  is_v_multi_iter = is_multi_iter(v)

  if not is_bp_multi_iter:
    bp = [bp, bp]

  # correct_multi_bp(0)
  correct_multi_bp(-1)

  if not is_v_multi_iter:
    v = [v, v]

  l_x = len(x)
  l_bp = len(bp)
  l_v = len(v)

  # print(f'bp: {bp}')
  if l_v <= 1:
    v = [v[-1], v[-1]]

  if l_bp < l_x or not hasattr(bp[0], '__iter__') or len(bp[0]) <= 1:
    # return interp(x[0], bp[0][0], v[0])
    # idx = range(len(x)) if is_multi_iter(x) else 0
    # idx = [0] if is_multi_iter(x) else 0
    idx = 0
  else:
    idx = find_nearest_index(bp[0], x[0])

  # print(f'indexes: {idx}')

  if hasattr(idx, '__iter__'):
    return [interp(x[-1], bp[-1][-1], v[min(l_v - 1, i)]) for i in set(idx)]
  else:
    return interp(x[-1], bp[-1][-1], v[min(l_v - 1, idx)])

  # return [interp(x[-1], bp[-1][i], v[i]) for i in set(idx)] if hasattr(idx, '__iter__') else interp(x[-1], bp[-1][idx], v[idx])
  # return interp(x[1], bp[1][idx], v[idx])

class BreakPointSourceKeys:
  VEGO = 'vego'
  AEGO = 'aego'
  DESIRED_STEER = 'desired_steer'

class ParamModifierKeys:
  ABS = '_abs'
  DEGREES = '_deg'
  RADIANS = '_rad'

class ValueTypes:
  number = [float, int]
  none_or_number = [type(None), float, int]
  list_of_numbers = [list, float, int]

class Param:
  def __init__(self, default, allowed_types, description=None, live=False, hidden=False, depends_on=None):
    self.default = default
    if not isinstance(allowed_types, list):
      allowed_types = [allowed_types]
    self.allowed_types = allowed_types
    self.description = description
    self.hidden = hidden
    self.live = live
    self.depends_on = depends_on
    self.children = []
    self._create_attrs()

  def is_valid(self, value):
    if not self.has_allowed_types:
      return True
    if self.is_list and isinstance(value, list):
      for v in value:
        if type(v) not in self.allowed_types:
          return False
      return True
    else:
      return type(value) in self.allowed_types or value in self.allowed_types

  def _create_attrs(self):  # Create attributes and check Param is valid
    self.has_allowed_types = isinstance(self.allowed_types, list) and len(self.allowed_types) > 0
    self.has_description = self.description is not None
    self.is_list = list in self.allowed_types
    self.is_bool = bool in self.allowed_types
    if self.has_allowed_types:
      assert type(self.default) in self.allowed_types or self.default in self.allowed_types, 'Default value type must be in specified allowed_types!'

      if self.is_list and self.default:
        for v in self.default:
          assert type(v) in self.allowed_types, 'Default value type must be in specified allowed_types!'


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
      self.fork_params = {'camera_offset': Param(default=0.06, allowed_types=VT.number)}  # VT.number allows both floats and ints
    """
  
    VT = ValueTypes()
    self.fork_params = {
                        # LAT_KP_BP: Param([0., 25.,], [list, float, int], live=True),
                        # LAT_KP_V: Param([0.12, 0.12], [list, float, int], live=True),
                        # LAT_KI_BP: Param([0.,25.], [list, float, int], live=True),
                        # LAT_KI_V: Param([0., 0.0001], [list, float, int], live=True),
                        # LAT_KD_BP: Param([0.,25.], [list, float, int], live=True),
                        # LAT_KD_V: Param([0., 0.001], [list, float, int], live=True),
                        # LAT_KF: Param(6e-6, VT.number, live=True),
                        # MAX_LAT_ACCEL: Param(1.2, VT.number, live=True),
                        FRICTION: Param(0.05, VT.number, live=True),
                        # STEER_ACT_DELAY: Param(0.1, VT.number, live=True),
                        # STEER_RATE_COST: Param(0.5, VT.number, live=True),
                        # DEVICE_OFFSET: Param(0.0, VT.number, live=True),
                        
                        #SHOW_RATE_PARAMS: Param(False, [bool], live=True),
                        #ENABLE_RATE_PARAMS: Param(False, [bool], live=True, depends_on=SHOW_RATE_PARAMS),
                        #STOCK_DELTA_UP_DOWN: Param(6, VT.number, live=True ,depends_on=SHOW_RATE_PARAMS),
                        #STOCK_DELTA_UP: Param(25, VT.number, live=True ,depends_on=SHOW_RATE_PARAMS),
                        #STOCK_DELTA_DOWN: Param(50, VT.number, live=True ,depends_on=SHOW_RATE_PARAMS),
                        #STOCK_STEER_MAX: Param(363, VT.number, live=True ,depends_on=SHOW_RATE_PARAMS),
}

    self._params_file = '/data/op_params.json'
    self._backup_file = '/data/op_params_corrupt.json'
    self._last_read_time = sec_since_boot()
    self.read_frequency = 2.5  # max frequency to read with self.get(...) (sec)
    self._to_delete = ['lane_hug_direction', 'lane_hug_angle_offset', 'prius_use_lqr']  # a list of params you want to delete (unused)
    self._last_mod_time = 0.0
    self._run_init()  # restores, reads, and updates params

  def _run_init(self):  # does first time initializing of default params
    # Two required parameters for opEdit
    self.fork_params['username'] = Param(None, [type(None), str, bool], 'Your identifier provided with any crash logs sent to Sentry.\nHelps the developer reach out to you if anything goes wrong')
    self.fork_params['op_edit_live_mode'] = Param(False, bool, 'This parameter controls which mode opEdit starts in', hidden=False)
    self.params = self._get_all_params(default=True)  # in case file is corrupted

    for k, p in self.fork_params.items():
      d = p.depends_on
      while d:
        fp = self.fork_params[d]
        fp.children.append(k)
        d = fp.depends_on

    if travis:
      return

    to_write = False
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
      if self._write():
        os.chmod(self._params_file, 0o764)

  def get(self, key=None, force_live=False):  # any params you try to get MUST be in fork_params
    if PC:
      assert isinstance(self, opParams), f'Self is type: {type(self).__name__}, expected opParams'

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

  def delete(self, key):  # todo: might be obsolete. remove?
    if key in self.params:
      del self.params[key]
      self._write()

  def param_info(self, key):
    if key in self.fork_params:
      return self.fork_params[key]
    return Param(None, type(None))

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
    if os.path.isfile(self._params_file):
      try:
        mod_time = os.path.getmtime(self._params_file)
        if mod_time > self._last_mod_time:
          with open(self._params_file, "r") as f:
            self.params = json.loads(f.read())
          self._last_mod_time = mod_time
          return True
        else:
          return False
      except Exception as e:
        print("Unable to read file: " + str(e))
        return False
    else:
      return False

  def _write(self):
    if not travis or os.path.isdir("/data/"):
      try:
        with open(self._params_file, "w") as f:
          f.write(json.dumps(self.params, indent=2))  # can further speed it up by remove indentation but makes file hard to read
        return True
      except Exception as e:
        print("Unable to write file: " + str(e))
        return False

SHOW_INDI_PARAMS = 'show_indi_params'
ENABLE_INDI_BREAKPOINTS = 'enable_indi_breakpoints'

# LAT_KP_BP = 'lat_kp_bp'
# LAT_KP_V = 'lat_kp_v'
# LAT_KI_BP = 'lat_ki_bp'
# LAT_KI_V = 'lat_ki_v'
# LAT_KD_BP = 'lat_kd_bp'
# LAT_KD_V = 'lat_kd_v'
# LAT_KF = 'lat_kf'
# MAX_LAT_ACCEL = 'max_lat_accel'
FRICTION = 'friction'

#SHOW_RATE_PARAMS = 'show_rate_params'
#ENABLE_RATE_PARAMS = 'enable_rate_params'
#STOCK_DELTA_UP_DOWN = 'stock_delta_up_down'
# STOCK_DELTA_UP = 'stock_delta_up'
# STOCK_DELTA_DOWN = 'stock_delta_down'
# STOCK_STEER_MAX = 'stock_steer_max'
# STEER_ACT_DELAY = 'steer_act_delay'
# STEER_RATIO = 'steer ratio'
# STEER_RATE_COST = 'steer_rate_cost'
# DEVICE_OFFSET = 'device_offset'
