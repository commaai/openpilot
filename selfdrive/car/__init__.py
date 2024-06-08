# functions common among cars
from collections import namedtuple
from dataclasses import dataclass
from enum import IntFlag, ReprEnum, EnumType
from dataclasses import replace, is_dataclass, fields
from tokenize import tokenize
from copy import deepcopy
from typing import List, Any, Tuple

import capnp, inspect, re, io

from cereal import car
from openpilot.common.numpy_fast import clip, interp
from openpilot.common.utils import Freezable
from openpilot.selfdrive.car.docs_definitions import CarDocs


# kg of standard extra cargo to count for drive, gas, etc...
STD_CARGO_KG = 136.

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName
AngleRateLimit = namedtuple('AngleRateLimit', ['speed_bp', 'angle_v'])


def apply_hysteresis(val: float, val_steady: float, hyst_gap: float) -> float:
  if val > val_steady + hyst_gap:
    val_steady = val - hyst_gap
  elif val < val_steady - hyst_gap:
    val_steady = val + hyst_gap
  return val_steady


def create_button_events(cur_btn: int, prev_btn: int, buttons_dict: dict[int, capnp.lib.capnp._EnumModule],
                         unpressed_btn: int = 0) -> list[capnp.lib.capnp._DynamicStructBuilder]:
  events: list[capnp.lib.capnp._DynamicStructBuilder] = []

  if cur_btn == prev_btn:
    return events

  # Add events for button presses, multiple when a button switches without going to unpressed
  for pressed, btn in ((False, prev_btn), (True, cur_btn)):
    if btn != unpressed_btn:
      events.append(car.CarState.ButtonEvent(pressed=pressed,
                                             type=buttons_dict.get(btn, ButtonType.unknown)))
  return events


def gen_empty_fingerprint():
  return {i: {} for i in range(8)}


# these params were derived for the Civic and used to calculate params for other cars
class VehicleDynamicsParams:
  MASS = 1326. + STD_CARGO_KG
  WHEELBASE = 2.70
  CENTER_TO_FRONT = WHEELBASE * 0.4
  CENTER_TO_REAR = WHEELBASE - CENTER_TO_FRONT
  ROTATIONAL_INERTIA = 2500
  TIRE_STIFFNESS_FRONT = 192150
  TIRE_STIFFNESS_REAR = 202500


# TODO: get actual value, for now starting with reasonable value for
# civic and scaling by mass and wheelbase
def scale_rot_inertia(mass, wheelbase):
  return VehicleDynamicsParams.ROTATIONAL_INERTIA * mass * wheelbase ** 2 / (VehicleDynamicsParams.MASS * VehicleDynamicsParams.WHEELBASE ** 2)


# TODO: start from empirically derived lateral slip stiffness for the civic and scale by
# mass and CG position, so all cars will have approximately similar dyn behaviors
def scale_tire_stiffness(mass, wheelbase, center_to_front, tire_stiffness_factor):
  center_to_rear = wheelbase - center_to_front
  tire_stiffness_front = (VehicleDynamicsParams.TIRE_STIFFNESS_FRONT * tire_stiffness_factor) * mass / VehicleDynamicsParams.MASS * \
                         (center_to_rear / wheelbase) / (VehicleDynamicsParams.CENTER_TO_REAR / VehicleDynamicsParams.WHEELBASE)

  tire_stiffness_rear = (VehicleDynamicsParams.TIRE_STIFFNESS_REAR * tire_stiffness_factor) * mass / VehicleDynamicsParams.MASS * \
                        (center_to_front / wheelbase) / (VehicleDynamicsParams.CENTER_TO_FRONT / VehicleDynamicsParams.WHEELBASE)

  return tire_stiffness_front, tire_stiffness_rear


DbcDict = dict[str, str]


def dbc_dict(pt_dbc, radar_dbc, chassis_dbc=None, body_dbc=None) -> DbcDict:
  return {'pt': pt_dbc, 'radar': radar_dbc, 'chassis': chassis_dbc, 'body': body_dbc}


def apply_driver_steer_torque_limits(apply_torque, apply_torque_last, driver_torque, LIMITS):

  # limits due to driver torque
  driver_max_torque = LIMITS.STEER_MAX + (LIMITS.STEER_DRIVER_ALLOWANCE + driver_torque * LIMITS.STEER_DRIVER_FACTOR) * LIMITS.STEER_DRIVER_MULTIPLIER
  driver_min_torque = -LIMITS.STEER_MAX + (-LIMITS.STEER_DRIVER_ALLOWANCE + driver_torque * LIMITS.STEER_DRIVER_FACTOR) * LIMITS.STEER_DRIVER_MULTIPLIER
  max_steer_allowed = max(min(LIMITS.STEER_MAX, driver_max_torque), 0)
  min_steer_allowed = min(max(-LIMITS.STEER_MAX, driver_min_torque), 0)
  apply_torque = clip(apply_torque, min_steer_allowed, max_steer_allowed)

  # slow rate if steer torque increases in magnitude
  if apply_torque_last > 0:
    apply_torque = clip(apply_torque, max(apply_torque_last - LIMITS.STEER_DELTA_DOWN, -LIMITS.STEER_DELTA_UP),
                        apply_torque_last + LIMITS.STEER_DELTA_UP)
  else:
    apply_torque = clip(apply_torque, apply_torque_last - LIMITS.STEER_DELTA_UP,
                        min(apply_torque_last + LIMITS.STEER_DELTA_DOWN, LIMITS.STEER_DELTA_UP))

  return int(round(float(apply_torque)))


def apply_dist_to_meas_limits(val, val_last, val_meas,
                              STEER_DELTA_UP, STEER_DELTA_DOWN,
                              STEER_ERROR_MAX, STEER_MAX):
  # limits due to comparison of commanded val VS measured val (torque/angle/curvature)
  max_lim = min(max(val_meas + STEER_ERROR_MAX, STEER_ERROR_MAX), STEER_MAX)
  min_lim = max(min(val_meas - STEER_ERROR_MAX, -STEER_ERROR_MAX), -STEER_MAX)

  val = clip(val, min_lim, max_lim)

  # slow rate if val increases in magnitude
  if val_last > 0:
    val = clip(val,
               max(val_last - STEER_DELTA_DOWN, -STEER_DELTA_UP),
               val_last + STEER_DELTA_UP)
  else:
    val = clip(val,
               val_last - STEER_DELTA_UP,
               min(val_last + STEER_DELTA_DOWN, STEER_DELTA_UP))

  return float(val)


def apply_meas_steer_torque_limits(apply_torque, apply_torque_last, motor_torque, LIMITS):
  return int(round(apply_dist_to_meas_limits(apply_torque, apply_torque_last, motor_torque,
                                             LIMITS.STEER_DELTA_UP, LIMITS.STEER_DELTA_DOWN,
                                             LIMITS.STEER_ERROR_MAX, LIMITS.STEER_MAX)))


def apply_std_steer_angle_limits(apply_angle, apply_angle_last, v_ego, LIMITS):
  # pick angle rate limits based on wind up/down
  steer_up = apply_angle_last * apply_angle >= 0. and abs(apply_angle) > abs(apply_angle_last)
  rate_limits = LIMITS.ANGLE_RATE_LIMIT_UP if steer_up else LIMITS.ANGLE_RATE_LIMIT_DOWN

  angle_rate_lim = interp(v_ego, rate_limits.speed_bp, rate_limits.angle_v)
  return clip(apply_angle, apply_angle_last - angle_rate_lim, apply_angle_last + angle_rate_lim)


def common_fault_avoidance(fault_condition: bool, request: bool, above_limit_frames: int,
                           max_above_limit_frames: int, max_mismatching_frames: int = 1):
  """
  Several cars have the ability to work around their EPS limits by cutting the
  request bit of their LKAS message after a certain number of frames above the limit.
  """

  # Count up to max_above_limit_frames, at which point we need to cut the request for above_limit_frames to avoid a fault
  if request and fault_condition:
    above_limit_frames += 1
  else:
    above_limit_frames = 0

  # Once we cut the request bit, count additionally to max_mismatching_frames before setting the request bit high again.
  # Some brands do not respect our workaround without multiple messages on the bus, for example
  if above_limit_frames > max_above_limit_frames:
    request = False

  if above_limit_frames >= max_above_limit_frames + max_mismatching_frames:
    above_limit_frames = 0

  return above_limit_frames, request


def make_can_msg(addr, dat, bus):
  return [addr, 0, dat, bus]


def get_safety_config(safety_model, safety_param = None):
  ret = car.CarParams.SafetyConfig.new_message()
  ret.safetyModel = safety_model
  if safety_param is not None:
    ret.safetyParam = safety_param
  return ret


class CanBusBase:
  offset: int

  def __init__(self, CP, fingerprint: dict[int, dict[int, int]] | None) -> None:
    if CP is None:
      assert fingerprint is not None
      num = max([k for k, v in fingerprint.items() if len(v)], default=0) // 4 + 1
    else:
      num = len(CP.safetyConfigs)
    self.offset = 4 * (num - 1)


class CanSignalRateCalculator:
  """
  Calculates the instantaneous rate of a CAN signal by using the counter
  variable and the known frequency of the CAN message that contains it.
  """
  def __init__(self, frequency):
    self.frequency = frequency
    self.previous_counter = 0
    self.previous_value = 0
    self.rate = 0

  def update(self, current_value, current_counter):
    if current_counter != self.previous_counter:
      self.rate = (current_value - self.previous_value) * self.frequency

    self.previous_counter = current_counter
    self.previous_value = current_value

    return self.rate


@dataclass(frozen=True, kw_only=True)
class CarSpecs:
  mass: float  # kg, curb weight
  wheelbase: float  # meters
  steerRatio: float
  centerToFrontRatio: float = 0.5
  minSteerSpeed: float = 0.0  # m/s
  minEnableSpeed: float = -1.0  # m/s
  tireStiffnessFactor: float = 1.0

  def override(self, **kwargs):
    return replace(self, **kwargs)


@dataclass(order=True, kw_only=True)
class PlatformConfig(Freezable):
  car_docs: list[CarDocs]
  specs: CarSpecs

  dbc_dict: DbcDict

  flags: int = 0

  platform_str: str | None = None

  def __hash__(self) -> int:
    return hash(self.platform_str)

  def override(self, **kwargs):
    return replace(self, **kwargs)

  def copy(self):
    return PlatformConfigModifier(self)

  def init(self):
    pass

  def __post_init__(self):
    self.init()


class PlatformsType(EnumType):
  def __new__(metacls, cls, bases, classdict, *, boundary=None, _simple=False, **kwds):
    for key in classdict._member_names.keys():
      cfg: PlatformConfig = classdict[key]
      cfg.platform_str = key
      cfg.freeze()
    return super().__new__(metacls, cls, bases, classdict, boundary=boundary, _simple=_simple, **kwds)


class Platforms(str, ReprEnum, metaclass=PlatformsType):
  config: PlatformConfig

  def __new__(cls, platform_config: PlatformConfig):
    member = str.__new__(cls, platform_config.platform_str)
    member.config = platform_config
    member._value_ = platform_config.platform_str
    return member

  def __repr__(self):
    return f"<{self.__class__.__name__}.{self.name}>"

  @classmethod
  def create_dbc_map(cls) -> dict[str, DbcDict]:
    return {p: p.config.dbc_dict for p in cls}

  @classmethod
  def with_flags(cls, flags: IntFlag) -> set['Platforms']:
    return {p for p in cls if p.config.flags & flags}

class PlatformConfigModifier:
    """
    Keeps track of changes made to any attributes of a PlatformConfig in a dict & saves them to the source file.
    The attributes have to be objects of dataclasses, otherwise the changes may be missed.
    The primary goal is not performance, but rather to keep the diff as low as possible when editing platform configs.
    """
    def __init__(self, config):
        self._original_config = deepcopy(config)
        self._config = deepcopy(config)
        self._changed_fields = {}

    def __getattr__(self, name):
        return getattr(self._config, name)

    def __setattr__(self, name, value):
        # if it's change to an already existing attribute
        if name in {'_original_config', '_config', '_changed_fields'}:
            super().__setattr__(name, value)
        else:
            # if it's a new attrbiute
            setattr(self._config, name, value)
            self._changed_fields[name] = value

    # this will only work if all objects are of dataclasses, otherwise changes to that attribute will be missed
    def _compare_dataclasses(self, original, current, path=""):
        changes = []

        # iterate through the fields in the original object
        for field in fields(original):
            # get the values of the attribute, current & previous
            original_value = getattr(original, field.name)
            current_value = getattr(current, field.name)
            # we need the formatted name for referring to later
            current_path = f"{path}.{field.name}" if path else field.name

            # if this is a nested object
            if is_dataclass(original_value):
                # recursively check for any changes within it
                changes.extend(self._compare_dataclasses(original_value, current_value, current_path))
            elif original_value != current_value: # or if there are any changes, just note them down
                changes.append({'name': current_path, 'value': current_value})
        return changes

    def _get_fields(self, attribute):
        result = dict()
        for field in fields(attribute):
            if is_dataclass(field.type):
                result[field.name] = self._get_fields(field.type)
            else:
                result[field.name] = ''
        return result

    # generate the source code of an attribute
    def _get_source(self, fields):
        config = self._config
        for each in fields:
            config = getattr(config, each)
        return str(config)

    # parse the existing source code into a dict
    # each attribute will be marked with it's starting and ending position in the code string
    # the writer can use this to find the diff points
    def _parse_source(self, code: List[Any], attributes: Any, start: int, end: int):

        '''
        if you write an attribute like so

            HONDA_ACCORD = PlatformConfig(
                CarSpecs(
                mass=3279 * 1
                , # comma here
                wheelbase
                = # or assignment here
                2.83,
                # ...
                )
            )

        with the commas or assignments on a new line, this will fail.
        i know comma isn't stupid to write commas like that, just documenting
        '''

        parsed = dict()

        # we don't count the starting paranthesis in the logic, it's just harcoded here
        brackets = {')': 1, ']': 0, '}': 0}
        def bracket_tracker(token):
            bracket_map = {'(': ')', '[': ']', '{': '}'}
            if token in bracket_map.keys():
                brackets[bracket_map[token]] += 1
            elif token in bracket_map.values():
                brackets[token] -= 1

        while start < end:

            name = ''

            # find the attribute in the source code
            for index in range(start+1, end):
                if code[index].type == 62 and code[index].string == '\n':
                    continue # if it's a newline, then skip

                bracket_tracker(code[index].string)
                # since kw_only=True, the code will be written as attribute_name=<value>, which we can exploit to find it
                if code[index].type == 1 and code[index+1].type == 54 and code[index+1].string == '=':
                    name = code[index].string
                    if name in attributes.keys():
                        # if it's sure that we're dealing with a nested object
                        # are you thinking why I'm checking using isinstance also?
                        # well, imagine the source code is something=dict() or something=list(), then the pattern will flag it as a nested object. we don't want that
                        if code[index+2].type == 1 and code[index+3].type == 54 and code[index+3].string == '(' and isinstance(attributes[name], dict):
                            attribute_start = start = index + 3 # mark the start after the 'ClassName(' paranthesis
                            parsed[name] = {'start': code[start-1].start}
                        else: # otherwise mark that name token as the start
                            attribute_start = start = index + 2
                            parsed[name] = {'start': code[start].start}
                        break

            # find the end value
            for index in range(start, end):
                bracket_tracker(code[index].string)

                # if this attribute has ended, mark the end position
                if code[index].type == 54 and code[index].string == ',' and [value for value in brackets.values()] == [1, 0, 0]:
                    parsed[name]['end'] = code[index-1].end
                    attribute_end = index
                    start = index
                    break

                # if all the brackets are clear, then this is the last attribute.
                if code[index].type == 54 and code[index].string == ')' and all(value == 0 for value in brackets.values()):

                    # backtrack till we find anything other than a newline
                    index -= 1
                    while code[index].type == 62 and code[index].string == '\n':
                        index-=1

                    # and that should be the end of this attribute, because come on what else would it be?
                    parsed[name]['end'] = code[index].end
                    attribute_end = index + 1
                    end = start # and mark this attribute as finished
                    break

            try:
                if isinstance(attributes[name], dict): parsed[name]['attributes'] = self._parse_source(code, attributes[name], attribute_start, attribute_end)
                del attributes[name]
            except KeyError: break

        # any attribute not yet encountered is not in the source code
        for attribute in attributes.keys():
            parsed[attribute] = {'start': (0, 0), 'end': (0, 0)}

        return parsed

    @property
    def original_fields(self):
        return self._get_fields(self._original_config)

    # returns all the fields that have changed as a list containing dict({name: value})
    @property
    def changed_fields(self):
        changes = self._compare_dataclasses(self._original_config, self._config)
        for key, value in self._changed_fields.items():
            full_path = key
            if not any(change['name'] == full_path for change in changes):
                changes.append({'name': full_path, 'value': value})
        return changes

    # takes some code and inserts it at the given position
    # it moves anything currently in that position to be after the inserted string
    def _insert_code(self, source: str, name: str, value: Any, position: Tuple[int, int]):
        lines = source.split('\n')
        line = lines[position[0]-1]

        before = line[:position[1]]
        after = line[position[1]:]

        lines[position[0]-1] = f'{before} {name}={str(value)}{after}' if before.endswith(',') else f'{before}, {name}={str(value)}{after}'
        return '\n'.join(lines)

    # takes a line number and writes the supplied code to the end of the line
    def _append_code(self, source: str, name: str, value: any, line: int):
        lines = source.split('\n')

        lines[line-1] += f' {name}={str(value)}' if lines[line-1].endswith(',') else f', {name}={str(value)}'
        return '\n'.join(lines)

    # takes a string to replace & a position as argument, then replaces whatever code is in that position with the supplied code
    def _replace_code(self, source: str, replacement: str, start: Tuple[int, int], end: Tuple[int, int]):
        # split the source text into lines
        lines = source.split('\n')

        # extract line and position from start and end tuples
        start_line, start_pos = start
        end_line, end_pos = end

        # account for the discrepancy with line numbers in tokenize
        start_line -= 1
        end_line -= 1

        # extract the parts of the lines to keep
        before = lines[start_line][:start_pos]
        after = lines[end_line][end_pos:]

        # construct the changed code
        if start_line == end_line:
            new_line = before + replacement + after
            lines[start_line] = new_line
        else:
            # replace the lines from start_line to end_line
            lines[start_line] = before + replacement + after
            # remove the lines in between
            del lines[start_line + 1:end_line + 1]

        return '\n'.join(lines)

    def _diff_writer(self, source: str, parsed: dict, changes: List[Any]):

        # for a given change, this returns which attribute's source code has to be changed
        def get_attribute(change):
            attributes = change.split(".")
            value, prev, diff, codeExists, code = parsed, (None, 0), None, False, None

            for index, attribute in enumerate(attributes):
                try:
                    value = value[attribute]
                except KeyError:
                    value = value['attributes'][attribute]

                # while traversing, keep track of the last attribute written in the source code
                if value['start'] != (0, 0) and value['end'] != (0, 0): prev = (value, index)

                if value['start'] == (0, 0) and value['end'] == (0, 0) and prev[0] is not None:
                    value, diff = prev[0], self._get_source(attributes)
                    # if the parent just above this is already written in the source, we can simply insert this one to that with minimum diff
                    codeExists, code = prev[1] is index-1, self._get_source(attributes[:index]) # we'll also keep the full atrribute source code, for an edge case
                    break

            return (value, diff, codeExists, code)

        root_end = max(parsed.items(), key=lambda item: item[1]['end'])[1]['end']
        replacements = list() # why waste memory on a separate list? go to line 545
        for change in changes:
            (attribute, diff, codeExists, code) = get_attribute(change['name'])
            value = diff if diff is not None else str(f"\"{change['value']}\"" if isinstance(change['value'], str) else change['value'])

            #  if neither the attribute nor it's parents are written in the source
            if attribute['start'] == (0, 0) and attribute['end'] == (0, 0):
                # write the code below all the already written attributes
                source = self._append_code(source, change['name'], value, root_end[0])
            # if the code of a parent attribute exists
            elif codeExists:
                # get the end position of the last written child attribute of the parent
                end = max(attribute['attributes'].items(), key=lambda item: item[1]['end'])[1]['end']
                if end == (0, 0): # EDGE CASE: if it doesn't have any child attributes written, it's a variable.
                    replacements.append((code, (attribute['start'], attribute['end']))) # replace the variable with the full source code
                # write the code after the last written child attribute
                else: source = self._insert_code(source, change['name'].split(".")[-1], change['value'], end)
            else:
                replacements.append((value, (attribute['start'], attribute['end'])))

        if len(replacements) > 0:
            # yeah imagine you did the replacements in the loop: the whole thing will be a disaster.
            # remember, tokenize gives us the (line number, position) of any given token. if we were to replace the changes at the lines on the top,
            # it'd change the lines numbers and position numbers of all the tokens after it & in the lines below
            # thus, it'd become impossible to replace the changes after the first one.

            # well, the solution? easy, just replace everything in reverse! subsequent changes will only happen in tokens occuring before the last,
            # and so, no position values get messed up.
            for replacement in replacements[::-1]:
                source = self._replace_code(source, replacement[0], replacement[1][0], replacement[1][1])

        return source

    # the driver function ofc
    def save(self, config: PlatformConfig, platform: str):

        changes = self.changed_fields
        if len(changes) < 0: return

        # this platform variable just for searching is a mess i agree, but i couldn't find any other way
        pattern = rf'{platform} = ([a-zA-Z0-9]*)PlatformConfig\((?:[^()]+|\((?:[^()]+|\((?:[^()]+|\([^()]*\))*\))*\))*\)'
        file_path = inspect.getsourcefile(type(config))

        with open(file_path, 'r') as source_file:
            source_code = source_file.read()

        match = re.search(pattern, source_code, re.DOTALL)
        tokens = [token for token in tokenize(io.BytesIO(match.group().encode()).readline)]

        # start where the first ( is found
        start_index = next((index for index, token in enumerate(tokens) if token.string == '('), 0)

        code = self._parse_source(tokens, self.original_fields, start_index, len(tokens)-2)
        diff = self._diff_writer(match.group(), code, changes)

        before, after = source_code[:match.start()], source_code[match.end():]
        source_code = before + diff + after

        # when i tried using w+, it's throwing some error related to reading the file
        # a single file open call would have been better yes, but i couldn't get it to work
        with open(file_path, 'w') as source_file:
            source_file.write(source_code)
