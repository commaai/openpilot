# functions common among cars
from common.numpy_fast import clip
from selfdrive.config import Conversions as CV
# kg of standard extra cargo to count for drive, gas, etc...
STD_CARGO_KG = 136.

# FIXME: hardcoding honda civic 2016 touring params so they can be used to
# scale unknown params for other cars
class CivicParams:
  mass = 2923. * CV.LB_TO_KG + STD_CARGO_KG
  wheelbase = 2.70
  centerToFront = wheelbase * 0.4
  centerToRear = wheelbase - centerToFront
  rotationalInertia = 2500
  tireStiffnessFront = 192150
  tireStiffnessRear = 202500

# TODO: get actual value, for now starting with reasonable value for
# civic and scaling by mass and wheelbase
def scaleRotInertia(mass, wheelbase):
  return CivicParams.rotationalInertia * mass * wheelbase ** 2 / (CivicParams.mass * CivicParams.wheelbase ** 2)

# TODO: start from empirically derived lateral slip stiffness for the civic and scale by
# mass and CG position, so all cars will have approximately similar dyn behaviors
def scaleTireStiffness(mass, wheelbase, center_to_front, tire_stiffness_factor):
  center_to_rear = wheelbase - center_to_front
  tire_stiffness_front = (CivicParams.tireStiffnessFront * tire_stiffness_factor) * \
                        mass / CivicParams.mass * \
                        (center_to_rear / wheelbase) / (CivicParams.centerToRear / CivicParams.wheelbase)

  tire_stiffness_rear = (CivicParams.tireStiffnessRear * tire_stiffness_factor) * \
                       mass / CivicParams.mass * \
                       (center_to_front / wheelbase) / (CivicParams.centerToFront / CivicParams.wheelbase)

  return tire_stiffness_front, tire_stiffness_rear
    
def dbc_dict(pt_dbc, radar_dbc, chassis_dbc=None):
  return {'pt': pt_dbc, 'radar': radar_dbc, 'chassis': chassis_dbc}


def apply_std_steer_torque_limits(apply_torque, apply_torque_last, driver_torque, LIMITS):

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

  return int(round(apply_torque))


def apply_toyota_steer_torque_limits(apply_torque, apply_torque_last, motor_torque, LIMITS):

  # limits due to comparison of commanded torque VS motor reported torque
  max_lim = min(max(motor_torque + LIMITS.STEER_ERROR_MAX, LIMITS.STEER_ERROR_MAX), LIMITS.STEER_MAX)
  min_lim = max(min(motor_torque - LIMITS.STEER_ERROR_MAX, -LIMITS.STEER_ERROR_MAX), -LIMITS.STEER_MAX)

  apply_torque = clip(apply_torque, min_lim, max_lim)

  # slow rate if steer torque increases in magnitude
  if apply_torque_last > 0:
    apply_torque = clip(apply_torque,
                        max(apply_torque_last - LIMITS.STEER_DELTA_DOWN, -LIMITS.STEER_DELTA_UP),
                        apply_torque_last + LIMITS.STEER_DELTA_UP)
  else:
    apply_torque = clip(apply_torque,
                        apply_torque_last - LIMITS.STEER_DELTA_UP,
                        min(apply_torque_last + LIMITS.STEER_DELTA_DOWN, LIMITS.STEER_DELTA_UP))

  return int(round(apply_torque))


def crc8_pedal(data):
  crc = 0xFF    # standard init value
  poly = 0xD5   # standard crc8: x8+x7+x6+x4+x2+1
  size = len(data)
  for i in range(size-1, -1, -1):
    crc ^= data[i]
    for j in range(8):
      if ((crc & 0x80) != 0):
        crc = ((crc << 1) ^ poly) & 0xFF
      else:
        crc <<= 1
  return crc


def create_gas_command(packer, gas_amount, idx):
  # Common gas pedal msg generator
  enable = gas_amount > 0.001

  values = {
    "ENABLE": enable,
    "COUNTER_PEDAL": idx & 0xF,
  }

  if enable:
    values["GAS_COMMAND"] = gas_amount * 255.
    values["GAS_COMMAND2"] = gas_amount * 255.

  dat = packer.make_can_msg("GAS_COMMAND", 0, values)[2]

  dat = [ord(i) for i in dat]
  checksum = crc8_pedal(dat[:-1])
  values["CHECKSUM_PEDAL"] = checksum

  return packer.make_can_msg("GAS_COMMAND", 0, values)
