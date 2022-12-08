from common.numpy_fast import interp


def create_steer_command(packer, steer, steer_req):
  """Creates a CAN message for the Toyota Steer Command."""

  values = {
    "STEER_REQUEST": steer_req,
    "STEER_TORQUE_CMD": steer,
    "SET_ME_1": 1,
  }
  return packer.make_can_msg("STEERING_LKA", 0, values)


def create_lta_steer_command(packer, apply_steer, steer_angle, driver_torque, steer_req, op_params):
  """Creates a CAN message for the Toyota LTA Steer Command."""

  percentage = interp(abs(driver_torque), [50, 100], [100, 0])
  apply_steer = interp(percentage, [-10, 100], [steer_angle, apply_steer])
  values = {
    # seems to actually be 1. Even 1 on 2023 RAV4 2023 (TODO: check from data)
    "SETME_X1": op_params.get("SETME_X1"),

    # On a RAV4 2023, it seems to be always 1
    # But other cars it can change randomly?
    # TODO: figure that out
    "SETME_X3": op_params.get("SETME_X3"),

    # 100 when driver not touching wheel, 0 when driver touching wheel. ramps smoothly between
    # TODO: find actual breakpoints and determine how this affects the control
    "PERCENTAGE": op_params.get("PERCENTAGE"),

    # ramps to 0 smoothly then back on falling edge of STEER_REQUEST if BIT isn't 1
    # but on 2023 RAV4, it was constant 100 on falling edge and BIT was 0
    # TODO: figure out if important. torque wind down? maybe user torque blending?
    "SETME_X64": op_params.get("SETME_X64"),

    # TODO: need to understand this better, it's always 1.5-2x higher than angle cmd
    # TODO: revisit on 2023 RAV4
    "ANGLE": op_params.get("ANGLE"),

    # seems to just be desired angle cmd
    # TODO: does this have offset on cars where accurate steering angle signal has offset?
    # some tss2 don't have any offset on the accurate angle signal... (tss2.5)?
    "STEER_ANGLE_CMD": apply_steer,

    # stock system turns off steering after ~20 frames of override, else torque winds up
    "LTA_REQUEST": steer_req,

    # 1 when actively using LTA. 3 when LTA is activated for LKA. 0 when LTA_REQUEST is 0
    # TODO: see if 3 gets us any more torque, or better blending, or SOMETHING. EPS_STATUS doesn't change based on this, so maybe it doesn't do anything
    "LTA_REQUEST_TYPE": op_params.get("LTA_REQUEST_TYPE") if steer_req else 0,

    # 1 when STEER_REQUEST changes state (usually)
    # except not true on 2023 RAV4. TODO: revisit, could it be UI related?
    "BIT": 0,
  }
  return packer.make_can_msg("STEERING_LTA", 0, values)


def create_accel_command(packer, accel, pcm_cancel, standstill_req, lead, acc_type):
  # TODO: find the exact canceling bit that does not create a chime
  values = {
    "ACCEL_CMD": accel,
    "ACC_TYPE": acc_type,
    "DISTANCE": 0,
    "MINI_CAR": lead,
    "PERMIT_BRAKING": 1,
    "RELEASE_STANDSTILL": not standstill_req,
    "CANCEL_REQ": pcm_cancel,
    "ALLOW_LONG_PRESS": 1,
  }
  return packer.make_can_msg("ACC_CONTROL", 0, values)


def create_acc_cancel_command(packer):
  values = {
    "GAS_RELEASED": 0,
    "CRUISE_ACTIVE": 0,
    "STANDSTILL_ON": 0,
    "ACCEL_NET": 0,
    "CRUISE_STATE": 0,
    "CANCEL_REQ": 1,
  }
  return packer.make_can_msg("PCM_CRUISE", 0, values)


def create_fcw_command(packer, fcw):
  values = {
    "PCS_INDICATOR": 1,
    "FCW": fcw,
    "SET_ME_X20": 0x20,
    "SET_ME_X10": 0x10,
    "PCS_OFF": 1,
    "PCS_SENSITIVITY": 0,
  }
  return packer.make_can_msg("ACC_HUD", 0, values)


def create_ui_command(packer, steer, chime, left_line, right_line, left_lane_depart, right_lane_depart, enabled):
  values = {
    "TWO_BEEPS": chime,
    "LDA_ALERT": steer,
    "RIGHT_LINE": 3 if right_lane_depart else 1 if right_line else 2,
    "LEFT_LINE": 3 if left_lane_depart else 1 if left_line else 2,
    "BARRIERS" : 1 if enabled else 0,

    # static signals
    "SET_ME_X02": 2,
    "SET_ME_X01": 1,
    "LKAS_STATUS": 1,
    "REPEATED_BEEPS": 0,
    "LANE_SWAY_FLD": 7,
    "LANE_SWAY_BUZZER": 0,
    "LANE_SWAY_WARNING": 0,
    "LDA_FRONT_CAMERA_BLOCKED": 0,
    "TAKE_CONTROL": 0,
    "LANE_SWAY_SENSITIVITY": 2,
    "LANE_SWAY_TOGGLE": 1,
    "LDA_ON_MESSAGE": 0,
    "LDA_SPEED_TOO_LOW": 0,
    "LDA_SA_TOGGLE": 1,
    "LDA_SENSITIVITY": 2,
    "LDA_UNAVAILABLE": 0,
    "LDA_MALFUNCTION": 0,
    "LDA_UNAVAILABLE_QUIET": 0,
    "ADJUSTING_CAMERA": 0,
    "LDW_EXIST": 1,
  }
  return packer.make_can_msg("LKAS_HUD", 0, values)
