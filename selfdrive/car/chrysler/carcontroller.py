from common.numpy_fast import clip, interp
from selfdrive.boardd.boardd import can_list_to_can_capnp
# from selfdrive.car.toyota.toyotacan import make_can_msg, create_video_target,\
#                                            create_steer_command, create_ui_command, \
#                                            create_ipas_steer_command, create_accel_command, \
#                                            create_fcw_command
from selfdrive.car.chrysler.chryslercan import create_2d9, create_2a6, create_292
from selfdrive.car.chrysler.values import ECU, STATIC_MSGS
from selfdrive.can.packer import CANPacker

# Accel limits
ACCEL_HYST_GAP = 0.02  # don't change accel command for small oscilalitons within this value
ACCEL_MAX = 1.5  # 1.5 m/s2
ACCEL_MIN = -3.0 # 3   m/s2
ACCEL_SCALE = max(ACCEL_MAX, -ACCEL_MIN)

# Steer torque range is 1024+-230. The 1024 is added by our library.
STEER_MAX = 230
STEER_DELTA_UP = 4
STEER_DELTA_DOWN = 5
STEER_ERROR_MAX = 350     # max delta between torque cmd and torque motor

# Steer angle limits (tested at the Crows Landing track and considered ok)
ANGLE_MAX_BP = [0., 5.]
ANGLE_MAX_V = [510., 300.]
ANGLE_DELTA_BP = [0., 5., 15.]
ANGLE_DELTA_V = [5., .8, .15]     # windup limit
ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit

TARGET_IDS = [0x340, 0x341, 0x342, 0x343, 0x344, 0x345,
              0x363, 0x364, 0x365, 0x370, 0x371, 0x372,
              0x373, 0x374, 0x375, 0x380, 0x381, 0x382,
              0x383]


def accel_hysteresis(accel, accel_steady, enabled):

  # for small accel oscillations within ACCEL_HYST_GAP, don't change the accel command
  if not enabled:
    # send 0 when disabled, otherwise acc faults
    accel_steady = 0.
  elif accel > accel_steady + ACCEL_HYST_GAP:
    accel_steady = accel - ACCEL_HYST_GAP
  elif accel < accel_steady - ACCEL_HYST_GAP:
    accel_steady = accel + ACCEL_HYST_GAP
  accel = accel_steady

  return accel, accel_steady


class CarController(object):
  def __init__(self, dbc_name, car_fingerprint, enable_camera, enable_dsu, enable_apg):
    self.braking = False
    # redundant safety check with the board
    self.controls_allowed = True
    self.last_steer = 0
    self.last_angle = 0
    self.accel_steady = 0.
    self.car_fingerprint = car_fingerprint
    self.alert_active = False
    self.last_standstill = False
    self.standstill_req = False
    self.angle_control = False

    self.steer_angle_enabled = False
    self.ipas_reset_counter = 0

    self.fake_ecus = set()
    if enable_camera: self.fake_ecus.add(ECU.CAM)
    #if enable_dsu: self.fake_ecus.add(ECU.DSU)
    #if enable_apg: self.fake_ecus.add(ECU.APGS)

    self.packer = CANPacker(dbc_name)

  def update(self, sendcan, enabled, CS, frame, actuators,
             pcm_cancel_cmd, hud_alert, audible_alert):

    # *** compute control surfaces ***

    # gas and brake
    apply_accel = actuators.gas - actuators.brake
    apply_accel, self.accel_steady = accel_hysteresis(apply_accel, self.accel_steady, enabled)
    apply_accel = clip(apply_accel * ACCEL_SCALE, ACCEL_MIN, ACCEL_MAX)

    # steer torque
    apply_steer = int(round(actuators.steer * STEER_MAX))
    # TODO use these values to decide if we should use apply_steer or apply_angle
    print 'carcontroller apply_steer %s  actuators.steerAngle %s' % (apply_steer, actuators.steerAngle)

    # max_lim = min(max(CS.steer_torque_motor + STEER_ERROR_MAX, STEER_ERROR_MAX), STEER_MAX)
    # min_lim = max(min(CS.steer_torque_motor - STEER_ERROR_MAX, -STEER_ERROR_MAX), -STEER_MAX)

    # apply_steer = clip(apply_steer, min_lim, max_lim)

    # slow rate if steer torque increases in magnitude
    if self.last_steer > 0:
      apply_steer = clip(apply_steer, max(self.last_steer - STEER_DELTA_DOWN, - STEER_DELTA_UP), self.last_steer + STEER_DELTA_UP)
    else:
      apply_steer = clip(apply_steer, self.last_steer - STEER_DELTA_UP, min(self.last_steer + STEER_DELTA_DOWN, STEER_DELTA_UP))


    #self.steer_angle_enabled, self.ipas_reset_counter = \
    #  ipas_state_transition(self.steer_angle_enabled, enabled, CS.ipas_active, self.ipas_reset_counter)
    #print self.steer_angle_enabled, self.ipas_reset_counter, CS.ipas_active

    # steer angle
    self.steer_angle_enabled = True  #!!! TODO use if we are doing apply_angle (instead of apply_steer)
    if self.steer_angle_enabled:
      apply_angle = actuators.steerAngle
      angle_lim = interp(CS.v_ego, ANGLE_MAX_BP, ANGLE_MAX_V)
      apply_angle = clip(apply_angle, -angle_lim, angle_lim)

      # windup slower
      if self.last_angle * apply_angle > 0. and abs(apply_angle) > abs(self.last_angle):
        angle_rate_lim = interp(CS.v_ego, ANGLE_DELTA_BP, ANGLE_DELTA_V)
      else:
        angle_rate_lim = interp(CS.v_ego, ANGLE_DELTA_BP, ANGLE_DELTA_VU)

      apply_angle = clip(apply_angle, self.last_angle - angle_rate_lim, self.last_angle + angle_rate_lim)
#    else:
#      apply_angle = CS.angle_steers  # just sets it to the current steering angle?


    self.standstill_req = False #?

    self.last_steer = apply_steer
    self.last_angle = apply_angle
    self.last_accel = apply_accel
    self.last_standstill = CS.standstill

    can_sends = []

    #*** control msgs ***
    #print "steer", apply_steer, min_lim, max_lim, CS.steer_torque_motor

    # toyota can trace shows this message at 42Hz, with counter adding alternatively 1 and 2;
    # sending it at 100Hz seem to allow a higher rate limit, as the rate limit seems imposed
    # on consecutive messages
    # We could try this higher message rate to turn wheel more. but we're just sending angle?
    # can_sends.append(create_steer_command(self.packer, apply_steer, frame))
    # TODO verify units and see if we want apply_steer or apply_angle
    # TODO verify frame is the counter we want. 

    #*** static msgs ***
    # TODO send the static messages here:
    if (frame % 10 == 0):  # 0.1s period
      can_sends.append(create_2d9())
    if (frame % 25 == 0):  # 0.25s period
      can_sends.append(create_2a6(CS.gear_shifter, (apply_steer != 0)))
    #####!!!! TODO sending apply_angle=0 until we verify units
    apply_angle = 0
    can_sends.append(create_292(apply_angle, frame))


    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
