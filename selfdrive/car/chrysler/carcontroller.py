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
# degrees * 5.1 = car units
CAR_UNITS_PER_DEGREE = 3.0  # originally 5.1
STEER_MAX = 230 / CAR_UNITS_PER_DEGREE  # degrees
STEER_DELTA_UP = 2.5 / CAR_UNITS_PER_DEGREE  # degrees
STEER_DELTA_DOWN = 2.5 / CAR_UNITS_PER_DEGREE  # degrees
STEER_ERROR_MAX = 350     # max delta between torque cmd and torque motor

# Steer angle limits (tested at the Crows Landing track and considered ok)
ANGLE_MAX_BP = [0., 5.]
ANGLE_MAX_V = [510., 300.]
ANGLE_DELTA_BP = [0., 5., 15.]
ANGLE_DELTA_V = [5., .8, .15]     # windup limit
ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit


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
    self.send_new_status = False  # indicates we want to send 2a6 when we can.
    self.prev_2a6 = -9999  # long time ago.
    self.ccframe = 0
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
    # outp = 'carcontroller apply_steer %s  actuators.steerAngle %s' % (apply_steer, actuators.steerAngle)
    # print outp

    # max_lim = min(max(CS.steer_torque_motor + STEER_ERROR_MAX, STEER_ERROR_MAX), STEER_MAX)
    # min_lim = max(min(CS.steer_torque_motor - STEER_ERROR_MAX, -STEER_ERROR_MAX), -STEER_MAX)

    # apply_steer = clip(apply_steer, min_lim, max_lim)
    apply_steer = clip(apply_steer, -STEER_MAX, STEER_MAX)

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
      # outp = '  apply_angle:%s  angle_lim:%s  angle_rate_lim:%s  apply_steer:%s' % (apply_angle, angle_lim, angle_rate_lim, apply_steer)
      # print outp
      # outp = '  CS.angle_steers:%s  CS.v_ego:%s' % (CS.angle_steers, CS.v_ego)
      # print outp
#    else:
#      apply_angle = CS.angle_steers  # just sets it to the current steering angle


    self.standstill_req = False #?

    moving_fast = True  # for status message
    if CS.v_ego < 3.5:  # don't steer if going under 7.8mph to not lock out LKAS (was < 3)
      apply_angle = 0
      apply_steer = 0
      moving_fast = False

    if self.last_steer == 0 and apply_steer != 0:
      self.send_new_status = True
    self.last_steer = apply_steer
    self.last_angle = apply_angle
    self.last_accel = apply_accel
    self.last_standstill = CS.standstill

    can_sends = []

    #*** control msgs ***
    #print "steer", apply_steer, min_lim, max_lim, CS.steer_torque_motor
    # can_sends.append(create_steer_command(self.packer, apply_steer, frame))
    # TODO verify units and see if we want apply_steer or apply_angle

    # frame is 100Hz (0.01s period)
    if (self.ccframe % 10 == 0):  # 0.1s period
      new_msg = create_2d9(self.car_fingerprint)
      sendcan.send(can_list_to_can_capnp([new_msg], msgtype='sendcan').to_bytes())
      can_sends.append(new_msg)
    if (self.ccframe % 25 == 0) or self.send_new_status:  # 0.25s period
      if (self.ccframe - self.prev_2a6) < 20:  # at least 200ms (20 frames) since last 2a6.
        self.send_new_status = True  # will not send, so send next time.
        apply_steer = 0  # cannot steer yet, waiting for 2a6 to be sent.
        last_steer = 0
        last_angle = 0
      else:
        new_msg = create_2a6(CS.gear_shifter, apply_steer, moving_fast, self.car_fingerprint)
        sendcan.send(can_list_to_can_capnp([new_msg], msgtype='sendcan').to_bytes())
        can_sends.append(new_msg)
        self.send_new_status = False
        self.prev_2a6 = self.ccframe
    new_msg = create_292(int(apply_steer * CAR_UNITS_PER_DEGREE), frame, moving_fast)
    sendcan.send(can_list_to_can_capnp([new_msg], msgtype='sendcan').to_bytes())
    can_sends.append(new_msg)  # degrees * 5.1 -> car steering units
    for msg in can_sends:
      [addr, _, dat, _] = msg
      #outp  = ('make_can_msg:%s  len:%d  %s' % ('0x{:02x}'.format(addr), len(dat),
      #                                          ' '.join('{:02x}'.format(ord(c)) for c in dat)))


    self.ccframe += 1
    # sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
