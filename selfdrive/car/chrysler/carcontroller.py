import logging
from common.numpy_fast import clip, interp
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.chrysler.chryslercan import create_2d9, create_2a6, create_292
from selfdrive.car.chrysler.values import ECU, STATIC_MSGS
from selfdrive.can.packer import CANPacker

# Steer torque range is 1024+-230. The 1024 is added by our library.
# degrees * 3.0 = car units
CAR_UNITS_PER_DEGREE = 3.0  # originally 5.1
STEER_MAX = 230 / CAR_UNITS_PER_DEGREE  # degrees
STEER_DELTA_UP = 2.5 / CAR_UNITS_PER_DEGREE  # degrees
STEER_DELTA_DOWN = 2.5 / CAR_UNITS_PER_DEGREE  # degrees
MIN_STEER_MS = 3.8  # consolidate with interface.py low_speed_alert


class CarController(object):
  def __init__(self, dbc_name, car_fingerprint, enable_camera, enable_dsu, enable_apg):
    self.braking = False
    # redundant safety check with the board
    self.controls_allowed = True
    self.last_steer = 0
    self.send_new_status = False  # indicates we want to send 2a6 when we can.
    self.prev_2a6 = -9999  # long time ago.
    self.prev_frame = -1  # previous frame from interface from 220 frame
    self.ccframe = 0
    self.car_fingerprint = car_fingerprint
    self.alert_active = False

    self.fake_ecus = set()
    if enable_camera: self.fake_ecus.add(ECU.CAM)

    self.packer = CANPacker(dbc_name)

    logging.basicConfig(level=logging.DEBUG, filename="/tmp/chrylog", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('CarController init')

  def update(self, sendcan, enabled, CS, frame, actuators,
             pcm_cancel_cmd, hud_alert, audible_alert):

    # *** compute control surfaces ***
    # steer torque
    apply_steer = int(round(actuators.steer * STEER_MAX))
    apply_steer = clip(apply_steer, -STEER_MAX, STEER_MAX)

    # slow rate if steer torque increases in magnitude
    if self.last_steer > 0:
      apply_steer = clip(apply_steer, max(self.last_steer - STEER_DELTA_DOWN, - STEER_DELTA_UP), self.last_steer + STEER_DELTA_UP)
    else:
      apply_steer = clip(apply_steer, self.last_steer - STEER_DELTA_UP, min(self.last_steer + STEER_DELTA_DOWN, STEER_DELTA_UP))

    moving_fast = True  # for status message
    if CS.v_ego < MIN_STEER_MS:  # don't steer if going under 8.5mph to not lock out LKAS (was < 3)
      apply_steer = 0
      moving_fast = False

    if self.last_steer == 0 and apply_steer != 0:
      self.send_new_status = True
    self.last_steer = apply_steer

    if self.prev_frame == frame:
      logging.info('prev_frame == frame so skipping')
      return  # Do not reuse an old frame. This avoids repeating on shut-down.

    can_sends = []

    #*** control msgs ***

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
      else:
        new_msg = create_2a6(CS.gear_shifter, apply_steer, moving_fast, self.car_fingerprint)
        sendcan.send(can_list_to_can_capnp([new_msg], msgtype='sendcan').to_bytes())
        can_sends.append(new_msg)
        self.send_new_status = False
        self.prev_2a6 = self.ccframe
    new_msg = create_292(int(apply_steer * CAR_UNITS_PER_DEGREE), frame, moving_fast)
    self.prev_frame = frame  # save so we do not reuse frames
    sendcan.send(can_list_to_can_capnp([new_msg], msgtype='sendcan').to_bytes())
    can_sends.append(new_msg)  # degrees * 5.1 -> car steering units
    for msg in can_sends:
      [addr, _, dat, _] = msg
      outp  = ('make_can_msg:%s  len:%d  %s' % ('0x{:02x}'.format(addr), len(dat),
                                                ' '.join('{:02x}'.format(ord(c)) for c in dat)))
      logging.info(outp)


    self.ccframe += 1
    # sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
