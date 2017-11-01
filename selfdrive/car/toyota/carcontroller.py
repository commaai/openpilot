from common.numpy_fast import clip, interp
from common.realtime import sec_since_boot
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit
from selfdrive.car.toyota.toyotacan import make_can_msg, create_video_target,\
                                           create_steer_command, create_ui_command, \
                                           create_ipas_steer_command, create_accel_command


ACCEL_HYST_GAP = 0.02  # don't change accel command for small oscilalitons within this value
ACCEL_MAX = 1500  # 1.5 m/s2
ACCEL_MIN = -3000 # 3   m/s2
ACCEL_SCALE = max(ACCEL_MAX, -ACCEL_MIN)

STEER_MAX = 1500
STEER_DELTA_UP = 10        # 1.5s time to peak torque
STEER_DELTA_DOWN = 45      # lower than 50 otherwise the Rav4 faults (Prius seems ok though)
STEER_ERROR_MAX = 500      # max delta between torque cmd and torque motor

STEER_IPAS_MAX = 340
STEER_IPAS_DELTA_MAX = 3

class CAR:
  PRIUS = "TOYOTA PRIUS 2017"
  RAV4 = "TOYOTA RAV4 2017"

class ECU:
  CAM = 0 # camera
  DSU = 1 # driving support unit
  APGS = 2 # advanced parking guidance system


TARGET_IDS = [0x340, 0x341, 0x342, 0x343, 0x344, 0x345,
              0x363, 0x364, 0x365, 0x370, 0x371, 0x372,
              0x373, 0x374, 0x375, 0x380, 0x381, 0x382,
              0x383]

# addr, [ecu, bus, 1/freq*100, vl]
STATIC_MSGS = {0x141: (ECU.DSU, (CAR.PRIUS, CAR.RAV4), 1,   2, '\x00\x00\x00\x46'),
               0x128: (ECU.DSU, (CAR.PRIUS, CAR.RAV4), 1,   3, '\xf4\x01\x90\x83\x00\x37'),

               0x292: (ECU.APGS, (CAR.PRIUS, CAR.RAV4), 0,   3, '\x00\x00\x00\x00\x00\x00\x00\x9e'),
               0x283: (ECU.DSU, (CAR.PRIUS, CAR.RAV4), 0,   3, '\x00\x00\x00\x00\x00\x00\x8c'),
               0x2E6: (ECU.DSU, (CAR.PRIUS,), 0,   3, '\xff\xf8\x00\x08\x7f\xe0\x00\x4e'),
               0x2E7: (ECU.DSU, (CAR.PRIUS,), 0,   3, '\xa8\x9c\x31\x9c\x00\x00\x00\x02'),

               0x240: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               0x241: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               0x244: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               0x245: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               0x248: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 1,   5, '\x00\x00\x00\x00\x00\x00\x01'),
               0x344: (ECU.DSU, (CAR.PRIUS, CAR.RAV4), 0,   5, '\x00\x00\x01\x00\x00\x00\x00\x50'),

               0x160: (ECU.DSU, (CAR.PRIUS, CAR.RAV4), 1,   7, '\x00\x00\x08\x12\x01\x31\x9c\x51'),
               0x161: (ECU.DSU, (CAR.PRIUS, CAR.RAV4), 1,   7, '\x00\x1e\x00\x00\x00\x80\x07'),

               0x32E: (ECU.APGS, (CAR.PRIUS, CAR.RAV4), 0,  20, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               0x33E: (ECU.DSU, (CAR.PRIUS,), 0,  20, '\x0f\xff\x26\x40\x00\x1f\x00'),
               0x365: (ECU.DSU, (CAR.PRIUS,), 0,  20, '\x00\x00\x00\x80\x03\x00\x08'),
               0x365: (ECU.DSU, (CAR.RAV4,), 0,  20, '\x00\x00\x00\x80\xfc\x00\x08'),
               0x366: (ECU.DSU, (CAR.PRIUS,), 0,  20, '\x00\x00\x4d\x82\x40\x02\x00'),
               0x366: (ECU.DSU, (CAR.RAV4,), 0,  20, '\x00\x72\x07\xff\x09\xfe\x00'),

               0x367: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 0,  40, '\x06\x00'),

               0x414: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x17\x00'),
               0x489: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
               0x48a: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
               0x48b: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x66\x06\x08\x0a\x02\x00\x00\x00'),
               0x4d3: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x1C\x00\x00\x01\x00\x00\x00\x00'),
               0x130: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 1, 100, '\x00\x00\x00\x00\x00\x00\x38'),
               0x466: (ECU.CAM, (CAR.PRIUS, CAR.RAV4), 1, 100, '\x20\x20\xAD'),
               0x396: (ECU.APGS, (CAR.PRIUS, CAR.RAV4), 0, 100, '\xBD\x00\x00\x00\x60\x0F\x02\x00'),
               0x43A: (ECU.APGS, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x84\x00\x00\x00\x00\x00\x00\x00'),
               0x43B: (ECU.APGS, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               0x497: (ECU.APGS, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               0x4CC: (ECU.APGS, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x0D\x00\x00\x00\x00\x00\x00\x00'),
               0x411: (ECU.DSU, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x00\x20\x00\x00\x10\x00\x80\x00'),
               0x4CB: (ECU.DSU, (CAR.PRIUS, CAR.RAV4), 0, 100, '\x0c\x00\x00\x00\x00\x00\x00\x00'),
               0x470: (ECU.DSU, (CAR.PRIUS,), 1, 100, '\x00\x00\x02\x7a'),
              }


def check_ecu_msgs(fingerprint, candidate, ecu):
  # return True if fingerprint contains messages normally sent by a given ecu
  ecu_msgs = [x for x in STATIC_MSGS if (ecu == STATIC_MSGS[x][0] and 
                                         candidate in STATIC_MSGS[x][1] and 
                                         STATIC_MSGS[x][2] == 0)]

  return any(msg for msg in fingerprint if msg in ecu_msgs)


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


def process_hud_alert(hud_alert, audible_alert):
  # initialize to no alert
  hud1 = 0
  hud2 = 0
  if hud_alert in ['steerRequired', 'fcw']:
    if audible_alert == 'chimeRepeated':
      hud1 = 3
    else:
      hud1 = 1
  if audible_alert in ['beepSingle', 'chimeSingle', 'chimeDouble']:
    # TODO: find a way to send single chimes
    hud2 = 1

  return hud1, hud2


class CarController(object):
  def __init__(self, car_fingerprint, enable_camera, enable_dsu, enable_apg):
    self.braking = False
    # redundant safety check with the board
    self.controls_allowed = True
    self.last_steer = 0.
    self.accel_steady = 0.
    self.car_fingerprint = car_fingerprint
    self.alert_active = False

    self.fake_ecus = set()
    if enable_camera: self.fake_ecus.add(ECU.CAM)
    if enable_dsu: self.fake_ecus.add(ECU.DSU)
    if enable_apg: self.fake_ecus.add(ECU.APGS)

  def update(self, sendcan, enabled, CS, frame, actuators,
             pcm_cancel_cmd, hud_alert, audible_alert):

    # *** compute control surfaces ***
    tt = sec_since_boot()

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_accel = actuators.gas - actuators.brake
    apply_accel, self.accel_steady = accel_hysteresis(apply_accel, self.accel_steady, enabled)
    apply_accel = int(round(clip(apply_accel * ACCEL_SCALE, ACCEL_MIN, ACCEL_MAX)))

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_steer = int(round(actuators.steer * STEER_MAX))

    max_lim = min(max(CS.steer_torque_motor + STEER_ERROR_MAX, STEER_ERROR_MAX), STEER_MAX)
    min_lim = max(min(CS.steer_torque_motor - STEER_ERROR_MAX, -STEER_ERROR_MAX), -STEER_MAX)

    apply_steer = clip(apply_steer, min_lim, max_lim)

    # slow rate if steer torque increases in magnitude
    if self.last_steer > 0:
      apply_steer = clip(apply_steer, max(self.last_steer - STEER_DELTA_DOWN, - STEER_DELTA_UP), self.last_steer + STEER_DELTA_UP)
    else:
      apply_steer = clip(apply_steer, self.last_steer - STEER_DELTA_UP, min(self.last_steer + STEER_DELTA_DOWN, STEER_DELTA_UP))

    if not enabled and CS.pcm_acc_status:
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      pcm_cancel_cmd = 1

    # dropping torque immediately might cause eps to temp fault. On the other hand, safety_toyota
    # cuts steer torque immediately anyway TODO: monitor if this is a real issue
    if not enabled or CS.steer_error:
      apply_steer = 0

    self.last_steer = apply_steer
    self.last_accel = apply_accel

    can_sends = []

    #*** control msgs ***
    #print "steer", apply_steer, min_lim, max_lim, CS.steer_torque_motor

    # toyota can trace shows this message at 42Hz, with counter adding alternatively 1 and 2;
    # sending it at 100Hz seem to allow a higher rate limit, as the rate limit seems imposed
    # on consecutive messages
    if ECU.CAM in self.fake_ecus:
      can_sends.append(create_steer_command(apply_steer, frame))

    if ECU.APGS in self.fake_ecus:
      can_sends.append(create_ipas_steer_command(apply_steer))

    # accel cmd comes from DSU, but we can spam can to cancel the system even if we are using lat only control
    if (frame % 3 == 0 and ECU.DSU in self.fake_ecus) or (pcm_cancel_cmd and ECU.CAM in self.fake_ecus):
      if ECU.DSU in self.fake_ecus:
        can_sends.append(create_accel_command(apply_accel, pcm_cancel_cmd))
      else:
        can_sends.append(create_accel_command(0, pcm_cancel_cmd))

    if frame % 10 == 0 and ECU.CAM in self.fake_ecus:
      for addr in TARGET_IDS:
        can_sends.append(create_video_target(frame/10, addr))

    # ui mesg is at 100Hz but we send asap if:
    # - there is something to display
    # - there is something to stop displaying
    hud1, hud2 = process_hud_alert(hud_alert, audible_alert)
    if ((hud1 or hud2) and not self.alert_active) or \
       (not (hud1 or hud2) and self.alert_active):
      send_ui = True
      self.alert_active = not self.alert_active
    else:
      send_ui = False

    if (frame % 100 == 0 or send_ui) and ECU.CAM in self.fake_ecus:
      can_sends.append(create_ui_command(hud1, hud2))

    #*** static msgs ***

    for addr, (ecu, cars, bus, fr_step, vl) in STATIC_MSGS.iteritems():
      if frame % fr_step == 0 and ecu in self.fake_ecus and self.car_fingerprint in cars:
        # send msg!
        # special cases

        if fr_step == 5 and ecu == ECU.CAM and bus == 1:
          cnt = (((frame / 5) % 7) + 1) << 5
          vl = chr(cnt) + vl
        elif addr in (0x489, 0x48a) and bus == 0:
          # add counter for those 2 messages (last 4 bits)
          cnt = ((frame/100)%0xf) + 1
          if addr == 0x48a:
            # 0x48a has a 8 preceding the counter
            cnt += 1 << 7
          vl += chr(cnt)

        can_sends.append(make_can_msg(addr, vl, bus, False))


    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
