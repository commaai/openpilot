from common.numpy_fast import clip, interp
from common.realtime import sec_since_boot
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit
from selfdrive.car.toyota.toyotacan import make_can_msg, create_video_target,\
                                           create_steer_command, create_ui_command, \
                                           create_ipas_steer_command, create_accel_command, \
                                           create_fcw_command
from selfdrive.car.toyota.values import ECU, STATIC_MSGS
from common.fingerprints import TOYOTA as CAR


ACCEL_HYST_GAP = 0.02  # don't change accel command for small oscilalitons within this value
ACCEL_MAX = 1500  # 1.5 m/s2
ACCEL_MIN = -3000 # 3   m/s2
ACCEL_SCALE = max(ACCEL_MAX, -ACCEL_MIN)

STEER_MAX = 1500
STEER_DELTA_UP = 10        # 1.5s time to peak torque
STEER_DELTA_DOWN = 25      # always lower than 45 otherwise the Rav4 faults (Prius seems ok with 50)
STEER_ERROR_MAX = 350      # max delta between torque cmd and torque motor

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


def process_hud_alert(hud_alert, audible_alert):
  # initialize to no alert
  steer = 0
  fcw = 0
  sound1 = 0
  sound2 = 0

  if hud_alert == 'fcw':
    fcw = 1
  elif hud_alert == 'steerRequired':
    steer = 1

  if audible_alert == 'chimeRepeated':
    sound1 = 1
  elif audible_alert in ['beepSingle', 'chimeSingle', 'chimeDouble']:
    # TODO: find a way to send single chimes
    sound2 = 1

  return steer, fcw, sound1, sound2


class CarController(object):
  def __init__(self, car_fingerprint, enable_camera, enable_dsu, enable_apg):
    self.braking = False
    # redundant safety check with the board
    self.controls_allowed = True
    self.last_steer = 0.
    self.accel_steady = 0.
    self.car_fingerprint = car_fingerprint
    self.alert_active = False
    self.last_standstill = False
    self.standstill_req = False

    self.fake_ecus = set()
    if enable_camera: self.fake_ecus.add(ECU.CAM)
    if enable_dsu: self.fake_ecus.add(ECU.DSU)
    if enable_apg: self.fake_ecus.add(ECU.APGS)

  def update(self, sendcan, enabled, CS, frame, actuators,
             pcm_cancel_cmd, hud_alert, audible_alert):

    # *** compute control surfaces ***
    ts = sec_since_boot()

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

    # on entering standstill, send standstill request
    if CS.standstill and not self.last_standstill:
      self.standstill_req = True
    if CS.pcm_acc_status != 8:
      # pcm entered standstill or it's disabled
      self.standstill_req = False

    self.last_steer = apply_steer
    self.last_accel = apply_accel
    self.last_standstill = CS.standstill

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
        can_sends.append(create_accel_command(apply_accel, pcm_cancel_cmd, self.standstill_req))
      else:
        can_sends.append(create_accel_command(0, pcm_cancel_cmd, False))

    if frame % 10 == 0 and ECU.CAM in self.fake_ecus:
      for addr in TARGET_IDS:
        can_sends.append(create_video_target(frame/10, addr))

    # ui mesg is at 100Hz but we send asap if:
    # - there is something to display
    # - there is something to stop displaying
    alert_out = process_hud_alert(hud_alert, audible_alert)
    steer, fcw, sound1, sound2 = alert_out

    if (any(alert_out) and not self.alert_active) or \
       (not any(alert_out) and self.alert_active):
      send_ui = True
      self.alert_active = not self.alert_active
    else:
      send_ui = False

    if (frame % 100 == 0 or send_ui) and ECU.CAM in self.fake_ecus:
      can_sends.append(create_ui_command(steer, sound1, sound2))
      can_sends.append(create_fcw_command(fcw))

    #*** static msgs ***

    for addr, (ecu, cars, bus, fr_step, vl) in STATIC_MSGS.iteritems():
      if frame % fr_step == 0 and ecu in self.fake_ecus and self.car_fingerprint in cars:
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
