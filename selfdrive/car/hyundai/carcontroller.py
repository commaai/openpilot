from common.numpy_fast import clip, interp
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.hyundai.hyundaican import make_can_msg, create_video_target, create_steer_command
from selfdrive.car.hyundai.values import ECU, STATIC_MSGS
from selfdrive.can.packer import CANPacker


# Steer torque limits
STEER_MAX = 1500
STEER_DELTA_UP = 10       # 1.5s time to peak torque
STEER_DELTA_DOWN = 25     # always lower than 45 otherwise the Rav4 faults (Prius seems ok with 50)
STEER_ERROR_MAX = 350     # max delta between torque cmd and torque motor

# Steer angle limits (tested at the Crows Landing track and considered ok)
ANGLE_MAX_BP = [0., 5.]
ANGLE_MAX_V = [510., 300.]
ANGLE_DELTA_BP = [0., 5., 15.]
ANGLE_DELTA_V = [5., .8, .15]     # windup limit
ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit

TARGET_IDS = [0x340]



class CarController(object):
  def __init__(self, dbc_name, car_fingerprint, enable_camera):
    self.braking = False
    # redundant safety check with the board
    self.controls_allowed = False
    self.last_steer = 0
    self.car_fingerprint = car_fingerprint
    self.angle_control = False

    self.steer_angle_enabled = False
    self.ipas_reset_counter = 0

    self.fake_ecus = set()
    if enable_camera: self.fake_ecus.add(ECU.CAM)
    self.packer = CANPacker(dbc_name)

  def update(self, sendcan, enabled, CS, frame, actuators):

    # *** compute control surfaces ***

    # steer torque
    apply_steer = int(round(actuators.steer * STEER_MAX))

    max_lim = min(max(CS.steer_torque_motor + STEER_ERROR_MAX, STEER_ERROR_MAX), STEER_MAX)
    min_lim = max(min(CS.steer_torque_motor - STEER_ERROR_MAX, -STEER_ERROR_MAX), -STEER_MAX)

    apply_steer = clip(apply_steer, min_lim, max_lim)

    # slow rate if steer torque increases in magnitude
    if self.last_steer > 0:
      apply_steer = clip(apply_steer, max(self.last_steer - STEER_DELTA_DOWN, - STEER_DELTA_UP), self.last_steer + STEER_DELTA_UP)
    else:
      apply_steer = clip(apply_steer, self.last_steer - STEER_DELTA_UP, min(self.last_steer + STEER_DELTA_DOWN, STEER_DELTA_UP))

    # dropping torque immediately might cause eps to temp fault. On the other hand, safety_toyota
    # cuts steer torque immediately anyway TODO: monitor if this is a real issue
    # only cut torque when steer state is a known fault
    if not enabled:
      apply_steer = 0


    self.last_steer = apply_steer

    can_sends = []

    #*** control msgs ***
    #print "steer", apply_steer, min_lim, max_lim, CS.steer_torque_motor

    #counts from 0 to 15 then back to 0
    idx = (frame / P.STEER_STEP) % 16

    if not lkas_enabled:
      apply_steer = 0

     
    #Max steer = 1023
    if actuators.steer < 0:
      chksm_steer = 1024-abs(apply_steer)
    else:
      chksm_steer = apply_steer
      
    steer2 = (chksm_steer >> 8) & 0x7
    steer1 =  chksm_steer - (steer2 << 8)
    checksum = (idx + steer1 + steer2 + lkas_request) % 256


    can_sends.append(hyundaican.create_lkas11(self.packer, apply_steer, idx, checksum))
    can_sends.append(hyundaican.create_lkas12(self.packer))


    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
