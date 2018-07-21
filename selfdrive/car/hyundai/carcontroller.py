from common.numpy_fast import clip, interp
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.hyundai.hyundaican import make_can_msg, create_lkas11, create_lkas12
from selfdrive.car.hyundai.values import ECU, STATIC_MSGS, CAR
from selfdrive.can.packer import CANPacker


# Steer torque limits
STEER_MAX = 250
STEER_DELTA = 10      

TARGET_IDS = [0x340]


class CarController(object):
  def __init__(self, dbc_name, car_fingerprint, enable_camera):
    self.braking = False
    self.controls_allowed = True
    self.last_steer = 0
    self.car_fingerprint = car_fingerprint
    self.angle_control = False
    self.idx = 0
    self.lkas_request = 0
    self.lanes = 0
    self.steer_angle_enabled = False
    self.ipas_reset_counter = 0
    self.turning_inhibit = 0
    print self.car_fingerprint

    self.fake_ecus = set()
    if enable_camera: self.fake_ecus.add(ECU.CAM)
    self.packer = CANPacker(dbc_name)

  def update(self, sendcan, enabled, CS, frame, actuators):
    # Steering Torque Scaling
    apply_steer = int(round((actuators.steer * STEER_MAX) + 1024))

    # This is redundant clipping code, kept in case it needs to be advanced
    max_lim = 1024 + STEER_MAX
    min_lim = 1024 - STEER_MAX
    apply_steer = clip(apply_steer, min_lim, max_lim)

    # Very basic Rate Limiting
    # TODO: Revisit this
    if (apply_steer - self.last_steer) > STEER_DELTA:
      apply_steer = self.last_steer + STEER_DELTA
    elif (self.last_steer - apply_steer) > STEER_DELTA:
      apply_steer = self.last_steer - STEER_DELTA


    # Inhibits *outside of* alerts
    #    Because the Turning Indicator Status is based on Lights and not Stalk, latching is 
    #    needed for the disable to work.
    if CS.left_blinker_on == 1 or CS.right_blinker_on == 1:
      self.turning_inhibit = 150  # Disable for 1.5 Seconds after blinker turned off

    if self.turning_inhibit > 0:
      self.turning_inhibit = self.turning_inhibit - 1



    if not enabled or self.turning_inhibit > 0:
      apply_steer = 1024
      self.last_steer = 1024
      self.lanes = 0         # Lanes is shown on the LKAS screen on the Dash
    else:
      self.lanes = 3 * 4     # bit 0 = Right Lane, bit 1 = Left Lane, Offset by 2 bits in byte.

    self.last_steer = apply_steer

    can_sends = []

    # Limit Terminal Debugging to 5Hz
    if (frame % 20) == 0:
      print "controlsdDebug steer", actuators.steer, "bi", self.turning_inhibit, "spd", \
        CS.v_ego, "strAng", CS.angle_steers, "strToq", CS.steer_torque_driver

    # Index is 4 bits long, this is the counter
    self.idx = self.idx + 1
    if self.idx >= 16:
      self.idx = 0
    
    lkas11_byte4 = self.idx * 16

    # Split apply steer Word into 2 Bytes
    apply_steer_a = apply_steer & 0xFF
    apply_steer_b = (apply_steer >> 8) & 0xFF
    apply_steer_b = apply_steer_b

    #print "check", steer_chksum_a, steer_chksum_b

    # If requested to steer, turn on ActToi
    if apply_steer != 1024:
      apply_steer_b = apply_steer_b + 0x08

    # High Beam Assist State
    if CS.car_fingerprint == CAR.STINGER or CS.car_fingerprint == CAR.ELANTRA:
      # HBA Sys State
      apply_steer_b = apply_steer_b + 0x20
      lkas11_byte4 = lkas11_byte4 + 0x04


    # Create Checksum
    checksum = (self.lanes + 0x00 + apply_steer_a + apply_steer_b + \
      lkas11_byte4 + 0x00) % 256
    

    # Creake LKAS11 Message at 100Hz
    can_sends.append(create_lkas11(self.packer, self.lanes, \
      0x00, apply_steer_a, apply_steer_b, lkas11_byte4, \
      0x00, checksum, 0x18))
   

    # Create LKAS12 Message at 10Hz
    if (frame % 10) == 0:
      if CS.car_fingerprint == CAR.SORENTO:
        can_sends.append(create_lkas12(self.packer, 0x20, 0x00))
      if CS.car_fingerprint == CAR.STINGER:
        can_sends.append(create_lkas12(self.packer, 0x80, 0x05))


    # Send messages to canbus
    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
