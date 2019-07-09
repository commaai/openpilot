import time
import numbers

from selfdrive.can.libdbc_py import libdbc, ffi

CAN_INVALID_CNT = 5  # after so many consecutive CAN data with wrong checksum, counter or frequency, flag CAN invalidity

class CANParser(object):
  def __init__(self, dbc_name, signals, checks=None, bus=0, sendcan=False, tcp_addr="127.0.0.1", timeout=-1):
    if checks is None:
      checks = []

    self.can_valid = True
    self.can_invalid_cnt = CAN_INVALID_CNT
    self.vl = {}
    self.ts = {}

    self.dbc_name = dbc_name
    self.dbc = libdbc.dbc_lookup(dbc_name)
    self.msg_name_to_addres = {}
    self.address_to_msg_name = {}

    num_msgs = self.dbc[0].num_msgs
    for i in range(num_msgs):
      msg = self.dbc[0].msgs[i]

      name = ffi.string(msg.name)
      address = msg.address

      self.msg_name_to_addres[name] = address
      self.address_to_msg_name[address] = name

      self.vl[address] = {}
      self.vl[name] = {}
      self.ts[address] = {}
      self.ts[name] = {}

    # Convert message names into addresses
    for i in range(len(signals)):
      s = signals[i]
      if not isinstance(s[1], numbers.Number):
        s = (s[0], self.msg_name_to_addres[s[1]], s[2])
        signals[i] = s

    for i in range(len(checks)):
      c = checks[i]
      if not isinstance(c[0], numbers.Number):
        c = (self.msg_name_to_addres[c[0]], c[1])
        checks[i] = c

    sig_names = dict((name, ffi.new("char[]", name)) for name, _, _ in signals)

    signal_options_c = ffi.new("SignalParseOptions[]", [
      {
        'address': sig_address,
        'name': sig_names[sig_name],
        'default_value': sig_default,
      } for sig_name, sig_address, sig_default in signals])

    message_options = dict((address, 0) for _, address, _ in signals)
    message_options.update(dict(checks))

    message_options_c = ffi.new("MessageParseOptions[]", [
      {
        'address': msg_address,
        'check_frequency': freq,
      } for msg_address, freq in message_options.items()])

    self.can = libdbc.can_init(bus, dbc_name, len(message_options_c), message_options_c,
                               len(signal_options_c), signal_options_c, sendcan, tcp_addr, timeout)

    self.p_can_valid = ffi.new("bool*")

    value_count = libdbc.can_query(self.can, 0, self.p_can_valid, 0, ffi.NULL)
    self.can_values = ffi.new("SignalValue[%d]" % value_count)
    self.update_vl(0)

  def update_vl(self, sec):
    can_values_len = libdbc.can_query(self.can, sec, self.p_can_valid, len(self.can_values), self.can_values)
    assert can_values_len <= len(self.can_values)

    self.can_invalid_cnt += 1
    if self.p_can_valid[0]:
      self.can_invalid_cnt = 0
    self.can_valid = self.can_invalid_cnt < CAN_INVALID_CNT

    ret = set()
    for i in xrange(can_values_len):
      cv = self.can_values[i]
      address = cv.address
      # print("{0} {1}".format(hex(cv.address), ffi.string(cv.name)))
      name = ffi.string(cv.name)
      self.vl[address][name] = cv.value
      self.ts[address][name] = cv.ts

      sig_name = self.address_to_msg_name[address]
      self.vl[sig_name][name] = cv.value
      self.ts[sig_name][name] = cv.ts
      ret.add(address)
    return ret

  def update(self, sec, wait):
    """Returns if the update was successfull (e.g. no rcv timeout happened)"""
    r = libdbc.can_update(self.can, sec, wait)
    return bool(r >= 0), self.update_vl(sec)


if __name__ == "__main__":
  from common.realtime import sec_since_boot

  radar_messages = range(0x430, 0x43A) + range(0x440, 0x446)
  # signals = zip(['LONG_DIST'] * 16 + ['NEW_TRACK'] * 16 + ['LAT_DIST'] * 16 +
  #               ['REL_SPEED'] * 16, radar_messages * 4,
  #               [255] * 16 + [1] * 16 + [0] * 16 + [0] * 16)
  # checks = zip(radar_messages, [20]*16)

  # cp = CANParser("acura_ilx_2016_nidec", signals, checks, 1)


  # signals = [
  #   ("XMISSION_SPEED", 0x158, 0), #sig_name, sig_address, default
  #   ("WHEEL_SPEED_FL", 0x1d0, 0),
  #   ("WHEEL_SPEED_FR", 0x1d0, 0),
  #   ("WHEEL_SPEED_RL", 0x1d0, 0),
  #   ("STEER_ANGLE", 0x14a, 0),
  #   ("STEER_TORQUE_SENSOR", 0x18f, 0),
  #   ("GEAR", 0x191, 0),
  #   ("WHEELS_MOVING", 0x1b0, 1),
  #   ("DOOR_OPEN_FL", 0x405, 1),
  #   ("DOOR_OPEN_FR", 0x405, 1),
  #   ("DOOR_OPEN_RL", 0x405, 1),
  #   ("DOOR_OPEN_RR", 0x405, 1),
  #   ("CRUISE_SPEED_PCM", 0x324, 0),
  #   ("SEATBELT_DRIVER_LAMP", 0x305, 1),
  #   ("SEATBELT_DRIVER_LATCHED", 0x305, 0),
  #   ("BRAKE_PRESSED", 0x17c, 0),
  #   ("CAR_GAS", 0x130, 0),
  #   ("CRUISE_BUTTONS", 0x296, 0),
  #   ("ESP_DISABLED", 0x1a4, 1),
  #   ("HUD_LEAD", 0x30c, 0),
  #   ("USER_BRAKE", 0x1a4, 0),
  #   ("STEER_STATUS", 0x18f, 5),
  #   ("WHEEL_SPEED_RR", 0x1d0, 0),
  #   ("BRAKE_ERROR_1", 0x1b0, 1),
  #   ("BRAKE_ERROR_2", 0x1b0, 1),
  #   ("GEAR_SHIFTER", 0x191, 0),
  #   ("MAIN_ON", 0x326, 0),
  #   ("ACC_STATUS", 0x17c, 0),
  #   ("PEDAL_GAS", 0x17c, 0),
  #   ("CRUISE_SETTING", 0x296, 0),
  #   ("LEFT_BLINKER", 0x326, 0),
  #   ("RIGHT_BLINKER", 0x326, 0),
  #   ("COUNTER", 0x324, 0),
  #   ("ENGINE_RPM", 0x17C, 0)
  # ]
  # checks = [
  #   (0x14a, 100), # address, frequency
  #   (0x158, 100),
  #   (0x17c, 100),
  #   (0x191, 100),
  #   (0x1a4, 50),
  #   (0x326, 10),
  #   (0x1b0, 50),
  #   (0x1d0, 50),
  #   (0x305, 10),
  #   (0x324, 10),
  #   (0x405, 3),
  # ]

  # cp = CANParser("honda_civic_touring_2016_can_generated", signals, checks, 0)


  signals = [
    # sig_name, sig_address, default
    ("GEAR", 956, 0x20),
    ("BRAKE_PRESSED", 548, 0),
    ("GAS_PEDAL", 705, 0),

    ("WHEEL_SPEED_FL", 170, 0),
    ("WHEEL_SPEED_FR", 170, 0),
    ("WHEEL_SPEED_RL", 170, 0),
    ("WHEEL_SPEED_RR", 170, 0),
    ("DOOR_OPEN_FL", 1568, 1),
    ("DOOR_OPEN_FR", 1568, 1),
    ("DOOR_OPEN_RL", 1568, 1),
    ("DOOR_OPEN_RR", 1568, 1),
    ("SEATBELT_DRIVER_UNLATCHED", 1568, 1),
    ("TC_DISABLED", 951, 1),
    ("STEER_ANGLE", 37, 0),
    ("STEER_FRACTION", 37, 0),
    ("STEER_RATE", 37, 0),
    ("GAS_RELEASED", 466, 0),
    ("CRUISE_STATE", 466, 0),
    ("MAIN_ON", 467, 0),
    ("SET_SPEED", 467, 0),
    ("STEER_TORQUE_DRIVER", 608, 0),
    ("STEER_TORQUE_EPS", 608, 0),
    ("TURN_SIGNALS", 1556, 3),   # 3 is no blinkers
    ("LKA_STATE", 610, 0),
  ]
  checks = [
    (548, 40),
    (705, 33),

    (170, 80),
    (37, 80),
    (466, 33),
    (608, 50),
  ]

  cp = CANParser("toyota_rav4_2017_pt_generated", signals, checks, 0)

  # print(cp.vl)

  while True:
    cp.update(int(sec_since_boot()*1e9), True)
    # print(cp.vl)
    print(cp.ts)
    print(cp.can_valid)
    time.sleep(0.01)
