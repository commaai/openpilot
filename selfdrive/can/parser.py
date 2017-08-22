import os
import time
import subprocess
from collections import defaultdict

from cffi import FFI

can_dir = os.path.dirname(os.path.abspath(__file__))
libdbc_fn = os.path.join(can_dir, "libdbc.so")
subprocess.check_output(["make"], cwd=can_dir)

ffi = FFI()
ffi.cdef("""

typedef struct SignalParseOptions {
  uint32_t address;
  const char* name;
  double default_value;
} SignalParseOptions;

typedef struct MessageParseOptions {
  uint32_t address;
  int check_frequency;
} MessageParseOptions;

typedef struct SignalValue {
  uint32_t address;
  uint16_t ts;
  const char* name;
  double value;
} SignalValue;

void* can_init(int bus, const char* dbc_name,
              size_t num_message_options, const MessageParseOptions* message_options,
              size_t num_signal_options, const SignalParseOptions* signal_options);

void can_update(void* can, uint64_t sec, bool wait);

size_t can_query(void* can, uint64_t sec, bool *out_can_valid, size_t out_values_size, SignalValue* out_values);

""")

libdbc = ffi.dlopen(libdbc_fn)

class CANParser(object):
  def __init__(self, dbc_name, signals, checks=[], bus=0):
    self.can_valid = True
    self.vl = defaultdict(dict)
    self.ts = defaultdict(dict)

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
        'address': address,
        'check_frequency': freq,
      } for address, freq in message_options.iteritems()])

    self.can = libdbc.can_init(bus, dbc_name, len(message_options_c), message_options_c,
                               len(signal_options_c), signal_options_c)

    self.p_can_valid = ffi.new("bool*")

    value_count = libdbc.can_query(self.can, 0, self.p_can_valid, 0, ffi.NULL)
    self.can_values = ffi.new("SignalValue[%d]" % value_count)
    self.update_vl(0)
    # print "==="

  def update_vl(self, sec):

    can_values_len = libdbc.can_query(self.can, sec, self.p_can_valid, len(self.can_values), self.can_values)
    assert can_values_len <= len(self.can_values)

    self.can_valid = self.p_can_valid[0]

    # print can_values_len
    ret = set()
    for i in xrange(can_values_len):
      cv = self.can_values[i]
      address = cv.address
      # print hex(cv.address), ffi.string(cv.name)
      name = ffi.string(cv.name)
      self.vl[address][name] = cv.value
      self.ts[address][name] = cv.ts
      ret.add(address)
    return ret

  def update(self, sec, wait):
    libdbc.can_update(self.can, sec, wait)
    return self.update_vl(sec)

if __name__ == "__main__":
  from common.realtime import sec_since_boot

  radar_messages = range(0x430, 0x43A) + range(0x440, 0x446)
  # signals = zip(['LONG_DIST'] * 16 + ['NEW_TRACK'] * 16 + ['LAT_DIST'] * 16 +
  #               ['REL_SPEED'] * 16, radar_messages * 4,
  #               [255] * 16 + [1] * 16 + [0] * 16 + [0] * 16)
  # checks = zip(radar_messages, [20]*16)

  # cp = CANParser("acura_ilx_2016_nidec", signals, checks, 1)

  signals = [
    ("XMISSION_SPEED", 0x158, 0), #sig_name, sig_address, default 
    ("WHEEL_SPEED_FL", 0x1d0, 0),
    ("WHEEL_SPEED_FR", 0x1d0, 0),
    ("WHEEL_SPEED_RL", 0x1d0, 0),
    ("STEER_ANGLE", 0x14a, 0),
    ("STEER_TORQUE_SENSOR", 0x18f, 0),
    ("GEAR", 0x191, 0),
    ("WHEELS_MOVING", 0x1b0, 1),
    ("DOOR_OPEN_FL", 0x405, 1),
    ("DOOR_OPEN_FR", 0x405, 1),
    ("DOOR_OPEN_RL", 0x405, 1),
    ("DOOR_OPEN_RR", 0x405, 1),
    ("CRUISE_SPEED_PCM", 0x324, 0),
    ("SEATBELT_DRIVER_LAMP", 0x305, 1),
    ("SEATBELT_DRIVER_LATCHED", 0x305, 0),
    ("BRAKE_PRESSED", 0x17c, 0),
    ("CAR_GAS", 0x130, 0),
    ("CRUISE_BUTTONS", 0x296, 0),
    ("ESP_DISABLED", 0x1a4, 1),
    ("HUD_LEAD", 0x30c, 0),
    ("USER_BRAKE", 0x1a4, 0),
    ("STEER_STATUS", 0x18f, 5),
    ("WHEEL_SPEED_RR", 0x1d0, 0),
    ("BRAKE_ERROR_1", 0x1b0, 1),
    ("BRAKE_ERROR_2", 0x1b0, 1),
    ("GEAR_SHIFTER", 0x191, 0),
    ("MAIN_ON", 0x326, 0),
    ("ACC_STATUS", 0x17c, 0),
    ("PEDAL_GAS", 0x17c, 0),
    ("CRUISE_SETTING", 0x296, 0),
    ("LEFT_BLINKER", 0x326, 0),
    ("RIGHT_BLINKER", 0x326, 0),
    ("COUNTER", 0x324, 0),
    ("ENGINE_RPM", 0x17C, 0)
  ]
  checks = [
    (0x14a, 100), # address, frequency
    (0x158, 100),
    (0x17c, 100),
    (0x191, 100),
    (0x1a4, 50),
    (0x326, 10),
    (0x1b0, 50),
    (0x1d0, 50),
    (0x305, 10),
    (0x324, 10),
    (0x405, 3),
  ]

  cp = CANParser("honda_civic_touring_2016_can", signals, checks, 0)
  print cp.vl

  while True:
    cp.update(int(sec_since_boot()*1e9), True)
    # print cp.vl
    print cp.ts
    print cp.can_valid
    time.sleep(0.01)
