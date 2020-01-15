from panda.tests.safety import libpandasafety_py

MAX_WRONG_COUNTERS = 5

def make_msg(bus, addr, length=8):
  to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
  if addr >= 0x800:
    to_send[0].RIR = (addr << 3) | 5
  else:
    to_send[0].RIR = (addr << 21) | 1
  to_send[0].RDTR = length
  to_send[0].RDTR |= bus << 4

  return to_send

def test_relay_malfunction(test, addr):
  # input is a test class and the address that, if seen on bus 0, triggers
  # the relay_malfunction protection logic: both tx_hook and fwd_hook are
  # expected to return failure
  test.assertFalse(test.safety.get_relay_malfunction())
  test.safety.safety_rx_hook(make_msg(0, addr, 8))
  test.assertTrue(test.safety.get_relay_malfunction())
  for a in range(1, 0x800):
    for b in range(0, 3):
      test.assertFalse(test.safety.safety_tx_hook(make_msg(b, a, 8)))
      test.assertEqual(-1, test.safety.safety_fwd_hook(b, make_msg(b, a, 8)))

def test_manually_enable_controls_allowed(test):
  test.safety.set_controls_allowed(1)
  test.assertTrue(test.safety.get_controls_allowed())
  test.safety.set_controls_allowed(0)
  test.assertFalse(test.safety.get_controls_allowed())

def test_spam_can_buses(test, TX_MSGS):
  for addr in range(1, 0x800):
    for bus in range(0, 4):
      if all(addr != m[0] or bus != m[1] for m in TX_MSGS):
        test.assertFalse(test.safety.safety_tx_hook(make_msg(bus, addr, 8)))
