import time

from opendbc.car.structs import CarParams


def test_safety_nooutput(p):
  p.set_safety_mode(CarParams.SafetyModel.silent)
  p.set_can_loopback(True)

  # send a message on bus 0
  p.can_send(0x1aa, b"message", 0)

  # confirm receive nothing
  time.sleep(0.05)
  r = p.can_recv()
  # bus 192 is messages blocked by TX safety hook on bus 0
  assert len([x for x in r if x[2] != 192]) == 0
  assert len([x for x in r if x[2] == 192]) == 1
