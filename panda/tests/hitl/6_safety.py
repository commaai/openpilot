import time

from panda import Panda


def test_safety_nooutput(p):
  p.set_safety_mode(Panda.SAFETY_SILENT)
  p.set_can_loopback(True)

  # send a message on bus 0
  p.can_send(0x1aa, b"message", 0)

  # confirm receive nothing
  time.sleep(0.05)
  r = p.can_recv()
  # bus 192 is messages blocked by TX safety hook on bus 0
  assert len([x for x in r if x[2] != 192]) == 0
  assert len([x for x in r if x[2] == 192]) == 1


def test_canfd_safety_modes(p):
  # works on all pandas
  p.set_safety_mode(Panda.SAFETY_TOYOTA)
  assert p.health()['safety_mode'] == Panda.SAFETY_TOYOTA

  # shouldn't be able to set a CAN-FD safety mode on non CAN-FD panda
  p.set_safety_mode(Panda.SAFETY_HYUNDAI_CANFD)
  expected_mode = Panda.SAFETY_HYUNDAI_CANFD if p.get_type() in Panda.H7_DEVICES else Panda.SAFETY_SILENT
  assert p.health()['safety_mode'] == expected_mode
