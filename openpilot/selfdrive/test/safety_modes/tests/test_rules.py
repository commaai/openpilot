from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.safety_modes.rules import check, dropped_on_can_loss, engaged_at_edge, fingerprinted, no_new_faults, none_blocked
from openpilot.selfdrive.test.safety_modes.tests.helpers import can, msg, panda

CAR = 'TOYOTA_COROLLA_TSS2'


QUIET = 3.
DT_LAST = 0.1


def drive(engage_at: float = 2.5, allowed_at: float = 2., end: float = 15., drop_after: float = 1.5, disengage_at: float = 99.) -> list:
  # 10 Hz from t=0, the panda allows controls at allowed_at, openpilot engages at engage_at
  msgs = [msg('carParams', 0., carFingerprint=CAR)]
  for i in range(int(end * 10)):
    t = i / 10
    msgs += [panda(t, t >= allowed_at), msg('selfdriveState', t, enabled=engage_at <= t < disengage_at), can(t, [(0x2e4, 128)])]
    if t < end - QUIET:
      msgs.append(can(t, [(0xaa, 0)]))
  # the panda drops controls drop_after s after the car's last frame
  for m in msgs:
    if m.which() == 'pandaStates' and m.logMonoTime * 1e-9 > end - QUIET - DT_LAST + drop_after:
      m.pandaStates[0].controlsAllowed = False
      m.pandaStates[0].safetyRxChecksInvalid = True
  return msgs


class TestRules(OpenpilotTestCase):
  def test_clean(self):
    assert all(ok for ok, _ in check(drive(), CAR))

  def test_fingerprint(self):
    assert not fingerprinted(drive(), 'HONDA_CIVIC')[0]
    assert not fingerprinted([msg('selfdriveState', 0.)], CAR)[0]

  def test_engage_late(self):
    assert engaged_at_edge(drive(engage_at=3.9))[0]
    assert not engaged_at_edge(drive(engage_at=4.1))[0]
    assert not engaged_at_edge(drive(engage_at=10.))[0]
    assert not engaged_at_edge(drive(allowed_at=20.))[0]

  def test_blocked(self):
    assert none_blocked(drive() + [can(1., [(0x2e4, 192)])])[0]
    assert not none_blocked(drive() + [can(3., [(0x2e4, 192)])])[0]
    # after the car's CAN stops the panda drops controls and blocks what openpilot still sends
    assert none_blocked(drive() + [can(14., [(0x2e4, 192)])])[0]

  def test_blocked_after_disallowed(self):
    # the car disengages at t=8, what openpilot sent before it saw that is blocked
    msgs = [m for m in drive() if not (m.which() == 'pandaStates' and 8. <= m.logMonoTime * 1e-9 < 9.)]
    msgs += [panda(t / 10, False) for t in range(80, 90)]
    def blocked(t):
      return none_blocked(sorted(msgs + [can(t, [(0x2e4, 192)])], key=lambda m: m.logMonoTime))[0]
    assert blocked(8.05)
    assert not blocked(7.85)

  def test_faults(self):
    msgs = sorted(drive(), key=lambda m: m.logMonoTime)
    assert no_new_faults([panda(0., False, ['heartbeatLoopWatchdog'])] + msgs + [panda(9., True, ['heartbeatLoopWatchdog'])])[0]
    assert not no_new_faults(msgs + [panda(9., True, ['relayMalfunction'])])[0]

  def test_can_loss(self):
    assert dropped_on_can_loss(drive())[0]
    # a timer running fast or slow, or no tick at all
    assert not dropped_on_can_loss(drive(drop_after=0.5))[0]
    assert not dropped_on_can_loss(drive(drop_after=2.8))[0]
    assert not dropped_on_can_loss(drive(drop_after=10.))[0]

  def test_can_loss_openpilot_first(self):
    # openpilot sees the CAN missing and disengages, the panda follows its heartbeat without rx checks invalid
    msgs = drive(drop_after=0.5, disengage_at=12.2)
    for m in msgs:
      if m.which() == 'pandaStates':
        m.pandaStates[0].safetyRxChecksInvalid = False
    assert dropped_on_can_loss(msgs)[0]
    assert not dropped_on_can_loss(drive(drop_after=2.8, disengage_at=12.2))[0]
    # the panda's own timer still has to be in time when openpilot disengaged first
    assert not dropped_on_can_loss(drive(drop_after=0.5, disengage_at=12.2))[0]
