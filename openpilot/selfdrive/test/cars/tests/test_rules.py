from openpilot.selfdrive.test.cars.rules import (can_valid, check, engaged_at_edge, fingerprinted, no_new_faults, none_blocked,
                                                 stayed_engaged)
from openpilot.selfdrive.test.cars.tests.helpers import can, msg, panda

CAR = 'TOYOTA_COROLLA_TSS2'


def drive(engage_at: float = 2.5, allowed_at: float = 2., end: float = 6.) -> list:
  # 10 Hz from t=0, the panda allows controls at allowed_at, openpilot engages at engage_at
  msgs = [msg('carParams', 0., carFingerprint=CAR)]
  for i in range(int(end * 10)):
    t = i / 10
    msgs += [panda(t, t >= allowed_at), msg('selfdriveState', t, enabled=t >= engage_at), msg('carState', t, canValid=t >= 1.),
             can(t, [(0x2e4, 128)])]
  return msgs


class TestRules:
  def test_clean(self):
    assert all(ok for ok, _ in check(drive(), CAR, edge=True))

  def test_fingerprint(self):
    assert not fingerprinted(drive(), 'HONDA_CIVIC')[0]
    assert not fingerprinted([msg('carState', 0.)], CAR)[0]

  def test_can_valid(self):
    assert not can_valid([msg('carState', 0., canValid=False)])[0]
    assert not can_valid(drive() + [msg('carState', 5.55, canValid=False)])[0]

  def test_engage_late(self):
    assert engaged_at_edge(drive(engage_at=3.9))[0]
    assert not engaged_at_edge(drive(engage_at=4.1))[0]
    assert not engaged_at_edge(drive(engage_at=10.))[0]
    assert not engaged_at_edge(drive(allowed_at=10.))[0]

  def test_no_edge_skips_engage(self):
    assert len(check(drive(engage_at=10.), CAR, edge=False)) == 5
    assert all(ok for ok, _ in check(drive(engage_at=10., allowed_at=10.), CAR, edge=False))

  def test_disengaged(self):
    assert not stayed_engaged(drive() + [msg('selfdriveState', 5.55, enabled=False)])[0]
    assert not stayed_engaged(drive() + [panda(5.55, False)])[0]

  def test_blocked(self):
    assert none_blocked(drive() + [can(1., [(0x2e4, 192)])])[0]
    assert not none_blocked(drive() + [can(3., [(0x2e4, 192)])])[0]

  def test_faults(self):
    msgs = sorted(drive(), key=lambda m: m.logMonoTime)
    assert no_new_faults([panda(0., False, ['heartbeatLoopWatchdog'])] + msgs + [panda(9., True, ['heartbeatLoopWatchdog'])])[0]
    assert not no_new_faults(msgs + [panda(9., True, ['relayMalfunction'])])[0]
