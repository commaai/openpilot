from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.safety_modes.rules import SENT, check, dropped_on_can_loss, engaged_at_edge, fingerprinted, no_new_faults, none_died, \
  sends
from openpilot.selfdrive.test.safety_modes.tests.helpers import can, manager, msg, panda

CAR = 'TOYOTA_COROLLA_TSS2'
# when the fake drive's panda allows controls
EDGE = 2.


QUIET = 3.
DT_LAST = 0.1


def drive(engage_at: float = 2.5, allowed_at: float = 2., end: float = 15., drop_after: float = 1.5, disengage_at: float = 99.,
          mode_at: float = 0.5) -> list:
  # 10 Hz from t=0, pandad sets the car's safety mode at mode_at, the panda allows controls at allowed_at,
  # openpilot engages at engage_at and disengages at disengage_at
  cp = msg('carParams', 0., carFingerprint=CAR)
  cp.carParams.init('safetyConfigs', 1)[0].safetyModel = 'toyota'
  msgs = [cp]
  for i in range(int(end * 10)):
    t = i / 10
    mode = 'toyota' if t >= mode_at else 'elm327'
    msgs += [panda(t, t >= allowed_at, mode=mode), msg('selfdriveState', t, enabled=engage_at <= t < disengage_at), can(t, [(0x2e4, 128)]),
             manager(t, {'card': (True, 1)})]
    if t < end - QUIET:
      msgs.append(can(t, [(0xaa, 0)]))
  # the panda drops controls drop_after s after the car's last frame
  for m in msgs:
    if m.which() == 'pandaStates' and m.logMonoTime * 1e-9 > end - QUIET - DT_LAST + drop_after:
      m.pandaStates[0].controlsAllowed = False
      m.pandaStates[0].safetyRxChecksInvalid = True
  return msgs


def blocked(t: float) -> list:
  return drive() + [can(t, [(0x2e4, 192)])]


class TestRules(OpenpilotTestCase):
  def test_clean(self):
    assert all(ok for ok, _ in check(drive(), CAR, 2.))

  def test_fingerprint(self):
    assert fingerprinted(drive(), 'HONDA_CIVIC')[0] is False
    assert fingerprinted([msg('selfdriveState', 0.)], CAR)[0] is False

  def test_engage_late(self):
    assert engaged_at_edge(drive(engage_at=3.9), EDGE)[0]
    assert engaged_at_edge(drive(engage_at=4.1), EDGE)[0] is False
    assert engaged_at_edge(drive(engage_at=10.), EDGE)[0] is False
    assert engaged_at_edge(drive(allowed_at=20.), EDGE)[0] is False

  def test_edge_where_the_route_had_it(self):
    # the DUT panda allows controls when the route's did, not at some later engagement
    assert engaged_at_edge(drive(), 2. + 2.9)[0]
    assert engaged_at_edge(drive(), 2. + 3.1)[0] is False

  def test_nothing_to_check(self):
    # controls not allowed when the CAN stopped: skipped, not passed
    assert dropped_on_can_loss(drive(allowed_at=99., engage_at=99.))[0] is None

  def test_blocked_reported(self):
    ok, evidence = sends(blocked(8.))
    assert ok and "1 blocked, first 0x2e4" in evidence
    assert "none blocked" in sends(blocked(14.))[1]

  def test_car_mode(self):
    assert sends(drive(mode_at=99.))[0] is False

  def test_nothing_sent(self):
    # a card that never sends doesn't pass
    assert sends(drive())[0]
    assert sends([m for m in drive() if m.which() != 'can' or m.can[0].src < SENT])[0] is False

  def test_died(self):
    alive = [manager(t, {'card': (True, 1)}) for t in (1., 2., 3.)]
    assert none_died(alive)[0]
    # starting up isn't dying
    assert none_died([manager(0., {'card': (False, 0)})] + alive)[0]
    # stopped, or crashed and restarted by manager
    assert none_died(alive + [manager(4., {'card': (False, 1)})])[0] is False
    assert none_died(alive + [manager(4., {'card': (True, 2)})])[0] is False
    assert none_died([])[0] is False

  def test_faults(self):
    msgs = sorted(drive(), key=lambda m: m.logMonoTime)
    assert no_new_faults([panda(0., False, ['heartbeatLoopWatchdog'])] + msgs + [panda(9., True, ['heartbeatLoopWatchdog'])])[0]
    assert no_new_faults(msgs + [panda(9., True, ['relayMalfunction'])])[0] is False

  def test_can_loss(self):
    assert dropped_on_can_loss(drive())[0]
    # a timer running fast or slow, or no tick at all
    assert dropped_on_can_loss(drive(drop_after=0.5))[0] is False
    # an rx check's last frame up to 0.1 s before the car's last one
    assert dropped_on_can_loss(drive(drop_after=0.85))[0]
    assert dropped_on_can_loss(drive(drop_after=0.75))[0] is False
    assert dropped_on_can_loss(drive(drop_after=2.8))[0] is False
    assert dropped_on_can_loss(drive(drop_after=10.))[0] is False

  def test_can_loss_openpilot_first(self):
    # openpilot sees the CAN missing and disengages, the panda follows its heartbeat without rx checks invalid
    msgs = drive(drop_after=0.5, disengage_at=12.2)
    for m in msgs:
      if m.which() == 'pandaStates':
        m.pandaStates[0].safetyRxChecksInvalid = False
    assert dropped_on_can_loss(msgs)[0]
    assert dropped_on_can_loss(drive(drop_after=2.8, disengage_at=12.2))[0] is False
    # the panda's own timer still has to be in time when openpilot disengaged first
    assert dropped_on_can_loss(drive(drop_after=0.5, disengage_at=12.2))[0] is False
