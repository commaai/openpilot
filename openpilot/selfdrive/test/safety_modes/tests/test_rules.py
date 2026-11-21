from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.safety_modes.rules import check, dropped_on_can_loss, engaged_at_edge, fingerprinted, no_new_faults, none_blocked
from openpilot.selfdrive.test.safety_modes.tests.helpers import can, msg, panda, sendcan

CAR = 'TOYOTA_COROLLA_TSS2'
# when the fake drive's panda allows controls
EDGE = 2.


QUIET = 3.
DT_LAST = 0.1


def drive(engage_at: float = 2.5, allowed_at: float = 2., end: float = 15., drop_after: float = 1.5, disengage_at: float = 99.,
          gas_at: float = 99.) -> list:
  # 10 Hz from t=0, the panda allows controls at allowed_at, openpilot engages at engage_at, the driver brakes at disengage_at
  # and presses gas at gas_at
  msgs = [msg('carParams', 0., carFingerprint=CAR)]
  for i in range(int(end * 10)):
    t = i / 10
    cs = msg('carState', t, gasPressed=t >= gas_at, brakePressed=t >= disengage_at)
    cs.carState.cruiseState.enabled = engage_at <= t < disengage_at
    msgs += [panda(t, t >= allowed_at), msg('selfdriveState', t, enabled=engage_at <= t < disengage_at), cs, can(t, [(0x2e4, 128)])]
    if t < end - QUIET:
      msgs.append(can(t, [(0xaa, 0)]))
  # the panda drops controls drop_after s after the car's last frame
  for m in msgs:
    if m.which() == 'pandaStates' and m.logMonoTime * 1e-9 > end - QUIET - DT_LAST + drop_after:
      m.pandaStates[0].controlsAllowed = False
      m.pandaStates[0].safetyRxChecksInvalid = True
  return msgs


def blocked(msgs: list, *frames: tuple[float, float, int]) -> bool | None:
  # (sent, blocked, address) of frames openpilot sent and the panda blocked
  return none_blocked(msgs + [m for s, b, addr in frames for m in (sendcan(s, [(addr, 0)]), can(b, [(addr, 192)]))])[0]


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
    # never engaged, controls not allowed when the CAN stopped: skipped, not passed
    assert none_blocked(drive(engage_at=99.))[0] is None
    assert dropped_on_can_loss(drive(allowed_at=99., engage_at=99.))[0] is None

  def test_blocked(self):
    # before openpilot engaged and after the car's CAN stopped nothing is checked, while engaged a block nothing explains fails
    assert blocked(drive(), (0.9, 1., 0x2e4))
    assert blocked(drive(), (13.9, 14., 0x2e4))
    assert blocked(drive(), (2.9, 3., 0x2e4)) is False
    # a blocked frame openpilot never sent
    assert none_blocked(drive(disengage_at=8.) + [can(8.01, [(0x2e4, 192)])])[0] is False

  def test_blocked_at_disengage(self):
    # the driver brakes, openpilot's carState shows it at t=8. what it sent before was on its way, batched or not
    assert blocked(drive(disengage_at=8.), (7.99, 8.01, 0x2e4))
    assert blocked(drive(disengage_at=8.), (7.98, 8.01, 0x2e4), (7.99, 8.01, 0x2e4))
    # sent after its carState showed the brake: acting on what it knew before
    assert blocked(drive(disengage_at=8.), (8.01, 8.02, 0x2e4)) is False
    # an event long after doesn't explain it
    assert blocked(drive(disengage_at=8.), (6.99, 7., 0x2e4)) is False

  def test_blocked_at_gas(self):
    # openpilot stays engaged, the panda blocks its acceleration while gas is pressed
    assert blocked(drive(gas_at=8.), (7.99, 8.01, 0x343))
    assert blocked(drive(gas_at=8.), (8.01, 8.02, 0x343)) is False

  def test_blocked_at_cruise_off(self):
    # the car's cruise turns off at t=10, openpilot's cancel it sent before is blocked
    msgs = drive()
    for m in msgs:
      if m.which() == 'carState':
        m.carState.cruiseState.enabled = 2.5 <= m.logMonoTime * 1e-9 < 10.
    assert blocked(msgs, (9.99, 10.01, 0x4f1))
    assert blocked(msgs, (10.01, 10.02, 0x4f1)) is False

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
