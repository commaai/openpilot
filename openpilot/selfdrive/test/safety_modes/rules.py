# the rules a drive's recording passes, each a (passed, evidence) check, passed None when there was nothing to check
# seconds after the DUT panda's controls_allowed edge by which openpilot must be enabled
ENGAGE_LIMIT = 2.
# seconds the DUT panda's edge may be from the route's, the panda samples at 10 Hz and the replay starts within a second of the recording
EDGE_LIMIT = 3.
# a can message's src: the car's frames on bus 0-2, what the panda sent 128 + bus, what it blocked 192 + bus
BUSES = 3
SENT = 128
BLOCKED = 192
# once the car's CAN stops, the panda flags its rx checks lagging 1 s after their own last frame, checked at 1 Hz, and drops
# controls_allowed. they're >= 10 Hz, the last can come 0.1 s before the car's last frame
CAN_LOSS_DROP = (0.9, 2.5)

Check = tuple[bool | None, str]


def _t(msgs, m) -> float:
  return (m.logMonoTime - msgs[0].logMonoTime) * 1e-9


def _of(msgs, which: str) -> list:
  return [m for m in msgs if m.which() == which and (which != 'pandaStates' or len(m.pandaStates))]


def _car_can(msgs) -> list[float]:
  return [_t(msgs, m) for m in _of(msgs, 'can') if any(c.src < BUSES for c in m.can)]


def _first_engaged(msgs) -> float | None:
  return next((_t(msgs, m) for m in _of(msgs, 'selfdriveState') if m.selfdriveState.enabled), None)


def fingerprinted(msgs, car_model: str) -> Check:
  cps = _of(msgs, 'carParams')
  if not cps:
    return False, "no carParams, openpilot didn't start the car"
  fingerprint = cps[-1].carParams.carFingerprint
  return fingerprint == car_model, f"fingerprinted {fingerprint}, route is {car_model}"


def engaged_at_edge(msgs, expected: float) -> Check:
  allowed = [(_t(msgs, m), m.pandaStates[0].controlsAllowed) for m in _of(msgs, 'pandaStates')]
  edge = next((t for (_, before), (t, a) in zip([(0., False)] + allowed, allowed, strict=False) if a and not before), None)
  if edge is None:
    return False, "the DUT panda never allowed controls"
  if abs(edge - expected) > EDGE_LIMIT:
    return False, f"panda allowed controls at t={edge:.2f}, the route's panda at t={expected:.2f}"
  engaged = _first_engaged(msgs)
  if engaged is None:
    return False, f"panda allowed controls at t={edge:.2f}, openpilot never engaged"
  return abs(engaged - edge) <= ENGAGE_LIMIT, f"panda allowed controls at t={edge:.2f}, openpilot engaged at t={engaged:.2f}"


def none_blocked(msgs) -> Check:
  # openpilot sends and no frame is blocked while the car's CAN runs, after it stops the panda drops controls on its own timer
  last = max(_car_can(msgs), default=0.)
  tx = [(_t(msgs, m), c) for m in _of(msgs, 'can') if _t(msgs, m) <= last for c in m.can if c.src >= SENT]
  blocked = [(t, c) for t, c in tx if c.src >= BLOCKED]
  if not blocked:
    return len(tx) > 0, f"{len(tx)} TX frames sent, none blocked"
  t, c = blocked[0]
  return False, f"{len(blocked)} TX frames blocked, first {c.address:#x} bus {c.src - BLOCKED} at t={t:.2f}, data {bytes(c.dat).hex()}"


def no_new_faults(msgs) -> Check:
  states = _of(msgs, 'pandaStates')
  if not states:
    return False, "no pandaStates"
  before = {str(f) for f in states[0].pandaStates[0].faults}
  new = sorted({str(f) for m in states for f in m.pandaStates[0].faults} - before)
  start = f", at start {', '.join(sorted(before))}" if before else ""
  return not new, (f"new panda faults {', '.join(new)}" if new else "no new panda faults") + start


def dropped_on_can_loss(msgs) -> Check:
  # the panda's own timer and safety tick: controls_allowed drops CAN_LOSS_DROP s after the last car frame it received
  received = _car_can(msgs)
  if not received:
    return False, "no car CAN received"
  last = received[-1]
  after = [m for m in _of(msgs, 'pandaStates') if _t(msgs, m) > last]
  if not after or not after[0].pandaStates[0].controlsAllowed:
    return None, f"controls weren't allowed when the car's CAN stopped at t={last:.2f}, nothing to time"
  drop = next((m for m in after if not m.pandaStates[0].controlsAllowed), None)
  if drop is None:
    return False, f"car CAN stopped at t={last:.2f}, the panda still allowed controls {_t(msgs, after[-1]) - last:.2f} s later"
  delay = _t(msgs, drop) - last
  rx_invalid = drop.pandaStates[0].safetyRxChecksInvalid
  if rx_invalid:
    # the panda's own timer
    ok = CAN_LOSS_DROP[0] <= delay <= CAN_LOSS_DROP[1]
    expected = f"{CAN_LOSS_DROP[0]:g}-{CAN_LOSS_DROP[1]:g} s"
    return ok, f"car CAN stopped at t={last:.2f}, rx checks invalid, controls dropped {delay:.2f} s later (expected {expected})"
  # else openpilot saw the CAN missing first and disengaged, the panda follows its heartbeat
  disengaged = next((_t(msgs, m) for m in _of(msgs, 'selfdriveState') if _t(msgs, m) > last and not m.selfdriveState.enabled), None)
  ok = disengaged is not None and disengaged < _t(msgs, drop) and delay <= CAN_LOSS_DROP[1]
  seen = "never disengaged" if disengaged is None else f"disengaged {disengaged - last:.2f} s later"
  return ok, f"car CAN stopped at t={last:.2f}, openpilot {seen}, controls dropped {delay:.2f} s later with rx checks valid (limit {CAN_LOSS_DROP[1]:g} s)"


def check(msgs, car_model: str, edge: float) -> list[Check]:
  # edge: when the route's panda allowed controls, seconds from the start of the recording
  msgs = sorted(msgs, key=lambda m: m.logMonoTime)
  if not msgs:
    return [(False, "nothing recorded")]
  return [fingerprinted(msgs, car_model), engaged_at_edge(msgs, edge), none_blocked(msgs), no_new_faults(msgs), dropped_on_can_loss(msgs)]
