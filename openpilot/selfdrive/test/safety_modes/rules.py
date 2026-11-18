# the rules a drive's recording passes, each a (passed, evidence) check
# seconds after the DUT panda's controls_allowed edge by which openpilot must be enabled
ENGAGE_LIMIT = 2.
# the panda echoes what it blocked with src 192 + bus
BLOCKED = 192
# once the car's CAN stops, the panda flags its rx checks lagging after 1 s, checked at 1 Hz, and drops controls_allowed
CAN_LOSS_DROP = (1., 2.5)

Check = tuple[bool, str]


def _t(msgs, m) -> float:
  return (m.logMonoTime - msgs[0].logMonoTime) * 1e-9


def _of(msgs, which: str) -> list:
  return [m for m in msgs if m.which() == which and (which != 'pandaStates' or len(m.pandaStates))]


def _first_engaged(msgs) -> float | None:
  return next((_t(msgs, m) for m in _of(msgs, 'selfdriveState') if m.selfdriveState.enabled), None)


def fingerprinted(msgs, car_model: str) -> Check:
  cps = _of(msgs, 'carParams')
  if not cps:
    return False, "no carParams, openpilot didn't start the car"
  fingerprint = cps[-1].carParams.carFingerprint
  return fingerprint == car_model, f"fingerprinted {fingerprint}, route is {car_model}"


def engaged_at_edge(msgs) -> Check:
  allowed = [(_t(msgs, m), m.pandaStates[0].controlsAllowed) for m in _of(msgs, 'pandaStates')]
  edge = next((t for (_, before), (t, a) in zip([(0., False)] + allowed, allowed, strict=False) if a and not before), None)
  if edge is None:
    return False, "the DUT panda never allowed controls"
  engaged = _first_engaged(msgs)
  if engaged is None:
    return False, f"panda allowed controls at t={edge:.2f}, openpilot never engaged"
  return abs(engaged - edge) <= ENGAGE_LIMIT, f"panda allowed controls at t={edge:.2f}, openpilot engaged at t={engaged:.2f}"


def none_blocked(msgs) -> Check:
  engaged = _first_engaged(msgs)
  if engaged is None:
    return True, "never engaged, no TX checked"
  # until the car's CAN stops, after it the panda rightly blocks what openpilot still sends
  last = max((_t(msgs, m) for m in _of(msgs, 'can') if any(c.src < 3 for c in m.can)), default=0.)
  blocked = [(_t(msgs, m), c.address, c.src - BLOCKED) for m in _of(msgs, 'can') if engaged <= _t(msgs, m) <= last for c in m.can if c.src >= BLOCKED]
  if blocked:
    t, addr, bus = blocked[0]
    return False, f"{len(blocked)} TX frames blocked while engaged, first {addr:#x} bus {bus} at t={t:.2f}"
  return True, "no TX frames blocked while engaged"


def no_new_faults(msgs) -> Check:
  states = _of(msgs, 'pandaStates')
  if not states:
    return False, "no pandaStates"
  before = {str(f) for f in states[0].pandaStates[0].faults}
  new = sorted({str(f) for m in states for f in m.pandaStates[0].faults} - before)
  return not new, f"new panda faults {', '.join(new)}" if new else "no new panda faults"


def dropped_on_can_loss(msgs) -> Check:
  # the panda's own timer and safety tick: controls_allowed drops CAN_LOSS_DROP s after the last car frame it received
  received = [_t(msgs, m) for m in _of(msgs, 'can') if any(c.src < 3 for c in m.can)]
  if not received:
    return False, "no car CAN received"
  last = received[-1]
  after = [m for m in _of(msgs, 'pandaStates') if _t(msgs, m) > last]
  if not after or not after[0].pandaStates[0].controlsAllowed:
    return True, f"controls weren't allowed when the car's CAN stopped at t={last:.2f}, nothing to time"
  drop = next((m for m in after if not m.pandaStates[0].controlsAllowed), None)
  if drop is None:
    return False, f"car CAN stopped at t={last:.2f}, the panda still allowed controls {_t(msgs, after[-1]) - last:.2f} s later"
  delay = _t(msgs, drop) - last
  rx_invalid = drop.pandaStates[0].safetyRxChecksInvalid
  ok = CAN_LOSS_DROP[0] <= delay <= CAN_LOSS_DROP[1] and rx_invalid
  expected = f"{CAN_LOSS_DROP[0]:g}-{CAN_LOSS_DROP[1]:g} s"
  return ok, f"car CAN stopped at t={last:.2f}, controls dropped {delay:.2f} s later (expected {expected}), rx checks invalid {rx_invalid}"


def check(msgs, car_model: str) -> list[Check]:
  msgs = sorted(msgs, key=lambda m: m.logMonoTime)
  if not msgs:
    return [(False, "nothing recorded")]
  return [fingerprinted(msgs, car_model), engaged_at_edge(msgs), none_blocked(msgs), no_new_faults(msgs), dropped_on_can_loss(msgs)]
