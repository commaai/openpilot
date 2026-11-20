# the rules a drive's recording passes, each a (passed, evidence) check
# seconds after the DUT panda's controls_allowed edge by which openpilot must be enabled
ENGAGE_LIMIT = 2.
# the panda echoes what it blocked with src 192 + bus
BLOCKED = 192
# once the car's CAN stops, the panda flags its rx checks lagging after 1 s, checked at 1 Hz, and drops controls_allowed
CAN_LOSS_DROP = (1., 2.5)
# seconds around openpilot's disengagement the frame openpilot sent before it saw it reaches the panda
IN_FLIGHT = 0.1

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
  # while the car's CAN runs. when the car disengages the panda sees it on its CAN before openpilot does, the one frame of
  # each message openpilot already sent is rightly blocked. a second is openpilot actuating after it should have seen it
  last = max((_t(msgs, m) for m in _of(msgs, 'can') if any(c.src < 3 for c in m.can)), default=0.)
  states = [(_t(msgs, m), m.selfdriveState.enabled) for m in _of(msgs, 'selfdriveState')]
  disengages = [t for (_, before), (t, enabled) in zip(states, states[1:], strict=False) if before and not enabled and t > engaged]
  in_flight: set[tuple[float, int, int]] = set()
  blocked = []
  for m in _of(msgs, 'can'):
    t = _t(msgs, m)
    if not engaged <= t <= last:
      continue
    for c in m.can:
      if c.src < BLOCKED:
        continue
      d = next((d for d in disengages if abs(t - d) <= IN_FLIGHT), None)
      if d is not None and (d, c.address, c.src) not in in_flight:
        in_flight.add((d, c.address, c.src))
      else:
        blocked.append((t, c))
  if blocked:
    t, c = blocked[0]
    return False, f"{len(blocked)} TX frames blocked, first {c.address:#x} bus {c.src - BLOCKED} at t={t:.2f}, data {bytes(c.dat).hex()}"
  return True, f"no TX frames blocked, {len(in_flight)} in flight at {len(disengages)} disengagements"


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
  if rx_invalid:
    # the panda's own timer
    ok = CAN_LOSS_DROP[0] <= delay <= CAN_LOSS_DROP[1]
    expected = f"{CAN_LOSS_DROP[0]:g}-{CAN_LOSS_DROP[1]:g} s"
    return ok, f"car CAN stopped at t={last:.2f}, rx checks invalid, controls dropped {delay:.2f} s later (expected {expected})"
  # else openpilot saw the CAN missing first and disengaged, the panda follows its heartbeat
  disengaged = next((_t(msgs, m) for m in _of(msgs, 'selfdriveState') if _t(msgs, m) > last and not m.selfdriveState.enabled), None)
  ok = disengaged is not None and disengaged < _t(msgs, drop) and delay <= CAN_LOSS_DROP[1]
  after = "never disengaged" if disengaged is None else f"disengaged {disengaged - last:.2f} s later"
  return ok, f"car CAN stopped at t={last:.2f}, openpilot {after}, controls dropped {delay:.2f} s later with rx checks valid (limit {CAN_LOSS_DROP[1]:g} s)"


def check(msgs, car_model: str) -> list[Check]:
  msgs = sorted(msgs, key=lambda m: m.logMonoTime)
  if not msgs:
    return [(False, "nothing recorded")]
  return [fingerprinted(msgs, car_model), engaged_at_edge(msgs), none_blocked(msgs), no_new_faults(msgs), dropped_on_can_loss(msgs)]
