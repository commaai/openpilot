# the rules a drive's recording passes, each a (passed, evidence) check
# seconds after the DUT panda's controls_allowed edge by which openpilot must be enabled
ENGAGE_LIMIT = 2.
# the panda echoes what it blocked with src 192 + bus
BLOCKED = 192

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
  blocked = [(_t(msgs, m), c.address, c.src - BLOCKED) for m in _of(msgs, 'can') if _t(msgs, m) >= engaged for c in m.can if c.src >= BLOCKED]
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


def check(msgs, car_model: str) -> list[Check]:
  msgs = sorted(msgs, key=lambda m: m.logMonoTime)
  if not msgs:
    return [(False, "nothing recorded")]
  return [fingerprinted(msgs, car_model), engaged_at_edge(msgs), none_blocked(msgs), no_new_faults(msgs)]
