from openpilot.cereal.messaging import new_message


def msg(which: str, t: float, size: int | None = None, **fields):
  m = new_message(which, size) if size is not None else new_message(which)
  m.logMonoTime = int(t * 1e9)
  for name, value in fields.items():
    setattr(m.pandaStates[0] if which == 'pandaStates' else getattr(m, which), name, value)
  return m


def panda(t: float, allowed: bool, faults: list[str] | None = None):
  m = msg('pandaStates', t, 1, controlsAllowed=allowed)
  m.pandaStates[0].faults = faults or []
  return m


def can(t: float, frames: list[tuple[int, int]], dat: bytes = bytes(8)):
  m = new_message('can', len(frames))
  m.logMonoTime = int(t * 1e9)
  for c, (addr, src) in zip(m.can, frames, strict=True):
    c.address, c.dat, c.src = addr, dat, src
  return m
