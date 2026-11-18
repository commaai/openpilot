from opendbc.car.tests.routes import CarTestRoute
from openpilot.selfdrive.test.cars.route import ENGAGED, LEAD, allowed_samples, bin_frames, find_edge, segments_to_try
from openpilot.selfdrive.test.cars.tests.helpers import can, msg, panda


def samples(*states: tuple[float, bool]) -> list[tuple[float, bool]]:
  return allowed_samples([panda(t, a) for t, a in states])


class TestWindow:
  def test_segments(self):
    assert segments_to_try(CarTestRoute("r", None)) == [2, 1, 0, 3, 4, 5, 6, 7, 8, 9]
    assert segments_to_try(CarTestRoute("r", None, 5)) == [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]

  def test_old_logs(self):
    m = msg('pandaStateDEPRECATED', 1., controlsAllowed=True)
    assert allowed_samples([m]) == [(1., True)]

  def test_edge(self):
    s = samples((0., False), (20., True), (40., True))
    assert find_edge(s, 0., 60.) == 20.

  def test_needs_lead(self):
    s = samples((0., False), (10., True), (30., False), (40., True), (55., True))
    assert find_edge(s, 0., 60.) == 40.
    assert find_edge(s, 0., 40. + ENGAGED - 1) is None

  def test_needs_hold(self):
    s = samples((0., False), (20., True), (20. + ENGAGED - 1, False))
    assert find_edge(s, 0., 60.) is None

  def test_no_edge_when_always_allowed(self):
    assert find_edge(samples((LEAD + 1, True), (LEAD + 20, True)), 0., 60.) is None


class TestFrames:
  def test_binning(self):
    msgs = [can(10.0, [(1, 0)]), can(10.015, [(2, 1), (3, 128), (4, 2)]), can(10.035, [(5, 0)]), can(10.05, [(6, 0)]), msg('carState', 10.)]
    frames = bin_frames(msgs, 10., 0.05)
    assert [[addr for addr, _, _ in f] for f in frames] == [[1], [2, 4], [], [5], []]
    assert frames[0][0] == (1, bytes(8), 0)
