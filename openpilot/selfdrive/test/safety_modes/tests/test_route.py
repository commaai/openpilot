from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.safety_modes.route import ENGAGED, LEAD, bin_frames, car_mode_samples, find_edge
from openpilot.selfdrive.test.safety_modes.tests.helpers import can, msg, panda


def samples(*states: tuple[float, bool]) -> list[tuple[float, bool]]:
  return car_mode_samples([panda(t, a) for t, a in states])


class TestWindow(OpenpilotTestCase):
  def test_old_logs(self):
    m = msg('pandaStateDEPRECATED', 1., controlsAllowed=True, safetyModel='toyota')
    assert car_mode_samples([m]) == [(1., True)]

  def test_car_mode_only(self):
    # the old device's own switch into the car's safety mode isn't an engagement
    assert car_mode_samples([panda(0., False, mode='elm327'), panda(1., True)]) == [(1., True)]

  def test_edge(self):
    assert find_edge(samples((0., False), (20., True), (40., True)), 60.) == 20.

  def test_needs_lead_in_car_mode(self):
    assert find_edge(samples((5., False), (5. + LEAD - 1, True), (30., True)), 60.) is None
    edge = find_edge(samples((5., False), (5. + LEAD + 0.1, True), (30., True)), 60.)
    assert edge is not None and abs(edge - (5. + LEAD + 0.1)) < 1e-6

  def test_needs_room(self):
    assert find_edge(samples((0., False), (20., True)), 20. + ENGAGED - 1) is None

  def test_needs_cruise_off_lead(self):
    # the replay would start with cruise on
    assert find_edge(samples((0., False), (20., True), (40., True)), 60., cruise_on=[12.]) is None
    assert find_edge(samples((0., False), (20., True), (40., True)), 60., cruise_on=[19.5]) == 20.

  def test_not_resumed(self):
    assert find_edge(samples((0., False), (20., True), (40., True)), 60., resumes=[19.5]) is None
    assert find_edge(samples((0., False), (20., True), (40., True)), 60., resumes=[15.]) == 20.

  def test_needs_hold(self):
    assert find_edge(samples((0., False), (20., True), (20. + ENGAGED - 1, False)), 60.) is None


class TestFrames:
  def test_binning(self):
    msgs = [can(10.0, [(1, 0)]), can(10.015, [(2, 1), (3, 128), (4, 2)]), can(10.035, [(5, 0)]), can(10.05, [(6, 0)]), msg('carState', 10.)]
    frames = bin_frames(msgs, 10., 0.05)
    assert [[addr for addr, _, _ in f] for f in frames] == [[1], [2, 4], [], [5], []]
    assert frames[0][0] == (1, bytes(8), 0)
