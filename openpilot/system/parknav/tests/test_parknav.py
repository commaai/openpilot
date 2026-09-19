import math

import pytest

from openpilot.cereal import log
from openpilot.system.parknav.geometry import geodetic_to_local_ned, wrap_angle
from openpilot.selfdrive.modeld.park_nav import NAV_MAX_SPEED, NavDesireInjector


def test_wrap_angle():
  assert wrap_angle(0.) == 0.
  assert abs(wrap_angle(math.pi)) == pytest.approx(math.pi)
  assert abs(wrap_angle(-math.pi)) == pytest.approx(math.pi)  # IEEE remainder keeps the endpoint  assert abs(wrap_angle(3 * math.pi)) == pytest.approx(math.pi)
  assert wrap_angle(math.pi + 1e-6) == pytest.approx(-math.pi + 1e-6, abs=1e-5)
  assert wrap_angle(-3 * math.pi) == pytest.approx(math.pi, abs=1e-9)
  assert wrap_angle(math.pi / 2) == pytest.approx(math.pi / 2)
  assert wrap_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)


def test_geodetic_local_ned():
  north, east = geodetic_to_local_ned(37., -122., 37., -122.)
  assert north == pytest.approx(0., abs=1e-6)
  assert east == pytest.approx(0., abs=1e-6)

  # one degree east at lat 37 ~ 88.8 km east
  north, east = geodetic_to_local_ned(37., -121., 37., -122.)
  assert north == pytest.approx(0., abs=1.)
  assert east == pytest.approx(88.9e3, rel=0.01)

  # one degree north ~ 111 km north
  north, east = geodetic_to_local_ned(38., -122., 37., -122.)
  assert north == pytest.approx(111.3e3, rel=0.01)
  assert east == pytest.approx(0., abs=1.)


class FakeNavSignal:
  def __init__(self, rel_bearing, valid=True):
    self.relBearing = rel_bearing
    self.valid = valid


class FakeCS:
  def __init__(self, left_blindspot=False, right_blindspot=False):
    self.leftBlindspot = left_blindspot
    self.rightBlindspot = right_blindspot


def test_nav_desire_hysteresis():
  injector = NavDesireInjector()
  cs = FakeCS()

  assert injector.update(FakeNavSignal(math.radians(5.)), True, cs, True, 5., log.Desire.none) == log.Desire.none
  assert injector.update(FakeNavSignal(math.radians(20.)), True, cs, True, 5., log.Desire.none) == log.Desire.turnRight
  assert injector.update(FakeNavSignal(math.radians(12.)), True, cs, True, 5., log.Desire.none) == log.Desire.turnRight
  assert injector.update(FakeNavSignal(math.radians(5.)), True, cs, True, 5., log.Desire.none) == log.Desire.none


def test_nav_desire_direction_and_blindspot():
  injector = NavDesireInjector()
  cs = FakeCS()
  assert injector.update(FakeNavSignal(-math.radians(20.)), True, cs, True, 5., log.Desire.none) == log.Desire.turnLeft

  assert injector.update(FakeNavSignal(-math.radians(20.)), True, FakeCS(left_blindspot=True), True, 5.,
                         log.Desire.none) == log.Desire.none


def test_blinker_precedence_and_speed_gate():
  injector = NavDesireInjector()
  cs = FakeCS()

  # blinker desire always wins
  assert injector.update(FakeNavSignal(math.radians(30.)), True, cs, True, 5.,
                         log.Desire.turnLeft) == log.Desire.turnLeft

  assert injector.update(FakeNavSignal(math.radians(30.)), True, cs, True, NAV_MAX_SPEED + 1.,
                         log.Desire.none) == log.Desire.none

  assert injector.update(FakeNavSignal(math.radians(30.), valid=False), True, cs, True, 5.,
                         log.Desire.none) == log.Desire.none
  assert injector.update(FakeNavSignal(math.radians(30.)), False, cs, True, 5., log.Desire.none) == log.Desire.none

  assert injector.update(FakeNavSignal(math.radians(30.)), True, cs, False, 5., log.Desire.none) == log.Desire.none


def test_reactivation_repulses():
  injector = NavDesireInjector()
  cs = FakeCS()
  # new activation is a fresh rising edge for the model's desire pulse
  assert injector.update(FakeNavSignal(math.radians(20.)), True, cs, True, 5., log.Desire.none) == log.Desire.turnRight
  assert injector.update(FakeNavSignal(0.), True, cs, True, 5., log.Desire.none) == log.Desire.none
  assert injector.update(FakeNavSignal(math.radians(20.)), True, cs, True, 5., log.Desire.none) == log.Desire.turnRight
