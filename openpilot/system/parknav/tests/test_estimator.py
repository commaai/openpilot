import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import log, messaging
from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.system.parknav import parknavd as nav


@pytest.fixture
def inputs(monkeypatch):
  clock = [100.]
  monkeypatch.setattr(nav.time, 'monotonic', lambda: clock[0])
  dm = messaging.new_message('deviceMotion').deviceMotion
  dm.orientationNED.valid = True
  dm.orientationNED.zStd = 0.01
  dm.velocityDevice.x = 2.
  gps = messaging.new_message('gpsLocation').gpsLocation
  gps.hasFix = True
  gps.latitude = 37.
  gps.longitude = -122.
  gps.speed = 2.
  gps.bearingDeg = 90.
  gps.bearingAccuracyDeg = 1.

  class SM(dict):
    updated = {'gpsLocation': True}
    valid = {'gpsLocation': True}

  sm = SM(deviceMotion=dm, gpsLocation=gps)
  calibrator = PoseCalibrator()
  calibrator.calib_valid = True
  return SimpleNamespace(clock=clock, sm=sm, calibrator=calibrator, state=nav.ParkNavEstimator())


def test_heading_requires_gnss_anchor(inputs):
  i = inputs
  pose = i.calibrator.build_calibrated_pose(Pose.from_device_motion(i.sm['deviceMotion']))
  assert math.isnan(pose.orientation.yaw_std)
  nav.update_pose(i.state, i.sm, i.calibrator)
  assert i.state.yaw_valid
  assert i.state.yaw_initialized
  i.state.last_fix_t = i.clock[0]
  i.state.pos_ned = np.array([20., 0.])
  assert not nav.make_signal(i.state, (37., -122.)).parkNavSignal.valid
  nav.update_gps(i.state, i.sm, None)
  assert i.state.heading_anchored
  assert i.state.yaw == pytest.approx(math.pi / 2)
  msg = nav.make_signal(i.state, (37., -122.))
  assert msg.parkNavSignal.valid
  with log.Event.from_bytes(msg.to_bytes()) as decoded:
    assert decoded.which() == 'parkNavSignal'
    assert decoded.parkNavSignal.valid


def test_course_is_applied_once_and_pose_tracks_turn(inputs):
  i = inputs
  nav.update_pose(i.state, i.sm, i.calibrator)
  nav.update_gps(i.state, i.sm, None)
  i.sm.updated['gpsLocation'] = False
  for step in range(1, 21):
    i.clock[0] += 0.05
    i.sm['deviceMotion'].orientationNED.z = math.radians(45 * step / 20)
    nav.update_pose(i.state, i.sm, i.calibrator)
    nav.update_gps(i.state, i.sm, None)
  assert i.state.yaw == pytest.approx(math.radians(135))
  i.sm.updated['gpsLocation'] = True
  i.sm['gpsLocation'].bearingDeg = 140.
  nav.update_gps(i.state, i.sm, None)
  assert i.state.yaw == pytest.approx(math.radians(136))


def test_integration_uses_absolute_yaw(inputs):
  i = inputs
  nav.update_pose(i.state, i.sm, i.calibrator)
  assert i.state.pos_ned == pytest.approx([0., 0.])
  nav.update_gps(i.state, i.sm, None)
  i.clock[0] += 0.1
  nav.update_pose(i.state, i.sm, i.calibrator)
  assert i.state.pos_ned == pytest.approx([0., 0.2], abs=1e-8)


@pytest.mark.parametrize('accuracy', [5., 20., -1., float('nan')])
def test_reject_uncertain_course(inputs, accuracy):
  i = inputs
  nav.update_pose(i.state, i.sm, i.calibrator)
  i.sm['gpsLocation'].bearingAccuracyDeg = accuracy
  nav.update_gps(i.state, i.sm, None)
  assert not i.state.heading_anchored


@pytest.mark.parametrize('distance', [0., 2., nav.ARRIVAL_RADIUS])
def test_arrival_disables_turns(inputs, distance):
  i = inputs
  i.state.yaw_valid = i.state.heading_anchored = True
  i.state.last_fix_t = i.clock[0]
  i.state.pos_ned = np.array([distance, 0.])
  signal = nav.make_signal(i.state, (37., -122.)).parkNavSignal
  assert not signal.valid
  assert signal.relBearing == 0.
  assert signal.lateralOffset == 0.


def test_fix_expiry_and_calibration_loss(inputs):
  i = inputs
  nav.update_pose(i.state, i.sm, i.calibrator)
  nav.update_gps(i.state, i.sm, (37.001, -122.))
  assert nav.make_signal(i.state, (37.001, -122.)).parkNavSignal.valid
  i.clock[0] += nav.MAX_FIX_AGE
  assert not nav.make_signal(i.state, (37.001, -122.)).parkNavSignal.valid
  i.calibrator.calib_valid = False
  nav.update_pose(i.state, i.sm, i.calibrator)
  assert not i.state.yaw_valid


@pytest.mark.parametrize('std', [float('nan'), float('inf'), -0.1, nav.MAX_HEADING_STD])
def test_reject_invalid_pose_uncertainty(inputs, std):
  i = inputs
  i.sm['deviceMotion'].orientationNED.zStd = std
  nav.update_pose(i.state, i.sm, i.calibrator)
  nav.update_gps(i.state, i.sm, None)
  assert not i.state.yaw_valid
  assert not i.state.heading_anchored


@pytest.mark.parametrize('field,value', [('hasFix', False), ('speed', 0.), ('bearingDeg', float('nan'))])
def test_reject_unusable_course(inputs, field, value):
  i = inputs
  nav.update_pose(i.state, i.sm, i.calibrator)
  setattr(i.sm['gpsLocation'], field, value)
  nav.update_gps(i.state, i.sm, None)
  assert not i.state.heading_anchored
