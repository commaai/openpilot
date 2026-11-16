import numpy as np

from openpilot.cereal import messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.locationd.helpers import Measurement, Pose
from openpilot.selfdrive.selfdrived.helpers import ExcessiveActuationCheck, ExcessiveActuationType


def make_pose(accel_x: float = 0.0, yaw_rate: float = 0.0) -> Pose:
  zero = Measurement(np.zeros(3), np.zeros(3))
  return Pose(zero, zero, Measurement(np.array([accel_x, 0.0, 0.0]), np.zeros(3)), Measurement(np.array([0.0, 0.0, yaw_rate]), np.zeros(3)))


class TestExcessiveActuationCheck:
  def setup_method(self):
    self.check = ExcessiveActuationCheck()
    self.sm = {s: messaging.new_message(s) for s in ('carControl', 'controlsState', 'vehicleParameters')}
    self.sm = {s: getattr(m, s) for s, m in self.sm.items()}
    self.sm['vehicleParameters'].valid = True
    self.CS = messaging.new_message('carState').carState
    self.CS.vEgo = 20.0

  def run(self, seconds: float, accel: float = 0.0, lat_accel: float = 0.0, steer_lat_accel: float = 0.0,
          long_active: bool = True, lat_active: bool = True, gas_pressed: bool = False):
    self.sm['carControl'].longActive = long_active
    self.sm['carControl'].latActive = lat_active
    self.sm['controlsState'].curvature = steer_lat_accel / self.CS.vEgo ** 2
    self.CS.aEgo = accel
    self.CS.gasPressed = gas_pressed
    pose = make_pose(accel, lat_accel / self.CS.vEgo)
    results = [self.check.update(self.sm, self.CS, pose) for _ in range(round(seconds / DT_CTRL))]
    return next((r for r in results if r is not None), None)

  def test_no_actuation(self):
    assert self.run(5.0) is None

  def test_excessive_accel(self):
    assert self.run(1.0) is None
    assert self.run(0.5, accel=4.5) == ExcessiveActuationType.LONGITUDINAL

  def test_excessive_decel(self):
    assert self.run(1.0) is None
    assert self.run(0.5, accel=-7.5) == ExcessiveActuationType.LONGITUDINAL

  def test_short_accel_peak(self):
    assert self.run(1.0) is None
    assert self.run(0.2, accel=4.5) is None
    assert self.run(1.0) is None

  def test_gas_override_release(self):
    assert self.run(1.0) is None
    assert self.run(1.0, accel=7.0, long_active=False, gas_pressed=True) is None
    for accel in np.linspace(6.0, 3.5, 10):
      assert self.run(0.05, accel=float(accel)) is None
    assert self.run(0.5, accel=4.5) is None
    assert self.run(0.5, accel=4.5) == ExcessiveActuationType.LONGITUDINAL

  def test_excessive_lateral(self):
    assert self.run(1.5) is None
    assert self.run(0.5, lat_accel=7.0, steer_lat_accel=6.0) == ExcessiveActuationType.LATERAL
    self.setup_method()
    assert self.run(1.5) is None
    assert self.run(0.5, lat_accel=-7.0, steer_lat_accel=-6.0) == ExcessiveActuationType.LATERAL

  def test_lateral_steering_underestimates(self):
    assert self.run(1.5) is None
    assert self.run(0.5, lat_accel=7.0, steer_lat_accel=3.5) == ExcessiveActuationType.LATERAL

  def test_device_rotation(self):
    assert self.run(1.5) is None
    assert self.run(1.0, lat_accel=12.0, steer_lat_accel=0.5) is None
    assert self.run(1.0, lat_accel=12.0, steer_lat_accel=-4.0) is None

  def test_invalid_vehicle_parameters(self):
    self.sm['vehicleParameters'].valid = False
    assert self.run(1.5) is None
    assert self.run(1.0, lat_accel=7.0, steer_lat_accel=6.0) is None

  def test_lateral_after_engage(self):
    assert self.run(1.0, lat_active=False) is None
    assert self.run(0.5, lat_accel=7.0, steer_lat_accel=6.0) is None
