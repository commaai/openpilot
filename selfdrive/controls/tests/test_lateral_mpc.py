import pytest
import numpy as np
from openpilot.selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc
from openpilot.selfdrive.controls.lib.drive_helpers import CAR_ROTATION_RADIUS
from openpilot.selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import N as LAT_MPC_N


def run_mpc(lat_mpc=None, v_ref=30., x_init=0., y_init=0., psi_init=0., curvature_init=0.,
            lane_width=3.6, poly_shift=0.):

  if lat_mpc is None:
    lat_mpc = LateralMpc()
  lat_mpc.set_weights(1., .1, 0.0, .05, 800)

  y_pts = poly_shift * np.ones(LAT_MPC_N + 1)
  heading_pts = np.zeros(LAT_MPC_N + 1)
  curv_rate_pts = np.zeros(LAT_MPC_N + 1)

  x0 = np.array([x_init, y_init, psi_init, curvature_init])
  p = np.column_stack([v_ref * np.ones(LAT_MPC_N + 1),
                      CAR_ROTATION_RADIUS * np.ones(LAT_MPC_N + 1)])

  # converge in no more than 10 iterations
  for _ in range(10):
    lat_mpc.run(x0, p,
                y_pts, heading_pts, curv_rate_pts)
  return lat_mpc.x_sol


class TestLateralMpc:

  def _assert_null(self, sol, curvature=1e-6):
    for i in range(len(sol)):
      assert sol[0,i,1] == pytest.approx(0, abs=curvature)
      assert sol[0,i,2] == pytest.approx(0, abs=curvature)
      assert sol[0,i,3] == pytest.approx(0, abs=curvature)

  def _assert_simmetry(self, sol, curvature=1e-6):
    for i in range(len(sol)):
      assert sol[0,i,1] == pytest.approx(-sol[1,i,1], abs=curvature)
      assert sol[0,i,2] == pytest.approx(-sol[1,i,2], abs=curvature)
      assert sol[0,i,3] == pytest.approx(-sol[1,i,3], abs=curvature)
      assert sol[0,i,0] == pytest.approx(sol[1,i,0], abs=curvature)

  def test_straight(self):
    sol = run_mpc()
    self._assert_null(np.array([sol]))

  def test_y_symmetry(self):
    sol = []
    for y_init in [-0.5, 0.5]:
      sol.append(run_mpc(y_init=y_init))
    self._assert_simmetry(np.array(sol))

  def test_poly_symmetry(self):
    sol = []
    for poly_shift in [-1., 1.]:
      sol.append(run_mpc(poly_shift=poly_shift))
    self._assert_simmetry(np.array(sol))

  def test_curvature_symmetry(self):
    sol = []
    for curvature_init in [-0.1, 0.1]:
      sol.append(run_mpc(curvature_init=curvature_init))
    self._assert_simmetry(np.array(sol))

  def test_psi_symmetry(self):
    sol = []
    for psi_init in [-0.1, 0.1]:
      sol.append(run_mpc(psi_init=psi_init))
    self._assert_simmetry(np.array(sol))

  def test_no_overshoot(self):
    y_init = 1.
    sol = run_mpc(y_init=y_init)
    for y in list(sol[:,1]):
      assert y_init >= abs(y)

  def test_switch_convergence(self):
    lat_mpc = LateralMpc()
    sol = run_mpc(lat_mpc=lat_mpc, poly_shift=3.0, v_ref=7.0)
    right_psi_deg = np.degrees(sol[:,2])
    sol = run_mpc(lat_mpc=lat_mpc, poly_shift=-3.0, v_ref=7.0)
    left_psi_deg = np.degrees(sol[:,2])
    np.testing.assert_almost_equal(right_psi_deg, -left_psi_deg, decimal=3)
