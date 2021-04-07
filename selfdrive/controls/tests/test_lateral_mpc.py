import unittest
import numpy as np
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_N, CAR_ROTATION_RADIUS


def run_mpc(v_ref=30., x_init=0., y_init=0., psi_init=0., curvature_init=0.,
            lane_width=3.6, poly_shift=0.):

  libmpc = libmpc_py.libmpc
  libmpc.init()
  libmpc.set_weights(1., 1., 1.)


  mpc_solution = libmpc_py.ffi.new("log_t *")

  y_pts = poly_shift * np.ones(MPC_N + 1)
  heading_pts = np.zeros(MPC_N + 1)


  cur_state = libmpc_py.ffi.new("state_t *")
  cur_state.x = x_init
  cur_state.y = y_init
  cur_state.psi = psi_init
  cur_state.curvature = curvature_init

  # converge in no more than 20 iterations
  for _ in range(20):
    libmpc.run_mpc(cur_state, mpc_solution, v_ref,
                   CAR_ROTATION_RADIUS,
                   list(y_pts), list(heading_pts))

  return mpc_solution


class TestLateralMpc(unittest.TestCase):

  def _assert_null(self, sol, curvature=1e-6):
    for i in range(len(sol[0].y)):
      self.assertAlmostEqual(sol[0].y[i], 0., delta=curvature)
      self.assertAlmostEqual(sol[0].psi[i], 0., delta=curvature)
      self.assertAlmostEqual(sol[0].curvature[i], 0., delta=curvature)

  def _assert_simmetry(self, sol, curvature=1e-6):
    for i in range(len(sol[0][0].y)):
      self.assertAlmostEqual(sol[0][0].y[i], -sol[1][0].y[i], delta=curvature)
      self.assertAlmostEqual(sol[0][0].psi[i], -sol[1][0].psi[i], delta=curvature)
      self.assertAlmostEqual(sol[0][0].curvature[i], -sol[1][0].curvature[i], delta=curvature)
      self.assertAlmostEqual(sol[0][0].x[i], sol[1][0].x[i], delta=curvature)

  def _assert_identity(self, sol, ignore_y=False, curvature=1e-6):
    for i in range(len(sol[0][0].y)):
      self.assertAlmostEqual(sol[0][0].psi[i], sol[1][0].psi[i], delta=curvature)
      self.assertAlmostEqual(sol[0][0].curvature[i], sol[1][0].curvature[i], delta=curvature)
      self.assertAlmostEqual(sol[0][0].x[i], sol[1][0].x[i], delta=curvature)
      if not ignore_y:
        self.assertAlmostEqual(sol[0][0].y[i], sol[1][0].y[i], delta=curvature)

  def test_straight(self):
    sol = run_mpc()
    self._assert_null(sol)

  def test_y_symmetry(self):
    sol = []
    for y_init in [-0.5, 0.5]:
      sol.append(run_mpc(y_init=y_init))
    self._assert_simmetry(sol)

  def test_poly_symmetry(self):
    sol = []
    for poly_shift in [-1., 1.]:
      sol.append(run_mpc(poly_shift=poly_shift))
    self._assert_simmetry(sol)

  def test_curvature_symmetry(self):
    sol = []
    for curvature_init in [-0.1, 0.1]:
      sol.append(run_mpc(curvature_init=curvature_init))
    self._assert_simmetry(sol)

  def test_psi_symmetry(self):
    sol = []
    for psi_init in [-0.1, 0.1]:
      sol.append(run_mpc(psi_init=psi_init))
    self._assert_simmetry(sol)

  def test_y_shift_vs_poly_shift(self):
    shift = 1.
    sol = []
    sol.append(run_mpc(y_init=shift))
    sol.append(run_mpc(poly_shift=-shift))
    # need larger curvature than standard, otherwise it false triggers.
    # this is acceptable because the 2 cases are very different from the optimizer standpoint
    self._assert_identity(sol, ignore_y=True, curvature=1e-5)

  def test_no_overshoot(self):
    y_init = 1.
    sol = run_mpc(y_init=y_init)
    for y in list(sol[0].y):
      self.assertGreaterEqual(y_init, abs(y))


if __name__ == "__main__":
  unittest.main()
