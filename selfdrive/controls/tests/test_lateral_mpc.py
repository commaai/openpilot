import unittest
import numpy as np
from selfdrive.car.honda.interface import CarInterface
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.lane_planner import calc_d_poly


def run_mpc(v_ref=30., x_init=0., y_init=0., psi_init=0., delta_init=0.,
            l_prob=1., r_prob=1., p_prob=1.,
            poly_l=np.array([0., 0., 0., 1.8]), poly_r=np.array([0., 0., 0., -1.8]), poly_p=np.array([0., 0., 0., 0.]),
            lane_width=3.6, poly_shift=0.):

  libmpc = libmpc_py.libmpc
  libmpc.init(1.0, 3.0, 1.0, 1.0)

  mpc_solution = libmpc_py.ffi.new("log_t *")

  p_l = poly_l.copy()
  p_l[3] += poly_shift

  p_r = poly_r.copy()
  p_r[3] += poly_shift

  p_p = poly_p.copy()
  p_p[3] += poly_shift

  d_poly = calc_d_poly(p_l, p_r, p_p, l_prob, r_prob, lane_width)

  CP = CarInterface.get_params("HONDA CIVIC 2016 TOURING", {})
  VM = VehicleModel(CP)

  v_ref = v_ref
  curvature_factor = VM.curvature_factor(v_ref)

  l_poly = libmpc_py.ffi.new("double[4]", map(float, p_l))
  r_poly = libmpc_py.ffi.new("double[4]", map(float, p_r))
  d_poly = libmpc_py.ffi.new("double[4]", map(float, d_poly))

  cur_state = libmpc_py.ffi.new("state_t *")
  cur_state[0].x = x_init
  cur_state[0].y = y_init
  cur_state[0].psi = psi_init
  cur_state[0].delta = delta_init

  # converge in no more than 20 iterations
  for _ in range(20):
    libmpc.run_mpc(cur_state, mpc_solution, l_poly, r_poly, d_poly, l_prob, r_prob,
                   curvature_factor, v_ref, lane_width)

  return mpc_solution


class TestLateralMpc(unittest.TestCase):

  def _assert_null(self, sol, delta=1e-6):
    for i in range(len(sol[0].y)):
      self.assertAlmostEqual(sol[0].y[i], 0., delta=delta)
      self.assertAlmostEqual(sol[0].psi[i], 0., delta=delta)
      self.assertAlmostEqual(sol[0].delta[i], 0., delta=delta)

  def _assert_simmetry(self, sol, delta=1e-6):
    for i in range(len(sol[0][0].y)):
      self.assertAlmostEqual(sol[0][0].y[i], -sol[1][0].y[i], delta=delta)
      self.assertAlmostEqual(sol[0][0].psi[i], -sol[1][0].psi[i], delta=delta)
      self.assertAlmostEqual(sol[0][0].delta[i], -sol[1][0].delta[i], delta=delta)
      self.assertAlmostEqual(sol[0][0].x[i], sol[1][0].x[i], delta=delta)

  def _assert_identity(self, sol, ignore_y=False, delta=1e-6):
    for i in range(len(sol[0][0].y)):
      self.assertAlmostEqual(sol[0][0].psi[i], sol[1][0].psi[i], delta=delta)
      self.assertAlmostEqual(sol[0][0].delta[i], sol[1][0].delta[i], delta=delta)
      self.assertAlmostEqual(sol[0][0].x[i], sol[1][0].x[i], delta=delta)
      if not ignore_y:
        self.assertAlmostEqual(sol[0][0].y[i], sol[1][0].y[i], delta=delta)

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

  def test_delta_symmetry(self):
    sol = []
    for delta_init in [-0.1, 0.1]:
      sol.append(run_mpc(delta_init=delta_init))
    self._assert_simmetry(sol)

  def test_psi_symmetry(self):
    sol = []
    for psi_init in [-0.1, 0.1]:
      sol.append(run_mpc(psi_init=psi_init))
    self._assert_simmetry(sol)

  def test_prob_symmetry(self):
    sol = []
    lane_width = 3.
    for r_prob in [0., 1.]:
      sol.append(run_mpc(r_prob=r_prob, l_prob=1.-r_prob, lane_width=lane_width))
    self._assert_simmetry(sol)

  def test_y_shift_vs_poly_shift(self):
    shift = 1.
    sol = []
    sol.append(run_mpc(y_init=shift))
    sol.append(run_mpc(poly_shift=-shift))
    # need larger delta than standard, otherwise it false triggers.
    # this is acceptable because the 2 cases are very different from the optimizer standpoint
    self._assert_identity(sol, ignore_y=True, delta=1e-5)

  def test_no_overshoot(self):
    y_init = 1.
    sol = run_mpc(y_init=y_init)
    for y in list(sol[0].y):
      self.assertGreaterEqual(y_init, abs(y))


if __name__ == "__main__":
  unittest.main()
