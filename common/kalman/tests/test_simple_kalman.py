import unittest
import random
import timeit
import numpy as np

from openpilot.common.kalman.simple_kalman import KF1D
from openpilot.common.kalman.simple_kalman_old import KF1D as KF1D_old


class TestSimpleKalman(unittest.TestCase):
  def setUp(self):
    dt = 0.01
    x0_0 = 0.0
    x1_0 = 0.0
    A0_0 = 1.0
    A0_1 = dt
    A1_0 = 0.0
    A1_1 = 1.0
    C0_0 = 1.0
    C0_1 = 0.0
    K0_0 = 0.12287673
    K1_0 = 0.29666309

    self.kf_old = KF1D_old(x0=np.array([[x0_0], [x1_0]]),
                           A=np.array([[A0_0, A0_1], [A1_0, A1_1]]),
                           C=np.array([C0_0, C0_1]),
                           K=np.array([[K0_0], [K1_0]]))

    self.kf = KF1D(x0=[[x0_0], [x1_0]],
                   A=[[A0_0, A0_1], [A1_0, A1_1]],
                   C=[C0_0, C0_1],
                   K=[[K0_0], [K1_0]])

  def test_getter_setter(self):
    self.kf.x = [[1.0], [1.0]]
    self.assertEqual(self.kf.x, [[1.0], [1.0]])

  def update_returns_state(self):
    x = self.kf.update(100)
    self.assertEqual(x, self.kf.x)

  def test_old_equal_new(self):
    for _ in range(1000):
      v_wheel = random.uniform(0, 200)

      x_old = self.kf_old.update(v_wheel)
      x = self.kf.update(v_wheel)

      # Compare the output x, verify that the error is less than 1e-4
      np.testing.assert_almost_equal(x_old[0], x[0])
      np.testing.assert_almost_equal(x_old[1], x[1])

  def test_new_is_faster(self):
    setup = """
import numpy as np

from openpilot.common.kalman.simple_kalman import KF1D
from openpilot.common.kalman.simple_kalman_old import KF1D as KF1D_old

dt = 0.01
x0_0 = 0.0
x1_0 = 0.0
A0_0 = 1.0
A0_1 = dt
A1_0 = 0.0
A1_1 = 1.0
C0_0 = 1.0
C0_1 = 0.0
K0_0 = 0.12287673
K1_0 = 0.29666309

kf_old = KF1D_old(x0=np.array([[x0_0], [x1_0]]),
                  A=np.array([[A0_0, A0_1], [A1_0, A1_1]]),
                  C=np.array([C0_0, C0_1]),
                  K=np.array([[K0_0], [K1_0]]))

kf = KF1D(x0=[[x0_0], [x1_0]],
          A=[[A0_0, A0_1], [A1_0, A1_1]],
          C=[C0_0, C0_1],
          K=[[K0_0], [K1_0]])
    """
    kf_speed = timeit.timeit("kf.update(1234)", setup=setup, number=10000)
    kf_old_speed = timeit.timeit("kf_old.update(1234)", setup=setup, number=10000)
    self.assertTrue(kf_speed < kf_old_speed / 4)

if __name__ == "__main__":
  unittest.main()
