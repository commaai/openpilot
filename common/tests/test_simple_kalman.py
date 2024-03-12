import unittest

from openpilot.common.simple_kalman import KF1D


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

    self.kf = KF1D(x0=[[x0_0], [x1_0]],
                   A=[[A0_0, A0_1], [A1_0, A1_1]],
                   C=[C0_0, C0_1],
                   K=[[K0_0], [K1_0]])

  def test_getter_setter(self):
    self.kf.set_x([[1.0], [1.0]])
    self.assertEqual(self.kf.x, [[1.0], [1.0]])

  def update_returns_state(self):
    x = self.kf.update(100)
    self.assertEqual(x, self.kf.x)


if __name__ == "__main__":
  unittest.main()
