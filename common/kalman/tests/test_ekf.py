import numpy as np
import numpy.matlib
import unittest
import timeit

from common.kalman.ekf import EKF, SimpleSensor, FastEKF1D

class TestEKF(EKF):
  def __init__(self, var_init, Q):
    super(TestEKF, self).__init__(False)
    self.identity = numpy.matlib.identity(2)
    self.state = numpy.matlib.zeros((2, 1))
    self.covar = self.identity * var_init

    self.process_noise = numpy.matlib.diag(Q)

  def calc_transfer_fun(self, dt):
    tf = numpy.matlib.identity(2)
    tf[0, 1] = dt
    return tf, tf


class EKFTest(unittest.TestCase):
  def test_update_scalar(self):
    ekf = TestEKF(1e3, [0.1, 1])
    dt = 1. / 100

    sensor = SimpleSensor(0, 1, 2)
    readings = map(sensor.read, np.arange(100, 300))

    for reading in readings:
      ekf.update_scalar(reading)
      ekf.predict(dt)

    np.testing.assert_allclose(ekf.state, [[300], [100]], 1e-4)
    np.testing.assert_allclose(
      ekf.covar,
      np.asarray([[0.0563, 0.10278], [0.10278, 0.55779]]),
      atol=1e-4)

  def test_unbiased(self):
    ekf = TestEKF(1e3, [0., 0.])
    dt = np.float64(1. / 100)

    sensor = SimpleSensor(0, 1, 2)
    readings = map(sensor.read, np.arange(1000))

    for reading in readings:
      ekf.update_scalar(reading)
      ekf.predict(dt)

    np.testing.assert_allclose(ekf.state, [[1000.], [100.]], 1e-4)


class FastEKF1DTest(unittest.TestCase):
  def test_correctness(self):
    dt = 1. / 100
    reading = SimpleSensor(0, 1, 2).read(100)

    ekf = TestEKF(1e3, [0.1, 1])
    fast_ekf = FastEKF1D(dt, 1e3, [0.1, 1])

    ekf.update_scalar(reading)
    fast_ekf.update_scalar(reading)
    self.assertAlmostEqual(ekf.state[0]   , fast_ekf.state[0])
    self.assertAlmostEqual(ekf.state[1]   , fast_ekf.state[1])
    self.assertAlmostEqual(ekf.covar[0, 0], fast_ekf.covar[0])
    self.assertAlmostEqual(ekf.covar[0, 1], fast_ekf.covar[2])
    self.assertAlmostEqual(ekf.covar[1, 1], fast_ekf.covar[1])

    ekf.predict(dt)
    fast_ekf.predict(dt)
    self.assertAlmostEqual(ekf.state[0]   , fast_ekf.state[0])
    self.assertAlmostEqual(ekf.state[1]   , fast_ekf.state[1])
    self.assertAlmostEqual(ekf.covar[0, 0], fast_ekf.covar[0])
    self.assertAlmostEqual(ekf.covar[0, 1], fast_ekf.covar[2])
    self.assertAlmostEqual(ekf.covar[1, 1], fast_ekf.covar[1])

  def test_speed(self):
    setup = """
import numpy as np
from common.kalman.tests.test_ekf import TestEKF
from common.kalman.ekf import SimpleSensor, FastEKF1D

dt = 1. / 100
reading = SimpleSensor(0, 1, 2).read(100)

var_init, Q = 1e3, [0.1, 1]
ekf = TestEKF(var_init, Q)
fast_ekf = FastEKF1D(dt, var_init, Q)
    """

    timeit.timeit("""
ekf.update_scalar(reading)
ekf.predict(dt)
    """, setup=setup, number=1000)

    ekf_speed = timeit.timeit("""
ekf.update_scalar(reading)
ekf.predict(dt)
    """, setup=setup, number=20000)

    timeit.timeit("""
fast_ekf.update_scalar(reading)
fast_ekf.predict(dt)
    """, setup=setup, number=1000)

    fast_ekf_speed = timeit.timeit("""
fast_ekf.update_scalar(reading)
fast_ekf.predict(dt)
    """, setup=setup, number=20000)

    assert fast_ekf_speed < ekf_speed / 4

if __name__ == "__main__":
  unittest.main()
