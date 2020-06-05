#!/usr/bin/env python3

import numpy as np
import unittest

from common.transformations.transformations import euler2quat_single, quat2euler_single  # pylint: disable=no-name-in-module


class TestOrientationCython(unittest.TestCase):
  def test_euler2quat_quat2euler(self):
    euler = [0.1, 0.2, 0.3]
    quat = euler2quat_single(euler)
    euler_new = quat2euler_single(quat)
    np.testing.assert_allclose(euler, euler_new)


if __name__ == "__main__":
  unittest.main()
