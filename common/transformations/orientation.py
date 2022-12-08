# pylint: skip-file
import numpy as np
from typing import Callable

from common.transformations.transformations import (ecef_euler_from_ned_single,
                                                    euler2quat_single,
                                                    euler2rot_single,
                                                    ned_euler_from_ecef_single,
                                                    quat2euler_single,
                                                    quat2rot_single,
                                                    rot2euler_single,
                                                    rot2quat_single)


def numpy_wrap(function, input_shape, output_shape) -> Callable[..., np.ndarray]:
  """Wrap a function to take either an input or list of inputs and return the correct shape"""
  def f(*inps):
    *args, inp = inps
    inp = np.array(inp)
    shape = inp.shape

    if len(shape) == len(input_shape):
      out_shape = output_shape
    else:
      out_shape = (shape[0],) + output_shape

    # Add empty dimension if inputs is not a list
    if len(shape) == len(input_shape):
      inp.shape = (1, ) + inp.shape

    result = np.asarray([function(*args, i) for i in inp])
    result.shape = out_shape
    return result
  return f


euler2quat = numpy_wrap(euler2quat_single, (3,), (4,))
quat2euler = numpy_wrap(quat2euler_single, (4,), (3,))
quat2rot = numpy_wrap(quat2rot_single, (4,), (3, 3))
rot2quat = numpy_wrap(rot2quat_single, (3, 3), (4,))
euler2rot = numpy_wrap(euler2rot_single, (3,), (3, 3))
rot2euler = numpy_wrap(rot2euler_single, (3, 3), (3,))
ecef_euler_from_ned = numpy_wrap(ecef_euler_from_ned_single, (3,), (3,))
ned_euler_from_ecef = numpy_wrap(ned_euler_from_ecef_single, (3,), (3,))

quats_from_rotations = rot2quat
quat_from_rot = rot2quat
rotations_from_quats = quat2rot
rot_from_quat = quat2rot
euler_from_rot = rot2euler
euler_from_quat = quat2euler
rot_from_euler = euler2rot
quat_from_euler = euler2quat
