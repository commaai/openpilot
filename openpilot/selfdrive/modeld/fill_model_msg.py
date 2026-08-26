import ctypes
import sys
import weakref
from pathlib import Path

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants


_OUTPUT_NAMES = (
  'plan', 'plan_stds',
  'lane_lines', 'lane_lines_stds', 'lane_lines_prob',
  'road_edges', 'road_edges_stds',
  'lead', 'lead_stds', 'lead_prob',
  'desire_state', 'desire_pred', 'meta',
  'pose', 'pose_stds',
  'wide_from_device_euler', 'wide_from_device_euler_stds',
  'road_transform', 'road_transform_stds',
)


class ModelOutputs(ctypes.Structure):
  _fields_ = [(name, ctypes.c_void_p) for name in _OUTPUT_NAMES] + [
    ('raw_pred', ctypes.c_void_p),
    ('raw_pred_size', ctypes.c_size_t),
  ]


class PublishData(ctypes.Structure):
  _fields_ = [
    ('timestamp_eof', ctypes.c_uint64),
    ('vipc_frame_id', ctypes.c_uint32),
    ('vipc_frame_id_extra', ctypes.c_uint32),
    ('frame_id', ctypes.c_uint32),
    ('frame_drop_perc', ctypes.c_float),
    ('model_execution_time', ctypes.c_float),
    ('desired_curvature', ctypes.c_float),
    ('desired_acceleration', ctypes.c_float),
    ('valid', ctypes.c_uint8),
    ('camera_odometry_valid', ctypes.c_uint8),
    ('big', ctypes.c_uint8),
    ('should_stop', ctypes.c_uint8),
    ('lane_change_state', ctypes.c_uint8),
    ('lane_change_direction', ctypes.c_uint8),
  ]


_suffix = '.dylib' if sys.platform == 'darwin' else '.so'
_lib = ctypes.CDLL(Path(__file__).with_name(f'libmodel_publisher{_suffix}'))
_lib.model_publisher_create.argtypes = []
_lib.model_publisher_create.restype = ctypes.c_void_p
_lib.model_publisher_destroy.argtypes = [ctypes.c_void_p]
_lib.model_publisher_destroy.restype = None
_lib.model_publisher_publish.argtypes = [ctypes.c_void_p, ctypes.POINTER(ModelOutputs), ctypes.POINTER(PublishData), ctypes.POINTER(ctypes.c_double)]
_lib.model_publisher_publish.restype = ctypes.c_bool
_lib.model_publisher_last_error.argtypes = []
_lib.model_publisher_last_error.restype = ctypes.c_char_p

_PATH_FIT = np.polynomial.polynomial.polyfit(ModelConstants.T_IDXS, np.eye(ModelConstants.IDX_N), ModelConstants.POLY_PATH_DEGREE)


class ModelPublisher:
  def __init__(self):
    self.publisher = _lib.model_publisher_create()
    if not self.publisher:
      raise RuntimeError(_lib.model_publisher_last_error().decode())
    self._finalizer = weakref.finalize(self, _lib.model_publisher_destroy, self.publisher)
    self._finalizer.atexit = False

  def publish(self, outputs, desired_curvature, desired_acceleration, should_stop,
              vipc_frame_id, vipc_frame_id_extra, frame_id, frame_drop,
              timestamp_eof, model_execution_time, valid, big, vipc_dropped_frames,
              lane_change_state, lane_change_direction):
    raw_pred = outputs.get('raw_pred')
    pointers = ModelOutputs(
      *(outputs[name].ctypes.data for name in _OUTPUT_NAMES),
      raw_pred.ctypes.data if raw_pred is not None else None,
      raw_pred.nbytes if raw_pred is not None else 0,
    )
    data = PublishData(
      timestamp_eof, vipc_frame_id, vipc_frame_id_extra, frame_id,
      frame_drop * 100, model_execution_time, desired_curvature, desired_acceleration,
      valid, valid and vipc_dropped_frames < 1, big, should_stop,
      lane_change_state, lane_change_direction,
    )
    path_coefficients = _PATH_FIT @ outputs['plan'][0, :, :3]
    if not _lib.model_publisher_publish(self.publisher, ctypes.byref(pointers), ctypes.byref(data),
                                        path_coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_double))):
      raise RuntimeError(_lib.model_publisher_last_error().decode())


def fill_xyz_poly(builder, degree, x, y, z):
  xyz = np.stack([x, y, z], axis=1)
  coeffs = np.polynomial.polynomial.polyfit(ModelConstants.T_IDXS, xyz, deg=degree)
  builder.xCoefficients = coeffs[:, 0].tolist()
  builder.yCoefficients = coeffs[:, 1].tolist()
  builder.zCoefficients = coeffs[:, 2].tolist()


def fill_lane_line_meta(builder, lane_lines, lane_line_probs):
  builder.leftY = lane_lines[1].y[0]
  builder.leftProb = lane_line_probs[1]
  builder.rightY = lane_lines[2].y[0]
  builder.rightProb = lane_line_probs[2]
