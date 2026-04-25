"""
``fill_pose_msg`` validity and odometry field wiring.

LOW-LEVEL §7.1 Phase B; does not modify ``test_fill_model_msg.py``.

Maps: R1.
"""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.modeld.fill_model_msg import fill_pose_msg
from openpilot.selfdrive.modeld.tests.modeld_test_fixtures import DummyBuilder, minimal_net_output_data


def test_fill_pose_msg_valid_when_calib_seen_and_no_drops():
  msg = DummyBuilder()
  outs = minimal_net_output_data(batch=1)

  fill_pose_msg(msg, outs, vipc_frame_id=9, vipc_dropped_frames=0, timestamp_eof=555, live_calib_seen=True)

  assert msg.valid is True
  assert msg.cameraOdometry.frameId == 9
  assert msg.cameraOdometry.timestampEof == 555


def test_fill_pose_msg_invalid_when_dropped_frames():
  msg = DummyBuilder()
  outs = minimal_net_output_data(batch=1)

  fill_pose_msg(msg, outs, vipc_frame_id=9, vipc_dropped_frames=1, timestamp_eof=555, live_calib_seen=True)

  assert msg.valid is False


def test_fill_pose_msg_invalid_without_live_calibration():
  msg = DummyBuilder()
  outs = minimal_net_output_data(batch=1)

  fill_pose_msg(msg, outs, vipc_frame_id=9, vipc_dropped_frames=0, timestamp_eof=555, live_calib_seen=False)

  assert msg.valid is False


def test_fill_pose_msg_pose_lists_match_constants_widths():
  msg = DummyBuilder()
  outs = minimal_net_output_data(batch=1)

  fill_pose_msg(msg, outs, vipc_frame_id=1, vipc_dropped_frames=0, timestamp_eof=1, live_calib_seen=True)

  odo = msg.cameraOdometry
  assert len(odo.trans) == 3
  assert len(odo.rot) == 3
  assert len(odo.wideFromDeviceEuler) == 3
  assert len(odo.roadTransformTrans) == 3


def test_fill_pose_msg_lists_match_net_output_tensors():
  """Every odometry list field should mirror batch-0 slices from ``net_output_data``."""
  msg = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  fill_pose_msg(msg, outs, vipc_frame_id=42, vipc_dropped_frames=0, timestamp_eof=9_000_001, live_calib_seen=True)

  odo = msg.cameraOdometry
  np.testing.assert_allclose(odo.trans, outs["pose"][0, :3].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(odo.rot, outs["pose"][0, 3:].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(odo.transStd, outs["pose_stds"][0, :3].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(odo.rotStd, outs["pose_stds"][0, 3:].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(odo.wideFromDeviceEuler, outs["wide_from_device_euler"][0, :].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(
    odo.wideFromDeviceEulerStd,
    outs["wide_from_device_euler_stds"][0, :].tolist(),
    rtol=0,
    atol=0,
  )
  np.testing.assert_allclose(odo.roadTransformTrans, outs["road_transform"][0, :3].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(
    odo.roadTransformTransStd,
    outs["road_transform_stds"][0, :3].tolist(),
    rtol=0,
    atol=0,
  )
  assert odo.frameId == 42
  assert odo.timestampEof == 9_000_001
