from openpilot.selfdrive.monitoring.helpers import face_orientation_from_net


def test_face_orientation_from_net_with_valid_input():
  """Test that face_orientation_from_net works with valid input"""
  angles_desc = [0.1, 0.2, 0.3]  # pitch, yaw, roll
  pos_desc = [0.0, 0.0]
  rpy_calib = [0.0, 0.01, 0.02]  # roll, pitch, yaw calibration

  roll, pitch, yaw = face_orientation_from_net(angles_desc, pos_desc, rpy_calib)


def test_face_orientation_from_net_short_angles():
  """Test that face_orientation_from_net handles short angles_desc safely"""
  angles_desc = [0.1]  # Only one element, should cause unpacking error in the original code
  pos_desc = [0.0, 0.0]
  rpy_calib = [0.0, 0.01, 0.02]

  roll, pitch, yaw = face_orientation_from_net(angles_desc, pos_desc, rpy_calib)

  # Should return default values when angles_desc is too short
  assert (roll, pitch, yaw) == (0.0, 0.0, 0.0)


def test_face_orientation_from_net_short_rpy_calib():
  """Test that face_orientation_from_net handles short rpy_calib safely"""
  angles_desc = [0.1, 0.2, 0.3]  # pitch, yaw, roll
  pos_desc = [0.0, 0.0]
  rpy_calib = [0.0]  # Only one element, should cause indexing error in the original code

  # This should handle short rpy_calib by using 0.0 for missing values
  roll, pitch, yaw = face_orientation_from_net(angles_desc, pos_desc, rpy_calib)
