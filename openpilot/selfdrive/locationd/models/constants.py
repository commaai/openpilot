import os

GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'generated'))

class ObservationKind:
  PHONE_GYRO = 4
  PHONE_ACCEL = 10
  CAMERA_ODO_TRANSLATION = 13
  CAMERA_ODO_ROTATION = 14

  ROAD_FRAME_XY_SPEED = 24  # (x, y) [m/s]
  ROAD_FRAME_YAW_RATE = 25  # [rad/s]
  STEER_ANGLE = 26  # [rad]
  ANGLE_OFFSET_FAST = 27  # [rad]
  STIFFNESS = 28  # [-]
  STEER_RATIO = 29  # [-]
  ROAD_FRAME_X_SPEED = 30  # (x) [m/s]
  ROAD_ROLL = 31  # [rad]
