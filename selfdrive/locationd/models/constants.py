import os

GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'generated'))

class ObservationKind:
  UNKNOWN = 0
  NO_OBSERVATION = 1
  GPS_NED = 2
  ODOMETRIC_SPEED = 3
  PHONE_GYRO = 4
  GPS_VEL = 5
  PSEUDORANGE_GPS = 6
  PSEUDORANGE_RATE_GPS = 7
  SPEED = 8
  NO_ROT = 9
  PHONE_ACCEL = 10
  ORB_POINT = 11
  ECEF_POS = 12
  CAMERA_ODO_TRANSLATION = 13
  CAMERA_ODO_ROTATION = 14
  ORB_FEATURES = 15
  MSCKF_TEST = 16
  FEATURE_TRACK_TEST = 17
  LANE_PT = 18
  IMU_FRAME = 19
  PSEUDORANGE_GLONASS = 20
  PSEUDORANGE_RATE_GLONASS = 21
  PSEUDORANGE = 22
  PSEUDORANGE_RATE = 23
  ECEF_VEL = 31
  ECEF_ORIENTATION_FROM_GPS = 32
  EARTH_RADIUS_WHEN_NO_GPS = 33

  ROAD_FRAME_XY_SPEED = 24  # (x, y) [m/s]
  ROAD_FRAME_YAW_RATE = 25  # [rad/s]
  STEER_ANGLE = 26  # [rad]
  ANGLE_OFFSET_FAST = 27  # [rad]
  STIFFNESS = 28  # [-]
  STEER_RATIO = 29  # [-]
  ROAD_FRAME_X_SPEED = 30  # (x) [m/s]

  names = [
    'Unknown',
    'No observation',
    'GPS NED',
    'Odometric speed',
    'Phone gyro',
    'GPS velocity',
    'GPS pseudorange',
    'GPS pseudorange rate',
    'Speed',
    'No rotation',
    'Phone acceleration',
    'ORB point',
    'ECEF pos',
    'camera odometric translation',
    'camera odometric rotation',
    'ORB features',
    'MSCKF test',
    'Feature track test',
    'Lane ecef point',
    'imu frame eulers',
    'GLONASS pseudorange',
    'GLONASS pseudorange rate',

    'Road Frame x,y speed',
    'Road Frame yaw rate',
    'Steer Angle',
    'Fast Angle Offset',
    'Stiffness',
    'Steer Ratio',
  ]

  @classmethod
  def to_string(cls, kind):
    return cls.names[kind]


SAT_OBS = [ObservationKind.PSEUDORANGE_GPS,
           ObservationKind.PSEUDORANGE_RATE_GPS,
           ObservationKind.PSEUDORANGE_GLONASS,
           ObservationKind.PSEUDORANGE_RATE_GLONASS]
