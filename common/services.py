# TODO: these port numbers are hardcoded in c, fix this

# LogRotate: 8001 is a PUSH PULL socket between loggerd and visiond

class Service(object):
  def __init__(self, port, should_log):
    self.port = port
    self.should_log = should_log

# all ZMQ pub sub
service_list = {
  # frame syncing packet
  "frame": Service(8002, True),
  # accel, gyro, and compass
  "sensorEvents": Service(8003, True),
  # GPS data, also global timestamp
  "gpsNMEA": Service(8004, True),
  # CPU+MEM+GPU+BAT temps
  "thermal": Service(8005, True),
  # List(CanData), list of can messages
  "can": Service(8006, True),
  "live100": Service(8007, True),
  # random events we want to log
  #"liveEvent": Service(8008, True),
  "model": Service(8009, True),
  "features": Service(8010, True),
  "health": Service(8011, True),
  "live20": Service(8012, True),
  #"liveUI": Service(8014, True),
  "encodeIdx": Service(8015, True),
  "liveTracks": Service(8016, True),
  "sendcan": Service(8017, True),
  "logMessage": Service(8018, True),
  "liveCalibration": Service(8019, True),
  "androidLog": Service(8020, True),
}

# manager -- base process to manage starting and stopping of all others
#   subscribes: health
#   publishes:  thermal

# boardd -- communicates with the car
#   subscribes: sendcan
#   publishes:  can, health

# visiond -- talks to the cameras, runs the model, saves the videos
#   subscribes: liveCalibration, sensorEvents
#   publishes:  frame, encodeIdx, model, features

# controlsd -- actually drives the car
#   subscribes: can, thermal, model, live20
#   publishes:  sendcan, live100

# radard -- processes the radar data
#   subscribes: can, live100, model
#   publishes:  live20, liveTracks

# sensord -- publishes the IMU and GPS
#   publishes:  sensorEvents, gpsNMEA

# calibrationd -- places the camera box
#   subscribes: features, live100
#   publishes:  liveCalibration

# **** LOGGING SERVICE ****

# loggerd
#   subscribes: EVERYTHING

# **** NON VITAL SERVICES ****

# ui
#   subscribes: live100, live20, liveCalibration, model, (raw frames)

# uploader
#   communicates through file system with loggerd

# logmessaged -- central logging service, can log to cloud
#   publishes:  logMessage

# logcatd -- fetches logcat info from android
#   publishes:  androidLog
