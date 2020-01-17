from cereal import log as capnp_log

def write_can_to_msg(data, src, msg):
  if not isinstance(data[0], Sequence):
    data = [data]

  can_msgs = msg.init('can', len(data))
  for i, d in enumerate(data):
    if d[0] < 0: continue # ios bug
    cc = can_msgs[i]
    cc.address = d[0]
    cc.busTime = 0
    cc.dat = hex_to_str(d[2])
    if len(d) == 4:
      cc.src = d[3]
      cc.busTime = d[1]
    else:
      cc.src = src

def convert_old_pkt_to_new(old_pkt):
  m, d = old_pkt
  msg = capnp_log.Event.new_message()

  if len(m) == 3:
    _, pid, t = m
    msg.logMonoTime = t
  else:
    t, pid = m
    msg.logMonoTime = int(t * 1e9)

  last_velodyne_time = None

  if pid == PID_OBD:
    write_can_to_msg(d, 0, msg)
  elif pid == PID_CAM:
    frame = msg.init('frame')
    frame.frameId = d[0]
    frame.timestampEof = msg.logMonoTime
  # iOS
  elif pid == PID_IGPS:
    loc = msg.init('gpsLocation')
    loc.latitude = d[0]
    loc.longitude = d[1]
    loc.speed = d[2]
    loc.timestamp = int(m[0]*1000.0)   # on iOS, first number is wall time in seconds
    loc.flags = 1 | 4  # has latitude, longitude, and speed.
  elif pid == PID_IMOTION:
    user_acceleration = d[:3]
    gravity = d[3:6]

    # iOS separates gravity from linear acceleration, so we recombine them.
    # Apple appears to use this constant for the conversion.
    g = -9.8
    acceleration = [g*(a + b) for a, b in zip(user_acceleration, gravity)]

    accel_event = msg.init('sensorEvents', 1)[0]
    accel_event.acceleration.v = acceleration
  # android
  elif pid == PID_GPS:
    if len(d) <= 6 or d[-1] == "gps":
      loc = msg.init('gpsLocation')
      loc.latitude = d[0]
      loc.longitude = d[1]
      loc.speed = d[2]
      if len(d) > 6:
        loc.timestamp = d[6]
      loc.flags = 1 | 4  # has latitude, longitude, and speed.
  elif pid == PID_ACCEL:
    val = d[2] if type(d[2]) != type(0.0) else d
    accel_event = msg.init('sensorEvents', 1)[0]
    accel_event.acceleration.v = val
  elif pid == PID_GYRO:
    val = d[2] if type(d[2]) != type(0.0) else d
    gyro_event = msg.init('sensorEvents', 1)[0]
    gyro_event.init('gyro').v = val
  elif pid == PID_LIDAR:
    lid = msg.init('lidarPts')
    lid.idx = d[3]
  elif pid == PID_APPLANIX:
    loc = msg.init('liveLocation')
    loc.status = d[18]

    loc.lat, loc.lon, loc.alt = d[0:3]
    loc.vNED = d[3:6]

    loc.roll = d[6]
    loc.pitch = d[7]
    loc.heading = d[8]

    loc.wanderAngle = d[9]
    loc.trackAngle = d[10]

    loc.speed = d[11]

    loc.gyro = d[12:15]
    loc.accel = d[15:18]
  elif pid == PID_IBAROMETER:
    pressure_event = msg.init('sensorEvents', 1)[0]
    _, pressure = d[0:2]
    pressure_event.init('pressure').v = [pressure] # Kilopascals
  elif pid == PID_IINIT and len(d) == 4:
    init_event = msg.init('initData')
    init_event.deviceType = capnp_log.InitData.DeviceType.chffrIos

    build_info = init_event.init('iosBuildInfo')
    build_info.appVersion = d[0]
    build_info.appBuild = int(d[1])
    build_info.osVersion = d[2]
    build_info.deviceModel = d[3]

  return msg.as_reader()
