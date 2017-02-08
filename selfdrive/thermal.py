"""Methods for reading system thermal information."""
import selfdrive.messaging as messaging

def read_tz(x):
  with open("/sys/devices/virtual/thermal/thermal_zone%d/temp" % x) as f:
    ret = max(0, int(f.read()))
  return ret

def read_thermal():
  dat = messaging.new_message()
  dat.init('thermal')
  dat.thermal.cpu0 = read_tz(5)
  dat.thermal.cpu1 = read_tz(7)
  dat.thermal.cpu2 = read_tz(10)
  dat.thermal.cpu3 = read_tz(12)
  dat.thermal.mem = read_tz(2)
  dat.thermal.gpu = read_tz(16)
  dat.thermal.bat = read_tz(29)
  return dat
