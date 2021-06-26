#!/usr/bin/env python3.7
'''
GPS cord converter: https://gist.github.com/jp1017/71bd0976287ce163c11a7cb963b04dd8
'''
import cereal.messaging as messaging
import os
import time
import datetime
import signal
import threading
import math
import zipfile

pi = 3.1415926535897932384626
x_pi = 3.14159265358979324 * 3000.0 / 180.0
a = 6378245.0
ee = 0.00669342162296594323

GPX_LOG_PATH = '/sdcard/gpx_logs/'

LOG_DELAY = 0.1 # secs, lower for higher accuracy, 0.1 seems fine
LOG_LENGTH = 60 # mins, higher means it keeps more data in the memory, will take more time to write into a file too.
LOST_SIGNAL_COUNT_LENGTH = 30 # secs, if we lost signal for this long, perform output to data
MIN_MOVE_SPEED_KMH = 5 # km/h, min speed to trigger logging

# do not change
LOST_SIGNAL_COUNT_MAX = LOST_SIGNAL_COUNT_LENGTH / LOG_DELAY # secs,
LOGS_PER_FILE = LOG_LENGTH * 60 / LOG_DELAY # e.g. 3 * 60 / 0.1 = 1800 points per file
MIN_MOVE_SPEED_MS = MIN_MOVE_SPEED_KMH / 3.6

class WaitTimeHelper:
  ready_event = threading.Event()
  shutdown = False

  def __init__(self):
    signal.signal(signal.SIGTERM, self.graceful_shutdown)
    signal.signal(signal.SIGINT, self.graceful_shutdown)
    signal.signal(signal.SIGHUP, self.graceful_shutdown)

  def graceful_shutdown(self, signum, frame):
    self.shutdown = True
    self.ready_event.set()

def main():
  # init
  sm = messaging.SubMaster(['gpsLocationExternal'])
  log_count = 0
  logs = list()
  lost_signal_count = 0
  wait_helper = WaitTimeHelper()
  started_time = datetime.datetime.utcnow().isoformat()
  # outside_china_checked = False
  # outside_china = False
  while True:
    sm.update()
    if sm.updated['gpsLocationExternal']:
      gps = sm['gpsLocationExternal']

      # do not log when no fix or accuracy is too low, add lost_signal_count
      if gps.flags % 2 == 0 or gps.accuracy > 5.:
        if log_count > 0:
          lost_signal_count += 1
      else:
        lng = gps.longitude
        lat = gps.latitude
        # if not outside_china_checked:
        #   outside_china = out_of_china(lng, lat)
        #   outside_china_checked = True
        # if not outside_china:
        #   lng, lat = wgs84togcj02(lng, lat)
        logs.append([datetime.datetime.utcfromtimestamp(gps.timestamp*0.001).isoformat(), lat, lng, gps.altitude])
        log_count += 1
        lost_signal_count = 0
    '''
    write to log if
    1. reach per file limit
    2. lost signal for a certain time (e.g. under cover car park?)
    '''
    if log_count > 0 and (log_count >= LOGS_PER_FILE or lost_signal_count >= LOST_SIGNAL_COUNT_MAX):
      # output
      to_gpx(logs, started_time)
      lost_signal_count = 0
      log_count = 0
      logs.clear()
      started_time = datetime.datetime.utcnow().isoformat()

    time.sleep(LOG_DELAY)
    if wait_helper.shutdown:
      break
  # when process end, we store any logs.
  if log_count > 0:
    to_gpx(logs, started_time)

'''
check to see if it's in china
'''
def out_of_china(lng, lat):
  if lng < 72.004 or lng > 137.8347:
    return True
  elif lat < 0.8293 or lat > 55.8271:
    return True
  return False

def transform_lat(lng, lat):
  ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
  ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
  ret += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
  ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
  return ret

def transform_lng(lng, lat):
  ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
  ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
  ret += (20.0 * math.sin(lng * pi) + 40.0 * math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
  ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 * math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
  return ret

'''
Convert wgs84 to gcj02 (
'''
def wgs84togcj02(lng, lat):
  if out_of_china(lng, lat):
    return lng, lat
  dlat = transform_lat(lng - 105.0, lat - 35.0)
  dlng = transform_lng(lng - 105.0, lat - 35.0)
  radlat = lat / 180.0 * pi
  magic = math.sin(radlat)
  magic = 1 - ee * magic * magic
  sqrtmagic = math.sqrt(magic)
  dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
  dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
  mglat = lat + dlat
  mglng = lng + dlng
  return mglng, mglat

'''
write logs to a gpx file and zip it
'''
def to_gpx(logs, timestamp):
  if len(logs) > 0:
    if not os.path.exists(GPX_LOG_PATH):
      os.makedirs(GPX_LOG_PATH)
    filename = timestamp.replace(':','-')
    str = ''
    str += "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>\n"
    str += "<gpx xmlns=\"http://www.topografix.com/GPX/1/1\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd\" version=\"1.1\">\n"
    str += "\t<trk>\n"
    str += "\t\t<trkseg>\n"
    for trkpt in logs:
      str += "\t\t\t<trkpt time=\"%sZ\" lat=\"%s\" lon=\"%s\" ele=\"%s\" />\n" % (trkpt[0], trkpt[1], trkpt[2], trkpt[3])
    str += "\t\t</trkseg>\n"
    str += "\t</trk>\n"
    str += "</gpx>\n"
    try:
      zi = zipfile.ZipInfo('%sZ.gpx' % filename, time.localtime())
      zi.compress_type = zipfile.ZIP_DEFLATED
      zf = zipfile.ZipFile('%s%sZ.zip' % (GPX_LOG_PATH, filename), mode='w')
      zf.writestr(zi, str)
      zf.close()
    except:
      pass

if __name__ == "__main__":
  main()
