#!/usr/bin/env python3
# The MIT License
#
# Copyright (c) 2019-, Rick Lan, dragonpilot community, and a number of other of contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cereal.messaging as messaging
import os
import datetime
import signal
import threading
from common.realtime import Ratekeeper
from cereal import log
from system.swaglog import cloudlog

# customisable values
GPX_LOG_PATH = '/data/media/0/gpx_logs/'
LOG_HERTZ = 10  # 10 hz = 0.1 sec, higher for higher accuracy, 10hz seems fine
LOG_LENGTH = 10  # mins, higher means it keeps more data in the memory, will take more time to write into a file too.
LOST_SIGNAL_COUNT_LENGTH = 10  # secs, output log file if we lost signal for this long

# do not change
LOST_SIGNAL_COUNT_MAX = LOST_SIGNAL_COUNT_LENGTH * LOG_HERTZ  # secs,
LOGS_PER_FILE = LOG_LENGTH * 60 * LOG_HERTZ  # e.g. 10 * 60 * 10 = 6000 points per file


_DEBUG = False
_CLOUDLOG_DEBUG = True


def _debug(msg, log_to_cloud=True):
  if _CLOUDLOG_DEBUG and log_to_cloud:
    cloudlog.debug(msg)
  if _DEBUG:
    print(msg)


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


class GpxD:
  def __init__(self, sm):
    self.sm = sm
    self.rk = Ratekeeper(LOG_HERTZ, print_delay_threshold=None)

    self.log_count = 0
    self.logs = list()
    self.lost_signal_count = 0
    self.wait_helper = WaitTimeHelper()
    self.started_time = datetime.datetime.utcnow().isoformat()
    self.v_ego_prev = 0.
    self.pause = True

  def log(self):
    location = self.sm['liveLocationKalman']
    gps = self.sm['gpsLocationExternal']
    v_ego = self.sm['carState'].vEgo

    if abs(v_ego) > 0.:
      self.pause = False

    location_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid
    _debug("gpxd: location_valid - %s" % location_valid)
    if not location_valid or self.pause:
      if self.log_count > 0:
        self.lost_signal_count += 1
    else:
      lat = location.positionGeodetic.value[0]
      lon = location.positionGeodetic.value[1]
      alt = gps.altitude

      _debug("gpxd: logged - %s %s %s %s" % (datetime.datetime.utcfromtimestamp(location.unixTimestampMillis*0.001).isoformat(), str(lat), str(lon), str(alt)))
      self.logs.append([datetime.datetime.utcfromtimestamp(location.unixTimestampMillis*0.001).isoformat(), str(lat), str(lon), str(alt)])
      self.log_count += 1
      self.lost_signal_count = 0

    if not self.pause and abs(v_ego) < 0.01:
      _debug("gpxd: paused")
      self.pause = True

    self.v_ego_prev = v_ego

  def write_log(self, force = False):
    if self.log_count == 0:
      return

    if force or (self.log_count >= LOGS_PER_FILE or self.lost_signal_count >= LOST_SIGNAL_COUNT_MAX):
      _debug("gpxd: save to log")
      self._write_gpx()
      self.lost_signal_count = 0
      self.log_count = 0
      self.logs.clear()
      self.started_time = datetime.datetime.utcnow().isoformat()

  def _write_gpx(self):
    if len(self.logs) > 1:
      if not os.path.exists(GPX_LOG_PATH):
        os.makedirs(GPX_LOG_PATH)
      filename = self.started_time.replace(':','-')
      str = ''
      str += "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"yes\"?>\n"
      str += "<gpx version=\"1.1\" creator=\"sunnypilot https://github.com/sunnypilot/openpilot\" xmlns=\"http://www.topografix.com/GPX/1/1\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd\">\n"
      str += "<trk>\n"
      str += "  <name>" + self.started_time + "</name>"
      str += "  <trkseg>\n"
      for trkpt in self.logs:
        str += self._trkpt_template(trkpt[1], trkpt[2], trkpt[3], trkpt[0])
      str += "  </trkseg>\n"
      str += "</trk>\n"
      str += "</gpx>\n"
      try:
        f = open('%s%sZ.gpx' % (GPX_LOG_PATH, filename), 'w')
        f.write(str)
        f.close()
      except:
        pass

  def _trkpt_template(self, lat, lon, ele, time):
    str = ""
    str += "    <trkpt lat=\"" + lat + "\" lon=\"" + lon + "\">\n"
    str += "      <ele>" + ele + "</ele>\n"
    str += "      <time>" + time + "</time>\n"
    str += "    </trkpt>\n"
    return str

  def gpxd_thread(self):
    while True:
      self.sm.update(0)
      self.log()
      self.write_log()
      if self.wait_helper.shutdown:
        self.write_log(True)
        break
      self.rk.keep_time()


def main(sm=None):
  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'gpsLocationExternal', 'carState'])

  gpx = GpxD(sm)
  gpx.gpxd_thread()


if __name__ == "__main__":
  main()
