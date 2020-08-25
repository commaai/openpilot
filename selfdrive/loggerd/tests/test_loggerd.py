# flake8: noqa

import unittest
import numpy as np
import os
import time
import signal
import shutil
import subprocess

from common.basedir import BASEDIR
from common.params import Params
from common.hardware import EON, TICI

CAMERAD_KILL_TIMEOUT = 15

# baseline file sizes for a 2s segment
if EON:
  CAMERA_NAME = ['fcamera', 'dcamera']
  CAMERA_BASELINE_SIZE = [1253786, 650920] # in bytes
elif TICI:
  CAMERA_NAME = ['fcamera', 'dcamera', 'ecamera']
  CAMERA_BASELINE_SIZE = [1253786, 1253786, 1253786]
else:
  raise NotImplementedError("unknown hardware type")

rTOL = 0.1 # tolerate a 10% fluctuation based on content

params = Params()
ret = params.get("RecordFront")
if ret is None:
  has_dcam = False
else:
  has_dcam = bool(int(ret))

n_seg = 100

# TODO: add more test
class TestLoggerd(unittest.TestCase):

  # 0. test recording for 100 segments
  def test_rotations(self):
    proc_cam = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
    time.sleep(1.0)
    logtmp = open("/tmp/logtmp.txt", 'w+')
    proc_log = subprocess.Popen([os.path.join(BASEDIR, "selfdrive/loggerd/loggerd"), '--test_loggerd'], cwd=os.path.join(BASEDIR, "selfdrive/loggerd"),\
                                stdout=logtmp, universal_newlines=True)

    def terminate(signalNumber, frame, passed=False):
      # if receive keyboard interrupt, exit everything
      print('got SIGINT, exiting..')
      proc_cam.send_signal(signal.SIGINT)
      proc_log.send_signal(signal.SIGTERM)
      kill_start = time.time()
      while proc_cam.poll() is None:
        if time.time() - kill_start > CAMERAD_KILL_TIMEOUT:
          from selfdrive.swaglog import cloudlog
          cloudlog.critical("FORCE REBOOTING PHONE!")
          os.system("date >> /sdcard/unkillable_reboot")
          # os.system("reboot")
          raise RuntimeError
        continue
      if not passed:
        print('----')
        print('unittest terminated.')
        print('----')
        self.assertTrue(False)

    signal.signal(signal.SIGINT, terminate)

    start_time = time.time()
    while time.time() - start_time < 2.0 * n_seg:
      print("%d/%d" % (int(time.time() - start_time), int(2.0 * n_seg)), end='\r')
      time.sleep(0.5)
      # pass

    terminate(0, 0, passed=True)

    # processing
    with open("/tmp/logtmp.txt", "r") as f:
      logtmp_str = f.read()
    route_prefix = None
    for row in logtmp_str.split('\n'):
      if "logging to" in row:
        route_prefix = row.split("logging to ")[1][:-1]
        assert(os.path.isdir(route_prefix + '0'))
        break
    assert(route_prefix is not None)

    has_files = True
    for i in range(n_seg):
      for cidx in range(len(CAMERA_NAME)):
        if cidx == 1 and not has_dcam:
          continue
        has_files = has_files and os.path.isfile(route_prefix + '%d/%s.hevc' % (i, CAMERA_NAME[cidx]))

    correct_size = True
    fsz = []
    for _ in CAMERA_NAME:
      fsz.append([])
    for i in range(n_seg):
      for cidx in range(len(CAMERA_NAME)):
        if cidx == 1 and not has_dcam:
          continue
        fsize = os.path.getsize(route_prefix + '%d/%s.hevc' % (i, CAMERA_NAME[cidx]))
        correct_size = correct_size and CAMERA_BASELINE_SIZE[cidx] * (1 - rTOL) < fsize < CAMERA_BASELINE_SIZE[cidx] * (1 + rTOL)
        fsz[cidx].append(fsize)

    print("checked files in %s" % route_prefix[:-2])
    for cidx in range(len(CAMERA_NAME)):
      szs = np.array(fsz[cidx])
      print("%s.hevc: count=%d, avg_sz=%d, max_sz=%d, min_sz=%d" % (CAMERA_NAME[cidx], len(szs), szs.mean(), szs.max(), szs.min()))

    for i in range(n_seg):
      shutil.rmtree(route_prefix + '%d' % i)

    self.assertTrue(has_files and correct_size)

if __name__ == "__main__":
  print('--- testing loggerd ---')
  unittest.main()
