#!/usr/bin/env python3
import numpy as np
import os
import random
import string
import subprocess
import time
import unittest
from collections import defaultdict
from pathlib import Path

import cereal.messaging as messaging
from cereal import log
from cereal.services import service_list
from common.basedir import BASEDIR
from common.params import Params
from common.timeout import Timeout
from selfdrive.hardware import TICI
from selfdrive.loggerd.config import ROOT
from selfdrive.manager.process_config import managed_processes
from selfdrive.version import get_version
from tools.lib.logreader import LogReader
from cereal.visionipc.visionipc_pyx import VisionIpcServer, VisionStreamType  # pylint: disable=no-name-in-module, import-error
from common.transformations.camera import eon_f_frame_size, tici_f_frame_size, \
                                          eon_d_frame_size, tici_d_frame_size, tici_e_frame_size

SentinelType = log.Sentinel.SentinelType

CEREAL_SERVICES = [f for f in log.Event.schema.union_fields if f in service_list
                   and service_list[f].should_log and "encode" not in f.lower()]


class TestLoggerd(unittest.TestCase):
  def _get_latest_log_dir(self):
    log_dirs = sorted(Path(ROOT).iterdir(), key=lambda f: f.stat().st_mtime)
    return log_dirs[-1]

  def _get_log_dir(self, x):
    for l in x.splitlines():
      for p in l.split(' '):
        path = Path(p.strip())
        if path.is_dir():
          return path
    return None

  def _get_log_fn(self, x):
    for l in x.splitlines():
      for p in l.split(' '):
        path = Path(p.strip())
        if path.is_file():
          return path
    return None

  def _gen_bootlog(self):
    with Timeout(5):
      out = subprocess.check_output("./bootlog", cwd=os.path.join(BASEDIR, "selfdrive/loggerd"), encoding='utf-8')

    log_fn = self._get_log_fn(out)

    # check existence
    assert log_fn is not None

    return log_fn

  def _check_init_data(self, msgs):
    msg = msgs[0]
    self.assertEqual(msg.which(), 'initData')

  def _check_sentinel(self, msgs, route):
    start_type = SentinelType.startOfRoute if route else SentinelType.startOfSegment
    self.assertTrue(msgs[1].sentinel.type == start_type)

    end_type = SentinelType.endOfRoute if route else SentinelType.endOfSegment
    self.assertTrue(msgs[-1].sentinel.type == end_type)

  def test_init_data_values(self):
    os.environ["CLEAN"] = random.choice(["0", "1"])

    dongle  = ''.join(random.choice(string.printable) for n in range(random.randint(1, 100)))
    fake_params = [
      # param, initData field, value
      ("DongleId", "dongleId", dongle),
      ("GitCommit", "gitCommit", "commit"),
      ("GitBranch", "gitBranch", "branch"),
      ("GitRemote", "gitRemote", "remote"),
    ]
    params = Params()
    for k, _, v in fake_params:
      params.put(k, v)

    lr = list(LogReader(str(self._gen_bootlog())))
    initData = lr[0].initData

    self.assertTrue(initData.dirty != bool(os.environ["CLEAN"]))
    self.assertEqual(initData.version, get_version())

    if os.path.isfile("/proc/cmdline"):
      with open("/proc/cmdline") as f:
        self.assertEqual(list(initData.kernelArgs), f.read().strip().split(" "))

      with open("/proc/version") as f:
        self.assertEqual(initData.kernelVersion, f.read())

    for _, k, v in fake_params:
      self.assertEqual(getattr(initData, k), v)

  def test_rotation(self):
    os.environ["LOGGERD_TEST"] = "1"
    Params().put("RecordFront", "1")

    expected_files = {"rlog.bz2", "qlog.bz2", "qcamera.ts", "fcamera.hevc", "dcamera.hevc"}
    streams = [(VisionStreamType.VISION_STREAM_ROAD, tici_f_frame_size if TICI else eon_f_frame_size, "roadCameraState"),
              (VisionStreamType.VISION_STREAM_DRIVER, tici_d_frame_size if TICI else eon_d_frame_size, "driverCameraState")]
    if TICI:
      expected_files.add("ecamera.hevc")
      streams.append((VisionStreamType.VISION_STREAM_WIDE_ROAD, tici_e_frame_size, "wideRoadCameraState"))

    pm = messaging.PubMaster(["roadCameraState", "driverCameraState", "wideRoadCameraState"])
    vipc_server = VisionIpcServer("camerad")
    for stream_type, frame_size, _ in streams:
      vipc_server.create_buffers(stream_type, 40, False, *(frame_size))
    vipc_server.start_listener()

    for _ in range(5):
      num_segs = random.randint(2, 5)
      length = random.randint(1, 3)
      os.environ["LOGGERD_SEGMENT_LENGTH"] = str(length)
      managed_processes["loggerd"].start()

      fps = 20.0
      for n in range(1, int(num_segs*length*fps)+1):
        for stream_type, frame_size, state in streams:
          dat = np.empty(int(frame_size[0]*frame_size[1]*3/2), dtype=np.uint8)
          vipc_server.send(stream_type, dat[:].flatten().tobytes(), n, n/fps, n/fps)

          camera_state = messaging.new_message(state)
          frame = getattr(camera_state, state)
          frame.frameId = n
          pm.send(state, camera_state)
        time.sleep(1.0/fps)

      managed_processes["loggerd"].stop()

      route_path = str(self._get_latest_log_dir()).rsplit("--", 1)[0]
      for n in range(num_segs):
        p = Path(f"{route_path}--{n}")
        logged = {f.name for f in p.iterdir() if f.is_file()}
        diff = logged ^ expected_files
        self.assertEqual(len(diff), 0, f"didn't get all expected files. run={_} seg={n} {route_path=}, {diff=}\n{logged=} {expected_files=}")

  def test_bootlog(self):
    # generate bootlog with fake launch log
    launch_log = ''.join(str(random.choice(string.printable)) for _ in range(100))
    with open("/tmp/launch_log", "w") as f:
      f.write(launch_log)

    bootlog_path = self._gen_bootlog()
    lr = list(LogReader(str(bootlog_path)))

    # check length
    assert len(lr) == 2  # boot + initData

    self._check_init_data(lr)

    # check msgs
    bootlog_msgs = [m for m in lr if m.which() == 'boot']
    assert len(bootlog_msgs) == 1

    # sanity check values
    boot = bootlog_msgs.pop().boot
    assert abs(boot.wallTimeNanos - time.time_ns()) < 5*1e9 # within 5s
    assert boot.launchLog == launch_log

    for fn in ["console-ramoops", "pmsg-ramoops-0"]:
      path = Path(os.path.join("/sys/fs/pstore/", fn))
      if path.is_file():
        expected_val = open(path, "rb").read()
        bootlog_val = [e.value for e in boot.pstore.entries if e.key == fn][0]
        self.assertEqual(expected_val, bootlog_val)

  def test_qlog(self):
    qlog_services = [s for s in CEREAL_SERVICES if service_list[s].decimation is not None]
    no_qlog_services = [s for s in CEREAL_SERVICES if service_list[s].decimation is None]

    services = random.sample(qlog_services, random.randint(2, min(10, len(qlog_services)))) + \
               random.sample(no_qlog_services, random.randint(2, min(10, len(no_qlog_services))))

    pm = messaging.PubMaster(services)

    # sleep enough for the first poll to time out
    # TOOD: fix loggerd bug dropping the msgs from the first poll
    managed_processes["loggerd"].start()
    for s in services:
      while not pm.all_readers_updated(s):
        time.sleep(0.1)

    sent_msgs = defaultdict(list)
    for _ in range(random.randint(2, 10) * 100):
      for s in services:
        try:
          m = messaging.new_message(s)
        except Exception:
          m = messaging.new_message(s, random.randint(2, 10))
        pm.send(s, m)
        sent_msgs[s].append(m)
      time.sleep(0.01)

    time.sleep(1)
    managed_processes["loggerd"].stop()

    qlog_path = os.path.join(self._get_latest_log_dir(), "qlog.bz2")
    lr = list(LogReader(qlog_path))

    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    recv_msgs = defaultdict(list)
    for m in lr:
      recv_msgs[m.which()].append(m)

    for s, msgs in sent_msgs.items():
      recv_cnt = len(recv_msgs[s])

      if s in no_qlog_services:
        # check services with no specific decimation aren't in qlog
        self.assertEqual(recv_cnt, 0, f"got {recv_cnt} {s} msgs in qlog")
      else:
        # check logged message count matches decimation
        expected_cnt = (len(msgs) - 1) // service_list[s].decimation + 1
        self.assertEqual(recv_cnt, expected_cnt, f"expected {expected_cnt} msgs for {s}, got {recv_cnt}")

  def test_rlog(self):
    services = random.sample(CEREAL_SERVICES, random.randint(5, 10))
    pm = messaging.PubMaster(services)

    # sleep enough for the first poll to time out
    # TODO: fix loggerd bug dropping the msgs from the first poll
    managed_processes["loggerd"].start()
    for s in services:
      while not pm.all_readers_updated(s):
        time.sleep(0.1)

    sent_msgs = defaultdict(list)
    for _ in range(random.randint(2, 10) * 100):
      for s in services:
        try:
          m = messaging.new_message(s)
        except Exception:
          m = messaging.new_message(s, random.randint(2, 10))
        pm.send(s, m)
        sent_msgs[s].append(m)

    time.sleep(2)
    managed_processes["loggerd"].stop()

    lr = list(LogReader(os.path.join(self._get_latest_log_dir(), "rlog.bz2")))

    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    # check all messages were logged and in order
    lr = lr[2:-1] # slice off initData and both sentinels
    for m in lr:
      sent = sent_msgs[m.which()].pop(0)
      sent.clear_write_flag()
      self.assertEqual(sent.to_bytes(), m.as_builder().to_bytes())


if __name__ == "__main__":
  unittest.main()
