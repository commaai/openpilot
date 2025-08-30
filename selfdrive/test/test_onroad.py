import math
import json
import os
import pytest
import shutil
import subprocess
import time
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from tabulate import tabulate

from cereal import log
import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.basedir import BASEDIR
from openpilot.common.timeout import Timeout
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.events import EVENTS, ET
from openpilot.selfdrive.test.helpers import set_params_enabled, release_only
from openpilot.system.hardware.hw import Paths
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.log_time_series import msgs_to_time_series

"""
CPU usage budget
* each process is entitled to at least 8%
* total CPU usage of openpilot (sum(PROCS.values())
  should not exceed MAX_TOTAL_CPU
"""

TEST_DURATION = 25
LOG_OFFSET = 8

MAX_TOTAL_CPU = 287.  # total for all 8 cores
PROCS = {
  # Baseline CPU usage by process
  "selfdrive.controls.controlsd": 16.0,
  "selfdrive.selfdrived.selfdrived": 16.0,
  "selfdrive.car.card": 26.0,
  "./loggerd": 14.0,
  "./encoderd": 13.0,
  "./camerad": 10.0,
  "selfdrive.controls.plannerd": 8.0,
  "./ui": 18.0,
  "system.sensord.sensord": 13.0,
  "selfdrive.controls.radard": 2.0,
  "selfdrive.modeld.modeld": 22.0,
  "selfdrive.modeld.dmonitoringmodeld": 18.0,
  "system.hardware.hardwared": 4.0,
  "selfdrive.locationd.calibrationd": 2.0,
  "selfdrive.locationd.torqued": 5.0,
  "selfdrive.locationd.locationd": 25.0,
  "selfdrive.locationd.paramsd": 9.0,
  "selfdrive.locationd.lagd": 11.0,
  "selfdrive.ui.soundd": 3.0,
  "selfdrive.ui.feedback.feedbackd": 1.0,
  "selfdrive.monitoring.dmonitoringd": 4.0,
  "./proclogd": 2.0,
  "system.logmessaged": 1.0,
  "system.tombstoned": 0,
  "./logcatd": 1.0,
  "system.micd": 5.0,
  "system.timed": 0,
  "selfdrive.pandad.pandad": 0,
  "system.statsd": 1.0,
  "system.loggerd.uploader": 15.0,
  "system.loggerd.deleter": 1.0,
  "./pandad": 19.0,
  "system.qcomgpsd.qcomgpsd": 1.0,
}

TIMINGS = {
  # rtols: max/min, rsd
  "can": [2.5, 0.35],
  "pandaStates": [2.5, 0.35],
  "peripheralState": [2.5, 0.35],
  "sendcan": [2.5, 0.35],
  "carState": [2.5, 0.35],
  "carControl": [2.5, 0.35],
  "controlsState": [2.5, 0.35],
  "longitudinalPlan": [2.5, 0.5],
  "driverAssistance": [2.5, 0.5],
  "roadCameraState": [2.5, 0.35],
  "driverCameraState": [2.5, 0.35],
  "modelV2": [2.5, 0.35],
  "driverStateV2": [2.5, 0.40],
  "livePose": [2.5, 0.35],
  "liveParameters": [2.5, 0.35],
  "wideRoadCameraState": [1.5, 0.35],
}

LOGS_SIZE = {  # MB per segment
  "qlog.zst": 0.5,
  "rlog.zst": 8.1,
  "qcamera.ts": 2.3,
}
LOGS_SIZE.update(dict.fromkeys(['ecamera.hevc', 'fcamera.hevc', 'dcamera.hevc'], 76.5))


def cputime_total(ct):
  return ct.cpuUser + ct.cpuSystem + ct.cpuChildrenUser + ct.cpuChildrenSystem


@pytest.mark.tici
@pytest.mark.skip_tici_setup
class TestOnroad:

  @classmethod
  def setup_class(cls):
    if "DEBUG" in os.environ:
      segs = filter(lambda x: os.path.exists(os.path.join(x, "rlog.zst")), Path(Paths.log_root()).iterdir())
      segs = sorted(segs, key=lambda x: x.stat().st_mtime)
      cls.lr = list(LogReader(os.path.join(segs[-1], "rlog.zst")))
      cls.ts = msgs_to_time_series(cls.lr)
      return

    # setup env
    params = Params()
    params.remove("CurrentRoute")
    params.put_bool("RecordFront", True)
    set_params_enabled()
    os.environ['REPLAY'] = '1'
    os.environ['TESTING_CLOSET'] = '1'
    if os.path.exists(Paths.log_root()):
      shutil.rmtree(Paths.log_root())

    # start manager and run openpilot for TEST_DURATION
    proc = None
    try:
      manager_path = os.path.join(BASEDIR, "system/manager/manager.py")
      cls.manager_st = time.monotonic()
      proc = subprocess.Popen(["python", manager_path])

      sm = messaging.SubMaster(['carState'])
      with Timeout(30, "controls didn't start"):
        while not sm.seen['carState']:
          sm.update(1000)

      route = params.get("CurrentRoute")
      assert route is not None

      segs = list(Path(Paths.log_root()).glob(f"{route}--*"))
      assert len(segs) == 1

      time.sleep(TEST_DURATION)
    finally:
      if proc is not None:
        proc.terminate()
        if proc.wait(60) is None:
          proc.kill()

    cls.lr = list(LogReader(os.path.join(str(segs[0]), "rlog.zst")))
    st = time.monotonic()
    cls.ts = msgs_to_time_series(cls.lr)
    print("msgs to time series", time.monotonic() - st)
    log_path = segs[0]

    cls.log_sizes = {}
    for f in log_path.iterdir():
      assert f.is_file()
      cls.log_sizes[f] = f.stat().st_size / 1e6

    cls.msgs = defaultdict(list)
    for m in cls.lr:
      cls.msgs[m.which()].append(m)

  def test_service_frequencies(self, subtests):
    for s, msgs in self.msgs.items():
      if s in ('initData', 'sentinel'):
        continue

      # skip gps services for now
      if s in ('ubloxGnss', 'ubloxRaw', 'gnssMeasurements', 'gpsLocation', 'gpsLocationExternal', 'qcomGnss'):
        continue

      with subtests.test(service=s):
        assert len(msgs) >= math.floor(SERVICE_LIST[s].frequency*int(TEST_DURATION*0.8))

  def test_manager_starting_time(self):
    st = self.ts['managerState']['t'][0]
    assert (st - self.manager_st) < 12.5, f"manager.py took {st - self.manager_st}s to publish the first 'managerState' msg"

  def test_cloudlog_size(self):
    msgs = self.msgs['logMessage']

    total_size = sum(len(m.as_builder().to_bytes()) for m in msgs)
    assert total_size < 3.5e5

    cnt = Counter(json.loads(m.logMessage)['filename'] for m in msgs)
    big_logs = [f for f, n in cnt.most_common(3) if n / sum(cnt.values()) > 30.]
    assert len(big_logs) == 0, f"Log spam: {big_logs}"

  def test_log_sizes(self, subtests):
    # TODO: this isn't super stable between different devices
    for f, sz in self.log_sizes.items():
      rate = LOGS_SIZE[f.name]/60.
      minn = rate * TEST_DURATION * 0.5
      maxx = rate * TEST_DURATION * 1.5
      with subtests.test(file=f.name):
        assert minn < sz <  maxx

  def test_ui_timings(self):
    result = "\n"
    result += "------------------------------------------------\n"
    result += "-------------- UI Draw Timing ------------------\n"
    result += "------------------------------------------------\n"

    ts = self.ts['uiDebug']['drawTimeMillis']
    result += f"min  {min(ts):.2f}ms\n"
    result += f"max  {max(ts):.2f}ms\n"
    result += f"std  {np.std(ts):.2f}ms\n"
    result += f"mean {np.mean(ts):.2f}ms\n"
    result += "------------------------------------------------\n"
    print(result)

    assert max(ts) < 250.
    assert np.mean(ts) < 10.
    #self.assertLess(np.std(ts), 5.)

    # some slow frames are expected since camerad/modeld can preempt ui
    veryslow = [x for x in ts if x > 40.]
    assert len(veryslow) < 5, f"Too many slow frame draw times: {veryslow}"

  def test_cpu_usage(self, subtests):
    print("\n------------------------------------------------")
    print("------------------ CPU Usage -------------------")
    print("------------------------------------------------")

    plogs_by_proc = defaultdict(list)
    for pl in self.msgs['procLog']:
      for x in pl.procLog.procs:
        if len(x.cmdline) > 0:
          n = list(x.cmdline)[0]
          plogs_by_proc[n].append(x)

    cpu_ok = True
    dt = (self.msgs['procLog'][-1].logMonoTime - self.msgs['procLog'][0].logMonoTime) / 1e9
    header = ['process', 'usage', 'expected', 'max allowed', 'test result']
    rows = []
    for proc_name, expected in PROCS.items():

      error = ""
      usage = 0.
      x = plogs_by_proc[proc_name]
      if len(x) > 2:
        cpu_time = cputime_total(x[-1]) - cputime_total(x[0])
        usage = cpu_time / dt * 100.

        max_allowed = max(expected * 1.8, expected + 5.0)
        if usage > max_allowed:
          error = "❌ USING MORE CPU THAN EXPECTED ❌"
          cpu_ok = False

      else:
        error = "❌ NO METRICS FOUND ❌"
        cpu_ok = False

      rows.append([proc_name, usage, expected, max_allowed, error or "✅"])
    print(tabulate(rows, header, tablefmt="simple_grid", stralign="center", numalign="center", floatfmt=".2f"))

    # Ensure there's no missing procs
    all_procs = {p.name for p in self.msgs['managerState'][0].managerState.processes if p.shouldBeRunning}
    for p in all_procs:
      with subtests.test(proc=p):
        assert any(p in pp for pp in PROCS.keys()), f"Expected CPU usage missing for {p}"

    # total CPU check
    procs_tot = sum([(max(x) if isinstance(x, tuple) else x) for x in PROCS.values()])
    with subtests.test(name="total CPU"):
      assert procs_tot < MAX_TOTAL_CPU, "Total CPU budget exceeded"
    print("------------------------------------------------")
    print(f"Total allocated CPU usage is {procs_tot}%, budget is {MAX_TOTAL_CPU}%, {MAX_TOTAL_CPU-procs_tot:.1f}% left")
    print("------------------------------------------------")

    assert cpu_ok

  def test_memory_usage(self):
    print("\n------------------------------------------------")
    print("--------------- Memory Usage -------------------")
    print("------------------------------------------------")
    offset = int(SERVICE_LIST['deviceState'].frequency * LOG_OFFSET)
    mems = [m.deviceState.memoryUsagePercent for m in self.msgs['deviceState'][offset:]]
    print("Memory usage: ", mems)

    # check for big leaks. note that memory usage is
    # expected to go up while the MSGQ buffers fill up
    assert np.average(mems) <= 65, "Average memory usage above 65%"
    assert np.max(np.diff(mems)) <= 4, "Max memory increase too high"
    assert np.average(np.diff(mems)) <= 1, "Average memory increase too high"

  def test_camera_frame_timings(self, subtests):
    # test timing within a single camera
    result = "\n"
    result += "------------------------------------------------\n"
    result += "-----------------  SOF Timing ------------------\n"
    result += "------------------------------------------------\n"
    for name in ['roadCameraState', 'wideRoadCameraState', 'driverCameraState']:
      ts = self.ts[name]['timestampSof']
      d_ms = np.diff(ts) / 1e6
      d50 = np.abs(d_ms-50)
      result += f"{name} sof delta vs 50ms: min  {min(d50):.2f}ms\n"
      result += f"{name} sof delta vs 50ms: max  {max(d50):.2f}ms\n"
      result += f"{name} sof delta vs 50ms: mean {d50.mean():.2f}ms\n"
      with subtests.test(camera=name):
        assert max(d50) < 5.0, f"high SOF delta vs 50ms: {max(d50)}"
    result += "------------------------------------------------\n"
    print(result)

  def test_camera_sync(self, subtests):
    cam_states = ['roadCameraState', 'wideRoadCameraState', 'driverCameraState']
    encode_cams = ['roadEncodeIdx', 'wideRoadEncodeIdx', 'driverEncodeIdx']
    for cams in (cam_states, encode_cams):
      with subtests.test(cams=cams):
        # sanity checks within a single cam
        for cam in cams:
          with subtests.test(test="frame_skips", camera=cam):
            assert set(np.diff(self.ts[cam]['frameId'])) == {1, }, "Frame ID skips"

            # EOF > SOF
            eof_sof_diff = self.ts[cam]['timestampEof'] - self.ts[cam]['timestampSof']
            assert np.all(eof_sof_diff > 0)
            assert np.all(eof_sof_diff < 50*1e6)

        first_fid = {min(self.ts[c]['frameId']) for c in cams}
        assert len(first_fid) == 1, "Cameras don't start on same frame ID"
        if cam.endswith('CameraState'):
          # camerad guarantees that all cams start on frame ID 0
          # (note loggerd also needs to start up fast enough to catch it)
          assert next(iter(first_fid)) < 100, "Cameras start on frame ID too high"

        # we don't do a full segment rotation, so these might not match exactly
        last_fid = {max(self.ts[c]['frameId']) for c in cams}
        assert max(last_fid) - min(last_fid) < 10

        start, end = min(first_fid), min(last_fid)
        for i in range(end-start):
          ts = {c: round(self.ts[c]['timestampSof'][i]/1e6, 1) for c in cams}
          diff = (max(ts.values()) - min(ts.values()))
          assert diff < 2, f"Cameras not synced properly: frame_id={start+i}, {diff=:.1f}ms, {ts=}"

  def test_camera_encoder_matches(self, subtests):
    # sanity check that the frame metadata is consistent with the encoded frames
    pairs = [('roadCameraState', 'roadEncodeIdx'),
             ('wideRoadCameraState', 'wideRoadEncodeIdx'),
             ('driverCameraState', 'driverEncodeIdx')]
    for cam, enc in pairs:
      with subtests.test(camera=cam, encoder=enc):
        cam_frames = {fid: (sof, eof) for fid, sof, eof in zip(
          self.ts[cam]['frameId'],
          self.ts[cam]['timestampSof'],
          self.ts[cam]['timestampEof'],
          strict=True,
        )}
        for i, fid in enumerate(self.ts[enc]['frameId']):
          cam_sof, cam_eof = cam_frames[fid]
          enc_sof, enc_eof = self.ts[enc]['timestampSof'][i], self.ts[enc]['timestampEof'][i]
          assert enc_sof == cam_sof, f"SOF mismatch: frameId={fid}, enc_sof={enc_sof}, cam_sof={cam_sof}"
          assert enc_eof == cam_eof, f"EOF mismatch: frameId={fid}, enc_eof={enc_eof}, cam_eof={cam_eof}"

  def test_mpc_execution_timings(self):
    result = "\n"
    result += "------------------------------------------------\n"
    result += "-----------------  MPC Timing ------------------\n"
    result += "------------------------------------------------\n"

    cfgs = [("longitudinalPlan", 0.05, 0.05),]
    for (s, instant_max, avg_max) in cfgs:
      ts = [getattr(m, s).solverExecutionTime for m in self.msgs[s]]
      assert max(ts) < instant_max, f"high '{s}' execution time: {max(ts)}"
      assert np.mean(ts) < avg_max, f"high avg '{s}' execution time: {np.mean(ts)}"
      result += f"'{s}' execution time: min  {min(ts):.5f}s\n"
      result += f"'{s}' execution time: max  {max(ts):.5f}s\n"
      result += f"'{s}' execution time: mean {np.mean(ts):.5f}s\n"
    result += "------------------------------------------------\n"
    print(result)

  def test_model_execution_timings(self, subtests):
    result = "\n"
    result += "------------------------------------------------\n"
    result += "----------------- Model Timing -----------------\n"
    result += "------------------------------------------------\n"
    cfgs = [
      # since multiple processes use the GPU and can preempt each other,
      # these numbers are not fully self-contained.
      ("modelV2", 0.06, 0.040),

      # can miss cycles here and there, just important the avg frequency is 20Hz
      ("driverStateV2", 0.3, 0.05),
    ]
    for (s, instant_max, avg_max) in cfgs:
      ts = [getattr(m, s).modelExecutionTime for m in self.msgs[s]]
      # TODO some init can happen in first iteration
      ts = ts[1:]
      result += f"'{s}' execution time: min  {min(ts):.5f}s\n"
      result += f"'{s}' execution time: max {max(ts):.5f}s\n"
      result += f"'{s}' execution time: mean {np.mean(ts):.5f}s\n"
      with subtests.test(s):
        assert max(ts) < instant_max, f"high '{s}' execution time: {max(ts)}"
        assert np.mean(ts) < avg_max, f"high avg '{s}' execution time: {np.mean(ts)}"
    result += "------------------------------------------------\n"
    print(result)

  def test_timings(self):
    passed = True
    print("\n------------------------------------------------")
    print("----------------- Service Timings --------------")
    print("------------------------------------------------")

    header = ['service', 'max', 'min', 'mean', 'expected mean', 'rsd', 'max allowed rsd', 'test result']
    rows = []
    for s, (maxmin, rsd) in TIMINGS.items():
      offset = int(SERVICE_LIST[s].frequency * LOG_OFFSET)
      msgs = [m.logMonoTime for m in self.msgs[s][offset:]]
      if not len(msgs):
        raise Exception(f"missing {s}")

      ts = np.diff(msgs) / 1e9
      dt = 1 / SERVICE_LIST[s].frequency

      errors = []
      if not np.allclose(np.mean(ts), dt, rtol=0.03, atol=0):
        errors.append("❌ FAILED MEAN TIMING CHECK ❌")
      if not np.allclose([np.max(ts), np.min(ts)], dt, rtol=maxmin, atol=0):
        errors.append("❌ FAILED MAX/MIN TIMING CHECK ❌")
      if (np.std(ts)/dt) > rsd:
        errors.append("❌ FAILED RSD TIMING CHECK ❌")
      passed = not errors and passed
      rows.append([s, *(np.array([np.max(ts), np.min(ts), np.mean(ts), dt])*1e3), np.std(ts)/dt, rsd, "\n".join(errors) or "✅"])

    print(tabulate(rows, header, tablefmt="simple_grid", stralign="center", numalign="center", floatfmt=".2f"))
    assert passed

  @release_only
  def test_startup(self):
    startup_alert = self.ts['selfdriveState']['alertText1'][0]
    expected = EVENTS[log.OnroadEvent.EventName.startup][ET.PERMANENT].alert_text_1
    assert startup_alert == expected, "wrong startup alert"

  def test_engagable(self):
    no_entries = Counter()
    for m in self.msgs['onroadEvents']:
      for evt in m.onroadEvents:
        if evt.noEntry:
          no_entries[evt.name] += 1

    offset = int(SERVICE_LIST['selfdriveState'].frequency * LOG_OFFSET)
    eng = [m.selfdriveState.engageable for m in self.msgs['selfdriveState'][offset:]]
    assert all(eng), \
           f"Not engageable for whole segment:\n- selfdriveState.engageable: {Counter(eng)}\n- No entry events: {no_entries}"
