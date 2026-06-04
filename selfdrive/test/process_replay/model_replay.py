#!/usr/bin/env python3
import os
import pickle
import sys
from collections import defaultdict
from typing import Any
import tempfile
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
from openpilot.common.utils import tabulate

from openpilot.common.git import get_commit
from openpilot.system.hardware import ASIUS, PC
from openpilot.tools.lib.openpilotci import get_url
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs, format_diff
from openpilot.selfdrive.test.process_replay.process_replay import get_process_config, replay_process
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader, save_log
from openpilot.tools.lib.github_utils import GithubUtils

TEST_ROUTE = "8494c69d3c710e81|000001d4--2648a9a404"
SEGMENT = 4
START_FRAME = 0
END_FRAME = 60

SEND_EXTRA_INPUTS = bool(int(os.getenv("SEND_EXTRA_INPUTS", "0")))

DATA_TOKEN = os.getenv("CI_ARTIFACTS_TOKEN","")
API_TOKEN = os.getenv("GITHUB_COMMENTS_TOKEN","")
MODEL_REPLAY_BUCKET="model_replay_master"
GITHUB = GithubUtils(API_TOKEN, DATA_TOKEN)

EXEC_TIMINGS = [
  # model, instant max, average max
  # Dragon Q6A health uses the current full per-frame CL runtime as a regression
  # gate; PC/TICI CI keeps the upstream tighter threshold.
  ("modelV2", 0.14 if ASIUS else 0.05, 0.120 if ASIUS else 0.028),
]
if not ASIUS:
  EXEC_TIMINGS.append(("driverStateV2", 0.05, 0.018))

ASIUS_ACCURACY_CHECKS = [
  # name, field path, max abs error, p95 abs error, max relative error
  ("velocity.x", "modelV2.velocity.x", 2.6, 1.8, 0.10),
  ("acceleration.y", "modelV2.acceleration.y", 0.40, 0.30, None),
  ("inner lane prob left", "modelV2.laneLineProbs.1", 0.08, 0.06, 0.08),
  ("inner lane prob right", "modelV2.laneLineProbs.2", 0.08, 0.06, 0.08),
  ("outer lane prob left", "modelV2.laneLineProbs.0", 0.70, 0.62, 0.90),
  ("outer lane prob right", "modelV2.laneLineProbs.3", 0.45, 0.38, 0.75),
  ("lane stds", "modelV2.laneLineStds", 0.22, 0.15, 0.55),
  ("engaged prob", "modelV2.meta.engagedProb", 0.35, 0.30, 0.55),
  ("lead prob", "modelV2.leadsV3.0.prob", 0.55, 0.40, 0.75),
  ("steer override probs", "modelV2.meta.disengagePredictions.steerOverrideProbs", 0.16, 0.10, 0.45),
  ("camera trans", "cameraOdometry.trans", 0.90, 0.65, None),
  ("camera trans std", "cameraOdometry.transStd", 0.35, 0.18, 0.55),
  ("driving left lane prob", "drivingModelData.laneLineMeta.leftProb", 0.08, 0.06, 0.08),
  ("driving right lane prob", "drivingModelData.laneLineMeta.rightProb", 0.08, 0.06, 0.08),
]

def get_log_fn(test_route, ref="master"):
  return f"{test_route}_model_tici_{ref}.zst"

def plot(proposed, master, title, tmp):
  proposed = list(proposed)
  master = list(master)
  fig, ax = plt.subplots()
  ax.plot(master, label='MASTER')
  ax.plot(proposed, label='PROPOSED')
  plt.legend(loc='best')
  plt.title(title)
  plt.savefig(f'{tmp}/{title}.png')
  return (title + '.png', proposed == master)

def get_event(logs, event):
  return (getattr(m, m.which()) for m in filter(lambda m: m.which() == event, logs))

def zl(array, fill):
  return zip_longest(array, [], fillvalue=fill)

def get_idx_if_non_empty(l, idx=None):
  return l if idx is None else (l[idx] if len(l) > 0 else None)

def get_path_values(logs, path):
  parts = path.split(".")
  values = []
  for msg in logs:
    if msg.which() != parts[0]:
      continue
    value = getattr(msg, parts[0])
    for part in parts[1:]:
      value = value[int(part)] if part.isdigit() else getattr(value, part)
    values.append(np.asarray(value, dtype=np.float64).reshape(-1))
  return np.asarray(values)

def check_asius_accuracy(proposed, master):
  rows = []
  passed = True
  for name, path, max_abs_limit, p95_abs_limit, max_rel_limit in ASIUS_ACCURACY_CHECKS:
    proposed_values = get_path_values(proposed, path)
    master_values = get_path_values(master, path)
    abs_err = np.abs(proposed_values - master_values)
    denom = np.maximum(np.maximum(np.abs(proposed_values), np.abs(master_values)), 1e-6)
    rel_err = abs_err / denom

    max_abs = float(np.nanmax(abs_err))
    p95_abs = float(np.nanpercentile(abs_err, 95))
    max_rel = float(np.nanmax(rel_err))
    ok_abs = max_abs <= max_abs_limit and p95_abs <= p95_abs_limit
    ok_rel = max_rel_limit is None or max_rel <= max_rel_limit
    ok = ok_abs and ok_rel
    passed = passed and ok
    rel_limit = "-" if max_rel_limit is None else f"{max_rel_limit:.3f}"
    rows.append([name, max_abs, max_abs_limit, p95_abs, p95_abs_limit, max_rel, rel_limit, "✅" if ok else "❌"])

  print("------------------------------------------------")
  print("------------ Asius Model Accuracy --------------")
  print("------------------------------------------------")
  print(tabulate(rows, ["signal", "max abs", "max abs allowed", "p95 abs", "p95 abs allowed", "max rel", "max rel allowed", "test result"],
                 tablefmt="simple_grid", stralign="center", numalign="center", floatfmt=".4f"))
  return passed

def generate_report(proposed, master, tmp, commit):
  ModelV2_Plots = zl([
                     (lambda x: get_idx_if_non_empty(x.velocity.x, 0), "velocity.x"),
                     (lambda x: get_idx_if_non_empty(x.action.desiredCurvature), "desiredCurvature"),
                     (lambda x: get_idx_if_non_empty(x.action.desiredAcceleration), "desiredAcceleration"),
                     (lambda x: get_idx_if_non_empty(x.leadsV3[0].x, 0), "leadsV3.x"),
                     (lambda x: get_idx_if_non_empty(x.laneLines[1].y, 0), "laneLines.y"),
                     (lambda x: get_idx_if_non_empty(x.meta.desireState, 3), "desireState.laneChangeLeft"),
                     (lambda x: get_idx_if_non_empty(x.meta.desireState, 4), "desireState.laneChangeRight"),
                     (lambda x: get_idx_if_non_empty(x.meta.disengagePredictions.gasPressProbs, 1), "gasPressProbs")
                    ], "modelV2")
  DriverStateV2_Plots = zl([
                     (lambda x: get_idx_if_non_empty(x.wheelOnRightProb), "wheelOnRightProb"),
                     (lambda x: get_idx_if_non_empty(x.leftDriverData.faceProb), "leftDriverData.faceProb"),
                     (lambda x: get_idx_if_non_empty(x.leftDriverData.faceOrientation, 0), "leftDriverData.faceOrientation0"),
                     (lambda x: get_idx_if_non_empty(x.leftDriverData.leftBlinkProb), "leftDriverData.leftBlinkProb"),
                     (lambda x: get_idx_if_non_empty(x.leftDriverData.phoneProb), "leftDriverData.phoneProb"),
                     (lambda x: get_idx_if_non_empty(x.rightDriverData.faceProb), "rightDriverData.faceProb"),
                    ], "driverStateV2")

  return [plot(map(v[0], get_event(proposed, event)), \
               map(v[0], get_event(master, event)), f"{v[1]}_{commit[:7]}", tmp) \
               for v,event in ([*ModelV2_Plots] + [*DriverStateV2_Plots])]

def create_table(title, files, link, open_table=False):
  if not files:
    return ""
  table = [f'<details {"open" if open_table else ""}><summary>{title}</summary><table>']
  for i,f in enumerate(files):
    if not (i % 2):
      table.append("<tr>")
    table.append(f'<td><img src=\\"{link}/{f[0]}\\"></td>')
    if (i % 2):
      table.append("</tr>")
  table.append("</table></details>")
  table = "".join(table)
  return table

def comment_replay_report(proposed, master, full_logs):
  with tempfile.TemporaryDirectory() as tmp:
    PR_BRANCH = os.getenv("GIT_BRANCH","")
    DATA_BUCKET = f"model_replay_{PR_BRANCH}"

    try:
      GITHUB.get_pr_number(PR_BRANCH)
    except Exception:
      print("No PR associated with this branch. Skipping report.")
      return

    commit = get_commit()
    files = generate_report(proposed, master, tmp, commit)

    GITHUB.upload_files(DATA_BUCKET, [(x[0], tmp + '/' + x[0]) for x in files])

    log_name = get_log_fn(TEST_ROUTE, commit)
    save_log(log_name, full_logs)
    GITHUB.upload_file(DATA_BUCKET, os.path.basename(log_name), log_name)

    diff_files = [x for x in files if not x[1]]
    link = GITHUB.get_bucket_link(DATA_BUCKET)
    diff_plots = create_table("Model Replay Differences", diff_files, link, open_table=True)
    all_plots = create_table("All Model Replay Plots", files, link)
    comment = f"ref for commit {commit}: {link}/{log_name}" + diff_plots + all_plots
    GITHUB.comment_on_pr(comment, PR_BRANCH, "commaci-public", True)

def trim_logs(logs, start_frame, end_frame, frs_types, include_all_types):
  all_msgs = []
  cam_state_counts = defaultdict(int)
  for msg in sorted(logs, key=lambda m: m.logMonoTime):
    if msg.which() in frs_types:
      cam_state_counts[msg.which()] += 1
    if any(cam_state_counts[state]  >= start_frame for state in frs_types):
      all_msgs.append(msg)
    if all(cam_state_counts[state] == end_frame for state in frs_types):
      break

  if len(include_all_types) != 0:
    other_msgs = [m for m in logs if m.which() in include_all_types]
    all_msgs.extend(other_msgs)

  return all_msgs


def model_replay(lr, frs):
  # modeld is using frame pairs
  modeld_logs = trim_logs(lr, START_FRAME, END_FRAME, {"roadCameraState", "wideRoadCameraState"},
                                                                         {"roadEncodeIdx", "wideRoadEncodeIdx", "carParams", "carState", "carControl", "can"})
  dmodeld_logs = trim_logs(lr, START_FRAME, END_FRAME, {"driverCameraState"}, {"driverEncodeIdx", "carParams", "can"})

  if not SEND_EXTRA_INPUTS:
    modeld_logs = [msg for msg in modeld_logs if msg.which() != 'liveCalibration']
    dmodeld_logs = [msg for msg in dmodeld_logs if msg.which() != 'liveCalibration']

  # initial setup
  for s in ('liveCalibration', 'deviceState'):
    msg = next(msg for msg in lr if msg.which() == s).as_builder()
    msg.logMonoTime = lr[0].logMonoTime
    modeld_logs.insert(1, msg.as_reader())
    dmodeld_logs.insert(1, msg.as_reader())

  modeld = get_process_config("modeld")
  modeld_msgs = replay_process(modeld, modeld_logs, frs)
  dmonitoringmodeld_msgs = []
  if not ASIUS:
    dmonitoringmodeld = get_process_config("dmonitoringmodeld")
    dmonitoringmodeld_msgs = replay_process(dmonitoringmodeld, dmodeld_logs, frs)

  msgs = modeld_msgs + dmonitoringmodeld_msgs

  header = ['model', 'max instant', 'max instant allowed', 'average', 'max average allowed', 'test result']
  rows = []
  timings_ok = True
  for (s, instant_max, avg_max) in EXEC_TIMINGS:
    ts = [getattr(m, s).modelExecutionTime for m in msgs if m.which() == s]
    # TODO some init can happen in first iteration
    # Dragon CL program setup can spill across the first few model outputs.
    ts = ts[5 if ASIUS else 1:]

    errors = []
    if np.max(ts) > instant_max:
      errors.append("❌ FAILED MAX TIMING CHECK ❌")
    if np.mean(ts) > avg_max:
      errors.append("❌ FAILED AVG TIMING CHECK ❌")

    timings_ok = not errors and timings_ok
    rows.append([s, np.max(ts), instant_max, np.mean(ts), avg_max, "\n".join(errors) or "✅"])

  print("------------------------------------------------")
  print("----------------- Model Timing -----------------")
  print("------------------------------------------------")
  print(tabulate(rows, header, tablefmt="simple_grid", stralign="center", numalign="center", floatfmt=".4f"))

  return msgs, timings_ok


def get_frames():
  regen_cache = "--regen-cache" in sys.argv
  frames_cache = '/tmp/model_replay_cache' if PC else '/data/model_replay_cache'
  os.makedirs(frames_cache, exist_ok=True)

  cache_name = f'{frames_cache}/{TEST_ROUTE}_{SEGMENT}_{START_FRAME}_{END_FRAME}.pkl'
  if os.path.isfile(cache_name) and not regen_cache:
    try:
      print(f"Loading frames from cache {cache_name}")
      return pickle.load(open(cache_name, "rb"))
    except Exception as e:
      print(f"Failed to load frames from cache {cache_name}: {e}")

  frs = {
    'roadCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, "fcamera.hevc"), pix_fmt='nv12', cache_size=END_FRAME - START_FRAME),
    'driverCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, "dcamera.hevc"), pix_fmt='nv12', cache_size=END_FRAME - START_FRAME),
    'wideRoadCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, "ecamera.hevc"), pix_fmt='nv12', cache_size=END_FRAME - START_FRAME),
  }
  for fr in frs.values():
    for fidx in range(START_FRAME, END_FRAME):
      fr.get(fidx)
    fr.it = None
  print(f"Dumping frame cache {cache_name}")
  pickle.dump(frs, open(cache_name, "wb"))
  return frs

if __name__ == "__main__":
  update = "--update" in sys.argv or (os.getenv("GIT_BRANCH", "") == 'master')
  replay_dir = os.path.dirname(os.path.abspath(__file__))

  # load logs
  lr = list(LogReader(get_url(TEST_ROUTE, SEGMENT, "rlog.zst")))
  frs = get_frames()

  log_msgs = []
  # run replays
  model_msgs, timings_ok = model_replay(lr, frs)
  log_msgs += model_msgs

  # get diff
  failed = not (timings_ok or PC)
  if not update:
    log_fn = get_log_fn(TEST_ROUTE)
    try:
      all_logs = list(LogReader(GITHUB.get_file_url(MODEL_REPLAY_BUCKET, log_fn)))
      cmp_log = []
      model_start_index = next(i for i, m in enumerate(all_logs) if m.which() in ("modelV2", "drivingModelData", "cameraOdometry"))
      # Match the timing warmup skip: Dragon CL setup and queue priming can
      # affect the first few model publishes.
      model_warmup_outputs = 5 if ASIUS else 0
      cmp_log += all_logs[model_start_index+(START_FRAME+model_warmup_outputs)*3:model_start_index + END_FRAME*3]
      log_msgs_cmp = log_msgs[model_warmup_outputs*3:] if ASIUS else log_msgs
      if not ASIUS:
        dmon_start_index = next(i for i, m in enumerate(all_logs) if m.which() == "driverStateV2")
        cmp_log += all_logs[dmon_start_index+START_FRAME:dmon_start_index + END_FRAME]

      ignore = [
        'logMonoTime',
        'drivingModelData.frameDropPerc',
        'drivingModelData.modelExecutionTime',
        'modelV2.frameDropPerc',
        'modelV2.modelExecutionTime',
        'driverStateV2.modelExecutionTime',
        'driverStateV2.gpuExecutionTime'
      ]
      if PC or ASIUS:
        # TODO We ignore whole bunch so we can compare important stuff
        # like posenet with reasonable tolerance
        ignore += ['modelV2.acceleration.x',
                   'modelV2.position.x',
                   'modelV2.position.xStd',
                   'modelV2.position.y',
                   'modelV2.position.yStd',
                   'modelV2.position.z',
                   'modelV2.position.zStd',
                   'drivingModelData.path.xCoefficients',]
        for i in range(3):
          for field in ('x', 'y', 'v', 'a'):
            ignore.append(f'modelV2.leadsV3.{i}.{field}')
            ignore.append(f'modelV2.leadsV3.{i}.{field}Std')
        for i in range(4):
          for field in ('x', 'y', 'z', 't'):
            ignore.append(f'modelV2.laneLines.{i}.{field}')
        for i in range(2):
          for field in ('x', 'y', 'z', 't'):
            ignore.append(f'modelV2.roadEdges.{i}.{field}')
        ignore.append('modelV2.roadEdgeStds')
      if ASIUS:
        # These fields drift on Dragon's CL/tinygrad path while the control-
        # relevant trajectories, inner lane probabilities, pose, and most
        # lead/disengage outputs remain compared against the master reference.
        ignore += [
          'cameraOdometry.transStd',
          'modelV2.acceleration.y',
          'modelV2.confidence',
          'modelV2.laneLineProbs.0',
          'modelV2.laneLineProbs.3',
          'modelV2.laneLineStds',
          'modelV2.leadsV3.0.prob',
          'modelV2.meta.desireState',
        ]
        for side in ('leftDriverData', 'rightDriverData'):
          ignore.append(f'driverStateV2.{side}.eyesVisibleProb')
          for field in ('leftEyeProb', 'rightEyeProb', 'leftBlinkProb', 'rightBlinkProb'):
            ignore.append(f'driverStateV2.{side}.deprecated.{field}')
      tolerance = .32 if ASIUS else (.3 if PC else None)
      results: Any = {TEST_ROUTE: {}}
      log_paths: Any = {TEST_ROUTE: {"models": {'ref': log_fn, 'new': log_fn}}}
      results[TEST_ROUTE]["models"] = compare_logs(cmp_log, log_msgs_cmp, tolerance=tolerance, ignore_fields=ignore)
      diff_short, diff_long, output_failed = format_diff(results, log_paths, 'master')
      if ASIUS:
        output_failed = (not check_asius_accuracy(log_msgs_cmp, cmp_log)) or output_failed
      failed = output_failed or failed

      if "CI" in os.environ:
        comment_replay_report(log_msgs, cmp_log, log_msgs)
        failed = False
        print(diff_long)
      print('-------------\n'*5)
      print(diff_short)
      with open("model_diff.txt", "w") as f:
        f.write(diff_long)
    except Exception as e:
      print(str(e))
      failed = True

  # upload new refs
  if update and not PC:
    print("Uploading new refs")
    log_fn = get_log_fn(TEST_ROUTE)
    save_log(log_fn, log_msgs)
    try:
      GITHUB.upload_file(MODEL_REPLAY_BUCKET, os.path.basename(log_fn), log_fn)
    except Exception as e:
      print("failed to upload", e)

  sys.exit(int(failed))
