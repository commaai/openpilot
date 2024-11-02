#!/usr/bin/env python3
import os
import sys
from collections import defaultdict
from typing import Any
import tempfile
from itertools import zip_longest
from pathlib import Path
import time
import pickle
import numpy as np

import matplotlib.pyplot as plt

from openpilot.common.git import get_commit
from openpilot.system.hardware import PC, HARDWARE
from openpilot.tools.lib.openpilotci import get_url
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs, format_diff
from openpilot.selfdrive.test.process_replay.process_replay import get_process_config, replay_process
from openpilot.tools.lib.framereader import FrameReader, NumpyFrameReader
from openpilot.tools.lib.logreader import LogReader, save_log
from openpilot.tools.lib.github_utils import GithubUtils

TEST_ROUTE = "2f4452b03ccb98f0|2022-12-03--13-45-30"
SEGMENT = 6
MAX_FRAMES = 400 if PC else 400

NO_MODEL = "NO_MODEL" in os.environ
SEND_EXTRA_INPUTS = bool(int(os.getenv("SEND_EXTRA_INPUTS", "0")))

DATA_TOKEN = os.getenv("CI_ARTIFACTS_TOKEN","")
API_TOKEN = os.getenv("GITHUB_COMMENTS_TOKEN","")
MODEL_REPLAY_BUCKET="model_replay_master_tmp"
GITHUB = GithubUtils(API_TOKEN, DATA_TOKEN)


def get_log_fn(test_route, ref="master"):
  return f"{test_route}_model_tici_{ref}.bz2"

def plot(proposed, master, title, tmp):
  proposed = list(proposed)
  master = list(master)
  fig, ax = plt.subplots()
  ax.plot(proposed, label='PROPOSED')
  ax.plot(master, label='MASTER')
  plt.legend(loc='best')
  plt.title(title)
  plt.savefig(f'{tmp}/{title}.png')
  return (title + '.png', proposed == master)

def get_event(logs, event):
  return (getattr(m, m.which()) for m in filter(lambda m: m.which() == event, logs))

def zl(array, fill):
  return zip_longest(array, [], fillvalue=fill)

def generate_report(proposed, master, tmp, commit):
  ModelV2_Plots = zl([
                     (lambda x: x.velocity.x[0], "velocity.x"),
                     (lambda x: x.action.desiredCurvature, "desiredCurvature"),
                     (lambda x: x.leadsV3[0].x[0], "leadsV3.x"),
                     (lambda x: x.laneLines[1].y[0], "laneLines.y"),
                     (lambda x: x.meta.disengagePredictions.gasPressProbs[1], "gasPressProbs")
                    ], "modelV2")

  return [plot(map(v[0], get_event(proposed, event)), \
               map(v[0], get_event(master, event)), f"{v[1]}_{commit[:7]}", tmp) \
               for v,event in [*ModelV2_Plots]]

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
    GITHUB.comment_on_pr(comment, PR_BRANCH)

def trim_logs_to_max_frames(logs, max_frames, frs_types, include_all_types):
  all_msgs = []
  cam_state_counts = defaultdict(int)
  # keep adding messages until cam states are equal to MAX_FRAMES
  for msg in sorted(logs, key=lambda m: m.logMonoTime):
    all_msgs.append(msg)
    if msg.which() in frs_types:
      cam_state_counts[msg.which()] += 1

    if all(cam_state_counts[state] == max_frames for state in frs_types):
      break

  if len(include_all_types) != 0:
    other_msgs = [m for m in logs if m.which() in include_all_types]
    all_msgs.extend(other_msgs)

  return all_msgs


def model_replay(lr, frs):
  logs = trim_logs_to_max_frames(lr, MAX_FRAMES, {"roadCameraState", "wideRoadCameraState", "driverCameraState"},
                                                 {"roadEncodeIdx", "wideRoadEncodeIdx", "carParams", "driverEncodeIdx"})

  if not SEND_EXTRA_INPUTS:
    logs = [msg for msg in logs if msg.which() != 'liveCalibration']

  # initial setup
  for s in ('liveCalibration', 'deviceState'):
    msg = next(msg for msg in lr if msg.which() == s).as_builder()
    msg.logMonoTime = lr[0].logMonoTime
    logs.insert(1, msg.as_reader())

  modeld = get_process_config("modeld")
  dmonitoringmodeld = get_process_config("dmonitoringmodeld")

  return replay_process([modeld, dmonitoringmodeld], logs, frs)

def get_logs_and_frames(cache=False):
  TICI = os.path.isfile('/TICI')
  CACHE="/data/model_replay_cache" if TICI else '/tmp/model_replay_cache'
  Path(CACHE).mkdir(parents=True, exist_ok=True)

  LOG_CACHE = f"{CACHE}/rlog"
  if cache and os.path.isfile(LOG_CACHE):
    with open(LOG_CACHE, "rb") as f:
      lr = pickle.load(f)
  else:
    lr = list(LogReader(get_url(TEST_ROUTE, SEGMENT, "rlog.bz2")))
    with open(LOG_CACHE, "wb") as f:
      pickle.dump(lr, f)

  videos = ["fcamera.hevc", "dcamera.hevc", "ecamera.hevc"]
  if cache:
    for v in videos:
      if not os.path.isfile(f"{CACHE}/{v}"):
        os.system(f"wget {get_url(TEST_ROUTE, SEGMENT, v)} -P {CACHE}")

  cams = ["roadCameraState", "driverCameraState", "wideRoadCameraState"]
  frs = {c : FrameReader(f"{CACHE}/{v}", readahead=True) for c,v in zip(cams, videos, strict=True)}
  for k,v in frs.items():
    f = v.get(0, 401, pix_fmt="nv12")
    np.save(f'{CACHE}/pregen_{k}_0', f[1:201])
    np.save(f'{CACHE}/pregen_{k}_1', f[201:])

  frs = {c : NumpyFrameReader(f"{CACHE}/pregen_{c}") for c,v in zip(cams, videos, strict=True)}

  return lr,frs


if __name__ == "__main__":
  HARDWARE.set_power_save(False)

  update = "--update" in sys.argv or (os.getenv("GIT_BRANCH", "") == 'master')
  replay_dir = os.path.dirname(os.path.abspath(__file__))

  lr,frs = get_logs_and_frames("CI" in os.environ or True)

  log_msgs = []
  # run replays
  if not NO_MODEL:
    st = time.monotonic()
    log_msgs += model_replay(lr, frs)
    print("MODEL_REPLAY: ", time.monotonic() - st)

  # get diff
  failed = False
  if not update:
    log_fn = get_log_fn(TEST_ROUTE)
    try:
      all_logs = list(LogReader(GITHUB.get_file_url(MODEL_REPLAY_BUCKET, log_fn)))
      cmp_log = []

      # logs are ordered based on type: modelV2, drivingModelData, driverStateV2
      if not NO_MODEL:
        start_index = next(i for i, m in enumerate(all_logs) if m.which() in ("modelV2", "drivingModelData", "cameraOdometry", "driverStateV2"))
        cmp_log += all_logs[start_index:start_index + MAX_FRAMES*4]

      ignore = [
        'logMonoTime',
        'drivingModelData.frameDropPerc',
        'drivingModelData.modelExecutionTime',
        'modelV2.frameDropPerc',
        'modelV2.modelExecutionTime',
        'driverStateV2.modelExecutionTime',
        'driverStateV2.gpuExecutionTime'
      ]
      if PC:
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
      tolerance = .3 if PC else None
      results: Any = {TEST_ROUTE: {}}
      st = time.monotonic()
      log_paths: Any = {TEST_ROUTE: {"models": {'ref': log_fn, 'new': log_fn}}}
      results[TEST_ROUTE]["models"] = compare_logs(cmp_log, log_msgs, tolerance=tolerance, ignore_fields=ignore)
      print("Compare Logs: ", time.monotonic() - st)
      diff_short, diff_long, failed = format_diff(results, log_paths, 'master')
      print("Format Diff: ", time.monotonic() - st)

      if "CI" in os.environ:
        st = time.monotonic()
        comment_replay_report(log_msgs, cmp_log, log_msgs)
        print("Comment Replay Report: ", time.monotonic() - st)
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
