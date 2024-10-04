#!/usr/bin/env python3
import os
import sys
from collections import defaultdict
from typing import Any
import tempfile
from itertools import zip_longest

import requests
#import matplotlib
#matplotlib.use('inline')
import matplotlib.pyplot as plt

from openpilot.common.git import get_commit
from openpilot.common.run import run_cmd
from openpilot.system.hardware import PC
from openpilot.tools.lib.openpilotci import BASE_URL, get_url
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs, format_diff
from openpilot.selfdrive.test.process_replay.process_replay import get_process_config, replay_process
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader, save_log

TEST_ROUTE = "2f4452b03ccb98f0|2022-12-03--13-45-30"
SEGMENT = 6
MAX_FRAMES = 100 if PC else 600

NO_MODEL = "NO_MODEL" in os.environ
SEND_EXTRA_INPUTS = bool(int(os.getenv("SEND_EXTRA_INPUTS", "0")))

UPDATE_REF=f"model_replay_{os.environ['GIT_BRANCH']}"


def get_log_fn(test_route):
  return f"{test_route}_model_tici_master.bz2"

def plot(proposed, master, title, tmp):
  fig, ax = plt.subplots()
  ax.plot(list(proposed), label='PROPOSED')
  ax.plot(list(master), label='MASTER')
  plt.legend(loc='best')
  plt.title(title)
  plt.savefig(f'{tmp}/{title}.png')
  return title

def get_event(logs, event):
  return (getattr(m, m.which()) for m in filter(lambda m: m.which() == event, logs))

def zl(array, fill):
  return zip_longest(array, [], fillvalue=fill)

def generate_report(proposed, master, tmp):
  ModelV2_Plots = zl([
                     (lambda x: x.action.desiredCurvature, "title1"),
                     (lambda x: x.meta.disengagePredictions.gasPressProbs[1], "title2"),
                     (lambda x: x.velocity.x[0], "title3"),
                     (lambda x: x.action.desiredCurvature, "title4"),
                     (lambda x: x.leadsV3[0].x[0], "title5")
                    ], "modelV2")

  return [plot(map(v[0], get_event(proposed, event)), \
               map(v[0], get_event(master, event)), v[1], tmp) \
               for v,event in [*ModelV2_Plots]]

def comment_replay_report(proposed, master):
  with tempfile.TemporaryDirectory() as tmp:
    GIT_BRANCH=f"model_replay_{os.environ['GIT_BRANCH']}"
    GIT_PATH=tmp
    GIT_TOKEN=os.environ['GIT_TOKEN']
    API_ROUTE="https://api.github.com/repos/commaai/openpilot"

    run_cmd(["git", "clone", "--depth=1", "-b", "master", "https://github.com/commaai/ci-artifacts", tmp])

    # create report
    files = generate_report(proposed, master, tmp)

    # save report
    run_cmd(["git", "-C", GIT_PATH, "checkout", "-b", GIT_BRANCH])
    run_cmd(["git", "-C", GIT_PATH, "add", "."])
    run_cmd(["git", "-C", GIT_PATH, "commit", "-m", "model replay artifacts"])
    run_cmd(["git", "-C", GIT_PATH, "push", "-f", f"https://commaci-public:{GIT_TOKEN}@github.com/commaai/ci-artifacts", GIT_BRANCH])

    headers = {"Authorization": f"token {GIT_TOKEN}", "Accept": "application/vnd.github+json"}
    comment = f'{"body": "<img src=\\"https://raw.githubusercontent.com/commaai/ci-artifacts/{GIT_BRANCH}/{files[0]}.png\\">"}'

    # get PR number
    r = requests.get(f'{API_ROUTE}/commits/{GIT_BRANCH}/pulls', headers=headers)
    assert r.ok, r.status_code
    pr_number = r.json()[0]['number']

    # comment on PR
    r = requests.get(f'{API_ROUTE}/issues/{pr_number}/comments', headers=headers)
    assert r.ok, r.status_code
    comments = [x['id'] for x in r.json() if x['user']['login'] == 'commaci-public']
    if comments:
      r = requests.patch(f'{API_ROUTE}/issues/comments/{comments[0]}', headers=headers, data=comment)
    else:
      r = requests.post(f'{API_ROUTE}/issues/{pr_number}/comments', headers=headers, data=comment)
    assert r.ok, r.status_code

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
  # modeld is using frame pairs
  modeld_logs = trim_logs_to_max_frames(lr, MAX_FRAMES, {"roadCameraState", "wideRoadCameraState"}, {"roadEncodeIdx", "wideRoadEncodeIdx", "carParams"})
  dmodeld_logs = trim_logs_to_max_frames(lr, MAX_FRAMES, {"driverCameraState"}, {"driverEncodeIdx", "carParams"})

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
  dmonitoringmodeld = get_process_config("dmonitoringmodeld")

  modeld_msgs = replay_process(modeld, modeld_logs, frs)
  dmonitoringmodeld_msgs = replay_process(dmonitoringmodeld, dmodeld_logs, frs)
  return modeld_msgs + dmonitoringmodeld_msgs


if __name__ == "__main__":
  update = "--update" in sys.argv or (os.getenv("GIT_BRANCH", "") == 'master')
  replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(replay_dir, "model_replay_ref_commit")

  # load logs
  lr = list(LogReader(get_url(TEST_ROUTE, SEGMENT, "rlog.bz2")))
  frs = {
    'roadCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, "fcamera.hevc"), readahead=True),
    'driverCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, "dcamera.hevc"), readahead=True),
    'wideRoadCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, "ecamera.hevc"), readahead=True)
  }

  log_msgs = []
  # run replays
  if not NO_MODEL:
    log_msgs += model_replay(lr, frs)

  # get diff
  failed = False
  if not update:
    log_fn = get_log_fn(TEST_ROUTE)
    try:
      all_logs = list(LogReader(BASE_URL + log_fn))
      cmp_log = []

      # logs are ordered based on type: modelV2, drivingModelData, driverStateV2
      if not NO_MODEL:
        model_start_index = next(i for i, m in enumerate(all_logs) if m.which() in ("modelV2", "drivingModelData", "cameraOdometry"))
        cmp_log += all_logs[model_start_index:model_start_index + MAX_FRAMES*3]
        dmon_start_index = next(i for i, m in enumerate(all_logs) if m.which() == "driverStateV2")
        cmp_log += all_logs[dmon_start_index:dmon_start_index + MAX_FRAMES]

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
      log_paths: Any = {TEST_ROUTE: {"models": {'ref': BASE_URL + log_fn, 'new': log_fn}}}
      results[TEST_ROUTE]["models"] = compare_logs(cmp_log, log_msgs, tolerance=tolerance, ignore_fields=ignore)
      diff_short, diff_long, failed = format_diff(results, log_paths, ref_commit)

      if "CI" in os.environ:
        if not PC:
          comment_replay_report(log_msgs, cmp_log)
          quit()
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
    from openpilot.tools.lib.openpilotci import upload_file

    print("Uploading new refs")

    log_fn = get_log_fn(TEST_ROUTE)
    save_log(log_fn, log_msgs)
    try:
      upload_file(log_fn, os.path.basename(log_fn), overwrite=True)
    except Exception as e:
      print("failed to upload", e)

  sys.exit(int(failed))
