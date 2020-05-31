#!/usr/bin/env python3

import time

from tqdm import tqdm

import selfdrive.manager as manager
from cereal.messaging import PubMaster, recv_one, sub_sock
from tools.lib.framereader import FrameReader


def rreplace(s, old, new, occurrence):
  li = s.rsplit(old, occurrence)
  return new.join(li)


def regen_model(msgs, pm, frame_reader, model_sock):
  # Send some livecalibration messages to initalize visiond
  for msg in msgs:
    if msg.which() == 'liveCalibration':
      pm.send('liveCalibration', msg.as_builder())

  out_msgs = []
  fidx = 0
  for msg in tqdm(msgs):
    w = msg.which()

    if w == 'frame':
      msg = msg.as_builder()

      img = frame_reader.get(fidx, pix_fmt="rgb24")[0][:, ::-1]

      msg.frame.image = img.flatten().tobytes()

      pm.send(w, msg)
      model = recv_one(model_sock)
      fidx += 1
      out_msgs.append(model)
    elif w == 'liveCalibration':
      pm.send(w, msg.as_builder())

  return out_msgs


def inject_model(msgs, segment_name):
  if segment_name.count('--') == 2:
    segment_name = rreplace(segment_name, '--', '/', 1)
  frame_reader = FrameReader('cd:/'+segment_name.replace("|", "/") + "/fcamera.hevc")

  manager.start_managed_process('camerad')
  manager.start_managed_process('modeld')
  # TODO do better than just wait for modeld to boot
  time.sleep(5)

  pm = PubMaster(['liveCalibration', 'frame'])
  model_sock = sub_sock('model')
  try:
    out_msgs = regen_model(msgs, pm, frame_reader, model_sock)
  except (KeyboardInterrupt, SystemExit, Exception) as e:
    manager.kill_managed_process('modeld')
    time.sleep(2)
    manager.kill_managed_process('camerad')
    raise e
  manager.kill_managed_process('modeld')
  time.sleep(2)
  manager.kill_managed_process('camerad')


  new_msgs = []
  midx = 0
  for msg in msgs:
    if (msg.which() == 'model') and (midx < len(out_msgs)):
      model = out_msgs[midx].as_builder()
      model.logMonoTime = msg.logMonoTime
      model = model.as_reader()
      new_msgs.append(model)
      midx += 1
    else:
      new_msgs.append(msg)

  print(len(new_msgs), len(list(msgs)))
  assert abs(len(new_msgs) - len(list(msgs))) < 2

  return new_msgs
