#!/usr/bin/env python3

import time

from tqdm import tqdm

from selfdrive.manager.process_config import managed_processes
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

    if w == 'roadCameraState':
      msg = msg.as_builder()

      img = frame_reader.get(fidx, pix_fmt="rgb24")[0][:,:,::-1]

      msg.roadCameraState.image = img.flatten().tobytes()

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

  managed_processes['camerad'].start()
  managed_processes['modeld'].start()
  # TODO do better than just wait for modeld to boot
  time.sleep(5)

  pm = PubMaster(['liveCalibration', 'roadCameraState'])
  model_sock = sub_sock('model')
  try:
    out_msgs = regen_model(msgs, pm, frame_reader, model_sock)
  except (KeyboardInterrupt, SystemExit, Exception) as e:
    managed_processes['modeld'].stop()
    time.sleep(2)
    managed_processes['camerad'].stop()
    raise e
  managed_processes['modeld'].stop()
  time.sleep(2)
  managed_processes['camerad'].stop()

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
