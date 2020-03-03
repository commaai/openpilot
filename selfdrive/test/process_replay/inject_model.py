#!/usr/bin/env python3

import os
import time


from tqdm import tqdm
from cereal.messaging import PubMaster, recv_one, sub_sock
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader
import subprocess



def inject_model(segment_name):
  frame_reader = FrameReader('cd:/'+segment_name.replace("|", "/") + "/fcamera.hevc")
  msgs = list(LogReader('cd:/'+segment_name.replace("|", "/") + "/rlog.bz2"))

  import selfdrive.camerad
  dir_path = os.path.dirname(selfdrive.camerad.__file__)
  os.chdir(dir_path)
  camerad_thread = subprocess.Popen('exec ./camerad', shell=True)
  time.sleep(2)
  import selfdrive.modeld
  dir_path = os.path.dirname(selfdrive.modeld.__file__)
  os.chdir(dir_path)
  modeld_thread = subprocess.Popen('exec ./modeld', shell=True)
  # TODO do better than just wait for modeld to boot
  time.sleep(5)

  pm = PubMaster(['liveCalibration', 'frame'])
  model_sock = sub_sock('model')
  frame_id_lookup = {}

  # Read encodeIdx
  for msg in msgs:
    if msg.which() == 'encodeIdx':
      frame_id_lookup[msg.encodeIdx.frameId] = (msg.encodeIdx.segmentNum, msg.encodeIdx.segmentId)

  # Send some livecalibration messages to initalize visiond
  for msg in msgs:
    if msg.which() == 'liveCalibration':
      pm.send('liveCalibration', msg.as_builder())

  time.sleep(1.0)

  out_msgs = []
  fidx = 0
  for msg in tqdm(msgs):
    w = msg.which()


    if w == 'frame':
      msg = msg.as_builder()

      img = frame_reader.get(fidx, pix_fmt="rgb24")[0][:,::-1]

      msg.frame.image = img.flatten().tobytes()

      pm.send(w, msg)
      model = recv_one(model_sock)
      fidx += 1

      model = model.as_builder()
      model.logMonoTime = msg.logMonoTime
      model = model.as_reader()
      out_msgs.append(model)
    elif w == 'liveCalibration':
      pm.send(w, msg.as_builder())
      out_msgs.append(msg)
    else:
      out_msgs.append(msg)
  modeld_thread.kill()
  time.sleep(1)
  camerad_thread.kill()
  return out_msgs



if __name__ == "__main__":
  inject_model("0375fdf7b1ce594d|2019-06-13--08-32-25/3")
