#!/usr/bin/env python3
import time
from tqdm import tqdm

import cereal.messaging as messaging
import selfdrive.manager as manager
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

def camera_replay(lr, fr):

  pm = messaging.PubMaster(['frame', 'liveCalibration'])
  sm = messaging.SubMaster(['model'])

  # TODO: add dmonitoringmodeld
  print("preparing procs")
  manager.prepare_managed_process("camerad")
  manager.prepare_managed_process("modeld")
  try:
    print("starting procs")
    manager.start_managed_process("camerad")
    manager.start_managed_process("modeld")
    time.sleep(5)
    print("procs started")

    cal = [msg for msg in lr if msg.which() == "liveCalibration"]
    for msg in cal[:5]:
      pm.send(msg.which(), msg.as_builder())

    log_msgs = []
    frame_idx = 0
    for msg in tqdm(lr):
      if msg.which() == "liveCalibrationd":
        print("sending ", msg.which())
        pm.send(msg.which(), msg.as_builder())
      elif msg.which() == "frame":
        print("sending ", msg.which())
        # recv one
        f = msg.as_builder()
        img = fr.get(frame_idx, pix_fmt="rgb24")[0][:, ::, -1]
        f.frame.image = img.flatten().tobytes()
        frame_idx += 1

        pm.send(msg.which(), f)
        print("wait for odel packet ")
        log_msgs.append(messaging.recv_one(sm.sock['model']))
        print("got model packet")

      if frame_idx >= fr.frame_count:
        break
  except KeyboardInterrupt:
    pass

  print("replay done")
  manager.kill_managed_process('modeld')
  time.sleep(2)
  manager.kill_managed_process('camerad')
  return log_msgs

if __name__ == "__main__":
  lr = LogReader("77611a1fac303767_2020-05-11--16-37-07--0--rlog.bz2")
  fr = FrameReader("77611a1fac303767_2020-05-11--16-37-07--0--fcamera.hevc")
  print(camera_replay(list(lr), fr))

