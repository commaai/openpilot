#!/usr/bin/env python3
# type: ignore
import cereal.messaging as messaging

all_sockets = ['roadCameraState', 'driverCameraState', 'wideRoadCameraState']
prev_id = [None,None,None]
this_id = [None,None,None]
dt = [None,None,None]
num_skipped = [0,0,0]

if __name__ == "__main__":
  sm = messaging.SubMaster(all_sockets)
  while True:
    sm.update()

    for i in range(len(all_sockets)):
      if not sm.updated[all_sockets[i]]:
        continue
      this_id[i] = sm[all_sockets[i]].frameId
      if prev_id[i] is None:
        prev_id[i] = this_id[i]
        continue
      dt[i] = this_id[i] - prev_id[i]
      if dt[i] != 1:
        num_skipped[i] += dt[i] - 1
        print(all_sockets[i] ,dt[i] - 1, num_skipped[i])
      prev_id[i] = this_id[i]
