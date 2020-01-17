#!/usr/bin/env python3
import zmq

import cereal.messaging as messaging
from cereal.services import service_list

if __name__ == "__main__":
  poller = zmq.Poller()

  fsock = messaging.sub_sock("frame", poller)
  msock = messaging.sub_sock("model", poller)

  frmTimes = {}
  proc = []

  last100 = []

  while 1:
    polld = poller.poll(timeout=1000)
    for sock, mode in polld:
      if mode != zmq.POLLIN:
        continue
      if sock == fsock:
        f = messaging.recv_one(sock)
        frmTimes[f.frame.frameId] = f.frame.timestampEof
      else:
        proc.append(messaging.recv_one(sock))
        nproc = []
        for mm in proc:
          fid = mm.model.frameId

          if fid in frmTimes:
            tm = (mm.logMonoTime-frmTimes[fid])/1e6
            del frmTimes[fid]
            last100.append(tm)
            last100 = last100[-100:]
            print("%10d: %.2f ms    min: %.2f  max: %.2f" %  (fid, tm, min(last100), max(last100)))
          else:
            nproc.append(mm)
        proc = nproc

