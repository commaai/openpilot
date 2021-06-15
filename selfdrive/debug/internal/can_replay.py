#!/usr/bin/env python3
import os
import time
from multiprocessing import Process
from tqdm import tqdm

os.environ['TESTING_CLOSET'] = '1'
os.environ['FILEREADER_CACHE'] = '1'

from common.realtime import config_realtime_process, Ratekeeper, DT_CTRL
from selfdrive.boardd.boardd import can_capnp_to_can_list
from tools.lib.logreader import LogReader

from panda import Panda
try:
  from panda_jungle import PandaJungle  # pylint: disable=import-error
except Exception:
  PandaJungle = None  # type: ignore

ROUTE = "77611a1fac303767/2020-03-24--09-50-38"
NUM_SEGS = 2 # route has 82 segments available

print("Loading log...")
CAN_MSGS = []
for i in tqdm(list(range(1, NUM_SEGS+1))):
  log_url = f"https://commadataci.blob.core.windows.net/openpilotci/{ROUTE}/{i}/rlog.bz2"
  lr = LogReader(log_url)
  CAN_MSGS += [can_capnp_to_can_list(m.can) for m in lr if m.which() == 'can']

# set both to cycle ignition
IGN_ON = int(os.getenv("ON", "0"))
IGN_OFF = int(os.getenv("OFF", "0"))

def send_thread(sender, core):
  config_realtime_process(core, 55)

  if "Jungle" in str(type(sender)):
    for i in [0, 1, 2, 3, 0xFFFF]:
      sender.can_clear(i)
    sender.set_ignition(False)
    time.sleep(5)
    sender.set_ignition(True)
    sender.set_panda_power(True)
  else:
    sender.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  sender.set_can_loopback(False)

  ignition = None
  if IGN_ON > 0 and IGN_OFF > 0:
    ignition = True
    print(f"Cycling ignition: on for {IGN_ON}s, off for {IGN_OFF}s")

  log_idx = 0
  rk = Ratekeeper(1 / DT_CTRL, print_delay_threshold=None)
  while True:
    # handle ignition cycling
    if ignition is not None:
      ign = (rk.frame*DT_CTRL) % (IGN_ON + IGN_OFF) < IGN_ON
      if ign != ignition:
        ignition = ign
        sender.set_ignition(ignition)

    snd = CAN_MSGS[log_idx]
    log_idx = (log_idx + 1) % len(CAN_MSGS)
    snd = list(filter(lambda x: x[-1] <= 2, snd))
    sender.can_send_many(snd)

    # Drain panda message buffer
    sender.can_recv()
    rk.keep_time()

def connect():
  serials = {}
  while True:
    # look for new devices
    for p in [Panda, PandaJungle]:
      if p is None:
        continue

      for s in p.list():
        if s not in serials:
          print("starting send thread for", s)
          serials[s] = Process(target=send_thread, args=(p(s), 3))
          serials[s].start()

    # try to join all send procs
    cur_serials = serials.copy()
    for s, p in cur_serials.items():
      p.join(0.01)
      if p.exitcode is not None:
        del serials[s]

    time.sleep(1)

if __name__ == "__main__":
  time.sleep(1)

  if "FLASH" in os.environ and PandaJungle is not None:
    for s in PandaJungle.list():
      PandaJungle(s).flash()

  while True:
    try:
      connect()
    except Exception:
      pass
