import os
import threading
import time
from itertools import chain
from cereal.messaging import SubMaster, PubMaster
import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.pandad.panda.state_manager import PandaStateManager
from openpilot.selfdrive.pandad.panda.peripheral import PeripheralManager
from openpilot.selfdrive.pandad.panda.safety import PandaSafetyManager

FAKE_SEND = os.getenv("FAKESEND") == "1"

class PandaRunner:
  def __init__(self, serials, pandas):
    self.pandas = pandas
    self.sm = SubMaster(["selfdriveState", "deviceState", "driverCameraState"])
    self.pm = PubMaster(["can", "pandaStates", "peripheralState"])
    self.lock = threading.Lock()
    self.state_mgr = PandaStateManager(pandas, serials, self.lock)
    self.periph_mgr = PeripheralManager(pandas, self.lock)
    self.safety_mgr = PandaSafetyManager(pandas, self.lock)

  def _can_send(self, evt):
    sock = messaging.sub_sock('sendcan', timeout=100)
    while not evt.is_set():
      data = sock.receive()
      if data:
        msg = messaging.log_from_bytes(data)
        cans = [(c.address, c.dat, c.src) for c in msg.sendcan]
        age = (time.monotonic_ns() - msg.logMonoTime) / 1e9
        if age < 1 and not FAKE_SEND:
          with self.lock:
            for p in self.pandas:
              p.can_send_many(cans)

  def _can_recv(self, evt):
    while not evt.is_set():
      with self.lock:
        cans = list(chain.from_iterable(p.can_recv() for p in self.pandas))

      if cans:
        msg = messaging.new_message('can', len(cans))
        msg.valid = True
        for i, (addr, dat, src) in enumerate(cans):
          can = msg.can[i]
          can.address = addr
          can.dat = dat
          can.src = src
        self.pm.send("can", msg)

  def run(self, evt):
    threads = [
      threading.Thread(target=self._can_send, args=(evt,)),
      threading.Thread(target=self._can_recv, args=(evt,)),
    ]
    for t in threads:
      t.start()

    rk = Ratekeeper(20)
    while not evt.is_set():
      self.sm.update(0)
      engaged = self.sm.all_checks() and self.sm["selfdriveState"].enabled

      self.periph_mgr.process(self.sm)

      # Process panda state at 10 Hz
      if not rk.frame % 2:
        self.state_mgr.process(engaged, self.pm)
        self.safety_mgr.configure_safety_mode();

      # Send out peripheralState at 2Hz
      if not rk.frame % 10:
        self.periph_mgr.send_state(self.pm)

      rk.keep_time()

    evt.set()
    for t in threads:
      t.join()
