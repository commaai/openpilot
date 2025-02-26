import os
import threading
import time
from itertools import chain
from panda import Panda
from cereal.messaging import SubMaster, PubMaster
import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.pandad.panda.state_manager import PandaStateManager
from openpilot.selfdrive.pandad.panda.peripheral import PeripheralManager
from openpilot.selfdrive.pandad.panda.safety import PandaSafetyManager

FAKE_SEND = os.getenv("FAKESEND") == "1"

class PandaRunner:
  def __init__(self, serials, pandas):
    self.pandas = pandas
    self.serials = set(serials)
    self.sm = SubMaster(["selfdriveState", "deviceState", "driverCameraState"])
    self.pm = PubMaster(["can", "pandaStates", "peripheralState"])
    self.lock = threading.Lock()
    self.state_mgr = PandaStateManager(pandas, self.lock)
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
        for i, can_info in enumerate(cans):
          can = msg.can[i]
          can.address, can.dat, can.src = can_info
        self.pm.send("can", msg)

  def run(self, evt):
    threads = [
      threading.Thread(target=self._can_send, args=(evt,)),
      threading.Thread(target=self._can_recv, args=(evt,)),
    ]
    for t in threads:
      t.start()

    rk = Ratekeeper(20)
    try:
      while not evt.is_set():
        self.sm.update(0)
        engaged = self.sm.all_checks() and self.sm["selfdriveState"].enabled

        self.periph_mgr.process(self.sm)

        # Process panda state at 10 Hz
        if not rk.frame % 2:
          ignition = self.state_mgr.process(engaged, self.pm)
          if not ignition:
            with self.lock:
              current_serials = set(Panda.list())
            if current_serials != self.serials:
              cloudlog.warning("Reconnecting to new panda")
              evt.set()

          self.safety_mgr.configure_safety_mode();

        # Send out peripheralState at 2Hz
        if not rk.frame % 10:
          self.periph_mgr.send_state(self.pm)

        rk.keep_time()
    except Exception as e:
      cloudlog.error(f"Exception in main loop: {e}")
    finally:
      evt.set()
      for t in threads:
        t.join()
