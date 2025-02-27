import os
import threading
import time
from itertools import chain
from panda import Panda
from panda.python import PANDA_BUS_CNT
from cereal import car
from cereal.messaging import SubMaster, PubMaster
import cereal.messaging as messaging
from openpilot.common.params import Params
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
    self.hw_types = [int.from_bytes(p.get_type(), 'big') for p in pandas]
    for panda in self.pandas:
      if os.getenv("BOARDD_LOOPBACK"):
        panda.set_can_loopback(True)
      for i in range(PANDA_BUS_CNT):
        panda.set_canfd_auto(i, True)

    self.sm = SubMaster(["selfdriveState", "deviceState", "driverCameraState"])
    self.pm = PubMaster(["can", "pandaStates", "peripheralState"])

    self.lock = threading.Lock()
    self.state_mgr = PandaStateManager(pandas, self.hw_types, self.lock)
    self.periph_mgr = PeripheralManager(pandas[0], self.hw_types[0], self.lock)
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
        elif age >= 1:
          cloudlog.error(f"Dropping stale sendcan message, age: {age:.2f}s")


  def _can_recv(self):
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
    thread = threading.Thread(target=self._can_send, args=(evt,))
    thread.start()

    rk = Ratekeeper(100, print_delay_threshold=None)
    engaged = False

    try:
      while not evt.is_set():
        self._can_recv()

        # Process peripheral state at 20 Hz
        if rk.frame % 5 == 0:
          self.sm.update(0)
          engaged = self.sm.all_checks() and self.sm["selfdriveState"].enabled
          self.periph_mgr.process(self.sm)

        # Process panda state at 10 Hz
        if rk.frame % 10 == 0:
          ignition = self.state_mgr.process(engaged, self.pm)
          if not ignition:
            with self.lock:
              current_serials = set(Panda.list())
            if current_serials != self.serials:
              cloudlog.warning("Reconnecting to new panda")
              evt.set()

          self.safety_mgr.configure_safety_mode()

        # Send out peripheralState at 2Hz
        if rk.frame % 50 == 0:
          self.periph_mgr.send_state(self.pm)

        rk.keep_time()
    except Exception as e:
      cloudlog.error(f"Exception in main loop: {e}")
    finally:
      evt.set()
      self.periph_mgr.cleanup()
      thread.join()

      # Close relay on exit to prevent a fault
      is_onroad = Params().get_bool("IsOnroad")
      if is_onroad and not engaged:
        for p in self.pandas:
          p.set_safety_mode(car.CarParams.SafetyModel.noOutput)
