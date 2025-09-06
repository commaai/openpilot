import os
import time

from cereal import car
from cereal.messaging import SubMaster, PubMaster
import cereal.messaging as messaging
from panda import Panda
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.pandad.state_manager import PandaStateManager
from openpilot.selfdrive.pandad.peripheral import PeripheralManager
from openpilot.selfdrive.pandad.safety import PandaSafetyManager

FAKE_SEND = os.getenv("FAKESEND") == "1"

class PandaRunner:
  def __init__(self, serials, pandas):
    self.pandas = pandas
    self.usb_pandas = {p.get_usb_serial() for p in pandas if not p.spi}
    self.hw_types = [int.from_bytes(p.get_type(), 'big') for p in pandas]

    for panda in self.pandas:
      if os.getenv("BOARDD_LOOPBACK"):
        panda.set_can_loopback(True)
      for i in range(3):
        panda.set_canfd_auto(i, True)

    self.sm = SubMaster(["selfdriveState", "deviceState", "driverCameraState"])
    self.pm = PubMaster(["can", "pandaStates", "peripheralState"])
    self.sendcan_sock = messaging.sub_sock('sendcan', timeout=10)
    self.sendcan_buffer = None

    self.state_mgr = PandaStateManager(pandas, self.hw_types)
    self.periph_mgr = PeripheralManager(pandas[0], self.hw_types[0])
    self.safety_mgr = PandaSafetyManager(pandas)

  def _can_send(self):
    # TODO: this needs to have a strict timeout of <10ms and handle NACKs well (buffer the data)
    while (msg := messaging.recv_one_or_none(self.sendcan_sock)):
      # drop msg if too old
      if (time.monotonic_ns() - msg.logMonoTime) / 1e9 > 1.0:
        cloudlog.warning("skipping CAN send, too old")
        continue

      # Group CAN messages by panda based on bus offset
      panda_msgs = [[] for _ in self.pandas]
      for c in msg.sendcan:
        panda_idx = c.src // 4  # Each panda handles 4 buses
        if panda_idx < len(self.pandas):
          # Adjust bus number for the panda (remove offset)
          adjusted_bus = c.src % 4
          panda_msgs[panda_idx].append((c.address, c.dat, adjusted_bus))

      # Send messages to each panda
      for panda_idx, can_msgs in enumerate(panda_msgs):
        if can_msgs:
          self.pandas[panda_idx].can_send_many(can_msgs)

  def _can_recv(self):
    cans = []
    for panda_idx, p in enumerate(self.pandas):
      bus_offset = panda_idx * 4  # Each panda gets 4 buses
      for address, dat, src in p.can_recv():
        if src >= 192:  # Rejected message
          base_bus = src - 192
          adjusted_src = base_bus + bus_offset + 192
        elif src >= 128:  # Returned message
          base_bus = src - 128
          adjusted_src = base_bus + bus_offset + 128
        else:  # Normal message
          adjusted_src = src + bus_offset
        cans.append((address, dat, adjusted_src))

    msg = messaging.new_message('can', len(cans) if cans else 0)
    msg.valid = True
    if cans:
      for i, can_info in enumerate(cans):
        can = msg.can[i]
        can.address, can.dat, can.src = can_info
    self.pm.send("can", msg)

  def run(self, evt):
    rk = Ratekeeper(100, print_delay_threshold=None)
    engaged = False

    try:
      while not evt.is_set():
        # receive CAN messages
        self._can_recv()

        # send CAN messages
        self._can_send()

        # Process peripheral state at 20 Hz
        if rk.frame % 5 == 0:
          self.sm.update(0)
          engaged = self.sm.all_checks() and self.sm["selfdriveState"].enabled
          self.periph_mgr.process(self.sm)

        # Process panda state at 10 Hz
        if rk.frame % 10 == 0:
          ignition = self.state_mgr.process(engaged, self.pm)
          if not ignition and rk.frame % 100 == 0:
            if set(Panda.list(usb_only=True)) != self.usb_pandas:
              cloudlog.warning("Reconnecting to new panda")
              evt.set()
              break

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

      # Close relay on exit to prevent a fault
      is_onroad = Params().get_bool("IsOnroad")
      if is_onroad and not engaged:
        for p in self.pandas:
          p.set_safety_mode(car.CarParams.SafetyModel.noOutput)
