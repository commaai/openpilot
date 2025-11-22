#!/usr/bin/env python3
import concurrent.futures

from panda import PandaJungle, PandaJungleDFU, McuType
from panda.tests.libs.resetter import Resetter

SERIALS = {
  '180019001451313236343430',  # jungle v2
}

def recover(s):
  with PandaJungleDFU(s) as pd:
    pd.recover()

def flash(s):
  with PandaJungle(s) as p:
    p.flash()
    return p.get_mcu_type()

# Reset + flash all CI hardware to get it into a consistent state
# * port 1: jungles-under-test
# * port 2: USB hubs
# * port 3: HITL pandas and their jungles
if __name__ == "__main__":
  with Resetter() as r:
    # everything off
    for i in range(1, 4):
      r.enable_power(i, 0)
    r.cycle_power(ports=[1, 2], dfu=True)
    for s in SERIALS:
      assert PandaJungle.wait_for_dfu(PandaJungleDFU.st_serial_to_dfu_serial(s, McuType.H7), timeout=10)
    dfu_serials = PandaJungleDFU.list()
    assert len(dfu_serials) == len(SERIALS)

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(dfu_serials)) as exc:
      list(exc.map(recover, dfu_serials, timeout=30))

      # power cycle for H7 bootloader bug
      r.cycle_power(ports=[1, 2])

      # wait for them to come back up
      for s in SERIALS:
        assert PandaJungle.wait_for_panda(s, timeout=10)
      assert set(PandaJungle.list()) >= SERIALS
      mcu_types = list(exc.map(flash, SERIALS, timeout=20))
      assert set(mcu_types) == {McuType.H7, }
