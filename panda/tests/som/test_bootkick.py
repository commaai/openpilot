import time
import pytest

from panda import Panda, PandaJungle

PANDA_SERIAL = "28002d000451323431333839"
JUNGLE_SERIAL = "26001c001451313236343430"

OBDC_PORT = 1

@pytest.fixture(autouse=True, scope="function")
def pj():
  jungle = PandaJungle(JUNGLE_SERIAL)
  jungle.flash()

  jungle.reset()
  jungle.set_ignition(False)

  yield jungle

  #jungle.set_panda_power(False)
  jungle.close()

@pytest.fixture(scope="function")
def p(pj):
  # note that the 3X's panda lib isn't updated, which
  # shold be fine since it only uses stable APIs
  pj.set_panda_power(True)
  assert Panda.wait_for_panda(PANDA_SERIAL, 10)
  p = Panda(PANDA_SERIAL)
  p.flash()
  p.reset()
  yield p
  p.close()

def setup_state(panda, jungle, state):
  jungle.set_panda_power(0)

  if state == "off":
    wait_for_full_poweroff(jungle)
  elif state == "normal boot":
    jungle.set_panda_individual_power(OBDC_PORT, 1)
  elif state == "QDL":
    time.sleep(0.5)
    jungle.set_panda_individual_power(OBDC_PORT, 1)
  elif state == "ready to bootkick":
    wait_for_full_poweroff(jungle)
    jungle.set_panda_individual_power(OBDC_PORT, 1)
    wait_for_boot(panda, jungle)
    set_som_shutdown_flag(panda)
    panda.set_safety_mode(Panda.SAFETY_SILENT)
    panda.send_heartbeat()
    wait_for_som_shutdown(panda, jungle)
  else:
    raise ValueError(f"unkown state: {state}")


def wait_for_som_shutdown(panda, jungle):
  st = time.monotonic()
  while panda.read_som_gpio():
    # can take a while for the SOM to fully shutdown
    if time.monotonic() - st > 120:
      raise Exception("SOM didn't shutdown in time")
    if check_som_boot_flag(panda):
      raise Exception(f"SOM rebooted instead of shutdown: {time.monotonic() - st}s")
    time.sleep(0.5)
    dt = time.monotonic() - st
    print("waiting for shutdown", round(dt))
  dt = time.monotonic() - st
  print(f"took {dt:.2f}s for SOM to shutdown")

def wait_for_full_poweroff(jungle, timeout=30):
  st = time.monotonic()

  time.sleep(15)
  while PANDA_SERIAL in Panda.list():
    if time.monotonic() - st > timeout:
      raise Exception("took too long for device to turn off")

  health = jungle.health()
  assert all(health[f"ch{i}_power"] < 0.1 for i in range(1, 7))

def check_som_boot_flag(panda):
  h = panda.health()
  return h['safety_mode'] == Panda.SAFETY_ELM327 and h['safety_param'] == 30

def set_som_shutdown_flag(panda):
  panda.set_can_data_speed_kbps(0, 1000)

def wait_for_boot(panda, jungle, reset_expected=False, bootkick=False, timeout=120):
  st = time.monotonic()

  Panda.wait_for_panda(PANDA_SERIAL, timeout)
  panda.reconnect()
  if bootkick:
    assert panda.health()['uptime'] > 20
  else:
    assert panda.health()['uptime'] < 3

  for i in range(3):
    assert not check_som_boot_flag(panda)
    time.sleep(1)

  # wait for SOM to bootup
  while not check_som_boot_flag(panda):
    if time.monotonic() - st > timeout:
      raise Exception("SOM didn't boot in time")
    time.sleep(1.0)

  assert panda.health()['som_reset_triggered'] == reset_expected

def test_cold_boot(p, pj):
  setup_state(p, pj, "off")
  setup_state(p, pj, "normal boot")
  wait_for_boot(p, pj)

def test_bootkick_ignition_line(p, pj):
  setup_state(p, pj, "ready to bootkick")
  pj.set_ignition(True)
  wait_for_boot(p, pj, bootkick=True)

@pytest.mark.skip("test isn't reliable yet")
def test_bootkick_can_ignition(p, pj):
  setup_state(p, pj, "ready to bootkick")
  for _ in range(10):
    # Mazda ignition signal
    pj.can_send(0x9E, b'\xc0\x00\x00\x00\x00\x00\x00\x00', 0)
    time.sleep(0.5)
  wait_for_boot(p, pj, bootkick=True)

def test_recovery_from_qdl(p, pj):
  setup_state(p, pj, "ready to bootkick")

  # put into QDL using the FORCE_USB_BOOT pin
  for i in range(10):
    pj.set_header_pin(i, 1)

  # try to boot
  time.sleep(1)
  pj.set_ignition(True)
  time.sleep(3)

  # release FORCE_USB_BOOT
  for i in range(10):
    pj.set_header_pin(i, 0)

  # normally, this GPIO is set immediately since it's first enabled in the ABL
  for i in range(17):
    assert not p.read_som_gpio()
    time.sleep(1)

  # should boot after 45s
  wait_for_boot(p, pj, reset_expected=True, bootkick=True, timeout=120)
