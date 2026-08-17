import time
import pytest

from panda import Panda

pytestmark = [
  pytest.mark.test_panda_types(Panda.INTERNAL_DEVICES),
]

MAX_RPM = 5000

@pytest.mark.timeout(2*60)
def test_fan_curve(p):
  # ensure fan curve is (roughly) linear

  rpms = []
  for power in (30, 70, 100):
    # wait until fan spins up
    p.set_fan_power(power)
    for _ in range(20):
      time.sleep(1)
      if p.get_fan_rpm() > 1000:
        break
    time.sleep(2)  # wait for RPM to converge
    rpms.append(p.get_fan_rpm())

  print(rpms)
  diffs = [b - a for a, b in zip(rpms, rpms[1:])]
  assert all(x > 0 for x in diffs), f"Fan RPMs not strictly increasing: {rpms=}"
  assert rpms[-1] > (0.75*MAX_RPM)


def test_fan_cooldown(p):
  # if the fan cooldown doesn't work, we get high frequency noise on the tach line
  # while the rotor spins down. this makes sure it never goes beyond the expected max RPM
  p.set_fan_power(100)
  time.sleep(3)
  p.set_fan_power(0)
  for _ in range(5):
    assert p.get_fan_rpm() <= MAX_RPM*2
    time.sleep(0.5)
