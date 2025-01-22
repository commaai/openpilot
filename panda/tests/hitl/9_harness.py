import time
import pytest
import itertools

from panda import Panda
from panda.tests.hitl.conftest import PandaGroup

# TODO: test relay

@pytest.mark.panda_expect_can_error
@pytest.mark.test_panda_types(PandaGroup.GEN2)
def test_harness_status(p, panda_jungle):
  # map from jungle orientations to panda orientations
  orientation_map = {
    Panda.HARNESS_STATUS_NC: Panda.HARNESS_STATUS_NC,
  }

  # this shouldn't be parameterized since we don't want the panda to be reset
  # between the tests.
  for ignition, orientation in itertools.product([True, False], [Panda.HARNESS_STATUS_NC, Panda.HARNESS_STATUS_NORMAL, Panda.HARNESS_STATUS_FLIPPED]):
    print()
    p.set_safety_mode(Panda.SAFETY_ELM327)
    panda_jungle.set_harness_orientation(orientation)
    panda_jungle.set_ignition(ignition)

    # wait for orientation detection
    time.sleep(0.25)

    health = p.health()
    detected_orientation = health['car_harness_status']
    print(f"orientation set: {orientation} detected: {detected_orientation}")

    if detected_orientation not in orientation_map:
      assert detected_orientation != Panda.HARNESS_STATUS_NC
      other = {Panda.HARNESS_STATUS_NORMAL: Panda.HARNESS_STATUS_FLIPPED, Panda.HARNESS_STATUS_FLIPPED: Panda.HARNESS_STATUS_NORMAL}
      orientation_map.update({
        orientation: detected_orientation,
        other[orientation]: other[detected_orientation],
      })

    # Orientation
    assert orientation_map[detected_orientation] == orientation

    # Line ignition
    assert health['ignition_line'] == (False if orientation == Panda.HARNESS_STATUS_NC else ignition)

    # CAN traffic
    if orientation != Panda.HARNESS_STATUS_NC:
      for bus in range(3):
        panda_jungle.can_send(0x123, f"{bus}".encode(), bus)
      time.sleep(0.5)

      msgs = p.can_recv()
      buses = {int(dat): bus for _, dat, bus in msgs if bus <= 3}
      print(msgs)

      # jungle doesn't actually switch buses when switching orientation
      flipped = orientation == Panda.HARNESS_STATUS_FLIPPED
      assert buses[0] == (2 if flipped else 0)
      assert buses[2] == (0 if flipped else 2)

    # SBU voltages
    supply_voltage_mV = 1800 if p.get_type() in [Panda.HW_TYPE_TRES, ] else 3300

    if orientation == Panda.HARNESS_STATUS_NC:
      assert health['sbu1_voltage_mV'] > 0.9 * supply_voltage_mV
      assert health['sbu2_voltage_mV'] > 0.9 * supply_voltage_mV
    else:
      relay_line = 'sbu1_voltage_mV' if (detected_orientation == Panda.HARNESS_STATUS_FLIPPED) else 'sbu2_voltage_mV'
      ignition_line = 'sbu2_voltage_mV' if (detected_orientation == Panda.HARNESS_STATUS_FLIPPED) else 'sbu1_voltage_mV'

      assert health[relay_line] < 0.1 * supply_voltage_mV
      assert health[ignition_line] > health[relay_line]
      if ignition:
        assert health[ignition_line] < 0.3 * supply_voltage_mV
      else:
        assert health[ignition_line] > 0.9 * supply_voltage_mV
