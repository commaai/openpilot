import time
import unittest

from opendbc.car.structs import CarParams
from panda.tests.hitl.base import PandaTestCase, panda_test

# keep in sync with PERMANENT_FAULTS, see board/sys/sys.h
PERMANENT_FAULT_BITS = ()
# board/drivers/self_test.h injection index: fault bit
SELF_TEST_FAULT_BITS = {0: 9, 1: 10, 2: 11, 3: 12, 4: 12, 5: 14}
# board/main_comms.h 0xef injection index: fault bit
INTERRUPT_FAULT_BITS = {0: 17, 1: 17, 2: 1, 3: 16, 4: 18, 5: 18, 6: 1}
FAULT_SIREN_MALFUNCTION_BIT = 25
FAULT_MPU_BIT = 28
# board/main_comms.h 0xf5 injection index: fault bit
READBACK_FAULT_BITS = {0: 13, 1: 29, 2: 24, 3: 29, 4: 18, 5: 18, 6: 24, 7: 18}


class TestSafety(PandaTestCase):
  def test_safety_nooutput(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.silent)
    p.set_can_loopback(True)

    # send a message on bus 0
    p.can_send(0x1aa, b"message", 0)

    # confirm receive nothing
    time.sleep(0.05)
    r = p.can_recv()
    # bus 192 is messages blocked by TX safety hook on bus 0
    assert len([x for x in r if x[2] != 192]) == 0
    assert len([x for x in r if x[2] == 192]) == 1

  @unittest.skipIf(len(PERMANENT_FAULT_BITS) == 0, "no permanent faults")
  def test_fault_safe_state(self):
    p = self.p
    for fault_bit in PERMANENT_FAULT_BITS:
      p.reset()
      p.set_safety_mode(CarParams.SafetyModel.allOutput)
      h = p.health()
      assert h['safety_mode'] == CarParams.SafetyModel.allOutput
      assert not (h['faults'] & (1 << fault_bit))

      t = time.monotonic()
      p.inject_fault(fault_bit)
      h = p.health()
      print(f"fault bit {fault_bit}: safe state after {(time.monotonic() - t) * 1e3:.1f} ms")
      assert h['faults'] & (1 << fault_bit)
      assert h['fault_status'] == 2
      assert h['safety_mode'] == CarParams.SafetyModel.silent
      assert not h['controls_allowed']

      p.set_safety_mode(CarParams.SafetyModel.allOutput)
      assert p.health()['safety_mode'] == CarParams.SafetyModel.silent
    p.reset()

  def test_temporary_fault_no_safe_state(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.inject_fault(FAULT_SIREN_MALFUNCTION_BIT)
    h = p.health()
    assert h['fault_status'] == 1
    assert h['safety_mode'] == CarParams.SafetyModel.allOutput
    p.reset()

  def test_hardfault_resets_to_silent(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.inject_hardfault()
    p.reconnect()
    h = p.health()
    assert h['hardfault_reset']
    assert h['safety_mode'] == CarParams.SafetyModel.silent
    assert not h['controls_allowed']

  def test_canfd_auto_bus_out_of_range(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.set_canfd_auto(0xFFFF, True)
    h = p.health()
    assert not h['hardfault_reset']
    assert h['safety_mode'] == CarParams.SafetyModel.allOutput

  def test_mpu_fault(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    assert not (p.health()['faults'] & (1 << FAULT_MPU_BIT))
    p.inject_mpu_fault()
    if FAULT_MPU_BIT in PERMANENT_FAULT_BITS:
      p.reconnect()
      h = p.health()
      assert h['hardfault_reset']
      assert h['safety_mode'] == CarParams.SafetyModel.silent
    else:
      # flagged at the next safety hook, raised by mpu_check() at 1 Hz
      time.sleep(2.5)
      h = p.health()
      assert h['faults'] & (1 << FAULT_MPU_BIT)
      assert h['safety_mode'] == CarParams.SafetyModel.allOutput
    p.reset()

  def test_stack_overflow_resets_to_silent(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.inject_stack_overflow()
    p.reconnect()
    h = p.health()
    assert h['hardfault_reset']
    assert h['safety_mode'] == CarParams.SafetyModel.silent
    assert not h['controls_allowed']

  def test_nmi_resets_to_silent(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.inject_nmi()
    p.reconnect()
    h = p.health()
    assert h['nmi_reset']
    assert h['safety_mode'] == CarParams.SafetyModel.silent
    assert not h['controls_allowed']

  @panda_test(timeout=2*60)
  def test_self_test_failure(self):
    p = self.p
    for test, fault_bit in SELF_TEST_FAULT_BITS.items():
      p.reset()
      p.set_safety_mode(CarParams.SafetyModel.allOutput)
      assert not (p.health()['faults'] & (1 << fault_bit))
      p.inject_self_test_failure(test)

      # heartbeat keeps allOutput, so only the fault can set SILENT
      t = time.monotonic()
      while not (p.health()['faults'] & (1 << fault_bit)) and (time.monotonic() - t) < 30:
        p.send_heartbeat(engaged=False)
        time.sleep(0.1)
      print(f"self-test {test}: fault after {time.monotonic() - t:.1f} s")
      h = p.health()
      assert h['faults'] & (1 << fault_bit)
      if fault_bit in PERMANENT_FAULT_BITS:
        assert h['safety_mode'] == CarParams.SafetyModel.silent
      else:
        assert h['safety_mode'] == CarParams.SafetyModel.allOutput
    p.reset()

  @panda_test(timeout=2*60)
  def test_interrupt_fault(self):
    p = self.p
    for test, fault_bit in INTERRUPT_FAULT_BITS.items():
      p.reset()
      p.set_safety_mode(CarParams.SafetyModel.allOutput)
      # the missing interrupt check skips the first window
      time.sleep(2.5)
      h = p.health()
      assert h['faults'] == 0, f"faults {h['faults']:#x}"
      p.inject_interrupt_fault(test)

      t = time.monotonic()
      while not ((h := p.health())['faults'] & (1 << fault_bit)) and (time.monotonic() - t) < 5:
        time.sleep(0.05)
      print(f"interrupt fault {test}: fault after {time.monotonic() - t:.1f} s")
      assert h['faults'] & (1 << fault_bit)
      if fault_bit in PERMANENT_FAULT_BITS:
        assert h['safety_mode'] == CarParams.SafetyModel.silent
      else:
        assert h['safety_mode'] == CarParams.SafetyModel.allOutput
    p.reset()

  @panda_test(timeout=3*60)
  def test_readback_fault(self):
    p = self.p
    for test, fault_bit in READBACK_FAULT_BITS.items():
      p.reset()
      # harness detection only reads the ADC while the relay isn't driven
      p.set_safety_mode(CarParams.SafetyModel.noOutput)
      time.sleep(2.5)
      h = p.health()
      assert h['faults'] == 0, f"faults {h['faults']:#x}"
      p.inject_readback_fault(test)

      t = time.monotonic()
      while not ((h := p.health())['faults'] & (1 << fault_bit)) and (time.monotonic() - t) < 8:
        time.sleep(0.05)
      print(f"readback fault {test}: fault after {time.monotonic() - t:.1f} s")
      assert h['faults'] & (1 << fault_bit)
      assert h['safety_mode'] == (CarParams.SafetyModel.silent if fault_bit in PERMANENT_FAULT_BITS else CarParams.SafetyModel.noOutput)
    p.reset()
