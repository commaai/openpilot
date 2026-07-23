from __future__ import annotations

import argparse
from collections import defaultdict
import threading
import usb1

from opendbc.car.structs import CarParams
import pytest

from openpilot.cereal import messaging
from panda import Panda

import openpilot.tools.cm5.usb_pandad as usb_pandad
from openpilot.tools.cm5.usb_pandad import (
  ReceiveOnlyPanda,
  ReceiveOnlyViolation,
  SILENT,
  UsbPandaPublisher,
  parse_can_data_speeds,
  parse_can_speeds,
)


def health(*, safety_mode=SILENT):
  return {
    "uptime": 12,
    "voltage": 12000,
    "current": 100,
    "safety_tx_blocked": 0,
    "safety_rx_invalid": 0,
    "tx_buffer_overflow": 0,
    "rx_buffer_overflow": 0,
    "faults": 0,
    "ignition_line": 1,
    "ignition_can": 0,
    "controls_allowed": 0,
    "car_harness_status": 1,
    "safety_mode": safety_mode,
    "safety_param": 0,
    "fault_status": 0,
    "power_save_enabled": 0,
    "heartbeat_lost": 0,
    "alternative_experience": 0,
    "interrupt_load": 0.0,
    "fan_power": 0,
    "safety_rx_checks_invalid": 0,
    "spi_error_count": 0,
    "sbu1_voltage_mV": 0,
    "sbu2_voltage_mV": 0,
    "som_reset_triggered": 0,
    "sound_output_level": 0,
  }


def can_health():
  return {
    "bus_off": 0,
    "bus_off_cnt": 0,
    "error_warning": 0,
    "error_passive": 0,
    "last_error": "No error",
    "last_stored_error": "No error",
    "last_data_error": "No error",
    "last_data_stored_error": "No error",
    "receive_error_cnt": 0,
    "transmit_error_cnt": 0,
    "total_error_cnt": 0,
    "total_tx_lost_cnt": 0,
    "total_rx_lost_cnt": 0,
    "total_tx_cnt": 0,
    "total_rx_cnt": 4,
    "total_fwd_cnt": 0,
    "total_tx_checksum_error_cnt": 0,
    "can_speed": 500,
    "can_data_speed": 2000,
    "canfd_enabled": 0,
    "brs_enabled": 0,
    "canfd_non_iso": 0,
    "irq0_call_rate": 0,
    "irq1_call_rate": 0,
    "irq2_call_rate": 0,
    "can_core_reset_count": 0,
  }


class FakePanda:
  def __init__(self, *, safety_mode=SILENT, panda_type=Panda.HW_TYPE_RED_PANDA, bootstub=False):
    self._health = health(safety_mode=safety_mode)
    self._can_health = [can_health() for _ in range(3)]
    self._panda_type = panda_type
    self.bootstub = bootstub
    self.calls = []

  def get_type(self):
    return self._panda_type

  def is_connected_usb(self):
    return True

  def set_heartbeat_disabled(self):
    self.calls.append(("set_heartbeat_disabled",))

  def set_power_save(self, enabled):
    self.calls.append(("set_power_save", enabled))

  def set_can_speed_kbps(self, bus, speed):
    self.calls.append(("set_can_speed_kbps", bus, speed))
    self._can_health[bus]["can_speed"] = speed

  def set_can_data_speed_kbps(self, bus, speed):
    self.calls.append(("set_can_data_speed_kbps", bus, speed))
    self._can_health[bus]["can_data_speed"] = speed

  def set_canfd_non_iso(self, bus, enabled):
    self.calls.append(("set_canfd_non_iso", bus, enabled))
    self._can_health[bus]["canfd_non_iso"] = enabled

  def set_canfd_auto(self, bus, enabled):
    self.calls.append(("set_canfd_auto", bus, enabled))

  def set_safety_mode(self, mode):
    self.calls.append(("set_safety_mode", mode))
    self._health["safety_mode"] = mode

  def can_recv(self):
    return [(0x123, b"\x01\x02", 0)]

  def health(self):
    return self._health.copy()

  def can_health(self, bus):
    return self._can_health[bus].copy()

  def get_fan_rpm(self):
    return 1234


class FakePubMaster:
  def __init__(self):
    self.messages = defaultdict(list)

  def send(self, service, message):
    if isinstance(message, bytes):
      message = messaging.log_from_bytes(message)
    else:
      message = message.as_reader()
    self.messages[service].append(message)


def test_configure_is_usb_receive_only():
  panda = FakePanda()
  publisher = UsbPandaPublisher(panda, FakePubMaster())

  publisher.configure()

  assert panda.calls == [
    ("set_safety_mode", CarParams.SafetyModel.silent),
    ("set_heartbeat_disabled",),
    ("set_power_save", 0),
    ("set_can_speed_kbps", 0, 500),
    ("set_can_data_speed_kbps", 0, 2000),
    ("set_canfd_non_iso", 0, False),
    ("set_canfd_auto", 0, False),
    ("set_can_speed_kbps", 1, 500),
    ("set_can_data_speed_kbps", 1, 2000),
    ("set_canfd_non_iso", 1, False),
    ("set_canfd_auto", 1, False),
    ("set_can_speed_kbps", 2, 500),
    ("set_can_data_speed_kbps", 2, 2000),
    ("set_canfd_non_iso", 2, False),
    ("set_canfd_auto", 2, False),
    ("set_safety_mode", CarParams.SafetyModel.silent),
  ]


def test_receive_only_panda_sets_silent_before_can_reset(monkeypatch):
  calls = []
  monkeypatch.setattr(Panda, "set_safety_mode", lambda _panda, mode, _param=0: calls.append(("safety", mode)))
  monkeypatch.setattr(Panda, "can_reset_communications", lambda _panda: calls.append(("reset",)))

  ReceiveOnlyPanda.__new__(ReceiveOnlyPanda).can_reset_communications()

  assert calls == [("safety", SILENT), ("reset",)]


def test_receive_only_panda_blocks_transmit_and_other_safety_modes(monkeypatch):
  panda = ReceiveOnlyPanda.__new__(ReceiveOnlyPanda)
  monkeypatch.setattr(Panda, "set_safety_mode", lambda *_args: None)

  with pytest.raises(ReceiveOnlyViolation, match="transmission is disabled"):
    panda.can_send(0x123, b"\x00", 0)
  with pytest.raises(ReceiveOnlyViolation, match="transmission is disabled"):
    panda.can_send_many([(0x123, b"\x00", 0)])
  with pytest.raises(ReceiveOnlyViolation, match="except SILENT"):
    panda.set_safety_mode(CarParams.SafetyModel.noOutput)


def test_receive_only_panda_checks_can_packet_version_before_usb_read():
  class UnreadableHandle:
    def bulkRead(self, *_args):
      raise AssertionError("bulkRead must not run after a packet-version mismatch")

  panda = ReceiveOnlyPanda.__new__(ReceiveOnlyPanda)
  panda.can_version = Panda.CAN_PACKET_VERSION + 1
  panda._handle = UnreadableHandle()
  panda.can_rx_overflow_buffer = b""

  with pytest.raises(RuntimeError, match="CAN packet version mismatch"):
    panda.can_recv()


def test_bus_speed_parser_rejects_values_firmware_cannot_program():
  assert parse_can_speeds("125,250,500") == (125, 250, 500)
  assert parse_can_data_speeds("1000,2000,5000") == (1000, 2000, 5000)
  with pytest.raises(argparse.ArgumentTypeError, match="supported values"):
    parse_can_speeds("333,500,500")
  with pytest.raises(argparse.ArgumentTypeError, match="supported values"):
    parse_can_data_speeds("8000,2000,2000")


@pytest.mark.parametrize("panda", [
  FakePanda(panda_type=Panda.HW_TYPE_BODY),
  FakePanda(panda_type=b"\x00"),
  FakePanda(bootstub=True),
])
def test_rejects_unsupported_or_bootstub_panda(panda):
  with pytest.raises(ReceiveOnlyViolation):
    UsbPandaPublisher(panda, FakePubMaster())


def test_run_once_publishes_can_and_health():
  panda = FakePanda()
  pm = FakePubMaster()
  publisher = UsbPandaPublisher(panda, pm, monotonic=lambda: 10.0)

  publisher.run_once()

  assert [(frame.address, bytes(frame.dat), frame.src) for frame in pm.messages["can"][0].can] == [(0x123, b"\x01\x02", 0)]
  state = pm.messages["pandaStates"][0].pandaStates[0]
  assert state.pandaType == "redPanda"
  assert state.safetyModel == "silent"
  assert state.ignitionLine
  assert state.canState0.totalRxCnt == 4
  assert pm.messages["peripheralState"][0].peripheralState.fanSpeedRpm == 1234


def test_run_once_fails_closed_if_silent_mode_drifts():
  panda = FakePanda(safety_mode=CarParams.SafetyModel.noOutput)
  pm = FakePubMaster()
  publisher = UsbPandaPublisher(panda, pm, monotonic=lambda: 10.0)

  with pytest.raises(ReceiveOnlyViolation, match="receive-only verification failed"):
    publisher.run_once()

  assert panda.calls == [("set_safety_mode", CarParams.SafetyModel.silent)]
  assert not pm.messages


def test_configure_rejects_prior_transmission_counter():
  panda = FakePanda()
  panda._can_health[1]["total_tx_cnt"] = 1

  with pytest.raises(ReceiveOnlyViolation, match="transmitted or forwarded"):
    UsbPandaPublisher(panda, FakePubMaster()).configure()


@pytest.mark.parametrize("loss_source", ["usb", "bus"])
def test_configure_rejects_confirmed_receive_loss(loss_source):
  panda = FakePanda()
  if loss_source == "usb":
    panda._health["rx_buffer_overflow"] = 1
  else:
    panda._can_health[2]["total_rx_lost_cnt"] = 1

  with pytest.raises(ReceiveOnlyViolation, match="recording is incomplete"):
    UsbPandaPublisher(panda, FakePubMaster()).configure()


def test_configure_applies_per_bus_bitrates_and_non_iso():
  panda = FakePanda()
  publisher = UsbPandaPublisher(
    panda, FakePubMaster(), can_speeds=(125, 250, 500),
    can_data_speeds=(500, 1000, 2000), canfd_non_iso_buses=frozenset({1}),
  )

  publisher.configure()

  assert [state["can_speed"] for state in panda._can_health] == [125, 250, 500]
  assert [state["can_data_speed"] for state in panda._can_health] == [500, 1000, 2000]
  assert [state["canfd_non_iso"] for state in panda._can_health] == [False, True, False]


def test_empty_can_batch_is_not_published():
  panda = FakePanda()
  panda.can_recv = list
  pm = FakePubMaster()

  UsbPandaPublisher(panda, pm, monotonic=lambda: 0.0).run_once()

  assert not pm.messages["can"]


def test_sustained_panda_outage_exits_for_supervisor(monkeypatch):
  times = iter((0.0, 0.0, 2.0))
  monkeypatch.setattr(usb_pandad.time, "monotonic", lambda: next(times))
  monkeypatch.setattr(usb_pandad, "_select_usb_serial", lambda _serial: (_ for _ in ()).throw(RuntimeError("missing")))

  with pytest.raises(RuntimeError, match="unavailable for 1 seconds"):
    usb_pandad.run(None, threading.Event(), reconnect_delay=0, outage_timeout=1)


@pytest.mark.parametrize("failure", [RuntimeError("USB read failed"), usb1.USBErrorOverflow()])
def test_established_session_io_failure_exits_for_supervisor(monkeypatch, failure):
  class FailingPanda:
    def set_safety_mode(self, _mode):
      pass

    def close(self):
      pass

  class FailingPublisher:
    def __init__(self, *_args, **_kwargs):
      pass

    def configure(self):
      pass

    def run_once(self):
      raise failure

  monkeypatch.setattr(usb_pandad, "_select_usb_serial", lambda _serial: "panda")
  monkeypatch.setattr(usb_pandad, "ReceiveOnlyPanda", lambda *_args, **_kwargs: FailingPanda())
  monkeypatch.setattr(usb_pandad, "UsbPandaPublisher", FailingPublisher)
  monkeypatch.setattr(usb_pandad.messaging, "PubMaster", lambda _services: object())

  expected = ReceiveOnlyViolation if isinstance(failure, usb1.USBErrorOverflow) else RuntimeError
  with pytest.raises(expected, match="recording is incomplete"):
    usb_pandad.run(None, threading.Event(), reconnect_delay=0, outage_timeout=30)
