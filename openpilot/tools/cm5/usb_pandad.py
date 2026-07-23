#!/usr/bin/env python3
"""Receive-only USB Panda publisher for the CM5 dashcam runtime.

This process intentionally has no ``sendcan`` subscription and never calls a
Panda CAN transmit API. Panda is kept in SILENT safety mode, which also puts its
CAN controllers in listen-only mode.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
import usb1
from collections.abc import Callable
from typing import Any

from opendbc.car.structs import CarParams
from panda import Panda, unpack_can_buffer
from panda.python import ensure_can_packet_version

from openpilot.cereal import log, messaging
from openpilot.selfdrive.pandad import can_list_to_can_capnp


PANDA_STATE_INTERVAL = 0.1
PERIPHERAL_STATE_INTERVAL = 0.5
CAN_POLL_INTERVAL = 0.01
PANDA_CAN_CNT = 3
DEFAULT_OUTAGE_TIMEOUT = 30.0
SILENT = CarParams.SafetyModel.silent
DEFAULT_CAN_SPEEDS = (500, 500, 500)
DEFAULT_CAN_DATA_SPEEDS = (2000, 2000, 2000)
SUPPORTED_PANDA_TYPES = {Panda.HW_TYPE_RED_PANDA}
SUPPORTED_CAN_SPEEDS = frozenset({10, 20, 50, 100, 125, 250, 500, 1000})
SUPPORTED_CAN_DATA_SPEEDS = SUPPORTED_CAN_SPEEDS | {2000, 5000}

LEC_ERROR_CODES = {
  "No error": "noError",
  "Stuff error": "stuffError",
  "Form error": "formError",
  "AckError": "ackError",
  "Bit1Error": "bit1Error",
  "Bit0Error": "bit0Error",
  "CRCError": "crcError",
  "NoChange": "noChange",
}


class ReceiveOnlyViolation(RuntimeError):
  """A fail-closed violation of the passive-CAN boundary."""


def _parse_bus_speeds(value: str, *, allowed: frozenset[int], description: str) -> tuple[int, ...]:
  try:
    speeds = tuple(int(part.strip()) for part in value.split(","))
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"{description} must be comma-separated integers in kbit/s") from exc
  if len(speeds) != PANDA_CAN_CNT or any(speed not in allowed for speed in speeds):
    choices = ",".join(map(str, sorted(allowed)))
    raise argparse.ArgumentTypeError(f"expected {PANDA_CAN_CNT} {description}; supported values are {choices} kbit/s")
  return speeds


def parse_can_speeds(value: str) -> tuple[int, ...]:
  return _parse_bus_speeds(value, allowed=SUPPORTED_CAN_SPEEDS, description="CAN arbitration speeds")


def parse_can_data_speeds(value: str) -> tuple[int, ...]:
  return _parse_bus_speeds(value, allowed=SUPPORTED_CAN_DATA_SPEEDS, description="CAN-FD data speeds")


def parse_bus_indices(value: str) -> frozenset[int]:
  if value.strip().lower() in ("", "none"):
    return frozenset()
  try:
    buses = frozenset(int(part.strip()) for part in value.split(","))
  except ValueError as exc:
    raise argparse.ArgumentTypeError("CAN bus indexes must be comma-separated integers") from exc
  if any(bus < 0 or bus >= PANDA_CAN_CNT for bus in buses):
    raise argparse.ArgumentTypeError(f"CAN bus indexes must be between 0 and {PANDA_CAN_CNT - 1}")
  return buses


class ReceiveOnlyPanda(Panda):
  """Assert SILENT before Panda's constructor resets or configures CAN."""
  def set_safety_mode(self, mode=SILENT, param=0):
    if mode != SILENT or param != 0:
      raise ReceiveOnlyViolation("receive-only Panda refuses every safety mode except SILENT with parameter 0")
    return super().set_safety_mode(mode, param)

  def can_reset_communications(self):
    self.set_safety_mode(SILENT)
    return super().can_reset_communications()

  def can_send_many(self, *_args, **_kwargs):
    raise ReceiveOnlyViolation("CAN transmission is disabled in the CM5 dashcam runtime")

  def can_send(self, *_args, **_kwargs):
    raise ReceiveOnlyViolation("CAN transmission is disabled in the CM5 dashcam runtime")

  @ensure_can_packet_version
  def can_recv(self):
    """Perform one USB receive without Panda's unbounded internal retry loop."""
    dat = self._handle.bulkRead(1, 16384)
    messages, self.can_rx_overflow_buffer = unpack_can_buffer(self.can_rx_overflow_buffer + dat)
    return messages


def _panda_type(raw_type: bytes) -> int:
  value = raw_type[0] if raw_type else 0
  if value > log.PandaState.PandaType.cuatro:
    return log.PandaState.PandaType.unknown
  return value


def _faults(mask: int) -> list[int]:
  return [bit for bit in range(27) if mask & (1 << bit)]


def _fill_can_state(state: Any, health: dict[str, Any]) -> None:
  state.busOff = bool(health["bus_off"])
  state.busOffCnt = health["bus_off_cnt"]
  state.errorWarning = bool(health["error_warning"])
  state.errorPassive = bool(health["error_passive"])
  state.lastError = LEC_ERROR_CODES[health["last_error"]]
  state.lastStoredError = LEC_ERROR_CODES[health["last_stored_error"]]
  state.lastDataError = LEC_ERROR_CODES[health["last_data_error"]]
  state.lastDataStoredError = LEC_ERROR_CODES[health["last_data_stored_error"]]
  state.receiveErrorCnt = health["receive_error_cnt"]
  state.transmitErrorCnt = health["transmit_error_cnt"]
  state.totalErrorCnt = health["total_error_cnt"]
  state.totalTxLostCnt = health["total_tx_lost_cnt"]
  state.totalRxLostCnt = health["total_rx_lost_cnt"]
  state.totalTxCnt = health["total_tx_cnt"]
  state.totalRxCnt = health["total_rx_cnt"]
  state.totalFwdCnt = health["total_fwd_cnt"]
  state.canSpeed = health["can_speed"]
  state.canDataSpeed = health["can_data_speed"]
  state.canfdEnabled = bool(health["canfd_enabled"])
  state.brsEnabled = bool(health["brs_enabled"])
  state.canfdNonIso = bool(health["canfd_non_iso"])
  state.irq0CallRate = health["irq0_call_rate"]
  state.irq1CallRate = health["irq1_call_rate"]
  state.irq2CallRate = health["irq2_call_rate"]
  state.canCoreResetCnt = health["can_core_reset_count"]


class UsbPandaPublisher:
  def __init__(self, panda: Panda, pm: messaging.PubMaster, monotonic: Callable[[], float] = time.monotonic,
               can_speeds: tuple[int, ...] = DEFAULT_CAN_SPEEDS,
               can_data_speeds: tuple[int, ...] = DEFAULT_CAN_DATA_SPEEDS,
               canfd_non_iso_buses: frozenset[int] = frozenset()):
    self.panda = panda
    self.pm = pm
    self.monotonic = monotonic
    if getattr(panda, "bootstub", False):
      raise ReceiveOnlyViolation("USB Panda is in bootstub mode; flash matching application firmware before recording")
    raw_type = panda.get_type()
    if raw_type not in SUPPORTED_PANDA_TYPES:
      panda_type = raw_type.hex() if raw_type else "unknown"
      raise ReceiveOnlyViolation(f"unsupported Panda hardware type 0x{panda_type}; the CM5 recorder requires an external Red Panda")
    if len(can_speeds) != PANDA_CAN_CNT or len(can_data_speeds) != PANDA_CAN_CNT:
      raise ValueError(f"expected {PANDA_CAN_CNT} arbitration and data speeds")
    self.panda_type = _panda_type(raw_type)
    self.can_speeds = can_speeds
    self.can_data_speeds = can_data_speeds
    self.canfd_non_iso_buses = canfd_non_iso_buses
    self.next_panda_state = 0.0
    self.next_peripheral_state = 0.0

  def configure(self) -> None:
    if not self.panda.is_connected_usb():
      raise RuntimeError("CM5 usb_pandad requires an external USB Panda")

    # Eliminate any window in which a Panda left in another mode can participate
    # on the bus before the rest of its receive configuration is applied.
    self.panda.set_safety_mode(SILENT)

    # Keep the device awake without heartbeats. A heartbeat re-enables Panda's
    # heartbeat checks, so this process deliberately never sends one.
    self.panda.set_heartbeat_disabled()
    self.panda.set_power_save(0)

    for bus in range(PANDA_CAN_CNT):
      self.panda.set_can_speed_kbps(bus, self.can_speeds[bus])
      self.panda.set_can_data_speed_kbps(bus, self.can_data_speeds[bus])
      self.panda.set_canfd_non_iso(bus, bus in self.canfd_non_iso_buses)
      # This option only affects outgoing frames, which are forbidden here.
      self.panda.set_canfd_auto(bus, False)

    # Reassert after configuration. SILENT selects the no-output safety hooks
    # and keeps CAN listen-only (including no ACK transmission).
    self.panda.set_safety_mode(SILENT)
    health = self.panda.health()
    can_health = [self.panda.can_health(bus) for bus in range(PANDA_CAN_CNT)]
    self._verify_receive_only(health, can_health)

  def _verify_receive_only(self, health: dict[str, Any], can_health: list[dict[str, Any]]) -> None:
    if health["safety_mode"] != SILENT or bool(health["controls_allowed"]):
      try:
        self.panda.set_safety_mode(SILENT)
      finally:
        raise ReceiveOnlyViolation(
          f"Panda receive-only verification failed: safety_mode={health['safety_mode']} " +
          f"controls_allowed={health['controls_allowed']}"
        )
    if health["rx_buffer_overflow"] != 0:
      raise ReceiveOnlyViolation(
        f"Panda USB receive buffer overflowed {health['rx_buffer_overflow']} time(s); CAN recording is incomplete"
      )

    for bus, state in enumerate(can_health):
      if state["total_tx_cnt"] != 0 or state["total_fwd_cnt"] != 0:
        raise ReceiveOnlyViolation(
          f"Panda bus {bus} transmitted or forwarded frames since boot " +
          f"(tx={state['total_tx_cnt']}, forwarded={state['total_fwd_cnt']}); power-cycle it before passive recording"
        )
      if state["total_rx_lost_cnt"] != 0:
        raise ReceiveOnlyViolation(
          f"Panda bus {bus} lost {state['total_rx_lost_cnt']} received frame(s); CAN recording is incomplete"
        )
      if state["can_speed"] != self.can_speeds[bus] or state["can_data_speed"] != self.can_data_speeds[bus]:
        raise RuntimeError(
          f"Panda bus {bus} bitrate readback mismatch: expected {self.can_speeds[bus]}/{self.can_data_speeds[bus]} " +
          f"kbit/s, got {state['can_speed']}/{state['can_data_speed']}"
        )
      if bool(state["canfd_non_iso"]) != (bus in self.canfd_non_iso_buses):
        raise RuntimeError(f"Panda bus {bus} CAN-FD ISO mode readback mismatch")

  def _build_panda_states(self, health: dict[str, Any], can_health: list[dict[str, Any]]):
    msg = messaging.new_message("pandaStates", 1)
    msg.valid = True
    state = msg.pandaStates[0]
    state.voltage = health["voltage"]
    state.current = health["current"]
    state.uptime = health["uptime"]
    state.safetyTxBlocked = health["safety_tx_blocked"]
    state.safetyRxInvalid = health["safety_rx_invalid"]
    state.ignitionLine = bool(health["ignition_line"])
    state.ignitionCan = bool(health["ignition_can"])
    state.controlsAllowed = bool(health["controls_allowed"])
    state.txBufferOverflow = health["tx_buffer_overflow"]
    state.rxBufferOverflow = health["rx_buffer_overflow"]
    state.pandaType = self.panda_type
    state.safetyModel = health["safety_mode"]
    state.safetyParam = health["safety_param"]
    state.faultStatus = health["fault_status"]
    state.powerSaveEnabled = bool(health["power_save_enabled"])
    state.heartbeatLost = bool(health["heartbeat_lost"])
    state.alternativeExperience = health["alternative_experience"]
    state.harnessStatus = health["car_harness_status"]
    state.interruptLoad = health["interrupt_load"]
    state.fanPower = health["fan_power"]
    state.safetyRxChecksInvalid = bool(health["safety_rx_checks_invalid"])
    state.spiErrorCount = health["spi_error_count"]
    state.sbu1Voltage = health["sbu1_voltage_mV"] / 1000.0
    state.sbu2Voltage = health["sbu2_voltage_mV"] / 1000.0
    state.soundOutputLevel = health["sound_output_level"]
    state.faults = _faults(health["faults"])

    can_states = (state.canState0, state.canState1, state.canState2)
    for bus, can_state in enumerate(can_states):
      _fill_can_state(can_state, can_health[bus])
    return msg

  def _build_peripheral_state(self, health: dict[str, Any]):
    msg = messaging.new_message("peripheralState")
    msg.valid = True
    msg.peripheralState.pandaType = self.panda_type
    msg.peripheralState.voltage = health["voltage"]
    msg.peripheralState.current = health["current"]
    try:
      msg.peripheralState.fanSpeedRpm = self.panda.get_fan_rpm()
    except Exception:
      msg.peripheralState.fanSpeedRpm = 0
    return msg

  def run_once(self) -> None:
    now = self.monotonic()
    if now >= self.next_panda_state:
      health = self.panda.health()
      can_health = [self.panda.can_health(bus) for bus in range(PANDA_CAN_CNT)]
      self._verify_receive_only(health, can_health)
      self.pm.send("pandaStates", self._build_panda_states(health, can_health))
      self.next_panda_state = now + PANDA_STATE_INTERVAL

      if now >= self.next_peripheral_state:
        self.pm.send("peripheralState", self._build_peripheral_state(health))
        self.next_peripheral_state = now + PERIPHERAL_STATE_INTERVAL

    frames = self.panda.can_recv()
    if frames:
      self.pm.send("can", can_list_to_can_capnp(frames))


def _select_usb_serial(requested: str | None) -> str:
  serials = sorted(Panda.list(usb_only=True))
  if requested is not None:
    if requested not in serials:
      raise RuntimeError(f"USB Panda {requested} is not connected")
    return requested
  if len(serials) != 1:
    raise RuntimeError(f"expected exactly one USB Panda, found {len(serials)}: {serials}")
  return serials[0]


def run(serial: str | None, stop_event: threading.Event, reconnect_delay: float = 1.0,
        outage_timeout: float = DEFAULT_OUTAGE_TIMEOUT,
        can_speeds: tuple[int, ...] = DEFAULT_CAN_SPEEDS,
        can_data_speeds: tuple[int, ...] = DEFAULT_CAN_DATA_SPEEDS,
        canfd_non_iso_buses: frozenset[int] = frozenset()) -> None:
  unavailable_since = time.monotonic()
  while not stop_event.is_set():
    panda: Panda | None = None
    recording_started = False
    try:
      selected_serial = _select_usb_serial(serial)
      # Own the safety/power setup explicitly in configure() instead of relying
      # on Panda's general-purpose "outside openpilot" defaults.
      panda = ReceiveOnlyPanda(selected_serial, cli=False, disable_checks=False, can_speed_kbps=can_speeds[0])
      publisher = UsbPandaPublisher(
        panda, messaging.PubMaster(["can", "pandaStates", "peripheralState"]),
        can_speeds=can_speeds, can_data_speeds=can_data_speeds, canfd_non_iso_buses=canfd_non_iso_buses,
      )
      publisher.configure()
      # From this point onward a reconnect could discard queued frames while
      # still recovering before the supervisor watchdog expires. Treat every
      # session discontinuity as a route fault instead of silently reconnecting.
      recording_started = True
      logging.info("recording CAN from USB Panda %s in SILENT mode", selected_serial)
      while not stop_event.is_set():
        publisher.run_once()
        # Configuration alone is not proof of a working USB/CAN data path.
        # Clear the outage clock only after can_recv and health I/O succeed.
        unavailable_since = None
        stop_event.wait(CAN_POLL_INTERVAL)
    except ReceiveOnlyViolation:
      logging.exception("receive-only Panda invariant failed; stopping recorder")
      raise
    except usb1.USBErrorOverflow as exc:
      raise ReceiveOnlyViolation("host USB receive overflow; CAN recording is incomplete") from exc
    except Exception as exc:
      if recording_started and not stop_event.is_set():
        raise RuntimeError("USB Panda session interrupted after recording started; CAN recording is incomplete") from exc
      if not stop_event.is_set():
        logging.exception("USB Panda disconnected or unavailable; retrying")
        now = time.monotonic()
        unavailable_since = now if unavailable_since is None else unavailable_since
        if outage_timeout > 0 and now - unavailable_since >= outage_timeout:
          raise RuntimeError(f"USB Panda unavailable for {outage_timeout:g} seconds") from exc
    finally:
      if panda is not None:
        try:
          panda.set_safety_mode(SILENT)
        except Exception:
          pass
        panda.close()

    stop_event.wait(reconnect_delay)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="CM5 receive-only USB Panda publisher")
  parser.add_argument("--serial", default=os.getenv("DASHCAM_PANDA_SERIAL") or None, help="USB Panda serial (required when more than one is attached)")
  parser.add_argument("--reconnect-delay", type=float, default=1.0)
  parser.add_argument("--outage-timeout", type=float, default=float(os.getenv("DASHCAM_PANDA_OUTAGE_TIMEOUT", DEFAULT_OUTAGE_TIMEOUT)))
  parser.add_argument("--can-speeds", type=parse_can_speeds, default=parse_can_speeds(os.getenv("DASHCAM_CAN_SPEEDS", "500,500,500")))
  parser.add_argument("--can-data-speeds", type=parse_can_data_speeds, default=parse_can_data_speeds(os.getenv("DASHCAM_CAN_DATA_SPEEDS", "2000,2000,2000")))
  parser.add_argument(
    "--canfd-non-iso-buses", type=parse_bus_indices,
    default=parse_bus_indices(os.getenv("DASHCAM_CANFD_NON_ISO_BUSES", "")),
    help="comma-separated CAN bus indexes using non-ISO CAN-FD (default: none)",
  )
  # PythonProcess calls main() after importing this module and retains the
  # supervisor's argv. Only parse command-line arguments for direct execution.
  args = parser.parse_args([] if argv is None else argv)

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  stop_event = threading.Event()
  for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda _signum, _frame: stop_event.set())

  run(
    args.serial, stop_event, args.reconnect_delay, args.outage_timeout,
    args.can_speeds, args.can_data_speeds, args.canfd_non_iso_buses,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
