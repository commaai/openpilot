#!/usr/bin/env python3
"""Standalone passive dashcam runtime for a V4L2 camera and USB Panda."""

from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time

from panda import Panda

from openpilot.cereal import messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.system.manager.dashcam_process_config import build_dashcam_processes
from openpilot.system.manager.process import ManagerProcess, NativeProcess
from openpilot.tools.cm5.usb_pandad import parse_bus_indices, parse_can_data_speeds, parse_can_speeds


DEFAULT_STATE_ROOT = Path.home() / ".comma-cm5-dashcam"
SHUTDOWN_ORDER = ("deleter", "loggerd", "encoderd", "webcamerad", "pandad", "logmessaged")
LOGGER_STOP_TIMEOUT = 20.0
RUNTIME_FAULT_SIGNAL = signal.SIGUSR1
WATCHDOG_SERVICES = ("roadCameraState", "roadEncodeData", "qRoadEncodeData", "pandaStates")
DEFAULT_WATCHDOG_STARTUP_TIMEOUT = 30.0
DEFAULT_WATCHDOG_STALE_TIMEOUT = 5.0
WATCHDOG_RATE_WINDOW = 5.0
WATCHDOG_MIN_RATES = {
  "roadCameraState": 15.0,
  "roadEncodeData": 15.0,
  "qRoadEncodeData": 15.0,
  "pandaStates": 7.0,
}


class DashcamAlreadyRunningError(RuntimeError):
  pass


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError("must be greater than zero")
  return parsed


def nonnegative_float(value: str) -> float:
  parsed = float(value)
  if parsed < 0:
    raise argparse.ArgumentTypeError("must be zero or greater")
  return parsed


def positive_float(value: str) -> float:
  parsed = float(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError("must be greater than zero")
  return parsed


def percentage(value: str) -> float:
  parsed = nonnegative_float(value)
  if parsed > 100:
    raise argparse.ArgumentTypeError("must not exceed 100")
  return parsed


def camera_device(camera: str | int) -> Path:
  value = str(camera)
  return Path(f"/dev/video{value}") if value.isdecimal() else Path(value).expanduser()


def panda_selection_errors(requested_serial: str | None, serials: list[str] | None = None) -> list[str]:
  try:
    serials = sorted(Panda.list(usb_only=True) if serials is None else serials)
  except Exception as exc:
    return [f"failed to enumerate USB Panda: {exc}"]
  if requested_serial is not None and requested_serial not in serials:
    return [f"requested USB Panda {requested_serial} is not connected (found: {serials})"]
  if requested_serial is None and len(serials) != 1:
    return [f"expected exactly one USB Panda, found {len(serials)}: {serials}"]
  return []


def required_mount_errors(required_mount: Path | None, log_root: Path) -> list[str]:
  if required_mount is None:
    return []

  mount = required_mount.expanduser().resolve()
  logs = log_root.expanduser().resolve()
  errors = []
  if not mount.is_dir():
    errors.append(f"required recording mount does not exist: {mount}")
  elif not os.path.ismount(mount):
    errors.append(f"recording path is not a mount point: {mount}")
  if logs != mount and mount not in logs.parents:
    errors.append(f"log root {logs} is outside required mount {mount}")
  return errors


def acquire_instance_lock(path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  lock = path.open("a+", encoding="utf-8")
  try:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError as exc:
    lock.close()
    raise DashcamAlreadyRunningError(f"another dashcam runtime holds {path}") from exc
  lock.seek(0)
  lock.truncate()
  lock.write(f"{os.getpid()}\n")
  lock.flush()
  os.fsync(lock.fileno())
  return lock


def quarantine_interrupted_segments(log_root: Path) -> list[Path]:
  """Move segments carrying stale writer locks out of the active route set."""
  interrupted = []
  quarantine_root = log_root / ".interrupted"
  for segment in sorted(log_root.iterdir()):
    if segment.name.startswith(".") or not segment.is_dir() or segment.is_symlink():
      continue
    try:
      has_locks = any(path.is_file() for path in segment.glob("*.lock"))
    except OSError:
      continue
    if not has_locks:
      continue

    quarantine_root.mkdir(mode=0o750, exist_ok=True)
    destination = quarantine_root / segment.name
    suffix = 1
    while destination.exists():
      destination = quarantine_root / f"{segment.name}.{suffix}"
      suffix += 1
    segment.rename(destination)
    interrupted.append(destination)
  return interrupted


def local_device_id() -> str:
  """Build a stable, local-only 16-character device identifier."""
  identity = platform.node()
  machine_id = Path("/etc/machine-id")
  try:
    identity = machine_id.read_text().strip() or identity
  except OSError:
    pass
  return hashlib.sha256(f"cm5-dashcam:{identity}".encode()).hexdigest()[:16]


def configure_environment(args: argparse.Namespace) -> None:
  log_root = args.log_root.expanduser().resolve()
  params_root = args.params_root.expanduser().resolve()
  log_root.mkdir(parents=True, exist_ok=True)
  params_root.mkdir(parents=True, exist_ok=True)

  os.environ["LOG_ROOT"] = str(log_root)
  os.environ["PARAMS_ROOT"] = str(params_root)
  os.environ["OPENPILOT_PREFIX"] = args.prefix
  os.environ["ROAD_CAM"] = str(args.camera)
  os.environ["USE_WEBCAM"] = "1"
  os.environ["DASHCAM_ENCODER"] = args.encoder
  os.environ["DASHCAM_ENCODER_PRESET"] = args.encoder_preset
  os.environ["DASHCAM_MAIN_BITRATE"] = str(args.main_bitrate)
  os.environ["DASHCAM_MIN_FREE_BYTES"] = str(int(args.min_free_gb * 1024**3))
  os.environ["DASHCAM_MIN_FREE_PERCENT"] = str(args.min_free_percent)
  os.environ["DASHCAM_PANDA_OUTAGE_TIMEOUT"] = str(args.panda_outage_timeout)
  os.environ["DASHCAM_CAN_SPEEDS"] = ",".join(map(str, args.can_speeds))
  os.environ["DASHCAM_CAN_DATA_SPEEDS"] = ",".join(map(str, args.can_data_speeds))
  os.environ["DASHCAM_CANFD_NON_ISO_BUSES"] = ",".join(map(str, sorted(args.canfd_non_iso_buses)))
  os.environ["DASHCAM_REQUIRED_CAN_BUSES"] = ",".join(map(str, sorted(args.required_can_buses)))
  os.environ["DASHCAM_CAN_STALE_TIMEOUT"] = str(args.can_stale_timeout)
  if args.panda_serial:
    os.environ["DASHCAM_PANDA_SERIAL"] = args.panda_serial
  else:
    os.environ.pop("DASHCAM_PANDA_SERIAL", None)

  # Mark this namespace as the reduced dashcam runtime. Its dedicated USB Panda
  # publisher contains no CAN transmit or sendcan subscription path.
  os.environ["DASHCAM"] = "1"

  device_id = args.device_id or local_device_id()
  os.environ["DONGLE_ID"] = device_id

  params = Params()
  params.put("DongleId", device_id, block=True)
  params.put_bool("RecordAudio", False, block=True)


def preflight(processes: dict[str, ManagerProcess], camera: str | int, log_root: Path,
              panda_serial: str | None = None) -> list[str]:
  errors: list[str] = []
  missing_modules = [module for module in ("av", "cv2") if importlib.util.find_spec(module) is None]
  if missing_modules:
    errors.append(f"missing webcam Python modules: {', '.join(missing_modules)} (install `av` and `opencv-python-headless`)")

  camera_path = camera_device(camera)
  if not camera_path.exists():
    errors.append(f"camera not found: {camera_path}")

  if not os.access(log_root, os.W_OK):
    errors.append(f"log root is not writable: {log_root}")

  errors.extend(panda_selection_errors(panda_serial))

  for process in processes.values():
    if isinstance(process, NativeProcess):
      executable = Path(BASEDIR) / process.cwd / process.cmdline[0]
      if not executable.is_file() or not os.access(executable, os.X_OK):
        errors.append(f"missing executable for {process.name}: {executable} (run `scons -u --dashcam`)")
  return errors


def stale_data_paths(seen: dict[str, bool], recv_time: dict[str, float], *, started_at: float, now: float,
                     startup_timeout: float, stale_timeout: float) -> list[str]:
  stale = []
  for service in WATCHDOG_SERVICES:
    if seen.get(service, False):
      if now - recv_time.get(service, 0.0) > stale_timeout:
        stale.append(service)
    elif now - started_at > startup_timeout:
      stale.append(service)
  return stale


def low_rate_data_paths(samples: dict[str, deque[float]], *, now: float,
                        window: float = WATCHDOG_RATE_WINDOW) -> list[str]:
  low_rate = []
  for service, minimum_rate in WATCHDOG_MIN_RATES.items():
    cutoff = now - window
    while samples[service] and samples[service][0] < cutoff:
      samples[service].popleft()
    if len(samples[service]) < minimum_rate * window:
      low_rate.append(service)
  return low_rate


def stale_can_buses(required_buses: frozenset[int], last_seen: dict[int, float], *, started_at: float, now: float,
                    startup_timeout: float, stale_timeout: float) -> list[int]:
  return [
    bus for bus in sorted(required_buses)
    if (bus not in last_seen and now - started_at > startup_timeout) or
       (bus in last_seen and now - last_seen[bus] > stale_timeout)
  ]


class DashcamRuntime:
  def __init__(self, processes: dict[str, ManagerProcess], *,
               watchdog_startup_timeout: float = DEFAULT_WATCHDOG_STARTUP_TIMEOUT,
               watchdog_stale_timeout: float = DEFAULT_WATCHDOG_STALE_TIMEOUT,
               required_can_buses: frozenset[int] = frozenset(), can_stale_timeout: float = 5.0):
    self.processes = processes
    self.watchdog_startup_timeout = watchdog_startup_timeout
    self.watchdog_stale_timeout = watchdog_stale_timeout
    self.required_can_buses = required_can_buses
    self.can_stale_timeout = can_stale_timeout
    self.stop_event = threading.Event()
    self.stop_signal: int | None = None
    self.logger_stopping = False
    self.runtime_failed = False

  def mark_failed(self) -> None:
    self.runtime_failed = True

  def request_stop(self, signum: int | None = None, _frame=None) -> None:
    power_signal = getattr(signal, "SIGPWR", None)
    if signum is not None and (self.stop_signal is None or signum == power_signal):
      # A power-failure signal upgrades an earlier ordinary shutdown request.
      self.stop_signal = signum
    if power_signal is not None and signum == power_signal and self.logger_stopping and not self.runtime_failed:
      # Upgrade a logger that already received SIGINT/SIGTERM during shutdown.
      self.processes["loggerd"].signal(power_signal)
    self.stop_event.set()

  def start(self) -> None:
    for process in self.processes.values():
      print(f"starting {process.name}", flush=True)
      process.start()

  def run(self) -> None:
    sm = messaging.SubMaster(list(WATCHDOG_SERVICES), poll="roadCameraState")
    started_at = time.monotonic()
    samples = {service: deque() for service in WATCHDOG_SERVICES}
    last_can_rx_count: dict[int, int] = {}
    last_can_seen: dict[int, float] = {}
    try:
      while True:
        # If a signal arrived between iterations, skip the potentially blocking
        # poll but still inspect child exit codes before accepting the stop.
        if not self.stop_event.is_set():
          sm.update(1000)
        failed = [p for p in self.processes.values() if p.proc is not None and p.proc.exitcode is not None]
        if failed:
          details = ", ".join(f"{p.name}={p.proc.exitcode}" for p in failed if p.proc is not None)
          raise RuntimeError(f"dashcam process exited unexpectedly: {details}")
        if self.stop_event.is_set():
          break

        now = time.monotonic()
        for service in WATCHDOG_SERVICES:
          if sm.updated[service]:
            samples[service].append(now)
        stale = stale_data_paths(
          sm.seen, sm.recv_time, started_at=started_at, now=now,
          startup_timeout=self.watchdog_startup_timeout, stale_timeout=self.watchdog_stale_timeout,
        )
        if stale:
          raise RuntimeError(f"dashcam data path stalled: {', '.join(stale)}")
        if now - started_at > self.watchdog_startup_timeout:
          low_rate = low_rate_data_paths(samples, now=now)
          if low_rate:
            raise RuntimeError(f"dashcam data rate too low: {', '.join(low_rate)}")
        if self.required_can_buses and sm.updated["pandaStates"] and len(sm["pandaStates"]) == 1:
          state = sm["pandaStates"][0]
          can_states = (state.canState0, state.canState1, state.canState2)
          for bus in self.required_can_buses:
            count = can_states[bus].totalRxCnt
            previous = last_can_rx_count.get(bus)
            if count > 0 and (previous is None or count > previous):
              last_can_seen[bus] = now
            last_can_rx_count[bus] = count
        stale_buses = stale_can_buses(
          self.required_can_buses, last_can_seen, started_at=started_at, now=now,
          startup_timeout=self.watchdog_startup_timeout, stale_timeout=self.can_stale_timeout,
        )
        if stale_buses:
          raise RuntimeError(f"required CAN bus traffic stalled: {', '.join(map(str, stale_buses))}")
    except Exception:
      self.runtime_failed = True
      raise

  def shutdown(self) -> None:
    power_signal = getattr(signal, "SIGPWR", None)
    logger_process = self.processes[SHUTDOWN_ORDER[1]]
    if self.runtime_failed:
      # Use the fault signal as loggerd's actual stop signal. Sending SIGINT or
      # SIGPWR as a second stop could be delivered first and start finalization
      # before loggerd latches the incomplete-route marker.
      self.logger_stopping = True
      logger_process.stop(block=False, sig=RUNTIME_FAULT_SIGNAL)
    # Stop retention before loggerd can unlock the final segment. On a runtime
    # fault the sticky logger marker above must be delivered first, since a
    # blocking deleter stop could otherwise straddle a segment rotation.
    self.processes[SHUTDOWN_ORDER[0]].stop(block=True)
    # Close loggerd before publishers so it writes the end sentinel cleanly.
    # Read stop_signal after the blocking deleter stop so a SIGPWR received
    # during that window becomes loggerd's sole initial stop signal.
    logger_signal = power_signal if power_signal is not None and self.stop_signal == power_signal else None
    if not self.runtime_failed:
      logger_process.stop(block=False, sig=logger_signal)
      self.logger_stopping = True
    try:
      # Cover a SIGPWR delivered while the first nonblocking stop call was in
      # progress, before request_stop could forward directly.
      if not self.runtime_failed and power_signal is not None and self.stop_signal == power_signal and logger_signal != power_signal:
        logger_process.signal(power_signal)
      logger_process.stop(block=True, timeout=LOGGER_STOP_TIMEOUT)
    finally:
      self.logger_stopping = False
    for name in SHUTDOWN_ORDER[2:-1]:
      self.processes[name].stop(block=False)
    for name in SHUTDOWN_ORDER[2:-1]:
      self.processes[name].stop(block=True)
    self.processes[SHUTDOWN_ORDER[-1]].stop(block=True)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run passive openpilot camera and CAN recording")
  parser.add_argument("--camera", default="0", help="V4L2 index or stable device path (default: /dev/video0)")
  parser.add_argument("--log-root", type=Path, default=DEFAULT_STATE_ROOT / "media/0/realdata", help="directory in which loggerd creates route segments")
  parser.add_argument("--params-root", type=Path, default=DEFAULT_STATE_ROOT / "params", help="dedicated openpilot Params directory")
  parser.add_argument("--require-mount", type=Path, help="refuse to start unless this parent of log-root is a mounted filesystem")
  parser.add_argument("--prefix", default="cm5_dashcam", help="message/Params namespace; keep unique from another openpilot instance")
  parser.add_argument("--device-id", help="local 16-character logger ID; defaults to a hash of /etc/machine-id")
  parser.add_argument("--encoder", choices=("libx264",), default="libx264", help="FFmpeg H.264 encoder name (default: libx264)")
  parser.add_argument("--encoder-preset", default="ultrafast", help="libx264 CPU/quality preset (default: ultrafast)")
  parser.add_argument("--main-bitrate", type=positive_int, default=5_000_000, help="full-resolution H.264 bitrate in bits/s (default: 5000000)")
  parser.add_argument(
    "--min-free-gb", type=nonnegative_float, default=10.0,
    help="delete oldest completed segments below this free-space reserve (default: 10)",
  )
  parser.add_argument("--min-free-percent", type=percentage, default=10.0, help="delete oldest completed segments below this free percentage (default: 10)")
  parser.add_argument(
    "--panda-serial", default=os.getenv("DASHCAM_PANDA_SERIAL") or None,
    help="USB Panda serial; required when more than one Panda is attached",
  )
  parser.add_argument(
    "--panda-outage-timeout", type=nonnegative_float,
    default=float(os.getenv("DASHCAM_PANDA_OUTAGE_TIMEOUT", 30.0)),
    help="restart the recorder after this many seconds without Panda (default: 30)",
  )
  parser.add_argument(
    "--can-speeds", type=parse_can_speeds,
    default=parse_can_speeds(os.getenv("DASHCAM_CAN_SPEEDS", "500,500,500")),
    help="comma-separated arbitration bitrates for Panda buses 0,1,2 in kbit/s",
  )
  parser.add_argument(
    "--can-data-speeds", type=parse_can_data_speeds,
    default=parse_can_data_speeds(os.getenv("DASHCAM_CAN_DATA_SPEEDS", "2000,2000,2000")),
    help="comma-separated CAN-FD data bitrates for Panda buses 0,1,2 in kbit/s",
  )
  parser.add_argument(
    "--canfd-non-iso-buses", type=parse_bus_indices,
    default=parse_bus_indices(os.getenv("DASHCAM_CANFD_NON_ISO_BUSES", "")),
    help="comma-separated Panda bus indexes using non-ISO CAN-FD (default: none)",
  )
  parser.add_argument(
    "--required-can-buses", type=parse_bus_indices,
    default=parse_bus_indices(os.getenv("DASHCAM_REQUIRED_CAN_BUSES", "")),
    help="comma-separated bus indexes that must carry traffic; set for the target vehicle",
  )
  parser.add_argument(
    "--can-stale-timeout", type=positive_float,
    default=float(os.getenv("DASHCAM_CAN_STALE_TIMEOUT", 5.0)),
    help="restart if a required CAN bus is silent for this many seconds (default: 5)",
  )
  parser.add_argument(
    "--watchdog-startup-timeout", type=positive_float,
    default=float(os.getenv("DASHCAM_WATCHDOG_STARTUP_TIMEOUT", DEFAULT_WATCHDOG_STARTUP_TIMEOUT)),
    help="seconds allowed for camera, encoder, and Panda health streams to start (default: 30)",
  )
  parser.add_argument(
    "--watchdog-stale-timeout", type=positive_float,
    default=float(os.getenv("DASHCAM_WATCHDOG_STALE_TIMEOUT", DEFAULT_WATCHDOG_STALE_TIMEOUT)),
    help="restart after a camera, encoder, or Panda health stream stalls for this many seconds (default: 5)",
  )
  parser.add_argument("--skip-preflight", action="store_true", help="start despite missing camera or unbuilt binaries")
  return parser


def main() -> int:
  args = build_parser().parse_args()
  mount_errors = required_mount_errors(args.require_mount, args.log_root)
  if mount_errors:
    for error in mount_errors:
      print(f"error: {error}", file=sys.stderr)
    return 2

  configure_environment(args)
  processes = build_dashcam_processes()

  errors = preflight(processes, args.camera, args.log_root.expanduser().resolve(), args.panda_serial)
  if errors and not args.skip_preflight:
    for error in errors:
      print(f"error: {error}", file=sys.stderr)
    return 2

  try:
    instance_lock = acquire_instance_lock(args.params_root.expanduser().resolve() / "dashcam.lock")
  except DashcamAlreadyRunningError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2

  for interrupted in quarantine_interrupted_segments(args.log_root.expanduser().resolve()):
    print(f"warning: quarantined interrupted segment: {interrupted}", file=sys.stderr)

  runtime = DashcamRuntime(
    processes,
    watchdog_startup_timeout=args.watchdog_startup_timeout,
    watchdog_stale_timeout=args.watchdog_stale_timeout,
    required_can_buses=args.required_can_buses,
    can_stale_timeout=args.can_stale_timeout,
  )
  signal.signal(signal.SIGINT, runtime.request_stop)
  signal.signal(signal.SIGTERM, runtime.request_stop)
  if hasattr(signal, "SIGPWR"):
    signal.signal(signal.SIGPWR, runtime.request_stop)

  print(f"recording camera {camera_device(args.camera)} and Panda CAN to {os.environ['LOG_ROOT']}", flush=True)
  try:
    runtime.start()
    runtime.run()
  except Exception as exc:
    runtime.mark_failed()
    print(f"error: {exc}", file=sys.stderr)
    return_code = 1
  else:
    return_code = 0
  finally:
    runtime.shutdown()
    instance_lock.close()
  return return_code


if __name__ == "__main__":
  sys.exit(main())
