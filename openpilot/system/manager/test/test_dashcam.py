from argparse import Namespace
from collections import deque
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

import openpilot.system.manager.dashcam as dashcam
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.system.manager.dashcam import (
  DashcamAlreadyRunningError,
  DashcamRuntime,
  WATCHDOG_SERVICES,
  acquire_instance_lock,
  build_parser,
  camera_device,
  configure_environment,
  panda_selection_errors,
  quarantine_interrupted_segments,
  required_mount_errors,
  low_rate_data_paths,
  stale_can_buses,
  stale_data_paths,
)
from openpilot.system.manager.dashcam_process_config import DASHCAM_PROCESS_NAMES, USB_PANDAD_MODULE, build_dashcam_processes


def test_process_set_is_recording_only():
  processes = build_dashcam_processes()

  assert tuple(processes) == DASHCAM_PROCESS_NAMES
  assert processes["pandad"].cmdline[-2:] == ["-m", USB_PANDAD_MODULE]
  assert processes["logmessaged"].module == "openpilot.system.logmessaged"
  assert processes["webcamerad"].module == "openpilot.system.camerad.webcam.camerad"
  assert processes["loggerd"].cmdline == ["./loggerd"]
  assert processes["encoderd"].cmdline == ["./encoderd"]
  assert all(process.separate_process_group for process in processes.values())


def test_configure_environment_uses_dedicated_namespace(tmp_path: Path, monkeypatch):
  environment_keys = (
    "LOG_ROOT", "PARAMS_ROOT", "OPENPILOT_PREFIX", "ROAD_CAM", "USE_WEBCAM", "DASHCAM", "DONGLE_ID",
    "DASHCAM_ENCODER", "DASHCAM_ENCODER_PRESET", "DASHCAM_MAIN_BITRATE",
    "DASHCAM_MIN_FREE_BYTES", "DASHCAM_MIN_FREE_PERCENT",
    "DASHCAM_PANDA_SERIAL", "DASHCAM_PANDA_OUTAGE_TIMEOUT",
    "DASHCAM_CAN_SPEEDS", "DASHCAM_CAN_DATA_SPEEDS", "DASHCAM_CANFD_NON_ISO_BUSES",
    "DASHCAM_REQUIRED_CAN_BUSES", "DASHCAM_CAN_STALE_TIMEOUT",
  )
  for key in environment_keys:
    monkeypatch.setenv(key, "test-original")

  args = Namespace(
    log_root=tmp_path / "logs",
    params_root=tmp_path / "params",
    prefix="dashcam_test",
    camera=7,
    device_id="0123456789abcdef",
    encoder="libx264",
    encoder_preset="ultrafast",
    main_bitrate=4_000_000,
    min_free_gb=12.5,
    min_free_percent=15.0,
    panda_serial="abcdef123456",
    panda_outage_timeout=45.0,
    can_speeds=(125, 250, 500),
    can_data_speeds=(500, 1000, 2000),
    canfd_non_iso_buses=frozenset({1}),
    required_can_buses=frozenset({0, 2}),
    can_stale_timeout=7.5,
  )

  configure_environment(args)

  assert (tmp_path / "logs").is_dir()
  assert os.environ["LOG_ROOT"] == str((tmp_path / "logs").resolve())
  assert os.environ["ROAD_CAM"] == "7"
  assert os.environ["DASHCAM"] == "1"
  assert os.environ["DASHCAM_ENCODER"] == "libx264"
  assert os.environ["DASHCAM_ENCODER_PRESET"] == "ultrafast"
  assert os.environ["DASHCAM_MAIN_BITRATE"] == "4000000"
  assert os.environ["DASHCAM_MIN_FREE_BYTES"] == str(int(12.5 * 1024**3))
  assert os.environ["DASHCAM_MIN_FREE_PERCENT"] == "15.0"
  assert os.environ["DASHCAM_PANDA_SERIAL"] == "abcdef123456"
  assert os.environ["DASHCAM_PANDA_OUTAGE_TIMEOUT"] == "45.0"
  assert os.environ["DASHCAM_CAN_SPEEDS"] == "125,250,500"
  assert os.environ["DASHCAM_CAN_DATA_SPEEDS"] == "500,1000,2000"
  assert os.environ["DASHCAM_CANFD_NON_ISO_BUSES"] == "1"
  assert os.environ["DASHCAM_REQUIRED_CAN_BUSES"] == "0,2"
  assert os.environ["DASHCAM_CAN_STALE_TIMEOUT"] == "7.5"
  assert Params().get("DongleId") == "0123456789abcdef"


def test_camera_device_accepts_index_and_stable_path():
  assert camera_device("2") == Path("/dev/video2")
  assert camera_device("/dev/v4l/by-id/road-camera") == Path("/dev/v4l/by-id/road-camera")


def test_panda_selection_requires_one_or_requested_serial():
  assert panda_selection_errors(None, ["one"]) == []
  assert panda_selection_errors(None, []) == ["expected exactly one USB Panda, found 0: []"]
  assert panda_selection_errors("two", ["one", "two"]) == []
  assert panda_selection_errors("missing", ["one"]) == ["requested USB Panda missing is not connected (found: ['one'])"]


def test_panda_defaults_can_come_from_service_environment(monkeypatch):
  monkeypatch.setenv("DASHCAM_PANDA_SERIAL", "service-panda")
  monkeypatch.setenv("DASHCAM_PANDA_OUTAGE_TIMEOUT", "75")
  monkeypatch.setenv("DASHCAM_CAN_SPEEDS", "125,250,500")
  monkeypatch.setenv("DASHCAM_CAN_DATA_SPEEDS", "500,1000,2000")
  monkeypatch.setenv("DASHCAM_CANFD_NON_ISO_BUSES", "1")
  monkeypatch.setenv("DASHCAM_REQUIRED_CAN_BUSES", "0,2")
  monkeypatch.setenv("DASHCAM_CAN_STALE_TIMEOUT", "8")

  args = build_parser().parse_args([])

  assert args.panda_serial == "service-panda"
  assert args.panda_outage_timeout == 75.0
  assert args.can_speeds == (125, 250, 500)
  assert args.can_data_speeds == (500, 1000, 2000)
  assert args.canfd_non_iso_buses == frozenset({1})
  assert args.required_can_buses == frozenset({0, 2})
  assert args.can_stale_timeout == 8.0


def test_required_mount_rejects_non_mount(tmp_path: Path):
  assert required_mount_errors(tmp_path, tmp_path / "realdata") == [f"recording path is not a mount point: {tmp_path}"]
  assert required_mount_errors(Path("/"), tmp_path / "realdata") == []


def test_instance_lock_is_exclusive(tmp_path: Path):
  first = acquire_instance_lock(tmp_path / "dashcam.lock")
  try:
    with pytest.raises(DashcamAlreadyRunningError):
      acquire_instance_lock(tmp_path / "dashcam.lock")
  finally:
    first.close()


def test_interrupted_segments_are_quarantined(tmp_path: Path):
  complete = tmp_path / "00000000--0123456789--0"
  complete.mkdir()
  (complete / "rlog.zst").touch()
  interrupted = tmp_path / "00000000--0123456789--1"
  interrupted.mkdir()
  (interrupted / "rlog.lock").touch()

  moved = quarantine_interrupted_segments(tmp_path)

  assert complete.is_dir()
  assert moved == [tmp_path / ".interrupted" / interrupted.name]
  assert (moved[0] / "rlog.lock").is_file()
  assert not interrupted.exists()


def test_systemd_and_udev_files_keep_safety_boundaries():
  service = (Path(BASEDIR) / "openpilot/tools/cm5/systemd/openpilot-cm5-dashcam.service").read_text()
  rules = (Path(BASEDIR) / "openpilot/tools/cm5/udev/60-openpilot-cm5.rules").read_text()

  for setting in (
    "RequiresMountsFor=/mnt/dashcam", "KillMode=mixed", "NoNewPrivileges=true", "ProtectSystem=strict",
    "RestrictAddressFamilies=AF_UNIX AF_NETLINK", "StartLimitIntervalSec=0", "KillSignal=SIGPWR", "TimeoutStopSec=60",
  ):
    assert setting in service
  assert "--require-mount /mnt/dashcam" in service
  assert 'MODE:="0660"' in rules
  assert 'MODE="0666"' not in rules


def test_data_path_watchdog_has_startup_grace_and_detects_stalls():
  assert WATCHDOG_SERVICES == ("roadCameraState", "roadEncodeData", "qRoadEncodeData", "pandaStates")
  services = dict.fromkeys(WATCHDOG_SERVICES, False)
  recv_time = dict.fromkeys(services, 0.0)

  assert stale_data_paths(services, recv_time, started_at=0, now=20, startup_timeout=30, stale_timeout=5) == []
  assert stale_data_paths(services, recv_time, started_at=0, now=31, startup_timeout=30, stale_timeout=5) == list(services)

  services = dict.fromkeys(services, True)
  recv_time = {"roadCameraState": 28.0, "roadEncodeData": 20.0, "qRoadEncodeData": 29.0, "pandaStates": 29.0}
  assert stale_data_paths(services, recv_time, started_at=0, now=30, startup_timeout=30, stale_timeout=5) == ["roadEncodeData"]



def test_data_path_watchdog_rejects_sustained_low_rate():
  samples = {service: deque(i / 20 for i in range(101)) for service in WATCHDOG_SERVICES}
  samples["qRoadEncodeData"] = deque(i / 10 for i in range(51))

  assert low_rate_data_paths(samples, now=5.0) == ["qRoadEncodeData"]


def test_required_can_bus_watchdog_is_configurable():
  required = frozenset({0, 2})
  assert stale_can_buses(required, {}, started_at=0, now=20, startup_timeout=30, stale_timeout=5) == []
  assert stale_can_buses(required, {0: 29, 2: 20}, started_at=0, now=31, startup_timeout=30, stale_timeout=5) == [2]
  assert stale_can_buses(required, {}, started_at=0, now=31, startup_timeout=30, stale_timeout=5) == [0, 2]


class FakeProcess:
  def __init__(self, name: str, calls: list[tuple]):
    self.name = name
    self.calls = calls
    self.on_stop = None

  def stop(self, *, block: bool, sig: int | None = None, timeout: float = 5.0):
    self.calls.append(("stop", self.name, block, sig, timeout))
    if self.on_stop is not None:
      self.on_stop(block, sig)

  def signal(self, sig: int):
    self.calls.append(("signal", self.name, sig))


def test_stop_request_does_not_mask_exited_child(monkeypatch):
  monkeypatch.setattr(dashcam.messaging, "SubMaster", lambda *_args, **_kwargs: object())
  processes = {name: SimpleNamespace(name=name, proc=None) for name in DASHCAM_PROCESS_NAMES}
  processes["pandad"].proc = SimpleNamespace(exitcode=7)
  runtime = DashcamRuntime(processes)  # type: ignore[arg-type]
  runtime.request_stop(signal.SIGTERM)

  with pytest.raises(RuntimeError, match="pandad=7"):
    runtime.run()

  assert runtime.runtime_failed


def test_logger_closes_before_publishers():
  calls: list[tuple] = []
  processes = {name: FakeProcess(name, calls) for name in DASHCAM_PROCESS_NAMES}

  DashcamRuntime(processes).shutdown()  # type: ignore[arg-type]

  assert calls[:3] == [
    ("stop", "deleter", True, None, 5.0),
    ("stop", "loggerd", False, None, 5.0),
    ("stop", "loggerd", True, None, 20.0),
  ]
  assert calls[3:6] == [
    ("stop", "encoderd", False, None, 5.0),
    ("stop", "webcamerad", False, None, 5.0),
    ("stop", "pandad", False, None, 5.0),
  ]
  assert calls[-1] == ("stop", "logmessaged", True, None, 5.0)


def test_runtime_failure_marks_logger_segment_incomplete():
  calls: list[tuple] = []
  processes = {name: FakeProcess(name, calls) for name in DASHCAM_PROCESS_NAMES}
  runtime = DashcamRuntime(processes)  # type: ignore[arg-type]
  runtime.mark_failed()

  runtime.shutdown()

  fault_stop = ("stop", "loggerd", False, signal.SIGUSR1, 5.0)
  assert calls.count(fault_stop) == 1
  assert calls.index(fault_stop) < calls.index(("stop", "deleter", True, None, 5.0))
  assert not [call for call in calls if call[0] == "signal" and call[1] == "loggerd"]


def test_sigpwr_is_forwarded_only_to_logger(monkeypatch):
  power_signal = 30
  monkeypatch.setattr(signal, "SIGPWR", power_signal, raising=False)
  calls: list[tuple] = []
  processes = {name: FakeProcess(name, calls) for name in DASHCAM_PROCESS_NAMES}
  runtime = DashcamRuntime(processes)  # type: ignore[arg-type]

  runtime.request_stop(signal.SIGTERM)
  runtime.request_stop(power_signal)
  runtime.shutdown()

  assert ("stop", "loggerd", False, power_signal, 5.0) in calls
  assert not [call for call in calls if call[0] == "stop" and call[1] != "loggerd" and call[3] is not None]


def test_late_sigpwr_upgrades_logger_shutdown(monkeypatch):
  power_signal = 30
  monkeypatch.setattr(signal, "SIGPWR", power_signal, raising=False)
  calls: list[tuple] = []
  processes = {name: FakeProcess(name, calls) for name in DASHCAM_PROCESS_NAMES}
  runtime = DashcamRuntime(processes)  # type: ignore[arg-type]
  runtime.request_stop(signal.SIGTERM)
  processes["loggerd"].on_stop = lambda block, _sig: runtime.request_stop(power_signal) if not block else None

  runtime.shutdown()

  assert ("stop", "loggerd", False, None, 5.0) in calls
  assert ("signal", "loggerd", power_signal) in calls
