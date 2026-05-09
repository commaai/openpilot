"""Tests for WifiManager._ensure_wpa_supplicant attach-first bringup.

Pins the contract: we never kill a wpa_supplicant daemon we didn't spawn,
and we never spawn a second daemon if one is already answering on the
ctrl socket. Designed to coexist with a future systemd/OpenRC-managed
wpa_supplicant on tici.
"""
import re
import subprocess

import pytest

from openpilot.system.ui.lib import wifi_manager as wifi_manager_module
from openpilot.system.ui.lib import wpa_ctrl as wpa_ctrl_module
from openpilot.system.ui.lib.wifi_manager import WPA_AP_CONF, WPA_SUPPLICANT_CONF


def _patch_pgrep_false_then_true_for_sta(mocker):
  """Patch wpa_ctrl._wpa_supplicant_running to return False for the AP config
  and a False-then-True sequence for the STA config.

  ensure_wpa_supplicant calls _wpa_supplicant_running(WPA_AP_CONF) once (AP
  adoption gate, must be False to skip), then _wpa_supplicant_running(
  WPA_SUPPLICANT_CONF) twice: once as the fast-path gate (must be False so
  we fall through to spawn) and again in the post-spawn retry loop (must be
  True so the attach is allowed once our spawn completes)."""
  sta_calls = [0]
  def side_effect(conf):
    if conf == WPA_AP_CONF:
      return False
    sta_calls[0] += 1
    return sta_calls[0] > 1
  mocker.patch.object(wpa_ctrl_module, "_wpa_supplicant_running", side_effect=side_effect)


def _patch_bringup_sideeffects(wm, mocker):
  """Mock the side-effect calls in the spawn fallback path."""
  mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")
  mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=True)
  mocker.patch.object(wpa_ctrl_module.glob, "glob", return_value=[])
  mocker.patch.object(wpa_ctrl_module, "_unmanage_wlan0")
  _patch_pgrep_false_then_true_for_sta(mocker)
  mocker.patch.object(wpa_ctrl_module.time, "sleep")
  wm._exit = False
  # Fixture lacks scan/state threads — ensure GC-triggered __del__ → stop()
  # doesn't crash when _exit flips to False for the duration of the test.
  wm._scan_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
  wm._state_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
  return mock_run


def _patch_sta_daemon_alive(wm, mocker):
  """Fast-path setup: AP daemon absent, STA (our) daemon alive.

  Flips _exit=False so the fast path actually runs — the new should_exit()
  gate before AP/STA fast-attach would otherwise return None immediately.
  os.path.exists is mocked True so the wait-for-wlan0 loop exits without
  needing _exit=True to short-circuit it."""
  mocker.patch.object(wpa_ctrl_module, "_wpa_supplicant_running",
                      side_effect=lambda conf: conf == WPA_SUPPLICANT_CONF)
  mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=True)
  wm._exit = False
  # Fixture lacks scan/state threads — ensure GC-triggered __del__ → stop()
  # doesn't crash when _exit flips to False for the duration of the test.
  wm._scan_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
  wm._state_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))


class TestAttachFirst:
  def test_attach_success_skips_nmcli_pkill_and_spawn(self, wm, mocker):
    """Fast path: when our own daemon is already running, we attach
    directly. No nmcli, no pkill, no spawn — we do not disturb NM at all,
    because there's nothing to release. (pgrep for our AP-config daemon
    is allowed — it's read-only and gates the AP-mode fast path.)"""
    _patch_sta_daemon_alive(wm, mocker)
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")

    wm._ensure_wpa_supplicant()

    assert wm._ctrl is ctrl
    ctrl.open.assert_called_once()
    commands = [tuple(call.args[0]) for call in mock_run.call_args_list]
    mutating = [c for c in commands if c[:1] == ("sudo",) or "iptables" in c]
    assert mutating == [], f"fast path must not mutate anything: {mutating}"

  def test_attach_success_enables_networks(self, wm, mocker):
    """On attach, all networks are re-enabled (no RECONFIGURE — that would
    be rude on a system-managed daemon)."""
    _patch_sta_daemon_alive(wm, mocker)
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mocker.patch.object(wpa_ctrl_module.subprocess, "run")

    wm._ensure_wpa_supplicant()

    requests = [call.args[0] for call in ctrl.request.call_args_list]
    assert "ENABLE_NETWORK all" in requests
    assert "RECONFIGURE" not in requests

  def test_attach_success_swallows_request_errors(self, wm, mocker):
    """ENABLE_NETWORK failures must not fail the attach."""
    _patch_sta_daemon_alive(wm, mocker)
    ctrl = mocker.MagicMock()
    ctrl.request.side_effect = OSError("permission denied")
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mocker.patch.object(wpa_ctrl_module.subprocess, "run")

    # Should not raise
    wm._ensure_wpa_supplicant()

    assert wm._ctrl is ctrl

  def test_exit_signaled_skips_fast_attach_when_wlan0_already_up(self, wm, mocker):
    """P1 regression: if stop() is requested while wlan0 already exists and
    our STA daemon is already running, the fast-attach path must not bind
    ctrl. Otherwise _init_wifi_state can fire _handle_connected and start
    udhcpc after shutdown was requested. The wait-for-wlan0 loop's
    should_exit() check only fires when wlan0 is missing — once it's up the
    loop exits without calling should_exit(), so a dedicated gate before
    AP/STA fast-attach is required."""
    mocker.patch.object(wpa_ctrl_module, "_wpa_supplicant_running", return_value=True)
    mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=True)
    wpa_ctrl_cls = mocker.patch.object(wpa_ctrl_module, "WpaCtrl")
    # Fixture leaves _exit=True; that's exactly the post-stop() condition we want.

    result = wm._ensure_wpa_supplicant()

    assert result is None
    wpa_ctrl_cls.assert_not_called()

  def test_our_daemon_missing_falls_through_to_spawn(self, wm, mocker):
    """Regression guard: when no daemon we own is alive, we must NOT
    attach — even if the socket file still exists — because that would
    latch onto NM's wpa_supplicant, which NM is about to tear down."""
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mock_run = _patch_bringup_sideeffects(wm, mocker)

    wm._ensure_wpa_supplicant()

    # Fast-path pgrep for our STA config is False, so we never call WpaCtrl
    # before the spawn path runs.
    commands = [tuple(call.args[0]) for call in mock_run.call_args_list]
    spawn_cmds = [cmd for cmd in commands if cmd[:2] == ("sudo", "wpa_supplicant")]
    assert spawn_cmds, f"no spawn in {commands}"


class TestSpawnFallback:
  def test_no_owned_daemon_falls_through_to_spawn(self, wm, mocker):
    """When no daemon we own is running, we spawn wpa_supplicant with our
    config."""
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mock_run = _patch_bringup_sideeffects(wm, mocker)

    wm._ensure_wpa_supplicant()

    assert wm._ctrl is ctrl
    commands = [tuple(call.args[0]) for call in mock_run.call_args_list]
    spawn_cmds = [cmd for cmd in commands if cmd[:2] == ("sudo", "wpa_supplicant")]
    assert spawn_cmds, f"no spawn call in {commands}"
    assert WPA_SUPPLICANT_CONF in spawn_cmds[0]

  def test_spawn_fallback_never_bare_killalls_wpa_supplicant(self, wm, mocker):
    """Critical: we must never `sudo killall wpa_supplicant` — that would
    stomp on a system-managed daemon we're supposed to coexist with."""
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mock_run = _patch_bringup_sideeffects(wm, mocker)

    wm._ensure_wpa_supplicant()

    commands = [tuple(call.args[0]) for call in mock_run.call_args_list]
    for cmd in commands:
      if "killall" in cmd:
        assert "wpa_supplicant" not in cmd, \
          f"bare killall stomps on system daemons: {cmd}"

  def test_spawn_fallback_pkill_targets_our_config(self, wm, mocker):
    """The pkill fallback must target only processes running our config,
    so a baked system daemon on a different config survives. The CONF path
    is regex-escaped to avoid over-match on metacharacters."""
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mock_run = _patch_bringup_sideeffects(wm, mocker)

    wm._ensure_wpa_supplicant()

    commands = [tuple(call.args[0]) for call in mock_run.call_args_list]
    pkill_cmds = [cmd for cmd in commands if cmd[:2] == ("sudo", "pkill")]
    assert pkill_cmds, f"no pkill fallback in {commands}"
    escaped = re.escape(WPA_SUPPLICANT_CONF)
    assert any(escaped in arg for cmd in pkill_cmds for arg in cmd), \
      f"pkill doesn't narrow to our escaped config: {pkill_cmds}"

  def test_spawn_waits_for_nm_teardown_before_spawning(self, wm, mocker):
    """After _unmanage_wlan0, we must poll until the ctrl socket is gone
    before cleaning up or spawning. Otherwise we race NM's asynchronous
    wpa_supplicant deinit and spawn into a still-occupied ctrl_iface.
    This test pins that the wait loop runs and gives up once the socket
    is gone."""
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mock_run = _patch_bringup_sideeffects(wm, mocker)

    # Simulate the socket disappearing after 3 polls by sequencing
    # os.path.exists return values. /sys/class/net/wlan0 check at the top
    # of ensure_wpa_supplicant always returns True (fixture _exit=False
    # would loop forever otherwise, but the fixture sets _exit=True which
    # skips that loop). The ctrl-socket wait loop is the one we care about.
    exists_calls = iter([True, True, True, False] + [True] * 50)
    mocker.patch.object(wpa_ctrl_module.os.path, "exists",
                        side_effect=lambda _: next(exists_calls))

    wm._ensure_wpa_supplicant()

    commands = [tuple(call.args[0]) for call in mock_run.call_args_list]
    spawn_cmds = [cmd for cmd in commands if cmd[:2] == ("sudo", "wpa_supplicant")]
    assert spawn_cmds, f"no spawn after socket clear: {commands}"

  def test_spawn_then_reattach_loop(self, wm, mocker):
    """After spawn, the retry loop attaches successfully to the new daemon."""
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    _patch_bringup_sideeffects(wm, mocker)

    wm._ensure_wpa_supplicant()

    assert wm._ctrl is ctrl
    # ENABLE_NETWORK is called on the post-spawn attach
    requests = [call.args[0] for call in ctrl.request.call_args_list]
    assert "ENABLE_NETWORK all" in requests

  def test_post_spawn_refuses_foreign_daemon(self, wm, mocker):
    """Regression guard for codex P1: if NM never released the ctrl socket
    (wait loop timed out) and our spawn failed, the retry loop must NOT
    attach to the foreign daemon still occupying /var/run/wpa_supplicant/
    wlan0. _wpa_supplicant_running gates the attach; when it returns
    False the whole post-spawn window, we end with no ctrl bound."""
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    # Don't use the False-then-True helper — here pgrep must return False
    # for every call so neither the fast path nor the post-spawn retry
    # attach succeeds.
    mocker.patch.object(wpa_ctrl_module.subprocess, "run")
    mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=True)
    mocker.patch.object(wpa_ctrl_module.glob, "glob", return_value=[])
    mocker.patch.object(wpa_ctrl_module, "_unmanage_wlan0")
    mocker.patch.object(wpa_ctrl_module, "_wpa_supplicant_running", return_value=False)
    mocker.patch.object(wpa_ctrl_module.time, "sleep")
    wm._exit = False
    wm._scan_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
    wm._state_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))

    wm._ensure_wpa_supplicant()

    # We must not have latched onto the foreign daemon on the socket.
    assert wm._ctrl is not ctrl
    ctrl.open.assert_not_called()

  def test_returns_when_stop_requested_before_wlan0_appears(self, mocker):
    """If shutdown is requested while wlan0 is still absent (cold boot), the
    wait loop must bail without ever calling _unmanage_wlan0 / pkill / ip flush.
    Otherwise a WifiManager.stop() during early boot can mutate networking
    after teardown and race the next lifecycle."""
    mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=False)
    mocker.patch.object(wpa_ctrl_module.time, "sleep")
    mock_unmanage = mocker.patch.object(wpa_ctrl_module, "_unmanage_wlan0")
    mock_pkill = mocker.patch.object(wpa_ctrl_module, "_pkill_wpa_supplicant")
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")

    result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: True, "/tmp/ignored")

    assert result is None
    mock_unmanage.assert_not_called()
    mock_pkill.assert_not_called()
    mock_run.assert_not_called()

  def test_returns_when_stop_requested_after_wlan0_exists(self, mocker):
    """If wlan0 is already present but stop() is requested before bringup
    reaches the NM teardown path, _unmanage_wlan0 / pkill / ip flush / spawn
    must not run. Covers the common case where stop() arrives after the
    wait loop already observed wlan0."""
    mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=True)
    mocker.patch.object(wpa_ctrl_module.time, "sleep")
    mocker.patch.object(wpa_ctrl_module, "_wpa_supplicant_running", return_value=False)
    mock_unmanage = mocker.patch.object(wpa_ctrl_module, "_unmanage_wlan0")
    mock_pkill = mocker.patch.object(wpa_ctrl_module, "_pkill_wpa_supplicant")
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")

    result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: True, "/tmp/ignored")

    assert result is None
    mock_unmanage.assert_not_called()
    mock_pkill.assert_not_called()
    mock_run.assert_not_called()


class TestMonitorRespawn:
  def test_monitor_respawns_when_no_owned_daemon(self, wm, mocker):
    """If _ctrl is None and no owned daemon is running (crash/kill, or failed
    initial bringup), the monitor must call _ensure_wpa_supplicant so wifi
    doesn't stay dead forever."""
    wm._ctrl = None
    wm._exit = False
    wm._monitor_epoch = 0
    mocker.patch.object(wifi_manager_module, "_wpa_supplicant_running", return_value=False)
    mocker.patch.object(wifi_manager_module.time, "sleep")

    def stop_after_ensure():
      wm._exit = True
    ensure = mocker.patch.object(wm, "_ensure_wpa_supplicant", side_effect=stop_after_ensure)

    wm._monitor_state()

    ensure.assert_called()

  def test_monitor_does_not_respawn_when_owned_daemon_running(self, wm, mocker):
    """If an owned daemon is running, the monitor must attach — not spawn."""
    wm._ctrl = None
    wm._exit = False
    wm._monitor_epoch = 0
    mocker.patch.object(wifi_manager_module, "_wpa_supplicant_running", return_value=True)
    mocker.patch.object(wifi_manager_module.time, "sleep")
    ctrl = mocker.MagicMock()
    def stop_after_attach():
      wm._exit = True
      return ctrl
    mocker.patch.object(wifi_manager_module, "try_attach_ctrl", side_effect=stop_after_attach)
    ensure = mocker.patch.object(wm, "_ensure_wpa_supplicant")

    wm._monitor_state()

    ensure.assert_not_called()

  def test_monitor_clears_ctrl_on_error_so_next_iter_recovers(self, wm, mocker):
    """If WpaCtrlMonitor.open() raises (daemon crashed), the monitor must drop
    self._ctrl so the next iteration's _ctrl-is-None branch re-attaches or
    respawns. Otherwise a dead socket wedges the loop forever."""
    dead_ctrl = mocker.MagicMock()
    wm._ctrl = dead_ctrl
    wm._exit = False
    wm._monitor_epoch = 0
    mocker.patch.object(wifi_manager_module, "_wpa_supplicant_running", return_value=False)
    mocker.patch.object(wifi_manager_module.time, "sleep",
                        side_effect=lambda *_: setattr(wm, "_exit", True))
    mocker.patch.object(wifi_manager_module, "WpaCtrlMonitor",
                        side_effect=OSError("socket gone"))

    wm._monitor_state()

    assert wm._ctrl is None
    dead_ctrl.close.assert_called_once()

  def test_monitor_skips_recovery_while_tethering_active(self, wm, mocker):
    """During _start_tethering, _ctrl is closed and the STA daemon is killed
    before the AP daemon is up, with _tethering_active=True. The monitor must
    not spawn its own STA supplicant in that gap or it races AP bringup."""
    wm._ctrl = None
    wm._tethering_active = True
    wm._exit = False
    wm._monitor_epoch = 0
    pgrep = mocker.patch.object(wifi_manager_module, "_wpa_supplicant_running", return_value=False)
    mocker.patch.object(wifi_manager_module.time, "sleep",
                        side_effect=lambda *_: setattr(wm, "_exit", True))
    ensure = mocker.patch.object(wm, "_ensure_wpa_supplicant")

    wm._monitor_state()

    ensure.assert_not_called()
    # The pgrep that would gate spawn vs attach must also be skipped — we don't
    # touch the daemon at all while tethering is in transition.
    pgrep.assert_not_called()

  def test_monitor_respawns_after_repeated_attach_failures_with_pgrep_true(self, wm, mocker):
    """P1 regression: pgrep can keep finding the daemon after its ctrl socket has
    been deleted (NM-driven deinit, /var/run cleanup, etc.). try_attach_ctrl
    returning None forever then wedges wifi recovery. The monitor must give up
    on the stale process after a bounded number of attempts and call
    _ensure_wpa_supplicant so the kill-and-respawn path runs."""
    wm._ctrl = None
    wm._exit = False
    wm._monitor_epoch = 0
    mocker.patch.object(wifi_manager_module, "_wpa_supplicant_running", return_value=True)
    mocker.patch.object(wifi_manager_module, "try_attach_ctrl", return_value=None)
    mocker.patch.object(wifi_manager_module.time, "sleep")

    def stop_after_ensure():
      wm._exit = True
    ensure = mocker.patch.object(wm, "_ensure_wpa_supplicant", side_effect=stop_after_ensure)

    wm._monitor_state()

    ensure.assert_called()


class TestMultipleDaemonsPrevented:
  def test_attach_short_circuits_before_pkill_and_spawn(self, wm, mocker):
    """Regression guard: when our daemon is alive and attach succeeds, we
    must not mutate anything — no nmcli, no pkill, no spawn."""
    _patch_sta_daemon_alive(wm, mocker)
    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")
    mock_unmanage = mocker.patch.object(wpa_ctrl_module, "_unmanage_wlan0")

    wm._ensure_wpa_supplicant()

    commands = [tuple(call.args[0]) for call in mock_run.call_args_list]
    mutating = [c for c in commands if c[:1] == ("sudo",) or "iptables" in c]
    assert mutating == [], f"fast path must not mutate anything: {mutating}"
    mock_unmanage.assert_not_called()


class TestAPModeAdoption:
  def test_ap_attach_failure_aborts_instead_of_running_sta_cleanup(self, wm, mocker):
    """P2 regression: when WPA_AP_CONF is detected but try_attach_ctrl() fails
    transiently, falling through to STA cleanup kills dnsmasq / flushes wlan0
    / unmanages NM, tearing down a live hotspot. After bounded retries the
    function must abort (return None) so the caller's monitor retries the
    whole bringup, leaving the hotspot intact."""
    def pgrep_side_effect(conf):
      return conf == WPA_AP_CONF
    mocker.patch.object(wpa_ctrl_module, "_wpa_supplicant_running", side_effect=pgrep_side_effect)
    mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=True)
    mocker.patch.object(wpa_ctrl_module.time, "sleep")
    mocker.patch.object(wpa_ctrl_module, "try_attach_ctrl", return_value=None)
    mock_unmanage = mocker.patch.object(wpa_ctrl_module, "_unmanage_wlan0")
    mock_pkill = mocker.patch.object(wpa_ctrl_module, "_pkill_wpa_supplicant")
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")

    result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: False, "/tmp/ignored")

    assert result is None
    mock_unmanage.assert_not_called()
    mock_pkill.assert_not_called()
    spawn_cmds = [c for c in mock_run.call_args_list
                  if len(c.args[0]) >= 2 and c.args[0][0] == "sudo" and c.args[0][1] == "wpa_supplicant"]
    assert not spawn_cmds, f"AP-attach-failure must not trigger STA spawn: {spawn_cmds}"

  def test_ap_daemon_alive_adopts_before_sta_cleanup(self, wm, mocker):
    """P1 regression: if tethering was active before UI restart, the AP
    daemon (WPA_AP_CONF) still owns wlan0. ensure_wpa_supplicant must
    detect this and attach directly — otherwise the STA cleanup path kills
    dnsmasq / flushes wlan0 / pkills STA-config and tears down the
    hotspot while it's still running."""
    # Only the AP daemon exists. Our STA daemon check must still return False.
    def pgrep_side_effect(conf):
      return conf == WPA_AP_CONF
    mocker.patch.object(wpa_ctrl_module, "_wpa_supplicant_running", side_effect=pgrep_side_effect)
    mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=True)
    # _exit=False so the new should_exit() gate before AP adoption doesn't
    # short-circuit; thread mocks so __del__ → stop() doesn't crash on GC.
    wm._exit = False
    wm._scan_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
    wm._state_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))

    ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=ctrl)
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")
    mock_unmanage = mocker.patch.object(wpa_ctrl_module, "_unmanage_wlan0")

    wm._ensure_wpa_supplicant()

    # We attached to the AP daemon...
    assert wm._ctrl is ctrl
    ctrl.open.assert_called_once()
    # ...and did nothing destructive. Especially: no killall dnsmasq, no
    # ip addr flush wlan0, no pkill wpa_supplicant, no NM unmanage.
    commands = [tuple(call.args[0]) for call in mock_run.call_args_list]
    mutating = [c for c in commands if c[:1] == ("sudo",) or "iptables" in c]
    assert mutating == [], f"AP adoption must not mutate anything: {mutating}"
    mock_unmanage.assert_not_called()


def _patch_tethering_sideeffects(wm, mocker):
  """Silence all the subprocess / filesystem plumbing _start_tethering
  executes so we can exercise just the ctrl-socket bringup check."""
  mocker.patch.object(wifi_manager_module.subprocess, "run")
  popen = mocker.patch.object(wifi_manager_module.subprocess, "Popen")
  # Simulate a live dnsmasq so the post-spawn liveness gate doesn't trip.
  popen.return_value.poll.return_value = None
  mocker.patch.object(wifi_manager_module.time, "sleep")
  mocker.patch.object(wifi_manager_module.os, "open", return_value=0)

  class _DummyFd:
    def __enter__(self):
      return self

    def __exit__(self, *a):
      return False

    def write(self, _data):
      return None

  mocker.patch.object(wifi_manager_module.os, "fdopen", return_value=_DummyFd())
  wm._tethering_ssid = "weedle-test"
  wm._tethering_psk = "hotspot-psk-1234"
  wm._ipv4_forward = False
  wm._monitor_epoch = 0


class TestTetheringBringupVerification:
  def test_start_tethering_raises_when_attached_daemon_is_not_ap(self, wm, mocker):
    """If a surviving STA daemon still owns wlan0, our AP spawn fails but
    attach still succeeds against the old daemon. STATUS reports mode=station,
    so bringup must raise (so set_tethering_active's rollback runs)."""
    _patch_tethering_sideeffects(wm, mocker)
    sta_ctrl = mocker.MagicMock()
    sta_ctrl.request.return_value = "wpa_state=COMPLETED\nmode=station\nssid=NotOurs\n"
    mocker.patch.object(wifi_manager_module, "WpaCtrl", return_value=sta_ctrl)

    with pytest.raises(RuntimeError, match="did not take over wlan0"):
      wm._start_tethering()

    sta_ctrl.close.assert_called_once()
    # We must NOT publish the stale ctrl as our own — otherwise callers
    # (monitor thread, connect path) would keep talking to the STA daemon
    # thinking it's our AP.
    assert wm._ctrl is None

  def test_start_tethering_accepts_ap_mode(self, wm, mocker):
    """Happy path: STATUS says mode=AP → attach is accepted and state flips
    to CONNECTED."""
    _patch_tethering_sideeffects(wm, mocker)
    ap_ctrl = mocker.MagicMock()
    ap_ctrl.request.return_value = f"wpa_state=COMPLETED\nmode=AP\nssid={wm._tethering_ssid}\n"
    mocker.patch.object(wifi_manager_module, "WpaCtrl", return_value=ap_ctrl)

    wm._start_tethering()

    assert wm._ctrl is ap_ctrl
    ap_ctrl.close.assert_not_called()
    from openpilot.system.ui.lib.wifi_manager import ConnectStatus
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == wm._tethering_ssid

  def test_start_tethering_raises_when_status_request_fails(self, wm, mocker):
    """A daemon that answers the socket but errors on STATUS is also unsafe
    to keep — we raise and close the ctrl."""
    _patch_tethering_sideeffects(wm, mocker)
    broken_ctrl = mocker.MagicMock()
    broken_ctrl.request.side_effect = OSError("broken pipe")
    mocker.patch.object(wifi_manager_module, "WpaCtrl", return_value=broken_ctrl)

    with pytest.raises(RuntimeError, match="STATUS failed"):
      wm._start_tethering()

    broken_ctrl.close.assert_called_once()
    assert wm._ctrl is None

  def test_start_tethering_raises_when_dnsmasq_exits_immediately(self, wm, mocker):
    """dnsmasq failing to bind (busy port, stale leasefile, etc.) must surface
    as a bringup failure so rollback runs. Otherwise the UI would advertise
    tethering while clients can never obtain a DHCP lease."""
    _patch_tethering_sideeffects(wm, mocker)
    wifi_manager_module.subprocess.Popen.return_value.poll.return_value = 2
    wifi_manager_module.subprocess.Popen.return_value.returncode = 2

    with pytest.raises(RuntimeError, match="dnsmasq exited"):
      wm._start_tethering()

    assert wm._dnsmasq_proc is None
    assert wm._ctrl is None

  def test_start_tethering_raises_when_masquerade_install_fails(self, wm, mocker):
    """If the MASQUERADE insert fails, the AP comes up but clients can't reach
    the uplink. _start_tethering must raise so the caller's rollback runs,
    rather than reporting a healthy hotspot with broken sharing."""
    _patch_tethering_sideeffects(wm, mocker)

    def fake_run(cmd, *args, **kwargs):
      if (len(cmd) >= 4 and cmd[:2] == ["sudo", "iptables"]
          and "-A" in cmd and "MASQUERADE" in cmd and kwargs.get("check")):
        raise subprocess.CalledProcessError(1, cmd)
      return mocker.MagicMock(returncode=0)
    wifi_manager_module.subprocess.run.side_effect = fake_run

    with pytest.raises(subprocess.CalledProcessError):
      wm._start_tethering()

  def test_start_tethering_installs_source_based_masquerade(self, wm, mocker):
    """NAT rule must match on source subnet, not -o <iface>, so a mid-
    session uplink change (cable unplug, SIM drop, 3G→4G) doesn't strand
    tethered clients. Mirrors NetworkManager's shared-connection rule."""
    _patch_tethering_sideeffects(wm, mocker)
    ap_ctrl = mocker.MagicMock()
    ap_ctrl.request.return_value = f"wpa_state=COMPLETED\nmode=AP\nssid={wm._tethering_ssid}\n"
    mocker.patch.object(wifi_manager_module, "WpaCtrl", return_value=ap_ctrl)

    wm._start_tethering()

    from openpilot.system.ui.lib.wifi_manager import (
      TETHERING_NAT_COMMENT,
      TETHERING_SUBNET,
    )
    commands = [tuple(c.args[0]) for c in wifi_manager_module.subprocess.run.call_args_list]
    # The insert must be present in the exact NM-style form.
    insert_cmds = [c for c in commands
                   if len(c) >= 5 and c[0] == "sudo" and c[1] == "iptables"
                   and "-A" in c and "POSTROUTING" in c and "MASQUERADE" in c]
    assert insert_cmds, f"no MASQUERADE insert in commands: {commands}"
    cmd = insert_cmds[0]
    assert "-s" in cmd and TETHERING_SUBNET in cmd, f"missing source subnet: {cmd}"
    assert "!" in cmd and "-d" in cmd, f"missing negated destination: {cmd}"
    # And it must NOT bind to a specific uplink interface.
    assert "-o" not in cmd, f"MASQUERADE should not bind to -o <iface>: {cmd}"
    # Comment tag for iptables -S hygiene.
    assert TETHERING_NAT_COMMENT in cmd, f"missing comment tag: {cmd}"


class TestStopTetheringRollback:
  """_stop_tethering must restore STA mode via _ensure_wpa_supplicant: attach
  to our own STA daemon if one is alive, otherwise unmanage + spawn. It must
  never attach to NM's wpa_supplicant (NM would tear it down under us)."""

  def _patch_common(self, wm, mocker):
    mocker.patch.object(wifi_manager_module.time, "sleep")
    mocker.patch.object(wifi_manager_module, "_generate_wpa_conf")
    mocker.patch.object(wpa_ctrl_module, "_unmanage_wlan0")
    mocker.patch.object(wpa_ctrl_module.time, "sleep")
    mocker.patch.object(wpa_ctrl_module.os.path, "exists", return_value=True)
    mocker.patch.object(wpa_ctrl_module.glob, "glob", return_value=[])
    wm._dnsmasq_proc = None
    wm._tethering_upstream_iface = "wwan0"
    wm._monitor_epoch = 0
    wm._exit = False
    wm._scan_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
    wm._state_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))

  def test_rollback_attaches_to_our_own_surviving_daemon(self, wm, mocker):
    """If our STA daemon survived _start_tethering (AP bringup failed before
    killing it), rollback must attach to it without spawning a second."""
    self._patch_common(wm, mocker)
    _patch_sta_daemon_alive(wm, mocker)
    mocker.patch.object(wifi_manager_module.subprocess, "run")
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")
    existing_ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=existing_ctrl)

    wm._stop_tethering()

    assert wm._ctrl is existing_ctrl
    existing_ctrl.request.assert_any_call("ENABLE_NETWORK all")
    spawn_cmds = [c for c in mock_run.call_args_list
                  if len(c.args[0]) >= 2 and c.args[0][0] == "sudo" and c.args[0][1] == "wpa_supplicant"]
    assert not spawn_cmds, f"must not spawn when our daemon is alive: {spawn_cmds}"

  def test_rollback_spawns_when_no_daemon_we_own(self, wm, mocker):
    """If no daemon we own is running (regardless of whether NM's or a
    system daemon is present), rollback must unmanage NM and spawn our own
    STA daemon. Attaching to a foreign daemon would be torn down when NM
    releases wlan0."""
    self._patch_common(wm, mocker)
    _patch_pgrep_false_then_true_for_sta(mocker)
    mocker.patch.object(wifi_manager_module.subprocess, "run")
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")
    new_ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=new_ctrl)

    wm._stop_tethering()

    assert wm._ctrl is new_ctrl
    spawn_cmds = [c for c in mock_run.call_args_list
                  if len(c.args[0]) >= 2 and c.args[0][0] == "sudo" and c.args[0][1] == "wpa_supplicant"]
    assert spawn_cmds, "spawn must run when no owned daemon is alive"
    assert WPA_SUPPLICANT_CONF in spawn_cmds[0].args[0]

  def test_rollback_never_bare_killall_wpa_supplicant(self, wm, mocker):
    """Invariant: rollback must never `killall wpa_supplicant`. pkill is
    allowed because it's narrowed to our config path via regex."""
    self._patch_common(wm, mocker)
    _patch_pgrep_false_then_true_for_sta(mocker)
    mocker.patch.object(wifi_manager_module.subprocess, "run")
    mock_run = mocker.patch.object(wpa_ctrl_module.subprocess, "run")
    new_ctrl = mocker.MagicMock()
    mocker.patch.object(wpa_ctrl_module, "WpaCtrl", return_value=new_ctrl)

    wm._stop_tethering()

    for call in mock_run.call_args_list:
      cmd = call.args[0]
      if "killall" in cmd:
        assert "wpa_supplicant" not in cmd, f"bare killall would stomp system daemon: {cmd}"
