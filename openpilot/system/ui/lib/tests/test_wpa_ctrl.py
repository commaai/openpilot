from pathlib import Path
import socket
import tempfile
import threading
import time
from typing import cast
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from openpilot.system.ui.lib import wpa_ctrl as wpa_ctrl_module
from openpilot.system.ui.lib.wpa_ctrl import (
  SecurityType,
  WpaCtrl,
  decode_ssid,
  normalize_ssid,
  parse_event_network_id,
  parse_event_ssid,
  parse_scan_results,
  parse_status,
  flags_to_security_type,
  dbm_to_percent,
)


class TestParseStatus(TestCase):
  def test_values(self):
    cases = (
      ("", {}),
      ("ssid=My=Network\n", {"ssid": "My=Network"}),
      ('ssid=My \\"Home\\"\n', {"ssid": 'My "Home"'}),
      (
        "wpa_state=COMPLETED\nssid=MyNet\nip_address=10.0.0.5\n",
        {"wpa_state": "COMPLETED", "ssid": "MyNet", "ip_address": "10.0.0.5"},
      ),
      (
        "wpa_state=COMPLETED\nssid=caf\\xc3\\xa9\nip_address=10.0.0.5\n",  # codespell:ignore caf
        {"wpa_state": "COMPLETED", "ssid": "café", "ip_address": "10.0.0.5"},
      ),
      (
        "bssid=00:11:22:33:44:55\nssid=\\x41\n",
        {"bssid": "00:11:22:33:44:55", "ssid": "A"},
      ),
    )
    for raw, expected in cases:
      with self.subTest(raw=raw):
        assert parse_status(raw) == expected


class TestParseEventSsid(TestCase):
  def test_values(self):
    cases = (
      ('id=0 ssid="MyNetwork" reason=WRONG_KEY', "MyNetwork"),
      ("id=0 reason=WRONG_KEY", None),
      (r'id=0 ssid="My \"Home\"" reason=WRONG_KEY', 'My "Home"'),
      (r'id=0 ssid="caf\xc3\xa9" reason=WRONG_KEY', "café"),  # codespell:ignore caf
    )
    for event, expected in cases:
      with self.subTest(event=event):
        assert parse_event_ssid(event) == expected


class TestParseEventNetworkId(TestCase):
  def test_values(self):
    cases = (
      ('id=42 ssid="MyNetwork" reason=WRONG_KEY', "42"),
      ('ssid="MyNetwork" reason=WRONG_KEY', None),
    )
    for event, expected in cases:
      with self.subTest(event=event):
        assert parse_event_network_id(event) == expected


class TestFlagsToSecurityType(TestCase):
  def test_security_types(self):
    cases = (
      ("[WPA2-PSK-CCMP][ESS]", SecurityType.WPA),
      ("[RSN-PSK-CCMP]", SecurityType.WPA),
      ("[WPA-PSK-TKIP]", SecurityType.WPA),
      ("[WPA2-PSK-CCMP][SAE]", SecurityType.WPA),
      ("[RSN-PSK-CCMP][SAE-CCMP]", SecurityType.WPA),
      ("[WPA2-PSK+SAE-CCMP][ESS]", SecurityType.WPA),
      ("[WPA2-PSK+EAP-CCMP][ESS]", SecurityType.WPA),
      ("[WPA2-PSK-SHA256+SAE-CCMP][ESS]", SecurityType.UNSUPPORTED),
      ("[WPA2-PSK-SHA256-CCMP][ESS]", SecurityType.UNSUPPORTED),
      ("[WPA-PSK-SHA256-TKIP][ESS]", SecurityType.UNSUPPORTED),
      ("[WPA2-PSK-CCMP][WPA2-PSK-SHA256-CCMP][ESS]", SecurityType.WPA),
      ("[SAE]", SecurityType.UNSUPPORTED),
      ("[SAE-CCMP]", SecurityType.UNSUPPORTED),
      ("[OWE-CCMP][ESS]", SecurityType.UNSUPPORTED),
      ("[OWE-TRANSITION][ESS]", SecurityType.UNSUPPORTED),
      ("[DPP][ESS]", SecurityType.UNSUPPORTED),
      ("[OSEN][ESS]", SecurityType.UNSUPPORTED),
      ("[FILS-SHA256][ESS]", SecurityType.UNSUPPORTED),
      ("[ESS]", SecurityType.OPEN),
      ("", SecurityType.OPEN),
      ("[WPA2-EAP-CCMP]", SecurityType.UNSUPPORTED),
      ("[802.1X]", SecurityType.UNSUPPORTED),
    )
    for flags, expected in cases:
      assert flags_to_security_type(flags) == expected


class TestDbmToPercent(TestCase):
  def test_values(self):
    cases = ((-120, 0), (-100, 0), (-92, 14), (-81, 32), (-74, 44), (-70, 50), (-40, 100), (-30, 100))
    for dbm, expected in cases:
      with self.subTest(dbm=dbm):
        assert dbm_to_percent(dbm) == expected


class TestParseScanResults(TestCase):
  HEADER = "bssid / frequency / signal level / flags / ssid\n"

  def test_basic(self):
    raw = self.HEADER + "00:11:22:33:44:55\t2437\t-65\t[WPA2-PSK-CCMP][ESS]\tMyNetwork\n"
    results = parse_scan_results(raw)
    assert len(results) == 1
    r = results[0]
    assert r.bssid == "00:11:22:33:44:55"
    assert r.freq == 2437
    assert r.signal == -65
    assert r.ssid == "MyNetwork"

  def test_ssid_values(self):
    cases = (
      ("00:11:22:33:44:55\t2437\t-65\t[ESS]\t\n", ""),
      (f"00:11:22:33:44:55\t2437\t-65\t[ESS]\t{'\\x00' * 32}\n", ""),
      ('00:11:22:33:44:55\t2437\t-65\t[ESS]\tcaf\\xc3\\xa9 \\"home\\"\n', 'café "home"'),
      ("00:11:22:33:44:55\t2437\t-65\t[ESS]\tMyNet \n", "MyNet "),
      ("00:11:22:33:44:55\t2437\t-65\t[ESS]\n", ""),
      ("garbage\n00:11:22:33:44:55\t2437\t-65\t[ESS]\tGood\n", "Good"),
    )
    for body, expected in cases:
      with self.subTest(body=body):
        results = parse_scan_results(self.HEADER + body)
        assert len(results) == 1
        assert results[0].ssid == expected

class TestDecodeSsid(TestCase):
  def test_values(self):
    cases = (
      ("MyNetwork", "MyNetwork"),
      ("", ""),
      ("\\x41\\x42", "AB"),
      ("caf\\xc3\\xa9", "café"),  # codespell:ignore caf
      ("\\xe6\\x97\\xa5\\xe6\\x9c\\xac", "日本"),
      ("\\xf0\\x9f\\x9a\\x97", "🚗"),
      ("\\x1Z", "\x01Z"),
      ("\\xA", "\x0a"),
      ("\\xGZ", "GZ"),
      ("\\101", "A"),
      ("\\0X", "\x00X"),
      ("\\78", "\x078"),
      ("\\\\", "\\"),
      ('\\"', '"'),
      ("\\n", "\n"),
      ("\\r", "\r"),
      ("\\t", "\t"),
      ("\\e", "\x1b"),
      ("a\\qb", "aqb"),
      ("abc\\", "abc"),
      ("\\x00" * 32, ""),
      ("A\\x00B", "A\x00B"),
    )
    for encoded, expected in cases:
      with self.subTest(encoded=encoded):
        assert decode_ssid(encoded) == expected

  def test_invalid_utf8_preserves_identity(self):
    for encoded, expected in (("\\xFF", b"\xff"), ("\\x80", b"\x80")):
      with self.subTest(encoded=encoded):
        decoded = decode_ssid(encoded)
        assert decoded.encode("utf-8", errors="surrogateescape") == expected
        assert normalize_ssid(decoded) == "\ufffd"


class TestWpaConfig(TestCase):
  def setUp(self):
    self.path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "wpa_supplicant.conf"

  def generate(self, ssid, profile):
    store = MagicMock()
    store.get_profiles.return_value = [(ssid, profile)]
    wpa_ctrl_module.generate_wpa_conf(store, str(self.path))
    return self.path.read_text()

  def test_emits_saved_network_priority(self):
    assert "  priority=42\n" in self.generate("Preferred", {"psk": "password123", "hidden": False, "priority": 42})

  def test_emits_saved_profile_identifier(self):
    assert '  id_str="11111111-1111-1111-1111-111111111111"\n' in self.generate(
      "Preferred", {"psk": "password123", "uuid": "11111111-1111-1111-1111-111111111111"},
    )

  def test_grants_control_access_to_netdev_group(self):
    assert "ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n" in self.generate("Test", {"psk": "password123"})

  def test_emits_saved_bssid_restriction(self):
    assert "  bssid=00:11:22:33:44:55\n" in self.generate(
      "Pinned", {"psk": "password123", "bssid": "00:11:22:33:44:55"},
    )

  def test_encodes_control_characters_in_ssid_losslessly(self):
    assert f"  ssid={b'Line\nBreak\r'.hex()}\n" in self.generate("Line\nBreak\r", {"psk": "password123"})


class _RacySock:
  def __init__(self):
    self._lock = threading.Lock()
    self._last_sent: bytes = b""

  def send(self, data: bytes):
    with self._lock:
      self._last_sent = data

  def recv(self, _bufsize: int) -> bytes:
    time.sleep(0.005)
    with self._lock:
      return b"REPLY:" + self._last_sent


class TestWpaCtrlRequestSerialization(TestCase):
  def test_request_pairs_reply_with_command_under_concurrency(self):
    ctrl = WpaCtrl()
    ctrl._sock = cast(socket.socket, _RacySock())

    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def worker(cmd: str):
      try:
        results[cmd] = ctrl.request(cmd)
      except BaseException as exc:
        errors.append(exc)

    threads = [threading.Thread(target=worker, args=(cmd,))
               for cmd in ("STATUS", "SCAN_RESULTS", "LIST_NETWORKS", "PING")]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=5)

    assert not errors, errors
    for cmd in ("STATUS", "SCAN_RESULTS", "LIST_NETWORKS", "PING"):
      assert results[cmd] == f"REPLY:{cmd}", \
        f"concurrent request mispaired reply for {cmd}: {results[cmd]}"

    ctrl._sock = None

  def test_interrupt_wakes_blocked_request(self):
    request_started = threading.Event()
    interrupted = threading.Event()

    class BlockingSock:
      def send(self, _data):
        request_started.set()

      def recv(self, _bufsize):
        interrupted.wait(1)
        raise OSError("interrupted")

      def shutdown(self, _how):
        interrupted.set()

      def close(self):
        pass

    ctrl = WpaCtrl()
    ctrl._sock = cast(socket.socket, BlockingSock())
    errors = []

    def run_request():
      try:
        ctrl.request("STATUS")
      except OSError as exc:
        errors.append(str(exc))

    request = threading.Thread(target=run_request)
    request.start()
    assert request_started.wait(1)

    ctrl.interrupt()
    request.join(1)

    assert interrupted.is_set()
    assert not request.is_alive()
    assert errors == ["interrupted"]
    ctrl._sock = None


class TestNetworkManagerCompatibility(TestCase):
  def test_unmanage_skips_when_nmcli_is_absent(self):
    with (
      patch.object(wpa_ctrl_module.shutil, "which", return_value=None),
      patch.object(wpa_ctrl_module.subprocess, "run") as run,
    ):
      assert wpa_ctrl_module._unmanage_wlan0()

      run.assert_not_called()

  def test_unmanage_uses_discovered_nmcli(self):
    with (
      patch.object(wpa_ctrl_module.shutil, "which", return_value="/usr/bin/nmcli"),
      patch.object(wpa_ctrl_module.subprocess, "run") as run,
    ):
      run.return_value.returncode = 0

      assert wpa_ctrl_module._unmanage_wlan0()
      run.assert_called_once_with(
        ["sudo", "/usr/bin/nmcli", "dev", "set", "wlan0", "managed", "no"],
        capture_output=True,
      )

  def test_unmanage_failure_is_nonfatal(self):
    with (
      patch.object(wpa_ctrl_module.shutil, "which", return_value="/usr/bin/nmcli"),
      patch.object(wpa_ctrl_module.subprocess, "run") as run,
    ):
      run.return_value.returncode = 10

      assert not wpa_ctrl_module._unmanage_wlan0()


class TestTetheringDnsmasqOwnership(TestCase):
  def test_process_patterns_do_not_match_sudo_parent(self):
    assert wpa_ctrl_module.TETHERING_DNSMASQ_PATTERN.startswith("^dnsmasq ")

    with patch.object(wpa_ctrl_module.subprocess, "run", return_value=MagicMock(returncode=0)) as run:
      assert wpa_ctrl_module.wpa_supplicant_running(wpa_ctrl_module.WPA_SUPPLICANT_CONF)
      running_pattern = run.call_args.args[0][2]
      wpa_ctrl_module.stop_wpa_supplicant(wpa_ctrl_module.WPA_SUPPLICANT_CONF)
      kill_pattern = run.call_args.args[0][-1]

    assert running_pattern == kill_pattern
    assert running_pattern.startswith("^wpa_supplicant ")

  def test_stop_targets_only_openpilot_tethering(self):
    with patch.object(wpa_ctrl_module.subprocess, "run") as run:
      wpa_ctrl_module.stop_tethering_dnsmasq()

      run.assert_called_once_with(
        ["sudo", "pkill", "-f", wpa_ctrl_module.TETHERING_DNSMASQ_PATTERN],
        check=False,
      )


class TestSupplicantBringup(TestCase):
  def test_reconciles_existing_station_configuration(self):
    ctrl = MagicMock()
    ctrl.request.side_effect = lambda command: (
      "wpa_state=COMPLETED\nmode=station\nssid=TestNet\n" if command == "STATUS" else "OK"
    )
    station_reconfigured = MagicMock()
    with (
      patch.object(wpa_ctrl_module.os.path, "exists", return_value=True),
      patch.object(
        wpa_ctrl_module,
        "wpa_supplicant_running",
        side_effect=lambda conf: conf == wpa_ctrl_module.WPA_SUPPLICANT_CONF,
      ),
      patch.object(wpa_ctrl_module, "try_attach_ctrl", return_value=ctrl),
      patch.object(wpa_ctrl_module, "_unmanage_wlan0") as unmanage,
      patch.object(wpa_ctrl_module, "stop_wpa_supplicant") as kill,
      patch.object(wpa_ctrl_module.subprocess, "run") as run,
    ):
      result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: False, station_reconfigured)

      assert result is ctrl
      assert call("RECONFIGURE") in ctrl.request.call_args_list
      station_reconfigured.assert_called_once_with("TestNet")
      unmanage.assert_not_called()
      kill.assert_not_called()
      run.assert_not_called()

  def test_failed_station_reconciliation_restarts_daemon(self):
    stale_ctrl = MagicMock()
    stale_ctrl.request.return_value = "FAIL"
    fresh_ctrl = MagicMock()
    station_checks = 0

    def running(conf):
      nonlocal station_checks
      if conf == wpa_ctrl_module.WPA_AP_CONF:
        return False
      station_checks += 1
      return station_checks in (1, 3)

    with (
      patch.object(
        wpa_ctrl_module.os.path,
        "exists",
        side_effect=lambda path: path == "/sys/class/net/wlan0",
      ),
      patch.object(wpa_ctrl_module, "wpa_supplicant_running", side_effect=running),
      patch.object(wpa_ctrl_module, "try_attach_ctrl", side_effect=[stale_ctrl, fresh_ctrl]),
      patch.object(wpa_ctrl_module, "_unmanage_wlan0", return_value=True),
      patch.object(wpa_ctrl_module, "stop_wpa_supplicant") as kill,
      patch.object(wpa_ctrl_module, "stop_tethering_dnsmasq"),
      patch.object(wpa_ctrl_module.time, "sleep"),
      patch.object(wpa_ctrl_module.subprocess, "run") as run,
    ):
      result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: False)

    assert result is fresh_ctrl
    stale_ctrl.close.assert_called_once()
    kill.assert_called_once_with(wpa_ctrl_module.WPA_SUPPLICANT_CONF)
    assert [
      "sudo", "wpa_supplicant", "-B", "-i", "wlan0",
      "-c", wpa_ctrl_module.WPA_SUPPLICANT_CONF, "-D", "nl80211",
    ] in [item.args[0] for item in run.call_args_list]

  def test_attaches_existing_hotspot_before_station_cleanup(self):
    ctrl = MagicMock()
    with (
      patch.object(wpa_ctrl_module.os.path, "exists", return_value=True),
      patch.object(
        wpa_ctrl_module,
        "wpa_supplicant_running",
        side_effect=lambda conf: conf == wpa_ctrl_module.WPA_AP_CONF,
      ),
      patch.object(wpa_ctrl_module, "try_attach_ctrl", return_value=ctrl),
      patch.object(wpa_ctrl_module, "_unmanage_wlan0") as unmanage,
      patch.object(wpa_ctrl_module, "stop_wpa_supplicant") as kill,
    ):
      result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: False)

      assert result is ctrl
      unmanage.assert_not_called()
      kill.assert_not_called()

  def test_unreachable_hotspot_falls_back_to_station_bringup(self):
    ctrl = MagicMock()
    abandoned_ap = MagicMock()
    ap_running = True
    station_running = False

    def running(conf):
      return ap_running if conf == wpa_ctrl_module.WPA_AP_CONF else station_running

    def stop(conf):
      nonlocal ap_running, station_running
      if conf == wpa_ctrl_module.WPA_AP_CONF:
        ap_running = False
      else:
        station_running = False

    def run(command, **_kwargs):
      nonlocal station_running
      if command[:2] == ["sudo", "wpa_supplicant"]:
        station_running = True
      return MagicMock(returncode=0)

    with (
      patch.object(
        wpa_ctrl_module.os.path,
        "exists",
        side_effect=lambda path: path == "/sys/class/net/wlan0",
      ),
      patch.object(wpa_ctrl_module, "wpa_supplicant_running", side_effect=running),
      patch.object(wpa_ctrl_module, "try_attach_ctrl", side_effect=lambda: None if ap_running else ctrl),
      patch.object(wpa_ctrl_module, "_unmanage_wlan0", return_value=True),
      patch.object(wpa_ctrl_module, "stop_wpa_supplicant", side_effect=stop) as kill,
      patch.object(wpa_ctrl_module, "stop_tethering_dnsmasq"),
      patch.object(wpa_ctrl_module.time, "sleep"),
      patch.object(wpa_ctrl_module.subprocess, "run", side_effect=run),
    ):
      result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: False, on_abandoned_ap=abandoned_ap)

    assert result is ctrl
    abandoned_ap.assert_called_once()
    assert call(wpa_ctrl_module.WPA_AP_CONF) in kill.call_args_list
    assert call(wpa_ctrl_module.WPA_SUPPLICANT_CONF) in kill.call_args_list

  def test_spawns_owned_station_daemon(self):
    ctrl = MagicMock()
    station_checks = 0

    def running(conf):
      nonlocal station_checks
      if conf == wpa_ctrl_module.WPA_AP_CONF:
        return False
      station_checks += 1
      return station_checks > 1

    with (
      patch.object(
        wpa_ctrl_module.os.path,
        "exists",
        side_effect=lambda path: path == "/sys/class/net/wlan0",
      ),
      patch.object(wpa_ctrl_module, "wpa_supplicant_running", side_effect=running),
      patch.object(wpa_ctrl_module, "try_attach_ctrl", return_value=ctrl),
      patch.object(wpa_ctrl_module, "_unmanage_wlan0", return_value=True),
      patch.object(wpa_ctrl_module, "stop_wpa_supplicant") as kill,
      patch.object(wpa_ctrl_module, "stop_tethering_dnsmasq"),
      patch.object(wpa_ctrl_module.time, "sleep"),
      patch.object(wpa_ctrl_module.subprocess, "run") as run,
    ):
      result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: False)

      assert result is ctrl
      kill.assert_called_once_with(wpa_ctrl_module.WPA_SUPPLICANT_CONF)
      assert [
        "sudo", "wpa_supplicant", "-B", "-i", "wlan0",
        "-c", wpa_ctrl_module.WPA_SUPPLICANT_CONF, "-D", "nl80211",
      ] in [item.args[0] for item in run.call_args_list]

  def test_failed_networkmanager_handoff_does_not_mutate_interface(self):
    with (
      patch.object(
        wpa_ctrl_module.os.path,
        "exists",
        side_effect=lambda path: path == "/sys/class/net/wlan0",
      ),
      patch.object(wpa_ctrl_module, "wpa_supplicant_running", return_value=False),
      patch.object(wpa_ctrl_module, "_unmanage_wlan0", return_value=False),
      patch.object(wpa_ctrl_module, "stop_wpa_supplicant") as kill,
      patch.object(wpa_ctrl_module, "stop_tethering_dnsmasq") as stop_dnsmasq,
      patch.object(wpa_ctrl_module.time, "sleep"),
      patch.object(wpa_ctrl_module.subprocess, "run") as run,
    ):
      result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: False)

    assert result is None
    kill.assert_not_called()
    stop_dnsmasq.assert_not_called()
    run.assert_not_called()

  def test_foreign_control_socket_is_not_attached_after_handoff(self):
    def exists(path):
      return path in ("/sys/class/net/wlan0", "/var/run/wpa_supplicant/wlan0")

    with (
      patch.object(wpa_ctrl_module.os.path, "exists", side_effect=exists),
      patch.object(wpa_ctrl_module, "wpa_supplicant_running", return_value=False),
      patch.object(wpa_ctrl_module, "try_attach_ctrl") as attach,
      patch.object(wpa_ctrl_module, "_unmanage_wlan0", return_value=True),
      patch.object(wpa_ctrl_module, "stop_wpa_supplicant"),
      patch.object(wpa_ctrl_module, "stop_tethering_dnsmasq"),
      patch.object(wpa_ctrl_module.time, "sleep"),
      patch.object(wpa_ctrl_module.subprocess, "run"),
    ):
      result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: False)

    assert result is None
    attach.assert_not_called()

  def test_exit_before_interface_mutation(self):
    with (
      patch.object(wpa_ctrl_module.os.path, "exists", return_value=True),
      patch.object(wpa_ctrl_module, "_unmanage_wlan0") as unmanage,
      patch.object(wpa_ctrl_module, "stop_wpa_supplicant") as kill,
      patch.object(wpa_ctrl_module.subprocess, "run") as run,
    ):
      result = wpa_ctrl_module.ensure_wpa_supplicant(lambda: True)

      assert result is None
      unmanage.assert_not_called()
      kill.assert_not_called()
      assert all(item.args[0][0] == "pgrep" for item in run.call_args_list)
