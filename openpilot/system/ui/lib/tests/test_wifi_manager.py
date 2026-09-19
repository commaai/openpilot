import configparser
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import types
import unittest.mock
from collections.abc import Callable
from dataclasses import replace
from typing import cast

from openpilot.common.test import OpenpilotTestCase, Mocker
from openpilot.system.ui.lib import wifi_manager
from openpilot.system.ui.lib.wifi_manager import (WpaCtrl, SecurityType, decode_ssid, parse_scan_results, dbm_to_percent,
                                                  security_type_from_flags, wpa_psk)

SCAN_HEADER = "bssid / frequency / signal level / flags / ssid\n"


class FakeSupplicant:
  # unix datagram server speaking the wpa_supplicant control protocol from a scripted reply table
  def __init__(self, path: str):
    self.path = path
    self.requests: list[str] = []
    self.replies: dict[str, str | Callable[[str], str]] = {"PING": "PONG"}
    self.status: dict[str, str] = {"wpa_state": "DISCONNECTED"}
    self.networks: dict[int, dict[str, str]] = {}
    self.on_request: Callable[[str], None] | None = None
    self._attached: list[bytes] = []
    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    self._sock.bind(path)
    self._sock.settimeout(0.1)
    self._exit = False
    self._thread = threading.Thread(target=self._serve, daemon=True)
    self._thread.start()

  def _reply(self, cmd: str) -> str:
    if cmd == "STATUS":
      return "".join(f"{k}={v}\n" for k, v in self.status.items())
    if cmd == "ADD_NETWORK":
      nid = max(self.networks, default=-1) + 1
      self.networks[nid] = {}
      return f"{nid}\n"
    if cmd.startswith("SET_NETWORK "):
      _, nid, key, value = cmd.split(" ", 3)
      self.networks[int(nid)][key] = value
      return "OK\n"
    if cmd.startswith("REMOVE_NETWORK "):
      reply = self.replies.get(cmd)
      if reply is not None:
        return reply if isinstance(reply, str) else reply(cmd)
      self.networks.pop(int(cmd.split()[1]), None)
      return "OK\n"
    if cmd == "LIST_NETWORKS":
      lines = ["network id / ssid / bssid / flags"]
      for nid, net in self.networks.items():
        lines.append(f"{nid}\t{bytes.fromhex(net.get('ssid', '')).decode()}\tany\t")
      return "\n".join(lines) + "\n"
    reply = self.replies.get(cmd)
    if reply is None:
      reply = self.replies.get(cmd.split(" ", 1)[0], "OK\n")
    return reply if isinstance(reply, str) else reply(cmd)

  def _serve(self):
    while not self._exit:
      try:
        data, addr = self._sock.recvfrom(65536)
      except TimeoutError:
        continue
      cmd = data.decode()
      if cmd == "ATTACH":
        self._attached.append(addr)
        self._sock.sendto(b"OK\n", addr)
        continue
      self.requests.append(cmd)
      if self.on_request is not None:
        self.on_request(cmd)
      reply = self._reply(cmd)
      self._sock.sendto(reply.encode() if reply.endswith("\n") else (reply + "\n").encode(), addr)

  def emit(self, event: str):
    for addr in list(self._attached):
      self._sock.sendto(f"<3>{event}".encode(), addr)

  def close(self):
    self._exit = True
    self._thread.join()
    self._sock.close()


def wait_for(cond: Callable[[], bool], timeout: float = 5.0):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if cond():
      return
    time.sleep(0.02)
  raise AssertionError("condition not met in time")


class TestParsers(OpenpilotTestCase):
  def test_decode_ssid(self):
    self.assertEqual(decode_ssid("plain"), "plain")
    self.assertEqual(decode_ssid("caf\\xc3\\xa9"), "café")  # codespell:ignore caf
    self.assertEqual(decode_ssid("a\\\\b\\\"c\\tq"), 'a\\b"c\tq')

  def test_parse_scan_results_keeps_dbm_and_flags(self):
    raw = SCAN_HEADER + "aa:bb\t2412\t-45\t[WPA2-PSK-CCMP][ESS]\tHome\n" + "cc:dd\t5180\t-80\t[ESS]\tCaf\\xc3\\xa9\n" + "ee:ff\t2437\t-60\t[ESS]\t\n"
    self.assertEqual(parse_scan_results(raw), [("Home", -45, "[WPA2-PSK-CCMP][ESS]"), ("Café", -80, "[ESS]"), ("", -60, "[ESS]")])

  def test_dbm_to_percent_matches_networkmanager_scale(self):
    self.assertEqual([dbm_to_percent(d) for d in (-30, -40, -70, -100, -110)], [100, 100, 50, 0, 0])

  def test_security_type_from_flags(self):
    cases = {"[ESS]": SecurityType.OPEN, "[WPS][ESS]": SecurityType.OPEN, "[WPA2-PSK-CCMP][ESS]": SecurityType.WPA,
             "[WPA2-PSK+SAE-CCMP][ESS]": SecurityType.WPA, "[WPA-PSK-TKIP][WPA2-PSK-CCMP][ESS]": SecurityType.WPA,
             "[WPA2-SAE-CCMP][ESS]": SecurityType.UNSUPPORTED, "[WPA2-EAP-CCMP][ESS]": SecurityType.UNSUPPORTED, "[WEP][ESS]": SecurityType.UNSUPPORTED}
    for flags, expected in cases.items():
      with self.subTest(flags=flags):
        self.assertEqual(security_type_from_flags(flags), expected)

  def test_wpa_psk(self):
    # reference vector from IEEE 802.11-2020 Annex J
    self.assertEqual(wpa_psk("IEEE", "password"), "f42c6fc52df0ebef9ebb4b90b38a5f902e83fe1b135a70e23aed762e9710a12e")
    self.assertEqual(wpa_psk("x", "F" * 64), "f" * 64)


class TestWpaCtrl(OpenpilotTestCase):
  def test_request_and_attach(self):
    with tempfile.TemporaryDirectory() as d:
      fake = FakeSupplicant(os.path.join(d, "wlan0"))
      self.addCleanup(fake.close)
      ctrl = WpaCtrl(fake.path)
      self.addCleanup(ctrl.close)
      self.assertEqual(ctrl.request("PING"), "PONG")
      self.assertTrue(ctrl.ok("SCAN"))
      events = ctrl.attach()
      fake.emit("CTRL-EVENT-SCAN-RESULTS")
      self.assertEqual(events.recv(1024), b"<3>CTRL-EVENT-SCAN-RESULTS")
      self.assertEqual(fake.requests, ["PING", "SCAN"])


KEYFILE_A = """[connection]
id=openpilot connection Home
uuid=11111111-1111-1111-1111-111111111111
type=wifi

[wifi]
mode=infrastructure
ssid=Home

[wifi-security]
key-mgmt=wpa-psk
psk=password123

[ipv4]
method=auto
"""

NETPLAN_KEYFILE = """[connection]
id=openpilot connection Café
type=wifi
uuid=22222222-2222-2222-2222-222222222222
interface-name=wlan0
metered=1

[wifi]
ssid=67;97;102;195;169;
hidden=true

[wifi-security]
key-mgmt=wpa-psk
psk=cafepass1
"""

OPEN_KEYFILE = """[connection]
id=Open
uuid=33333333-3333-3333-3333-333333333333
type=wifi
metered=2

[wifi]
ssid=Open
"""

HOTSPOT_KEYFILE = """[connection]
id=Hotspot
uuid=44444444-4444-4444-4444-444444444444
type=wifi
interface-name=wlan0
autoconnect=false

[wifi]
band=bg
mode=ap
ssid=weedle

[wifi-security]
key-mgmt=wpa-psk
psk=swagswagcomma

[ipv4]
method=shared
address1=192.168.43.1/24,192.168.43.1
never-default=true
"""


def fake_sudo(record: list[list[str]]):
  # run install/rm locally without sudo, record everything else without running it
  def _sudo(*cmd: str, check: bool = True):
    record.append(list(cmd))
    if cmd[0] == "iptables-legacy" and "-C" in cmd:
      return subprocess.CompletedProcess(cmd, 1, "", "")
    if cmd[0] in ("install", "rm"):
      return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.CompletedProcess(cmd, 0, "", "")
  return _sudo


def profile_dirs(mocker: Mocker):
  d = tempfile.mkdtemp()
  persistent, runtime, netplan = (os.path.join(d, n) for n in ("persistent", "runtime", "netplan"))
  for p in (persistent, runtime, netplan):
    os.makedirs(p)
  mocker.patch.object(wifi_manager, "PROFILE_DIRS", (persistent, runtime))
  mocker.patch.object(wifi_manager, "NETPLAN_DIR", netplan)
  mocker.patch.object(wifi_manager, "sudo_read", lambda path: open(path).read())
  sudo_calls: list[list[str]] = []
  mocker.patch.object(wifi_manager, "_sudo", fake_sudo(sudo_calls))
  yield {"persistent": persistent, "runtime": runtime, "netplan": netplan, "sudo": sudo_calls}
  shutil.rmtree(d)


def write(path: str, content: str):
  with open(path, "w") as f:
    f.write(content)


class TestProfiles(OpenpilotTestCase):
  def test_read_profiles_from_both_dirs(self, profile_dirs):
    write(os.path.join(profile_dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    write(os.path.join(profile_dirs["persistent"], "Hotspot.nmconnection"), HOTSPOT_KEYFILE)
    write(os.path.join(profile_dirs["runtime"], "netplan-NM-22222222-2222-2222-2222-222222222222-Caf.nmconnection"), NETPLAN_KEYFILE)
    write(os.path.join(profile_dirs["runtime"], "lo.nmconnection"), "[connection]\nid=lo\nuuid=5\ntype=loopback\n")
    write(os.path.join(profile_dirs["runtime"], "broken.nmconnection"), "[connection\nid=x")
    profiles = {p.ssid: p for p in wifi_manager.read_profiles()}
    self.assertEqual(set(profiles), {"Home", "Café", "weedle"})
    self.assertEqual(profiles["Home"].psk, "password123")
    self.assertFalse(profiles["Home"].hidden)
    self.assertEqual(profiles["Home"].metered, wifi_manager.MeteredType.UNKNOWN)
    self.assertTrue(profiles["Café"].hidden)
    self.assertEqual(profiles["Café"].metered, wifi_manager.MeteredType.YES)
    self.assertTrue(profiles["weedle"].is_ap)
    self.assertEqual(profiles["weedle"].psk, "swagswagcomma")

  def test_write_profile_creates_networkmanager_keyfile(self, profile_dirs):
    profile = wifi_manager.Profile(path="", uuid="55555555-5555-5555-5555-555555555555", ssid="My Café/2", psk="secret99", hidden=True,
                                   metered=wifi_manager.MeteredType.NO, is_ap=False)
    saved = wifi_manager.write_profile(profile)
    self.assertEqual(saved.path, os.path.join(profile_dirs["persistent"], "My%20Caf%C3%A9%2F2.nmconnection"))
    self.assertEqual(profile_dirs["sudo"][-1][:3], ["install", "-m", "600"])
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(saved.path)
    self.assertEqual(cp["connection"]["type"], "wifi")
    self.assertEqual(cp["connection"]["uuid"], profile.uuid)
    self.assertEqual(cp["connection"]["metered"], "2")
    self.assertEqual(cp["wifi"]["ssid"], "77;121;32;67;97;102;195;169;47;50;")
    self.assertEqual(cp["wifi"]["hidden"], "true")
    self.assertEqual(cp["wifi-security"]["psk"], "secret99")
    self.assertEqual(cp["ipv4"]["method"], "auto")
    self.assertEqual([p.ssid for p in wifi_manager.read_profiles()], ["My Café/2"])

  def test_write_hotspot_profile_matches_upstream_shape(self, profile_dirs):
    saved = wifi_manager.write_profile(wifi_manager.Profile("", "6666", "weedle-abcd", "swagswagcomma", False, wifi_manager.MeteredType.UNKNOWN, True))
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(saved.path)
    self.assertEqual(cp["wifi"]["mode"], "ap")
    self.assertEqual(cp["wifi"]["ssid"], "weedle-abcd")
    self.assertEqual(cp["ipv4"]["method"], "shared")
    self.assertEqual(cp["ipv4"]["address1"], "192.168.43.1/24,192.168.43.1")
    self.assertEqual(cp["connection"]["autoconnect"], "false")
    self.assertNotIn("metered", cp["connection"])

  def test_write_profile_replaces_netplan_sources(self, profile_dirs):
    runtime = os.path.join(profile_dirs["runtime"], "netplan-NM-22222222-2222-2222-2222-222222222222-Caf.nmconnection")
    write(runtime, NETPLAN_KEYFILE)
    yaml = os.path.join(profile_dirs["netplan"], "90-NM-22222222-2222-2222-2222-222222222222.yaml")
    write(yaml, "network: {}\n")
    (profile,) = wifi_manager.read_profiles()
    saved = wifi_manager.write_profile(replace(profile, metered=wifi_manager.MeteredType.NO))
    self.assertFalse(os.path.exists(runtime))
    self.assertFalse(os.path.exists(yaml))
    self.assertEqual([(p.path, p.metered) for p in wifi_manager.read_profiles()], [(saved.path, wifi_manager.MeteredType.NO)])

  def test_remove_profile_removes_netplan_yaml(self, profile_dirs):
    runtime = os.path.join(profile_dirs["runtime"], "netplan-NM-22222222-2222-2222-2222-222222222222-Caf.nmconnection")
    write(runtime, NETPLAN_KEYFILE)
    yaml = os.path.join(profile_dirs["netplan"], "90-NM-22222222-2222-2222-2222-222222222222.yaml")
    write(yaml, "network: {}\n")
    wifi_manager.remove_profile(wifi_manager.read_profiles()[0])
    self.assertEqual(os.listdir(profile_dirs["runtime"]) + os.listdir(profile_dirs["netplan"]), [])

  def test_pid_alive(self):
    with tempfile.NamedTemporaryFile("w", suffix=".pid") as f:
      f.write(f"{os.getpid()}\n")
      f.flush()
      self.assertTrue(wifi_manager._pid_alive(f.name))
      f.seek(0)
      f.write("999999\n")
      f.flush()
      self.assertFalse(wifi_manager._pid_alive(f.name))
    self.assertFalse(wifi_manager._pid_alive("/nonexistent.pid"))


def manager_env(mocker: Mocker, profile_dirs):
  d = tempfile.mkdtemp()
  env = types.SimpleNamespace(dirs=profile_dirs, sudo=profile_dirs["sudo"], fake=None, popen=[], ctrl_path=os.path.join(d, "wlan0"),
                              wpa_pid=os.path.join(d, "wlan0.pid"), udhcpc_pid=os.path.join(d, "udhcpc.pid"), dnsmasq_pid=os.path.join(d, "dnsmasq.pid"),
                              spawn_status={"wpa_state": "DISCONNECTED"}, managers=[], supplicant=None, supplicants=[], dnsmasq=None, keep_pids=set(),
                              popen_kwargs=[],
                              operations=[])
  mocker.patch.object(wifi_manager, "WPA_CTRL_PATH", env.ctrl_path)
  mocker.patch.object(wifi_manager, "WPA_PID_PATH", env.wpa_pid)
  mocker.patch.object(wifi_manager, "UDHCPC_PID_PATH", env.udhcpc_pid)
  mocker.patch.object(wifi_manager, "DNSMASQ_PID_PATH", env.dnsmasq_pid)
  mocker.patch.object(wifi_manager, "Params", None)
  mocker.patch.object(wifi_manager, "SCAN_PERIOD_SECONDS", 0.1)
  mocker.patch.object(wifi_manager, "HANDOFF_TIMEOUT_SECONDS", 1)

  def alive(path):
    write(path, f"{os.getpid()}\n")

  def spawn_fake():
    env.fake = FakeSupplicant(env.ctrl_path)
    env.fake.on_request = env.operations.append
    env.fake.status = dict(env.spawn_status)
    alive(env.wpa_pid)

  env.alive = alive
  env.spawn_fake = spawn_fake

  inner = fake_sudo(env.sudo)

  def _sudo(*cmd, check=True):
    if cmd[0] == "wpa_supplicant":
      spawn_fake()
    env.operations.append(" ".join(cmd))
    result = inner(*cmd, check=check)
    if cmd[0] == "kill" and len(cmd) == 2:
      for path in (env.udhcpc_pid, env.dnsmasq_pid):
        if path not in env.keep_pids and os.path.exists(path):
          with open(path) as f:
            if f.read().strip() == cmd[1]:
              os.unlink(path)
    return result

  mocker.patch.object(wifi_manager, "_sudo", _sudo)

  real_run = subprocess.run
  real_popen = subprocess.Popen

  def run(cmd, **kwargs):
    if cmd[:3] == ["sudo", "wpa_supplicant", "-B"]:
      env.supplicant = (cmd, kwargs)
      env.supplicants.append(env.supplicant)
      spawn_fake()
      return subprocess.CompletedProcess(cmd, 0, "", "")
    if cmd[:2] == ["sudo", "dnsmasq"]:
      env.dnsmasq = (cmd, kwargs)
      return subprocess.CompletedProcess(cmd, 0, "", "")
    return real_run(cmd, **kwargs)

  mocker.patch.object(wifi_manager.subprocess, "run", run)

  def popen(cmd, **kwargs):
    if cmd[:2] == ["sudo", "udhcpc"]:
      env.popen.append(cmd)
      env.popen_kwargs.append(kwargs)
      return unittest.mock.MagicMock()
    return real_popen(cmd, **kwargs)

  mocker.patch.object(wifi_manager.subprocess, "Popen", popen)
  yield env
  for wm in env.managers:
    wm.stop()
  if env.fake is not None:
    env.fake.close()
  shutil.rmtree(d)


def start_manager(env):
  wm = wifi_manager.WifiManager()
  env.managers.append(wm)
  wait_for(lambda: wm._ready)
  return wm


def drain(wm, events: list):
  wm.process_callbacks()
  return events


class TestLifecycle(OpenpilotTestCase):
  def test_fresh_start_hands_off_and_loads_saved_networks(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    write(os.path.join(manager_env.dirs["persistent"], "Open.nmconnection"), OPEN_KEYFILE)
    write(os.path.join(manager_env.dirs["persistent"], "Hotspot.nmconnection"), HOTSPOT_KEYFILE)
    write(os.path.join(manager_env.dirs["runtime"], "netplan-NM-2222-Caf.nmconnection"), NETPLAN_KEYFILE)
    wm = start_manager(manager_env)
    self.assertEqual(manager_env.sudo, [["nmcli", "dev", "set", "wlan0", "managed", "no"]])
    self.assertEqual(manager_env.supplicant, (["sudo", "wpa_supplicant", "-B", "-i", "wlan0", "-D", "nl80211", "-c", wifi_manager.WPA_CONF_PATH,
                                               "-P", manager_env.wpa_pid],
                                              {"check": True, "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                                               "stderr": subprocess.DEVNULL, "start_new_session": True}))
    nets = {bytes.fromhex(n["ssid"]).decode(): n for n in manager_env.fake.networks.values()}
    self.assertEqual(set(nets), {"Home", "Open", "Café"})
    self.assertEqual(nets["Home"]["psk"], wpa_psk("Home", "password123"))
    self.assertEqual(nets["Open"]["key_mgmt"], "NONE")
    self.assertEqual(nets["Café"]["scan_ssid"], "1")
    self.assertIn("ENABLE_NETWORK all", manager_env.fake.requests)  # codespell:ignore assertin
    self.assertEqual(manager_env.popen, [["sudo", "udhcpc", "-i", "wlan0", "-f", "-R", "-s", wifi_manager.UDHCPC_SCRIPT_PATH, "-p", manager_env.udhcpc_pid]])
    self.assertEqual(manager_env.popen_kwargs, [{"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                                                  "start_new_session": True}])
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertTrue(wm.is_connection_saved("Home"))
    self.assertTrue(wm.is_connection_saved("weedle"))
    self.assertFalse(wm.is_connection_saved("Elsewhere"))
    self.assertFalse(wm.is_tethering_active())

  def test_adopts_connected_station_without_respawning(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A.replace("type=wifi\n", "type=wifi\nmetered=2\n"))
    manager_env.spawn_status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "ip_address": "10.0.0.5", "id": "0"}
    manager_env.spawn_fake()
    manager_env.fake.networks = {0: {"ssid": b"Home".hex()}}
    manager_env.alive(manager_env.udhcpc_pid)
    wm = start_manager(manager_env)
    self.assertEqual([c for c in manager_env.sudo if c[0] != "install"], [["kill", "-USR1", str(os.getpid())]])
    self.assertEqual(manager_env.popen, [])
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState("Home", wifi_manager.ConnectStatus.CONNECTED))
    self.assertEqual(wm.connected_ssid, "Home")
    self.assertEqual(wm.ipv4_address, "10.0.0.5")
    self.assertEqual(wm.current_network_metered, wifi_manager.MeteredType.NO)
    self.assertEqual(wm._network_ids, {0: "Home"})

  def test_adopts_associating_station_as_connecting(self, manager_env):
    manager_env.spawn_status = {"wpa_state": "4WAY_HANDSHAKE", "ssid": "Home", "mode": "station"}
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    self.assertEqual(wm.connecting_to_ssid, "Home")
    self.assertEqual(wm.ipv4_address, "")

  def test_scan_results_update_sorted_networks(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    write(os.path.join(manager_env.dirs["persistent"], "Hotspot.nmconnection"), HOTSPOT_KEYFILE)
    wm = start_manager(manager_env)
    updates = []
    wm.add_callbacks(networks_updated=updates.append)
    manager_env.fake.replies["SCAN_RESULTS"] = (SCAN_HEADER + "aa\t2412\t-75\t[WPA2-PSK-CCMP][ESS]\tHome\n" + "ab\t5180\t-50\t[WPA2-PSK-CCMP][ESS]\tHome\n"
                                                + "cc\t2437\t-40\t[ESS]\tCoffee\n" + "dd\t2437\t-90\t[WPA2-PSK-CCMP][ESS]\tweedle\n"
                                                + "ee\t2437\t-60\t[WPA2-EAP-CCMP][ESS]\tOffice\n" + "ff\t2437\t-30\t[ESS]\t\n")
    wait_for(lambda: "SCAN" in manager_env.fake.requests)
    manager_env.fake.emit("CTRL-EVENT-SCAN-RESULTS")
    wait_for(lambda: drain(wm, updates))
    self.assertEqual(updates[-1], [wifi_manager.Network("weedle", 100, SecurityType.WPA, True), wifi_manager.Network("Home", 84, SecurityType.WPA, False),
                                   wifi_manager.Network("Coffee", 100, SecurityType.OPEN, False),
                                   wifi_manager.Network("Office", 67, SecurityType.UNSUPPORTED, False)])
    self.assertEqual(updates[-1], wm.networks)

  def test_stop_leaves_daemons_running(self, manager_env):
    manager_env.spawn_status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "ip_address": "10.0.0.5", "id": "0"}
    manager_env.spawn_fake()
    manager_env.alive(manager_env.udhcpc_pid)
    wm = start_manager(manager_env)
    wm.stop()
    self.assertNotIn(["kill", str(os.getpid())], manager_env.sudo)
    self.assertFalse(wm._scan_thread.is_alive())
    self.assertFalse(wm._monitor_thread.is_alive())
    self.assertEqual(WpaCtrl(manager_env.ctrl_path).request("PING"), "PONG")

  def test_stop_waits_for_initialization(self, manager_env):
    manager_env.spawn_fake()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

    def ping(cmd):
      entered.set()
      release.wait()
      return "PONG"

    manager_env.fake.replies["PING"] = ping
    wm = wifi_manager.WifiManager()
    manager_env.managers.append(wm)
    stopper = None
    try:
      self.assertTrue(entered.wait(2))
      stopper = threading.Thread(target=lambda: (wm.stop(), stopped.set()))
      stopper.start()
      self.assertFalse(stopped.wait(0.2))
    finally:
      release.set()
      if stopper is not None:
        stopper.join(2)
    self.assertTrue(stopped.is_set())
    self.assertFalse(wm._init_thread.is_alive())
    self.assertFalse(wm._scan_thread.is_alive())
    self.assertFalse(wm._monitor_thread.is_alive())
    self.assertEqual(wm._ctrl._sock.fileno(), -1)
    self.assertEqual(wm._events.fileno(), -1)
    self.assertEqual(manager_env.popen, [["sudo", "udhcpc", "-i", "wlan0", "-f", "-R", "-s", wifi_manager.UDHCPC_SCRIPT_PATH, "-p", manager_env.udhcpc_pid]])

  def test_stop_waits_for_inflight_refresh(self, manager_env):
    wm = start_manager(manager_env)
    wm.set_active(False)
    handled = threading.Event()
    handle_event = wm._handle_event

    def handle(event):
      handled.set()
      handle_event(event)

    wm._handle_event = handle
    wm._events.settimeout(0.01)
    manager_env.fake.emit("CTRL-EVENT-IGNORE")
    self.assertTrue(handled.wait(2))
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

    def scan_results(cmd):
      entered.set()
      release.wait()
      return SCAN_HEADER + "aa\t2412\t-40\t[ESS]\tAfterStop\n"

    manager_env.fake.replies["SCAN_RESULTS"] = scan_results
    wm.set_active(True)
    stopper = None
    try:
      self.assertTrue(entered.wait(2))
      stopper = threading.Thread(target=lambda: (wm.stop(), stopped.set()))
      stopper.start()
      self.assertFalse(stopped.wait(0.2))
    finally:
      release.set()
      if stopper is not None:
        stopper.join(2)
    self.assertTrue(stopped.is_set())
    self.assertFalse(wm._scan_thread.is_alive())
    self.assertFalse(wm._monitor_thread.is_alive())

  def test_process_callbacks_keeps_concurrent_enqueue(self, manager_env):
    wm = start_manager(manager_env)
    entered, release, delivered = threading.Event(), threading.Event(), []

    class GateList(list):
      def append(self, value):
        entered.set()
        release.wait()
        super().append(value)

    wm._callback_queue = GateList()
    producer = threading.Thread(target=lambda: wm._enqueue_callbacks([lambda: delivered.append(True)]))
    drainer = None
    producer.start()
    completed = threading.Event()

    def drain_callbacks():
      wm.process_callbacks()
      completed.set()

    try:
      self.assertTrue(entered.wait(2))
      drainer = threading.Thread(target=drain_callbacks)
      drainer.start()
      self.assertFalse(completed.wait(0.2))
    finally:
      release.set()
      producer.join(2)
      if drainer is not None:
        drainer.join(2)
    self.assertFalse(producer.is_alive())
    self.assertFalse(drainer.is_alive())
    self.assertEqual(delivered, [True])

  def test_handoff_waits_for_control_path_removal(self, mocker):
    finished = threading.Event()
    mocker.patch.object(wifi_manager, "HANDOFF_TIMEOUT_SECONDS", 1)
    mocker.patch.object(wifi_manager, "CTRL_TIMEOUT_SECONDS", 0.1)
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "wlan0")
      stale = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
      stale.bind(path)
      stale.close()
      mocker.patch.object(wifi_manager, "WPA_CTRL_PATH", path)
      waiter = threading.Thread(target=lambda: (wifi_manager.WifiManager._wait_for_handoff(cast(wifi_manager.WifiManager, types.SimpleNamespace())),
                                                finished.set()))
      waiter.start()
      try:
        self.assertFalse(finished.wait(0.2))
      finally:
        if os.path.exists(path):
          os.unlink(path)
        waiter.join(2)
      self.assertTrue(finished.is_set())


def profile_files(env):
  # Exclude the hotspot profile from station-profile assertions.
  return sorted(f for f in os.listdir(env.dirs["persistent"]) if f != "weedle.nmconnection")


class TestStation(OpenpilotTestCase):
  def test_connect_persists_after_first_address(self, manager_env):
    wm = start_manager(manager_env)
    activated, forgotten = [], []
    wm.add_callbacks(activated=lambda: activated.append(True), forgotten=forgotten.append)
    manager_env.alive(manager_env.udhcpc_pid)
    wm.connect_to_network("Home", "password123")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    (nid,) = manager_env.fake.networks
    self.assertEqual(manager_env.fake.networks[nid], {"ssid": b"Home".hex(), "psk": wpa_psk("Home", "password123")})
    self.assertEqual(wm.connecting_to_ssid, "Home")
    self.assertEqual(profile_files(manager_env), [])

    manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "id": str(nid)}
    manager_env.fake.emit(f"CTRL-EVENT-CONNECTED - Connection to aa:bb completed [id={nid} id_str=]")
    wait_for(lambda: ["kill", "-USR1", str(os.getpid())] in manager_env.sudo)
    self.assertEqual(wm.connecting_to_ssid, "Home")
    self.assertEqual(profile_files(manager_env), [])
    wait_for(lambda: manager_env.fake.requests.count("ENABLE_NETWORK all") >= 2)

    manager_env.fake.status["ip_address"] = "10.0.0.9"
    wait_for(lambda: wm.connected_ssid == "Home")
    wait_for(lambda: drain(wm, activated))
    self.assertEqual(wm.ipv4_address, "10.0.0.9")
    self.assertEqual(profile_files(manager_env), ["Home.nmconnection"])
    self.assertTrue(wm.is_connection_saved("Home"))
    self.assertEqual(next(p.psk for p in wifi_manager.read_profiles() if p.ssid == "Home"), "password123")
    self.assertEqual(forgotten, ["Home"])

  def test_wrong_password_asks_once_and_saves_nothing(self, manager_env):
    wm = start_manager(manager_env)
    need_auth = []
    wm.add_callbacks(need_auth=need_auth.append)
    wm.connect_to_network("Home", "wrongpass")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    (nid,) = manager_env.fake.networks
    n = len(manager_env.fake.requests)
    manager_env.fake.emit(f'CTRL-EVENT-SSID-TEMP-DISABLED id={nid} ssid="Home" auth_failures=1 duration=10 reason=WRONG_KEY')
    wait_for(lambda: drain(wm, need_auth) == ["Home"])
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertEqual(manager_env.fake.networks, {})
    self.assertEqual(profile_files(manager_env), [])
    self.assertEqual([r for r in manager_env.fake.requests[n:] if not r.startswith(("SCAN", "STATUS"))], [f"REMOVE_NETWORK {nid}", "ENABLE_NETWORK all"])

  def test_activate_saved_network_selects_it(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    wm = start_manager(manager_env)
    wm.activate_connection("Home")
    wait_for(lambda: "SELECT_NETWORK 0" in manager_env.fake.requests)
    self.assertEqual(wm.connecting_to_ssid, "Home")

    wm.activate_connection("Unknown", block=True)
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertEqual([r for r in manager_env.fake.requests if r.startswith("SELECT")], ["SELECT_NETWORK 0"])

  def test_forget_removes_every_source(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    write(os.path.join(manager_env.dirs["runtime"], "netplan-NM-22222222-2222-2222-2222-222222222222-Caf.nmconnection"), NETPLAN_KEYFILE)
    yaml = os.path.join(manager_env.dirs["netplan"], "90-NM-22222222-2222-2222-2222-222222222222.yaml")
    write(yaml, "network: {}\n")
    wm = start_manager(manager_env)
    forgotten = []
    wm.add_callbacks(forgotten=forgotten.append)
    for ssid in ("Café", "Home"):
      with self.subTest(ssid=ssid):
        nid = next(i for i, s in wm._network_ids.items() if s == ssid)
        wm.forget_connection(ssid, block=True)
        self.assertIn(f"REMOVE_NETWORK {nid}", manager_env.fake.requests)  # codespell:ignore assertin
        self.assertFalse(wm.is_connection_saved(ssid))
        self.assertEqual(drain(wm, forgotten)[-1], ssid)
    self.assertFalse(os.path.exists(yaml))
    self.assertEqual(os.listdir(manager_env.dirs["runtime"]) + profile_files(manager_env), [])

  def test_forget_connected_network_disconnects(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    manager_env.spawn_status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "ip_address": "10.0.0.5", "id": "0"}
    manager_env.spawn_fake()
    manager_env.fake.networks = {0: {"ssid": b"Home".hex()}}
    wm = start_manager(manager_env)
    disconnected, forgotten = [], []
    wm.add_callbacks(disconnected=lambda: disconnected.append(True),
                     forgotten=lambda ssid: forgotten.append((ssid, wm.wifi_state, wm.ipv4_address)))
    wm.forget_connection("Home", block=True)
    drain(wm, forgotten)
    self.assertEqual(forgotten, [("Home", wifi_manager.WifiState(), "")])
    self.assertEqual(disconnected, [True])
    manager_env.fake.status = {"wpa_state": "DISCONNECTED"}
    handled = threading.Event()
    handle_event = wm._handle_event

    def handle(event):
      handle_event(event)
      handled.set()

    wm._handle_event = handle
    manager_env.fake.emit("CTRL-EVENT-DISCONNECTED bssid=aa:bb reason=3 locally_generated=1")
    self.assertTrue(handled.wait(2))
    drain(wm, disconnected)
    self.assertEqual(disconnected, [True])
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertEqual(wm.ipv4_address, "")
    self.assertEqual(profile_files(manager_env), [])

  def test_dhcp_timeout_reports_disconnected(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "DHCP_TIMEOUT_SECONDS", 0.5)
    wm = start_manager(manager_env)
    disconnected = []
    wm.add_callbacks(disconnected=lambda: disconnected.append(True))
    wm.connect_to_network("Home", "password123")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    (nid,) = manager_env.fake.networks
    manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "id": str(nid)}
    manager_env.fake.emit(f"CTRL-EVENT-CONNECTED - Connection to aa:bb completed [id={nid} id_str=]")
    wait_for(lambda: drain(wm, disconnected))
    self.assertIn(f"DISABLE_NETWORK {nid}", manager_env.fake.requests)  # codespell:ignore assertin
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertEqual(profile_files(manager_env), [])

  def test_link_loss_reports_disconnected(self, manager_env):
    manager_env.spawn_status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "ip_address": "10.0.0.5", "id": "0"}
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    disconnected = []
    wm.add_callbacks(disconnected=lambda: disconnected.append(True))
    manager_env.fake.status = {"wpa_state": "SCANNING"}
    manager_env.fake.emit("CTRL-EVENT-DISCONNECTED bssid=aa:bb reason=4")
    wait_for(lambda: drain(wm, disconnected))
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertEqual(wm.ipv4_address, "")

  def test_stop_during_dhcp_wait_does_not_report_disconnect(self, manager_env):
    wm = start_manager(manager_env)
    disconnected = []
    wm.add_callbacks(disconnected=lambda: disconnected.append(True))
    wm.connect_to_network("Home", "password123")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    (nid,) = manager_env.fake.networks
    manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "id": str(nid)}
    manager_env.fake.emit(f"CTRL-EVENT-CONNECTED - Connection to aa:bb completed [id={nid} id_str=]")
    wait_for(lambda: manager_env.fake.requests.count("ENABLE_NETWORK all") >= 2)
    wm.stop()
    self.assertNotIn(f"DISABLE_NETWORK {nid}", manager_env.fake.requests)
    self.assertEqual(drain(wm, disconnected), [])

  def test_stale_connected_event_keeps_new_selection(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    manager_env.spawn_status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "ip_address": "10.0.0.5", "id": "0"}
    manager_env.spawn_fake()
    manager_env.fake.networks = {0: {"ssid": b"Home".hex()}}
    wm = start_manager(manager_env)
    activated = []
    wm.add_callbacks(activated=lambda: activated.append(True))
    wm.connect_to_network("New", "password123")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    wm._refresh_status()
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState("New", wifi_manager.ConnectStatus.CONNECTING))

    wm._handle_event("CTRL-EVENT-CONNECTED - Connection to aa:bb completed [id=0 id_str=]")
    self.assertEqual(wm._pending.ssid, "New")
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState("New", wifi_manager.ConnectStatus.CONNECTING))
    self.assertEqual(drain(wm, activated), [])

  def test_stale_dhcp_timeout_keeps_later_selection(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "DHCP_TIMEOUT_SECONDS", 0.1)
    wm = start_manager(manager_env)
    wm.set_active(False)
    disconnected = []
    wm.add_callbacks(disconnected=lambda: disconnected.append(True))
    wm.connect_to_network("A", "password123")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    (a_id,) = manager_env.fake.networks
    manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "A", "mode": "station", "id": str(a_id)}

    original_status = wm._status
    entered, release = threading.Event(), threading.Event()
    calls = 0

    def gated_status():
      nonlocal calls
      calls += 1
      status = original_status()
      if calls == 2:
        entered.set()
        release.wait()
      return status

    wm._status = gated_status
    handler = threading.Thread(target=lambda: wm._handle_event(f"CTRL-EVENT-CONNECTED - Connection to aa:bb completed [id={a_id} id_str=]"))
    handler.start()
    try:
      self.assertTrue(entered.wait(2))
      wm.connect_to_network("B", "password123")
      wait_for(lambda: "SELECT_NETWORK 1" in manager_env.fake.requests)
      release.set()
      handler.join(2)
    finally:
      release.set()
      handler.join(2)
    self.assertFalse(handler.is_alive())
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState("B", wifi_manager.ConnectStatus.CONNECTING))
    self.assertEqual(wm._pending.ssid, "B")
    self.assertNotIn(f"DISABLE_NETWORK {a_id}", manager_env.fake.requests)
    self.assertEqual(drain(wm, disconnected), [])

  def test_dhcp_timeout_does_not_clear_late_selection(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "DHCP_TIMEOUT_SECONDS", 0.1)
    wm = start_manager(manager_env)
    wm.set_active(False)
    wm.connect_to_network("A", "password123")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    (a_id,) = manager_env.fake.networks
    manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "A", "mode": "station", "id": str(a_id)}

    entered, release, selected = threading.Event(), threading.Event(), threading.Event()
    warning = wifi_manager.cloudlog.warning

    def gate(message):
      entered.set()
      release.wait()
      warning(message)

    mocker.patch.object(wifi_manager.cloudlog, "warning", gate)
    handler = threading.Thread(target=lambda: wm._handle_event(f"CTRL-EVENT-CONNECTED - Connection to aa:bb completed [id={a_id} id_str=]"))
    selector = threading.Thread(target=lambda: (wm.connect_to_network("B", "password123"), selected.set()))
    handler.start()
    try:
      self.assertTrue(entered.wait(2))
      selector.start()
      selected.wait(0.2)
      release.set()
      handler.join(2)
      selector.join(2)
    finally:
      release.set()
      handler.join(2)
      selector.join(2)
    self.assertFalse(handler.is_alive())
    self.assertFalse(selector.is_alive())
    wait_for(lambda: "SELECT_NETWORK 1" in manager_env.fake.requests)
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState("B", wifi_manager.ConnectStatus.CONNECTING))
    self.assertEqual(wm._pending.ssid, "B")

  def test_stale_dhcp_success_keeps_new_pending_connection(self, manager_env):
    wm = start_manager(manager_env)
    wm.set_active(False)
    activated = []
    wm.add_callbacks(activated=lambda: activated.append(True))
    wm.connect_to_network("A", "password123")
    wait_for(lambda: "SELECT_NETWORK 0" in manager_env.fake.requests)
    manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "A", "mode": "station", "id": "0", "ip_address": "10.0.0.9"}

    original_status = wm._status
    entered, release = threading.Event(), threading.Event()
    calls = 0

    def gated_status():
      nonlocal calls
      calls += 1
      status = original_status()
      if calls == 2:
        entered.set()
        release.wait()
      return status

    wm._status = gated_status
    handler = threading.Thread(target=wm._on_associated)
    handler.start()
    try:
      self.assertTrue(entered.wait(2))
      wm.connect_to_network("B", "password123")
      wait_for(lambda: "SELECT_NETWORK 1" in manager_env.fake.requests)
      manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "B", "mode": "station", "id": "1", "ip_address": "10.0.0.10"}
      wm._refresh_status()
      self.assertIsNone(wm._selected)
      self.assertEqual(wm._pending.ssid, "B")
      release.set()
      handler.join(2)
    finally:
      release.set()
      handler.join(2)
    self.assertFalse(handler.is_alive())
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState("B", wifi_manager.ConnectStatus.CONNECTED))
    self.assertEqual(wm.ipv4_address, "10.0.0.10")
    self.assertEqual(wm._pending.ssid, "B")
    self.assertEqual(profile_files(manager_env), [])
    self.assertEqual(drain(wm, activated), [])

  def test_stale_dhcp_timeout_keeps_new_pending_connection(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "DHCP_TIMEOUT_SECONDS", 0.1)
    for name, saved in (("pending", False), ("saved", True)):
      with self.subTest(name=name):
        if manager_env.fake is not None:
          manager_env.fake.networks = {}
          manager_env.fake.requests.clear()
        if saved:
          write(os.path.join(manager_env.dirs["persistent"], "B.nmconnection"), KEYFILE_A.replace("Home", "B"))
        wm = start_manager(manager_env)
        wm.set_active(False)
        activated, disconnected = [], []
        wm.add_callbacks(activated=lambda activated=activated: activated.append(True),
                         disconnected=lambda disconnected=disconnected: disconnected.append(True))
        wm.connect_to_network("A", "password123")
        wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
        a_id = next(nid for nid, ssid in wm._network_ids.items() if ssid == "A")
        manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "A", "mode": "station", "id": str(a_id)}

        original_status = wm._status
        entered, release = threading.Event(), threading.Event()
        calls = 0

        def gated_status(original_status=original_status, entered=entered, release=release):
          nonlocal calls
          calls += 1
          status = original_status()
          if calls == 2:
            entered.set()
            release.wait()
          return status

        wm._status = gated_status
        handler = threading.Thread(target=wm._on_associated)
        handler.start()
        try:
          self.assertTrue(entered.wait(2))
          if saved:
            wm.activate_connection("B", block=True)
          else:
            wm.connect_to_network("B", "password123")
          wait_for(lambda wm=wm: any(ssid == "B" for ssid in wm._network_ids.values()))
          b_id = next(nid for nid, ssid in wm._network_ids.items() if ssid == "B")
          wait_for(lambda b_id=b_id: f"SELECT_NETWORK {b_id}" in manager_env.fake.requests)
          manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "B", "mode": "station", "id": str(b_id), "ip_address": "10.0.0.10"}
          wm._refresh_status()
          self.assertIsNone(wm._selected)
          if saved:
            self.assertIsNone(wm._pending)
          else:
            self.assertEqual(wm._pending.ssid, "B")
          release.set()
          handler.join(2)
        finally:
          release.set()
          handler.join(2)
        self.assertFalse(handler.is_alive())
        self.assertNotIn(f"DISABLE_NETWORK {a_id}", manager_env.fake.requests)
        self.assertEqual(wm.wifi_state, wifi_manager.WifiState("B", wifi_manager.ConnectStatus.CONNECTED))
        self.assertEqual(wm.ipv4_address, "10.0.0.10")
        if saved:
          self.assertIsNone(wm._pending)
        else:
          self.assertEqual(wm._pending.ssid, "B")
        self.assertEqual(profile_files(manager_env), [] if not saved else ["B.nmconnection"])
        self.assertEqual((drain(wm, activated), drain(wm, disconnected)), ([], []))
        wm.stop()

  def test_select_failure_resets_state(self, manager_env, mocker):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    wm = start_manager(manager_env)
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    manager_env.fake.replies["SELECT_NETWORK"] = "FAIL"
    for name, select in (("activate", lambda: wm.activate_connection("Home", block=True)),
                         ("connect", lambda: wm.connect_to_network("New", "password123"))):
      with self.subTest(name=name):
        failure.reset_mock()
        select()
        wait_for(lambda: failure.called)
        wait_for(lambda: wm.wifi_state == wifi_manager.WifiState())
        self.assertIn(name, failure.call_args.args[0])  # codespell:ignore assertin

  def test_forget_remove_failure_preserves_network(self, manager_env, mocker):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    manager_env.spawn_status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "ip_address": "10.0.0.5", "id": "0"}
    manager_env.spawn_fake()
    manager_env.fake.networks = {0: {"ssid": b"Home".hex()}}
    wm = start_manager(manager_env)
    forgotten = []
    wm.add_callbacks(forgotten=forgotten.append)
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    manager_env.fake.status = {"wpa_state": "DISCONNECTED"}
    manager_env.fake.replies["REMOVE_NETWORK 0"] = "FAIL"
    n = len(manager_env.fake.requests)
    wm.forget_connection("Home", block=True)
    self.assertTrue(failure.called)
    self.assertEqual(wm._network_ids, {0: "Home"})
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertTrue(wm.is_connection_saved("Home"))
    self.assertIn("STATUS", manager_env.fake.requests[n:])  # codespell:ignore assertin
    self.assertEqual(drain(wm, forgotten), ["Home"])

  def test_wrong_key_remove_failure_keeps_network(self, manager_env, mocker):
    wm = start_manager(manager_env)
    need_auth = []
    wm.add_callbacks(need_auth=need_auth.append)
    wm.connect_to_network("Home", "wrongpass")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    (nid,) = manager_env.fake.networks
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    manager_env.fake.status = {"wpa_state": "DISCONNECTED"}
    manager_env.fake.replies[f"REMOVE_NETWORK {nid}"] = "FAIL"
    n = len(manager_env.fake.requests)
    manager_env.fake.emit(f'CTRL-EVENT-SSID-TEMP-DISABLED id={nid} ssid="Home" auth_failures=1 duration=10 reason=WRONG_KEY')
    wait_for(lambda: failure.called)
    wait_for(lambda: wm.wifi_state == wifi_manager.WifiState())
    self.assertEqual(wm._network_ids, {nid: "Home"})
    self.assertIn("STATUS", manager_env.fake.requests[n:])  # codespell:ignore assertin
    self.assertEqual(drain(wm, need_auth), [])

  def test_terminal_command_failures_reset_state(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "DHCP_TIMEOUT_SECONDS", 0.1)
    wm = start_manager(manager_env)
    activated, need_auth, disconnected = [], [], []
    wm.add_callbacks(activated=lambda: activated.append(True), need_auth=need_auth.append, disconnected=lambda: disconnected.append(True))
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")

    def fail(cmd):
      manager_env.fake.status = {"wpa_state": "DISCONNECTED"}
      return "FAIL"

    def reset():
      failure.reset_mock()
      manager_env.fake.replies.clear()
      manager_env.fake.networks = {0: {"ssid": b"Home".hex()}}
      wm._network_ids = {0: "Home"}
      wm._selected = "Home"
      wm._pending = None
      wm._wifi_state = wifi_manager.WifiState("Home", wifi_manager.ConnectStatus.CONNECTING)
      wm._ipv4_address = ""
      del activated[:]
      del need_auth[:]
      del disconnected[:]

    cases = (
      ("associated-enable", lambda: manager_env.fake.replies.update({"ENABLE_NETWORK all": fail}),
       lambda: setattr(manager_env.fake, "status", {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "ip_address": "10.0.0.5", "id": "0"}),
       lambda: wm._on_associated()),
      ("wrong-key-enable", lambda: manager_env.fake.replies.update({"ENABLE_NETWORK all": fail}),
       lambda: setattr(manager_env.fake, "status", {"wpa_state": "DISCONNECTED"}), lambda: wm._on_wrong_key(0)),
      ("timeout-disable", lambda: manager_env.fake.replies.update({"DISABLE_NETWORK 0": fail}),
       lambda: setattr(manager_env.fake, "status", {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "id": "0"}), lambda: wm._on_associated()),
    )
    for name, configure, status, action in cases:
      with self.subTest(name=name):
        reset()
        configure()
        status()
        n = len(manager_env.fake.requests)
        action()
        self.assertTrue(failure.called)
        self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
        self.assertIn("STATUS", manager_env.fake.requests[n:])  # codespell:ignore assertin
        wm.process_callbacks()
        self.assertEqual((activated, need_auth, disconnected), ([], [], []))

  def test_terminal_callbacks_enqueue_before_later_selection(self, manager_env):
    def wrong_key(wm):
      wm.connect_to_network("A", "wrongpass")
      wait_for(lambda: "SELECT_NETWORK 0" in manager_env.fake.requests)
      return wm._need_auth, lambda: wm._on_wrong_key(0), ["need_auth"]

    def forget(wm):
      manager_env.fake.networks = {0: {"ssid": b"A".hex()}}
      with wm._lock:
        wm._network_ids = {0: "A"}
        wm._wifi_state = wifi_manager.WifiState("A", wifi_manager.ConnectStatus.CONNECTED)
      return wm._disconnected, lambda: wm.forget_connection("A", block=True), ["disconnected", "forgotten"]

    for name, setup in (("wrong-key", wrong_key), ("forget", forget)):
      with self.subTest(name=name):
        wm = start_manager(manager_env)
        wm.set_active(False)
        need_auth, disconnected, forgotten = [], [], []
        wm.add_callbacks(need_auth=need_auth.append, disconnected=lambda disconnected=disconnected: disconnected.append(True), forgotten=forgotten.append)
        gate, release, selected = threading.Event(), threading.Event(), threading.Event()
        queued = []
        original_enqueue = wm._enqueue_callbacks
        gated_callbacks, action, expected = setup(wm)
        wm.process_callbacks()
        need_auth.clear()
        disconnected.clear()
        forgotten.clear()
        select_requests = manager_env.fake.requests.count("SELECT_NETWORK 0")
        names = {id(wm._need_auth): "need_auth", id(wm._disconnected): "disconnected", id(wm._forgotten): "forgotten"}

        def enqueue(cbs, *args, gated_callbacks=gated_callbacks, gate=gate, release=release, original_enqueue=original_enqueue, names=names, queued=queued):
          if cbs is gated_callbacks:
            gate.set()
            release.wait()
          original_enqueue(cbs, *args)
          if (callback := names.get(id(cbs))) is not None:
            queued.append(callback)

        def select(wm=wm, selected=selected):
          wm.connect_to_network("B", "password123")
          selected.set()

        wm._enqueue_callbacks = enqueue
        handler = threading.Thread(target=action)
        selector = threading.Thread(target=select)
        handler.start()
        try:
          self.assertTrue(gate.wait(2))
          selector.start()
          self.assertFalse(selected.wait(0.2))
          release.set()
          handler.join(2)
          selector.join(2)
        finally:
          release.set()
          handler.join(2)
          selector.join(2)
        self.assertFalse(handler.is_alive())
        self.assertFalse(selector.is_alive())
        self.assertEqual(wm.wifi_state, wifi_manager.WifiState("B", wifi_manager.ConnectStatus.CONNECTING))
        self.assertEqual(wm._pending.ssid, "B")
        wait_for(lambda select_requests=select_requests: manager_env.fake.requests.count("SELECT_NETWORK 0") > select_requests)
        self.assertEqual(queued, expected + ["forgotten"])
        wm.process_callbacks()
        self.assertEqual((need_auth, disconnected, forgotten), (["A"], [], ["B"]) if name == "wrong-key" else ([], [True], ["A", "B"]))
        wm.stop()

  def test_delayed_workers_do_not_select_stale_network(self, manager_env):
    for name, profiles in (("connect", ("C",)), ("activate", ("B", "C"))):
      with self.subTest(name=name):
        for filename in os.listdir(manager_env.dirs["persistent"]):
          os.unlink(os.path.join(manager_env.dirs["persistent"], filename))
        for ssid in profiles:
          write(os.path.join(manager_env.dirs["persistent"], f"{ssid}.nmconnection"), KEYFILE_A.replace("Home", ssid))
        if manager_env.fake is not None:
          manager_env.fake.networks = {}
          manager_env.fake.requests.clear()
          manager_env.fake.status = {"wpa_state": "DISCONNECTED"}
        wm = start_manager(manager_env)
        wm.set_active(False)
        workers = []
        thread_class = wifi_manager.threading.Thread

        def tracked_thread(*args, thread_class=thread_class, workers=workers, **kwargs):
          worker = thread_class(*args, **kwargs)
          workers.append(worker)
          return worker

        with unittest.mock.patch.object(wifi_manager.threading, "Thread", tracked_thread):
          if name == "connect":
            forget = wm.forget_connection
            entered, release = threading.Event(), threading.Event()

            def gated_forget(ssid, block=False, forget=forget, entered=entered, release=release):
              forget(ssid, block=block)
              if ssid == "B":
                entered.set()
                release.wait()

            wm.forget_connection = gated_forget
            wm.connect_to_network("B", "password123")
            self.assertTrue(entered.wait(2))
            worker = workers[-1]
            try:
              wm.activate_connection("C", block=True)
            finally:
              release.set()
          else:
            with wm._lock:
              wm.activate_connection("B")
              worker = workers[-1]
              wm.activate_connection("C", block=True)
          worker.join(2)
        self.assertFalse(worker.is_alive())
        c_id = next(nid for nid, ssid in wm._network_ids.items() if ssid == "C")
        self.assertEqual(wm.wifi_state, wifi_manager.WifiState("C", wifi_manager.ConnectStatus.CONNECTING))
        self.assertEqual([r for r in manager_env.fake.requests if r.startswith("SELECT_NETWORK")], [f"SELECT_NETWORK {c_id}"])
        self.assertNotIn("B", wm._network_ids.values())
        wm.stop()

  def test_connect_replaces_saved_network_without_clearing_selection(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    wm = start_manager(manager_env)
    wm.connect_to_network("Home", "replacement")
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    (nid,) = manager_env.fake.networks
    self.assertEqual(wm.connecting_to_ssid, "Home")
    self.assertEqual(wm._pending.psk, "replacement")
    self.assertEqual(profile_files(manager_env), [])

    manager_env.fake.status = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "id": str(nid)}
    manager_env.fake.emit(f"CTRL-EVENT-CONNECTED - Connection to aa:bb completed [id={nid} id_str=]")
    wait_for(lambda: manager_env.fake.requests.count("ENABLE_NETWORK all") >= 2)
    self.assertEqual(profile_files(manager_env), [])

    manager_env.fake.status["ip_address"] = "10.0.0.9"
    wait_for(lambda: wm.connected_ssid == "Home")
    self.assertEqual(next(p.psk for p in wifi_manager.read_profiles() if p.ssid == "Home"), "replacement")


AP_STATUS = {"wpa_state": "COMPLETED", "ssid": "weedle", "mode": "AP", "id": "5"}
STATION_STATUS = {"wpa_state": "COMPLETED", "ssid": "Home", "mode": "station", "ip_address": "10.0.0.5", "id": "0"}


def ap_on_select(fake):
  def reply(cmd):
    nid = int(cmd.split()[1])
    if fake.networks[nid].get("mode") == "2":
      fake.status = {**AP_STATUS, "id": str(nid)}
    return "OK\n"
  fake.replies["SELECT_NETWORK"] = reply


class TestTethering(OpenpilotTestCase):
  def test_hotspot_profile_is_created_on_first_start(self, manager_env):
    wm = start_manager(manager_env)
    self.assertEqual(os.listdir(manager_env.dirs["persistent"]), ["weedle.nmconnection"])
    self.assertEqual(wm.tethering_password, "swagswagcomma")
    self.assertTrue(wifi_manager.read_profiles()[0].is_ap)
    self.assertEqual(manager_env.fake.networks, {})

  def test_tethering_on_then_off(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {0: {"ssid": b"Home".hex()}}
    manager_env.alive(manager_env.udhcpc_pid)
    wm = start_manager(manager_env)
    activated, disconnected = [], []
    wm.add_callbacks(activated=lambda: activated.append(True), disconnected=lambda: disconnected.append(True))
    ap_on_select(manager_env.fake)
    del manager_env.sudo[:]

    wm.set_tethering_active(True)
    wait_for(lambda: wm.connected_ssid == "weedle", timeout=10)
    wait_for(lambda: drain(wm, activated))
    self.assertTrue(wm.is_tethering_active())
    self.assertEqual(wm.ipv4_address, "192.168.43.1")
    ap = next(n for n in manager_env.fake.networks.values() if n.get("mode") == "2")
    self.assertEqual(ap, {"ssid": b"weedle".hex(), "mode": "2", "frequency": "2437", "key_mgmt": "WPA-PSK", "proto": "RSN", "pairwise": "CCMP",
                          "psk": wpa_psk("weedle", "swagswagcomma")})
    self.assertEqual(manager_env.sudo[0], ["kill", str(os.getpid())])
    flush = ["ip", "addr", "flush", "dev", "wlan0"]
    add = ["ip", "addr", "add", "192.168.43.1/24", "dev", "wlan0"]
    self.assertIn(flush, manager_env.sudo)  # codespell:ignore assertin
    self.assertIn(add, manager_env.sudo)  # codespell:ignore assertin
    self.assertLess(manager_env.sudo.index(flush), manager_env.sudo.index(add))
    self.assertEqual(manager_env.dnsmasq,
                     (["sudo", "dnsmasq", "--interface=wlan0", "--bind-interfaces", "--except-interface=lo", "--dhcp-range=192.168.43.2,192.168.43.254,24h",
                       f"--pid-file={manager_env.dnsmasq_pid}"],
                      {"check": True, "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "start_new_session": True}))
    self.assertIn(["iptables-legacy", "-t", "nat", "-A", *wifi_manager.TETHERING_NAT_RULE], manager_env.sudo)  # codespell:ignore assertin
    self.assertEqual(manager_env.sudo[-1], ["sysctl", "net.ipv4.ip_forward=0"])

    wm.set_ipv4_forward(True)
    self.assertEqual(manager_env.sudo[-1], ["sysctl", "net.ipv4.ip_forward=1"])

    manager_env.alive(manager_env.dnsmasq_pid)
    del manager_env.sudo[:]
    del manager_env.operations[:]
    manager_env.fake.replies.pop("SELECT_NETWORK")
    n = len(manager_env.fake.requests)
    wm.set_tethering_active(False)
    wait_for(lambda: drain(wm, disconnected), timeout=10)
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertFalse(wm.is_tethering_active())
    self.assertEqual(manager_env.sudo[0], ["kill", str(os.getpid())])
    self.assertIn(["iptables-legacy", "-t", "nat", "-D", *wifi_manager.TETHERING_NAT_RULE], manager_env.sudo)  # codespell:ignore assertin
    self.assertIn(["ip", "addr", "flush", "dev", "wlan0"], manager_env.sudo)  # codespell:ignore assertin
    self.assertNotIn("2", [net.get("mode") for net in manager_env.fake.networks.values()])
    self.assertIn("ENABLE_NETWORK all", manager_env.fake.requests[n:])  # codespell:ignore assertin
    self.assertLess(manager_env.operations.index("ENABLE_NETWORK all"), manager_env.operations.index("ip addr flush dev wlan0"))
    self.assertTrue(["kill", "-USR1", str(os.getpid())] in manager_env.sudo or manager_env.popen)

  def test_adopts_running_hotspot(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Hotspot.nmconnection"), HOTSPOT_KEYFILE)
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    wm = start_manager(manager_env)
    self.assertTrue(wm.is_tethering_active())
    self.assertEqual(wm.ipv4_address, "192.168.43.1")
    self.assertEqual(manager_env.popen, [])
    self.assertEqual(manager_env.dnsmasq[0][:2], ["sudo", "dnsmasq"])
    self.assertEqual(manager_env.dnsmasq[1], {"check": True, "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                                               "stderr": subprocess.DEVNULL, "start_new_session": True})
    self.assertIn(["ip", "addr", "replace", "192.168.43.1/24", "dev", "wlan0"], manager_env.sudo)  # codespell:ignore assertin
    self.assertEqual([c[0] for c in manager_env.sudo], ["ip", "iptables-legacy", "iptables-legacy", "sysctl"])

  def test_tethering_password_change_restarts_hotspot(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Hotspot.nmconnection"), HOTSPOT_KEYFILE)
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    wm = start_manager(manager_env)
    ap_on_select(manager_env.fake)
    wm.set_tethering_password("newpass123")
    wait_for(lambda: wm.tethering_password == "newpass123")
    wait_for(lambda: any(n.get("psk") == wpa_psk("weedle", "newpass123") for n in list(manager_env.fake.networks.values())), timeout=10)
    self.assertIn("REMOVE_NETWORK 5", manager_env.fake.requests)  # codespell:ignore assertin
    self.assertEqual(wifi_manager.read_profiles()[0].psk, "newpass123")
    wait_for(lambda: wm.connected_ssid == "weedle")

  def test_tethering_password_change_restarts_connecting_hotspot(self, manager_env):
    wm = start_manager(manager_env)
    wm.set_tethering_active(True)
    wait_for(lambda: any(network.get("mode") == "2" for network in manager_env.fake.networks.values()))
    ap_on_select(manager_env.fake)
    wm.set_tethering_password("newpass123")
    wait_for(lambda: wm.tethering_password == "newpass123")
    wait_for(lambda: wm.connected_ssid == "weedle")
    self.assertEqual(next(network["psk"] for network in manager_env.fake.networks.values() if network.get("mode") == "2"),
                     wpa_psk("weedle", "newpass123"))

  def test_tethering_start_uses_password_written_before_worker_runs(self, manager_env):
    wm = start_manager(manager_env)
    start_tethering = wm._start_tethering
    entered, release = threading.Event(), threading.Event()

    def delayed_start(*args, **kwargs):
      entered.set()
      release.wait()
      return start_tethering(*args, **kwargs)

    wm._start_tethering = delayed_start
    wm.set_tethering_active(True)
    try:
      self.assertTrue(entered.wait(2))
      wm.set_tethering_password("newpass123")
      wait_for(lambda: wm.tethering_password == "newpass123")
    finally:
      release.set()
    wait_for(lambda: any(network.get("mode") == "2" and "psk" in network for network in manager_env.fake.networks.values()))
    self.assertEqual(next(network["psk"] for network in manager_env.fake.networks.values() if network.get("mode") == "2"),
                     wpa_psk("weedle", "newpass123"))

  def test_tethering_start_failure_resets_state(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "AP_TIMEOUT_SECONDS", 0.3)
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    wm.set_tethering_active(True)
    wait_for(lambda: wm.connecting_to_ssid == "weedle")
    wait_for(lambda: not wm.is_tethering_active(), timeout=10)
    wait_for(lambda: failure.called)
    wm.set_active(True)
    wait_for(lambda: wm.connected_ssid == "Home")

  def test_tethering_retries_transient_ready_status(self, manager_env):
    wm = start_manager(manager_env)
    ap_on_select(manager_env.fake)
    status = wm._status
    calls = 0

    def transient_status():
      nonlocal calls
      calls += 1
      if calls == 2:
        return {**AP_STATUS, "wpa_state": "ASSOCIATING"}
      return status()

    wm._status = transient_status
    wm.set_tethering_active(True)
    wait_for(lambda: wm.connected_ssid == "weedle")
    self.assertGreaterEqual(calls, 4)

  def test_tethering_start_refuses_live_dhcp(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "CTRL_TIMEOUT_SECONDS", 0.1)
    wm = start_manager(manager_env)
    manager_env.alive(manager_env.udhcpc_pid)
    manager_env.keep_pids.add(manager_env.udhcpc_pid)
    ap_on_select(manager_env.fake)
    hotspot = wm._hotspot_profile()
    with wm._lock:
      wm._selected = hotspot.ssid
      wm._wifi_state = wifi_manager.WifiState(hotspot.ssid, wifi_manager.ConnectStatus.CONNECTING)
    with self.assertRaisesRegex(RuntimeError, "udhcpc"):
      wm._start_tethering()
    self.assertFalse(any(n.get("mode") == "2" for n in manager_env.fake.networks.values()))

  def test_tethering_stop_refuses_live_dnsmasq(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "CTRL_TIMEOUT_SECONDS", 0.1)
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    manager_env.alive(manager_env.dnsmasq_pid)
    manager_env.keep_pids.add(manager_env.dnsmasq_pid)
    wm = start_manager(manager_env)
    with self.assertRaisesRegex(RuntimeError, "dnsmasq"):
      wm._stop_tethering()
    self.assertEqual(wm.connected_ssid, "weedle")

  def test_tethering_select_failure_reconciles_without_callbacks(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "AP_TIMEOUT_SECONDS", 0.1)
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    activated, disconnected = [], []
    wm.add_callbacks(activated=lambda: activated.append(True), disconnected=lambda: disconnected.append(True))
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    manager_env.fake.replies["SELECT_NETWORK"] = "FAIL"
    wm.set_tethering_active(True)
    wait_for(lambda: failure.called)
    wait_for(lambda: wm.connected_ssid == "Home")
    self.assertEqual(drain(wm, activated), [])
    self.assertEqual(disconnected, [])

  def test_tethering_stop_remove_failure_reconciles_without_callback(self, manager_env, mocker):
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    wm = start_manager(manager_env)
    disconnected = []
    wm.add_callbacks(disconnected=lambda: disconnected.append(True))
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    manager_env.fake.replies["REMOVE_NETWORK 5"] = "FAIL"
    del manager_env.operations[:]
    wm.set_tethering_active(False)
    wait_for(lambda: failure.called)
    wait_for(lambda: wm.connected_ssid == "weedle")
    self.assertEqual(drain(wm, disconnected), [])
    self.assertIn(5, manager_env.fake.networks)  # codespell:ignore assertin
    self.assertNotIn("ENABLE_NETWORK all", manager_env.operations)
    self.assertNotIn("ip addr flush dev wlan0", manager_env.operations)

  def test_tethering_stop_enable_failure_reconciles_without_callback(self, manager_env, mocker):
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    wm = start_manager(manager_env)
    disconnected = []
    wm.add_callbacks(disconnected=lambda: disconnected.append(True))
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    manager_env.fake.replies["ENABLE_NETWORK all"] = "FAIL"
    del manager_env.operations[:]
    wm.set_tethering_active(False)
    wait_for(lambda: failure.called)
    self.assertEqual(drain(wm, disconnected), [])
    self.assertNotIn("ip addr flush dev wlan0", manager_env.operations)

  def test_connect_replaces_running_hotspot(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    wm = start_manager(manager_env)
    activated = []
    wm.add_callbacks(activated=lambda: activated.append(True))

    def select_network(cmd):
      nid = int(cmd.split()[1])
      if manager_env.fake.networks[nid].get("mode") == "2":
        manager_env.fake.status = {**AP_STATUS, "id": str(nid)}
      else:
        manager_env.fake.status = {**STATION_STATUS, "id": str(nid)}
      return "OK\n"

    manager_env.fake.replies["SELECT_NETWORK"] = select_network
    wm.connect_to_network("Home", "password123")
    wait_for(lambda: any(network.get("ssid") == b"Home".hex() for network in manager_env.fake.networks.values()))
    self.assertFalse(any(network.get("mode") == "2" for network in manager_env.fake.networks.values()))
    self.assertEqual(wm.connecting_to_ssid, "Home")
    self.assertEqual(wm._pending.ssid, "Home")
    wm._on_associated()
    wait_for(lambda: drain(wm, activated))
    self.assertEqual(wm.connected_ssid, "Home")

  def test_activate_replaces_running_hotspot(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    wm = start_manager(manager_env)
    activated = []
    wm.add_callbacks(activated=lambda: activated.append(True))

    def select_network(cmd):
      nid = int(cmd.split()[1])
      if manager_env.fake.networks[nid].get("mode") == "2":
        manager_env.fake.status = {**AP_STATUS, "id": str(nid)}
      else:
        manager_env.fake.status = {**STATION_STATUS, "id": str(nid)}
      return "OK\n"

    manager_env.fake.replies["SELECT_NETWORK"] = select_network
    wm.activate_connection("Home")
    wait_for(lambda: any(network.get("ssid") == b"Home".hex() for network in manager_env.fake.networks.values()))
    self.assertFalse(any(network.get("mode") == "2" for network in manager_env.fake.networks.values()))
    self.assertEqual(wm.connecting_to_ssid, "Home")
    self.assertIsNone(wm._pending)
    wm._on_associated()
    wait_for(lambda: drain(wm, activated))
    self.assertEqual(wm.connected_ssid, "Home")

  def test_delayed_tethering_start_keeps_new_station_selection(self, manager_env):
    wm = start_manager(manager_env)
    ap_on_select(manager_env.fake)
    entered, release = threading.Event(), threading.Event()
    start_tethering = wm._start_tethering

    def delayed_start(*args, **kwargs):
      entered.set()
      release.wait()
      start_tethering(*args, **kwargs)

    wm._start_tethering = delayed_start
    wm.set_tethering_active(True)
    self.assertTrue(entered.wait(2))
    try:
      wm.connect_to_network("Home", "password123")
    finally:
      release.set()
    wait_for(lambda: wm.connecting_to_ssid == "Home")
    time.sleep(0.2)
    self.assertFalse(any(n.get("mode") == "2" for n in manager_env.fake.networks.values()))

  def test_tethering_success_rechecks_claim_before_services(self, manager_env):
    wm = start_manager(manager_env)
    ap_on_select(manager_env.fake)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    status = wm._status
    start_tethering = wm._start_tethering
    ap_thread = None

    def wrapped_start_tethering():
      nonlocal ap_thread
      ap_thread = threading.get_ident()
      try:
        start_tethering()
      finally:
        finished.set()

    def gated_status():
      result = status()
      if threading.get_ident() == ap_thread:
        entered.set()
        release.wait()
      return result

    wm._start_tethering = wrapped_start_tethering
    wm._status = gated_status
    wm.set_tethering_active(True)
    try:
      self.assertTrue(entered.wait(2))
      wm.connect_to_network("Home", "password123")
      wait_for(lambda: any(net.get("ssid") == b"Home".hex() for net in manager_env.fake.networks.values()))
      with wm._lock:
        pass
      manager_env.operations.clear()
    finally:
      release.set()
    self.assertTrue(finished.wait(2))
    self.assertNotIn("ip addr flush dev wlan0", manager_env.operations)
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState("Home", wifi_manager.ConnectStatus.CONNECTING))

  def test_tethering_waits_for_its_selected_network_id(self, manager_env):
    wm = start_manager(manager_env)

    def wrong_id(cmd):
      manager_env.fake.status = {**AP_STATUS, "id": "99"}
      return "OK\n"

    manager_env.fake.replies["SELECT_NETWORK"] = wrong_id
    wm.set_tethering_active(True)
    wait_for(lambda: manager_env.fake.networks)
    time.sleep(0.3)
    self.assertEqual(wm.connecting_to_ssid, "weedle")
    (nid,) = manager_env.fake.networks
    manager_env.fake.status["id"] = str(nid)
    wait_for(lambda: wm.connected_ssid == "weedle")

  def test_tethering_timeout_keeps_new_station_selection(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "AP_TIMEOUT_SECONDS", 0.2)
    wm = start_manager(manager_env)
    disconnected = []
    wm.add_callbacks(disconnected=lambda: disconnected.append(True))
    wm.set_tethering_active(True)
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    wm.connect_to_network("Home", "password123")
    wait_for(lambda: wm.connecting_to_ssid == "Home")
    time.sleep(0.4)
    self.assertEqual(wm.connecting_to_ssid, "Home")
    self.assertEqual(drain(wm, disconnected), [])

  def test_pending_tethering_off_removes_ap_and_restarts_dhcp(self, manager_env):
    wm = start_manager(manager_env)
    manager_env.alive(manager_env.udhcpc_pid)
    wm.set_tethering_active(True)
    wait_for(lambda: any(network.get("mode") == "2" for network in manager_env.fake.networks.values()))
    (nid,) = manager_env.fake.networks
    status = wm._status
    status_entered, status_release = threading.Event(), threading.Event()
    worker_entered, worker_release = threading.Event(), threading.Event()
    thread_class = wifi_manager.threading.Thread
    workers = []

    def gated_status():
      status_entered.set()
      status_release.wait()
      return status()

    def delayed_thread(*args, **kwargs):
      target = kwargs.pop("target")
      worker = thread_class(target=lambda: (worker_entered.set(), worker_release.wait(), target()), **kwargs)
      workers.append(worker)
      return worker

    wm._status = gated_status
    try:
      self.assertTrue(status_entered.wait(2))
      with unittest.mock.patch.object(wifi_manager.threading, "Thread", delayed_thread):
        wm.set_tethering_active(False)
        self.assertTrue(worker_entered.wait(2))
      status_release.set()
      time.sleep(0.3)
      self.assertIn(nid, manager_env.fake.networks)  # codespell:ignore assertin
    finally:
      status_release.set()
      worker_release.set()
    for worker in workers:
      worker.join(2)
      self.assertFalse(worker.is_alive())
    wait_for(lambda: nid not in manager_env.fake.networks)
    wait_for(lambda: len(manager_env.popen) == 2)
    self.assertIn(f"REMOVE_NETWORK {nid}", manager_env.fake.requests)  # codespell:ignore assertin
    self.assertIn("ENABLE_NETWORK all", manager_env.fake.requests)  # codespell:ignore assertin

  def test_delayed_tethering_stop_keeps_new_station_selection(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    wm = start_manager(manager_env)
    activated = []
    wm.add_callbacks(activated=lambda: activated.append(True))
    manager_env.fake.replies["SELECT_NETWORK"] = lambda cmd: (manager_env.fake.status.update({**STATION_STATUS, "id": cmd.split()[1]}), "OK\n")[1]
    entered, release = threading.Event(), threading.Event()
    thread_class = wifi_manager.threading.Thread
    workers = []

    def delayed_thread(*args, **kwargs):
      target = kwargs.pop("target")
      worker = thread_class(target=lambda: (entered.set(), release.wait(), target()), **kwargs)
      workers.append(worker)
      return worker

    with unittest.mock.patch.object(wifi_manager.threading, "Thread", delayed_thread):
      wm.set_tethering_active(False)
      self.assertTrue(entered.wait(2))
    wm.connect_to_network("Home", "password123")
    try:
      wait_for(lambda: wm.connecting_to_ssid == "Home")
      wait_for(lambda: any(network.get("ssid") == b"Home".hex() for network in manager_env.fake.networks.values()))
      self.assertFalse(any(network.get("mode") == "2" for network in manager_env.fake.networks.values()))
      self.assertEqual(wm._pending.ssid, "Home")
      wm._on_associated()
      wait_for(lambda: drain(wm, activated))
    finally:
      release.set()
    for worker in workers:
      worker.join(2)
      self.assertFalse(worker.is_alive())
    self.assertEqual(wm.connected_ssid, "Home")

  def test_delayed_tethering_stop_keeps_newer_ap(self, manager_env):
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    manager_env.fake.networks = {5: {"ssid": b"weedle".hex(), "mode": "2"}}
    wm = start_manager(manager_env)
    ap_on_select(manager_env.fake)
    entered, release = threading.Event(), threading.Event()
    thread_class = wifi_manager.threading.Thread
    workers = []

    def delayed_thread(*args, **kwargs):
      target = kwargs.pop("target")
      worker = thread_class(target=lambda: (entered.set(), release.wait(), target()), **kwargs)
      workers.append(worker)
      return worker

    try:
      with unittest.mock.patch.object(wifi_manager.threading, "Thread", delayed_thread):
        wm.set_tethering_active(False)
        self.assertTrue(entered.wait(2))
      wm.set_tethering_active(True)
      wait_for(lambda: wm.connected_ssid == "weedle")
      new_nid = max(manager_env.fake.networks)
      self.assertNotEqual(new_nid, 5)
    finally:
      release.set()
    for worker in workers:
      worker.join(2)
      self.assertFalse(worker.is_alive())
    self.assertEqual(wm.connected_ssid, "weedle")
    self.assertIn(new_nid, manager_env.fake.networks)  # codespell:ignore assertin

  def test_metering_does_not_follow_a_new_connection(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    write(os.path.join(manager_env.dirs["persistent"], "Café.nmconnection"), KEYFILE_A.replace("Home", "Café"))
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    entered, release = threading.Event(), threading.Event()
    thread_class = wifi_manager.threading.Thread
    workers = []

    def delayed_thread(*args, **kwargs):
      target = kwargs.pop("target")
      worker = thread_class(target=lambda: (entered.set(), release.wait(), target()), **kwargs)
      workers.append(worker)
      return worker

    try:
      with unittest.mock.patch.object(wifi_manager.threading, "Thread", delayed_thread):
        wm.set_current_network_metered(wifi_manager.MeteredType.YES)
        self.assertTrue(entered.wait(2))
        manager_env.fake.status = {**STATION_STATUS, "ssid": "Café", "id": "1"}
        wm._refresh_status()
    finally:
      release.set()
    for worker in workers:
      worker.join(2)
      self.assertFalse(worker.is_alive())
    metered = {p.ssid: p.metered for p in wifi_manager.read_profiles() if not p.is_ap}
    self.assertEqual(metered, {"Home": wifi_manager.MeteredType.UNKNOWN, "Café": wifi_manager.MeteredType.UNKNOWN})

  def test_stop_during_tethering_readiness_leaves_networking_running(self, manager_env, mocker):
    mocker.patch.object(wifi_manager, "AP_TIMEOUT_SECONDS", 0.2)
    wm = start_manager(manager_env)
    wm.set_tethering_active(True)
    wait_for(lambda: any(r.startswith("SELECT_NETWORK") for r in manager_env.fake.requests))
    n = len(manager_env.fake.requests)
    del manager_env.operations[:]
    wm.stop()
    time.sleep(0.4)
    self.assertNotIn("ENABLE_NETWORK all", manager_env.fake.requests[n:])
    self.assertNotIn("ip addr flush dev wlan0", manager_env.operations)


class TestMetering(OpenpilotTestCase):
  def test_metered_rewrites_connected_profile(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    self.assertEqual(wm.current_network_metered, wifi_manager.MeteredType.UNKNOWN)
    wm.set_current_network_metered(wifi_manager.MeteredType.YES)
    wait_for(lambda: wm.current_network_metered == wifi_manager.MeteredType.YES)
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"))
    self.assertEqual(cp["connection"]["metered"], "1")
    self.assertEqual(cp["wifi-security"]["psk"], "password123")

  def test_metered_migrates_netplan_profile(self, manager_env):
    runtime = os.path.join(manager_env.dirs["runtime"], "netplan-NM-22222222-2222-2222-2222-222222222222-Caf.nmconnection")
    write(runtime, NETPLAN_KEYFILE)
    yaml = os.path.join(manager_env.dirs["netplan"], "90-NM-22222222-2222-2222-2222-222222222222.yaml")
    write(yaml, "network: {}\n")
    # codespell:ignore-next-line caf
    manager_env.spawn_status = {"wpa_state": "COMPLETED", "ssid": "Caf\\xc3\\xa9", "mode": "station", "ip_address": "10.0.0.7", "id": "0"}
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    self.assertEqual(wm.connected_ssid, "Café")
    wm.set_current_network_metered(wifi_manager.MeteredType.NO)
    wait_for(lambda: wm.current_network_metered == wifi_manager.MeteredType.NO)
    self.assertFalse(os.path.exists(runtime))
    self.assertFalse(os.path.exists(yaml))
    (profile,) = [p for p in wifi_manager.read_profiles() if not p.is_ap]
    self.assertEqual((profile.ssid, profile.psk, profile.hidden, profile.metered, profile.uuid),
                     ("Café", "cafepass1", True, wifi_manager.MeteredType.NO, "22222222-2222-2222-2222-222222222222"))

  def test_metered_ignored_while_tethering(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Hotspot.nmconnection"), HOTSPOT_KEYFILE)
    manager_env.spawn_status = dict(AP_STATUS)
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    before = manager_env.sudo[:]
    wm.set_current_network_metered(wifi_manager.MeteredType.YES)
    time.sleep(0.3)
    self.assertEqual(manager_env.sudo, before)
    self.assertEqual(wm.current_network_metered, wifi_manager.MeteredType.UNKNOWN)


class TestHealth(OpenpilotTestCase):
  def test_reattaching_healthy_supplicant_preserves_pending_hotspot_without_services(self, manager_env):
    wm = start_manager(manager_env)
    hotspot = wm._hotspot_profile()
    events = wm._events
    manager_env.operations.clear()
    manager_env.popen.clear()
    with wm._lock:
      wm._selected, wm._pending = hotspot.ssid, hotspot
      wm._wifi_state = wifi_manager.WifiState(hotspot.ssid, wifi_manager.ConnectStatus.CONNECTING)
      wm._events = None
    wait_for(lambda: wm._events is not None and wm._events is not events, timeout=10)
    self.assertEqual(wm._selected, hotspot.ssid)
    self.assertEqual(wm._pending, hotspot)
    self.assertIsNone(manager_env.dnsmasq)
    self.assertEqual(manager_env.popen, [])
    self.assertFalse(any(command.startswith(("ip ", "iptables-legacy ")) for command in manager_env.operations))

  def test_reattaching_nonready_access_point_does_not_start_dhcp(self, manager_env):
    wm = start_manager(manager_env)
    events = wm._events
    manager_env.fake.status = {"wpa_state": "ASSOCIATING", "ssid": "weedle", "mode": "AP", "id": "5"}
    manager_env.popen.clear()
    with wm._lock:
      wm._events = None
    events.close()
    wait_for(lambda: wm._events is not None and wm._events is not events, timeout=10)
    self.assertEqual(manager_env.popen, [])

  def test_missing_event_recovery_waits_between_failed_attempts(self, manager_env, mocker):
    wm = start_manager(manager_env)
    events = wm._events
    attempts = []
    checked = threading.Event()

    def failed_recovery():
      attempts.append(time.monotonic())
      checked.set()

    mocker.patch.object(wm, "_check_daemons", side_effect=failed_recovery)
    with wm._lock:
      wm._events = None
    events.close()
    try:
      self.assertTrue(checked.wait(2))
      time.sleep(0.1)
    finally:
      wm.stop()
    self.assertEqual(len(attempts), 1)

  def test_fresh_supplicant_restart_clears_pending_hotspot_state(self, manager_env):
    wm = start_manager(manager_env)
    hotspot = wm._hotspot_profile()
    with wm._lock:
      wm._selected, wm._pending = hotspot.ssid, hotspot
      wm._wifi_state = wifi_manager.WifiState(hotspot.ssid, wifi_manager.ConnectStatus.CONNECTING)
    first = manager_env.fake
    first.close()
    os.unlink(first.path)
    os.unlink(manager_env.wpa_pid)
    wait_for(lambda: manager_env.fake is not first, timeout=10)
    wait_for(lambda: wm._selected is None and wm._pending is None, timeout=10)
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertIsNone(wm._selected)
    self.assertIsNone(wm._pending)
    self.assertIsNone(manager_env.dnsmasq)
    self.assertNotIn("ip addr flush dev wlan0", manager_env.operations)

  def test_dhcp_restart_failure_is_logged_and_refreshes_status(self, manager_env, mocker):
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    manager_env.alive(manager_env.udhcpc_pid)
    wm = start_manager(manager_env)
    os.unlink(manager_env.udhcpc_pid)
    with wm._lock:
      wm._selected = "Later"
      wm._wifi_state = wifi_manager.WifiState("Later", wifi_manager.ConnectStatus.CONNECTING)
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    refresh = mocker.patch.object(wm, "_refresh_status", wraps=wm._refresh_status)
    mocker.patch.object(wm, "_start_dhcp", side_effect=OSError("udhcpc failed"))
    wm._check_daemons()
    self.assertTrue(failure.called)
    self.assertTrue(refresh.called)
    self.assertEqual(wm.connecting_to_ssid, "Later")

  def test_supplicant_restart_failure_refreshes_status(self, manager_env, mocker):
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    wm = start_manager(manager_env)
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState("Home", wifi_manager.ConnectStatus.CONNECTED))
    manager_env.fake.status = {"wpa_state": "DISCONNECTED"}
    failure = mocker.patch.object(wifi_manager.cloudlog, "exception")
    refresh = mocker.patch.object(wm, "_refresh_status", wraps=wm._refresh_status)
    request = wm._ctrl.request
    mocker.patch.object(wm._ctrl, "request", side_effect=lambda cmd: "FAIL" if cmd == "PING" else request(cmd))
    mocker.patch.object(wm, "_connect_ctrl", side_effect=OSError("restart failed"))
    wm._check_daemons()
    self.assertTrue(failure.called)
    self.assertTrue(refresh.called)
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())
    self.assertEqual(wm.ipv4_address, "")

  def test_health_does_not_restart_dhcp_after_stop(self, manager_env):
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    manager_env.alive(manager_env.udhcpc_pid)
    wm = start_manager(manager_env)
    entered, release = threading.Event(), threading.Event()
    request = wm._ctrl.request

    def delayed_ping(cmd):
      if cmd == "PING":
        entered.set()
        release.wait()
        return "PONG"
      return request(cmd)

    wm._ctrl.request = delayed_ping
    os.unlink(manager_env.udhcpc_pid)
    health = threading.Thread(target=wm._check_daemons)
    stopped = threading.Thread(target=wm.stop)
    health.start()
    try:
      self.assertTrue(entered.wait(2))
      stopped.start()
      wait_for(lambda: wm._exit.is_set())
    finally:
      release.set()
    health.join(2)
    stopped.join(2)
    self.assertFalse(health.is_alive())
    self.assertFalse(stopped.is_alive())
    self.assertEqual(manager_env.popen, [])

  def test_missing_event_socket_reattaches_healthy_supplicant(self, manager_env):
    wm = start_manager(manager_env)
    ctrl = wm._ctrl
    events = wm._events
    with wm._lock:
      wm._events = None
    events.close()
    wait_for(lambda: wm._events is not None and wm._ctrl is not ctrl and ctrl._sock.fileno() == -1, timeout=10)

  def test_crashed_supplicant_is_respawned_with_saved_networks(self, manager_env):
    write(os.path.join(manager_env.dirs["persistent"], "Home.nmconnection"), KEYFILE_A)
    wm = start_manager(manager_env)
    first = manager_env.fake
    first.close()
    os.unlink(first.path)
    os.unlink(manager_env.wpa_pid)
    wait_for(lambda: manager_env.fake is not first, timeout=10)
    wait_for(lambda: len(manager_env.fake.networks) == 1 and "ENABLE_NETWORK all" in manager_env.fake.requests, timeout=10)
    self.assertEqual(bytes.fromhex(manager_env.fake.networks[0]["ssid"]).decode(), "Home")
    self.assertEqual(len(manager_env.supplicants), 2)
    self.assertEqual(wm.wifi_state, wifi_manager.WifiState())

  def test_dead_udhcpc_is_restarted_while_connected(self, manager_env):
    manager_env.spawn_status = dict(STATION_STATUS)
    manager_env.spawn_fake()
    manager_env.alive(manager_env.udhcpc_pid)
    wm = start_manager(manager_env)
    self.assertEqual(manager_env.popen, [])
    os.unlink(manager_env.udhcpc_pid)
    wait_for(lambda: manager_env.popen, timeout=10)
    self.assertEqual(manager_env.popen[0][:2], ["sudo", "udhcpc"])
    self.assertEqual(wm.connected_ssid, "Home")


class TestShippedFiles(OpenpilotTestCase):
  def test_supplicant_conf_has_no_secrets(self):
    with open(wifi_manager.WPA_CONF_PATH) as f:
      lines = [line.strip() for line in f if line.strip()]
    self.assertEqual(lines, ["ctrl_interface=DIR=/run/wpa_supplicant GROUP=netdev", "update_config=0"])

  def test_udhcpc_script_sets_wifi_metric(self):
    self.assertTrue(os.access(wifi_manager.UDHCPC_SCRIPT_PATH, os.X_OK))
    with tempfile.TemporaryDirectory() as d:
      fake_bin = os.path.join(d, "bin")
      os.makedirs(fake_bin)
      log = os.path.join(d, "calls")
      write(os.path.join(fake_bin, "busybox"), f'''#!/bin/sh
if [ "$1" = ip ] && [ "$2" = -4 ] && [ "$3" = route ] && [ "$4" = show ]; then
  echo "$@" >> {log}
  if [ "$subnet" != 255.255.255.255 ]; then
    printf '%s\\n' '10.0.0.0/25 proto kernel scope link src 10.0.0.20' '10.0.0.0/24 proto kernel scope link src 10.0.0.2'
  fi
elif [ "$1" = awk ]; then
  shift
  awk "$@"
else
  echo "$@" >> {log}
fi
''')
      os.chmod(os.path.join(fake_bin, "busybox"), 0o755)
      write(os.path.join(d, "default.script"), f'#!/bin/sh\necho "default $1" >> {log}\n')
      os.chmod(os.path.join(d, "default.script"), 0o755)
      env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "UDHCPC_DEFAULT_SCRIPT": os.path.join(d, "default.script"),
             "interface": "wlan0", "ip": "10.0.0.2", "router": "10.0.0.1 10.0.0.2", "subnet": "255.255.255.0"}
      subprocess.run([wifi_manager.UDHCPC_SCRIPT_PATH, "bound"], check=True, env=env)
      subprocess.run([wifi_manager.UDHCPC_SCRIPT_PATH, "deconfig"], check=True, env=env)
      with open(log) as f:
        self.assertEqual(f.read().splitlines(), ["default bound", "ip -4 route show dev wlan0 proto kernel scope link",
                                                 "ip -4 route del 10.0.0.0/24 dev wlan0",
                                                 "ip -4 route add 10.0.0.0/24 dev wlan0 proto kernel scope link src 10.0.0.2 metric 600",
                                                 "ip -4 route flush exact 0.0.0.0/0 dev wlan0",
                                                 "ip -4 route add default via 10.0.0.1 dev wlan0 metric 600", "default deconfig"])
      env["subnet"] = "255.255.255.255"
      subprocess.run([wifi_manager.UDHCPC_SCRIPT_PATH, "renew"], check=True, env=env)
      with open(log) as f:
        self.assertEqual(f.read().splitlines()[-4:], ["default renew", "ip -4 route show dev wlan0 proto kernel scope link",
                                                       "ip -4 route flush exact 0.0.0.0/0 dev wlan0",
                                                       "ip -4 route add default via 10.0.0.1 dev wlan0 onlink metric 600"])
