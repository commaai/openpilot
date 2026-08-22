import os
import re
import shutil
import socket
import select
import subprocess
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import atomic_write
from openpilot.common.wifi import WPA_CTRL_DIR, WPA_CTRL_PATH, WPA_PID_FILE, decode_wpa_ssid


RECV_BUF_SIZE = 32768
IEEE80211_MAX_SSID_BYTES = 32

WPA_SUPPLICANT_CONF = "/tmp/wpa_supplicant.conf"
WPA_AP_CONF = "/tmp/wpa_supplicant_ap.conf"
WPA_CTRL_INTERFACE = f"ctrl_interface=DIR={WPA_CTRL_DIR} GROUP=netdev"
TETHERING_DNSMASQ_PATTERN = r"^dnsmasq .*--dhcp-range=192\.168\.43\.2"


class SecurityType(IntEnum):
  OPEN = 0
  WPA = 1
  UNSUPPORTED = 2


@dataclass(frozen=True)
class ScanResult:
  bssid: str
  freq: int
  signal: int  # dBm
  flags: str
  ssid: str


class _WpaCtrlBase:
  """Shared socket lifecycle for wpa_supplicant control connections."""

  _counter = 0
  _counter_lock = threading.Lock()

  def __init__(self, ctrl_path: str = WPA_CTRL_PATH):
    self._ctrl_path = ctrl_path
    self._sock: socket.socket | None = None
    self._local_path: str = ""

  def _open_socket(self, prefix: str):
    with _WpaCtrlBase._counter_lock:
      _WpaCtrlBase._counter += 1
      idx = _WpaCtrlBase._counter
    self._local_path = f"/tmp/{prefix}_{os.getpid()}_{idx}"
    try:
      os.unlink(self._local_path)
    except OSError:
      pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
      sock.bind(self._local_path)
      sock.connect(self._ctrl_path)
    except Exception:
      sock.close()
      try:
        os.unlink(self._local_path)
      except OSError:
        pass
      self._local_path = ""
      raise
    self._sock = sock

  def _ensure_sock(self) -> socket.socket:
    if self._sock is None:
      raise RuntimeError("not opened")
    return self._sock

  def close(self):
    if self._sock is not None:
      try:
        self._sock.close()
      except OSError:
        pass
      self._sock = None
    if self._local_path:
      try:
        os.unlink(self._local_path)
      except OSError:
        pass
      self._local_path = ""

  def interrupt(self):
    """Wake a thread blocked on this socket without racing normal cleanup."""
    sock = self._sock
    if sock is not None:
      try:
        sock.shutdown(socket.SHUT_RDWR)
      except OSError:
        pass

  def __enter__(self):
    self.open()
    return self

  def __exit__(self, *_):
    self.close()

  def __del__(self):
    self.close()


class WpaCtrl(_WpaCtrlBase):
  """Synchronous wpa_supplicant control socket command client."""

  def __init__(self, ctrl_path: str = WPA_CTRL_PATH):
    super().__init__(ctrl_path)
    self._request_lock = threading.Lock()

  def open(self):
    self._open_socket("wpa_ctrl")
    self._sock.settimeout(10)

  def request(self, cmd: str) -> str:
    """Send command, return response string."""
    with self._request_lock:
      sock = self._ensure_sock()
      sock.send(cmd.encode())
      return sock.recv(RECV_BUF_SIZE).decode("utf-8", "replace")

  def close(self):
    # Let in-flight requests finish before closing the socket
    with self._request_lock:
      super().close()


class WpaCtrlMonitor(_WpaCtrlBase):
  """Async event stream from wpa_supplicant (ATTACH/DETACH protocol)."""

  def open(self):
    self._open_socket("wpa_mon")
    self._sock.settimeout(10)
    resp = self._raw_request("ATTACH")
    if not resp.startswith("OK"):
      self.close()
      raise RuntimeError(f"ATTACH failed: {resp}")

  def _raw_request(self, cmd: str) -> str:
    sock = self._ensure_sock()
    sock.send(cmd.encode())
    return sock.recv(RECV_BUF_SIZE).decode("utf-8", "replace")

  def pending(self, timeout: float = 0) -> bool:
    if self._sock is None:
      return False
    r, _, _ = select.select([self._sock], [], [], timeout)
    return len(r) > 0

  def recv(self, timeout: float = 1.0) -> str | None:
    if self._sock is None:
      return None
    r, _, _ = select.select([self._sock], [], [], timeout)
    if not r:
      return None
    data = self._sock.recv(RECV_BUF_SIZE).decode("utf-8", "replace")
    # Strip priority prefix like <3>
    if data.startswith("<") and ">" in data[:4]:
      data = data[data.index(">") + 1:]
    return data

  def close(self):
    if self._sock is not None:
      try:
        self._raw_request("DETACH")
      except (OSError, RuntimeError):
        pass
    super().close()


# Keep the public name while sharing one decoder with hardwared.
decode_ssid = decode_wpa_ssid


def parse_scan_results(raw: str) -> list[ScanResult]:
  """Parse wpa_supplicant SCAN_RESULTS output (tab-separated, first line is header)."""
  results = []
  # Preserve legal trailing spaces in the final SSID
  lines = raw.splitlines()
  if len(lines) < 2:
    return results
  for line in lines[1:]:
    parts = line.split("\t")
    if len(parts) < 4:
      continue
    try:
      results.append(ScanResult(
        bssid=parts[0],
        freq=int(parts[1]),
        signal=int(parts[2]),
        flags=parts[3],
        ssid=decode_ssid(parts[4]) if len(parts) > 4 else "",
      ))
    except (ValueError, IndexError):
      continue
  return results


def flags_to_security_type(flags: str) -> SecurityType:
  """Convert wpa_supplicant flags string to SecurityType.

  Examples: [WPA2-PSK-CCMP][WPA-PSK-CCMP], [ESS], [WPA2-PSK-CCMP+TKIP]
  """
  flags_upper = flags.upper()
  flag_groups = re.findall(r"\[([^\]]+)\]", flags_upper)

  # WEP → unsupported
  if "WEP" in flags_upper:
    return SecurityType.UNSUPPORTED

  # Transitional PSK+SAE networks remain usable through WPA-PSK
  if any(re.search(r"(?:^|\+)(?:(?:WPA2|RSN|WPA)-)?PSK(?!-SHA256)(?:[-+]|$)", group) for group in flag_groups):
    return SecurityType.WPA
  # Enterprise / 802.1X without a usable PSK suite → unsupported
  if "EAP" in flags_upper or "802.1X" in flags_upper:
    return SecurityType.UNSUPPORTED
  # SAE-only is unsupported by the current AGNOS stack
  if "SAE" in flags_upper:
    return SecurityType.UNSUPPORTED
  # Secured non-PSK modes must not fall through as open
  if any(mode in flags_upper for mode in ("OWE", "DPP", "OSEN", "FILS")):  # codespell:ignore fils
    return SecurityType.UNSUPPORTED

  # No security flags → open
  if "WPA" not in flags_upper and "RSN" not in flags_upper:
    return SecurityType.OPEN

  return SecurityType.UNSUPPORTED


def parse_status(raw: str) -> dict[str, str]:
  """Parse wpa_supplicant STATUS output (key=value lines). `ssid` is decoded."""
  result = {}
  for line in raw.strip().split("\n"):
    if "=" in line:
      key, _, value = line.partition("=")
      if key == "ssid":
        value = decode_ssid(value)
      result[key] = value
  return result


def dbm_to_percent(dbm: int) -> int:
  """Convert dBm to percentage [0, 100], matching NetworkManager's scale."""
  v = abs(max(-100, min(-40, dbm)) + 40)
  return 100 - (100 * v) // 60


TEMP_DISABLED_SSID_RE = re.compile(r'\bssid="((?:\\.|[^"])*)"')
EVENT_NETWORK_ID_RE = re.compile(r"\bid=(\d+)\b")
WPA_STOP_ATTEMPTS = 60
WPA_STOP_POLL_INTERVAL_SECONDS = 0.05


def normalize_ssid(ssid: str) -> str:
  display_ssid = ssid.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
  return display_ssid.replace("\u2019", "'")  # for iPhone hotspots


def parse_event_ssid(event: str) -> str | None:
  """Extract ssid="…" from a wpa_supplicant control event (printf_encode'd), or None."""
  match = TEMP_DISABLED_SSID_RE.search(event)
  if match is None:
    return None
  return decode_ssid(match.group(1))


def parse_event_network_id(event: str) -> str | None:
  """Extract a numeric network ID from a wpa_supplicant control event, or None."""
  match = EVENT_NETWORK_ID_RE.search(event)
  return match.group(1) if match is not None else None


def _owned_wpa_pid(conf: str) -> int | None:
  try:
    pid = int(Path(WPA_PID_FILE).read_text().strip())
    if pid <= 1:
      return None
    args = [os.fsdecode(arg) for arg in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if arg]
  except (OSError, ValueError):
    return None

  def flag_value(flag: str) -> str | None:
    try:
      return args[args.index(flag) + 1]
    except (ValueError, IndexError):
      return None

  if (
    not args
    or os.path.basename(args[0]) != "wpa_supplicant"
    or flag_value("-i") != "wlan0"
    or flag_value("-c") != conf
    or flag_value("-P") != WPA_PID_FILE
  ):
    return None
  return pid


def wpa_supplicant_running(conf: str) -> bool:
  return _owned_wpa_pid(conf) is not None


def _process_start_time(pid: int) -> str | None:
  try:
    stat = Path(f"/proc/{pid}/stat").read_text()
  except OSError:
    return None
  end_comm = stat.rfind(")")
  fields = stat[end_comm + 2:].split() if end_comm != -1 else []
  return fields[19] if len(fields) > 19 else None


def _wait_for_process_exit(pid: int, start_time: str) -> bool:
  for _ in range(WPA_STOP_ATTEMPTS):
    current_start_time = _process_start_time(pid)
    if current_start_time is None:
      return True
    if current_start_time != start_time:
      raise RuntimeError(f"wpa_supplicant PID {pid} was reused before teardown completed")
    time.sleep(WPA_STOP_POLL_INTERVAL_SECONDS)
  return False


def prepare_wpa_runtime() -> None:
  subprocess.run(["sudo", "install", "-d", "-o", "root", "-g", "netdev", "-m", "775", WPA_CTRL_DIR], check=True)
  subprocess.run(["sudo", "rm", "-f", WPA_PID_FILE, WPA_CTRL_PATH], check=False)


def stop_wpa_supplicant(conf: str) -> None:
  pid = _owned_wpa_pid(conf)
  if pid is None:
    return
  start_time = _process_start_time(pid)
  if start_time is None:
    if os.path.exists(f"/proc/{pid}"):
      raise RuntimeError(f"failed to capture wpa_supplicant PID {pid} identity")
  else:
    subprocess.run(["sudo", "kill", "-TERM", "--", str(pid)], check=False)
    if not _wait_for_process_exit(pid, start_time):
      subprocess.run(["sudo", "kill", "-KILL", "--", str(pid)], check=False)
      if not _wait_for_process_exit(pid, start_time):
        raise RuntimeError(f"owned wpa_supplicant PID {pid} did not exit")
  subprocess.run(["sudo", "rm", "-f", WPA_PID_FILE, WPA_CTRL_PATH], check=True)


def tethering_dnsmasq_running() -> bool:
  return subprocess.run(["pgrep", "-f", TETHERING_DNSMASQ_PATTERN], capture_output=True).returncode == 0


def stop_tethering_dnsmasq() -> None:
  """Stop only the dnsmasq instance started for openpilot tethering."""
  subprocess.run(["sudo", "pkill", "-f", TETHERING_DNSMASQ_PATTERN], check=False)


def sanitize_for_conf(value: str) -> str:
  """Escape characters that could break wpa_supplicant.conf quoting."""
  return value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '').replace('\r', '')


def format_ssid_value(ssid: str) -> str:
  """Render an SSID as hexadecimal bytes for lossless wpa_supplicant parsing."""
  return ssid.encode("utf-8", errors="surrogateescape").hex()


def is_valid_ssid(ssid: str) -> bool:
  try:
    return 0 < len(ssid.encode("utf-8", errors="surrogateescape")) <= IEEE80211_MAX_SSID_BYTES
  except UnicodeEncodeError:
    return False


def _is_raw_psk(psk: str) -> bool:
  """True if psk is a pre-hashed 64-hex WPA PSK. Quoted 64-char values fail as
  too-long passphrases, so raw PSKs must be passed unquoted."""
  return len(psk) == 64 and all(c in "0123456789abcdefABCDEF" for c in psk)


def is_valid_psk(psk: str) -> bool:
  try:
    if any(unicodedata.category(char) == "Cc" for char in psk):
      return False
    return 8 <= len(psk.encode("utf-8")) <= 63 or _is_raw_psk(psk)
  except UnicodeEncodeError:
    return False


def format_psk_value(psk: str) -> str:
  """Render a psk value for wpa_supplicant: raw 64-hex unquoted, else quoted."""
  if _is_raw_psk(psk):
    return psk
  return f'"{sanitize_for_conf(psk)}"'


def generate_wpa_conf(store, path: str = WPA_SUPPLICANT_CONF):
  """Write wpa_supplicant.conf from a NetworkStore (STA networks only)."""
  lines = [
    WPA_CTRL_INTERFACE,
    "update_config=0",
    "p2p_disabled=1",
    "",
  ]

  for ssid, entry in store.get_profiles():
    psk = entry.get("psk", "")
    security = entry.get("security", SecurityType.WPA if psk else SecurityType.OPEN)
    hidden = entry.get("hidden", False)
    priority = entry.get("priority", 0)
    bssid = entry.get("bssid", "")
    ssid_value = format_ssid_value(ssid)
    if not ssid_value:
      continue
    lines.append("network={")
    lines.append(f"  ssid={ssid_value}")
    if security == SecurityType.WPA:
      lines.append(f'  psk={format_psk_value(psk)}')
      lines.append("  key_mgmt=WPA-PSK")
    else:
      lines.append("  key_mgmt=NONE")
    if hidden:
      lines.append("  scan_ssid=1")
    if bssid:
      lines.append(f"  bssid={bssid}")
    if profile_uuid := entry.get("uuid"):
      lines.append(f'  id_str="{sanitize_for_conf(profile_uuid)}"')
    lines.append(f"  priority={priority}")
    lines.append("}")
    lines.append("")

  with atomic_write(path, overwrite=True) as f:
    f.write("\n".join(lines))


def try_attach_ctrl() -> WpaCtrl | None:
  """Pure attach to a running wpa_supplicant ctrl socket. Never spawns, never kills."""
  try:
    ctrl = WpaCtrl()
    ctrl.open()
    return ctrl
  except OSError:
    return None


def _unmanage_wlan0() -> bool:
  """Tell NetworkManager to release wlan0 when it is present.

  An image without NetworkManager has no nmcli and wlan0 is already available,
  so skip the compatibility handoff and continue bringup.
  """
  nmcli = shutil.which("nmcli")
  if nmcli is None:
    cloudlog.info("nmcli not found; assuming NetworkManager is absent")
    return True
  result = subprocess.run(["sudo", nmcli, "dev", "set", "wlan0", "managed", "no"], capture_output=True)
  cloudlog.info(f"nmcli dev set wlan0 managed no: rc={result.returncode}")
  return result.returncode == 0


def ensure_wpa_supplicant(should_exit: Callable[[], bool], station_reconfigured: Callable[[str], None] | None = None,
                          on_abandoned_ap: Callable[[], None] | None = None) -> WpaCtrl | None:
  """Attach to a wpa_supplicant we own, or spawn one. Never attach to NM's daemon.
  Returns the attached WpaCtrl, or None if ownership cannot be acquired."""
  # Wait for wlan0 without allowing teardown to mutate it afterward
  while not os.path.exists("/sys/class/net/wlan0"):
    if should_exit():
      return None
    time.sleep(0.5)

  # Retry attaching to an adopted AP before tearing it down
  if wpa_supplicant_running(WPA_AP_CONF):
    for _ in range(3):
      if should_exit():
        return None
      ctrl = try_attach_ctrl()
      if ctrl is not None:
        return ctrl
      time.sleep(0.5)
    # Replace an owned AP whose control socket remains unreachable
    cloudlog.warning("AP daemon present but ctrl attach failed; killing it so STA spawn can recover")
    stop_wpa_supplicant(WPA_AP_CONF)
    if on_abandoned_ap is not None:
      try:
        on_abandoned_ap()
      except Exception:
        cloudlog.exception("Failed to clean up abandoned AP services")

  # Reuse our station daemon without disturbing NetworkManager
  if wpa_supplicant_running(WPA_SUPPLICANT_CONF):
    if should_exit():
      return None
    ctrl = try_attach_ctrl()
    if ctrl is not None:
      try:
        status = parse_status(ctrl.request("STATUS"))
        station_ssid = status.get("ssid") if status.get("wpa_state") == "COMPLETED" and status.get("mode") == "station" else None
        response = ctrl.request("RECONFIGURE").strip()
        if response.startswith("OK"):
          if station_ssid is not None and station_reconfigured is not None:
            station_reconfigured(station_ssid)
          return ctrl
        cloudlog.warning(f"Station configuration reconciliation failed: {response}")
      except Exception:
        cloudlog.exception("Failed to reconcile running station configuration")
      ctrl.close()

  # Stop before mutating network state
  if should_exit():
    return None

  if not _unmanage_wlan0():
    cloudlog.warning("NetworkManager handoff failed; deferring station bringup")
    return None

  stop_wpa_supplicant(WPA_SUPPLICANT_CONF)
  prepare_wpa_runtime()
  stop_tethering_dnsmasq()
  subprocess.run(["sudo", "ip", "addr", "flush", "dev", "wlan0"], check=False)
  time.sleep(0.5)

  subprocess.run([
    "sudo", "wpa_supplicant", "-B", "-i", "wlan0",
    "-c", WPA_SUPPLICANT_CONF, "-P", WPA_PID_FILE, "-D", "nl80211",
  ], check=False)

  # Never attach to a daemon not using our config
  for _ in range(30):
    if should_exit():
      return None
    if wpa_supplicant_running(WPA_SUPPLICANT_CONF):
      ctrl = try_attach_ctrl()
      if ctrl is not None:
        try:
          ctrl.request("ENABLE_NETWORK all")
        except Exception:
          pass
        return ctrl
    time.sleep(1)
  cloudlog.error("wpa_supplicant did not start after 30 attempts")
  return None
