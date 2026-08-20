import os
from pathlib import Path
import subprocess
import threading

from openpilot.common.swaglog import cloudlog

DHCP_SCRIPT = os.path.join(os.path.dirname(__file__), "udhcpc.script")
DHCP_DEFAULT_SCRIPT = "/etc/udhcpc/default.script"
DHCP_RUNTIME_DIR = "/run/openpilot-wifi"


class DhcpClient:
  """Manage udhcpc for DHCP on wlan0."""

  # Match udhcpc's -T retry timeout
  DISCOVER_TIMEOUT_SECONDS = 3
  DISCOVER_ATTEMPTS = 5

  def __init__(self, iface: str = "wlan0"):
    self._iface = iface
    self._pid_file = os.path.join(DHCP_RUNTIME_DIR, f"udhcpc-{iface}.pid")
    self._proc: subprocess.Popen | None = None
    self._adopted = False
    self._client_thread: threading.Thread | None = None
    self._client_stop = threading.Event()
    self._ipv6_enabled: bool | None = None

  def _start_client_thread(self):
    self._client_stop.clear()
    self._client_thread = threading.Thread(target=self._monitor_client, daemon=True)
    self._client_thread.start()

  def _owned_pid(self) -> int | None:
    try:
      pid = int(Path(self._pid_file).read_text().strip())
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
      or os.path.basename(args[0]) != "udhcpc"
      or flag_value("-i") != self._iface
      or flag_value("-p") != self._pid_file
      or flag_value("-s") != DHCP_SCRIPT
    ):
      return None
    return pid

  def _client_running(self) -> bool:
    if self._proc is not None and self._proc.poll() is None:
      return True
    return self._owned_pid() is not None

  def _prepare_runtime(self):
    subprocess.run(["sudo", "install", "-d", "-o", "root", "-g", "root", "-m", "755", DHCP_RUNTIME_DIR], check=True)
    subprocess.run(["sudo", "rm", "-f", self._pid_file], check=False)

  def _flush_address(self):
    subprocess.run(["sudo", "ip", "-4", "addr", "flush", "dev", self._iface], capture_output=True, check=False)

  def _flush_lease(self):
    subprocess.run(["sudo", "ip", "-4", "route", "flush", "dev", self._iface], capture_output=True, check=False)
    self._flush_address()

  def clear_ipv6_state(self):
    delete_default_route = ["sudo", "ip", "-6", "route", "del", "default", "dev", self._iface]
    for command in (
      ["sudo", "ip", "-6", "addr", "flush", "dev", self._iface, "scope", "global"],
      # Delete router-advertised defaults before flushing routes
      delete_default_route,
      ["sudo", "ip", "-6", "route", "flush", "dev", self._iface],
    ):
      try:
        result = subprocess.run(command, capture_output=True, check=False)
        missing_default_route = (
          command == delete_default_route
          and result.returncode == 2
          and b"No such process" in result.stderr
        )
        if result.returncode != 0 and not missing_default_route:
          cloudlog.warning(f"Failed to clear {self._iface} IPv6 state (rc={result.returncode})")
      except OSError:
        cloudlog.exception(f"Failed to clear {self._iface} IPv6 state")

  def set_ipv6_enabled(self, enabled: bool):
    if self._ipv6_enabled == enabled:
      return
    disabled = "0" if enabled else "1"
    subprocess.run(["sudo", "sysctl", f"net.ipv6.conf.{self._iface}.disable_ipv6={disabled}"], check=True)
    actual = Path(f"/proc/sys/net/ipv6/conf/{self._iface}/disable_ipv6").read_text().strip()
    if actual != disabled:
      raise RuntimeError(f"Failed to set IPv6 enabled={enabled} on {self._iface} (actual={actual!r})")
    self._ipv6_enabled = enabled
    if not enabled:
      self.clear_ipv6_state()

  def _spawn(self) -> bool:
    if not os.access(DHCP_DEFAULT_SCRIPT, os.X_OK):
      cloudlog.error(f"udhcpc default script is not executable: {DHCP_DEFAULT_SCRIPT}")
      return False
    try:
      self._prepare_runtime()
      self._proc = subprocess.Popen(
        ["sudo", "udhcpc", "-i", self._iface, "-f",
         "-t", str(self.DISCOVER_ATTEMPTS), "-T", str(self.DISCOVER_TIMEOUT_SECONDS),
         "-p", self._pid_file, "-s", DHCP_SCRIPT],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
      )
    except Exception:
      self._proc = None
      cloudlog.exception("Failed to start udhcpc")
      return False
    return True

  def _monitor_client(self):
    while not self._client_stop.wait(self.DISCOVER_TIMEOUT_SECONDS):
      if not self._client_running():
        self._flush_lease()
        self._spawn()

  def adopt(self) -> bool:
    if not self._client_running():
      return False
    self._adopted = True
    self._start_client_thread()
    return True

  def start(self):
    self.stop()
    self._spawn()
    self._start_client_thread()

  def stop(self):
    self._client_stop.set()
    if self._client_thread is not None:
      self._client_thread.join(timeout=self.DISCOVER_TIMEOUT_SECONDS)
      self._client_thread = None
    owned_pid = self._owned_pid()
    if self._proc is not None:
      try:
        self._proc.terminate()
        self._proc.wait(timeout=3)
      except Exception:
        try:
          self._proc.kill()
          self._proc.wait()
        except Exception:
          pass
      self._proc = None
    if owned_pid is not None:
      subprocess.run(["sudo", "kill", str(owned_pid)], check=False)
    subprocess.run(["sudo", "rm", "-f", self._pid_file], check=False)
    self._adopted = False
    self._flush_lease()
