import os
import re
import subprocess
import threading

from openpilot.common.swaglog import cloudlog

DHCP_SCRIPT = os.path.join(os.path.dirname(__file__), "udhcpc.script")
DHCP_DEFAULT_SCRIPT = "/etc/udhcpc/default.script"


class DhcpClient:
  """Manage udhcpc for DHCP on wlan0."""

  # Match udhcpc's -T retry timeout
  DISCOVER_TIMEOUT_SECONDS = 3
  DISCOVER_ATTEMPTS = 5

  def __init__(self, iface: str = "wlan0"):
    self._iface = iface
    self._proc: subprocess.Popen | None = None
    self._adopted = False
    self._client_thread: threading.Thread | None = None
    self._client_stop = threading.Event()

  def _start_client_thread(self):
    self._client_stop.clear()
    self._client_thread = threading.Thread(target=self._monitor_client, daemon=True)
    self._client_thread.start()

  def _client_running(self) -> bool:
    if self._proc is not None and self._proc.poll() is None:
      return True
    script = re.escape(DHCP_SCRIPT)
    result = subprocess.run(["pgrep", "-f", f"^udhcpc -i {self._iface}( |$).* -s {script}( |$)"], capture_output=True, check=False)
    return result.returncode == 0

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

  def _spawn(self) -> bool:
    if not os.access(DHCP_DEFAULT_SCRIPT, os.X_OK):
      cloudlog.error(f"udhcpc default script is not executable: {DHCP_DEFAULT_SCRIPT}")
      return False
    try:
      self._proc = subprocess.Popen(
        ["sudo", "udhcpc", "-i", self._iface, "-f",
         "-t", str(self.DISCOVER_ATTEMPTS), "-T", str(self.DISCOVER_TIMEOUT_SECONDS),
         "-s", DHCP_SCRIPT],
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
    self._adopted = False
    # Kill orphaned udhcpc children before flushing lease state
    subprocess.run(["sudo", "pkill", "-f", f"^udhcpc -i {self._iface}( |$)"], check=False)
    self._flush_lease()
