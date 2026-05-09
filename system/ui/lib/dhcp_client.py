"""udhcpc lifecycle for a single interface, with a one-shot default-route metric fixup."""
import subprocess
import threading
import time

from openpilot.common.swaglog import cloudlog


class DhcpClient:
  """Manage udhcpc for DHCP on wlan0."""

  # Above NM's eth0 default (metric 100) so ETH keeps priority; matches NM's wifi metric.
  DEFAULT_ROUTE_METRIC = 600

  def __init__(self, iface: str = "wlan0"):
    self._iface = iface
    self._proc: subprocess.Popen | None = None
    self._metric_thread: threading.Thread | None = None
    self._metric_stop = threading.Event()

  def start(self):
    self.stop()
    # If a previous UI/manager process crashed, its sudo udhcpc -f survives as
    # an orphan and self.stop() above can't reach it (we only track our own Popen).
    # Without this, two udhcpc instances would race lease renewals on the same
    # interface and the orphan can re-add the address after a managed flush.
    subprocess.run(["sudo", "pkill", "-f", f"udhcpc.*-i {self._iface}"], check=False)
    self._metric_stop.clear()
    try:
      self._proc = subprocess.Popen(
        ["sudo", "udhcpc", "-i", self._iface, "-f", "-t", "5", "-T", "3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
      )
    except Exception:
      cloudlog.exception("Failed to start udhcpc")
      return
    self._metric_thread = threading.Thread(target=self._fix_default_route_metric, daemon=True)
    self._metric_thread.start()

  def _fix_default_route_metric(self):
    """Replace udhcpc's metric-0 default route with metric 600.
    busybox udhcpc can't set a bind-time metric; without this, wlan0 silently beats
    eth0 on every DHCP bind. Same-router renewals skip the re-install, so a one-shot
    bump survives the lease."""
    deadline = time.monotonic() + 10.0
    while not self._metric_stop.is_set() and time.monotonic() < deadline:
      try:
        out = subprocess.check_output(
          ["ip", "-4", "route", "show", "default", "dev", self._iface],
          text=True, timeout=2,
        ).strip()
      except Exception:
        # Transient: udhcpc may not have installed the route yet, or the iface is briefly down.
        # Fall through to the throttled wait and retry until the deadline.
        cloudlog.exception("Failed to query wlan0 default route")
        out = ""

      for line in out.splitlines():
        parts = line.split()
        if "via" not in parts:
          continue
        try:
          gw = parts[parts.index("via") + 1]
        except IndexError:
          continue
        if "metric" in parts:
          try:
            if int(parts[parts.index("metric") + 1]) == self.DEFAULT_ROUTE_METRIC:
              return
          except (IndexError, ValueError):
            pass
        subprocess.run(
          ["sudo", "ip", "-4", "route", "flush", "exact", "0.0.0.0/0", "dev", self._iface],
          check=False,
        )
        subprocess.run(
          ["sudo", "ip", "-4", "route", "add", "default", "via", gw,
           "dev", self._iface, "metric", str(self.DEFAULT_ROUTE_METRIC)],
          check=False,
        )
        return

      self._metric_stop.wait(0.2)

  def stop(self):
    self._metric_stop.set()
    if self._metric_thread is not None:
      self._metric_thread.join(timeout=2)
      self._metric_thread = None
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
      subprocess.run(["sudo", "ip", "addr", "flush", "dev", self._iface], capture_output=True, check=False)
