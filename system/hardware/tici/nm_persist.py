"""Persist NM/netplan-emitted runtime wifi keyfiles to /data so saved networks
survive after NetworkManager and netplan are removed in a future OS release.

netplan generates *.nmconnection files into /run/NetworkManager/system-connections/
at boot, prefixed `netplan-NM-<UUID>...`. /run is tmpfs — the moment netplan stops
running, those connections vanish unless we've already persisted them. This module
copies each wifi keyfile to /data/etc/NetworkManager/system-connections/ under a
`<UUID>-<ssid>.nmconnection` name and deletes the upstream YAML in
/data/etc/netplan/, making the keyfile canonical.

Wifi only: eth0 and other types stay where they are. The eventual no-NM OS upgrade
ships its own eth0 config (systemd-networkd on AGNOS, runit service on vamOS); we
don't need to migrate it.

Idempotent: safe to run on every boot. Self-disabling: once netplan is gone, /run
contains no `netplan-NM-*` entries and this is a no-op.
"""
import configparser
import os
import re
import subprocess
import tempfile

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import sudo_read

RUN_DIR = "/run/NetworkManager/system-connections"
DATA_DIR = "/data/etc/NetworkManager/system-connections"
NETPLAN_DIR = "/data/etc/netplan"

# netplan-NM-<UUID>(-<ssid suffix>).nmconnection, with UUID in standard 8-4-4-4-12 form.
_NETPLAN_KEYFILE_RE = re.compile(
  r"^netplan-NM-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:-.*)?\.nmconnection$"
)


def _sanitize_ssid_for_filename(ssid: str) -> str:
  # SSID suffix is purely cosmetic; UUID is the stable handle. Replace path-tricky chars.
  return ssid.replace("/", "_").replace("\0", "_")


def _sudo_install(content: str, dest_path: str) -> None:
  with tempfile.NamedTemporaryFile(mode="w", dir="/tmp", delete=False) as f:
    f.write(content)
    tmp = f.name
  try:
    os.chmod(tmp, 0o600)
    dest_dir = os.path.dirname(dest_path)
    subprocess.run(["sudo", "install", "-d", "-m", "755", dest_dir], check=True)
    subprocess.run(["sudo", "install", "-o", "root", "-g", "root", "-m", "600",
                    tmp, dest_path], check=True)
  finally:
    try:
      os.unlink(tmp)
    except FileNotFoundError:
      pass


def _delete_netplan_yaml(file_uuid: str, netplan_dir: str) -> None:
  try:
    yamls = os.listdir(netplan_dir)
  except OSError:
    return
  suffix = f"NM-{file_uuid}.yaml"
  for yfname in yamls:
    if yfname.endswith(suffix):
      ypath = os.path.join(netplan_dir, yfname)
      subprocess.run(["sudo", "rm", "-f", ypath], check=False)


def persist_connections(run_dir: str = RUN_DIR, data_dir: str = DATA_DIR, netplan_dir: str = NETPLAN_DIR) -> None:
  """Copy netplan-emitted wifi keyfiles from run_dir to data_dir and delete the
  netplan YAMLs that produced them. Wifi only. Idempotent."""
  try:
    filenames = os.listdir(run_dir)
  except OSError:
    return  # /run/NetworkManager/system-connections doesn't exist (no NM running)

  for fname in filenames:
    m = _NETPLAN_KEYFILE_RE.match(fname)
    if m is None:
      continue
    file_uuid = m.group(1)

    raw = sudo_read(os.path.join(run_dir, fname))
    if not raw:
      continue

    cp = configparser.ConfigParser(interpolation=None)
    try:
      cp.read_string(raw)
    except configparser.Error:
      continue

    if cp.get("connection", "type", fallback="") != "wifi":
      continue

    ssid = cp.get("wifi", "ssid", fallback="")
    mode = cp.get("wifi", "mode", fallback="infrastructure")
    # Hotspot profiles (mode=ap) are owned by openpilot's tethering path, not by
    # NetworkStore. Persisting them here is dead clutter.
    if not ssid or mode == "ap":
      continue

    dest_fname = f"{file_uuid}-{_sanitize_ssid_for_filename(ssid)}.nmconnection"
    dest_path = os.path.join(data_dir, dest_fname)

    # sudo_read strips whitespace; raw came from sudo_read so already stripped.
    if sudo_read(dest_path) == raw:
      _delete_netplan_yaml(file_uuid, netplan_dir)
      continue

    try:
      _sudo_install(raw, dest_path)
    except Exception:
      cloudlog.exception("nm_persist: failed to install %s -> %s", fname, dest_path)
      continue

    _delete_netplan_yaml(file_uuid, netplan_dir)
