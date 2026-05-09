"""Persistent storage for saved WiFi networks, backed by .nmconnection files."""
import configparser
import os
import subprocess
import tempfile
import threading
import uuid
from enum import IntEnum

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import sudo_read


NM_CONNECTIONS_DIR = "/data/etc/NetworkManager/system-connections"

# Only key-mgmt values we can actually drive via wpa_supplicant. Anything else
# (wpa-eap, sae, ieee8021x, ...) gets skipped on load — coercing those to
# psk="" would render as key_mgmt=NONE in wpa_supplicant.conf, silently turning
# a secure profile into an open one for the same SSID and inviting open spoofing.
_SUPPORTED_KEY_MGMT = {"wpa-psk", "none"}


class MeteredType(IntEnum):
  UNKNOWN = 0
  YES = 1
  NO = 2


def _canonical_filename(file_uuid: str, ssid: str) -> str:
  """`<uuid>-<ssid>.nmconnection` matches netplan's runtime keyfile naming. UUID is the
  stable handle; the SSID suffix is purely cosmetic, so it gets sanitized lossily."""
  ssid_safe = ssid.replace("/", "_").replace("\0", "_")
  return f"{file_uuid}-{ssid_safe}.nmconnection"


class NetworkStore:
  """Persistent storage for saved WiFi networks using .nmconnection files."""

  def __init__(self, directory: str = NM_CONNECTIONS_DIR):
    self._directory = directory
    self._lock = threading.Lock()
    self._networks: dict[str, dict] = {}
    self._load()

  def _load(self):
    self._networks = {}
    try:
      filenames = os.listdir(self._directory)
    except OSError:
      return
    for fname in filenames:
      if not fname.endswith(".nmconnection"):
        continue
      fpath = os.path.join(self._directory, fname)
      try:
        cp = configparser.ConfigParser(interpolation=None)
        raw = sudo_read(fpath)
        if raw:
          cp.read_string(raw)
        else:
          cp.read(fpath)

        # NM keyfile schema documents [802-11-wireless]/[802-11-wireless-security]
        # as canonical with [wifi]/[wifi-security] as aliases. AGNOS NM 1.46 writes
        # the alias form for every wifi profile it generates; only hand-written or
        # imported-from-old-NM keyfiles use canonical names, and none ship with
        # AGNOS. We deliberately accept only the alias spelling so a keyfile we
        # can't round-trip (we only write [wifi] back) doesn't get half-migrated.
        if not cp.has_section("wifi"):
          continue
        ssid = cp.get("wifi", "ssid", fallback="")
        mode = cp.get("wifi", "mode", fallback="infrastructure")
        if not ssid or mode == "ap":
          continue

        # An open profile has no [wifi-security] section. A secure profile with a
        # key-mgmt we can't reproduce (wpa-eap, sae, ...) must be skipped entirely.
        psk = ""
        if cp.has_section("wifi-security"):
          key_mgmt = cp.get("wifi-security", "key-mgmt", fallback="").lower()
          if key_mgmt not in _SUPPORTED_KEY_MGMT:
            cloudlog.warning(f"NetworkStore: skipping {ssid!r} with unsupported key-mgmt={key_mgmt!r}")
            continue
          # NM stores WEP as `key-mgmt=none` plus `wep-key*`/`wep-key-type`/`auth-alg=shared`.
          # Loading those as open (psk="") would let _generate_wpa_conf demote the secured
          # SSID to key_mgmt=NONE — auto-association to an open spoof of the same SSID.
          wep_keys = ("wep-key0", "wep-key1", "wep-key2", "wep-key3", "wep-key-type", "auth-alg")
          if key_mgmt == "none" and any(cp.has_option("wifi-security", k) for k in wep_keys):
            cloudlog.warning(f"NetworkStore: skipping {ssid!r} (WEP profile, unsupported)")
            continue
          psk = cp.get("wifi-security", "psk", fallback="")
          # NM agent-managed secrets (psk-flags=1) live outside the keyfile. We can't
          # drive them via wpa_supplicant, and loading with psk="" would render as
          # key_mgmt=NONE — a secure profile silently demoted to open, inviting spoofs.
          if key_mgmt == "wpa-psk" and not psk:
            cloudlog.warning(f"NetworkStore: skipping {ssid!r} (wpa-psk with no inline secret)")
            continue

        # connection.autoconnect=false is user/provisioning intent — don't load it
        # only to have ENABLE_NETWORK all silently re-arm the auto-join.
        if not cp.getboolean("connection", "autoconnect", fallback=True):
          cloudlog.warning(f"NetworkStore: skipping {ssid!r} (connection.autoconnect=false)")
          continue

        # getint/getboolean can raise ValueError on malformed values; skip the bad profile.
        self._networks[ssid] = {
          "psk": psk,
          "metered": cp.getint("connection", "metered", fallback=0),
          "hidden": cp.getboolean("wifi", "hidden", fallback=False),
          "uuid": cp.get("connection", "uuid", fallback=""),
          # Remember the on-disk filename so save/remove stay consistent with legacy files.
          "_filename": fname,
        }
      except (configparser.Error, ValueError):
        continue

  def _render_nmconnection(self, ssid: str, entry: dict) -> tuple[str, dict]:
    file_uuid = entry.get("uuid") or str(uuid.uuid5(uuid.NAMESPACE_DNS, ssid))
    entry = dict(entry)
    entry["uuid"] = file_uuid

    canonical_fname = _canonical_filename(file_uuid, ssid)
    old_fname = entry.get("_filename")
    entry["_filename"] = canonical_fname

    cp = configparser.ConfigParser(interpolation=None)
    cp["connection"] = {
      "id": ssid,
      "uuid": file_uuid,
      "type": "wifi",
      "metered": str(entry.get("metered", 0)),
    }
    cp["wifi"] = {
      "ssid": ssid,
      "mode": "infrastructure",
      "hidden": str(entry.get("hidden", False)).lower(),
    }

    psk = entry.get("psk", "")
    if psk:
      cp["wifi-security"] = {
        "key-mgmt": "wpa-psk",
        "psk": psk,
      }

    cp["ipv4"] = {"method": "auto"}
    cp["ipv6"] = {"method": "auto"}

    with tempfile.NamedTemporaryFile(mode="w", dir="/tmp", delete=False) as f:
      cp.write(f)
      temp_path = f.name

    try:
      os.chmod(temp_path, 0o600)
      subprocess.run(["sudo", "install", "-d", "-m", "755", self._directory], check=True)
      subprocess.run(["sudo", "install", "-o", "root", "-g", "root", "-m", "600",
                      temp_path, os.path.join(self._directory, canonical_fname)], check=True)
    finally:
      try:
        os.unlink(temp_path)
      except FileNotFoundError:
        pass

    # Migrate from any prior naming (legacy percent-encoded, or earlier UUID with stale ssid suffix).
    if old_fname and old_fname != canonical_fname:
      old_path = os.path.join(self._directory, old_fname)
      result = subprocess.run(["sudo", "rm", "-f", old_path], check=False)
      # If cleanup fails (FS read-only, etc.) both files survive. _load's listdir
      # order would pick winners non-deterministically. Make both files hold the
      # same content so the duplicate is harmless; pin _filename to the legacy
      # name so future updates land there too and we keep retrying the cleanup.
      if result.returncode != 0:
        cloudlog.warning(f"NetworkStore: cleanup of legacy {old_fname} failed; mirroring content to keep both files in sync")
        try:
          subprocess.run(["sudo", "install", "-o", "root", "-g", "root", "-m", "600",
                          os.path.join(self._directory, canonical_fname), old_path], check=True)
        except Exception:
          cloudlog.exception("NetworkStore: failed to mirror migrated keyfile back to legacy path")
        entry["_filename"] = old_fname

    return file_uuid, entry

  def get_all(self) -> dict[str, dict]:
    with self._lock:
      return {k: dict(v) for k, v in self._networks.items()}

  def get(self, ssid: str) -> dict | None:
    with self._lock:
      entry = self._networks.get(ssid)
      return dict(entry) if entry else None

  def save_network(self, ssid: str, psk: str | None = None, metered: int | None = None, hidden: bool | None = None):
    # ConfigParser.write/get strips boundary whitespace on round-trip, so an SSID
    # or PSK with leading/trailing spaces would silently change on next restart —
    # the in-memory connection works for the session, then breaks after reload.
    # Refuse rather than corrupt; the user gets a clear failure to investigate.
    if ssid != ssid.strip():
      cloudlog.warning(f"NetworkStore: refusing to save SSID with boundary whitespace: {ssid!r}")
      return
    if psk is not None and psk != psk.strip():
      cloudlog.warning(f"NetworkStore: refusing to save PSK with boundary whitespace for {ssid!r}")
      return
    with self._lock:
      existing = dict(self._networks.get(ssid, {}))
      if psk is not None:
        existing["psk"] = psk
      elif "psk" not in existing:
        existing["psk"] = ""
      if metered is not None:
        existing["metered"] = metered
      elif "metered" not in existing:
        existing["metered"] = 0
      if hidden is not None:
        existing["hidden"] = hidden
      elif "hidden" not in existing:
        existing["hidden"] = False
      file_uuid, updated = self._render_nmconnection(ssid, existing)
      updated["uuid"] = file_uuid
      self._networks[ssid] = updated

  def remove(self, ssid: str) -> bool:
    with self._lock:
      entry = self._networks.get(ssid)
      if entry is None:
        return False
      # Migrated profiles can leave duplicate keyfiles around (legacy + canonical,
      # or the failed-cleanup mirror path). Removing only the tracked file leaves
      # the others on disk and _load would silently restore the network. Always
      # include the canonical and tracked paths plus any other files matching
      # this SSID on disk, then `rm -f` them all.
      paths: set[str] = set()
      paths.add(os.path.join(self._directory, _canonical_filename(entry.get("uuid", ""), ssid)))
      tracked = entry.get("_filename")
      if tracked:
        paths.add(os.path.join(self._directory, tracked))
      paths.update(self._find_keyfiles_for(ssid))
      for p in paths:
        result = subprocess.run(["sudo", "rm", "-f", p], check=False)
        # `rm -f` returns 0 even if the file is absent; non-zero is a real failure
        # (FS read-only, sudo broken). Leave the in-memory entry alone — _load on
        # next start would restore the file and silently re-enable auto-connect.
        if result.returncode != 0:
          cloudlog.warning(f"NetworkStore: failed to remove {p} (rc={result.returncode})")
          return False
      del self._networks[ssid]
      return True

  def _find_keyfiles_for(self, ssid: str) -> list[str]:
    """Return all .nmconnection paths in our directory whose [wifi] ssid == ssid."""
    matches = []
    try:
      filenames = os.listdir(self._directory)
    except OSError:
      return matches
    for fname in filenames:
      if not fname.endswith(".nmconnection"):
        continue
      fpath = os.path.join(self._directory, fname)
      raw = sudo_read(fpath)
      if not raw:
        continue
      cp = configparser.ConfigParser(interpolation=None)
      try:
        cp.read_string(raw)
      except configparser.Error:
        continue
      if cp.get("wifi", "ssid", fallback="") == ssid:
        matches.append(fpath)
    return matches

  def set_metered(self, ssid: str, metered: int):
    with self._lock:
      if ssid in self._networks:
        updated = dict(self._networks[ssid])
        updated["metered"] = metered
        file_uuid, updated = self._render_nmconnection(ssid, updated)
        updated["uuid"] = file_uuid
        self._networks[ssid] = updated

  def get_metered(self, ssid: str) -> MeteredType:
    with self._lock:
      entry = self._networks.get(ssid)
      if entry:
        m = entry.get("metered", 0)
        if m == MeteredType.YES:
          return MeteredType.YES
        elif m == MeteredType.NO:
          return MeteredType.NO
    return MeteredType.UNKNOWN

  def contains(self, ssid: str) -> bool:
    with self._lock:
      return ssid in self._networks

  def saved_ssids(self) -> set[str]:
    with self._lock:
      return set(self._networks.keys())
