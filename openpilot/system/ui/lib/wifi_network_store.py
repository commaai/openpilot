import configparser
import os
import re
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from enum import IntEnum

from openpilot.common.swaglog import cloudlog
from openpilot.common.nm_keyfile import decode_nm_keyfile_ssid, decode_nm_keyfile_string
from openpilot.common.utils import sudo_read
from openpilot.system.ui.lib.wpa_ctrl import SecurityType, is_valid_psk, is_valid_ssid


NM_CONNECTIONS_DIR = "/data/etc/NetworkManager/system-connections"
RUNTIME_CONNECTIONS_DIR = "/run/NetworkManager/system-connections"
NETPLAN_CONNECTIONS_DIR = "/data/etc/netplan"

# Never reinterpret unsupported secured profiles as open
_SUPPORTED_KEY_MGMT = {"wpa-psk", "none"}
_SUPPORTED_CONNECTION_OPTIONS = {
  "id", "uuid", "type", "autoconnect", "autoconnect-priority", "autoconnect-retries", "timestamp", "metered", "interface-name",
}
_SUPPORTED_WIFI_OPTIONS = {"ssid", "mode", "hidden", "bssid"}
_SUPPORTED_SECURITY_OPTIONS = {"key-mgmt", "psk", "psk-flags", "auth-alg"}
_SUPPORTED_IPV4_METHODS = {"auto"}
_SUPPORTED_IPV6_METHODS = {"auto", "ignore"}
_SUPPORTED_IPV4_OPTIONS = {"method", "dns-priority"}
_SUPPORTED_IPV6_OPTIONS = {"method", "addr-gen-mode"}
# Preserve NetworkManager's DNS priority for rollback compatibility
_OPENPILOT_DNS_PRIORITY = "600"
_TRANSACTION_REMNANT_RE = re.compile(r"^(?P<original>.+)\.openpilot-(?:update|forget)-[0-9a-f]{32}$")


class MeteredType(IntEnum):
  UNKNOWN = 0
  YES = 1
  NO = 2


def _canonical_filename(file_uuid: str, ssid: str) -> str:
  """`<uuid>-<ssid>.nmconnection` matches netplan's runtime keyfile naming. UUID is the
  stable handle; the SSID suffix is purely cosmetic, so it gets sanitized lossily."""
  ssid_safe = ssid.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
  ssid_safe = ssid_safe.replace("/", "_").replace("\0", "_")
  return f"{file_uuid}-{ssid_safe}.nmconnection"


def _parse_uuid(value: str) -> str | None:
  try:
    return str(uuid.UUID(value))
  except ValueError:
    return None


def _encode_keyfile_string(value: str) -> str:
  """Encode GLib keyfile string escapes, including boundary spaces."""
  leading_spaces = len(value) - len(value.lstrip(" "))
  trailing_spaces = len(value) - len(value.rstrip(" "))
  encoded = []
  for i, char in enumerate(value):
    if char == "\\":
      encoded.append("\\\\")
    elif char == "\n":
      encoded.append("\\n")
    elif char == "\r":
      encoded.append("\\r")
    elif char == "\t":
      encoded.append("\\t")
    elif char == " " and (i < leading_spaces or i >= len(value) - trailing_spaces):
      encoded.append("\\s")
    else:
      encoded.append(char)
  return "".join(encoded)


def _encode_keyfile_ssid(ssid: str) -> str:
  if not ssid:
    return ""
  return ";".join(str(b) for b in ssid.encode("utf-8", errors="surrogateescape")) + ";"


def _normalize_keyfile_sections(cp: configparser.ConfigParser):
  for alias, canonical in (
    ("wifi", "802-11-wireless"),
    ("wifi-security", "802-11-wireless-security"),
  ):
    if not cp.has_section(alias) and cp.has_section(canonical):
      cp[alias] = dict(cp[canonical])


def _keyfile_section(cp: configparser.ConfigParser, alias: str, canonical: str) -> str | None:
  if cp.has_section(alias):
    return alias
  if cp.has_section(canonical):
    return canonical
  return None


class NetworkStore:
  def __init__(self, directory: str = NM_CONNECTIONS_DIR, runtime_directory: str | None = None, netplan_directory: str | None = None):
    self._directory = directory
    # Import Netplan's runtime-only keyfiles on production stores
    self._runtime_directory = RUNTIME_CONNECTIONS_DIR if directory == NM_CONNECTIONS_DIR else None
    if runtime_directory is not None:
      self._runtime_directory = runtime_directory
    self._netplan_directory = NETPLAN_CONNECTIONS_DIR if directory == NM_CONNECTIONS_DIR else None
    if netplan_directory is not None:
      self._netplan_directory = netplan_directory
    self._lock = threading.Lock()
    self._mutation_lock = threading.Lock()
    self._networks: dict[str, dict] = {}
    self._profiles: dict[str, list[dict]] = {}
    self._recover_transaction_remnants()
    self._load()

  def _recover_transaction_remnants(self):
    directories = dict.fromkeys((
      self._directory,
      self._runtime_directory,
      self._netplan_directory,
    ))
    for directory in directories:
      if directory is None:
        continue
      try:
        filenames = sorted(os.listdir(directory))
      except OSError:
        continue
      for filename in filenames:
        match = _TRANSACTION_REMNANT_RE.fullmatch(filename)
        if match is None:
          continue
        remnant_path = os.path.join(directory, filename)
        original_path = os.path.join(directory, match.group("original"))
        command = ["sudo", "rm", "-f", remnant_path] if os.path.exists(original_path) else [
          "sudo", "mv", "-f", remnant_path, original_path,
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
          cloudlog.warning(f"NetworkStore: failed to recover transaction remnant {remnant_path} (rc={result.returncode})")

  def _load(self):
    self._networks = {}
    self._profiles = {}
    sources = [(self._directory, False)]
    if self._runtime_directory is not None:
      sources.append((self._runtime_directory, True))

    persistent_uuids: dict[str, set[str]] = {}
    for directory, imported in sources:
      try:
        filenames = sorted(os.listdir(directory))
      except OSError:
        continue
      for fname in filenames:
        self._load_keyfile(directory, fname, imported, persistent_uuids)

  def _find_netplan_filename(self, file_uuid: str) -> str | None:
    if self._netplan_directory is None or not file_uuid:
      return None
    expected = f"90-NM-{file_uuid}.yaml"
    if os.path.exists(os.path.join(self._netplan_directory, expected)):
      return expected
    try:
      filenames = sorted(os.listdir(self._netplan_directory))
    except OSError:
      return None
    pattern = re.compile(r"^\s*uuid\s*:\s*['\"]?([^'\"\s#]+)['\"]?\s*(?:#.*)?$", re.MULTILINE)
    yaml_filenames = [fname for fname in filenames if fname.endswith(".yaml")]
    read_failed = False
    for fname in yaml_filenames:
      try:
        raw = sudo_read(os.path.join(self._netplan_directory, fname))
      except OSError:
        read_failed = True
        continue
      if not raw:
        read_failed = True
      elif {_parse_uuid(value) for value in pattern.findall(raw)} == {file_uuid}:
        return fname
    return expected if read_failed else None

  def _load_keyfile(self, directory: str, fname: str, imported: bool, persistent_uuids: dict[str, set[str]]):
    if not fname.endswith(".nmconnection"):
      return
    fpath = os.path.join(directory, fname)
    try:
      cp = configparser.ConfigParser(interpolation=None)
      raw = sudo_read(fpath)
      if raw:
        cp.read_string(raw)
      else:
        cp.read(fpath)

      _normalize_keyfile_sections(cp)
      if not cp.has_section("wifi"):
        return
      ssid = decode_nm_keyfile_ssid(cp.get("wifi", "ssid", fallback=""))
      mode = cp.get("wifi", "mode", fallback="infrastructure")
      if not is_valid_ssid(ssid) or mode != "infrastructure":
        return
      if {key for key, value in cp.items("wifi") if value} - _SUPPORTED_WIFI_OPTIONS:
        cloudlog.warning(f"NetworkStore: skipping {ssid!r} with unsupported Wi-Fi options")
        return
      bssid = cp.get("wifi", "bssid", fallback="")
      if bssid and re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", bssid) is None:
        cloudlog.warning(f"NetworkStore: skipping {ssid!r} with invalid bssid={bssid!r}")
        return
      raw_uuid = cp.get("connection", "uuid", fallback="")
      file_uuid = _parse_uuid(raw_uuid)
      if file_uuid is None:
        cloudlog.warning(f"NetworkStore: skipping {ssid!r} with invalid uuid={raw_uuid!r}")
        return
      connection = dict(cp["connection"])
      connection_type = connection.get("type", "wifi").lower()
      interface_name = connection.get("interface-name", "")
      unsupported_connection_options = {key for key, value in connection.items() if value} - _SUPPORTED_CONNECTION_OPTIONS
      if (connection_type not in ("wifi", "802-11-wireless")
          or interface_name not in ("", "wlan0")
          or cp.getint("connection", "autoconnect-retries", fallback=0) != 0
          or unsupported_connection_options):
        cloudlog.warning(f"NetworkStore: skipping {ssid!r} with unsupported connection constraints")
        return
      # Skip secured profiles that cannot be reproduced safely.
      security = SecurityType.OPEN
      psk = ""
      if cp.has_section("wifi-security"):
        key_mgmt = cp.get("wifi-security", "key-mgmt", fallback="").lower()
        if key_mgmt not in _SUPPORTED_KEY_MGMT:
          cloudlog.warning(f"NetworkStore: skipping {ssid!r} with unsupported key-mgmt={key_mgmt!r}")
          return
        # key-mgmt=none can still represent WEP; never import it as open.
        wep_keys = ("wep-key0", "wep-key1", "wep-key2", "wep-key3", "wep-key-type", "auth-alg")
        if key_mgmt == "none" and any(cp.has_option("wifi-security", k) for k in wep_keys):
          cloudlog.warning(f"NetworkStore: skipping {ssid!r} (WEP profile, unsupported)")
          return
        unsupported_security_options = {key for key, value in cp.items("wifi-security") if value} - _SUPPORTED_SECURITY_OPTIONS
        auth_alg = cp.get("wifi-security", "auth-alg", fallback="").lower()
        psk_flags = cp.getint("wifi-security", "psk-flags", fallback=0)
        if unsupported_security_options or auth_alg not in ("", "open") or psk_flags != 0:
          cloudlog.warning(f"NetworkStore: skipping {ssid!r} with unsupported security constraints")
          return
        if key_mgmt == "wpa-psk":
          security = SecurityType.WPA
          psk = decode_nm_keyfile_string(cp.get("wifi-security", "psk", fallback=""))
          # Agent-managed secrets are unavailable to wpa_supplicant.
          if not is_valid_psk(psk):
            cloudlog.warning(f"NetworkStore: skipping {ssid!r} (wpa-psk with invalid inline secret)")
            return

      # Respect disabled autoconnect profiles
      if not cp.getboolean("connection", "autoconnect", fallback=True):
        cloudlog.warning(f"NetworkStore: skipping {ssid!r} (connection.autoconnect=false)")
        return

      ipv4 = dict(cp["ipv4"]) if cp.has_section("ipv4") else {"method": "auto"}
      ipv6 = dict(cp["ipv6"]) if cp.has_section("ipv6") else {"method": "auto"}
      ipv4_method = ipv4.get("method", "auto").lower()
      ipv6_method = ipv6.get("method", "auto").lower()
      unsupported_ipv4_options = {key for key, value in ipv4.items() if value} - _SUPPORTED_IPV4_OPTIONS
      unsupported_ipv6_options = {key for key, value in ipv6.items() if value} - _SUPPORTED_IPV6_OPTIONS
      ipv4_dns_priority = ipv4.get("dns-priority") or None
      ipv6_addr_gen_mode = (ipv6.get("addr-gen-mode") or "default").lower()
      if (ipv4_method not in _SUPPORTED_IPV4_METHODS
          or ipv6_method not in _SUPPORTED_IPV6_METHODS
          or unsupported_ipv4_options
          or unsupported_ipv6_options
          or ipv4_dns_priority not in (None, _OPENPILOT_DNS_PRIORITY)
          or ipv6_addr_gen_mode != "default"):
        cloudlog.warning(f"NetworkStore: skipping {ssid!r} with unsupported addressing configuration")
        return

      # Skip profiles with malformed integer or boolean values
      entry = {
        "psk": psk,
        "security": security,
        "metered": cp.getint("connection", "metered", fallback=0),
        "priority": cp.getint("connection", "autoconnect-priority", fallback=0),
        "hidden": cp.getboolean("wifi", "hidden", fallback=False),
        "bssid": bssid,
        "uuid": file_uuid,
        "_connection": connection,
        "_ipv4": ipv4,
        "_ipv6": ipv6,
        # Track the source filename for noncanonical profiles
        "_filename": None if imported else fname,
        "_runtime_filename": fname if imported else None,
        "_netplan_filename": self._find_netplan_filename(file_uuid) if imported else None,
      }
      profiles = self._profiles.setdefault(ssid, [])
      if imported and file_uuid in persistent_uuids.get(ssid, set()):
        persistent = next((profile for profile in profiles if profile.get("uuid") == file_uuid), None)
        if persistent is not None and persistent.get("_runtime_filename") is None:
          persistent["_runtime_filename"] = fname
          persistent["_netplan_filename"] = self._find_netplan_filename(file_uuid)
        return
      if any(profile.get("uuid") == file_uuid for profile in profiles):
        return
      profiles.append(entry)
      if not imported:
        persistent_uuids.setdefault(ssid, set()).add(file_uuid)
      self._networks.setdefault(ssid, entry)
    except (configparser.Error, ValueError):
      return

  def _install_keyfile(self, cp: configparser.ConfigParser, path: str):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
      cp.write(f)
      temp_path = f.name

    try:
      os.chmod(temp_path, 0o600)
      subprocess.run(["sudo", "install", "-d", "-m", "755", self._directory], check=True)
      subprocess.run(["sudo", "install", "-o", "root", "-g", "root", "-m", "600", temp_path, path], check=True)
    finally:
      try:
        os.unlink(temp_path)
      except FileNotFoundError:
        pass

  @contextmanager
  def _profile_update(self, ssid: str, profiles: list[dict]) -> Iterator[None]:
    if not profiles:
      yield
      return

    paths: set[str] = set()
    for profile in profiles:
      file_uuid = profile.get("uuid")
      if file_uuid:
        paths.add(os.path.join(self._directory, _canonical_filename(file_uuid, ssid)))
      filename = profile.get("_filename")
      if filename:
        paths.add(os.path.join(self._directory, filename))
      runtime_filename = profile.get("_runtime_filename")
      if self._runtime_directory is not None and runtime_filename:
        paths.add(os.path.join(self._runtime_directory, runtime_filename))
      netplan_filename = profile.get("_netplan_filename")
      if self._netplan_directory is not None and netplan_filename:
        paths.add(os.path.join(self._netplan_directory, netplan_filename))

    original_paths = {path for path in paths if os.path.exists(path)}
    token = uuid.uuid4().hex
    backups: dict[str, str] = {}
    try:
      for path in sorted(original_paths):
        backup_path = f"{path}.openpilot-update-{token}"
        result = subprocess.run([
          "sudo", "install", "-o", "root", "-g", "root", "-m", "600", path, backup_path,
        ], check=False)
        if result.returncode != 0:
          raise OSError(f"failed to back up {path}")
        backups[path] = backup_path
    except Exception as e:
      cleanup_failed = False
      for backup_path in backups.values():
        cleanup_failed |= subprocess.run(["sudo", "rm", "-f", backup_path], check=False).returncode != 0
      if cleanup_failed:
        raise OSError(f"failed to clean up profile backups for {ssid}") from e
      raise

    try:
      yield
    except Exception as e:
      rollback_failed = False
      for path in sorted(paths - original_paths):
        rollback_failed |= subprocess.run(["sudo", "rm", "-f", path], check=False).returncode != 0
      for path, backup_path in backups.items():
        rollback_failed |= subprocess.run(["sudo", "mv", "-f", backup_path, path], check=False).returncode != 0
      if rollback_failed:
        raise OSError(f"failed to roll back profile update for {ssid}") from e
      raise
    else:
      for backup_path in backups.values():
        result = subprocess.run(["sudo", "rm", "-f", backup_path], check=False)
        if result.returncode != 0:
          cloudlog.warning(f"NetworkStore: failed to clean up profile backup {backup_path} (rc={result.returncode})")

  def _render_nmconnection(self, ssid: str, entry: dict) -> tuple[str, dict]:
    file_uuid = entry.get("uuid")
    if not file_uuid:
      try:
        file_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, ssid))
      except UnicodeEncodeError:
        ssid_hex = ssid.encode("utf-8", errors="surrogateescape").hex()
        file_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, ssid_hex))
    entry = dict(entry)
    entry["uuid"] = file_uuid

    canonical_fname = _canonical_filename(file_uuid, ssid)
    canonical_path = os.path.join(self._directory, canonical_fname)
    canonical_existed = os.path.exists(canonical_path)
    stored_fname = entry.get("_filename")
    entry["_filename"] = canonical_fname

    cp = configparser.ConfigParser(interpolation=None)
    connection_id = ssid.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    connection = dict(entry.get("_connection", {}))
    connection.update({
      "id": _encode_keyfile_string(connection_id),
      "uuid": file_uuid,
      "type": "wifi",
      "metered": str(entry.get("metered", 0)),
      "autoconnect-priority": str(entry.get("priority", 0)),
    })
    cp["connection"] = connection
    wifi = {
      "ssid": _encode_keyfile_ssid(ssid),
      "mode": "infrastructure",
      "hidden": str(entry.get("hidden", False)).lower(),
    }
    if entry.get("bssid"):
      wifi["bssid"] = entry["bssid"]
    cp["wifi"] = wifi

    psk = entry.get("psk", "")
    security = entry.get("security", SecurityType.WPA if psk else SecurityType.OPEN)
    if security == SecurityType.WPA:
      cp["wifi-security"] = {
        "key-mgmt": "wpa-psk",
        "psk": _encode_keyfile_string(psk),
      }

    ipv4 = dict(entry.get("_ipv4", {"method": "auto"}))
    ipv4["dns-priority"] = _OPENPILOT_DNS_PRIORITY
    cp["ipv4"] = ipv4
    cp["ipv6"] = entry.get("_ipv6", {"method": "auto"})

    backup_path = None
    if canonical_existed:
      backup_path = f"{canonical_path}.openpilot-update-{uuid.uuid4().hex}"
      result = subprocess.run([
        "sudo", "install", "-o", "root", "-g", "root", "-m", "600", canonical_path, backup_path,
      ], check=False)
      if result.returncode != 0:
        raise OSError(f"failed to back up {canonical_path}")

    def cleanup_canonical_after_failure() -> bool:
      if backup_path is not None:
        return subprocess.run(["sudo", "mv", "-f", backup_path, canonical_path], check=False).returncode == 0
      return subprocess.run(["sudo", "rm", "-f", canonical_path], check=False).returncode == 0

    try:
      self._install_keyfile(cp, canonical_path)
    except Exception as e:
      if not cleanup_canonical_after_failure():
        raise OSError(f"failed to install and roll back {canonical_path}") from e
      raise

    runtime_filename = entry.get("_runtime_filename")
    if self._runtime_directory is not None and runtime_filename:
      runtime_path = os.path.join(self._runtime_directory, runtime_filename)
      result = subprocess.run(["sudo", "rm", "-f", runtime_path], check=False)
      if result.returncode != 0:
        if not cleanup_canonical_after_failure():
          raise OSError(f"failed to remove {runtime_path} and roll back {canonical_path}")
        raise OSError(f"failed to remove {runtime_path}")
      entry["_runtime_filename"] = None

    netplan_filename = entry.get("_netplan_filename")
    if self._netplan_directory is not None and netplan_filename:
      netplan_path = os.path.join(self._netplan_directory, netplan_filename)
      if not os.path.exists(netplan_path):
        if not cleanup_canonical_after_failure():
          raise OSError(f"failed to find {netplan_path} and roll back {canonical_path}")
        raise OSError(f"failed to find {netplan_path}")
      result = subprocess.run(["sudo", "rm", "-f", netplan_path], check=False)
      if result.returncode != 0:
        if not cleanup_canonical_after_failure():
          raise OSError(f"failed to remove {netplan_path} and roll back {canonical_path}")
        raise OSError(f"failed to remove {netplan_path}")
      entry["_netplan_filename"] = None

    # Keep one canonical filename for noncanonical profiles
    if stored_fname and stored_fname != canonical_fname:
      stored_path = os.path.join(self._directory, stored_fname)
      result = subprocess.run(["sudo", "rm", "-f", stored_path], check=False)
      # Mirror failed noncanonical cleanup so both copies remain equivalent
      if result.returncode != 0:
        cloudlog.warning(f"NetworkStore: cleanup of noncanonical {stored_fname} failed; mirroring content to keep both files in sync")
        try:
          subprocess.run(["sudo", "install", "-o", "root", "-g", "root", "-m", "600", os.path.join(self._directory, canonical_fname), stored_path], check=True)
        except Exception:
          cloudlog.exception("NetworkStore: failed to mirror keyfile to noncanonical path")
        entry["_filename"] = stored_fname

    if backup_path is not None:
      result = subprocess.run(["sudo", "rm", "-f", backup_path], check=False)
      if result.returncode != 0:
        if not cleanup_canonical_after_failure():
          raise OSError(f"failed to clean up {backup_path} and roll back {canonical_path}")
        raise OSError(f"failed to clean up {backup_path}")

    return file_uuid, entry

  def get_all(self) -> dict[str, dict]:
    with self._lock:
      return {k: dict(v) for k, v in self._networks.items()}

  def get_profiles(self) -> list[tuple[str, dict]]:
    with self._lock:
      return [
        (ssid, dict(entry))
        for ssid, profiles in self._profiles.items()
        for entry in profiles
      ]

  def get(self, ssid: str) -> dict | None:
    with self._lock:
      entry = self._networks.get(ssid)
      return dict(entry) if entry else None

  def get_tethering_password(self, ssid: str) -> str | None:
    for cp, _, _, _ in self._tethering_profiles(ssid):
      security_section = _keyfile_section(cp, "wifi-security", "802-11-wireless-security")
      assert security_section is not None
      password = decode_nm_keyfile_string(cp.get(security_section, "psk", fallback=""))
      if password:
        return password
    return None

  def _tethering_profiles(self, ssid: str) -> list[tuple[configparser.ConfigParser, str, str, str]]:
    profiles = []
    directories = [self._directory]
    if self._runtime_directory is not None:
      directories.append(self._runtime_directory)

    for directory in directories:
      try:
        filenames = os.listdir(directory)
      except OSError:
        continue
      for fname in filenames:
        if not fname.endswith(".nmconnection"):
          continue
        try:
          raw = sudo_read(os.path.join(directory, fname))
          if not raw:
            continue
          cp = configparser.ConfigParser(interpolation=None)
          cp.read_string(raw)
          wifi_section = _keyfile_section(cp, "wifi", "802-11-wireless")
          security_section = _keyfile_section(cp, "wifi-security", "802-11-wireless-security")
          if wifi_section is None or security_section is None:
            continue
          profile_ssid = decode_nm_keyfile_ssid(cp.get(wifi_section, "ssid", fallback=""))
          if cp.get(wifi_section, "mode", fallback="infrastructure") != "ap" or profile_ssid != ssid:
            continue
          if cp.get(security_section, "key-mgmt", fallback="").lower() != "wpa-psk":
            continue
          file_uuid = _parse_uuid(cp.get("connection", "uuid", fallback=""))
          if file_uuid is None:
            continue
          profiles.append((cp, directory, fname, file_uuid))
        except (configparser.Error, OSError, ValueError):
          continue
    return profiles

  def set_tethering_password(self, ssid: str, password: str) -> bool:
    with self._mutation_lock:
      profiles = self._tethering_profiles(ssid)
      if not profiles:
        return False

      cp, source_directory, source_filename, file_uuid = profiles[0]
      security_section = _keyfile_section(cp, "wifi-security", "802-11-wireless-security")
      assert security_section is not None
      cp[security_section]["psk"] = _encode_keyfile_string(password)

      if source_directory == self._directory:
        target_path = os.path.join(self._directory, source_filename)
      else:
        if not file_uuid:
          return False
        target_path = os.path.join(self._directory, _canonical_filename(file_uuid, ssid))
      target_existed = os.path.exists(target_path)

      runtime_profile = next((profile for profile in profiles
                              if profile[1] == self._runtime_directory and profile[3] == file_uuid), None)
      runtime_path = os.path.join(self._runtime_directory, runtime_profile[2]) if self._runtime_directory is not None and runtime_profile is not None else None
      netplan_filename = self._find_netplan_filename(file_uuid) if runtime_path is not None else None
      netplan_path = os.path.join(self._netplan_directory, netplan_filename) if self._netplan_directory is not None and netplan_filename else None
      if netplan_path is not None and not os.path.exists(netplan_path):
        return False

      token = uuid.uuid4().hex
      target_backup = f"{target_path}.openpilot-update-{token}" if target_existed else None
      if target_backup is not None:
        result = subprocess.run([
          "sudo", "install", "-o", "root", "-g", "root", "-m", "600", target_path, target_backup,
        ], check=False)
        if result.returncode != 0:
          raise OSError(f"failed to back up {target_path}")

      staged_sources: list[tuple[str, str]] = []
      try:
        self._install_keyfile(cp, target_path)
        for source_path in (runtime_path, netplan_path):
          if source_path is None:
            continue
          staged_path = f"{source_path}.openpilot-update-{token}"
          result = subprocess.run(["sudo", "mv", "-f", source_path, staged_path], check=False)
          if result.returncode != 0:
            raise OSError(f"failed to stage {source_path}")
          staged_sources.append((source_path, staged_path))
      except Exception as e:
        rollback_failed = False
        for source_path, staged_path in reversed(staged_sources):
          rollback_failed |= subprocess.run(["sudo", "mv", "-f", staged_path, source_path], check=False).returncode != 0
        if target_backup is not None:
          rollback_failed |= subprocess.run(["sudo", "mv", "-f", target_backup, target_path], check=False).returncode != 0
        else:
          rollback_failed |= subprocess.run(["sudo", "rm", "-f", target_path], check=False).returncode != 0
        if rollback_failed:
          raise OSError(f"failed to roll back tethering password update for {ssid}") from e
        raise

      for _, staged_path in staged_sources:
        if subprocess.run(["sudo", "rm", "-f", staged_path], check=False).returncode != 0:
          cloudlog.warning(f"NetworkStore: failed to clean up staged tethering source {staged_path}")
      if target_backup is not None and subprocess.run(["sudo", "rm", "-f", target_backup], check=False).returncode != 0:
        cloudlog.warning(f"NetworkStore: failed to clean up tethering backup {target_backup}")
      return True

  def save_network(self, ssid: str, psk: str | None = None, metered: int | None = None, hidden: bool | None = None,
                   security: SecurityType | None = None, profile_uuid: str | None = None):
    with self._mutation_lock:
      with self._lock:
        current = self._networks.get(ssid)
        profiles = list(self._profiles.get(ssid, []))
        existing = dict(current or {})
        if profile_uuid is not None:
          canonical_uuid = _parse_uuid(profile_uuid)
          if canonical_uuid is None:
            raise ValueError(f"invalid profile UUID: {profile_uuid!r}")
          if current is not None and current.get("uuid") != canonical_uuid:
            raise ValueError(f"profile UUID changed for {ssid!r}")
          existing["uuid"] = canonical_uuid
        if psk is not None:
          existing["psk"] = psk
          existing["security"] = security if security is not None else (SecurityType.WPA if psk else SecurityType.OPEN)
        else:
          existing.setdefault("psk", "")
          if security is not None:
            existing["security"] = security
          else:
            existing.setdefault("security", SecurityType.WPA if existing["psk"] else SecurityType.OPEN)
        if existing["security"] == SecurityType.OPEN:
          existing["psk"] = ""
        elif not is_valid_psk(existing["psk"]):
          raise ValueError(f"invalid WPA PSK for {ssid!r}")
        if metered is not None:
          existing["metered"] = metered
        elif "metered" not in existing:
          existing["metered"] = 0
        if hidden is not None:
          existing["hidden"] = hidden
        elif "hidden" not in existing:
          existing["hidden"] = False

      with self._profile_update(ssid, profiles):
        file_uuid, updated = self._render_nmconnection(ssid, existing)
        updated["uuid"] = file_uuid
        if current is None:
          profiles.append(updated)
        else:
          updated_profiles = []
          replaced_primary = False
          for profile in profiles:
            if profile is current:
              updated_profiles.append(updated)
              replaced_primary = True
            elif psk is not None and not profile.get("bssid"):
              duplicate = dict(profile)
              duplicate["psk"] = psk
              duplicate["security"] = SecurityType.WPA if psk else SecurityType.OPEN
              duplicate_uuid, duplicate = self._render_nmconnection(ssid, duplicate)
              duplicate["uuid"] = duplicate_uuid
              updated_profiles.append(duplicate)
            else:
              updated_profiles.append(profile)
          if not replaced_primary:
            updated_profiles.append(updated)
          profiles = updated_profiles
      with self._lock:
        self._profiles[ssid] = profiles
        self._networks[ssid] = updated

  def remove(self, ssid: str) -> bool:
    with self._mutation_lock:
      with self._lock:
        entry = self._networks.get(ssid)
        if entry is None:
          return False
        profiles = list(self._profiles.get(ssid, [entry]))

      # Remove every representation so duplicates cannot restore the network
      paths: set[str] = set()
      netplan_paths: set[str] = set()
      for profile in profiles:
        paths.add(os.path.join(self._directory, _canonical_filename(profile.get("uuid", ""), ssid)))
        tracked = profile.get("_filename")
        if tracked:
          paths.add(os.path.join(self._directory, tracked))
        runtime_filename = profile.get("_runtime_filename")
        if self._runtime_directory is not None and runtime_filename:
          paths.add(os.path.join(self._runtime_directory, runtime_filename))
        netplan_filename = profile.get("_netplan_filename")
        if self._netplan_directory is not None and netplan_filename:
          netplan_paths.add(os.path.join(self._netplan_directory, netplan_filename))
      for p in netplan_paths:
        if not os.path.exists(p):
          cloudlog.warning(f"NetworkStore: failed to find netplan source {p}")
          return False
      paths.update(netplan_paths)
      existing_paths = sorted(p for p in paths if os.path.exists(p))
      if len(existing_paths) > 1:
        token = uuid.uuid4().hex
        staged_paths = []
        for p in existing_paths:
          staged_path = f"{p}.openpilot-forget-{token}"
          result = subprocess.run(["sudo", "mv", "-f", p, staged_path], check=False)
          if result.returncode != 0:
            cloudlog.warning(f"NetworkStore: failed to stage {p} for removal (rc={result.returncode})")
            for original_path, rollback_path in reversed(staged_paths):
              rollback = subprocess.run(["sudo", "mv", "-f", rollback_path, original_path], check=False)
              if rollback.returncode != 0:
                cloudlog.warning(f"NetworkStore: failed to roll back {original_path} (rc={rollback.returncode})")
            return False
          staged_paths.append((p, staged_path))
        for _, staged_path in staged_paths:
          result = subprocess.run(["sudo", "rm", "-f", staged_path], check=False)
          if result.returncode != 0:
            cloudlog.warning(f"NetworkStore: failed to clean up staged profile {staged_path} (rc={result.returncode})")
      else:
        for p in existing_paths:
          result = subprocess.run(["sudo", "rm", "-f", p], check=False)
          # Keep the in-memory profile when disk removal fails
          if result.returncode != 0:
            cloudlog.warning(f"NetworkStore: failed to remove {p} (rc={result.returncode})")
            return False
      with self._lock:
        self._networks.pop(ssid, None)
        self._profiles.pop(ssid, None)
      return True

  def set_metered(self, ssid: str, metered: int):
    with self._mutation_lock:
      with self._lock:
        profiles = list(self._profiles.get(ssid, []))
        if not profiles:
          return
        primary = self._networks.get(ssid)
      with self._profile_update(ssid, profiles):
        updated_profiles = []
        updated_primary = None
        for current in profiles:
          updated = dict(current)
          updated["metered"] = metered
          file_uuid, updated = self._render_nmconnection(ssid, updated)
          updated["uuid"] = file_uuid
          updated_profiles.append(updated)
          if current is primary:
            updated_primary = updated
      with self._lock:
        self._profiles[ssid] = updated_profiles
        self._networks[ssid] = updated_primary or updated_profiles[0]

  def get_metered(self, ssid: str, profile_uuid: str | None = None) -> MeteredType:
    with self._lock:
      if profile_uuid is None:
        entry = self._networks.get(ssid)
      else:
        entry = next((profile for profile in self._profiles.get(ssid, []) if profile.get("uuid") == profile_uuid), None)
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
