import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openpilot.system.ui.lib import wifi_network_store as store_module
from openpilot.system.ui.lib.wifi_network_store import NetworkStore
from openpilot.system.ui.lib.wpa_ctrl import generate_wpa_conf


def profile_uuid(name: str) -> str:
  return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def write_profile(directory: str, filename: str, ssid: str, *,
                  file_uuid: str | None = None, psk: str | None = "password123",
                  key_mgmt: str = "wpa-psk", mode: str = "infrastructure",
                  autoconnect: bool = True, autoconnect_priority: int = 0,
                  bssid: str | None = None, extra_wifi: str = "",
                  extra_security: str = "", extra_connection: str = "", valid_uuid: bool = True) -> str:
  file_uuid = file_uuid or ssid
  if valid_uuid:
    file_uuid = profile_uuid(file_uuid)
  security = ""
  if psk is not None or extra_security:
    security = f"""
[wifi-security]
key-mgmt={key_mgmt}
{f"psk={psk}" if psk is not None else ""}
{extra_security}
"""
  content = f"""\
[connection]
id={ssid}
uuid={file_uuid}
type=wifi
autoconnect={str(autoconnect).lower()}
autoconnect-priority={autoconnect_priority}
{extra_connection}

[wifi]
ssid={ssid}
mode={mode}
{f"bssid={bssid}" if bssid is not None else ""}
{extra_wifi}
{security}
"""
  path = os.path.join(directory, filename)
  Path(path).write_text(content)
  return path


def require_entry(store: NetworkStore, ssid: str) -> dict:
  entry = store.get(ssid)
  assert entry is not None
  return entry


class TestNetworkStore(TestCase):
  def setUp(self):
    self.root = tempfile.mkdtemp()
    self.persistent = os.path.join(self.root, "persistent")
    self.runtime = os.path.join(self.root, "runtime")
    self.netplan = os.path.join(self.root, "netplan")
    for directory in (self.persistent, self.runtime, self.netplan):
      os.mkdir(directory)

  def tearDown(self):
    shutil.rmtree(self.root)

  def patch_reads(self):
    return patch.object(store_module, "sudo_read", side_effect=lambda path: Path(path).read_text())

  def make_store(self) -> NetworkStore:
    return NetworkStore(self.persistent, self.runtime, self.netplan)

  def run_file_command(self, command, **_):
    if command[:2] == ["sudo", "install"] and "-d" not in command:
      shutil.copyfile(command[-2], command[-1])
    elif command[:3] == ["sudo", "mv", "-f"]:
      Path(command[-2]).replace(command[-1])
    elif command[:3] == ["sudo", "rm", "-f"]:
      Path(command[-1]).unlink(missing_ok=True)
    return MagicMock(returncode=0)

  def test_loads_persistent_and_open_profiles(self):
    write_profile(self.persistent, "secure.nmconnection", "Secure")
    write_profile(self.persistent, "open.nmconnection", "Open", psk=None)

    with self.patch_reads():
      store = self.make_store()

    assert require_entry(store, "Secure")["psk"] == "password123"
    assert require_entry(store, "Open")["psk"] == ""

  def test_loads_canonical_networkmanager_sections(self):
    path = Path(write_profile(self.persistent, "canonical.nmconnection", "Canonical"))
    raw = path.read_text().replace("[wifi]", "[802-11-wireless]").replace("[wifi-security]", "[802-11-wireless-security]")
    path.write_text(raw)

    with self.patch_reads():
      store = self.make_store()

    assert require_entry(store, "Canonical")["psk"] == "password123"

  def test_loads_current_networkmanager_profile_defaults(self):
    Path(self.persistent, "saved.nmconnection").write_text(f"""\
[connection]
id=openpilot connection SavedNet
uuid={profile_uuid("SavedNet")}
type=wifi
autoconnect-retries=0
timestamp=1775127802

[wifi]
mode=infrastructure
ssid=83;97;118;101;100;78;101;116;
hidden=false
mac-address-blacklist=

[wifi-security]
auth-alg=open
key-mgmt=wpa-psk
psk=password123

[ipv4]
dns-priority=600
dns-search=
method=auto

[ipv6]
addr-gen-mode=default
dns-search=
method=ignore

[proxy]
""")

    with self.patch_reads():
      store = self.make_store()

    assert require_entry(store, "SavedNet")["psk"] == "password123"

  def test_skips_profiles_with_invalid_uuids(self):
    write_profile(self.persistent, "station.nmconnection", "Station", file_uuid="../../station", valid_uuid=False)
    write_profile(self.runtime, "hotspot.nmconnection", "weedle", file_uuid="/tmp/hotspot", mode="ap", valid_uuid=False)

    with self.patch_reads():
      store = self.make_store()

    assert store.get("Station") is None
    assert store.get_tethering_password("weedle") is None

  def test_enforces_connection_interface_constraint(self):
    write_profile(self.persistent, "wlan0.nmconnection", "Wlan0", extra_connection="interface-name=wlan0")
    write_profile(self.persistent, "wlan1.nmconnection", "Wlan1", extra_connection="interface-name=wlan1")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.set_metered("Wlan0", 1)

    assert store.get("Wlan1") is None
    raw = Path(self.persistent, f"{profile_uuid('Wlan0')}-Wlan0.nmconnection").read_text()
    assert "interface-name = wlan0" in raw

  def test_skips_profiles_with_invalid_psks(self):
    write_profile(self.persistent, "short.nmconnection", "Short", psk="short")
    write_profile(self.persistent, "nonhex.nmconnection", "NonHex", psk="x" * 64)
    write_profile(self.persistent, "oversized.nmconnection", "Oversized", psk="é" * 32)
    write_profile(self.persistent, "raw.nmconnection", "Raw", psk="a" * 64)

    with self.patch_reads():
      store = self.make_store()

    assert store.get("Short") is None
    assert store.get("NonHex") is None
    assert store.get("Oversized") is None
    assert require_entry(store, "Raw")["psk"] == "a" * 64

  def test_enforces_ssid_byte_limit(self):
    valid_ssid = "é" * 16
    oversized_ssid = "é" * 17
    write_profile(self.persistent, "valid.nmconnection", valid_ssid)
    write_profile(self.persistent, "oversized.nmconnection", oversized_ssid)

    with self.patch_reads():
      store = self.make_store()

    assert require_entry(store, valid_ssid)["psk"] == "password123"
    assert store.get(oversized_ssid) is None

  def test_loads_autoconnect_priority(self):
    write_profile(self.persistent, "preferred.nmconnection", "Preferred", autoconnect_priority=42)

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.set_metered("Preferred", 1)

    assert require_entry(store, "Preferred")["priority"] == 42
    raw = Path(self.persistent, f"{profile_uuid('Preferred')}-Preferred.nmconnection").read_text()
    assert "autoconnect-priority = 42" in raw

  def test_preserves_bssid_restriction(self):
    write_profile(self.persistent, "pinned.nmconnection", "Pinned", bssid="00:11:22:33:44:55")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.set_metered("Pinned", 1)

    assert require_entry(store, "Pinned")["bssid"] == "00:11:22:33:44:55"
    raw = Path(self.persistent, f"{profile_uuid('Pinned')}-Pinned.nmconnection").read_text()
    assert "bssid = 00:11:22:33:44:55" in raw

  def test_skips_profile_with_invalid_bssid(self):
    write_profile(self.persistent, "valid.nmconnection", "Valid")
    write_profile(self.persistent, "invalid.nmconnection", "Invalid", bssid="not-a-mac")

    with self.patch_reads():
      store = self.make_store()

    config_path = os.path.join(self.root, "wpa_supplicant.conf")
    generate_wpa_conf(store, config_path)

    assert store.get("Invalid") is None
    assert "ssid=56616c6964" in Path(config_path).read_text()

  def test_loads_printable_decimal_list_ssid(self):
    path = write_profile(self.persistent, "decimal.nmconnection", "placeholder")
    raw = Path(path).read_text().replace("ssid=placeholder", "ssid=65;66;67;")
    Path(path).write_text(raw)

    with self.patch_reads():
      store = self.make_store()

    assert require_entry(store, "ABC")["psk"] == "password123"

  def test_preserves_non_utf8_decimal_list_ssid(self):
    path = write_profile(self.persistent, "binary.nmconnection", "placeholder")
    raw = Path(path).read_text().replace("ssid=placeholder", "ssid=255;65;")
    Path(path).write_text(raw)
    ssid = b"\xffA".decode("utf-8", errors="surrogateescape")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.set_metered(ssid, 1)
    config_path = os.path.join(self.root, "wpa_supplicant.conf")
    generate_wpa_conf(store, config_path)

    assert require_entry(store, ssid)["psk"] == "password123"
    assert "ssid=ff41" in Path(config_path).read_text()
    keyfile = next(Path(self.persistent).glob("*.nmconnection"))
    assert "ssid = 255;65;" in keyfile.read_text()

  def test_skips_profiles_that_cannot_be_reproduced_safely(self):
    write_profile(self.persistent, "enterprise.nmconnection", "Enterprise", psk=None, key_mgmt="wpa-eap", extra_security="identity=user")
    write_profile(self.persistent, "agent-secret.nmconnection", "AgentSecret", psk=None, extra_security="psk-flags=1")
    write_profile(self.persistent, "wep.nmconnection", "Wep", psk=None, key_mgmt="none", extra_security="wep-key0=abcde")
    write_profile(self.persistent, "disabled.nmconnection", "Disabled", autoconnect=False)
    write_profile(self.persistent, "hotspot.nmconnection", "Hotspot", psk=None, mode="ap")

    with self.patch_reads():
      store = self.make_store()

    assert store.get_all() == {}

  def test_skips_non_infrastructure_profiles(self):
    for mode in ("adhoc", "mesh"):
      write_profile(self.persistent, f"{mode}.nmconnection", mode.title(), mode=mode)

    with self.patch_reads():
      store = self.make_store()

    assert store.get_all() == {}

  def test_skips_profile_with_unsupported_wifi_options(self):
    write_profile(
      self.persistent,
      "randomized.nmconnection",
      "Randomized",
      extra_wifi="cloned-mac-address=stable",
    )

    with self.patch_reads():
      store = self.make_store()

    assert store.get("Randomized") is None

  def test_skips_profile_with_unsupported_security_constraints(self):
    write_profile(
      self.persistent,
      "constrained.nmconnection",
      "Constrained",
      extra_security="pmf=3\nproto=rsn;\npairwise=ccmp;",
    )

    with self.patch_reads():
      store = self.make_store()

    assert store.get("Constrained") is None

  def test_reads_existing_tethering_password_without_importing_profile(self):
    write_profile(self.persistent, "hotspot.nmconnection", "weedle", psk="custom-password", mode="ap")

    with self.patch_reads():
      store = self.make_store()
      assert store.get_tethering_password("weedle") == "custom-password"
    assert store.get("weedle") is None

  def test_updates_persistent_tethering_password(self):
    path = Path(write_profile(self.persistent, "hotspot.nmconnection", "weedle", psk="old-password", mode="ap"))

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      assert store.set_tethering_password("weedle", "new-password")

    assert "psk = new-password" in path.read_text()

  def test_persists_runtime_tethering_profile_for_rollback(self):
    runtime_path = Path(write_profile(
      self.runtime, "netplan-hotspot.nmconnection", "weedle", file_uuid="hotspot-uuid", psk="old-password", mode="ap",
    ))
    hotspot_uuid = profile_uuid("hotspot-uuid")
    netplan_path = Path(self.netplan, f"90-NM-{hotspot_uuid}.yaml")
    netplan_path.write_text(f"network:\n  version: 2\n  networkmanager:\n    uuid: {hotspot_uuid}\n")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      assert store.set_tethering_password("weedle", "new-password")

    persistent_path = Path(self.persistent, f"{hotspot_uuid}-weedle.nmconnection")
    assert "psk = new-password" in persistent_path.read_text()
    assert not runtime_path.exists()
    assert not netplan_path.exists()

  def test_runtime_profiles_remain_live_sources_despite_stale_marker(self):
    runtime_path = write_profile(self.runtime, "netplan.nmconnection", "Runtime")
    Path(self.persistent, ".wpa_supplicant-import-complete").write_text("complete\n")

    with self.patch_reads(), patch.object(store_module.subprocess, "run") as run:
      store = self.make_store()

      assert require_entry(store, "Runtime")["psk"] == "password123"
      assert os.path.exists(runtime_path)
      assert os.listdir(self.persistent) == [".wpa_supplicant-import-complete"]
      run.assert_not_called()

  def test_persistent_profile_wins_runtime_duplicate(self):
    write_profile(self.persistent, "persistent.nmconnection", "Duplicate", psk="persistent")
    write_profile(self.runtime, "runtime.nmconnection", "Duplicate", psk="runtime")

    with self.patch_reads():
      store = self.make_store()

    assert require_entry(store, "Duplicate")["psk"] == "persistent"

  def test_edit_persistent_profile_removes_shadowed_runtime_copy(self):
    write_profile(self.persistent, "persistent.nmconnection", "Duplicate", file_uuid="shared-uuid", psk="persistent")
    runtime_path = Path(write_profile(self.runtime, "runtime.nmconnection", "Duplicate", file_uuid="shared-uuid", psk="runtime"))
    netplan_path = Path(self.netplan, f"90-NM-{profile_uuid('shared-uuid')}.yaml")
    netplan_path.write_text("network:\n  version: 2\n")

    with (
      self.patch_reads(),
      patch.object(store_module.subprocess, "run", return_value=MagicMock(returncode=0)) as run,
    ):
      store = self.make_store()
      assert require_entry(store, "Duplicate")["_runtime_filename"] == "runtime.nmconnection"
      store.set_metered("Duplicate", 1)

    removed = [item.args[0][-1] for item in run.call_args_list
               if item.args[0][:3] == ["sudo", "rm", "-f"]]
    assert str(runtime_path) in removed
    assert str(netplan_path) in removed

  def test_edit_persistent_profile_preserves_runtime_profile_with_different_uuid(self):
    write_profile(self.persistent, "persistent.nmconnection", "Duplicate", file_uuid="persistent-uuid", psk="persistent")
    runtime_path = Path(write_profile(self.runtime, "runtime.nmconnection", "Duplicate", file_uuid="runtime-uuid", psk="runtime"))
    netplan_path = Path(self.netplan, f"90-NM-{profile_uuid('runtime-uuid')}.yaml")
    netplan_path.write_text("network:\n  version: 2\n")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.set_metered("Duplicate", 1)

    assert runtime_path.exists()
    assert netplan_path.exists()

  def test_failed_noncanonical_cleanup_keeps_profile_copies_equivalent(self):
    stored_path = Path(write_profile(self.persistent, "stored.nmconnection", "Stored", file_uuid="stored-uuid"))

    def run(command, **kwargs):
      if command[:3] == ["sudo", "rm", "-f"] and command[-1] == str(stored_path):
        return MagicMock(returncode=1)
      return self.run_file_command(command, **kwargs)

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=run):
      store = self.make_store()
      store.set_metered("Stored", 1)

    canonical_path = Path(self.persistent, f"{profile_uuid('stored-uuid')}-Stored.nmconnection")
    assert canonical_path.read_text() == stored_path.read_text()
    assert require_entry(store, "Stored")["_filename"] == "stored.nmconnection"

  def test_emits_multiple_profiles_with_the_same_ssid(self):
    write_profile(
      self.persistent,
      "first.nmconnection",
      "Pinned",
      file_uuid="first-uuid",
      bssid="00:11:22:33:44:55",
    )
    write_profile(
      self.persistent,
      "second.nmconnection",
      "Pinned",
      file_uuid="second-uuid",
      bssid="66:77:88:99:aa:bb",
    )

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.set_metered("Pinned", 1)
    config_path = os.path.join(self.root, "wpa_supplicant.conf")
    generate_wpa_conf(store, config_path)
    config = Path(config_path).read_text()

    assert config.count("network={") == 2
    assert "bssid=00:11:22:33:44:55" in config
    assert "bssid=66:77:88:99:aa:bb" in config

  def test_metered_updates_every_profile_with_the_same_ssid(self):
    write_profile(self.persistent, "first.nmconnection", "Duplicate", file_uuid="first-uuid")
    write_profile(self.persistent, "second.nmconnection", "Duplicate", file_uuid="second-uuid")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.set_metered("Duplicate", 1)

    profiles = [entry for ssid, entry in store.get_profiles() if ssid == "Duplicate"]
    assert len(profiles) == 2
    assert all(entry["metered"] == 1 for entry in profiles)
    for file_uuid in (profile_uuid("first-uuid"), profile_uuid("second-uuid")):
      raw = Path(self.persistent, f"{file_uuid}-Duplicate.nmconnection").read_text()
      assert "metered = 1" in raw

  def test_replacement_psk_preserves_other_profile_credentials(self):
    write_profile(
      self.persistent,
      "first.nmconnection",
      "Duplicate",
      file_uuid="first-uuid",
      psk="stale-password",
      bssid="00:11:22:33:44:55",
    )
    write_profile(
      self.persistent,
      "second.nmconnection",
      "Duplicate",
      file_uuid="second-uuid",
      psk="alternate-password",
      bssid="66:77:88:99:aa:bb",
    )

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.save_network("Duplicate", psk="replacement-password")

    profiles = [entry for ssid, entry in store.get_profiles() if ssid == "Duplicate"]
    assert len(profiles) == 2
    assert {entry["bssid"] for entry in profiles} == {"00:11:22:33:44:55", "66:77:88:99:aa:bb"}
    assert {entry["bssid"]: entry["psk"] for entry in profiles} == {
      "00:11:22:33:44:55": "replacement-password",
      "66:77:88:99:aa:bb": "alternate-password",
    }
    assert "psk = replacement-password" in Path(self.persistent, f"{profile_uuid('first-uuid')}-Duplicate.nmconnection").read_text()
    assert "psk=alternate-password" in Path(self.persistent, "second.nmconnection").read_text()

  def test_replacement_psk_updates_every_unpinned_profile(self):
    write_profile(self.persistent, "first.nmconnection", "Duplicate", file_uuid="first-uuid", psk="stale-password")
    write_profile(self.persistent, "second.nmconnection", "Duplicate", file_uuid="second-uuid", psk="alternate-password")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.save_network("Duplicate", psk="replacement-password")

    profiles = [entry for ssid, entry in store.get_profiles() if ssid == "Duplicate"]
    assert len(profiles) == 2
    assert all(entry["psk"] == "replacement-password" for entry in profiles)
    for file_uuid in (profile_uuid("first-uuid"), profile_uuid("second-uuid")):
      raw = Path(self.persistent, f"{file_uuid}-Duplicate.nmconnection").read_text()
      assert "psk = replacement-password" in raw

  def test_readers_do_not_wait_for_profile_writes(self):
    write_profile(self.persistent, "saved.nmconnection", "Saved")
    render_started = threading.Event()
    release_render = threading.Event()
    read_finished = threading.Event()

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      render = store._render_nmconnection

      def blocking_render(*args, **kwargs):
        render_started.set()
        assert release_render.wait(1)
        return render(*args, **kwargs)

      with patch.object(store, "_render_nmconnection", side_effect=blocking_render):
        writer = threading.Thread(target=store.save_network, args=("Saved",), kwargs={"psk": "replacement-password"})
        reader = threading.Thread(target=lambda: (store.contains("Saved"), read_finished.set()))
        writer.start()
        assert render_started.wait(1)
        reader.start()
        read_completed_during_write = read_finished.wait(0.1)
        release_render.set()
        writer.join(1)
        reader.join(1)

    assert read_completed_during_write
    assert not writer.is_alive()
    assert not reader.is_alive()

  def test_unsupported_persistent_profile_blocks_runtime_duplicate(self):
    write_profile(self.persistent, "persistent.nmconnection", "Enterprise", psk=None, key_mgmt="wpa-eap", extra_security="identity=user")
    write_profile(self.runtime, "runtime.nmconnection", "Enterprise")

    with self.patch_reads():
      store = self.make_store()

    assert store.get("Enterprise") is None

  def test_persistent_profile_with_unsupported_wifi_options_blocks_runtime_duplicate(self):
    write_profile(self.persistent, "persistent.nmconnection", "Randomized", extra_wifi="cloned-mac-address=stable")
    write_profile(self.runtime, "runtime.nmconnection", "Randomized")

    with self.patch_reads():
      store = self.make_store()

    assert store.get("Randomized") is None

  def test_forget_runtime_profile_removes_netplan_source(self):
    write_profile(self.runtime, "netplan.nmconnection", "Runtime", file_uuid="runtime-uuid")
    netplan_path = Path(self.netplan, f"90-NM-{profile_uuid('runtime-uuid')}.yaml")
    netplan_path.write_text("network:\n  version: 2\n")

    with (
      self.patch_reads(),
      patch.object(store_module.subprocess, "run", return_value=MagicMock(returncode=0)) as run,
    ):
      store = self.make_store()

      assert store.remove("Runtime")

      staged = [args.args[0][-2] for args in run.call_args_list if args.args[0][:3] == ["sudo", "mv", "-f"]]
      assert str(netplan_path) in staged
      assert str(Path(self.runtime, "netplan.nmconnection")) in staged
      assert store.get("Runtime") is None

  def test_edit_runtime_profile_without_netplan_source(self):
    write_profile(self.runtime, "runtime.nmconnection", "Runtime", file_uuid="runtime-uuid")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.save_network("Runtime", psk="replacement-password")

    assert not Path(self.runtime, "runtime.nmconnection").exists()
    assert Path(self.persistent, f"{profile_uuid('runtime-uuid')}-Runtime.nmconnection").exists()
    assert require_entry(store, "Runtime")["_netplan_filename"] is None

  def test_forget_runtime_profile_without_netplan_source(self):
    write_profile(self.runtime, "runtime.nmconnection", "Runtime", file_uuid="runtime-uuid")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      assert store.remove("Runtime")

    assert not Path(self.runtime, "runtime.nmconnection").exists()
    assert not store.contains("Runtime")

  def test_forget_finds_renamed_netplan_source_by_uuid(self):
    write_profile(self.runtime, "netplan.nmconnection", "Runtime", file_uuid="runtime-uuid")
    netplan_path = Path(self.netplan, "provisioned-wifi.yaml")
    netplan_path.write_text(f"network:\n  version: 2\n  networkmanager:\n    uuid: {profile_uuid('runtime-uuid')}\n")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      assert store.remove("Runtime")

    assert not netplan_path.exists()

  def test_forget_ignores_unrelated_netplan_source(self):
    runtime_path = Path(write_profile(self.runtime, "netplan.nmconnection", "Runtime", file_uuid="runtime-uuid"))
    netplan_path = Path(self.netplan, "provisioned-wifi.yaml")
    netplan_path.write_text("network:\n  version: 2\n  wifis:\n    wlan0:\n      access-points:\n        Runtime: {}\n")

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      assert store.remove("Runtime")

    assert not runtime_path.exists()
    assert netplan_path.exists()

  def test_forget_refuses_unreadable_netplan_source(self):
    runtime_path = Path(write_profile(self.runtime, "netplan.nmconnection", "Runtime", file_uuid="runtime-uuid"))
    netplan_path = Path(self.netplan, "provisioned-wifi.yaml")
    netplan_path.write_text("network:\n  version: 2\n")

    def read(path):
      return "" if path == str(netplan_path) else Path(path).read_text()

    with (
      patch.object(store_module, "sudo_read", side_effect=read),
      patch.object(store_module.subprocess, "run", side_effect=self.run_file_command),
    ):
      store = self.make_store()
      assert not store.remove("Runtime")

    assert runtime_path.exists()
    assert netplan_path.exists()

  def test_forget_duplicate_runtime_profiles_removes_every_netplan_source(self):
    write_profile(self.runtime, "first.nmconnection", "Duplicate", file_uuid="first-uuid")
    write_profile(self.runtime, "second.nmconnection", "Duplicate", file_uuid="second-uuid")
    netplan_paths = {
      Path(self.netplan, f"90-NM-{profile_uuid('first-uuid')}.yaml"),
      Path(self.netplan, f"90-NM-{profile_uuid('second-uuid')}.yaml"),
    }
    for path in netplan_paths:
      path.write_text("network:\n  version: 2\n")

    with (
      self.patch_reads(),
      patch.object(store_module.subprocess, "run", return_value=MagicMock(returncode=0)) as run,
    ):
      store = self.make_store()
      assert store.remove("Duplicate")

    staged = {Path(item.args[0][-2]) for item in run.call_args_list
              if item.args[0][:3] == ["sudo", "mv", "-f"]}
    assert netplan_paths <= staged

  def test_forget_preserves_unsupported_profile_with_same_ssid(self):
    unsupported = Path(write_profile(
      self.persistent,
      "enterprise.nmconnection",
      "Duplicate",
      file_uuid="enterprise-uuid",
      psk=None,
      key_mgmt="wpa-eap",
      extra_security="identity=user",
    ))
    managed = Path(write_profile(self.persistent, "managed.nmconnection", "Duplicate", file_uuid="managed-uuid"))

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      assert store.remove("Duplicate")

    assert unsupported.exists()
    assert not managed.exists()

  def test_forget_keeps_profile_when_disk_removal_fails(self):
    write_profile(self.runtime, "netplan.nmconnection", "Runtime", file_uuid="runtime-uuid")

    with (
      self.patch_reads(),
      patch.object(store_module.subprocess, "run", return_value=MagicMock(returncode=1)),
    ):
      store = self.make_store()

      assert not store.remove("Runtime")
      assert store.get("Runtime") is not None

  def test_forget_rolls_back_earlier_removals(self):
    first_path = Path(write_profile(self.persistent, "a.nmconnection", "Duplicate", file_uuid="first-uuid"))
    second_path = Path(write_profile(self.persistent, "b.nmconnection", "Duplicate", file_uuid="second-uuid"))
    originals = {first_path, second_path}
    mutations = 0

    def run(command, **kwargs):
      nonlocal mutations
      if command[:3] == ["sudo", "rm", "-f"] and Path(command[-1]) in originals:
        if mutations == 1:
          return MagicMock(returncode=1)
        mutations += 1
      if command[:3] == ["sudo", "mv", "-f"]:
        source = Path(command[-2])
        if source in originals:
          if mutations == 1:
            return MagicMock(returncode=1)
          mutations += 1
        source.replace(command[-1])
        return MagicMock(returncode=0)
      return self.run_file_command(command, **kwargs)

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=run):
      store = self.make_store()
      assert not store.remove("Duplicate")

    assert first_path.exists()
    assert second_path.exists()
    assert store.contains("Duplicate")

  def test_edit_runtime_profile_installs_keyfile_before_removing_netplan(self):
    write_profile(self.runtime, "netplan.nmconnection", "Runtime", file_uuid="runtime-uuid")
    netplan_path = Path(self.netplan, f"90-NM-{profile_uuid('runtime-uuid')}.yaml")
    netplan_path.write_text("network:\n  version: 2\n")

    with (
      self.patch_reads(),
      patch.object(store_module.subprocess, "run", return_value=MagicMock(returncode=0)) as run,
    ):
      store = self.make_store()

      store.save_network("Runtime", psk="replacement")

      commands = [item.args[0] for item in run.call_args_list]
      install_index = next(i for i, command in enumerate(commands)
                           if command[:2] == ["sudo", "install"] and command[-1].endswith(f"{profile_uuid('runtime-uuid')}-Runtime.nmconnection"))
      runtime_remove_index = next(i for i, command in enumerate(commands)
                                  if command[:3] == ["sudo", "rm", "-f"] and command[-1] == str(Path(self.runtime, "netplan.nmconnection")))
      remove_index = next(i for i, command in enumerate(commands)
                          if command[:3] == ["sudo", "rm", "-f"] and command[-1] == str(netplan_path))
      assert install_index < runtime_remove_index < remove_index
      assert require_entry(store, "Runtime")["psk"] == "replacement"
      assert require_entry(store, "Runtime")["_runtime_filename"] is None
      assert require_entry(store, "Runtime")["_netplan_filename"] is None

  def test_edit_runtime_profile_rolls_back_when_runtime_remove_fails(self):
    runtime_path = Path(write_profile(self.runtime, "netplan.nmconnection", "Runtime", file_uuid="runtime-uuid"))
    netplan_path = Path(self.netplan, f"90-NM-{profile_uuid('runtime-uuid')}.yaml")
    netplan_path.write_text("network:\n  version: 2\n")

    def run(command, **_):
      return MagicMock(returncode=1 if command[-1] == str(runtime_path) else 0)

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=run) as process:
      store = self.make_store()
      keyfile_path = os.path.join(self.persistent, f"{profile_uuid('runtime-uuid')}-Runtime.nmconnection")

      with self.assertRaises(OSError):
        store.set_metered("Runtime", 1)

      commands = [item.args[0] for item in process.call_args_list]
      assert ["sudo", "rm", "-f", keyfile_path] in commands
      assert ["sudo", "rm", "-f", str(netplan_path)] not in commands
      assert require_entry(store, "Runtime")["metered"] == 0
      assert require_entry(store, "Runtime")["_runtime_filename"] == "netplan.nmconnection"

  def test_runtime_cleanup_failure_preserves_existing_persistent_profile(self):
    persistent_path = Path(write_profile(
      self.persistent, f"{profile_uuid('shared-uuid')}-Duplicate.nmconnection", "Duplicate", file_uuid="shared-uuid", psk="original-password",
    ))
    runtime_path = Path(write_profile(self.runtime, "runtime.nmconnection", "Duplicate", file_uuid="shared-uuid"))

    def run(command, **kwargs):
      if command[:3] == ["sudo", "rm", "-f"] and command[-1] == str(runtime_path):
        return MagicMock(returncode=1)
      return self.run_file_command(command, **kwargs)

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=run):
      store = self.make_store()

      with self.assertRaises(OSError):
        store.save_network("Duplicate", psk="replacement-password")

    assert persistent_path.exists()

  def test_runtime_cleanup_failure_preserves_noncanonical_persistent_profile(self):
    persistent_path = Path(write_profile(
      self.persistent, "saved.nmconnection", "Duplicate", file_uuid="shared-uuid", psk="original-password",
    ))
    runtime_path = Path(write_profile(self.runtime, "runtime.nmconnection", "Duplicate", file_uuid="shared-uuid"))

    def run(command, **kwargs):
      if command[:3] == ["sudo", "rm", "-f"] and command[-1] == str(runtime_path):
        return MagicMock(returncode=1)
      return self.run_file_command(command, **kwargs)

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=run):
      store = self.make_store()

      with self.assertRaises(OSError):
        store.set_metered("Duplicate", 1)

    assert persistent_path.exists()

  def test_edit_runtime_profile_rolls_back_when_netplan_remove_fails(self):
    write_profile(self.runtime, "netplan.nmconnection", "Runtime", file_uuid="runtime-uuid")
    netplan_path = Path(self.netplan, f"90-NM-{profile_uuid('runtime-uuid')}.yaml")
    netplan_path.write_text("network:\n  version: 2\n")

    def run(command, **_):
      return MagicMock(returncode=1 if command[-1] == str(netplan_path) else 0)

    with self.patch_reads(), patch.object(store_module.subprocess, "run", side_effect=run) as process:
      store = self.make_store()
      keyfile_path = os.path.join(self.persistent, f"{profile_uuid('runtime-uuid')}-Runtime.nmconnection")

      with self.assertRaises(OSError):
        store.save_network("Runtime", psk="replacement")

      commands = [item.args[0] for item in process.call_args_list]
      assert ["sudo", "rm", "-f", keyfile_path] in commands
      assert require_entry(store, "Runtime")["psk"] == "password123"
      assert require_entry(store, "Runtime")["_netplan_filename"] == f"90-NM-{profile_uuid('runtime-uuid')}.yaml"

  def test_accepts_only_networkmanager_dns_priority(self):
    rollback_path = write_profile(self.persistent, "rollback.nmconnection", "Rollback")
    custom_path = write_profile(self.persistent, "custom.nmconnection", "Custom")
    for path, priority in ((rollback_path, 600), (custom_path, 100)):
      with Path(path).open("a") as f:
        f.write(f"""\
[ipv4]
method=auto
dns-priority={priority}

[ipv6]
method=auto
""")

    with self.patch_reads():
      store = self.make_store()

    assert require_entry(store, "Rollback")["_ipv4"]["dns-priority"] == "600"
    assert store.get("Custom") is None

  def test_skips_profile_with_unsupported_automatic_ip_options(self):
    path = write_profile(self.persistent, "static.nmconnection", "Static")
    with Path(path).open("a") as f:
      f.write("""\
[ipv4]
method=auto
dns=1.1.1.1;9.9.9.9;

[ipv6]
method=auto
addr-gen-mode=stable-privacy
""")

    with self.patch_reads():
      store = self.make_store()

    assert store.get("Static") is None
    assert "dns=1.1.1.1;9.9.9.9;" in Path(path).read_text()
    assert "addr-gen-mode=stable-privacy" in Path(path).read_text()

  def test_skips_profile_with_unsupported_addressing(self):
    path = write_profile(self.persistent, "static.nmconnection", "Static")
    with Path(path).open("a") as f:
      f.write("""\
[ipv4]
method=manual
address1=192.168.50.10/24,192.168.50.1

[ipv6]
method=auto
""")

    with self.patch_reads():
      store = self.make_store()

    assert store.get("Static") is None
    assert "address1=192.168.50.10/24,192.168.50.1" in Path(path).read_text()

  def test_saved_profile_uses_nm_keyfile_compatible_name_and_mode(self):
    with patch.object(store_module.subprocess, "run", return_value=MagicMock(returncode=0)) as run:
      store = self.make_store()

      store.save_network("Cafe/Wifi", psk="password123")

      install = next(item.args[0] for item in run.call_args_list
                     if item.args[0][:2] == ["sudo", "install"] and "-d" not in item.args[0])
      assert install[-1].endswith("-Cafe_Wifi.nmconnection")
      assert install[install.index("-m") + 1] == "600"

  def test_new_profile_writes_rollback_dns_priority(self):
    with patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.save_network("Rollback", psk="password123")

    raw = next(Path(self.persistent).glob("*.nmconnection")).read_text()
    assert "dns-priority = 600" in raw

  def test_round_trips_boundary_whitespace_with_keyfile_escaping(self):
    with patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()

      store.save_network(" Cafe ", psk=" password123 ")
      with self.patch_reads():
        reloaded = self.make_store()

    assert require_entry(reloaded, " Cafe ")["psk"] == " password123 "
    raw = next(Path(self.persistent).glob("*.nmconnection")).read_text()
    assert "ssid = 32;67;97;102;101;32;" in raw
    assert r"psk = \spassword123\s" in raw

  def test_writes_non_ascii_ssid_as_utf8_byte_list(self):
    with patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.save_network("café", psk="password123", metered=1)

    raw = next(Path(self.persistent).glob("*.nmconnection")).read_text()
    assert "ssid = 99;97;102;195;169;" in raw

  def test_round_trips_decimal_list_like_literal_ssid(self):
    with patch.object(store_module.subprocess, "run", side_effect=self.run_file_command):
      store = self.make_store()
      store.save_network("65;66;67;", psk="password123")
      with self.patch_reads():
        reloaded = self.make_store()

    assert require_entry(reloaded, "65;66;67;")["psk"] == "password123"

  def test_get_returns_copy(self):
    store = self.make_store()
    store._networks["Test"] = {"psk": "password123"}

    entry = require_entry(store, "Test")
    entry["psk"] = "changed"

    assert require_entry(store, "Test")["psk"] == "password123"
