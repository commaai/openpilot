"""Tests for nm_persist (slurp netplan-derived wifi keyfiles to /data)."""
import os
import shutil
import subprocess
import tempfile

from openpilot.system.hardware.tici import nm_persist


WIFI_UUID = "48b912a7-7698-4a81-94c4-8cbe1604fd9d"
ETH_UUID = "10c8e2f8-797f-34d8-bb2a-09ca8afd9ddd"

WIFI_KEYFILE = """[connection]
id=Hotspot
type=wifi
uuid={uuid}

[wifi]
ssid={ssid}
mode=infrastructure

[wifi-security]
key-mgmt=wpa-psk
psk=swagswagcomma

[ipv4]
method=auto

[ipv6]
method=auto
"""

ETH_KEYFILE = """[connection]
id=Wired connection 1
type=ethernet
uuid={uuid}

[ethernet]
wake-on-lan=1

[ipv4]
method=auto
address1=192.168.2.2/24

[ipv6]
method=auto
"""

WIFI_NETPLAN_YAML = "network:\n  version: 2\n  wifis: {}\n"
ETH_NETPLAN_YAML = "network:\n  version: 2\n  ethernets: {}\n"


class TestNmPersist:
  def setup_method(self):
    self.tmp = tempfile.mkdtemp()
    self.run_dir = os.path.join(self.tmp, "run")
    self.data_dir = os.path.join(self.tmp, "data")
    self.netplan_dir = os.path.join(self.tmp, "netplan")
    for d in (self.run_dir, self.data_dir, self.netplan_dir):
      os.makedirs(d)

  def teardown_method(self):
    shutil.rmtree(self.tmp, ignore_errors=True)

  def _patch_io(self, mocker):
    """Replace sudo_read with a plain read and subprocess.run with a no-sudo cp/rm/mkdir."""
    def fake_sudo_read(path):
      try:
        with open(path) as f:
          return f.read().strip()
      except (OSError, FileNotFoundError):
        return ""
    mocker.patch.object(nm_persist, "sudo_read", side_effect=fake_sudo_read)

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
      # Strip leading "sudo" and -o/-g ownership pairs so the op succeeds as the test user.
      if cmd[:1] == ["sudo"]:
        cmd = cmd[1:]
      filtered = []
      skip = False
      for arg in cmd:
        if skip:
          skip = False
          continue
        if arg in ("-o", "-g"):
          skip = True
          continue
        filtered.append(arg)
      return real_run(filtered, **kwargs)
    mocker.patch.object(nm_persist.subprocess, "run", side_effect=fake_run)

  def _write_run_keyfile(self, fname: str, content: str) -> str:
    path = os.path.join(self.run_dir, fname)
    with open(path, "w") as f:
      f.write(content)
    return path

  def _write_netplan_yaml(self, fname: str, content: str) -> str:
    path = os.path.join(self.netplan_dir, fname)
    with open(path, "w") as f:
      f.write(content)
    return path

  def test_persists_wifi_and_deletes_netplan(self, mocker):
    self._patch_io(mocker)
    self._write_run_keyfile(f"netplan-NM-{WIFI_UUID}-weedle-bbc2.nmconnection",
                            WIFI_KEYFILE.format(uuid=WIFI_UUID, ssid="weedle-bbc2"))
    yaml_path = self._write_netplan_yaml(f"90-NM-{WIFI_UUID}.yaml", WIFI_NETPLAN_YAML)

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)

    assert os.path.exists(os.path.join(self.data_dir, f"{WIFI_UUID}-weedle-bbc2.nmconnection"))
    assert not os.path.exists(yaml_path)

  def test_skips_ethernet(self, mocker):
    self._patch_io(mocker)
    self._write_run_keyfile(f"netplan-NM-{ETH_UUID}.nmconnection", ETH_KEYFILE.format(uuid=ETH_UUID))
    yaml_path = self._write_netplan_yaml(f"90-NM-{ETH_UUID}.yaml", ETH_NETPLAN_YAML)

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)

    assert os.listdir(self.data_dir) == [], "ethernet keyfile must not be slurped"
    assert os.path.exists(yaml_path), "ethernet netplan YAML must be left in place"

  def test_skips_native_keyfiles(self, mocker):
    """Files without `netplan-NM-` prefix (lo, lte, esim, tailscale0, etc.) are owned
    by other producers and must not be touched."""
    self._patch_io(mocker)
    self._write_run_keyfile("lo.nmconnection", "[connection]\ntype=loopback\n")
    self._write_run_keyfile("lte.nmconnection", "[connection]\ntype=gsm\n")
    self._write_run_keyfile("tailscale0.nmconnection", "[connection]\ntype=tun\n")

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)

    assert os.listdir(self.data_dir) == []

  def test_idempotent_no_rewrite_when_identical(self, mocker):
    self._patch_io(mocker)
    content = WIFI_KEYFILE.format(uuid=WIFI_UUID, ssid="MyNet")
    self._write_run_keyfile(f"netplan-NM-{WIFI_UUID}-MyNet.nmconnection", content)
    self._write_netplan_yaml(f"90-NM-{WIFI_UUID}.yaml", WIFI_NETPLAN_YAML)

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)
    dest = os.path.join(self.data_dir, f"{WIFI_UUID}-MyNet.nmconnection")
    assert os.path.exists(dest)
    first_mtime = os.path.getmtime(dest)

    # Re-run: dest already matches source, no rewrite. Yaml stays gone.
    self._write_netplan_yaml(f"90-NM-{WIFI_UUID}.yaml", WIFI_NETPLAN_YAML)  # someone re-created it
    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)
    assert os.path.getmtime(dest) == first_mtime, "should not rewrite identical content"
    assert not os.path.exists(os.path.join(self.netplan_dir, f"90-NM-{WIFI_UUID}.yaml"))

  def test_no_run_dir_is_no_op(self, mocker):
    """Once NM/netplan are gone, /run/NetworkManager/system-connections doesn't exist."""
    self._patch_io(mocker)
    shutil.rmtree(self.run_dir)
    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)
    # No raise; data dir untouched.
    assert os.listdir(self.data_dir) == []

  def test_overwrites_when_content_differs(self, mocker):
    """If a stale keyfile is on disk from a previous run, persist replaces it."""
    self._patch_io(mocker)
    dest = os.path.join(self.data_dir, f"{WIFI_UUID}-MyNet.nmconnection")
    with open(dest, "w") as f:
      f.write("[connection]\nid=Stale\n")

    self._write_run_keyfile(f"netplan-NM-{WIFI_UUID}-MyNet.nmconnection",
                            WIFI_KEYFILE.format(uuid=WIFI_UUID, ssid="MyNet"))

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)

    with open(dest) as f:
      assert "id=Hotspot" in f.read(), "stale content must be replaced"

  def test_sanitizes_ssid_with_path_chars(self, mocker):
    self._patch_io(mocker)
    self._write_run_keyfile(f"netplan-NM-{WIFI_UUID}-weird.nmconnection",
                            WIFI_KEYFILE.format(uuid=WIFI_UUID, ssid="ev/il"))

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)

    assert os.path.exists(os.path.join(self.data_dir, f"{WIFI_UUID}-ev_il.nmconnection"))

  def test_skips_malformed_keyfile(self, mocker):
    self._patch_io(mocker)
    self._write_run_keyfile(f"netplan-NM-{WIFI_UUID}-bad.nmconnection", "not a valid keyfile [")

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)

    assert os.listdir(self.data_dir) == []

  def test_skips_ap_mode_hotspot(self, mocker):
    """Hotspot profiles are owned by openpilot's tethering path, not NetworkStore.
    Persisting them here would just be dead clutter."""
    self._patch_io(mocker)
    ap_keyfile = f"""[connection]
id=Hotspot
type=wifi
uuid={WIFI_UUID}

[wifi]
ssid=weedle-bbc2
mode=ap
"""
    self._write_run_keyfile(f"netplan-NM-{WIFI_UUID}-weedle-bbc2.nmconnection", ap_keyfile)
    yaml_path = self._write_netplan_yaml(f"90-NM-{WIFI_UUID}.yaml", WIFI_NETPLAN_YAML)

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)

    assert os.listdir(self.data_dir) == [], "AP-mode wifi must not be slurped"
    assert os.path.exists(yaml_path), "AP-mode netplan YAML must be left in place"

  def test_skips_keyfile_without_ssid(self, mocker):
    self._patch_io(mocker)
    self._write_run_keyfile(
      f"netplan-NM-{WIFI_UUID}-noSsid.nmconnection",
      f"[connection]\ntype=wifi\nuuid={WIFI_UUID}\n\n[wifi]\nmode=infrastructure\n",
    )

    nm_persist.persist_connections(self.run_dir, self.data_dir, self.netplan_dir)

    assert os.listdir(self.data_dir) == []
