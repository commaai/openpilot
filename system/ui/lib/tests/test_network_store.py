"""Tests for NetworkStore (saved WiFi network persistence)."""
import os
import tempfile

from pytest_mock import MockerFixture

from openpilot.system.ui.lib.wifi_network_store import NetworkStore
from openpilot.system.ui.lib.wpa_ctrl import _generate_wpa_conf, _format_psk_value, _is_raw_psk


class TestNetworkStore:
  def setup_method(self):
    self.tmpdir = tempfile.mkdtemp()

  def _make_store(self, mocker: MockerFixture):
    mocker.patch("subprocess.run")
    return NetworkStore(directory=self.tmpdir)

  def test_empty_store(self, mocker: MockerFixture):
    store = self._make_store(mocker)
    assert store.get_all() == {}

  def test_remove_nonexistent_returns_false(self, mocker: MockerFixture):
    store = self._make_store(mocker)
    assert store.remove("DoesNotExist") is False

  def test_remove_existing_returns_true(self, mocker: MockerFixture):
    store = self._make_store(mocker)
    store._networks["TestNet"] = {"psk": "pass123", "metered": 0, "hidden": False, "uuid": "abc"}
    mock_run = mocker.patch("subprocess.run", return_value=mocker.MagicMock(returncode=0))
    result = store.remove("TestNet")
    assert result is True
    assert "TestNet" not in store._networks
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[:3] == ["sudo", "rm", "-f"]

  def test_remove_uses_check_false(self, mocker: MockerFixture):
    """Verify remove uses check=False, not check=True (rm -f handles missing files)."""
    store = self._make_store(mocker)
    store._networks["TestNet"] = {"psk": "x", "metered": 0, "hidden": False, "uuid": "abc"}
    mock_run = mocker.patch("subprocess.run", return_value=mocker.MagicMock(returncode=0))
    store.remove("TestNet")
    kwargs = mock_run.call_args[1]
    assert kwargs.get("check") is False, "remove() should use check=False since rm -f handles missing files"

  def test_remove_keeps_in_memory_when_rm_fails(self, mocker: MockerFixture):
    """If `sudo rm` returns non-zero (e.g. FS read-only), the file persists on disk
    and _load would restore the entry on next start. Don't lose the in-memory mapping
    in that window — auto-connect to a "forgotten" network is the failure mode here."""
    store = self._make_store(mocker)
    store._networks["TestNet"] = {"psk": "x", "metered": 0, "hidden": False, "uuid": "abc"}
    mocker.patch("subprocess.run", return_value=mocker.MagicMock(returncode=1))
    result = store.remove("TestNet")
    assert result is False
    assert "TestNet" in store._networks, "in-memory entry must survive rm failure"

  def test_get_returns_copy(self, mocker: MockerFixture):
    store = self._make_store(mocker)
    store._networks["TestNet"] = {"psk": "pass123", "metered": 0, "hidden": False, "uuid": "abc"}
    entry = store.get("TestNet")
    assert entry is not None
    entry["psk"] = "CHANGED"
    assert store.get("TestNet")["psk"] == "pass123"

  def test_get_nonexistent_returns_none(self, mocker: MockerFixture):
    store = self._make_store(mocker)
    assert store.get("DoesNotExist") is None

  def test_load_reads_nmconnection_files(self, mocker: MockerFixture):
    """Write a real .nmconnection file and verify it loads."""
    content = """\
[connection]
id=MyWifi
uuid=test-uuid-123
type=wifi
metered=0

[wifi]
ssid=MyWifi
mode=infrastructure

[wifi-security]
key-mgmt=wpa-psk
psk=secret123

[ipv4]
method=auto
"""
    fpath = os.path.join(self.tmpdir, "MyWifi.nmconnection")
    with open(fpath, "w") as f:
      f.write(content)

    mocker.patch("openpilot.system.ui.lib.wifi_network_store.sudo_read", return_value=content)
    store = NetworkStore(directory=self.tmpdir)

    entry = store.get("MyWifi")
    assert entry is not None
    assert entry["psk"] == "secret123"
    assert entry["uuid"] == "test-uuid-123"
    assert entry["metered"] == 0

  def test_load_skips_malformed_profile_and_keeps_going(self, mocker: MockerFixture):
    """A single profile with a bad metered/hidden value must not abort _load."""
    bad = """\
[connection]
id=Bad
uuid=bad-uuid
type=wifi
metered=yes

[wifi]
ssid=Bad
mode=infrastructure
hidden=maybe
"""
    good = """\
[connection]
id=Good
uuid=good-uuid
type=wifi
metered=0

[wifi]
ssid=Good
mode=infrastructure
"""
    with open(os.path.join(self.tmpdir, "Bad.nmconnection"), "w") as f:
      f.write(bad)
    with open(os.path.join(self.tmpdir, "Good.nmconnection"), "w") as f:
      f.write(good)

    reads = {"Bad.nmconnection": bad, "Good.nmconnection": good}
    mocker.patch("openpilot.system.ui.lib.wifi_network_store.sudo_read",
                 side_effect=lambda p: reads[os.path.basename(p)])
    store = NetworkStore(directory=self.tmpdir)

    assert store.get("Bad") is None
    assert store.get("Good") is not None

  def test_save_uses_canonical_uuid_ssid_filename(self, mocker: MockerFixture):
    """New writes must land at <uuid>-<ssid>.nmconnection, matching netplan's runtime naming."""
    store = self._make_store(mocker)
    mock_run = mocker.patch("subprocess.run")
    store.save_network("MyNet", psk="hunter2")

    install_calls = [c for c in mock_run.call_args_list
                     if len(c.args[0]) >= 2 and c.args[0][:2] == ["sudo", "install"]
                     and "-d" not in c.args[0]]
    assert install_calls, "expected an install call for the keyfile"
    dest = install_calls[0].args[0][-1]
    fname = os.path.basename(dest)
    file_uuid = store.get("MyNet")["uuid"]
    assert fname == f"{file_uuid}-MyNet.nmconnection"

  def test_save_migrates_legacy_filename(self, mocker: MockerFixture):
    """Save replaces a legacy percent-encoded file with the canonical name and rms the old."""
    store = self._make_store(mocker)
    store._networks["MyNet"] = {
      "psk": "old", "metered": 0, "hidden": False, "uuid": "abcd-uuid",
      "_filename": "MyNet.nmconnection",  # legacy percent-encoded
    }
    mock_run = mocker.patch("subprocess.run", return_value=mocker.MagicMock(returncode=0))
    store.save_network("MyNet", psk="new")

    rm_calls = [c for c in mock_run.call_args_list
                if len(c.args[0]) >= 2 and c.args[0][:2] == ["sudo", "rm"]]
    assert any(c.args[0][-1].endswith("MyNet.nmconnection") and "abcd-uuid" not in c.args[0][-1]
               for c in rm_calls), "legacy file must be removed after canonical write"
    assert store.get("MyNet")["_filename"] == "abcd-uuid-MyNet.nmconnection"

  def test_save_migration_cleanup_failure_mirrors_content(self, mocker: MockerFixture):
    """If `sudo rm` of the legacy file fails after the canonical write, both files
    must hold identical content. Otherwise listdir order on next load would pick
    a stale winner non-deterministically."""
    store = self._make_store(mocker)
    store._networks["MyNet"] = {
      "psk": "old", "metered": 0, "hidden": False, "uuid": "abcd-uuid",
      "_filename": "MyNet.nmconnection",
    }
    calls = []

    def fake_run(cmd, *args, **kwargs):
      calls.append(cmd)
      # Fail the legacy-file rm; succeed everything else.
      if cmd[:2] == ["sudo", "rm"]:
        return mocker.MagicMock(returncode=1)
      return mocker.MagicMock(returncode=0)
    mocker.patch("subprocess.run", side_effect=fake_run)

    store.save_network("MyNet", psk="new")

    install_to_old = [c for c in calls
                      if c[:2] == ["sudo", "install"] and len(c) >= 2 and c[-1].endswith("MyNet.nmconnection")
                      and "abcd-uuid" not in c[-1]]
    assert install_to_old, "after rm failure, legacy file must be mirrored with current content"
    # _filename pinned to legacy so future writes keep retrying the cleanup.
    assert store.get("MyNet")["_filename"] == "MyNet.nmconnection"

  def test_save_sanitizes_ssid_in_filename(self, mocker: MockerFixture):
    store = self._make_store(mocker)
    mock_run = mocker.patch("subprocess.run")
    store.save_network("ev/il", psk="x")

    install_calls = [c for c in mock_run.call_args_list
                     if len(c.args[0]) >= 2 and c.args[0][:2] == ["sudo", "install"]
                     and "-d" not in c.args[0]]
    fname = os.path.basename(install_calls[0].args[0][-1])
    file_uuid = store.get("ev/il")["uuid"]
    assert fname == f"{file_uuid}-ev_il.nmconnection"

  def test_load_skips_unsupported_key_mgmt(self, mocker: MockerFixture):
    """Migrated profile with key-mgmt=wpa-eap (no PSK) must NOT load with psk="".
    Otherwise _generate_wpa_conf would emit key_mgmt=NONE for that SSID and the
    device would auto-associate to an open spoof of the same SSID."""
    eap = """\
[connection]
id=Enterprise
uuid=eap-uuid
type=wifi

[wifi]
ssid=Enterprise
mode=infrastructure

[wifi-security]
key-mgmt=wpa-eap
"""
    sae = """\
[connection]
id=WPA3Net
uuid=sae-uuid
type=wifi

[wifi]
ssid=WPA3Net
mode=infrastructure

[wifi-security]
key-mgmt=sae
psk=somepass
"""
    psk = """\
[connection]
id=Home
uuid=psk-uuid
type=wifi

[wifi]
ssid=Home
mode=infrastructure

[wifi-security]
key-mgmt=wpa-psk
psk=hunter2
"""
    files = {"a.nmconnection": eap, "b.nmconnection": sae, "c.nmconnection": psk}
    for fname, content in files.items():
      with open(os.path.join(self.tmpdir, fname), "w") as f:
        f.write(content)
    mocker.patch("openpilot.system.ui.lib.wifi_network_store.sudo_read",
                 side_effect=lambda p: files[os.path.basename(p)])

    store = NetworkStore(directory=self.tmpdir)

    assert store.get("Enterprise") is None, "wpa-eap must be skipped"
    assert store.get("WPA3Net") is None, "sae must be skipped — not driveable"
    assert store.get("Home") is not None, "wpa-psk must load"

  def test_load_skips_wpa_psk_without_inline_secret(self, mocker: MockerFixture):
    """NM agent-managed secrets (psk-flags=1) live outside the keyfile. Loading
    with psk="" would render as key_mgmt=NONE — silent demotion to open profile."""
    content = """\
[connection]
id=AgentSecret
uuid=as-uuid
type=wifi

[wifi]
ssid=AgentSecret
mode=infrastructure

[wifi-security]
key-mgmt=wpa-psk
psk-flags=1
"""
    fpath = os.path.join(self.tmpdir, "AgentSecret.nmconnection")
    with open(fpath, "w") as f:
      f.write(content)
    mocker.patch("openpilot.system.ui.lib.wifi_network_store.sudo_read", return_value=content)

    store = NetworkStore(directory=self.tmpdir)

    assert store.get("AgentSecret") is None, "wpa-psk without inline psk must not load as open"

  def test_load_skips_wep_profiles(self, mocker: MockerFixture):
    """NM stores WEP as key-mgmt=none + wep-key*. Loading with psk='' would render
    as key_mgmt=NONE in wpa_supplicant.conf — silent demotion to open, inviting
    auto-association to an open spoof of the same SSID."""
    content = """\
[connection]
id=OldWEP
uuid=wep-uuid
type=wifi

[wifi]
ssid=OldWEP
mode=infrastructure

[wifi-security]
key-mgmt=none
auth-alg=shared
wep-key0=cafebabe
wep-key-type=1
"""
    fpath = os.path.join(self.tmpdir, "OldWEP.nmconnection")
    with open(fpath, "w") as f:
      f.write(content)
    mocker.patch("openpilot.system.ui.lib.wifi_network_store.sudo_read", return_value=content)

    store = NetworkStore(directory=self.tmpdir)

    assert store.get("OldWEP") is None

  def test_load_skips_autoconnect_disabled(self, mocker: MockerFixture):
    """connection.autoconnect=false is explicit user/provisioning intent. ENABLE_NETWORK
    all would silently re-arm auto-join after upgrade — drop the entry instead."""
    content = """\
[connection]
id=Disabled
uuid=disabled-uuid
type=wifi
autoconnect=false

[wifi]
ssid=Disabled
mode=infrastructure

[wifi-security]
key-mgmt=wpa-psk
psk=hunter2
"""
    fpath = os.path.join(self.tmpdir, "Disabled.nmconnection")
    with open(fpath, "w") as f:
      f.write(content)
    mocker.patch("openpilot.system.ui.lib.wifi_network_store.sudo_read", return_value=content)

    store = NetworkStore(directory=self.tmpdir)

    assert store.get("Disabled") is None

  def test_load_open_network_no_security_section(self, mocker: MockerFixture):
    """An open profile has no [wifi-security] section; loads with empty psk."""
    content = """\
[connection]
id=OpenNet
uuid=open-uuid
type=wifi

[wifi]
ssid=OpenNet
mode=infrastructure
"""
    fpath = os.path.join(self.tmpdir, "OpenNet.nmconnection")
    with open(fpath, "w") as f:
      f.write(content)
    mocker.patch("openpilot.system.ui.lib.wifi_network_store.sudo_read", return_value=content)

    store = NetworkStore(directory=self.tmpdir)

    entry = store.get("OpenNet")
    assert entry is not None
    assert entry["psk"] == ""

  def test_load_skips_ap_mode(self, mocker: MockerFixture):
    content = """\
[connection]
id=Hotspot
uuid=ap-uuid
type=wifi

[wifi]
ssid=Hotspot
mode=ap
"""
    fpath = os.path.join(self.tmpdir, "Hotspot.nmconnection")
    with open(fpath, "w") as f:
      f.write(content)

    mocker.patch("openpilot.system.ui.lib.wifi_network_store.sudo_read", return_value=content)
    store = NetworkStore(directory=self.tmpdir)

    assert store.get("Hotspot") is None


class TestPskFormatting:
  """wpa_supplicant requires 64-hex PSKs unquoted and 8-63 char passphrases
  quoted (hostap config.c:620-694). A quoted 64-char value always FAILs."""

  def test_is_raw_psk_64_hex(self):
    assert _is_raw_psk("0123456789abcdef" * 4) is True

  def test_is_raw_psk_uppercase(self):
    assert _is_raw_psk("0123456789ABCDEF" * 4) is True

  def test_is_raw_psk_63_chars_false(self):
    assert _is_raw_psk("0" * 63) is False

  def test_is_raw_psk_65_chars_false(self):
    assert _is_raw_psk("0" * 65) is False

  def test_is_raw_psk_non_hex_false(self):
    # 64 chars but contains a non-hex char.
    assert _is_raw_psk("z" + "0" * 63) is False

  def test_format_passphrase_quoted(self):
    assert _format_psk_value("hello123") == '"hello123"'

  def test_format_raw_psk_unquoted(self):
    raw = "deadbeef" * 8
    assert _format_psk_value(raw) == raw

  def test_format_quotes_escaped_in_passphrase(self):
    assert _format_psk_value('pa"ss') == '"pa\\"ss"'


class _FakeStore:
  def __init__(self, networks):
    self._networks = networks

  def get_all(self):
    return self._networks


class TestGenerateWpaConf:
  def setup_method(self):
    self.tmpdir = tempfile.mkdtemp()
    self.path = os.path.join(self.tmpdir, "wpa.conf")

  def test_raw_hex_psk_written_unquoted(self):
    raw = "deadbeef" * 8
    store = _FakeStore({"RawNet": {"psk": raw}})
    _generate_wpa_conf(store, path=self.path)
    with open(self.path) as f:
      content = f.read()
    assert f"  psk={raw}" in content
    assert f'  psk="{raw}"' not in content
    assert "key_mgmt=WPA-PSK" in content

  def test_passphrase_written_quoted(self):
    store = _FakeStore({"SecNet": {"psk": "myp@ssw0rd"}})
    _generate_wpa_conf(store, path=self.path)
    with open(self.path) as f:
      content = f.read()
    assert '  psk="myp@ssw0rd"' in content

  def test_open_network_no_psk(self):
    store = _FakeStore({"OpenNet": {"psk": ""}})
    _generate_wpa_conf(store, path=self.path)
    with open(self.path) as f:
      content = f.read()
    assert "key_mgmt=NONE" in content
    assert "psk=" not in content
