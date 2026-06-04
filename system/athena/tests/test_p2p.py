import json

from openpilot.system.athena import p2p


APP_KEY = "D6xksRG9VaWxAesrqRjb9NePxwhrBLi72SSJyJqahPtw"
GITHUB_KEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC307aE+nuHzTAgaJhzSf5v7ZZQW9gaper private"


def test_identity_to_ssh_public_key_round_trip():
  ssh_key = p2p.identity_to_ssh_public_key(APP_KEY)

  assert p2p.ssh_public_key_to_identity(ssh_key) == APP_KEY
  assert p2p.is_generated_ssh_key(ssh_key)


def test_ed25519_base58_keys_are_fixed_width():
  key = p2p.bytes_to_base58(b"\x00" * 31 + b"\x01")

  assert len(key) == 44
  assert p2p.base58_to_bytes(key) == b"\x00" * 31 + b"\x01"
  assert not p2p.is_asius_dongle_id(key[1:])


def test_sync_ssh_keys_preserves_user_keys_and_tracks_authorized_peers(tmp_path, monkeypatch):
  monkeypatch.setattr(p2p, "PARAMS_DIR", tmp_path)
  other_app_key = p2p.bytes_to_base58(b"\x02" * 32)
  stale_generated_key = p2p.identity_to_ssh_public_key(other_app_key)
  legacy_generated_key = stale_generated_key.rsplit(" ", 1)[0] + " asius-app"
  p2p.write_raw_param(p2p.GITHUB_SSH_KEYS_PARAM, f"\n{GITHUB_KEY}\n{stale_generated_key}\n{legacy_generated_key}\n{GITHUB_KEY}\n")
  p2p.write_raw_param(p2p.ATHENA_AUTHORIZED_KEYS_PARAM, json.dumps({APP_KEY: {"publicKey": APP_KEY}}))

  keys = p2p.sync_ssh_keys()

  assert keys.splitlines() == [GITHUB_KEY, p2p.identity_to_ssh_public_key(APP_KEY)]
  assert p2p.load_github_ssh_peers() == {}
  assert p2p.load_authorized_peers() == {APP_KEY: {"publicKey": APP_KEY}}

  p2p.write_raw_param(p2p.ATHENA_AUTHORIZED_KEYS_PARAM, "{}")
  keys = p2p.sync_ssh_keys()

  assert keys == GITHUB_KEY


def test_authorized_peer_metadata(tmp_path, monkeypatch):
  monkeypatch.setattr(p2p, "PARAMS_DIR", tmp_path)
  monkeypatch.setattr(p2p, "wall_time", lambda: 1_234)

  peer = p2p.authorize_peer(APP_KEY, label="Karel phone")

  assert peer["publicKey"] == APP_KEY
  assert peer["label"] == "Karel phone"
  assert peer["createdAt"] == 1_234
  assert p2p.load_stored_authorized_peers()[APP_KEY]["label"] == "Karel phone"


def test_payload_timestamp_valid_rejects_old_messages(monkeypatch):
  monkeypatch.setattr(p2p, "wall_time", lambda: 1_000)

  assert p2p.payload_timestamp_valid(1_000)
  assert p2p.payload_timestamp_valid(941)
  assert not p2p.payload_timestamp_valid(939)
  assert not p2p.payload_timestamp_valid("1000")


def test_pairing_mode_window(tmp_path, monkeypatch):
  monkeypatch.setattr(p2p, "PARAMS_DIR", tmp_path)
  now = 1_000
  monkeypatch.setattr(p2p, "wall_time", lambda: now)

  assert not p2p.pairing_mode_active()
  assert p2p.enable_pairing_mode(duration_seconds=180) == 1_180
  assert p2p.pairing_mode_active()

  now = 1_181
  assert not p2p.pairing_mode_active()
