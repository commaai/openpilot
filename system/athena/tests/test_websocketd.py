import json

from openpilot.system.athena import websocketd
from openpilot.system.athena.identity import bytes_to_identity, identity_to_bytes, is_dongle_id


APP_KEY = "D6xksRG9VaWxAesrqRjb9NePxwhrBLi72SSJyJqahPtw"
GITHUB_KEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC307aE+nuHzTAgaJhzSf5v7ZZQW9gaper private"


def test_identity_to_ssh_public_key_round_trip():
  ssh_key = websocketd.identity_to_ssh_public_key(APP_KEY)

  assert websocketd.ssh_public_key_to_identity(ssh_key) == APP_KEY
  assert websocketd.is_generated_ssh_key(ssh_key)


def test_ed25519_base58_keys_are_fixed_width():
  key = bytes_to_identity(b"\x00" * 31 + b"\x01")

  assert len(key) == 44
  assert identity_to_bytes(key) == b"\x00" * 31 + b"\x01"
  assert not is_dongle_id(key[1:])


def test_sync_ssh_keys_preserves_user_keys_and_tracks_authorized_peers(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  other_app_key = bytes_to_identity(b"\x02" * 32)
  stale_generated_key = websocketd.identity_to_ssh_public_key(other_app_key)
  legacy_generated_key = stale_generated_key.rsplit(" ", 1)[0] + " asius-app"
  websocketd.write_raw_param(websocketd.GITHUB_SSH_KEYS_PARAM, f"\n{GITHUB_KEY}\n{stale_generated_key}\n{legacy_generated_key}\n{GITHUB_KEY}\n")
  websocketd.write_raw_param(websocketd.ATHENA_AUTHORIZED_KEYS_PARAM, json.dumps({APP_KEY: {"publicKey": APP_KEY}}))

  keys = websocketd.sync_ssh_keys()

  assert keys.splitlines() == [GITHUB_KEY, websocketd.identity_to_ssh_public_key(APP_KEY)]
  assert websocketd.load_authorized_peers() == {APP_KEY: {"publicKey": APP_KEY}}

  websocketd.write_raw_param(websocketd.ATHENA_AUTHORIZED_KEYS_PARAM, "{}")
  keys = websocketd.sync_ssh_keys()

  assert keys == GITHUB_KEY


def test_authorized_peer_metadata(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_234)

  peer = websocketd.authorize_peer(APP_KEY, label="Karel phone")

  assert peer["publicKey"] == APP_KEY
  assert peer["label"] == "Karel phone"
  assert peer["createdAt"] == 1_234
  assert websocketd.load_authorized_peers()[APP_KEY]["label"] == "Karel phone"


def test_payload_timestamp_valid_rejects_old_messages(monkeypatch):
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_000)

  assert websocketd.payload_timestamp_valid(1_000)
  assert websocketd.payload_timestamp_valid(941)
  assert not websocketd.payload_timestamp_valid(939)
  assert not websocketd.payload_timestamp_valid("1000")


def test_pairing_mode_window(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  now = 1_000
  monkeypatch.setattr(websocketd, "wall_time", lambda: now)

  assert not websocketd.pairing_mode_active()
  assert websocketd.enable_pairing_mode(duration_seconds=180) == 1_180
  assert websocketd.pairing_mode_active()

  now = 1_181
  assert not websocketd.pairing_mode_active()
