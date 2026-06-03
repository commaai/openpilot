import json

from openpilot.system.athena import p2p


APP_KEY = "D6xksRG9VaWxAesrqRjb9NePxwhrBLi72SSJyJqahPtw"
GITHUB_KEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC307aE+nuHzTAgaJhzSf5v7ZZQW9gaper private"


def test_identity_to_ssh_public_key_round_trip():
  ssh_key = p2p.identity_to_ssh_public_key(APP_KEY)

  assert p2p.ssh_public_key_to_identity(ssh_key) == APP_KEY
  assert p2p.is_generated_ssh_key(ssh_key)


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

  p2p.write_raw_param(p2p.ATHENA_AUTHORIZED_KEYS_PARAM, "{}")
  keys = p2p.sync_ssh_keys()

  assert keys == GITHUB_KEY
