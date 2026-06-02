import base64
import json
import os
from hashlib import sha256
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from openpilot.common.params import Params


ATHENA_AUTHORIZED_KEYS_PARAM = "AthenadAuthorizedKeys"
ATHENA_INBOX_SECRET_PARAM = "AthenadInboxSecret"
PARAMS_DIR = Path(os.getenv("PARAMS_DIR", "/data/params/d"))
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ASIUS_DONGLE_ID_LEN = 44


def base64url_encode(data: bytes) -> str:
  return base64.urlsafe_b64encode(data).decode().rstrip("=")


def base64url_decode(data: str) -> bytes:
  return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def base58_encode(value: int) -> str:
  if value == 0:
    return BASE58[0]

  out = []
  while value:
    value, rem = divmod(value, 58)
    out.append(BASE58[rem])
  return "".join(reversed(out))


def base58_decode(value: str) -> int:
  decoded = 0
  for char in value:
    decoded *= 58
    decoded += BASE58.index(char)
  return decoded


def is_asius_dongle_id(dongle_id: str | None) -> bool:
  return dongle_id is not None and len(dongle_id) == ASIUS_DONGLE_ID_LEN and all(char in BASE58 for char in dongle_id)


def random_base58_secret() -> str:
  return base58_encode(int.from_bytes(os.urandom(32), "big")).rjust(ASIUS_DONGLE_ID_LEN, BASE58[0])


def read_raw_param(key: str) -> str | None:
  try:
    return (PARAMS_DIR / key).read_text()
  except OSError:
    return None


def write_raw_param(key: str, value: str) -> None:
  PARAMS_DIR.mkdir(parents=True, exist_ok=True)
  path = PARAMS_DIR / key
  tmp_path = PARAMS_DIR / f".{key}.tmp"
  tmp_path.write_text(value)
  os.replace(tmp_path, path)


def get_inbox_secret(params: Params | None = None) -> str:
  secret = read_raw_param(ATHENA_INBOX_SECRET_PARAM)
  if isinstance(secret, str) and is_asius_dongle_id(secret):
    return secret

  secret = random_base58_secret()
  write_raw_param(ATHENA_INBOX_SECRET_PARAM, secret)
  return secret


def load_authorized_peers(params: Params | None = None) -> dict[str, dict[str, str]]:
  try:
    peers = read_raw_param(ATHENA_AUTHORIZED_KEYS_PARAM)
    return peers if isinstance(peers, dict) else json.loads(peers or "{}")
  except Exception:
    return {}


def save_authorized_peers(peers: dict[str, dict[str, str]], params: Params | None = None) -> None:
  write_raw_param(ATHENA_AUTHORIZED_KEYS_PARAM, json.dumps(peers))


def ssh_key_from_public_key(public_key: str) -> str:
  raw = base58_decode(public_key).to_bytes(32, "big")
  return f"ssh-ed25519 {base64.b64encode(raw).decode()} athena-{public_key}"


def authorize_peer(public_key: str, inbox_secret: str, params: Params | None = None) -> dict[str, str]:
  if not is_asius_dongle_id(public_key) or not is_asius_dongle_id(inbox_secret):
    raise ValueError("invalid Athena peer key")

  params = params or Params()
  peers = load_authorized_peers(params)
  peer = {"publicKey": public_key, "inboxSecret": inbox_secret}
  peers[public_key] = peer
  save_authorized_peers(peers, params)

  ssh_key = ssh_key_from_public_key(public_key)
  ssh_keys = params.get("GithubSshKeys") or ""
  if ssh_key not in ssh_keys.splitlines():
    params.put("GithubSshKeys", "\n".join([line for line in ssh_keys.splitlines() if line] + [ssh_key]), block=True)

  return peer


def encryption_key(public_key: str, inbox_secret: str) -> bytes:
  return sha256(f"athena-v1:{public_key}:{inbox_secret}".encode()).digest()


def encrypt_payload(text: str, public_key: str, inbox_secret: str) -> str:
  iv = os.urandom(12)
  ciphertext = AESGCM(encryption_key(public_key, inbox_secret)).encrypt(iv, text.encode(), None)
  return json.dumps({"v": 1, "alg": "A256GCM", "iv": base64url_encode(iv), "ciphertext": base64url_encode(ciphertext)})


def decrypt_payload(payload: str, public_key: str, inbox_secret: str) -> str | None:
  try:
    encrypted = json.loads(payload)
    if encrypted.get("v") != 1 or encrypted.get("alg") != "A256GCM":
      return None
    plaintext = AESGCM(encryption_key(public_key, inbox_secret)).decrypt(
      base64url_decode(encrypted["iv"]), base64url_decode(encrypted["ciphertext"]), None
    )
    return plaintext.decode()
  except Exception:
    return None
