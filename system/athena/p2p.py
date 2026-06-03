import base64
import json
import os
import time
from hashlib import sha256, sha512
from pathlib import Path

import jwt
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat, load_pem_private_key

from openpilot.common.params import Params


ATHENA_AUTHORIZED_KEYS_PARAM = "AthenadAuthorizedKeys"
ATHENA_ACL_EPOCH_PARAM = "AthenadAuthorizedKeysEpoch"
GITHUB_SSH_KEYS_PARAM = "GithubSshKeys"
PARAMS_DIR = Path(os.getenv("PARAMS_DIR", "/data/params/d"))
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
KEY_PREFIX = {
  "ed25519": "e",
}
KEY_BYTES = {
  "ed25519": 32,
}

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


def base58_to_bytes(value: str, algorithm: str = "ed25519") -> bytes:
  prefix = KEY_PREFIX[algorithm]
  if not value.startswith(prefix):
    raise ValueError(f"expected {algorithm} key prefix {prefix}")
  body = value[len(prefix):]
  if not body:
    raise ValueError("missing base58 key body")
  decoded = base58_decode(body)
  byte_length = KEY_BYTES[algorithm]
  if decoded >= 1 << (byte_length * 8):
    raise ValueError("invalid base58 value for byte length")
  return decoded.to_bytes(byte_length, "big")


def bytes_to_base58(value: bytes, algorithm: str = "ed25519") -> str:
  byte_length = KEY_BYTES[algorithm]
  if len(value) != byte_length:
    raise ValueError(f"expected {byte_length} {algorithm} bytes")
  return KEY_PREFIX[algorithm] + base58_encode(int.from_bytes(value, "big"))


def is_asius_dongle_id(dongle_id: str | None) -> bool:
  if dongle_id is None:
    return False
  try:
    base58_to_bytes(dongle_id, "ed25519")
    return True
  except Exception:
    return False


def random_base58_secret() -> str:
  return bytes_to_base58(os.urandom(32), "ed25519")


def read_ssh_string(data: bytes, offset: int) -> tuple[bytes, int]:
  length = int.from_bytes(data[offset:offset + 4], "big")
  start = offset + 4
  end = start + length
  return data[start:end], end


def ssh_public_key_to_identity(key: str) -> str | None:
  try:
    key_type, encoded, *_ = key.split()
    if key_type != "ssh-ed25519":
      return None

    data = base64.b64decode(encoded)
    parsed_type, offset = read_ssh_string(data, 0)
    if parsed_type != b"ssh-ed25519":
      return None

    public_key, _ = read_ssh_string(data, offset)
    return bytes_to_base58(public_key, "ed25519")
  except Exception:
    return None


def load_github_ssh_peers() -> dict[str, dict[str, str]]:
  keys = read_raw_param(GITHUB_SSH_KEYS_PARAM) or ""
  peers = {}
  for key in keys.splitlines():
    public_key = ssh_public_key_to_identity(key)
    if public_key is not None:
      peers[public_key] = {"publicKey": public_key}
  return peers


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


def get_acl_epoch(params: Params | None = None) -> int:
  try:
    return int(read_raw_param(ATHENA_ACL_EPOCH_PARAM) or "0")
  except ValueError:
    return 0


def bump_acl_epoch(params: Params | None = None) -> int:
  epoch = get_acl_epoch(params) + 1
  write_raw_param(ATHENA_ACL_EPOCH_PARAM, str(epoch))
  return epoch


def load_stored_authorized_peers(params: Params | None = None) -> dict[str, dict[str, str]]:
  try:
    raw = read_raw_param(ATHENA_AUTHORIZED_KEYS_PARAM)
    raw_peers = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    return {
      public_key: {"publicKey": public_key, **({"aclEpoch": peer["aclEpoch"]} if isinstance(peer, dict) and "aclEpoch" in peer else {})}
      for public_key, peer in raw_peers.items()
      if is_asius_dongle_id(public_key)
    }
  except Exception:
    return {}


def load_authorized_peers(params: Params | None = None) -> dict[str, dict[str, str]]:
  peers = load_github_ssh_peers()
  peers.update(load_stored_authorized_peers(params))
  return peers


def save_authorized_peers(peers: dict[str, dict[str, str]], params: Params | None = None) -> None:
  write_raw_param(ATHENA_AUTHORIZED_KEYS_PARAM, json.dumps(peers))


def authorize_peer(public_key: str, params: Params | None = None) -> dict[str, str]:
  if not is_asius_dongle_id(public_key):
    raise ValueError("invalid Athena peer key")

  params = params or Params()
  peers = load_stored_authorized_peers(params)
  peer = {"publicKey": public_key}
  peer["aclEpoch"] = str(bump_acl_epoch(params))
  peers[public_key] = peer
  save_authorized_peers(peers, params)

  return peer


def stable_json(value) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def identity_private_key() -> ed25519.Ed25519PrivateKey:
  from openpilot.common.api import get_key_pair
  _, private_key, _ = get_key_pair()
  key = load_pem_private_key(private_key.encode(), password=None)
  if not isinstance(key, ed25519.Ed25519PrivateKey):
    raise ValueError("Athena identity key is not Ed25519")
  return key


def sign_jwt(payload: dict, expiry_seconds: int) -> str:
  now = int(time.time())
  return jwt.encode({**payload, "iat": now, "nbf": now, "exp": now + expiry_seconds}, identity_private_key(), algorithm="EdDSA")


def pairing_token(recipient: str, acl_epoch: int, expiry_seconds: int = 300) -> str:
  return sign_jwt({"type": "pair", "to": recipient, "aclEpoch": acl_epoch}, expiry_seconds)


def verify_identity_signature(public_key: str, signature: str, data: bytes) -> bool:
  try:
    ed25519.Ed25519PublicKey.from_public_bytes(base58_to_bytes(public_key, "ed25519")).verify(base64url_decode(signature), data)
    return True
  except Exception:
    return False


def x25519_private_from_identity() -> x25519.X25519PrivateKey:
  raw = identity_private_key().private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
  digest = bytearray(sha512(raw).digest()[:32])
  digest[0] &= 248
  digest[31] &= 127
  digest[31] |= 64
  return x25519.X25519PrivateKey.from_private_bytes(bytes(digest))


def x25519_public_from_identity(public_key: str) -> x25519.X25519PublicKey:
  p = 2**255 - 19
  raw = bytearray(base58_to_bytes(public_key, "ed25519"))
  raw[31] &= 0x7f
  y = int.from_bytes(raw, "little")
  u = ((1 + y) * pow(1 - y, p - 2, p)) % p
  return x25519.X25519PublicKey.from_public_bytes(u.to_bytes(32, "little"))


def payload_key(shared: bytes, sender: str, recipient: str) -> bytes:
  return sha256(f"athena-v3:{base64url_encode(shared)}:{sender}:{recipient}".encode()).digest()


def encrypt_payload(text: str, sender: str, recipient: str) -> str:
  shared = x25519_private_from_identity().exchange(x25519_public_from_identity(recipient))
  iv = os.urandom(12)
  envelope = {
    "v": 3,
    "alg": "Ed25519-X25519-A256GCM",
    "from": sender,
    "to": recipient,
    "iv": base64url_encode(iv),
    "ts": int(time.time()),
  }
  aad = stable_json(envelope).encode()
  ciphertext = AESGCM(payload_key(shared, sender, recipient)).encrypt(iv, text.encode(), aad)
  signed = {**envelope, "ciphertext": base64url_encode(ciphertext)}
  signature = identity_private_key().sign(stable_json(signed).encode())
  return json.dumps({**signed, "sig": base64url_encode(signature)})


def verify_pair_token(token: str | None, recipient: str) -> bool:
  try:
    if token is None:
      return False
    unverified = jwt.decode(token, options={"verify_signature": False})
    if unverified.get("to") != recipient:
      return False
    public_key = ed25519.Ed25519PublicKey.from_public_bytes(base58_to_bytes(recipient, "ed25519"))
    verified = jwt.decode(token, public_key, algorithms=["EdDSA"])
    return verified.get("type") == "pair"
  except Exception:
    return False


def decrypt_payload(payload: str, sender: str, recipient: str) -> str | None:
  try:
    encrypted = json.loads(payload)
    if encrypted.get("v") == 3 and encrypted.get("alg") == "Ed25519-X25519-A256GCM":
      if encrypted.get("from") != sender or encrypted.get("to") != recipient:
        return None

      signature = encrypted["sig"]
      signed = {key: value for key, value in encrypted.items() if key != "sig"}
      if not verify_identity_signature(sender, signature, stable_json(signed).encode()):
        return None

      shared = x25519_private_from_identity().exchange(x25519_public_from_identity(sender))
      aad = stable_json({key: value for key, value in signed.items() if key != "ciphertext"}).encode()
      plaintext = AESGCM(payload_key(shared, sender, recipient)).decrypt(
        base64url_decode(encrypted["iv"]), base64url_decode(encrypted["ciphertext"]), aad
      )
      return plaintext.decode()

    return None
  except Exception:
    return None
