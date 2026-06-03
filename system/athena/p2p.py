import base64
import json
import os
import time
from hashlib import sha256
from pathlib import Path

import jwt
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat, load_pem_private_key

from openpilot.common.params import Params


ATHENA_AUTHORIZED_KEYS_PARAM = "AthenadAuthorizedKeys"
ATHENA_ACL_EPOCH_PARAM = "AthenadAuthorizedKeysEpoch"
ATHENA_BOX_PRIVATE_KEY_PARAM = "AthenadBoxPrivateKey"
PARAMS_DIR = Path(os.getenv("PARAMS_DIR", "/data/params/d"))
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ASIUS_DONGLE_ID_LEN = 44
REPLAY_WINDOW_SECONDS = 300
SEEN_NONCES: dict[str, int] = {}


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


def base58_to_bytes(value: str) -> bytes:
  return base58_decode(value).to_bytes(32, "big")


def bytes_to_base58(value: bytes) -> str:
  return base58_encode(int.from_bytes(value, "big")).rjust(ASIUS_DONGLE_ID_LEN, BASE58[0])


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


def get_acl_epoch(params: Params | None = None) -> int:
  try:
    return int(read_raw_param(ATHENA_ACL_EPOCH_PARAM) or "0")
  except ValueError:
    return 0


def bump_acl_epoch(params: Params | None = None) -> int:
  epoch = get_acl_epoch(params) + 1
  write_raw_param(ATHENA_ACL_EPOCH_PARAM, str(epoch))
  return epoch


def get_box_key_pair(params: Params | None = None) -> tuple[str, str]:
  private_key = read_raw_param(ATHENA_BOX_PRIVATE_KEY_PARAM)
  if isinstance(private_key, str) and is_asius_dongle_id(private_key):
    key = x25519.X25519PrivateKey.from_private_bytes(base58_to_bytes(private_key))
  else:
    key = x25519.X25519PrivateKey.generate()
    private_key = bytes_to_base58(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()))
    write_raw_param(ATHENA_BOX_PRIVATE_KEY_PARAM, private_key)

  public_key = bytes_to_base58(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))
  return private_key, public_key


def load_authorized_peers(params: Params | None = None) -> dict[str, dict[str, str]]:
  try:
    peers = read_raw_param(ATHENA_AUTHORIZED_KEYS_PARAM)
    raw_peers = peers if isinstance(peers, dict) else json.loads(peers or "{}")
    return {
      public_key: peer
      for public_key, peer in raw_peers.items()
      if is_asius_dongle_id(public_key) and is_asius_dongle_id(peer.get("boxPublicKey"))
    }
  except Exception:
    return {}


def save_authorized_peers(peers: dict[str, dict[str, str]], params: Params | None = None) -> None:
  write_raw_param(ATHENA_AUTHORIZED_KEYS_PARAM, json.dumps(peers))


def authorize_peer(public_key: str, box_public_key: str, params: Params | None = None) -> dict[str, str]:
  if not is_asius_dongle_id(public_key) or not is_asius_dongle_id(box_public_key):
    raise ValueError("invalid Athena peer key")

  params = params or Params()
  peers = load_authorized_peers(params)
  peer = {"publicKey": public_key, "boxPublicKey": box_public_key}
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


def verify_identity_signature(public_key: str, signature: str, data: bytes) -> bool:
  try:
    ed25519.Ed25519PublicKey.from_public_bytes(base58_to_bytes(public_key)).verify(base64url_decode(signature), data)
    return True
  except Exception:
    return False


def check_replay(public_key: str, nonce: str, ts: int) -> bool:
  now = int(time.time())
  for key, seen_ts in list(SEEN_NONCES.items()):
    if seen_ts + REPLAY_WINDOW_SECONDS < now:
      del SEEN_NONCES[key]

  if abs(now - int(ts)) > REPLAY_WINDOW_SECONDS:
    return False

  key = f"{public_key}:{nonce}"
  if key in SEEN_NONCES:
    return False
  SEEN_NONCES[key] = int(ts)
  return True


def payload_key(shared: bytes, sender: str, recipient: str, ephemeral_public_key: str) -> bytes:
  return sha256(f"athena-v2:{base64url_encode(shared)}:{sender}:{recipient}:{ephemeral_public_key}".encode()).digest()


def encrypt_payload(text: str, sender: str, recipient: str, recipient_box_public_key: str) -> str:
  ephemeral = x25519.X25519PrivateKey.generate()
  ephemeral_public_key = bytes_to_base58(ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))
  shared = ephemeral.exchange(x25519.X25519PublicKey.from_public_bytes(base58_to_bytes(recipient_box_public_key)))
  iv = os.urandom(12)
  envelope = {
    "v": 2,
    "alg": "X25519-A256GCM-Ed25519",
    "from": sender,
    "to": recipient,
    "eph": ephemeral_public_key,
    "iv": base64url_encode(iv),
    "ts": int(time.time()),
    "nonce": base64url_encode(os.urandom(16)),
  }
  aad = stable_json(envelope).encode()
  ciphertext = AESGCM(payload_key(shared, sender, recipient, ephemeral_public_key)).encrypt(iv, text.encode(), aad)
  signed = {**envelope, "ciphertext": base64url_encode(ciphertext)}
  signature = identity_private_key().sign(stable_json(signed).encode())
  return json.dumps({**signed, "sig": base64url_encode(signature)})


def verify_pair_token(token: str | None, recipient: str) -> bool:
  try:
    if token is None:
      return False
    unverified = jwt.decode(token, options={"verify_signature": False})
    if unverified.get("identity") != recipient:
      return False
    public_key = ed25519.Ed25519PublicKey.from_public_bytes(base58_to_bytes(recipient))
    verified = jwt.decode(token, public_key, algorithms=["EdDSA"])
    _, box_public_key = get_box_key_pair()
    return verified.get("type") == "pair" and verified.get("boxPublicKey") == box_public_key
  except Exception:
    return False


def decrypt_payload(payload: str, sender: str, recipient: str) -> str | None:
  try:
    encrypted = json.loads(payload)
    if encrypted.get("v") == 2 and encrypted.get("alg") == "X25519-A256GCM-Ed25519":
      if encrypted.get("from") != sender or encrypted.get("to") != recipient:
        return None

      signature = encrypted["sig"]
      signed = {key: value for key, value in encrypted.items() if key != "sig"}
      if not verify_identity_signature(sender, signature, stable_json(signed).encode()):
        return None

      private_key, _ = get_box_key_pair()
      shared = x25519.X25519PrivateKey.from_private_bytes(base58_to_bytes(private_key)).exchange(
        x25519.X25519PublicKey.from_public_bytes(base58_to_bytes(encrypted["eph"]))
      )
      aad = stable_json({key: value for key, value in signed.items() if key != "ciphertext"}).encode()
      plaintext = AESGCM(payload_key(shared, sender, recipient, encrypted["eph"])).decrypt(
        base64url_decode(encrypted["iv"]), base64url_decode(encrypted["ciphertext"]), aad
      )
      if not check_replay(sender, encrypted["nonce"], int(encrypted["ts"])):
        return None
      return plaintext.decode()

    return None
  except Exception:
    return None
