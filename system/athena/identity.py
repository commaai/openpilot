from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_public_key

BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ED25519_KEY_BYTES = 32
DONGLE_ID_LEN = 44


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


def identity_to_bytes(value: str) -> bytes:
  if len(value) != DONGLE_ID_LEN or not all(char in BASE58 for char in value):
    raise ValueError("invalid identity")

  decoded = base58_decode(value)
  if decoded >= 1 << (ED25519_KEY_BYTES * 8):
    raise ValueError("invalid identity")
  return decoded.to_bytes(ED25519_KEY_BYTES, "big")


def bytes_to_identity(value: bytes) -> str:
  if len(value) != ED25519_KEY_BYTES:
    raise ValueError("identity requires a 32-byte Ed25519 public key")
  return base58_encode(int.from_bytes(value, "big")).rjust(DONGLE_ID_LEN, BASE58[0])


def is_dongle_id(dongle_id: str | None) -> bool:
  if dongle_id is None:
    return False
  try:
    identity_to_bytes(dongle_id)
    return True
  except Exception:
    return False


def dongle_id_from_public_key(public_key: str) -> str:
  key = load_pem_public_key(public_key.encode())
  if not isinstance(key, ed25519.Ed25519PublicKey):
    raise ValueError("identity requires an Ed25519 public key")

  public_bytes = key.public_bytes(Encoding.Raw, PublicFormat.Raw)
  return bytes_to_identity(public_bytes)


def public_key_from_dongle_id(dongle_id: str) -> str:
  if not is_dongle_id(dongle_id):
    raise ValueError("invalid DongleId")

  key = ed25519.Ed25519PublicKey.from_public_bytes(identity_to_bytes(dongle_id))
  return key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
