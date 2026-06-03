#!/usr/bin/env python3
from pathlib import Path
import subprocess

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat, load_pem_public_key

from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.hardware import PC
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ED25519_PREFIX = "e"
ED25519_KEY_BYTES = 32


def is_registered_device() -> bool:
  dongle = Params().get("DongleId")
  return dongle not in (None, UNREGISTERED_DONGLE_ID)


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


def bytes_to_dongle_id(value: bytes) -> str:
  if len(value) != ED25519_KEY_BYTES:
    raise ValueError("Asius identity requires a 32-byte Ed25519 public key")
  return ED25519_PREFIX + base58_encode(int.from_bytes(value, "big"))


def dongle_id_to_bytes(dongle_id: str) -> bytes:
  if not dongle_id.startswith(ED25519_PREFIX):
    raise ValueError("invalid Asius DongleId")

  body = dongle_id[len(ED25519_PREFIX):]
  if not body or not all(char in BASE58 for char in body):
    raise ValueError("invalid Asius DongleId")

  encoded = base58_decode(body)
  if encoded >= 1 << (ED25519_KEY_BYTES * 8):
    raise ValueError("invalid Asius DongleId")
  return encoded.to_bytes(ED25519_KEY_BYTES, "big")


def is_asius_dongle_id(dongle_id: str | None) -> bool:
  if dongle_id is None:
    return False
  try:
    dongle_id_to_bytes(dongle_id)
    return True
  except Exception:
    return False


def dongle_id_from_public_key(public_key: str) -> str:
  key = load_pem_public_key(public_key.encode())
  if not isinstance(key, ed25519.Ed25519PublicKey):
    raise ValueError("Asius identity requires an Ed25519 public key")

  public_bytes = key.public_bytes(Encoding.Raw, PublicFormat.Raw)
  return bytes_to_dongle_id(public_bytes)


def public_key_from_dongle_id(dongle_id: str) -> str:
  if not is_asius_dongle_id(dongle_id):
    raise ValueError("invalid Asius DongleId")

  key = ed25519.Ed25519PublicKey.from_public_bytes(dongle_id_to_bytes(dongle_id))
  return key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()


def ed25519_key_paths(identity_dir: Path | None = None) -> tuple[Path, Path]:
  identity_dir = identity_dir or Path(Paths.persist_root()) / "comma"
  return identity_dir / "id_ed25519", identity_dir / "id_ed25519.pub"


def remount_persist(readwrite: bool) -> None:
  if Path(Paths.persist_root()) != Path("/persist"):
    return

  mode = "rw" if readwrite else "ro"
  subprocess.run(["sudo", "-n", "mount", "-o", f"remount,{mode}", "/persist"], check=True)


def create_ed25519_key_pair() -> str:
  private_key_path, public_key_path = ed25519_key_paths()
  private_key_path.parent.mkdir(parents=True, exist_ok=True)

  private_key = ed25519.Ed25519PrivateKey.generate()
  private_key_path.write_bytes(private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
  private_key_path.chmod(0o600)

  public_key = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
  public_key_path.write_bytes(public_key)
  return public_key.decode()


def ensure_ed25519_key_pair() -> str:
  private_key_path, public_key_path = ed25519_key_paths()
  if private_key_path.exists() and public_key_path.exists():
    return public_key_path.read_text()

  remounted = False
  try:
    remount_persist(True)
    remounted = True
    return create_ed25519_key_pair()
  finally:
    if remounted:
      try:
        remount_persist(False)
      except Exception:
        cloudlog.exception("failed to remount /persist read-only after Ed25519 key creation")


def register(show_spinner=False) -> str | None:
  params = Params()
  dongle_id: str | None = params.get("DongleId")

  try:
    public_key = ensure_ed25519_key_pair()
  except Exception:
    dongle_id = UNREGISTERED_DONGLE_ID
    cloudlog.exception("failed to create Asius Ed25519 identity")
  else:
    expected_dongle_id = dongle_id_from_public_key(public_key)
    if dongle_id != expected_dongle_id:
      dongle_id = expected_dongle_id

  if dongle_id:
    params.put("DongleId", dongle_id, block=True)
    set_offroad_alert("Offroad_UnregisteredHardware", (dongle_id == UNREGISTERED_DONGLE_ID) and not PC)
  return dongle_id


if __name__ == "__main__":
  print(register())
