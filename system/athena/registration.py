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
from openpilot.system.athena.identity import bytes_to_identity, identity_to_bytes, is_dongle_id


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"
FALLBACK_IDENTITY_DIR = Path("/data/persist/comma")


def is_registered_device() -> bool:
  dongle = Params().get("DongleId")
  return dongle not in (None, UNREGISTERED_DONGLE_ID)


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


def ed25519_key_paths(identity_dir: Path | None = None) -> tuple[Path, Path]:
  identity_dir = identity_dir or Path(Paths.persist_root()) / "comma"
  return identity_dir / "id_ed25519", identity_dir / "id_ed25519.pub"


def remount_persist(readwrite: bool) -> None:
  if Path(Paths.persist_root()) != Path("/persist"):
    return

  mode = "rw" if readwrite else "ro"
  subprocess.run(["sudo", "-n", "mount", "-o", f"remount,{mode}", "/persist"], check=True)


def create_ed25519_key_pair(identity_dir: Path | None = None) -> str:
  private_key_path, public_key_path = ed25519_key_paths(identity_dir)
  private_key_path.parent.mkdir(parents=True, exist_ok=True)

  private_key = ed25519.Ed25519PrivateKey.generate()
  private_key_path.write_bytes(private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
  private_key_path.chmod(0o600)

  public_key = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
  public_key_path.write_bytes(public_key)
  return public_key.decode()


def prepare_fallback_identity_dir(identity_dir: Path = FALLBACK_IDENTITY_DIR) -> None:
  try:
    identity_dir.mkdir(parents=True, exist_ok=True)
  except PermissionError:
    subprocess.run(["sudo", "-n", "mkdir", "-p", str(identity_dir)], check=True)
  if not PC:
    subprocess.run(["sudo", "-n", "chown", "comma:comma", str(identity_dir)], check=True)
  identity_dir.chmod(0o700)


def ensure_ed25519_key_pair() -> str:
  private_key_path, public_key_path = ed25519_key_paths()
  if private_key_path.exists() and public_key_path.exists():
    return public_key_path.read_text()

  fallback_private_key_path, fallback_public_key_path = ed25519_key_paths(FALLBACK_IDENTITY_DIR)
  if fallback_private_key_path.exists() and fallback_public_key_path.exists():
    return fallback_public_key_path.read_text()

  remounted = False
  try:
    remount_persist(True)
    remounted = True
    return create_ed25519_key_pair()
  except OSError:
    prepare_fallback_identity_dir()
    return create_ed25519_key_pair(FALLBACK_IDENTITY_DIR)
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
    cloudlog.exception("failed to create Ed25519 identity")
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
