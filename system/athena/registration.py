#!/usr/bin/env python3
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from openpilot.common.api import get_key_pair
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.hardware import PC
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.system.athena.identity import dongle_id_from_public_key


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"


def register(show_spinner=False) -> str | None:
  params = Params()
  dongle_id: str | None = params.get("DongleId")

  try:
    _, _, public_key = get_key_pair()
    if public_key is None:
      private_key_path = Path(Paths.persist_root()) / "comma" / "id_ed25519"
      public_key_path = private_key_path.with_suffix(".pub")
      private_key_path.parent.mkdir(parents=True, exist_ok=True)

      private_key = ed25519.Ed25519PrivateKey.generate()
      private_key_path.write_bytes(private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
      private_key_path.chmod(0o600)

      public_key = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
      public_key_path.write_bytes(public_key)
      public_key = public_key.decode()

    expected_dongle_id = dongle_id_from_public_key(public_key)
    if dongle_id != expected_dongle_id:
      dongle_id = expected_dongle_id
  except Exception:
    dongle_id = UNREGISTERED_DONGLE_ID
    cloudlog.exception("failed to create Ed25519 identity")

  if dongle_id:
    params.put("DongleId", dongle_id, block=True)
    set_offroad_alert("Offroad_UnregisteredHardware", (dongle_id == UNREGISTERED_DONGLE_ID) and not PC)
  return dongle_id


if __name__ == "__main__":
  print(register())
