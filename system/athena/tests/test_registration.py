from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from openpilot.common.params import Params
from openpilot.system.athena.identity import (
  DONGLE_ID_LEN,
  dongle_id_from_public_key,
  public_key_from_dongle_id,
)
from openpilot.system.athena.registration import (
  register,
  UNREGISTERED_DONGLE_ID,
)
from openpilot.system.hardware.hw import Paths


class TestRegistration:

  @pytest.fixture(autouse=True)
  def setup_identity_root(self, tmp_path, monkeypatch):
    self.params = Params()

    persist_root = tmp_path / "persist"
    monkeypatch.setattr(Paths, "persist_root", staticmethod(lambda: str(persist_root)))

    persist_dir = Path(Paths.persist_root()) / "comma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    self.priv_key = persist_dir / "id_ed25519"
    self.pub_key = persist_dir / "id_ed25519.pub"

  def _generate_keys(self) -> str:
    key = ed25519.Ed25519PrivateKey.generate()
    self.priv_key.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    self.pub_key.write_bytes(key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    return self.pub_key.read_text()

  def test_public_key_roundtrip(self):
    public_key = self._generate_keys()
    dongle = dongle_id_from_public_key(public_key)

    assert len(dongle) == DONGLE_ID_LEN
    assert public_key_from_dongle_id(dongle) == public_key

  def test_valid_cache(self):
    public_key = self._generate_keys()
    dongle = dongle_id_from_public_key(public_key)

    self.params.put("DongleId", dongle, block=True)
    assert register() == dongle

  def test_creates_missing_ed25519_keys(self):
    dongle = register()

    assert self.priv_key.exists()
    assert self.pub_key.exists()
    assert dongle == dongle_id_from_public_key(self.pub_key.read_text())
    assert self.params.get("DongleId") == dongle

  def test_missing_cache(self):
    public_key = self._generate_keys()
    dongle = dongle_id_from_public_key(public_key)

    assert register() == dongle
    assert register() == dongle
    assert self.params.get("DongleId") == dongle

  def test_invalid_cache_is_replaced(self):
    public_key = self._generate_keys()
    self.params.put("DongleId", "0000000000000000", block=True)

    dongle = register()
    assert dongle == dongle_id_from_public_key(public_key)

  def test_key_create_failure(self, monkeypatch):
    monkeypatch.setattr("openpilot.system.athena.registration.ed25519.Ed25519PrivateKey.generate", lambda: (_ for _ in ()).throw(OSError("no write")))

    dongle = register()
    assert dongle == UNREGISTERED_DONGLE_ID
    assert self.params.get("DongleId") == dongle
