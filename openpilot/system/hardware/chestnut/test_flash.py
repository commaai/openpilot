import hashlib

import pytest

from openpilot.system.hardware.chestnut.flash import EXPECTED_PRODUCT, EXPECTED_VERSION, FIRMWARE_PATH, FIRMWARE_SHA256, validate_wrapped


class TestChestnutFirmware:
  def test_bundled_image(self):
    image = FIRMWARE_PATH.read_bytes()
    assert hashlib.sha256(image).hexdigest() == FIRMWARE_SHA256
    validate_wrapped(image)

  def test_version_pin(self):
    image = FIRMWARE_PATH.read_bytes()
    assert EXPECTED_PRODUCT.encode() in image
    assert EXPECTED_PRODUCT == f"custom {EXPECTED_VERSION}-CLEAN"

  def test_version_matches_openpilot(self):
    from openpilot.common.hardware.usb import CHESTNUT_FW_VERSION
    assert CHESTNUT_FW_VERSION == EXPECTED_VERSION

  @pytest.mark.parametrize("corrupt", [4, 100, -6, -1])
  def test_corrupt_image_rejected(self, corrupt):
    image = bytearray(FIRMWARE_PATH.read_bytes())
    image[corrupt] ^= 0xFF
    with pytest.raises(ValueError):
      validate_wrapped(bytes(image))

  def test_truncated_image_rejected(self):
    image = FIRMWARE_PATH.read_bytes()
    with pytest.raises(ValueError):
      validate_wrapped(image[:-1])
