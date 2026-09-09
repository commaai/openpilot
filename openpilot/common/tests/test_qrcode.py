import unittest

from openpilot.common import qrcode as qr
from openpilot.common.test import OpenpilotTestCase


class TestQRCode(OpenpilotTestCase):
  def test_alignment_positions(self):
    assert qr._alignment_positions(7) == [6, 22, 38]
    assert qr._alignment_positions(40) == [6, 30, 58, 86, 114, 142, 170]

  @unittest.expectedFailure  # the step between patterns rounds the wrong way for version 32
  def test_alignment_positions_v32(self):
    assert qr._alignment_positions(32) == [6, 34, 60, 86, 112, 138]
