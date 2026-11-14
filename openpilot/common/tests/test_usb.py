from unittest import mock

from openpilot.common.hardware import usb
from openpilot.common.test import OpenpilotTestCase


class TestUsb(OpenpilotTestCase):
  def test_typec_orientation(self):
    for upstream, value, expected in [(True, "normal", 1), (True, "reverse", 2), (True, "unknown", 0), (True, None, 0),
                                      (False, "1", 1), (False, "2", 2), (False, "0", 0)]:
      with mock.patch.object(usb, "upstream_kernel", return_value=upstream), \
           mock.patch.object(usb, "read", return_value=value), \
           mock.patch.object(usb, "read_int", return_value=int(value or 0) if not upstream else 0):
        assert usb.typec_orientation() == expected, (upstream, value)
