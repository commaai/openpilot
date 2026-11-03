import ctypes
from unittest import mock

from openpilot.common import gpio
from openpilot.common.test import OpenpilotTestCase


class TestGpio(OpenpilotTestCase):
  def test_gpio_structs(self):
    # sizes from the kernel's include/uapi/linux/gpio.h
    assert ctypes.sizeof(gpio.gpio_v2_line_request) == 592
    assert ctypes.sizeof(gpio.gpio_v2_line_event) == 48
    assert ctypes.sizeof(gpio.gpioevent_request) == 48
    assert ctypes.sizeof(gpio.gpioevent_data) == 16
    # _IOWR(0xB4, 0x07, struct gpio_v2_line_request)
    assert 0xc0000000 | (ctypes.sizeof(gpio.gpio_v2_line_request) << 16) | (0xb4 << 8) | 0x07 == 0xc250b407

  def test_tlmm_base(self):
    gpio.tlmm_base.cache_clear()
    self.addCleanup(gpio.tlmm_base.cache_clear)
    with mock.patch.object(gpio, "upstream_kernel", return_value=False):
      assert gpio.tlmm_base() == 0
    gpio.tlmm_base.cache_clear()
    with mock.patch.object(gpio, "upstream_kernel", return_value=True), mock.patch("glob.glob", return_value=[]):
      with self.assertRaises(RuntimeError):
        gpio.tlmm_base()
