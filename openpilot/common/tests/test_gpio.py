import ctypes

from openpilot.common import gpio
from openpilot.common.test import OpenpilotTestCase


class TestGpio(OpenpilotTestCase):
  def test_ioctl_struct_sizes(self):
    # each ioctl number encodes the size of the kernel's struct, a struct that doesn't match it passes the wrong bytes
    for struct, request in ((gpio.gpiohandle_request, gpio.GPIO_GET_LINEHANDLE_IOCTL),
                            (gpio.gpiohandle_data, gpio.GPIOHANDLE_GET_LINE_VALUES_IOCTL),
                            (gpio.gpioevent_request, gpio.GPIO_GET_LINEEVENT_IOCTL)):
      assert ctypes.sizeof(struct) == (request >> 16) & 0x3fff, struct.__name__
