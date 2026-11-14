import os
import glob
import errno
import fcntl
import ctypes
from functools import cache

# the line is only held while it's set or read. the controller keeps driving it after it's released

def gpio_set(pin: int, high: bool) -> None:
  # direction and value in one step. sysfs took two writes, and "out" drove the pin low until the value was written
  try:
    os.close(_request_line(pin, GPIOHANDLE_REQUEST_OUTPUT, high))
  except Exception as e:
    print(f"Failed to set gpio {pin} value: {e}")

def gpio_read(pin: int) -> bool | None:
  val = None
  try:
    # no direction flag leaves the direction as it is
    fd = _request_line(pin, 0)
    try:
      data = gpiohandle_data()
      fcntl.ioctl(fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, data)
      val = bool(data.values[0])
    finally:
      os.close(fd)
  except Exception as e:
    print(f"Failed to read gpio {pin} value: {e}")

  return val

@cache
def get_irq_action(irq: int) -> list[str]:
  try:
    with open(f"/sys/kernel/irq/{irq}/actions") as f:
      actions = f.read().strip().split(',')
      return actions
  except FileNotFoundError:
    return []

def get_irqs_for_action(action: str) -> list[str]:
  ret = []
  with open("/proc/interrupts") as f:
    for l in f.readlines():
      irq = l.split(':')[0].strip()
      if irq.isdigit() and action in get_irq_action(irq):
        ret.append(irq)
  return ret

# *** gpiochip ***

# the SoC's GPIO controller, through the GPIO character device. pins are its line offsets,
# so they're the same on every kernel, unlike the global numbers in /sys/class/gpio
TLMM = "/sys/bus/platform/devices/3400000.pinctrl"

GPIO_GET_LINEHANDLE_IOCTL = 0xc16cb403
GPIO_GET_LINEEVENT_IOCTL = 0xc030b404
GPIOHANDLE_GET_LINE_VALUES_IOCTL = 0xc040b408
GPIOHANDLE_REQUEST_INPUT = 0x1
GPIOHANDLE_REQUEST_OUTPUT = 0x2
GPIOEVENT_REQUEST_BOTH_EDGES = 0x3

class gpiohandle_request(ctypes.Structure):
  _fields_ = [
    ("lineoffsets", ctypes.c_uint32 * 64),
    ("flags", ctypes.c_uint32),
    ("default_values", ctypes.c_uint8 * 64),
    ("consumer_label", ctypes.c_char * 32),
    ("lines", ctypes.c_uint32),
    ("fd", ctypes.c_int),
  ]

class gpiohandle_data(ctypes.Structure):
  _fields_ = [
    ("values", ctypes.c_uint8 * 64),
  ]

class gpioevent_data(ctypes.Structure):
  _fields_ = [
    ("timestamp", ctypes.c_uint64),
    ("id", ctypes.c_uint32),
  ]

class gpioevent_request(ctypes.Structure):
  _fields_ = [
    ("lineoffset", ctypes.c_uint32),
    ("handleflags", ctypes.c_uint32),
    ("eventflags", ctypes.c_uint32),
    ("label", ctypes.c_char * 32),
    ("fd", ctypes.c_int)
  ]

@cache
def gpiochip() -> str:
  chip, = glob.glob(f"{TLMM}/gpiochip*")
  return f"/dev/{os.path.basename(chip)}"

def _ioctl(request, rq) -> int:
  fd = os.open(gpiochip(), os.O_RDONLY)
  try:
    fcntl.ioctl(fd, request, rq)
  finally:
    os.close(fd)
  return int(rq.fd)

def _unexport(pin: int) -> bool:
  # TODO: remove once AGNOS stops exporting the pins openpilot drives. a pin exported in /sys/class/gpio
  # is busy for everyone else, and unexporting it leaves it as it is
  base, = glob.glob(f"{TLMM}/gpio/gpiochip*/base")
  with open(base) as f:
    number = int(f.read()) + pin
  if not os.path.isdir(f"/sys/class/gpio/gpio{number}"):
    return False
  with open("/sys/class/gpio/unexport", "w") as f:
    f.write(str(number))
  return True

def _request_line(pin: int, flags: int, value: bool = False) -> int:
  rq = gpiohandle_request()
  rq.lineoffsets[0] = pin
  rq.flags = flags
  rq.default_values[0] = value
  rq.consumer_label = b"openpilot"
  rq.lines = 1
  try:
    return _ioctl(GPIO_GET_LINEHANDLE_IOCTL, rq)
  except OSError as e:
    if e.errno != errno.EBUSY or not _unexport(pin):
      raise
  return _ioctl(GPIO_GET_LINEHANDLE_IOCTL, rq)

def gpiochip_get_ro_value_fd(label: str, pin: int) -> int:
  rq = gpioevent_request()
  rq.lineoffset = pin
  rq.handleflags = GPIOHANDLE_REQUEST_INPUT
  rq.eventflags = GPIOEVENT_REQUEST_BOTH_EDGES
  rq.label = label.encode('utf-8')[:31] + b'\0'
  return _ioctl(GPIO_GET_LINEEVENT_IOCTL, rq)
