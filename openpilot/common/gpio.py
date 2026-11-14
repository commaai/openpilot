import os
import glob
import fcntl
import ctypes
from functools import cache

from openpilot.common.kernel import upstream_kernel

TLMM = "/sys/bus/platform/devices/3400000.pinctrl"

@cache
def tlmm_base() -> int:
  # the upstream kernel assigns sysfs GPIO bases dynamically, the downstream one numbers the TLMM from 0
  if not upstream_kernel():
    return 0
  bases = glob.glob(f"{TLMM}/gpio/gpiochip*/base")
  if len(bases) != 1:
    raise RuntimeError(f"expected one sysfs GPIO chip for the TLMM, found {bases}")
  with open(bases[0]) as f:
    return int(f.read())

def gpio_init(pin: int, output: bool) -> None:
  try:
    with open(f"/sys/class/gpio/gpio{tlmm_base() + pin}/direction", 'wb') as f:
      f.write(b"out" if output else b"in")
  except Exception as e:
    print(f"Failed to set gpio {pin} direction: {e}")

def gpio_set(pin: int, high: bool) -> None:
  try:
    with open(f"/sys/class/gpio/gpio{tlmm_base() + pin}/value", 'wb') as f:
      f.write(b"1" if high else b"0")
  except Exception as e:
    print(f"Failed to set gpio {pin} value: {e}")

def gpio_read(pin: int) -> bool | None:
  val = None
  try:
    with open(f"/sys/class/gpio/gpio{tlmm_base() + pin}/value", 'rb') as f:
      val = bool(int(f.read().strip()))
  except Exception as e:
    print(f"Failed to set gpio {pin} value: {e}")

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

class gpioevent_data(ctypes.Structure):
  _fields_ = [
    ("timestamp", ctypes.c_uint64),  # CLOCK_REALTIME before 5.7
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

class gpio_v2_line_event(ctypes.Structure):
  _fields_ = [
    ("timestamp", ctypes.c_uint64),  # CLOCK_MONOTONIC
    ("id", ctypes.c_uint32),
    ("offset", ctypes.c_uint32),
    ("seqno", ctypes.c_uint32),
    ("line_seqno", ctypes.c_uint32),
    ("padding", ctypes.c_uint32 * 6),
  ]

class gpio_v2_line_request(ctypes.Structure):
  _fields_ = [
    ("offsets", ctypes.c_uint32 * 64),
    ("consumer", ctypes.c_char * 32),
    ("flags", ctypes.c_uint64),  # struct gpio_v2_line_config, without attributes
    ("num_attrs", ctypes.c_uint32),
    ("config_padding", ctypes.c_uint32 * 5),
    ("attrs", ctypes.c_uint8 * (24 * 10)),
    ("num_lines", ctypes.c_uint32),
    ("event_buffer_size", ctypes.c_uint32),
    ("padding", ctypes.c_uint32 * 5),
    ("fd", ctypes.c_int32),
  ]

def gpiochip_get_ro_value_fd(label: str, pin: int) -> int:
  # an fd to read both edges of a TLMM pin from, as gpio_v2_line_event on the upstream kernel, else gpioevent_data.
  # the upstream kernel has the v2 line ABI and may have the deprecated v1 ABI compiled out, the downstream 4.9 kernel only has v1
  chip = os.path.basename(glob.glob(f"{TLMM}/gpiochip*")[0])
  fd = os.open(f"/dev/{chip}", os.O_RDONLY)
  try:
    if upstream_kernel():
      GPIO_V2_LINE_FLAG_INPUT = 1 << 2
      GPIO_V2_LINE_FLAG_EDGE_RISING = 1 << 4
      GPIO_V2_LINE_FLAG_EDGE_FALLING = 1 << 5
      GPIO_V2_GET_LINE_IOCTL = 0xc250b407

      rq = gpio_v2_line_request()
      rq.offsets[0] = pin
      rq.num_lines = 1
      rq.consumer = label.encode('utf-8')[:31] + b'\0'
      rq.flags = GPIO_V2_LINE_FLAG_INPUT | GPIO_V2_LINE_FLAG_EDGE_RISING | GPIO_V2_LINE_FLAG_EDGE_FALLING
      fcntl.ioctl(fd, GPIO_V2_GET_LINE_IOCTL, rq)
    else:
      GPIOEVENT_REQUEST_BOTH_EDGES = 0x3
      GPIOHANDLE_REQUEST_INPUT = 0x1
      GPIO_GET_LINEEVENT_IOCTL = 0xc030b404

      rq = gpioevent_request()
      rq.lineoffset = pin
      rq.handleflags = GPIOHANDLE_REQUEST_INPUT
      rq.eventflags = GPIOEVENT_REQUEST_BOTH_EDGES
      rq.label = label.encode('utf-8')[:31] + b'\0'
      fcntl.ioctl(fd, GPIO_GET_LINEEVENT_IOCTL, rq)
  finally:
    os.close(fd)
  return int(rq.fd)
