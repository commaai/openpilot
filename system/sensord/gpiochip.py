import os
import fcntl
import ctypes

O_RDONLY = 0
GPIOEVENT_REQUEST_BOTH_EDGES = 0x3
GPIOHANDLE_REQUEST_INPUT = 0x1
GPIO_GET_LINEEVENT_IOCTL = 0xc030b404
class gpioevent_request(ctypes.Structure):
  _fields_ = [
      ("lineoffset", ctypes.c_uint32),
      ("handleflags", ctypes.c_uint32),
      ("eventflags", ctypes.c_uint32),
      ("label", ctypes.c_char * 32),
      ("fd", ctypes.c_int)
  ]

def gpiochip_get_ro_value_fd(label: str, gpiochip_id: int, pin: int) -> int:
  rq = gpioevent_request()
  rq.lineoffset = pin
  rq.handleflags = GPIOHANDLE_REQUEST_INPUT
  rq.eventflags = GPIOEVENT_REQUEST_BOTH_EDGES
  rq.label = label.encode('utf-8')[:31] + b'\0'

  fd = os.open(f"/dev/gpiochip{gpiochip_id}", O_RDONLY)
  fcntl.ioctl(fd, GPIO_GET_LINEEVENT_IOCTL, rq)
  os.close(fd)
  return rq.fd
