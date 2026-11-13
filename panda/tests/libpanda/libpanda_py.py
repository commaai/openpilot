import ctypes
import os

from panda import LEN_TO_DLC

libpanda_dir = os.path.dirname(os.path.abspath(__file__))
libpanda_fn = os.path.join(libpanda_dir, "libpanda.so")


class CANPacketHeader(ctypes.LittleEndianStructure):
  _pack_ = 1
  _fields_ = [
    ("fd", ctypes.c_uint8, 1),
    ("bus", ctypes.c_uint8, 3),
    ("data_len_code", ctypes.c_uint8, 4),
  ]


class CANPacketAddress(ctypes.LittleEndianStructure):
  _pack_ = 1
  _fields_ = [
    ("rejected", ctypes.c_uint32, 1),
    ("returned", ctypes.c_uint32, 1),
    ("extended", ctypes.c_uint32, 1),
    ("addr", ctypes.c_uint32, 29),
  ]


class CANPacketFields(ctypes.LittleEndianStructure):
  _pack_ = 1
  _anonymous_ = ("header", "address")
  _fields_ = [
    ("header", CANPacketHeader),
    ("address", CANPacketAddress),
    ("checksum", ctypes.c_uint8),
    ("data", ctypes.c_uint8 * 64),
  ]


class CANPacket(ctypes.Union):
  # Match CANPacket_t's packed fields and aligned(4) attribute.
  _anonymous_ = ("packet",)
  _fields_ = [("packet", CANPacketFields), ("_alignment", ctypes.c_uint32)]


class CANRing(ctypes.Structure):
  _fields_ = [
    ("w_ptr", ctypes.c_uint32),
    ("r_ptr", ctypes.c_uint32),
    ("fifo_size", ctypes.c_uint32),
    ("elems", ctypes.POINTER(CANPacket)),
  ]


libpanda = ctypes.CDLL(libpanda_fn)

for name in ("rx_q", "tx1_q", "tx2_q", "tx3_q"):
  setattr(libpanda, name, ctypes.POINTER(CANRing).in_dll(libpanda, name))

for name, argtypes, restype in (
  ("set_safety_hooks", [ctypes.c_uint16, ctypes.c_uint16], ctypes.c_int),
  ("can_pop", [ctypes.POINTER(CANRing), ctypes.POINTER(CANPacket)], ctypes.c_bool),
  ("can_push", [ctypes.POINTER(CANRing), ctypes.POINTER(CANPacket)], ctypes.c_bool),
  ("can_set_checksum", [ctypes.POINTER(CANPacket)], None),
  ("comms_can_read", [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32], ctypes.c_int),
  ("comms_can_write", [ctypes.c_char_p, ctypes.c_uint32], None),
  ("comms_can_reset", [], None),
  ("can_slots_empty", [ctypes.POINTER(CANRing)], ctypes.c_uint32),
):
  func = getattr(libpanda, name)
  func.argtypes = argtypes
  func.restype = restype


def make_CANPacket(addr: int, bus: int, dat) -> CANPacket:
  ret = CANPacket()
  ret.extended = 1 if addr >= 0x800 else 0
  ret.addr = addr
  ret.data_len_code = LEN_TO_DLC[len(dat)]
  ret.bus = bus
  ret.data[:len(dat)] = bytes(dat)
  libpanda.can_set_checksum(ret)
  return ret
