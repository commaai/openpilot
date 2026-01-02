import socket
import struct
import time

# /**
#  * struct canfd_frame - CAN flexible data rate frame structure
#  * @can_id: CAN ID of the frame and CAN_*_FLAG flags, see canid_t definition
#  * @len:    frame payload length in byte (0 .. CANFD_MAX_DLEN)
#  * @flags:  additional flags for CAN FD
#  * @__res0: reserved / padding
#  * @__res1: reserved / padding
#  * @data:   CAN FD frame payload (up to CANFD_MAX_DLEN byte)
#  */
# struct canfd_frame {
# 	canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
# 	__u8    len;     /* frame payload length in byte */
# 	__u8    flags;   /* additional flags for CAN FD */
# 	__u8    __res0;  /* reserved / padding */
# 	__u8    __res1;  /* reserved / padding */
# 	__u8    data[CANFD_MAX_DLEN] __attribute__((aligned(8)));
# };
CAN_HEADER_FMT = "=IBB2x"
CAN_HEADER_LEN = struct.calcsize(CAN_HEADER_FMT)
CAN_MAX_DLEN = 8
CANFD_MAX_DLEN = 64

CAN_CONFIRM_FLAG = 0x800
CAN_EFF_FLAG = 0x80000000

CANFD_BRS = 0x01 # bit rate switch (second bitrate for payload data)
CANFD_FDF = 0x04 # mark CAN FD for dual use of struct canfd_frame

# socket.SO_RXQ_OVFL is missing
# https://github.com/torvalds/linux/blob/47ac09b91befbb6a235ab620c32af719f8208399/include/uapi/asm-generic/socket.h#L61
SO_RXQ_OVFL = 40

import typing
@typing.no_type_check # mypy struggles with macOS here...
def create_socketcan(interface:str, recv_buffer_size:int) -> socket.socket:
  # settings mostly from https://github.com/linux-can/can-utils/blob/master/candump.c
  socketcan = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  socketcan.setblocking(False)
  socketcan.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
  socketcan.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 1)
  socketcan.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer_size)
  # TODO: why is it always 2x the requested size?
  assert socketcan.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF) == recv_buffer_size * 2
  # TODO: how to dectect and alert on buffer overflow?
  socketcan.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
  socketcan.bind((interface,))
  return socketcan

# Panda class substitute for socketcan device (to support using the uds/iso-tp/xcp/ccp library)
class SocketPanda():
  def __init__(self, interface:str="can0", recv_buffer_size:int=212992) -> None:
    self.interface = interface
    self.recv_buffer_size = recv_buffer_size
    self.socket = create_socketcan(interface, recv_buffer_size)

  def __del__(self):
    self.socket.close()

  def get_serial(self) -> tuple[int, int]:
    return (0, 0)

  def get_version(self) -> int:
    return 0

  def can_clear(self, bus:int) -> None:
    self.socket.close()
    self.socket = create_socketcan(self.interface, self.recv_buffer_size)

  def set_safety_mode(self, mode:int, param=0) -> None:
    pass

  def can_send_many(self, arr, *, fd=False, timeout=0) -> None:
    for msg in arr:
      self.can_send(*msg, fd=fd, timeout=timeout)

  def can_send(self, addr, dat, bus, *, fd=False, timeout=0) -> None:
    # Even if the CANFD_FDF flag is not set, the data still must be 8 bytes for classic CAN frames.
    data_len = CANFD_MAX_DLEN if fd else CAN_MAX_DLEN
    msg_len = len(dat)
    msg_dat = dat.ljust(data_len, b'\x00')

    # Set extended ID flag
    if addr > 0x7ff:
      addr |= CAN_EFF_FLAG

    # Set FD flags
    flags = CANFD_BRS | CANFD_FDF if fd else 0

    can_frame = struct.pack(CAN_HEADER_FMT, addr, msg_len, flags) + msg_dat

    # Try to send until timeout. sendto might block if the TX buffer is full.
    # TX buffer size can also be adjusted through `ip link set can0 txqueuelen <size>` if needed
    start_t = time.monotonic()
    while (time.monotonic() - start_t < (timeout / 1000)) or (timeout == 0):
      try:
        self.socket.sendto(can_frame, (self.interface,))
        break
      except (BlockingIOError, OSError):
        continue
    else:
      raise TimeoutError


  def can_recv(self) -> list[tuple[int, bytes, int]]:
    msgs = list()
    while True:
      try:
        dat, _, msg_flags, _ = self.socket.recvmsg(self.recv_buffer_size)
        assert len(dat) >= CAN_HEADER_LEN, f"ERROR: received {len(dat)} bytes"

        can_id, msg_len, _ = struct.unpack(CAN_HEADER_FMT, dat[:CAN_HEADER_LEN])
        assert len(dat) >= CAN_HEADER_LEN + msg_len, f"ERROR: received {len(dat)} bytes, expected at least {CAN_HEADER_LEN + msg_len} bytes"

        msg_dat = dat[CAN_HEADER_LEN:CAN_HEADER_LEN+msg_len]
        bus = 128 if (msg_flags & CAN_CONFIRM_FLAG) else 0
        msgs.append((can_id, msg_dat, bus))
      except BlockingIOError:
        break # buffered data exhausted
    return msgs
