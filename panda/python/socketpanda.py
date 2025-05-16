import socket
import struct

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

CANFD_BRS = 0x01 # bit rate switch (second bitrate for payload data)
CANFD_FDF = 0x04 # mark CAN FD for dual use of struct canfd_frame

# socket.SO_RXQ_OVFL is missing
# https://github.com/torvalds/linux/blob/47ac09b91befbb6a235ab620c32af719f8208399/include/uapi/asm-generic/socket.h#L61
SO_RXQ_OVFL = 40

def create_socketcan(interface:str, recv_buffer_size:int, fd:bool) -> socket.socket:
  # settings mostly from https://github.com/linux-can/can-utils/blob/master/candump.c
  socketcan = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  if fd:
    socketcan.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
  socketcan.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer_size)
  # TODO: why is it always 2x the requested size?
  assert socketcan.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF) == recv_buffer_size * 2
  # TODO: how to dectect and alert on buffer overflow?
  socketcan.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
  socketcan.bind((interface,))
  return socketcan

# Panda class substitute for socketcan device (to support using the uds/iso-tp/xcp/ccp library)
class SocketPanda():
  def __init__(self, interface:str="can0", bus:int=0, fd:bool=False, recv_buffer_size:int=212992) -> None:
    self.interface = interface
    self.bus = bus
    self.fd = fd
    self.flags = CANFD_BRS | CANFD_FDF if fd else 0
    self.data_len = CANFD_MAX_DLEN if fd else CAN_MAX_DLEN
    self.recv_buffer_size = recv_buffer_size
    self.socket = create_socketcan(interface, recv_buffer_size, fd)

  def __del__(self):
    self.socket.close()

  def get_serial(self) -> tuple[int, int]:
    return (0, 0) # TODO: implemented in panda socketcan driver

  def get_version(self) -> int:
    return 0 # TODO: implemented in panda socketcan driver

  def can_clear(self, bus:int) -> None:
    # TODO: implemented in panda socketcan driver
    self.socket.close()
    self.socket = create_socketcan(self.interface, self.recv_buffer_size, self.fd)

  def set_safety_mode(self, mode:int, param=0) -> None:
    pass # TODO: implemented in panda socketcan driver

  def has_obd(self) -> bool:
    return False # TODO: implemented in panda socketcan driver

  def can_send(self, addr, dat, bus=0, timeout=0) -> None:
    msg_len = len(dat)
    msg_dat = dat.ljust(self.data_len, b'\x00')
    can_frame = struct.pack(CAN_HEADER_FMT, addr, msg_len, self.flags) + msg_dat
    self.socket.sendto(can_frame, (self.interface,))

  def can_recv(self) -> list[tuple[int, bytes, int]]:
    msgs = list()
    while True:
      try:
        dat, _ = self.socket.recvfrom(self.recv_buffer_size, socket.MSG_DONTWAIT)
        assert len(dat) == CAN_HEADER_LEN + self.data_len, f"ERROR: received {len(dat)} bytes"
        can_id, msg_len, _ = struct.unpack(CAN_HEADER_FMT, dat[:CAN_HEADER_LEN])
        msg_dat = dat[CAN_HEADER_LEN:CAN_HEADER_LEN+msg_len]
        msgs.append((can_id, msg_dat, self.bus))
      except BlockingIOError:
        break # buffered data exhausted
    return msgs
