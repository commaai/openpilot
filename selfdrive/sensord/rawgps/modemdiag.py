import os
import time
import select
from serial import Serial
from crcmod import mkCrcFun
from struct import pack, unpack_from, calcsize

class ModemDiag:
  def __init__(self):
    self.serial = self.open_serial()
    self.pend = b''

  def open_serial(self):
    def op():
      return Serial("/dev/ttyUSB0", baudrate=115200, rtscts=True, dsrdtr=True, timeout=0)
    try:
      serial = op()
    except Exception:
      # TODO: this is a hack to get around modemmanager's exclusive open
      print("unlocking serial...")
      os.system('sudo su -c \'echo "1-1.1:1.0" > /sys/bus/usb/drivers/option/unbind\'')
      os.system('sudo su -c \'echo "1-1.1:1.0" > /sys/bus/usb/drivers/option/bind\'')
      time.sleep(0.5)
      os.system("sudo chmod 666 /dev/ttyUSB0")
      serial = op()
    serial.flush()
    serial.reset_input_buffer()
    serial.reset_output_buffer()
    return serial

  ccitt_crc16 = mkCrcFun(0x11021, initCrc=0, xorOut=0xffff)
  ESCAPE_CHAR = b'\x7d'
  TRAILER_CHAR = b'\x7e'

  def hdlc_encapsulate(self, payload):
    payload += pack('<H', ModemDiag.ccitt_crc16(payload))
    payload = payload.replace(self.ESCAPE_CHAR, bytes([self.ESCAPE_CHAR[0], self.ESCAPE_CHAR[0] ^ 0x20]))
    payload = payload.replace(self.TRAILER_CHAR, bytes([self.ESCAPE_CHAR[0], self.TRAILER_CHAR[0] ^ 0x20]))
    payload += self.TRAILER_CHAR
    return payload

  def hdlc_decapsulate(self, payload):
    assert len(payload) >= 3
    assert payload[-1:] == self.TRAILER_CHAR
    payload = payload[:-1]
    payload = payload.replace(bytes([self.ESCAPE_CHAR[0], self.TRAILER_CHAR[0] ^ 0x20]), self.TRAILER_CHAR)
    payload = payload.replace(bytes([self.ESCAPE_CHAR[0], self.ESCAPE_CHAR[0] ^ 0x20]), self.ESCAPE_CHAR)
    assert payload[-2:] == pack('<H', ModemDiag.ccitt_crc16(payload[:-2]))
    return payload[:-2]

  def recv(self):
    # self.serial.read_until makes tons of syscalls!
    raw_payload = [self.pend]
    while self.TRAILER_CHAR not in raw_payload[-1]:
      select.select([self.serial.fd], [], [])
      raw = self.serial.read(0x10000)
      raw_payload.append(raw)
    raw_payload = b''.join(raw_payload)
    raw_payload, self.pend = raw_payload.split(self.TRAILER_CHAR, 1)
    raw_payload += self.TRAILER_CHAR
    unframed_message = self.hdlc_decapsulate(raw_payload)
    return unframed_message[0], unframed_message[1:]

  def send(self, packet_type, packet_payload):
    self.serial.write(self.hdlc_encapsulate(bytes([packet_type]) + packet_payload))

# *** end class ***

DIAG_LOG_F = 16
DIAG_LOG_CONFIG_F = 115
LOG_CONFIG_RETRIEVE_ID_RANGES_OP = 1
LOG_CONFIG_SET_MASK_OP = 3
LOG_CONFIG_SUCCESS_S = 0

def send_recv(diag, packet_type, packet_payload):
  diag.send(packet_type, packet_payload)
  while 1:
    opcode, payload = diag.recv()
    if opcode != DIAG_LOG_F:
      break
  return opcode, payload

def setup_logs(diag, types_to_log):
  opcode, payload = send_recv(diag, DIAG_LOG_CONFIG_F, pack('<3xI', LOG_CONFIG_RETRIEVE_ID_RANGES_OP))

  header_spec = '<3xII'
  operation, status = unpack_from(header_spec, payload)
  assert operation == LOG_CONFIG_RETRIEVE_ID_RANGES_OP
  assert status == LOG_CONFIG_SUCCESS_S

  log_masks = unpack_from('<16I', payload, calcsize(header_spec))

  for log_type, log_mask_bitsize in enumerate(log_masks):
    if log_mask_bitsize:
      log_mask = [0] * ((log_mask_bitsize+7)//8)
      for i in range(log_mask_bitsize):
        if ((log_type<<12)|i) in types_to_log:
          log_mask[i//8] |= 1 << (i%8)
      opcode, payload = send_recv(diag, DIAG_LOG_CONFIG_F, pack('<3xIII',
          LOG_CONFIG_SET_MASK_OP,
          log_type,
          log_mask_bitsize
      ) + bytes(log_mask))
      assert opcode == DIAG_LOG_CONFIG_F
      operation, status = unpack_from(header_spec, payload)
      assert operation == LOG_CONFIG_SET_MASK_OP
      assert status == LOG_CONFIG_SUCCESS_S
