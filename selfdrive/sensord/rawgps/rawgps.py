#!/usr/bin/env python3
import os
import time
from serial import Serial
from crcmod import mkCrcFun
from hexdump import hexdump
from struct import pack, unpack_from, calcsize, unpack

def unlock_serial():
  os.system('sudo su -c \'echo "1-1.1:1.0" > /sys/bus/usb/drivers/option/unbind\'')
  os.system('sudo su -c \'echo "1-1.1:1.0" > /sys/bus/usb/drivers/option/bind\'')
  time.sleep(0.5)
  os.system("sudo chmod 666 /dev/ttyUSB0")

def open_serial():
  # TODO: this is a hack to get around modemmanager's exclusive open
  try:
    return Serial("/dev/ttyUSB0", baudrate=115200, rtscts=True, dsrdtr=True)
  except Exception:
    print("unlocking serial...")
    unlock_serial()
    return Serial("/dev/ttyUSB0", baudrate=115200, rtscts=True, dsrdtr=True)

ccitt_crc16 = mkCrcFun(0x11021, initCrc=0, xorOut=0xffff)
ESCAPE_CHAR = b'\x7d'
TRAILER_CHAR = b'\x7e'

def hdlc_encapsulate(payload):
  payload += pack('<H', ccitt_crc16(payload))
  payload = payload.replace(ESCAPE_CHAR, bytes([ESCAPE_CHAR[0], ESCAPE_CHAR[0] ^ 0x20]))
  payload = payload.replace(TRAILER_CHAR, bytes([ESCAPE_CHAR[0], TRAILER_CHAR[0] ^ 0x20]))
  payload += TRAILER_CHAR
  return payload

def hdlc_decapsulate(payload):
  assert len(payload) >= 3
  assert payload[-1:] == TRAILER_CHAR
  payload = payload[:-1]
  payload = payload.replace(bytes([ESCAPE_CHAR[0], TRAILER_CHAR[0] ^ 0x20]), TRAILER_CHAR)
  payload = payload.replace(bytes([ESCAPE_CHAR[0], ESCAPE_CHAR[0] ^ 0x20]), ESCAPE_CHAR)
  assert payload[-2:] == pack('<H', ccitt_crc16(payload[:-2]))
  return payload[:-2]

DIAG_LOG_F = 16
DIAG_LOG_CONFIG_F = 115
LOG_CONFIG_RETRIEVE_ID_RANGES_OP = 1
LOG_CONFIG_SET_MASK_OP = 3
LOG_CONFIG_SUCCESS_S = 0

def recv(serial):
  raw_payload = []
  while 1:
    char_read = serial.read()
    raw_payload.append(char_read)
    if char_read.endswith(TRAILER_CHAR):
      break
  raw_payload = b''.join(raw_payload)
  unframed_message = hdlc_decapsulate(raw_payload)
  return unframed_message[0], unframed_message[1:]

def send_recv(serial, packet_type, packet_payload):
  serial.write(hdlc_encapsulate(bytes([packet_type]) + packet_payload))
  while 1:
    opcode, payload = recv(serial)
    if opcode != DIAG_LOG_F:
      break
  return opcode, payload

TYPES_FOR_RAW_PACKET_LOGGING = [
  0x1476,
  0x1477,
  0x1480,

  #0x1478,
  #0x1756,
  #0x1886,

  #0x14DE,
  #0x14E1,

  #0x1838,
  #0x147B,
  #0x147E,
  #0x1488,
  #0x1516,
]

def setup_rawgps():
  os.system("mmcli -m 0 --location-enable-gps-raw --location-enable-gps-nmea")
  opcode, payload = send_recv(serial, DIAG_LOG_CONFIG_F, pack('<3xI', LOG_CONFIG_RETRIEVE_ID_RANGES_OP))

  header_spec = '<3xII'
  operation, status = unpack_from(header_spec, payload)
  assert operation == LOG_CONFIG_RETRIEVE_ID_RANGES_OP
  assert status == LOG_CONFIG_SUCCESS_S

  log_masks = unpack_from('<16I', payload, calcsize(header_spec))
  print(log_masks)

  for log_type, log_mask_bitsize in enumerate(log_masks):
    if log_mask_bitsize:
      log_mask = [0] * ((log_mask_bitsize+7)//8)
      for i in range(log_mask_bitsize):
        if ((log_type<<12)|i) in TYPES_FOR_RAW_PACKET_LOGGING:
          log_mask[i//8] |= 1 << (i%8)
      opcode, payload = send_recv(serial, DIAG_LOG_CONFIG_F, pack('<3xIII',
          LOG_CONFIG_SET_MASK_OP,
          log_type,
          log_mask_bitsize
      ) + bytes(log_mask))
      operation, status = unpack_from(header_spec, payload)
      assert operation == LOG_CONFIG_SET_MASK_OP
      assert status == LOG_CONFIG_SUCCESS_S

svStructNames = ["svId", "observationState", "observations", 
  "goodObservations", "gpsParityErrorCount", "filterStages",
  "carrierNoise", "latency", "predetectInterval", "postdetections",
  "unfilteredMeasurementIntegral", "unfilteredMeasurementFraction", 
  "unfilteredTimeUncertainty", "unfilteredSpeed", "unfilteredSpeedUncertainty",
  "measurementStatus", "miscStatus", "multipathEstimate", 
  "azimuth", "elevation", "carrierPhaseCyclesIntegral", "carrierPhaseCyclesFraction",
  "fineSpeed", "fineSpeedUncertainty", "cycleSlipCount"]

if __name__ == "__main__":
  serial = open_serial()
  serial.flush()

  setup_rawgps()

  while 1:
    opcode, payload = recv(serial)
    assert opcode == DIAG_LOG_F
    (pending_msgs, log_outer_length), inner_log_packet = unpack_from('<BH', payload), payload[calcsize('<BH'):]
    (log_inner_length, log_type, log_time), log_payload = unpack_from('<HHQ', inner_log_packet), inner_log_packet[calcsize('<HHQ'):]
    print("%x len %d" % (log_type, len(log_payload)))

    if log_type == 0x1476:
      #hexdump(log_payload)
      pass

    if log_type == 0x1477: # or log_type == 0x1480:
      if log_type == 0x1477:
        dat = unpack("<BIHIffffB", log_payload[0:28])
        ll = 28
      else:
        dat = unpack("<BIBHIffffB", log_payload[0:29])
        ll = 29
      print(dat)
      sats = log_payload[ll:]
      L = 70
      assert len(sats)//dat[-1] == L
      for i in range(dat[-1]):
        sat = dict(zip(svStructNames, unpack("<BBBBHBHhBHIffffIBIffiHffBI", sats[L*i:L*i+L])[:-1]))
        print("  ", sat)


  #header_spec = '<3xII'
  #operation, status = unpack_from(header_spec, payload)
  #log_masks = unpack_from('<16I', payload, calcsize(header_spec))

  """
  opcode, payload = self.diag_input.send_recv(DIAG_LOG_CONFIG_F, pack('<3xIII',
    LOG_CONFIG_SET_MASK_OP,
    log_type,
    log_mask_bitsize
  ) + log_mask)
  """

