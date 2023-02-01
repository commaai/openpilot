#!/usr/bin/env python3
import time
import struct
import argparse

from selfdrive.sensord.pigeond import create_pigeon, init_baudrate, initialize_pigeon, run_receiving, add_ubx_checksum

GPS_RES_TRACK = 0x04
GPS_MAX_TRACK = 0x08
GPS_ENABLE_FREQ =  b"\x01\x00\x01\x01" # first byte enable
GLONASS_RES_TRACK = 0x10
GLONASS_MAX_TRACK = 0x12
GLONASS_ENABLE_FREQ =  b"\x01\x00\x01\x01"
QZSS_RES_TRACK = 0x00
QZSS_MAX_TRACK = 0x02
QZSS_ENABLE_FREQ =  b"\x01\x00\x01\x01"


def get_gnss_config(pigeon):
  # UBX-CFG-GNSS pull request
  hdr = b'\xb5\x62\x06\x3E'
  pigeon.send(hdr + b'\x00\x00\x44\xD2')
  while True:
    recv = pigeon.receive()
    if hdr in recv:
      return hdr + recv.split(hdr)[1]

def set_gnss_config(pigeon):
  # UBX-CFG-GNSS
  hdr  = b'\xB5\x62\x06\x3E'
  c    = b'\x00\x1c\x1c'
  msg  = struct.pack("I", (GPS_MAX_TRACK<<16) | (GPS_RES_TRACK<<8)) # GPS GNSS CFG
  msg += GPS_ENABLE_FREQ
  msg += b'\x01\x00\x00\x00' # GPS SBAS (disable)
  msg += b'\x00\x00\x01\x01'
  msg += b'\x02\x00\x00\x00' # GALILEO (disable)
  msg += b'\x00\x20\x00\x01'
  msg += b'\x03\x00\x00\x00' # BEIDOU (disable)
  msg += b'\x00\x00\x01\x01'
  msg += b'\x04\x00\x00\x00' # IMES (disable)
  msg += b'\x00\x00\x01\x01'
  msg += struct.pack("I", (QZSS_MAX_TRACK<<16) | (QZSS_RES_TRACK<<8) | 5) # QZSS (keep enabled with GPS)
  msg += QZSS_ENABLE_FREQ
  msg += struct.pack("I", (GLONASS_MAX_TRACK<<16) | (GLONASS_RES_TRACK<<8) | 6) # GLONASS
  msg += GLONASS_ENABLE_FREQ
  gnss_cfg = hdr + struct.pack("H", len(msg) + 4) + c + struct.pack("b", int(len(msg)/8)) + msg
  pigeon.send_with_ack(add_ubx_checksum(gnss_cfg))

  # UBX-CFG-CFG (saveMask)
  pigeon.send_with_ack(b'\xB5\x62\x06\x09\x0C\x00\x00\x00\x00\x00\x1A\x00\x00\x00\x00\x00\x00\x00\x35\x5F')

def verify_gnss_config(pigeon):
  def check_data(data, resTrkCh, maxTrkCh, flags):
    return data[1] == resTrkCh and data[2] == maxTrkCh and data[4:8]  == flags

  def check_block_config(block):
    if block[0] == 0: # GPS
      return check_data(block, GPS_RES_TRACK, GPS_MAX_TRACK, GPS_ENABLE_FREQ)
    elif block[0] in [1,2,3,4]: # SBAS, GALILEO, BEIDOU, IMES
      return block[4] == 0 # disabled
    elif block[0] == 5: # QZSS
      return check_data(block, QZSS_RES_TRACK, QZSS_MAX_TRACK, QZSS_ENABLE_FREQ)
    elif block[0] == 6: # GLONASS
      return check_data(block, GLONASS_RES_TRACK, GLONASS_MAX_TRACK, GLONASS_ENABLE_FREQ)

  data = get_gnss_config(pigeon)
  length = struct.unpack("H", data[4:6])[0]
  numConfigBlocks = data[9]
  configs = data[10:10+length-4]
  return all(check_block_config(configs[idx:idx+8]) for idx in range(0, numConfigBlocks*8, 8))

def apply_gnss_config(pigeon):
  if not verify_gnss_config(pigeon):
    print("set GNSS config")
    set_gnss_config(pigeon)
    return True
  return False

def main(clear_config=False):
  pigeon, pm = create_pigeon()

  if clear_config:
    print("reset ublox config")
    # clearMask
    pigeon.send_with_ack(b'\xB5\x62\x06\x09\x0C\x00\x1A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x005\xc7')
    # cold start, controlled software reset
    # TODO: verify if this is the same in pigeond
    pigeon.send(b"\xb5\x62\x06\x04\x04\x00\xff\xff\x01\x00\x0d\x5f")
    time.sleep(1)
    clear_config = False
    init_baudrate(pigeon)
    ret = pigeon.reset_device()
    print(f"Reset successful: {ret}")

  retry_gnss_setting = False
  while True:
    init_baudrate(pigeon)

    initialize_pigeon(pigeon)

    # UBX-NAV-ORB, enable more DEBUG messages
    pigeon.send_with_ack(b"\xB5\x62\x06\x01\x03\x00\x01\x34\x0AI\xB4")

    if apply_gnss_config(pigeon):
      if retry_gnss_setting:
        print("Could not set GNSS config")
      else:
        # GNSS config change needs a device reset, Software Reset
        pigeon.send(b"\xB5\x62\x06\x04\x04\x00\xFF\xFF\x01\x00\x0D\x5F")
        time.sleep(0.5)
        retry_gnss_setting = True
        continue
    break

  # start receiving data
  run_receiving(pigeon, pm)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Next level pigeond.")
  parser.add_argument("-c", "--clear_config", action='store_true', default=False, help='Clear ublox GNSS config')
  args = parser.parse_args()
  main(args.clear_config)
