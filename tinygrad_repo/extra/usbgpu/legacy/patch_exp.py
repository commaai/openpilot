#!/usr/bin/env python3
import sys
import zlib

def patch(input_filepath, output_filepath, patches):
  with open(input_filepath, 'rb') as infile: data = bytearray(infile.read())

  for offset, expected_bytes, new_bytes in patches:
    if len(expected_bytes) != len(new_bytes):
      print(len(expected_bytes), len(new_bytes))
      raise ValueError("Expected bytes and new bytes must be the same length")

    if offset + len(new_bytes) > len(data): return False
    current_bytes = data[offset:offset + len(expected_bytes)]
    assert bytes(current_bytes) == expected_bytes, f"Expected {expected_bytes} at offset {offset:x}, but got {current_bytes}"
    data[offset:offset + len(new_bytes)] = new_bytes

  checksum = sum(data[4:-6]) & 0xff
  crc32 = zlib.crc32(data[4:-6]).to_bytes(4, 'little')
  data[-5] = checksum
  data[-4] = crc32[0]
  data[-3] = crc32[1]
  data[-2] = crc32[2]
  data[-1] = crc32[3]

  with open(output_filepath, 'wb') as outfile:
    outfile.write(data)

  return True

patches = [
  # (0x3903 + 1 + 4, b'\x8a', b'\x8b'),
  # (0x3cf9 + 1 + 4, b'\x8a', b'\x8b'), # this is the one which triggered...

  (0x2a0d + 1 + 4, b'\x0a', b'\x05'), # write handle exit with code 5 (?)
  # (0x40e1 + 4, b'\x90\x06\xe6\x04\xf0\x78\x0d\xe6\xfe\x24\x71\x12\x1b\x0b\x60\x0b\x74\x08', b'\x7f\x00\x12\x53\x21\x12\x1c\xfc\x74\x01\xf6\x90\x90\x94\x74\x10\xf0\x22')
  # (0x29ad + 1 + 4, b'\x09', b'\x05'), # write handle exit with code 5 (?)
  # (0x40ef + 0 + 4, b'\x60', b'\x70'), # jz -> jnz
  # (0x40e1 + 0 + 4, b'\x90', b'\x22'), # jmp -> ret
  # (0x40fa + 0 + 4, b'\x80', b'\x22'),
  # (0x40e1 + 0 + 4, b'\x90\x06\xe6\x04\xf0', b'\x7f\x00\x02\x41\x7c'), # jmp -> ret
]

next_traphandler = 0
def add_traphandler(addr, sec):
  global next_traphandler, patches

  trap_addr = 0x6000 + next_traphandler * 0x20
  return_addr = addr + len(sec)
  cntr_addr = 0x3000 + next_traphandler
  patches += [
    (addr + 4, sec, b'\x02' + trap_addr.to_bytes(2, 'big') + b'\x22'*(len(sec)-3)),
    (trap_addr + 4, b'\x00' * (21 + len(sec)),
      b'\xc0\xe0\xc0\x82\xc0\x83\x90' + cntr_addr.to_bytes(2, 'big') + b'\xe0\x04\xf0\xd0\x83\xd0\x82\xd0\xe0' + sec + b'\x02' + return_addr.to_bytes(2, 'big')),
  ]
  next_traphandler += 1

# add_traphandler(0x0206, b'\xed\x54\x06') # fill_scsi_resp
# add_traphandler(0x40d9, b'\x78\x6a\xe6') # fill_scsi_to_usb_transport
# add_traphandler(0x4d44, b'\x78\x6a\xe6') # FUN_CODE_4d44
# add_traphandler(0x4784, b'\x78\x6a\xe6') # FUN_CODE_4784
# add_traphandler(0x3e81, b'\x90\xc5\x16') # FUN_CODE_3e81
# add_traphandler(0x32a5, b'\x78\x6a\xe6') # FUN_CODE_32a5
# add_traphandler(0x2a10, b'\x90\xc4\x51') # FUN_CODE_2a10
# add_traphandler(0x2608, b'\x12\x16\x87') # FUN_CODE_2608
# add_traphandler(0x0e78, b'\x90\xc8\x02') # main usb entry
# add_traphandler(0x102f, b'\x12\x18\x0d') # possible scsi entry parser
# add_traphandler(0x1198, b'\x12\x18\x0d') # close_to_scsi_parse_1_and_set_c47a_to_0xff caller to scsi
# add_traphandler(0x180d, b'\x90\x0a\x7d') # close_to_scsi_parse
# add_traphandler(0x1114, b'\x75\x37\x00') # entry into if ((DAT_EXTMEM_c802 >> 2 & 1) != 0) { in main usb entry
# add_traphandler(0x113a, b'\x90\x90\x00') # exit from scsi parse loop
# add_traphandler(0x117b, b'\xd0\x07\xd0\x06') # exit from main usb entry


# add_traphandler(0x2f81, b'\x90\x0a\x59') # main loop? 8
# add_traphandler(0xc7a7, b'\x90\x09\xfa') # call smth in write path 9
# add_traphandler(0x2fcb, b'\x90\x0a\x59') # if ((DAT_EXTMEM_0ae2 != 0) && (DAT_EXTMEM_0ae2 != 0x10)) {
# add_traphandler(0x2fc0, b'\x90\x0a\xe2') # submain loop 11
# add_traphandler(0x30be, b'\x90\x0a\x5a') # aft sub loop 12
# add_traphandler(0x3076, b'\x12\x03\x59') # call to call_wait_for_nvme??(); 13
# add_traphandler(0x30ad, b'\x12\x04\xe4') # call to call_wait_for_nvme??(); 14

# add_traphandler(0x2608, b'\x12\x16\x87') # FUN_CODE_2608
# add_traphandler(0x10ee, b'\x90\x04\x64') # iniside trap handler
# add_traphandler(0x10e0, b'\x90\xc8\x06') # iniside trap handler
# add_traphandler(0x4977, b'\x90\x0a\xa8') # waiter for nvme???

assert patch(sys.argv[1], sys.argv[2], patches) is True
