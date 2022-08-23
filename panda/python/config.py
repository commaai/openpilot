import os

BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

BOOTSTUB_ADDRESS = 0x8000000

BLOCK_SIZE_FX = 0x800
APP_ADDRESS_FX = 0x8004000
SECTOR_SIZES_FX = [0x4000 for _ in range(4)] + [0x10000] + [0x20000 for _ in range(11)]
DEVICE_SERIAL_NUMBER_ADDR_FX = 0x1FFF79C0
DEFAULT_FW_FN = os.path.join(BASEDIR, "board", "obj", "panda.bin.signed")
DEFAULT_BOOTSTUB_FN = os.path.join(BASEDIR, "board", "obj", "bootstub.panda.bin")

BLOCK_SIZE_H7 = 0x400
APP_ADDRESS_H7 = 0x8020000
SECTOR_SIZES_H7 = [0x20000 for _ in range(7)] # there is an 8th sector, but we use that for the provisioning chunk, so don't program over that!
DEVICE_SERIAL_NUMBER_ADDR_H7 = 0x080FFFC0
DEFAULT_H7_FW_FN = os.path.join(BASEDIR, "board", "obj", "panda_h7.bin.signed")
DEFAULT_H7_BOOTSTUB_FN = os.path.join(BASEDIR, "board", "obj", "bootstub.panda_h7.bin")
