#!/usr/bin/env python3
from panda import Panda
from panda.python.uds import UdsClient, NegativeResponseError, DATA_IDENTIFIER_TYPE

if __name__ == "__main__":
  address = 0x18da30f1 # Honda EPS
  panda = Panda()
  uds_client = UdsClient(panda, address, debug=False)

  print("tester present ...")
  uds_client.tester_present()

  try:
    print("")
    print("read data by id: boot software id ...")
    data = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.BOOT_SOFTWARE_IDENTIFICATION)
    print(data.decode('utf-8'))
  except NegativeResponseError as e:
    print(e)

  try:
    print("")
    print("read data by id: application software id ...")
    data = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION)
    print(data.decode('utf-8'))
  except NegativeResponseError as e:
    print(e)

  try:
    print("")
    print("read data by id: application data id ...")
    data = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)
    print(data.decode('utf-8'))
  except NegativeResponseError as e:
    print(e)

  try:
    print("")
    print("read data by id: boot software fingerprint ...")
    data = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.BOOT_SOFTWARE_FINGERPRINT)
    print(data.decode('utf-8'))
  except NegativeResponseError as e:
    print(e)

  try:
    print("")
    print("read data by id: application software fingerprint ...")
    data = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_FINGERPRINT)
    print(data.decode('utf-8'))
  except NegativeResponseError as e:
    print(e)

  try:
    print("")
    print("read data by id: application data fingerprint ...")
    data = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.APPLICATION_DATA_FINGERPRINT)
    print(data.decode('utf-8'))
  except NegativeResponseError as e:
    print(e)
