#!/usr/bin/env python3
import capnp
from dataclasses import dataclass, field
import struct
from typing import Dict, List

import panda.python.uds as uds


def p16(val):
  return struct.pack("!H", val)


class StandardQueries:
  TESTER_PRESENT_REQUEST = bytes([uds.SERVICE_TYPE.TESTER_PRESENT, 0x0])
  TESTER_PRESENT_RESPONSE = bytes([uds.SERVICE_TYPE.TESTER_PRESENT + 0x40, 0x0])

  SHORT_TESTER_PRESENT_REQUEST = bytes([uds.SERVICE_TYPE.TESTER_PRESENT])
  SHORT_TESTER_PRESENT_RESPONSE = bytes([uds.SERVICE_TYPE.TESTER_PRESENT + 0x40])

  DEFAULT_DIAGNOSTIC_REQUEST = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL,
                                      uds.SESSION_TYPE.DEFAULT])
  DEFAULT_DIAGNOSTIC_RESPONSE = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40,
                                      uds.SESSION_TYPE.DEFAULT, 0x0, 0x32, 0x1, 0xf4])

  EXTENDED_DIAGNOSTIC_REQUEST = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL,
                                       uds.SESSION_TYPE.EXTENDED_DIAGNOSTIC])
  EXTENDED_DIAGNOSTIC_RESPONSE = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40,
                                        uds.SESSION_TYPE.EXTENDED_DIAGNOSTIC, 0x0, 0x32, 0x1, 0xf4])

  UDS_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
    p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION)
  UDS_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
    p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION)

  OBD_VERSION_REQUEST = b'\x09\x04'
  OBD_VERSION_RESPONSE = b'\x49\x04'


@dataclass
class Request:
  request: List[bytes]
  response: List[bytes]
  whitelist_ecus: List[int] = field(default_factory=list)
  rx_offset: int = 0x8
  bus: int = 1


@dataclass
class FwQueryConfig:
  requests: List[Request]
  # Overrides and removes from essential ecus for specific models and ecus (exact matching)
  non_essential_ecus: Dict[capnp.lib.capnp._EnumModule, List[str]] = field(default_factory=dict)
