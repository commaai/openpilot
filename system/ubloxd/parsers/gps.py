"""
GPS subframe parser using pure Python bit operations.
Provides a subset of the Kaitai-generated API used by ubloxd.py.
"""

from __future__ import annotations

import struct


class _BitReader:
  def __init__(self, b: bytes):
    self.b = b
    self.bit = 0  # bit offset

  def read_bits_be(self, n: int) -> int:
    v = 0
    for _ in range(n):
      byte_index = self.bit // 8
      bit_index = 7 - (self.bit % 8)
      v = (v << 1) | ((self.b[byte_index] >> bit_index) & 1)
      self.bit += 1
    return v

  def align_to_byte(self) -> None:
    if self.bit % 8:
      self.bit += 8 - (self.bit % 8)

  def read_u1(self) -> int:
    self.align_to_byte()
    v = self.b[self.bit // 8]
    self.bit += 8
    return v

  def read_s1(self) -> int:
    v = self.read_u1()
    return struct.unpack("<b", bytes([v]))[0]

  def read_s2be(self) -> int:
    self.align_to_byte()
    i = self.bit // 8
    v = struct.unpack(">h", self.b[i:i+2])[0]
    self.bit += 16
    return v

  def read_s4be(self) -> int:
    self.align_to_byte()
    i = self.bit // 8
    v = struct.unpack(">i", self.b[i:i+4])[0]
    self.bit += 32
    return v

  def read_u2be(self) -> int:
    self.align_to_byte()
    i = self.bit // 8
    v = struct.unpack(">H", self.b[i:i+2])[0]
    self.bit += 16
    return v

  def read_u4be(self) -> int:
    self.align_to_byte()
    i = self.bit // 8
    v = struct.unpack(">I", self.b[i:i+4])[0]
    self.bit += 32
    return v


class Gps:
  class Tlm:
    def __init__(self, r: _BitReader):
      # TLM preamble must be 0x8B
      pre = r.read_u1()
      if pre != 0x8B:
        raise ValueError("Invalid GPS TLM preamble")
      self.tlm = r.read_bits_be(14)
      self.integrity_status = r.read_bits_be(1) != 0
      self.reserved = r.read_bits_be(1) != 0

  class How:
    def __init__(self, r: _BitReader):
      self.tow_count = r.read_bits_be(17)
      self.alert = r.read_bits_be(1) != 0
      self.anti_spoof = r.read_bits_be(1) != 0
      self.subframe_id = r.read_bits_be(3)
      self.reserved = r.read_bits_be(2)

  class Subframe1:
    def __init__(self, r: _BitReader):
      self.week_no = r.read_bits_be(10)
      self.code = r.read_bits_be(2)
      self.sv_accuracy = r.read_bits_be(4)
      self.sv_health = r.read_bits_be(6)
      self.iodc_msb = r.read_bits_be(2)
      self.l2_p_data_flag = r.read_bits_be(1) != 0
      self.reserved1 = r.read_bits_be(23)
      self.reserved2 = r.read_bits_be(24)
      self.reserved3 = r.read_bits_be(24)
      self.reserved4 = r.read_bits_be(16)
      r.align_to_byte()
      self.t_gd = r.read_s1()
      self.iodc_lsb = r.read_u1()
      self.t_oc = r.read_u2be()
      self.af_2 = r.read_s1()
      self.af_1 = r.read_s2be()
      af0_sign = r.read_bits_be(1) != 0
      af0_value = r.read_bits_be(21)
      self.af_0 = (af0_value - (1 << 21)) if af0_sign else af0_value
      self.reserved5 = r.read_bits_be(2)

  class Subframe2:
    def __init__(self, r: _BitReader):
      self.iode = r.read_u1()
      self.c_rs = r.read_s2be()
      self.delta_n = r.read_s2be()
      self.m_0 = r.read_s4be()
      self.c_uc = r.read_s2be()
      self.e = r.read_s4be()
      self.c_us = r.read_s2be()
      self.sqrt_a = r.read_u4be()
      self.t_oe = r.read_u2be()
      self.fit_interval_flag = r.read_bits_be(1) != 0
      self.aoda = r.read_bits_be(5)
      self.reserved = r.read_bits_be(2)

  class Subframe3:
    def __init__(self, r: _BitReader):
      self.c_ic = r.read_s2be()
      self.omega_0 = r.read_s4be()
      self.c_is = r.read_s2be()
      self.i_0 = r.read_s4be()
      self.c_rc = r.read_s2be()
      self.omega = r.read_s4be()
      od_sign = r.read_bits_be(1) != 0
      od_val = r.read_bits_be(23)
      self.omega_dot = (od_val - (1 << 23)) if od_sign else od_val
      r.align_to_byte()
      self.iode = r.read_u1()
      id_sign = r.read_bits_be(1) != 0
      id_val = r.read_bits_be(13)
      self.idot = (id_val - (1 << 13)) if id_sign else id_val
      self.reserved = r.read_bits_be(2)

  class Subframe4:
    class IonosphereData:
      def __init__(self, r: _BitReader):
        self.a0 = r.read_s1()
        self.a1 = r.read_s1()
        self.a2 = r.read_s1()
        self.a3 = r.read_s1()
        self.b0 = r.read_s1()
        self.b1 = r.read_s1()
        self.b2 = r.read_s1()
        self.b3 = r.read_s1()

    def __init__(self, r: _BitReader):
      self.data_id = r.read_bits_be(2)
      self.page_id = r.read_bits_be(6)
      r.align_to_byte()
      if self.page_id == 56:
        self.body = Gps.Subframe4.IonosphereData(r)

  @staticmethod
  def from_bytes(b: bytes) -> Gps:
    r = _BitReader(b)
    self = Gps()
    self.tlm = Gps.Tlm(r)
    self.how = Gps.How(r)
    sid = self.how.subframe_id
    if sid == 1:
      self.body = Gps.Subframe1(r)
    elif sid == 2:
      self.body = Gps.Subframe2(r)
    elif sid == 3:
      self.body = Gps.Subframe3(r)
    elif sid == 4:
      self.body = Gps.Subframe4(r)
    return self
