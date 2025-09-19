"""
GLONASS string parser using pure Python bit operations.
Matches the subset of fields used by ubloxd.py.
"""

from __future__ import annotations


class _BitReader:
  def __init__(self, b: bytes):
    self.b = b
    self.bit = 0

  def read_bits_be(self, n: int) -> int:
    v = 0
    for _ in range(n):
      byte_index = self.bit // 8
      bit_index = 7 - (self.bit % 8)
      v = (v << 1) | ((self.b[byte_index] >> bit_index) & 1)
      self.bit += 1
    return v

  def read_bool(self) -> bool:
    return self.read_bits_be(1) != 0


class Glonass:
  class StringNonImmediate:
    def __init__(self, r: _BitReader):
      self.data_1 = r.read_bits_be(64)
      self.data_2 = r.read_bits_be(8)

  class String1:
    def __init__(self, r: _BitReader):
      self.not_used = r.read_bits_be(2)
      self.p1 = r.read_bits_be(2)
      self.t_k = r.read_bits_be(12)
      xv_sign = r.read_bool()
      xv_val = r.read_bits_be(23)
      xa_sign = r.read_bool()
      xa_val = r.read_bits_be(4)
      x_sign = r.read_bool()
      x_val = r.read_bits_be(26)
      self.x_vel = -xv_val if xv_sign else xv_val
      self.x_accel = -xa_val if xa_sign else xa_val
      self.x = -x_val if x_sign else x_val

  class String2:
    def __init__(self, r: _BitReader):
      self.b_n = r.read_bits_be(3)
      self.p2 = r.read_bool()
      self.t_b = r.read_bits_be(7)
      self.not_used = r.read_bits_be(5)
      yv_sign = r.read_bool()
      yv_val = r.read_bits_be(23)
      ya_sign = r.read_bool()
      ya_val = r.read_bits_be(4)
      y_sign = r.read_bool()
      y_val = r.read_bits_be(26)
      self.y_vel = -yv_val if yv_sign else yv_val
      self.y_accel = -ya_val if ya_sign else ya_val
      self.y = -y_val if y_sign else y_val

  class String3:
    def __init__(self, r: _BitReader):
      self.p3 = r.read_bool()
      gn_sign = r.read_bool()
      gn_val = r.read_bits_be(10)
      self.not_used = r.read_bool()
      self.p = r.read_bits_be(2)
      self.l_n = r.read_bool()
      zv_sign = r.read_bool()
      zv_val = r.read_bits_be(23)
      za_sign = r.read_bool()
      za_val = r.read_bits_be(4)
      z_sign = r.read_bool()
      z_val = r.read_bits_be(26)
      self.gamma_n = -gn_val if gn_sign else gn_val
      self.z_vel = -zv_val if zv_sign else zv_val
      self.z_accel = -za_val if za_sign else za_val
      self.z = -z_val if z_sign else z_val

  class String4:
    def __init__(self, r: _BitReader):
      tn_sign = r.read_bool()
      tn_val = r.read_bits_be(21)
      dtn_sign = r.read_bool()
      dtn_val = r.read_bits_be(4)
      self.e_n = r.read_bits_be(5)
      self.not_used_1 = r.read_bits_be(14)
      self.p4 = r.read_bool()
      self.f_t = r.read_bits_be(4)
      self.not_used_2 = r.read_bits_be(3)
      self.n_t = r.read_bits_be(11)
      self.n = r.read_bits_be(5)
      self.m = r.read_bits_be(2)
      self.tau_n = -tn_val if tn_sign else tn_val
      self.delta_tau_n = -dtn_val if dtn_sign else dtn_val

  class String5:
    def __init__(self, r: _BitReader):
      self.n_a = r.read_bits_be(11)
      self.tau_c = r.read_bits_be(32)
      self.not_used = r.read_bool()
      self.n_4 = r.read_bits_be(5)
      self.tau_gps = r.read_bits_be(22)
      self.l_n = r.read_bool()

  @staticmethod
  def from_bytes(b: bytes) -> Glonass:
    r = _BitReader(b)
    self = Glonass()
    self.idle_chip = r.read_bool()
    self.string_number = r.read_bits_be(4)
    # Do not align here; match the Kaitai workaround
    if self.string_number == 1:
      self.data = Glonass.String1(r)
    elif self.string_number == 2:
      self.data = Glonass.String2(r)
    elif self.string_number == 3:
      self.data = Glonass.String3(r)
    elif self.string_number == 4:
      self.data = Glonass.String4(r)
    elif self.string_number == 5:
      self.data = Glonass.String5(r)
    else:
      self.data = Glonass.StringNonImmediate(r)
    self.hamming_code = r.read_bits_be(8)
    self.pad_1 = r.read_bits_be(11)
    self.superframe_number = r.read_bits_be(16)
    self.pad_2 = r.read_bits_be(8)
    self.frame_number = r.read_bits_be(8)
    return self
