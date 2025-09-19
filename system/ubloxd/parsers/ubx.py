"""
Pure-Python UBX payload parsers using struct and bit operations.
Exposes a minimal API compatible with ubloxd.py (no Kaitai dependency).
"""

from __future__ import annotations

import struct
from enum import IntEnum


class Ubx:
  class GnssType(IntEnum):
    gps = 0
    sbas = 1
    galileo = 2
    beidou = 3
    imes = 4
    qzss = 5
    glonass = 6

  class NavPvt:
    @staticmethod
    def from_bytes(b: bytes) -> Ubx.NavPvt:
      self = Ubx.NavPvt()
      (
        self.i_tow,
        self.year,
        self.month,
        self.day,
        self.hour,
        self.min,
        self.sec,
        self.valid,
        self.t_acc,
        self.nano,
        self.fix_type,
        self.flags,
        self.flags2,
        self.num_sv,
        self.lon,
        self.lat,
        self.height,
        self.h_msl,
        self.h_acc,
        self.v_acc,
        self.vel_n,
        self.vel_e,
        self.vel_d,
        self.g_speed,
        self.head_mot,
        self.s_acc,
        self.head_acc,
        self.p_dop,
        self.flags3,
      ) = struct.unpack_from(
        "<I HBBBBBB I i BBBB i i i i I I i i i i i I H B", b, 0
      )
      # Skip 5 reserved bytes, head_veh (i32), mag_dec (i16), mag_acc (u16)
      return self

  class RxmRawx:
    class Measurement:
      pass

    @staticmethod
    def from_bytes(b: bytes) -> Ubx.RxmRawx:
      self = Ubx.RxmRawx()
      self.rcv_tow, self.week = struct.unpack_from("<dH", b, 0)
      self.leap_s = struct.unpack_from("<b", b, 10)[0]
      self.num_meas = b[11]
      self.rec_stat = b[12]
      # skip 3 reserved bytes at 13..15
      self.meas: list[Ubx.RxmRawx.Measurement] = []
      off = 16
      for _ in range(self.num_meas):
        m = Ubx.RxmRawx.Measurement()
        m.pr_mes = struct.unpack_from("<d", b, off)[0]
        m.cp_mes = struct.unpack_from("<d", b, off + 8)[0]
        m.do_mes = struct.unpack_from("<f", b, off + 16)[0]
        gid = b[off + 20]
        m.gnss_id = Ubx.GnssType(gid)
        m.sv_id = b[off + 21]
        # off+22 reserved
        m.freq_id = b[off + 23]
        m.lock_time = struct.unpack_from("<H", b, off + 24)[0]
        m.cno = b[off + 26]
        m.pr_stdev = b[off + 27]
        m.cp_stdev = b[off + 28]
        m.do_stdev = b[off + 29]
        m.trk_stat = b[off + 30]
        # off+31 reserved
        self.meas.append(m)
        off += 32
      return self

  class RxmSfrbx:
    # Placeholder for typing; ubloxd constructs its own view for SFRBX parsing
    pass

  class NavSat:
    class Nav:
      pass

    @staticmethod
    def from_bytes(b: bytes) -> Ubx.NavSat:
      self = Ubx.NavSat()
      self.itow = struct.unpack_from("<I", b, 0)[0]
      self.version = b[4]
      self.num_svs = b[5]
      # 2 bytes reserved at 6..7
      self.svs: list[Ubx.NavSat.Nav] = []
      off = 8
      for _ in range(self.num_svs):
        n = Ubx.NavSat.Nav()
        gid = b[off + 0]
        n.gnss_id = Ubx.GnssType(gid)
        n.sv_id = b[off + 1]
        n.cno = b[off + 2]
        n.elev = struct.unpack_from("<b", b, off + 3)[0]
        n.azim = struct.unpack_from("<h", b, off + 4)[0]
        n.pr_res = struct.unpack_from("<h", b, off + 6)[0]
        n.flags = struct.unpack_from("<I", b, off + 8)[0]
        self.svs.append(n)
        off += 12
      return self

  class MonHw2:
    class ConfigSource(IntEnum):
      flash = 102
      otp = 111
      config_pins = 112
      rom = 113

    @staticmethod
    def from_bytes(b: bytes) -> Ubx.MonHw2:
      self = Ubx.MonHw2()
      self.ofs_i = struct.unpack_from("<b", b, 0)[0]
      self.mag_i = b[1]
      self.ofs_q = struct.unpack_from("<b", b, 2)[0]
      self.mag_q = b[3]
      self.cfg_source = Ubx.MonHw2.ConfigSource(b[4])
      # 3 reserved bytes at 5..7
      self.low_lev_cfg = struct.unpack_from("<I", b, 8)[0]
      # 8 reserved at 12..19
      self.post_status = struct.unpack_from("<I", b, 20)[0]
      # 4 reserved at 24..27
      return self

  class MonHw:
    class AntennaStatus(IntEnum):
      init = 0
      dontknow = 1
      ok = 2
      short = 3
      open = 4

    class AntennaPower(IntEnum):
      false = 0
      true = 1
      dontknow = 2

    @staticmethod
    def from_bytes(b: bytes) -> Ubx.MonHw:
      self = Ubx.MonHw()
      # Offsets per UBX-MON-HW message layout
      self.noise_per_ms = struct.unpack_from("<H", b, 16)[0]
      self.agc_cnt = struct.unpack_from("<H", b, 18)[0]
      self.a_status = Ubx.MonHw.AntennaStatus(b[20])
      self.a_power = Ubx.MonHw.AntennaPower(b[21])
      self.flags = b[22]
      self.jam_ind = b[45]
      return self
