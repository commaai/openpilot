"""
UBX protocol parser - declarative Python DSL implementation.
"""

from enum import IntEnum
from typing import Annotated

from openpilot.system.ubloxd.binary_struct import (
  BinaryStruct,
  array,
  binary_struct,
  bytes_field,
  const,
  enum,
  f32,
  f64,
  s8,
  s16,
  s32,
  substream,
  switch,
  u8,
  u16,
  u16be,
  u32,
)


class GnssType(IntEnum):
  gps = 0
  sbas = 1
  galileo = 2
  beidou = 3
  imes = 4
  qzss = 5
  glonass = 6


@binary_struct
class Ubx(BinaryStruct):
  GnssType = GnssType

  @binary_struct
  class RxmRawx(BinaryStruct):
    @binary_struct
    class Measurement(BinaryStruct):
      pr_mes: Annotated[float, f64]
      cp_mes: Annotated[float, f64]
      do_mes: Annotated[float, f32]
      gnss_id: Annotated[GnssType | int, enum(u8, GnssType)]
      sv_id: Annotated[int, u8]
      reserved2: Annotated[bytes, bytes_field(1)]
      freq_id: Annotated[int, u8]
      lock_time: Annotated[int, u16]
      cno: Annotated[int, u8]
      pr_stdev: Annotated[int, u8]
      cp_stdev: Annotated[int, u8]
      do_stdev: Annotated[int, u8]
      trk_stat: Annotated[int, u8]
      reserved3: Annotated[bytes, bytes_field(1)]

    rcv_tow: Annotated[float, f64]
    week: Annotated[int, u16]
    leap_s: Annotated[int, s8]
    num_meas: Annotated[int, u8]
    rec_stat: Annotated[int, u8]
    reserved1: Annotated[bytes, bytes_field(3)]
    meas: Annotated[list[Measurement], array(Measurement, count_field='num_meas')]

  @binary_struct
  class RxmSfrbx(BinaryStruct):
    gnss_id: Annotated[GnssType | int, enum(u8, GnssType)]
    sv_id: Annotated[int, u8]
    reserved1: Annotated[bytes, bytes_field(1)]
    freq_id: Annotated[int, u8]
    num_words: Annotated[int, u8]
    reserved2: Annotated[bytes, bytes_field(1)]
    version: Annotated[int, u8]
    reserved3: Annotated[bytes, bytes_field(1)]
    body: Annotated[list[int], array(u32, count_field='num_words')]

  @binary_struct
  class NavSat(BinaryStruct):
    @binary_struct
    class Nav(BinaryStruct):
      gnss_id: Annotated[GnssType | int, enum(u8, GnssType)]
      sv_id: Annotated[int, u8]
      cno: Annotated[int, u8]
      elev: Annotated[int, s8]
      azim: Annotated[int, s16]
      pr_res: Annotated[int, s16]
      flags: Annotated[int, u32]

    itow: Annotated[int, u32]
    version: Annotated[int, u8]
    num_svs: Annotated[int, u8]
    reserved: Annotated[bytes, bytes_field(2)]
    svs: Annotated[list[Nav], array(Nav, count_field='num_svs')]

  @binary_struct
  class NavPvt(BinaryStruct):
    i_tow: Annotated[int, u32]
    year: Annotated[int, u16]
    month: Annotated[int, u8]
    day: Annotated[int, u8]
    hour: Annotated[int, u8]
    min: Annotated[int, u8]
    sec: Annotated[int, u8]
    valid: Annotated[int, u8]
    t_acc: Annotated[int, u32]
    nano: Annotated[int, s32]
    fix_type: Annotated[int, u8]
    flags: Annotated[int, u8]
    flags2: Annotated[int, u8]
    num_sv: Annotated[int, u8]
    lon: Annotated[int, s32]
    lat: Annotated[int, s32]
    height: Annotated[int, s32]
    h_msl: Annotated[int, s32]
    h_acc: Annotated[int, u32]
    v_acc: Annotated[int, u32]
    vel_n: Annotated[int, s32]
    vel_e: Annotated[int, s32]
    vel_d: Annotated[int, s32]
    g_speed: Annotated[int, s32]
    head_mot: Annotated[int, s32]
    s_acc: Annotated[int, s32]
    head_acc: Annotated[int, u32]
    p_dop: Annotated[int, u16]
    flags3: Annotated[int, u8]
    reserved1: Annotated[bytes, bytes_field(5)]
    head_veh: Annotated[int, s32]
    mag_dec: Annotated[int, s16]
    mag_acc: Annotated[int, u16]

  @binary_struct
  class MonHw2(BinaryStruct):
    class ConfigSource(IntEnum):
      flash = 102
      otp = 111
      config_pins = 112
      rom = 113

    ofs_i: Annotated[int, s8]
    mag_i: Annotated[int, u8]
    ofs_q: Annotated[int, s8]
    mag_q: Annotated[int, u8]
    cfg_source: Annotated[ConfigSource | int, enum(u8, ConfigSource)]
    reserved1: Annotated[bytes, bytes_field(3)]
    low_lev_cfg: Annotated[int, u32]
    reserved2: Annotated[bytes, bytes_field(8)]
    post_status: Annotated[int, u32]
    reserved3: Annotated[bytes, bytes_field(4)]

  @binary_struct
  class MonHw(BinaryStruct):
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

    pin_sel: Annotated[int, u32]
    pin_bank: Annotated[int, u32]
    pin_dir: Annotated[int, u32]
    pin_val: Annotated[int, u32]
    noise_per_ms: Annotated[int, u16]
    agc_cnt: Annotated[int, u16]
    a_status: Annotated[AntennaStatus | int, enum(u8, AntennaStatus)]
    a_power: Annotated[AntennaPower | int, enum(u8, AntennaPower)]
    flags: Annotated[int, u8]
    reserved1: Annotated[bytes, bytes_field(1)]
    used_mask: Annotated[int, u32]
    vp: Annotated[bytes, bytes_field(17)]
    jam_ind: Annotated[int, u8]
    reserved2: Annotated[bytes, bytes_field(2)]
    pin_irq: Annotated[int, u32]
    pull_h: Annotated[int, u32]
    pull_l: Annotated[int, u32]

  magic: Annotated[bytes, const(bytes_field(2), b"\xb5\x62")]
  msg_type: Annotated[int, u16be]
  length: Annotated[int, u16]
  body: Annotated[
    object,
    substream(
      'length',
      switch(
        'msg_type',
        {
          0x0107: NavPvt,
          0x0213: RxmSfrbx,
          0x0215: RxmRawx,
          0x0A09: MonHw,
          0x0A0B: MonHw2,
          0x0135: NavSat,
        },
      ),
    ),
  ]
  checksum: Annotated[int, u16]
