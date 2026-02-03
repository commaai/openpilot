"""
UBX protocol parser
"""

from enum import IntEnum
from typing import Annotated

from openpilot.system.ubloxd import binary_struct as bs


class GnssType(IntEnum):
  gps = 0
  sbas = 1
  galileo = 2
  beidou = 3
  imes = 4
  qzss = 5
  glonass = 6


class Ubx(bs.BinaryStruct):
  GnssType = GnssType

  class RxmRawx(bs.BinaryStruct):
    class Measurement(bs.BinaryStruct):
      pr_mes: Annotated[float, bs.f64]
      cp_mes: Annotated[float, bs.f64]
      do_mes: Annotated[float, bs.f32]
      gnss_id: Annotated[GnssType | int, bs.enum(bs.u8, GnssType)]
      sv_id: Annotated[int, bs.u8]
      reserved2: Annotated[bytes, bs.bytes_field(1)]
      freq_id: Annotated[int, bs.u8]
      lock_time: Annotated[int, bs.u16]
      cno: Annotated[int, bs.u8]
      pr_stdev: Annotated[int, bs.u8]
      cp_stdev: Annotated[int, bs.u8]
      do_stdev: Annotated[int, bs.u8]
      trk_stat: Annotated[int, bs.u8]
      reserved3: Annotated[bytes, bs.bytes_field(1)]

    rcv_tow: Annotated[float, bs.f64]
    week: Annotated[int, bs.u16]
    leap_s: Annotated[int, bs.s8]
    num_meas: Annotated[int, bs.u8]
    rec_stat: Annotated[int, bs.u8]
    reserved1: Annotated[bytes, bs.bytes_field(3)]
    meas: Annotated[list[Measurement], bs.array(Measurement, count_field='num_meas')]

  class RxmSfrbx(bs.BinaryStruct):
    gnss_id: Annotated[GnssType | int, bs.enum(bs.u8, GnssType)]
    sv_id: Annotated[int, bs.u8]
    reserved1: Annotated[bytes, bs.bytes_field(1)]
    freq_id: Annotated[int, bs.u8]
    num_words: Annotated[int, bs.u8]
    reserved2: Annotated[bytes, bs.bytes_field(1)]
    version: Annotated[int, bs.u8]
    reserved3: Annotated[bytes, bs.bytes_field(1)]
    body: Annotated[list[int], bs.array(bs.u32, count_field='num_words')]

  class NavSat(bs.BinaryStruct):
    class Nav(bs.BinaryStruct):
      gnss_id: Annotated[GnssType | int, bs.enum(bs.u8, GnssType)]
      sv_id: Annotated[int, bs.u8]
      cno: Annotated[int, bs.u8]
      elev: Annotated[int, bs.s8]
      azim: Annotated[int, bs.s16]
      pr_res: Annotated[int, bs.s16]
      flags: Annotated[int, bs.u32]

    itow: Annotated[int, bs.u32]
    version: Annotated[int, bs.u8]
    num_svs: Annotated[int, bs.u8]
    reserved: Annotated[bytes, bs.bytes_field(2)]
    svs: Annotated[list[Nav], bs.array(Nav, count_field='num_svs')]

  class NavPvt(bs.BinaryStruct):
    i_tow: Annotated[int, bs.u32]
    year: Annotated[int, bs.u16]
    month: Annotated[int, bs.u8]
    day: Annotated[int, bs.u8]
    hour: Annotated[int, bs.u8]
    min: Annotated[int, bs.u8]
    sec: Annotated[int, bs.u8]
    valid: Annotated[int, bs.u8]
    t_acc: Annotated[int, bs.u32]
    nano: Annotated[int, bs.s32]
    fix_type: Annotated[int, bs.u8]
    flags: Annotated[int, bs.u8]
    flags2: Annotated[int, bs.u8]
    num_sv: Annotated[int, bs.u8]
    lon: Annotated[int, bs.s32]
    lat: Annotated[int, bs.s32]
    height: Annotated[int, bs.s32]
    h_msl: Annotated[int, bs.s32]
    h_acc: Annotated[int, bs.u32]
    v_acc: Annotated[int, bs.u32]
    vel_n: Annotated[int, bs.s32]
    vel_e: Annotated[int, bs.s32]
    vel_d: Annotated[int, bs.s32]
    g_speed: Annotated[int, bs.s32]
    head_mot: Annotated[int, bs.s32]
    s_acc: Annotated[int, bs.s32]
    head_acc: Annotated[int, bs.u32]
    p_dop: Annotated[int, bs.u16]
    flags3: Annotated[int, bs.u8]
    reserved1: Annotated[bytes, bs.bytes_field(5)]
    head_veh: Annotated[int, bs.s32]
    mag_dec: Annotated[int, bs.s16]
    mag_acc: Annotated[int, bs.u16]

  class MonHw2(bs.BinaryStruct):
    class ConfigSource(IntEnum):
      flash = 102
      otp = 111
      config_pins = 112
      rom = 113

    ofs_i: Annotated[int, bs.s8]
    mag_i: Annotated[int, bs.u8]
    ofs_q: Annotated[int, bs.s8]
    mag_q: Annotated[int, bs.u8]
    cfg_source: Annotated[ConfigSource | int, bs.enum(bs.u8, ConfigSource)]
    reserved1: Annotated[bytes, bs.bytes_field(3)]
    low_lev_cfg: Annotated[int, bs.u32]
    reserved2: Annotated[bytes, bs.bytes_field(8)]
    post_status: Annotated[int, bs.u32]
    reserved3: Annotated[bytes, bs.bytes_field(4)]

  class MonHw(bs.BinaryStruct):
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

    pin_sel: Annotated[int, bs.u32]
    pin_bank: Annotated[int, bs.u32]
    pin_dir: Annotated[int, bs.u32]
    pin_val: Annotated[int, bs.u32]
    noise_per_ms: Annotated[int, bs.u16]
    agc_cnt: Annotated[int, bs.u16]
    a_status: Annotated[AntennaStatus | int, bs.enum(bs.u8, AntennaStatus)]
    a_power: Annotated[AntennaPower | int, bs.enum(bs.u8, AntennaPower)]
    flags: Annotated[int, bs.u8]
    reserved1: Annotated[bytes, bs.bytes_field(1)]
    used_mask: Annotated[int, bs.u32]
    vp: Annotated[bytes, bs.bytes_field(17)]
    jam_ind: Annotated[int, bs.u8]
    reserved2: Annotated[bytes, bs.bytes_field(2)]
    pin_irq: Annotated[int, bs.u32]
    pull_h: Annotated[int, bs.u32]
    pull_l: Annotated[int, bs.u32]

  magic: Annotated[bytes, bs.const(bs.bytes_field(2), b"\xb5\x62")]
  msg_type: Annotated[int, bs.u16be]
  length: Annotated[int, bs.u16]
  body: Annotated[
    object,
    bs.substream(
      'length',
      bs.switch(
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
  checksum: Annotated[int, bs.u16]
