"""
Parses GPS navigation subframes per IS-GPS-200E specification.
https://www.gps.gov/technical/icwg/IS-GPS-200E.pdf
"""

from typing import Annotated

from openpilot.system.ubloxd import binary_struct as bs


class Gps(bs.BinaryStruct):
  class Tlm(bs.BinaryStruct):
    preamble: Annotated[bytes, bs.const(bs.bytes_field(1), b"\x8b")]
    tlm: Annotated[int, bs.bits(14)]
    integrity_status: Annotated[bool, bs.bits(1)]
    reserved: Annotated[bool, bs.bits(1)]

  class How(bs.BinaryStruct):
    tow_count: Annotated[int, bs.bits(17)]
    alert: Annotated[bool, bs.bits(1)]
    anti_spoof: Annotated[bool, bs.bits(1)]
    subframe_id: Annotated[int, bs.bits(3)]
    reserved: Annotated[int, bs.bits(2)]

  class Subframe1(bs.BinaryStruct):
    week_no: Annotated[int, bs.bits(10)]
    code: Annotated[int, bs.bits(2)]
    sv_accuracy: Annotated[int, bs.bits(4)]
    sv_health: Annotated[int, bs.bits(6)]
    iodc_msb: Annotated[int, bs.bits(2)]
    l2_p_data_flag: Annotated[bool, bs.bits(1)]
    reserved1: Annotated[int, bs.bits(23)]
    reserved2: Annotated[int, bs.bits(24)]
    reserved3: Annotated[int, bs.bits(24)]
    reserved4: Annotated[int, bs.bits(16)]
    t_gd: Annotated[int, bs.s8]
    iodc_lsb: Annotated[int, bs.u8]
    t_oc: Annotated[int, bs.u16be]
    af_2: Annotated[int, bs.s8]
    af_1: Annotated[int, bs.s16be]
    af_0_sign: Annotated[bool, bs.bits(1)]
    af_0_value: Annotated[int, bs.bits(21)]
    reserved5: Annotated[int, bs.bits(2)]

    @property
    def af_0(self) -> int:
      """Computed af_0 from sign-magnitude representation."""
      return (self.af_0_value - (1 << 21)) if self.af_0_sign else self.af_0_value

  class Subframe2(bs.BinaryStruct):
    iode: Annotated[int, bs.u8]
    c_rs: Annotated[int, bs.s16be]
    delta_n: Annotated[int, bs.s16be]
    m_0: Annotated[int, bs.s32be]
    c_uc: Annotated[int, bs.s16be]
    e: Annotated[int, bs.s32be]
    c_us: Annotated[int, bs.s16be]
    sqrt_a: Annotated[int, bs.u32be]
    t_oe: Annotated[int, bs.u16be]
    fit_interval_flag: Annotated[bool, bs.bits(1)]
    aoda: Annotated[int, bs.bits(5)]
    reserved: Annotated[int, bs.bits(2)]

  class Subframe3(bs.BinaryStruct):
    c_ic: Annotated[int, bs.s16be]
    omega_0: Annotated[int, bs.s32be]
    c_is: Annotated[int, bs.s16be]
    i_0: Annotated[int, bs.s32be]
    c_rc: Annotated[int, bs.s16be]
    omega: Annotated[int, bs.s32be]
    omega_dot_sign: Annotated[bool, bs.bits(1)]
    omega_dot_value: Annotated[int, bs.bits(23)]
    iode: Annotated[int, bs.u8]
    idot_sign: Annotated[bool, bs.bits(1)]
    idot_value: Annotated[int, bs.bits(13)]
    reserved: Annotated[int, bs.bits(2)]

    @property
    def omega_dot(self) -> int:
      """Computed omega_dot from sign-magnitude representation."""
      return (self.omega_dot_value - (1 << 23)) if self.omega_dot_sign else self.omega_dot_value

    @property
    def idot(self) -> int:
      """Computed idot from sign-magnitude representation."""
      return (self.idot_value - (1 << 13)) if self.idot_sign else self.idot_value

  class Subframe4(bs.BinaryStruct):
    class IonosphereData(bs.BinaryStruct):
      a0: Annotated[int, bs.s8]
      a1: Annotated[int, bs.s8]
      a2: Annotated[int, bs.s8]
      a3: Annotated[int, bs.s8]
      b0: Annotated[int, bs.s8]
      b1: Annotated[int, bs.s8]
      b2: Annotated[int, bs.s8]
      b3: Annotated[int, bs.s8]

    data_id: Annotated[int, bs.bits(2)]
    page_id: Annotated[int, bs.bits(6)]
    body: Annotated[object, bs.switch('page_id', {56: IonosphereData})]

  tlm: Tlm
  how: How
  body: Annotated[
    object,
    bs.switch(
      'how.subframe_id',
      {
        1: Subframe1,
        2: Subframe2,
        3: Subframe3,
        4: Subframe4,
      },
    ),
  ]
