"""
GPS subframe parser - declarative Python DSL implementation.

Parses GPS navigation subframes per IS-GPS-200E specification.
https://www.gps.gov/technical/icwg/IS-GPS-200E.pdf
"""

from typing import Annotated

from openpilot.system.ubloxd.binary_struct import (
  BinaryStruct,
  binary_struct,
  bits,
  bytes_field,
  const,
  s8,
  s16be,
  s32be,
  switch,
  u8,
  u16be,
  u32be,
)


@binary_struct
class Gps(BinaryStruct):
  @binary_struct
  class Tlm(BinaryStruct):
    preamble: Annotated[bytes, const(bytes_field(1), b"\x8b")]
    tlm: Annotated[int, bits(14)]
    integrity_status: Annotated[bool, bits(1)]
    reserved: Annotated[bool, bits(1)]

  @binary_struct
  class How(BinaryStruct):
    tow_count: Annotated[int, bits(17)]
    alert: Annotated[bool, bits(1)]
    anti_spoof: Annotated[bool, bits(1)]
    subframe_id: Annotated[int, bits(3)]
    reserved: Annotated[int, bits(2)]

  @binary_struct
  class Subframe1(BinaryStruct):
    week_no: Annotated[int, bits(10)]
    code: Annotated[int, bits(2)]
    sv_accuracy: Annotated[int, bits(4)]
    sv_health: Annotated[int, bits(6)]
    iodc_msb: Annotated[int, bits(2)]
    l2_p_data_flag: Annotated[bool, bits(1)]
    reserved1: Annotated[int, bits(23)]
    reserved2: Annotated[int, bits(24)]
    reserved3: Annotated[int, bits(24)]
    reserved4: Annotated[int, bits(16)]
    t_gd: Annotated[int, s8]
    iodc_lsb: Annotated[int, u8]
    t_oc: Annotated[int, u16be]
    af_2: Annotated[int, s8]
    af_1: Annotated[int, s16be]
    af_0_sign: Annotated[bool, bits(1)]
    af_0_value: Annotated[int, bits(21)]
    reserved5: Annotated[int, bits(2)]

    @property
    def af_0(self) -> int:
      """Computed af_0 from sign-magnitude representation."""
      return (self.af_0_value - (1 << 21)) if self.af_0_sign else self.af_0_value

  @binary_struct
  class Subframe2(BinaryStruct):
    iode: Annotated[int, u8]
    c_rs: Annotated[int, s16be]
    delta_n: Annotated[int, s16be]
    m_0: Annotated[int, s32be]
    c_uc: Annotated[int, s16be]
    e: Annotated[int, s32be]
    c_us: Annotated[int, s16be]
    sqrt_a: Annotated[int, u32be]
    t_oe: Annotated[int, u16be]
    fit_interval_flag: Annotated[bool, bits(1)]
    aoda: Annotated[int, bits(5)]
    reserved: Annotated[int, bits(2)]

  @binary_struct
  class Subframe3(BinaryStruct):
    c_ic: Annotated[int, s16be]
    omega_0: Annotated[int, s32be]
    c_is: Annotated[int, s16be]
    i_0: Annotated[int, s32be]
    c_rc: Annotated[int, s16be]
    omega: Annotated[int, s32be]
    omega_dot_sign: Annotated[bool, bits(1)]
    omega_dot_value: Annotated[int, bits(23)]
    iode: Annotated[int, u8]
    idot_sign: Annotated[bool, bits(1)]
    idot_value: Annotated[int, bits(13)]
    reserved: Annotated[int, bits(2)]

    @property
    def omega_dot(self) -> int:
      """Computed omega_dot from sign-magnitude representation."""
      return (self.omega_dot_value - (1 << 23)) if self.omega_dot_sign else self.omega_dot_value

    @property
    def idot(self) -> int:
      """Computed idot from sign-magnitude representation."""
      return (self.idot_value - (1 << 13)) if self.idot_sign else self.idot_value

  @binary_struct
  class Subframe4(BinaryStruct):
    @binary_struct
    class IonosphereData(BinaryStruct):
      a0: Annotated[int, s8]
      a1: Annotated[int, s8]
      a2: Annotated[int, s8]
      a3: Annotated[int, s8]
      b0: Annotated[int, s8]
      b1: Annotated[int, s8]
      b2: Annotated[int, s8]
      b3: Annotated[int, s8]

    data_id: Annotated[int, bits(2)]
    page_id: Annotated[int, bits(6)]
    body: Annotated[object, switch('page_id', {56: IonosphereData})]

  tlm: Tlm
  how: How
  body: Annotated[
    object,
    switch(
      'how.subframe_id',
      {
        1: Subframe1,
        2: Subframe2,
        3: Subframe3,
        4: Subframe4,
      },
    ),
  ]
