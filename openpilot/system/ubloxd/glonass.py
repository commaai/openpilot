"""
Parses GLONASS navigation strings per GLONASS ICD specification.
http://gauss.gge.unb.ca/GLONASS.ICD.pdf
https://www.unavco.org/help/glossary/docs/ICD_GLONASS_4.0_(1998)_en.pdf
"""

from typing import Annotated

from openpilot.system.ubloxd import binary_struct as bs


class Glonass(bs.BinaryStruct):
  class String1(bs.BinaryStruct):
    not_used: Annotated[int, bs.bits(2)]
    p1: Annotated[int, bs.bits(2)]
    t_k: Annotated[int, bs.bits(12)]
    x_vel_sign: Annotated[bool, bs.bits(1)]
    x_vel_value: Annotated[int, bs.bits(23)]
    x_accel_sign: Annotated[bool, bs.bits(1)]
    x_accel_value: Annotated[int, bs.bits(4)]
    x_sign: Annotated[bool, bs.bits(1)]
    x_value: Annotated[int, bs.bits(26)]

    @property
    def x_vel(self) -> int:
      """Computed x_vel from sign-magnitude representation."""
      return (self.x_vel_value * -1) if self.x_vel_sign else self.x_vel_value

    @property
    def x_accel(self) -> int:
      """Computed x_accel from sign-magnitude representation."""
      return (self.x_accel_value * -1) if self.x_accel_sign else self.x_accel_value

    @property
    def x(self) -> int:
      """Computed x from sign-magnitude representation."""
      return (self.x_value * -1) if self.x_sign else self.x_value

  class String2(bs.BinaryStruct):
    b_n: Annotated[int, bs.bits(3)]
    p2: Annotated[bool, bs.bits(1)]
    t_b: Annotated[int, bs.bits(7)]
    not_used: Annotated[int, bs.bits(5)]
    y_vel_sign: Annotated[bool, bs.bits(1)]
    y_vel_value: Annotated[int, bs.bits(23)]
    y_accel_sign: Annotated[bool, bs.bits(1)]
    y_accel_value: Annotated[int, bs.bits(4)]
    y_sign: Annotated[bool, bs.bits(1)]
    y_value: Annotated[int, bs.bits(26)]

    @property
    def y_vel(self) -> int:
      """Computed y_vel from sign-magnitude representation."""
      return (self.y_vel_value * -1) if self.y_vel_sign else self.y_vel_value

    @property
    def y_accel(self) -> int:
      """Computed y_accel from sign-magnitude representation."""
      return (self.y_accel_value * -1) if self.y_accel_sign else self.y_accel_value

    @property
    def y(self) -> int:
      """Computed y from sign-magnitude representation."""
      return (self.y_value * -1) if self.y_sign else self.y_value

  class String3(bs.BinaryStruct):
    p3: Annotated[bool, bs.bits(1)]
    gamma_n_sign: Annotated[bool, bs.bits(1)]
    gamma_n_value: Annotated[int, bs.bits(10)]
    not_used: Annotated[bool, bs.bits(1)]
    p: Annotated[int, bs.bits(2)]
    l_n: Annotated[bool, bs.bits(1)]
    z_vel_sign: Annotated[bool, bs.bits(1)]
    z_vel_value: Annotated[int, bs.bits(23)]
    z_accel_sign: Annotated[bool, bs.bits(1)]
    z_accel_value: Annotated[int, bs.bits(4)]
    z_sign: Annotated[bool, bs.bits(1)]
    z_value: Annotated[int, bs.bits(26)]

    @property
    def gamma_n(self) -> int:
      """Computed gamma_n from sign-magnitude representation."""
      return (self.gamma_n_value * -1) if self.gamma_n_sign else self.gamma_n_value

    @property
    def z_vel(self) -> int:
      """Computed z_vel from sign-magnitude representation."""
      return (self.z_vel_value * -1) if self.z_vel_sign else self.z_vel_value

    @property
    def z_accel(self) -> int:
      """Computed z_accel from sign-magnitude representation."""
      return (self.z_accel_value * -1) if self.z_accel_sign else self.z_accel_value

    @property
    def z(self) -> int:
      """Computed z from sign-magnitude representation."""
      return (self.z_value * -1) if self.z_sign else self.z_value

  class String4(bs.BinaryStruct):
    tau_n_sign: Annotated[bool, bs.bits(1)]
    tau_n_value: Annotated[int, bs.bits(21)]
    delta_tau_n_sign: Annotated[bool, bs.bits(1)]
    delta_tau_n_value: Annotated[int, bs.bits(4)]
    e_n: Annotated[int, bs.bits(5)]
    not_used_1: Annotated[int, bs.bits(14)]
    p4: Annotated[bool, bs.bits(1)]
    f_t: Annotated[int, bs.bits(4)]
    not_used_2: Annotated[int, bs.bits(3)]
    n_t: Annotated[int, bs.bits(11)]
    n: Annotated[int, bs.bits(5)]
    m: Annotated[int, bs.bits(2)]

    @property
    def tau_n(self) -> int:
      """Computed tau_n from sign-magnitude representation."""
      return (self.tau_n_value * -1) if self.tau_n_sign else self.tau_n_value

    @property
    def delta_tau_n(self) -> int:
      """Computed delta_tau_n from sign-magnitude representation."""
      return (self.delta_tau_n_value * -1) if self.delta_tau_n_sign else self.delta_tau_n_value

  class String5(bs.BinaryStruct):
    n_a: Annotated[int, bs.bits(11)]
    tau_c: Annotated[int, bs.bits(32)]
    not_used: Annotated[bool, bs.bits(1)]
    n_4: Annotated[int, bs.bits(5)]
    tau_gps: Annotated[int, bs.bits(22)]
    l_n: Annotated[bool, bs.bits(1)]

  class StringNonImmediate(bs.BinaryStruct):
    data_1: Annotated[int, bs.bits(64)]
    data_2: Annotated[int, bs.bits(8)]

  idle_chip: Annotated[bool, bs.bits(1)]
  string_number: Annotated[int, bs.bits(4)]
  data: Annotated[
    object,
    bs.switch(
      'string_number',
      {
        1: String1,
        2: String2,
        3: String3,
        4: String4,
        5: String5,
      },
      default=StringNonImmediate,
    ),
  ]
  hamming_code: Annotated[int, bs.bits(8)]
  pad_1: Annotated[int, bs.bits(11)]
  superframe_number: Annotated[int, bs.bits(16)]
  pad_2: Annotated[int, bs.bits(8)]
  frame_number: Annotated[int, bs.bits(8)]
