# http://gauss.gge.unb.ca/GLONASS.ICD.pdf
# some variables are misprinted but good in the old doc
# https://www.unavco.org/help/glossary/docs/ICD_GLONASS_4.0_(1998)_en.pdf
meta:
  id: glonass
  endian: be
  bit-endian: be
seq:
  - id: idle_chip
    type: b1
  - id: string_number
    type: b4
  - id: data
    type:
      switch-on: string_number
      cases:
        1: string_1
        2: string_2
        3: string_3
        4: string_4
        5: string_5
        _: string_non_immediate
  - id: hamming_code
    type: b8
  - id: pad_1
    type: b11
  - id: superframe_number
    type: b16
  - id: pad_2
    type: b8
  - id: frame_number
    type: b8

types:
  string_1:
    seq:
      - id: not_used
        type: b2
      - id: p1
        type: b2
      - id: t_k
        type: b12
      - id: x_vel_sign
        type: b1
      - id: x_vel_value
        type: b23
      - id: x_accel_sign
        type: b1
      - id: x_accel_value
        type: b4
      - id: x_sign
        type: b1
      - id: x_value
        type: b26
    instances:
      x_vel:
        value: 'x_vel_sign ? (x_vel_value * (-1)) : x_vel_value'
      x_accel:
        value: 'x_accel_sign ? (x_accel_value * (-1)) : x_accel_value'
      x:
        value: 'x_sign ? (x_value * (-1)) : x_value'
  string_2:
    seq:
      - id: b_n
        type: b3
      - id: p2
        type: b1
      - id: t_b
        type: b7
      - id: not_used
        type: b5
      - id: y_vel_sign
        type: b1
      - id: y_vel_value
        type: b23
      - id: y_accel_sign
        type: b1
      - id: y_accel_value
        type: b4
      - id: y_sign
        type: b1
      - id: y_value
        type: b26
    instances:
      y_vel:
        value: 'y_vel_sign ? (y_vel_value * (-1)) : y_vel_value'
      y_accel:
        value: 'y_accel_sign ? (y_accel_value * (-1)) : y_accel_value'
      y:
        value: 'y_sign ? (y_value * (-1)) : y_value'
  string_3:
    seq:
      - id: p3
        type: b1
      - id: gamma_n_sign
        type: b1
      - id: gamma_n_value
        type: b10
      - id: not_used
        type: b1
      - id: p
        type: b2
      - id: l_n
        type: b1
      - id: z_vel_sign
        type: b1
      - id: z_vel_value
        type: b23
      - id: z_accel_sign
        type: b1
      - id: z_accel_value
        type: b4
      - id: z_sign
        type: b1
      - id: z_value
        type: b26
    instances:
      gamma_n:
        value: 'gamma_n_sign ? (gamma_n_value * (-1)) : gamma_n_value'
      z_vel:
        value: 'z_vel_sign ? (z_vel_value * (-1)) : z_vel_value'
      z_accel:
        value: 'z_accel_sign ? (z_accel_value * (-1)) : z_accel_value'
      z:
        value: 'z_sign ? (z_value * (-1)) : z_value'
  string_4:
    seq:
      - id: tau_n_sign
        type: b1
      - id: tau_n_value
        type: b21
      - id: delta_tau_n_sign
        type: b1
      - id: delta_tau_n_value
        type: b4
      - id: e_n
        type: b5
      - id: not_used_1
        type: b14
      - id: p4
        type: b1
      - id: f_t
        type: b4
      - id: not_used_2
        type: b3
      - id: n_t
        type: b11
      - id: n
        type: b5
      - id: m
        type: b2
    instances:
      tau_n:
        value: 'tau_n_sign ? (tau_n_value * (-1)) : tau_n_value'
      delta_tau_n:
        value: 'delta_tau_n_sign ? (delta_tau_n_value * (-1)) : delta_tau_n_value'
  string_5:
    seq:
      - id: n_a
        type: b11
      - id: tau_c
        type: b32
      - id: not_used
        type: b1
      - id: n_4
        type: b5
      - id: tau_gps
        type: b22
      - id: l_n
        type: b1
  string_non_immediate:
    seq:
      - id: data_1
        type: b64
      - id: data_2
        type: b8
