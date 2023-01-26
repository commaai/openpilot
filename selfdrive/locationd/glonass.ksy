# http://gauss.gge.unb.ca/GLONASS.ICD.pdf
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
      - id: x_vel
        type: b24
      - id: x_speedup
        type: b5
      - id: x
        type: b27
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
      - id: y_vel
        type: b24
      - id: y_speedup
        type: b5
      - id: y
        type: b27
  string_3:
    seq:
      - id: p3
        type: b1
      - id: gamma_n
        type: b11
      - id: not_used
        type: b1
      - id: p
        type: b2
      - id: l_n
        type: b1
      - id: z_vel
        type: b24
      - id: z_speedup
        type: b5
      - id: z
        type: b27
  string_4:
    seq:
      - id: tau_n
        type: b22
      - id: delta_tau_n
        type: b5
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
  string_non_immediate:
    seq:
      - id: data_1
        type: b64
      - id: data_2
        type: b8