meta:
  id: gps
  endian: be
  bit-endian: be
seq:
  - id: tlm
    type: tlm
  - id: how
    type: how
  - id: body
    type:
      switch-on: how.subframe_id
      cases:
        1: subframe_1
        2: subframe_2
        3: subframe_3
types:
  tlm:
   seq:
     - id: magic
       contents: [0x8b]
     - id: tlm
       type: b14
     - id: integrity_status
       type: b1
     - id: reserved
       type: b1
  how:
    seq:
      - id: tow_count
        type: b17
      - id: alert
        type: b1
      - id: anti_spoof
        type: b1
      - id: subframe_id
        type: b3
      - id: reserved
        type: b2

  # https://www.gps.gov/technical/icwg/IS-GPS-200E.pdf
  # Page 77
  subframe_1:
    seq:
      # Word 3
      - id: week_no
        type: b10
      - id: code
        type: b2
      - id: sv_accuracy
        type: b4
      - id: sv_health
        type: b6
      - id: iodc_msb
        type: b2
      # Word 4
      - id: l2_p_data_flag
        type: b1
      - id: reserved1
        type: b23
      # Word 5
      - id: reserved2
        type: b24
      # Word 6
      - id: reserved3
        type: b24
      # Word 7
      - id: reserved4
        type: b16
      - id: t_gd
        type: s1
      # Word 8
      - id: iodc_lsb
        type: u1
      - id: t_oc
        type: u2
      # Word 9
      - id: af_2
        type: s1
      - id: af_1
        type: s2
      # Word 10
      - id: af_0
        type: b22
      - id: reserved5
        type: b2
  subframe_2:
    seq:
      # Word 3
      - id: iode
        type: u1
      - id: c_rs
        type: s2
      # Word 4 & 5
      - id: delta_n
        type: s2
      - id: m_0
        type: s4
      # Word 6 & 7
      - id: c_uc
        type: s2
      - id: e
        type: s4
      # Word 8 & 9
      - id: c_us
        type: s2
      - id: sqrt_a
        type: u4
      # Word 10
      - id: t_oe
        type: u2
      - id: fit_interval_flag
        type: b1
      - id: aoda
        type: b5
      - id: reserved
        type: b2

  subframe_3:
    seq:
      # Word 3 & 4
      - id: c_ic
        type: s2
      - id: omega_0
        type: s4
      # Word 5 & 6
      - id: c_is
        type: s2
      - id: i_0
        type: s4
      # Word 7 & 8
      - id: c_rc
        type: s2
      - id: omega
        type: s4
      # Word 9
      - id: omega_dot
        type: b24
      # Word 10
      - id: iode
        type: u1
      - id: idot
        type: b14
      - id: reserved
        type: b2


