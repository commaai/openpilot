meta:
  id: ubx
  endian: le
seq:
  - id: magic
    contents: [0xb5, 0x62]
  - id: msg_type
    type: u2be
  - id: length
    type: u2
  - id: body
    type:
      switch-on: msg_type
      cases:
        0x0107: nav_pvt
        0x0213: rxm_sfrbx
        0x0215: rxm_rawx
        0x0a09: mon_hw
        0x0a0b: mon_hw2
        0x0135: nav_sat
instances:
  checksum:
    pos: length + 6
    type: u2

types:
  mon_hw:
    seq:
      - id: pin_sel
        type: u4
      - id: pin_bank
        type: u4
      - id: pin_dir
        type: u4
      - id: pin_val
        type: u4
      - id: noise_per_ms
        type: u2
      - id: agc_cnt
        type: u2
      - id: a_status
        type: u1
        enum: antenna_status
      - id: a_power
        type: u1
        enum: antenna_power
      - id: flags
        type: u1
      - id: reserved1
        size: 1
      - id: used_mask
        type: u4
      - id: vp
        size: 17
      - id: jam_ind
        type: u1
      - id: reserved2
        size: 2
      - id: pin_irq
        type: u4
      - id: pull_h
        type: u4
      - id: pull_l
        type: u4
    enums:
      antenna_status:
        0: init
        1: dontknow
        2: ok
        3: short
        4: open
      antenna_power:
        0: off
        1: on
        2: dontknow

  mon_hw2:
    seq:
      - id: ofs_i
        type: s1
      - id: mag_i
        type: u1
      - id: ofs_q
        type: s1
      - id: mag_q
        type: u1
      - id: cfg_source
        type: u1
        enum: config_source
      - id: reserved1
        size: 3
      - id: low_lev_cfg
        type: u4
      - id: reserved2
        size: 8
      - id: post_status
        type: u4
      - id: reserved3
        size: 4

    enums:
      config_source:
        113: rom
        111: otp
        112: config_pins
        102: flash

  rxm_sfrbx:
    seq:
      - id: gnss_id
        type: u1
        enum: gnss_type
      - id: sv_id
        type: u1
      - id: reserved1
        size: 1
      - id: freq_id
        type: u1
      - id: num_words
        type: u1
      - id: reserved2
        size: 1
      - id: version
        type: u1
      - id: reserved3
        size: 1
      - id: body
        type: u4
        repeat: expr
        repeat-expr: num_words

  rxm_rawx:
    seq:
      - id: rcv_tow
        type: f8
      - id: week
        type: u2
      - id: leap_s
        type: s1
      - id: num_meas
        type: u1
      - id: rec_stat
        type: u1
      - id: reserved1
        size: 3
      - id: meas
        type: measurement
        size: 32
        repeat: expr
        repeat-expr: num_meas
    types:
      measurement:
        seq:
          - id: pr_mes
            type: f8
          - id: cp_mes
            type: f8
          - id: do_mes
            type: f4
          - id: gnss_id
            type: u1
            enum: gnss_type
          - id: sv_id
            type: u1
          - id: reserved2
            size: 1
          - id: freq_id
            type: u1
          - id: lock_time
            type: u2
          - id: cno
            type: u1
          - id: pr_stdev
            type: u1
          - id: cp_stdev
            type: u1
          - id: do_stdev
            type: u1
          - id: trk_stat
            type: u1
          - id: reserved3
            size: 1
  nav_sat:
    seq:
      - id: itow
        type: u4
      - id: version
        type: u1
      - id: num_svs
        type: u1
      - id: reserved
        size: 2
      - id: svs
        type: nav
        size: 12
        repeat: expr
        repeat-expr: num_svs
    types:
      nav:
        seq:
          - id: gnss_id
            type: u1
            enum: gnss_type
          - id: sv_id
            type: u1
          - id: cno
            type: u1
          - id: elev
            type: s1
          - id: azim
            type: s2
          - id: pr_res
            type: s2
          - id: flags
            type: u4

  nav_pvt:
    seq:
      - id: i_tow
        type: u4
      - id: year
        type: u2
      - id: month
        type: u1
      - id: day
        type: u1
      - id: hour
        type: u1
      - id: min
        type: u1
      - id: sec
        type: u1
      - id: valid
        type: u1
      - id: t_acc
        type: u4
      - id: nano
        type: s4
      - id: fix_type
        type: u1
      - id: flags
        type: u1
      - id: flags2
        type: u1
      - id: num_sv
        type: u1
      - id: lon
        type: s4
      - id: lat
        type: s4
      - id: height
        type: s4
      - id: h_msl
        type: s4
      - id: h_acc
        type: u4
      - id: v_acc
        type: u4
      - id: vel_n
        type: s4
      - id: vel_e
        type: s4
      - id: vel_d
        type: s4
      - id: g_speed
        type: s4
      - id: head_mot
        type: s4
      - id: s_acc
        type: s4
      - id: head_acc
        type: u4
      - id: p_dop
        type: u2
      - id: flags3
        type: u1
      - id: reserved1
        size: 5
      - id: head_veh
        type: s4
      - id: mag_dec
        type: s2
      - id: mag_acc
        type: u2
enums:
  gnss_type:
    0: gps
    1: sbas
    2: galileo
    3: beidou
    4: imes
    5: qzss
    6: glonass
