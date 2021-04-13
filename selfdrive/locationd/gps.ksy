meta:
  id: gps
  endian: be
  bit-endian: be
seq:
  - id: tlm
    type: tlm
  - id: how
    type: how


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

