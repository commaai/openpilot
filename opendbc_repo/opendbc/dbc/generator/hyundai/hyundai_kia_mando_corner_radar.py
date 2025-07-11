#!/usr/bin/env python3
from collections import namedtuple
import os

if __name__ == "__main__":
  dbc_name = os.path.basename(__file__).replace(".py", ".dbc")
  hyundai_path = os.path.dirname(os.path.realpath(__file__))
  with open(os.path.join(hyundai_path, dbc_name), "w", encoding='utf-8') as f:
    f.write("""
VERSION ""


NS_ :
    NS_DESC_
    CM_
    BA_DEF_
    BA_
    VAL_
    CAT_DEF_
    CAT_
    FILTER
    BA_DEF_DEF_
    EV_DATA_
    ENVVAR_DATA_
    SGTYPE_
    SGTYPE_VAL_
    BA_DEF_SGTYPE_
    BA_SGTYPE_
    SIG_TYPE_REF_
    VAL_TABLE_
    SIG_GROUP_
    SIG_VALTYPE_
    SIGTYPE_VALTYPE_
    BO_TX_BU_
    BA_DEF_REL_
    BA_REL_
    BA_DEF_DEF_REL_
    BU_SG_REL_
    BU_EV_REL_
    BU_BO_REL_
    SG_MUL_VAL_

BS_:

BU_: XXX
""")

    for a in [0x100, 0x200]:
      f.write(f"""
BO_ {a} RADAR_POINTS_METADATA_0x{a:x}: 64 RADAR
 SG_ SIGNAL_1 : 0|32@1+ (1,0) [0|255] "" XXX
 SG_ SIGNAL_2 : 32|32@1+ (1,0) [0|65535] "" XXX
 SG_ SIGNAL_3 : 64|4@1+ (1,0) [0|15] "" XXX
 SG_ SIGNAL_4 : 68|4@1+ (1,0) [0|15] "" XXX
 SG_ RADAR_POINT_COUNT : 72|8@1+ (1,0) [0|255] "" XXX
 SG_ SIGNAL_6 : 80|7@1+ (0.015625,0) [0|3] "" XXX
 SG_ SIGNAL_7 : 87|1@1+ (1,0) [0|1] "" XXX
 SG_ SIGNAL_8 : 88|3@1+ (1,0) [0|7] "" XXX
 SG_ SIGNAL_9 : 91|5@1+ (0.0625,0) [0|31] "" XXX
 SG_ SIGNAL_10 : 96|8@1+ (1,0) [0|255] "" XXX
 SG_ SIGNAL_11 : 104|7@1+ (0.015625,0) [0|127] "" XXX
 SG_ SIGNAL_12 : 111|2@1+ (1,0) [0|65535] "" XXX
 SG_ SIGNAL_13 : 113|7@1+ (0.015625,0) [0|127] "" XXX
 SG_ SIGNAL_14 : 120|7@1+ (0.015625,0) [0|127] "" XXX
 SG_ SIGNAL_15 : 127|3@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_16 : 130|2@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_17 : 133|2@0+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_18 : 134|1@0+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_19 : 135|3@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_20 : 138|8@1+ (1,0) [0|63] "" XXX
 SG_ SIGNAL_21 : 146|2@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_22 : 148|1@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_23 : 149|4@1+ (1,0) [0|7] "" XXX
 SG_ SIGNAL_24 : 153|1@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_25 : 154|2@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_26 : 157|2@0+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_27 : 158|7@1+ (0.125,0) [0|3] "" XXX
 SG_ SIGNAL_28 : 165|7@1+ (0.015625,0) [0|31] "" XXX
 SG_ SIGNAL_29 : 172|7@1+ (0.125,0) [0|3] "" XXX
 SG_ SIGNAL_30 : 179|7@1+ (0.015625,0) [0|1] "" XXX
 SG_ SIGNAL_31 : 186|4@1+ (1,0) [0|7] "" XXX
 SG_ SIGNAL_32 : 190|14@1+ (0.015625,0) [0|15] "" XXX
 SG_ SIGNAL_33 : 204|11@1+ (0.03125,0) [0|8191] "" XXX
 SG_ SIGNAL_34 : 215|2@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_35 : 217|7@1+ (1,0) [0|127] "" XXX
 SG_ SIGNAL_36 : 224|6@1+ (1,0) [0|63] "" XXX
 SG_ SIGNAL_37 : 230|6@1+ (0.2,0) [0|31] "" XXX
 SG_ SIGNAL_38 : 236|6@1+ (0.2,0) [0|7] "" XXX
 SG_ SIGNAL_39 : 242|8@1+ (1,-90) [0|255] "" XXX
 SG_ SIGNAL_40 : 250|6@1+ (1,0) [0|63] "" XXX
 SG_ SIGNAL_41 : 256|8@1+ (0.25,0) [0|255] "" XXX
 SG_ SIGNAL_42 : 264|3@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_43 : 267|12@1+ (0.01,0) [0|31] "" XXX
 SG_ SIGNAL_44 : 279|32@1+ (1,0) [0|63] "" XXX
 SG_ SIGNAL_45 : 311|1@1+ (1,0) [0|1] "" XXX
 SG_ SIGNAL_46 : 312|2@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_47 : 314|32@1+ (1,0) [0|255] "" XXX
 SG_ SIGNAL_48 : 346|6@1+ (1,0) [0|63] "" XXX
 SG_ SIGNAL_49 : 352|7@1+ (0.25,0) [0|127] "" XXX
 SG_ SIGNAL_50 : 359|6@1+ (0.03125,0) [0|31] "" XXX
 SG_ SIGNAL_51 : 365|10@1+ (0.125,0) [0|3] "" XXX
 SG_ SIGNAL_52 : 375|10@1+ (0.125,0) [0|63] "" XXX
 SG_ SIGNAL_53 : 385|7@1+ (1,0) [0|127] "" XXX
 SG_ SIGNAL_54 : 392|7@1+ (1,0) [0|127] "" XXX
 SG_ SIGNAL_55 : 399|8@1+ (0.00390625,0) [0|31] "" XXX
 SG_ SIGNAL_56 : 407|10@1+ (0.125,0) [0|63] "" XXX
 SG_ SIGNAL_57 : 417|1@1+ (1,0) [0|3] "" XXX
 SG_ SIGNAL_58 : 418|1@1+ (1,0) [0|3] "" XXX
""")

    # radar points are sent at 20 Hz in groups of 1 to 13 messages
    # each message has 5 radar points for a total of 65 points max
    # each radar point is 101 bits so the alignment is not consistent
    RadarPointSignal = namedtuple("RadarPointSignal", ["name", "start", "length", "scale", "offset"])
    radar_point_signals = (
      RadarPointSignal("DISTANCE", 7, 14, 1/64, 0),
      RadarPointSignal("", 21, 2, 1, 0),
      RadarPointSignal("", 23, 8, 1/512, -127/512),
      RadarPointSignal("REL_VELOCITY", 31, 13, 1/32, -66),
      RadarPointSignal("", 44, 2, 1, 0),
      RadarPointSignal("", 46, 2, 1, 0),
      RadarPointSignal("AZIMUTH", 48, 12, 1/512, -2047/512),
      RadarPointSignal("", 60, 2, 1, 0),
      RadarPointSignal("", 62, 1, 1, 0),
      RadarPointSignal("", 63, 7, 1, 0),
      RadarPointSignal("", 70, 1, 1, 0),
      RadarPointSignal("", 71, 6, 1, 0),
      RadarPointSignal("", 77, 2, 1, 0),
      RadarPointSignal("", 79, 8, 1/512, -127/512),
      RadarPointSignal("", 87, 1, 1, 0),
      RadarPointSignal("", 88, 2, 1, 0),
      RadarPointSignal("", 90, 3, 1, 0),
      # last 15 bits are controlled by LAYOUT_ID (seems to always zero, so below is layout 0)
      RadarPointSignal("", 93, 6, 1, 0),
      RadarPointSignal("", 99, 8, 1, 0),
      RadarPointSignal("", 107, 1, 1, 0),
    )
    radar_point_bit_count = sum([s.length for s in radar_point_signals])

    for a in [0x101, 0x201]:
      f.write(f"""
BO_ {a} RADAR_POINTS_0x{a:x}: 64 RADAR
 SG_ MESSAGE_ID : 0|5@1+ (1,0) [0|31] "" XXX
 SG_ LAYOUT_ID : 5|2@1+ (1,0) [0|3] "" XXX
""")
      bit_idx = radar_point_signals[0].start
      for i in range(5):
        signal_idx = 1
        for sig in radar_point_signals:
          if sig.name:
            sig_name = f"POINT_{i+1}_{sig.name}"
          else:
            sig_name = f"POINT_{i+1}_SIGNAL_{signal_idx}"
          signal_idx += 1

          sig_start_idx = i * radar_point_bit_count + sig.start
          assert bit_idx == sig_start_idx, f"signal overlap or gap!!! {bit_idx} != {sig_start_idx}"
          min_val = round(sig.offset, 10)
          max_val = round((2**sig.length - 1) * sig.scale + sig.offset, 10)

          f.write(f" SG_ {sig_name} : {sig_start_idx}|{sig.length}@1+ ({sig.scale},{sig.offset}) [{min_val}|{max_val}] \"\" XXX\n")
          bit_idx += sig.length

    # checksum is across a group of 0x100/200 and 0x101/201 messages (no checksums inside the other messages)
    # ccitt_crc16 = mkCrcFun(0x11021, initCrc=0xffff, xorOut=0x0000, rev=False)
    for a in [0x104, 0x204]:
      f.write(f"""
BO_ {a} RADAR_POINTS_CHECKSUM_0x{a:x}: 3 RADAR
 SG_ CRC16 : 0|16@1+ (1,0) [0|65535] "" XXX
""")
