#!/usr/bin/env python3
import os

if __name__ == "__main__":
  dbc_name = os.path.basename(__file__).replace(".py", ".dbc")
  rivian_path = os.path.dirname(os.path.realpath(__file__))
  with open(os.path.join(rivian_path, dbc_name), "w", encoding='utf-8') as f:
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

    # note: 0x501/0x502 seem to be special in 0x5XX range
    for a in range(0x500, 0x500 + 32):
        f.write(f"""
BO_ {a} RADAR_TRACK_{a:x}: 8 RADAR
 SG_ CHECKSUM : 0|8@1+ (1,0) [0|255] "" XXX
 SG_ COUNTER : 11|4@0+ (1,0) [0|15] "" XXX
 SG_ UNKNOWN_1 : 23|8@0- (1,0) [-128|127] "" XXX
 SG_ AZIMUTH : 28|10@0- (0.1,0) [-61.2|62.1] "" XXX
 SG_ STATE : 31|3@0+ (1,0) [0|7] "" XXX
 SG_ LONG_DIST : 34|11@0+ (0.1,0) [0|204.7] "" XXX
 SG_ STATE_2 : 55|1@0+ (1,0) [0|1] "" XXX
 SG_ REL_SPEED : 53|14@0- (0.01,0) [-81.92|81.92] "" XXX
    """)
