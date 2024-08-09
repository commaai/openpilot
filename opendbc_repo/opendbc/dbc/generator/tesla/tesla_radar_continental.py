#!/usr/bin/env python3

import os
from opendbc.dbc.generator.tesla.radar_common import get_radar_point_definition, get_val_definition

if __name__ == "__main__":
  dbc_name = os.path.basename(__file__).replace(".py", ".dbc")
  tesla_path = os.path.dirname(os.path.realpath(__file__))
  with open(os.path.join(tesla_path, dbc_name), "w", encoding='utf-8') as f:
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

BU_:  Autopilot Radar Diag

BO_ 1025 RadarStatus: 8 Radar
   SG_ carparkDetected : 29|1@1+ (1,0) [0|1] "" Autopilot
   SG_ decreaseBlockage : 25|1@1+ (1,0) [0|1] "" Autopilot
   SG_ horizontMisalignment : 8|12@1+ (0.00012207,-0.25) [-0.25|0.249878] "rad" Autopilot
   SG_ increaseBlockage : 24|1@1+ (1,0) [0|1] "" Autopilot
   SG_ lowPowerMode : 20|2@1+ (1,0) [0|3] "" Autopilot
   SG_ powerOnSelfTest : 22|1@1+ (1,0) [0|1] "" Autopilot
   SG_ sensorBlocked : 26|1@1+ (1,0) [0|1] "" Autopilot
   SG_ sensorInfoConsistBit : 30|1@1+ (1,0) [0|1] "" Autopilot
   SG_ sensorReplace : 31|1@1+ (1,0) [0|1] "" Autopilot
   SG_ shortTermUnavailable : 23|1@1+ (1,0) [0|1] "" Autopilot
   SG_ tunnelDetected : 28|1@1+ (1,0) [0|1] "" Autopilot
   SG_ vehDynamicsError : 27|1@1+ (1,0) [0|1] "" Autopilot
   SG_ verticalMisalignment : 0|8@1+ (0.00195313,-0.25) [-0.25|0.248047] "rad" Autopilot

BO_ 1617 Radar_udsResponse: 8 Radar
   SG_ Radar_udsResponseData : 7|64@0+ (1,0) [0|1.84467e+19] "" Diag

BO_ 1601 UDS_radcRequest: 8 Diag
   SG_ UDS_radcRequestData : 7|64@0+ (1,0) [0|1.84467e+19] "" Radar
""")

    POINT_RANGE = range(0x410, 0x45E + 1, 2)
    for i, base_id in enumerate(POINT_RANGE):
      f.write(get_radar_point_definition(base_id, f"RadarPoint{i}"))

    f.write("""
VAL_ 1025 lowPowerMode 1 "COMMANDED_LOW_POWER" 0 "DEFAULT_LOW_POWER" 2 "NORMAL_POWER" 3 "SNA";""")

    for base_id in list(POINT_RANGE):
      f.write(get_val_definition(base_id))
