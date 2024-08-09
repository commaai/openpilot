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


BO_ 769 TeslaRadarSguInfo: 8 Radar
 SG_ RADC_VerticalMisalignment : 0|8@1+ (1,0) [0|255] ""  Autopilot
 SG_ RADC_SCUTemperature : 8|8@1+ (1,-128) [-128|127] ""  Autopilot
 SG_ RADC_VMA_Plaus : 16|8@1+ (1,0) [0|255] ""  Autopilot
 SG_ RADC_SGU_ITC : 24|8@1+ (1,0) [0|255] ""  Autopilot
 SG_ RADC_HorizontMisalignment : 32|12@1+ (1,0) [0|4096] ""  Autopilot
 SG_ RADC_SensorDirty : 44|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_HWFail : 45|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_SGUFail : 46|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_SGUInfoConsistBit : 47|1@1+ (1,0) [0|1] ""  Autopilot

BO_ 770 TeslaRadarTguInfo: 8 Radar
 SG_ RADC_ACCTargObj1_sguIndex : 0|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ RADC_ACCTargObj2_sguIndex : 6|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ RADC_ACCTargObj3_sguIndex : 12|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ RADC_ACCTargObj4_sguIndex : 18|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ RADC_ACCTargObj5_sguIndex : 24|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ unused30 : 30|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_TGUInfoConsistBit : 31|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_ACCTargObj1_dBPower : 32|16@1+ (1,0) [0|65535] ""  Autopilot
 SG_ RADC_ACCTargObj5_dBPower : 48|16@1+ (1,0) [0|65535] ""  Autopilot

BO_ 1281 TeslaRadarAlertMatrix: 8 Radar
 SG_ RADC_a001_ecuInternalPerf : 0|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a002_flashPerformance : 1|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a003_vBatHigh : 2|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a004_adjustmentNotDone : 3|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a005_adjustmentReq : 4|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a006_adjustmentNotOk : 5|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a007_sensorBlinded : 6|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a008_plantModeActive : 7|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a009_configMismatch : 8|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a010_canBusOff : 9|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a011_bdyMIA : 10|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a012_espMIA : 11|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a013_gtwMIA : 12|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a014_sccmMIA : 13|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a015_adasMIA : 14|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a016_bdyInvalidCount : 15|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a017_adasInvalidCount : 16|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a018_espInvalidCount : 17|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a019_sccmInvalidCount : 18|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a020_bdyInvalidChkSm : 19|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a021_espInvalidChkSm : 20|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a022_sccmInvalidChkSm : 21|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a023_sccmInvalidChkSm : 22|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a024_absValidity : 23|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a025_ambTValidity : 24|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a026_brakeValidity : 25|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a027_CntryCdValidity : 26|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a028_espValidity : 27|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a029_longAccOffValidity : 28|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a030_longAccValidity : 29|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a031_odoValidity : 30|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a032_gearValidity : 31|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a033_steerAngValidity : 32|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a034_steerAngSpdValidity : 33|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a035_indctrValidity : 34|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a036_vehStandStillValidity : 35|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a037_vinValidity : 36|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a038_whlRotValidity : 37|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a039_whlSpdValidity : 38|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a040_whlStandStillValidity : 39|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a041_wiperValidity : 40|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a042_xwdValidity : 41|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a043_yawOffValidity : 42|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a044_yawValidity : 43|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a045_bsdSanity : 44|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a046_rctaSanity : 45|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a047_lcwSanity : 46|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a048_steerAngOffSanity : 47|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a049_tireSizeSanity : 48|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a050_velocitySanity : 49|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a051_yawSanity : 50|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a052_radomeHtrInop : 51|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a053_espmodValidity : 52|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a054_gtwmodValidity : 53|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a055_stwmodValidity : 54|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a056_bcmodValidity : 55|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a057_dimodValidity : 56|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a058_opmodValidity : 57|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a059_drmiInvalidChkSm : 58|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a060_drmiInvalidCount : 59|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a061_radPositionMismatch : 60|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ RADC_a062_strRackMismatch : 61|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ unused62 : 62|2@1+ (1,0) [0|3] ""  Autopilot
""")

    M_RANGE = range(0x310, 0x36D + 1, 3)
    for i, base_id in enumerate(M_RANGE):
      f.write(get_radar_point_definition(base_id, f"RadarPoint{i}"))

    L_RANGE = range(0x371, 0x37D + 1, 3)
    for i, base_id in enumerate(L_RANGE):
      f.write(get_radar_point_definition(base_id, f"ProcessedRadarPoint{i+1}"))

    f.write("""
BO_ 697 VIN_VIP_405HS: 8 Autopilot
 SG_ VIN_MuxID M : 0|8@1+ (1,0) [0|0] ""  Radar
 SG_ VIN_Part1 m16 : 47|24@0+ (1,0) [0|16777215] "" Radar
 SG_ VIN_Part2 m17 : 15|56@0+ (1,0) [0|7.2057594038E+16] "" Radar
 SG_ VIN_Part3 m18 : 15|56@0+ (1,0) [0|7.2057594038E+16] "" Radar

BO_ 681 Msg2A9_GTW_carConfig: 8 Autopilot
 SG_ Msg2A9_Always0x02 : 48|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg2A9_Always0x10 : 56|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg2A9_Always0x16 : 8|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg2A9_Always0x41 : 24|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg2A9_Value1_0x02 : 0|3@1+ (1,0) [0|0] "" Radar
 SG_ Msg2A9_FourWheelDrive : 3|2@1+ (1,0) [0|0] "" Radar
 SG_ Msg2A9_Value2_0x02 : 5|3@1+ (1,0) [0|0] ""  Radar
 SG_ Msg2A9_Always0x43 : 16|8@1+ (1,0) [0|0] ""  Radar

BO_ 409 Msg199_STW_ANGLHP_STAT: 8 Autopilot
 SG_ Msg199Always0x04 : 32|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg199Always0x20 : 16|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg199Always0x2F : 0|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg199Always0x67 : 8|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg199Always0xFF : 40|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg199Checksum : 56|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg199Counter : 52|4@1+ (1,0) [0|0] ""  Radar

BO_ 361 Msg169_ESP_wheelSpeeds: 8 Autopilot
 SG_ ESP_wheelSpeedFrL_HS : 0|13@1+ (0.04,0) [0|327.64] "km/h" Radar
 SG_ ESP_wheelSpeedFrR_HS : 13|13@1+ (0.04,0) [0|327.64] "km/h" Radar
 SG_ ESP_wheelSpeedReL_HS : 26|13@1+ (0.04,0) [0|327.64] "km/h" Radar
 SG_ ESP_wheelSpeedReR_HS : 39|13@1+ (0.04,0) [0|327.64] "km/h" Radar
 SG_ Msg169Checksum : 56|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg169Counter : 52|4@1+ (1,0) [0|0] ""  Radar

BO_ 345 Msg159_ESP_C: 8 Autopilot
 SG_ Msg159Always0x3A : 16|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg159Always0xA5 : 0|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg159Always0xCF : 32|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg159Always0xF4 : 8|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg159Counter : 44|4@1+ (1,0) [0|0] ""  Radar
 SG_ Msg159Checksum : 24|8@1+ (1,0) [0|0] ""  Radar

BO_ 329 Msg149_ESP_145h: 8 Autopilot
 SG_ Msg149Always0x02 : 16|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg149Always0x04 : 40|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg149Always0x26 : 8|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg149Always0x6A : 24|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg149Always0xAA : 32|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg149Always0xF : 48|4@1+ (1,0) [0|0] ""  Radar
 SG_ Msg149Checksum : 56|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg149Counter : 52|4@1+ (1,0) [0|0] ""  Radar

BO_ 297 Msg129_ESP_115h: 6 Autopilot
 SG_ Msg129Always0x20 : 24|8@1+ (1,0) [0|0] "" Radar
 SG_ Msg129Checksum : 40|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg129Counter : 36|4@1+ (1,0) [0|0] ""  Radar

BO_ 281 Msg119_DI_torque2: 6 Autopilot
 SG_ Msg119Always0x11 : 24|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg119Always0x1F : 8|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg119Always0x8 : 36|4@1+ (1,0) [0|0] ""  Radar
 SG_ Msg119Always0xF4 : 16|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg119Always0xFF : 0|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg119Checksum : 40|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg119Counter : 32|4@1+ (1,0) [0|0] ""  Radar

BO_ 265 Msg109_DI_torque1: 8 Autopilot
 SG_ Msg109Always0x80 : 24|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg109Checksum : 56|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg109Counter : 13|3@1+ (1,0) [0|0] ""  Radar

BO_ 521 Msg209_GTW_odo: 8 Autopilot
 SG_ Msg209Always0x61 : 8|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg209Always0x94 : 16|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg209Always0x52 : 24|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg209Always0x13 : 32|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg209Always0x03 : 40|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg209Always0x80 : 48|8@1+ (1,0) [0|0] ""  Radar

BO_ 537 Msg219_STW_ACTN_RQ: 8 Autopilot
 SG_ Msg219Counter : 52|4@1+ (1,0) [0|15] "" Radar
 SG_ Msg219CRC : 56|8@1+ (1,0) [0|0] "" Radar

BO_ 425 Msg1A9_DI_espControl: 5 Autopilot
 SG_ Msg1A9Always0x0C : 16|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg1A9Counter : 28|4@1+ (1,0) [0|0] ""  Radar
 SG_ Msg1A9Checksum : 32|8@1+ (1,0) [0|0] ""  Radar

BO_ 729 Msg2D9_BC_status : 8 Autopilot
 SG_ Msg2D9Always0x80 : 0|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg2D9Always0x40 : 8|8@1+ (1,0) [0|0] ""  Radar
 SG_ Msg2D9Always0x83 : 16|8@1+ (1,0) [0|0] ""  Radar

BO_ 1601 UDS_radarRequest: 8 Diag
 SG_ UDS_radarRequestData : 7|64@0+ (1,0) [0|0] "" Radar

BO_ 1617 Radar_udsResponse: 8 Radar
 SG_ Radar_udsResponseData : 7|64@0+ (1,0) [0|0] "" Diag

CM_ BO_ 697 "Start with MuxID 0x12, then 0x11 and finally 0x10 (VIN is then transmitted in the reverse order)";
CM_ BO_ 681 "Message sent every 1000 ms. All fixed bytes, no checksum, the byte for RWD or AWD needs to match VIN config";
CM_ BO_ 409 "Message sent every 10ms. Checksum : use all first 7 bytes with the SAE J1850 CRC algo";
CM_ BO_ 361 "Message sent every 10ms. Checksum : Sum of all first 7 bytes + 0x76";
CM_ BO_ 345 "Message sent every 20ms. Checksum : Sum of all first bytes + 0xc; place checksum in 4th octet";
CM_ BO_ 329 "Message sent every 20ms. Checksum : Sum of all first 7 bytes + 0x46";
CM_ BO_ 297 "Message sent every 20ms. Checksum : Sum of all first 5 bytes + 0x16";
CM_ BO_ 281 "Message sent every 10ms. Checksum : Sum of all first 5 bytes + 0x17";
CM_ BO_ 265 "Message sent every 10ms. Checksum : Sum of all first 7 bytes + 0x7";
CM_ BO_ 521 "Message sent every 100ms. All fixed bytes, no checksum.";
CM_ BO_ 537 "Message sent every 100ms. Checksum : use all first 7 bytes with the SAE J1850 CRC algo";
CM_ BO_ 425 "Message sent every 20ms. Checksum : Sum of all first 4 bytes + 0x38";
CM_ BO_ 729 "Message sent every 1000ms. All fixed bytes, no checksum.";

BA_DEF_ "BusType" STRING ;
BA_DEF_ BO_ "GenMsgCycleTime" INT 0 0;
BA_DEF_ SG_ "FieldType" STRING ;

BA_DEF_DEF_ "BusType" "CAN";
BA_DEF_DEF_ "FieldType" "";
BA_DEF_DEF_ "GenMsgCycleTime" 0;

BA_ "GenMsgCycleTime" BO_ 697 250;
BA_ "GenMsgCycleTime" BO_ 681 1000;
BA_ "GenMsgCycleTime" BO_ 409 10;
BA_ "GenMsgCycleTime" BO_ 361 10;
BA_ "GenMsgCycleTime" BO_ 345 20;
BA_ "GenMsgCycleTime" BO_ 329 20;
BA_ "GenMsgCycleTime" BO_ 297 20;
BA_ "GenMsgCycleTime" BO_ 281 10;
BA_ "GenMsgCycleTime" BO_ 265 10;
BA_ "GenMsgCycleTime" BO_ 521 100;
BA_ "GenMsgCycleTime" BO_ 537 100;
BA_ "GenMsgCycleTime" BO_ 425 20;
BA_ "GenMsgCycleTime" BO_ 729 1000;

VAL_ 681 Msg2A9_FourWheelDrive 3 "SNA" 2 "UNUSED" 1 "4WD" 0 "2WD" ;""")

    for base_id in list(M_RANGE) + list(L_RANGE):
      f.write(get_val_definition(base_id))
