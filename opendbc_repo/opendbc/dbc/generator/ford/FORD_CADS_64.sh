#!/bin/bash

OUT_FILENAME="../../FORD_CADS_64.dbc"


build_bo(){
id=$1
# bo=$(($id + 287))
bo=$(expr $id + 287)

len=64
if [ "$id" = "22" ]; then
   len=24
fi

cat <<EOF >> ${OUT_FILENAME}
BO_ ${bo} MRR_Detection_0${id}: ${len} MRR
EOF

build_bo_segment $id "01"
build_bo_segment $id "02"
build_bo_segment $id "03"
if [ "$id" != "22" ]; then
   build_bo_segment $id "04"
   build_bo_segment $id "05"
   build_bo_segment $id "06"
fi
echo "" >> ${OUT_FILENAME}

}

build_bo_segment(){
   id=$1
   seg=$2
   base=$((($2 - 1)*72))
cat <<EOF >> ${OUT_FILENAME}
   SG_ CAN_DET_CONFID_AZIMUTH_${id}_${seg} : $(expr $base + 33)|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_${id}_${seg} : $(expr $base + 56)|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_${id}_${seg} : $(expr $base + 48)|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_${id}_${seg} : $(expr $base + 49)|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_${id}_${seg} : $(expr $base + 0)|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_${id}_${seg} : $(expr $base + 47)|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_${id}_${seg} : $(expr $base + 31)|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_${id}_${seg} : $(expr $base + 15)|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_${id}_${seg} : $(expr $base + 7)|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_${id}_${seg} : $(expr $base + 17)|2@0+ (1,0) [0|3] "" IFV_VFP
EOF
}


build_ba(){
id=$1
# bo=$(($id + 287))
ba=$(expr $id + 287)

len=64
if [ "$id" = "22" ]; then
   len=24
fi

cat <<EOF >> ${OUT_FILENAME}
BA_ "GenMsgSendType" BO_ ${ba} 1;
BA_ "GenMsgILSupport" BO_ ${ba} 1;
BA_ "GenMsgNrOfRepetition" BO_ ${ba} 0;
BA_ "GenMsgCycleTime" BO_ ${ba} 0;
BA_ "NetworkInitialization" BO_ ${ba} 0;
BA_ "GenMsgDelayTime" BO_ ${ba} 0;
EOF

build_ba_segment $ba $id "01"
build_ba_segment $ba $id "02"
build_ba_segment $ba $id "03"
if [ "$id" != "22" ]; then
   build_ba_segment $ba $id "04"
   build_ba_segment $ba $id "05"
   build_ba_segment $ba $id "06"
fi

}

build_ba_segment(){
   ba=$1
   id=$2
   seg=$3
   
cat <<EOF >> ${OUT_FILENAME}
BA_ "GenSigVtEn" SG_ ${ba} CAN_DET_CONFID_AZIMUTH_${id}_${seg} "CAN_DET_CONFID_AZIMUTH_${id}_${seg}";
BA_ "GenSigVtName" SG_ ${ba} CAN_DET_CONFID_AZIMUTH_${id}_${seg} "CAN_DET_CONFID_AZIMUTH_${id}_${seg}";
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_CONFID_AZIMUTH_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_CONFID_AZIMUTH_${id}_${seg} "CAN_DET_CONFID_AZIMUTH_${id}_${seg}";
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_SUPER_RES_TARGET_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_SUPER_RES_TARGET_${id}_${seg} "CAN_DET_SUPER_RES_TARGET_${id}_${seg}";
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_ND_TARGET_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_ND_TARGET_${id}_${seg} "CAN_DET_ND_TARGET_${id}_${seg}";
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_HOST_VEH_CLUTTER_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_HOST_VEH_CLUTTER_${id}_${seg} "CAN_DET_HOST_VEH_CLUTTER_${id}_${seg}";
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_VALID_LEVEL_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_VALID_LEVEL_${id}_${seg} "CAN_DET_VALID_LEVEL_${id}_${seg}";
BA_ "GenSigStartValue" SG_ ${ba} CAN_DET_AZIMUTH_${id}_${seg} 0;
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_AZIMUTH_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_AZIMUTH_${id}_${seg} "CAN_DET_AZIMUTH_${id}_${seg}";
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_RANGE_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_RANGE_${id}_${seg} "CAN_DET_RANGE_${id}_${seg}";
BA_ "GenSigStartValue" SG_ ${ba} CAN_DET_RANGE_RATE_${id}_${seg} 0;
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_RANGE_RATE_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_RANGE_RATE_${id}_${seg} "CAN_DET_RANGE_RATE_${id}_${seg}";
BA_ "GenSigSendType" SG_ ${ba} CAN_DET_AMPLITUDE_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_DET_AMPLITUDE_${id}_${seg} "CAN_DET_AMPLITUDE_${id}_${seg}";
BA_ "GenSigSendType" SG_ ${ba} CAN_SCAN_INDEX_2LSB_${id}_${seg} 0;
BA_ "GenSigCmt" SG_ ${ba} CAN_SCAN_INDEX_2LSB_${id}_${seg} "CAN_SCAN_INDEX_2LSB_${id}_${seg}";
EOF
}

build_val(){
id=$1
# bo=$(($id + 287))
val=$(expr $id + 287)

cat <<EOF >> ${OUT_FILENAME}
VAL_ ${val} CAN_DET_CONFID_AZIMUTH_${id}_01 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ ${val} CAN_DET_CONFID_AZIMUTH_${id}_02 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ ${val} CAN_DET_CONFID_AZIMUTH_${id}_03 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
EOF
if [ "$id" != "22" ]; then
cat <<EOF >> ${OUT_FILENAME}
VAL_ ${val} CAN_DET_CONFID_AZIMUTH_${id}_04 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ ${val} CAN_DET_CONFID_AZIMUTH_${id}_05 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ ${val} CAN_DET_CONFID_AZIMUTH_${id}_06 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
EOF
fi

}


cat <<EOF > ${OUT_FILENAME}
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
BU_: MRR 
BO_ 1073741824 VECTOR__INDEPENDENT_SIG_MSG: 0 Vector__XXX
   SG_ New_Signal_943 : 0|8@1+ (1,0) [0|0] "" Vector__XXX
   SG_ CAN_SENSOR_VANGLE_OFFSET : 0|8@0+ (0.0625,-8) [-8|7.9375] "deg" Vector__XXX
   SG_ CAN_SENSOR_FOV_VER : 0|8@0+ (1,0) [0|255] "deg" Vector__XXX
   SG_ CAN_AUTO_ALIGN_VANGLE_QF : 0|1@0+ (1,0) [0|1] "" Vector__XXX
   SG_ CAN_AUTO_ALIGN_VANGLE_REF : 0|10@0+ (0.03125,-10) [-10|21.9688] "deg" Vector__XXX
   SG_ CAN_AUTO_ALIGN_VANGLE : 0|10@0+ (0.03125,-10) [-10|21.9688] "deg" Vector__XXX
   SG_ CAN_MMIC_Temp4 : 0|8@0+ (1,-50) [-50|205] "C" Vector__XXX
   SG_ CAN_MMIC_Temp3 : 0|8@0+ (1,-50) [-50|205] "C" Vector__XXX
   SG_ CAN_MMIC_Temp2 : 0|8@0+ (1,-50) [-50|205] "C" Vector__XXX
   SG_ CAN_Processor_Temp2 : 0|8@0+ (1,-50) [-50|205] "C" Vector__XXX
   SG_ CAN_CHECKSUM : 0|8@0+ (1,0) [0|255] "" Vector__XXX
   SG_ CAN_COUNTER : 0|4@0+ (1,0) [0|15] "" Vector__XXX
   SG_ CAN_VEHICLE_MODE : 0|4@0+ (1,0) [0|15] "" Vector__XXX
   SG_ CAN_USC_CAL_VER_MAJOR : 0|16@0+ (1,0) [0|65535] "" Vector__XXX
   SG_ CAN_USC_CAL_VER_MINOR : 0|16@0+ (1,0) [0|65535] "" Vector__XXX
   SG_ CAN_SMC_CAL_VER_MAJOR : 0|16@0+ (1,0) [0|65535] "" Vector__XXX
   SG_ CAN_HW_VERSION : 0|32@0+ (1,0) [0|4.29497e+09] "" Vector__XXX
   SG_ CAN_FAC_TGT_MTG_SPACE_VER : 0|8@0+ (1,-128) [-128|127] "cm" Vector__XXX
   SG_ CAN_ANGLE_MISALIGNMENT_VER : 0|10@0+ (0.03125,-10) [-10|21.9688] "deg" Vector__XXX
   SG_ CAN_ANGLE_MOUNTING_VOFFSET : 0|8@0+ (0.0625,-8) [-8|7.9375] "deg" Vector__XXX
   SG_ CAN_LATCH_FAULTS : 0|64@0+ (1,0) [0|100] "" Vector__XXX
   SG_ CAN_ACTIVE_FAULTS : 0|64@0+ (1,0) [0|1.84467e+19] "" Vector__XXX
   SG_ CAN_HISTORY_FAULTS : 0|64@0+ (1,0) [0|1.84467e+19] "" Vector__XXX
   SG_ CAN_SERV_ALIGN_ENABLE : 0|1@0+ (1,0) [0|1] "" Vector__XXX
   SG_ CAN_LONG_MOUNTING_OFFSET : 0|8@0+ (0.015625,-2) [-2|1.98438] "" Vector__XXX
   SG_ CAN_BEAMWIDTH_VERT : 0|7@0+ (0.125,0) [0|15.875] "deg" Vector__XXX
   SG_ CAN_VEHICLE_SPEED_CALC_QF : 0|2@0+ (1,0) [0|3] "" Vector__XXX

BO_ 34 Active_Fault_Latched_2: 8 MRR
   SG_ IPMA_PCAN_DataRangeCheck : 4|1@1+ (1,0) [0|1] "" External_Tool
   SG_ IPMA_PCAN_MissingMsg : 3|1@1+ (1,0) [0|1] "" External_Tool
   SG_ VINSignalCompareFailure : 2|1@1+ (1,0) [0|1] "" External_Tool
   SG_ ModuleNotConfiguredError : 1|1@1+ (1,0) [0|1] "" External_Tool
   SG_ CarCfgNotConfiguredError : 0|1@1+ (1,0) [0|1] "" External_Tool

BO_ 33 Active_Fault_Latched_1: 8 MRR
   SG_ Active_Flt_Latched_byte7_bit7 : 63|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte7_bit6 : 62|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte7_bit5 : 61|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte7_bit4 : 60|1@1+ (1,0) [0|1] "" External_Tool
   SG_ ARMtoDSPChksumFault : 59|1@1+ (1,0) [0|1] "" External_Tool
   SG_ DSPtoArmChksumFault : 58|1@1+ (1,0) [0|1] "" External_Tool
   SG_ HostToArmChksumFault : 57|1@1+ (1,0) [0|1] "" External_Tool
   SG_ ARMtoHostChksumFault : 56|1@1+ (1,0) [0|1] "" External_Tool
   SG_ LoopBWOutOfRange : 55|1@1+ (1,0) [0|1] "" External_Tool
   SG_ DSPOverrunFault : 54|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte6_bit5 : 53|1@1+ (1,0) [0|1] "" External_Tool
   SG_ TuningSensitivityFault : 52|1@1+ (1,0) [0|1] "" External_Tool
   SG_ SaturatedTuningFreqFault : 51|1@1+ (1,0) [0|1] "" External_Tool
   SG_ LocalOscPowerFault : 50|1@1+ (1,0) [0|1] "" External_Tool
   SG_ TransmitterPowerFault : 49|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte6_bit0 : 48|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte5_bit7 : 47|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte5_bit6 : 46|1@1+ (1,0) [0|1] "" External_Tool
   SG_ XCVRDeviceSPIFault : 45|1@1+ (1,0) [0|1] "" External_Tool
   SG_ FreqSynthesizerSPIFault : 44|1@1+ (1,0) [0|1] "" External_Tool
   SG_ AnalogConverterDevicSPIFault : 43|1@1+ (1,0) [0|1] "" External_Tool
   SG_ SidelobeBlockage : 42|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte5_bit1 : 41|1@1+ (1,0) [0|1] "" External_Tool
   SG_ MNRBlocked : 40|1@1+ (1,0) [0|1] "" External_Tool
   SG_ ECUTempHighFault : 39|1@1+ (1,0) [0|1] "" External_Tool
   SG_ TransmitterTempHighFault : 38|1@1+ (1,0) [0|1] "" External_Tool
   SG_ AlignmentRoutineFailedFault : 37|1@1+ (1,0) [0|1] "" External_Tool
   SG_ UnreasonableRadarData : 36|1@1+ (1,0) [0|1] "" External_Tool
   SG_ MicroprocessorTempHighFault : 35|1@1+ (1,0) [0|1] "" External_Tool
   SG_ VerticalAlignmentOutOfRange : 34|1@1+ (1,0) [0|1] "" External_Tool
   SG_ HorizontalAlignmentOutOfRange : 33|1@1+ (1,0) [0|1] "" External_Tool
   SG_ FactoryAlignmentMode : 32|1@1+ (1,0) [0|1] "" External_Tool
   SG_ BatteryLowFault : 31|1@1+ (1,0) [0|1] "" External_Tool
   SG_ BatteryHighFault : 30|1@1+ (1,0) [0|1] "" External_Tool
   SG_ v_1p25SupplyOutOfRange : 29|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte3_bit4 : 28|1@1+ (1,0) [0|1] "" External_Tool
   SG_ ThermistorOutOfRange : 27|1@1+ (1,0) [0|1] "" External_Tool
   SG_ v_3p3DACSupplyOutOfRange : 26|1@1+ (1,0) [0|1] "" External_Tool
   SG_ v_3p3RAWSupplyOutOfRange : 25|1@1+ (1,0) [0|1] "" External_Tool
   SG_ v_5_SupplyOutOfRange : 24|1@1+ (1,0) [0|1] "" External_Tool
   SG_ TransmitterIDFault : 23|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte2_bit6 : 22|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte2_bit5 : 21|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte2_bit4 : 20|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte2_bit3 : 19|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte2_bit2 : 18|1@1+ (1,0) [0|1] "" External_Tool
   SG_ PCANMissingMsgFault : 17|1@1+ (1,0) [0|1] "" External_Tool
   SG_ PCANBusOff : 16|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte1_bit7 : 15|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte1_bit6 : 14|1@1+ (1,0) [0|1] "" External_Tool
   SG_ InstructionSetCheckFault : 13|1@1+ (1,0) [0|1] "" External_Tool
   SG_ StackOverflowFault : 12|1@1+ (1,0) [0|1] "" External_Tool
   SG_ WatchdogFault : 11|1@1+ (1,0) [0|1] "" External_Tool
   SG_ PLLLockFault : 10|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte1_bit1 : 9|1@1+ (1,0) [0|1] "" External_Tool
   SG_ RAMMemoryTestFault : 8|1@1+ (1,0) [0|1] "" External_Tool
   SG_ USCValidationFault : 7|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte0_bit6 : 6|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte0_bit5 : 5|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte0_bit4 : 4|1@1+ (1,0) [0|1] "" External_Tool
   SG_ Active_Flt_Latched_byte0_bit3 : 3|1@1+ (1,0) [0|1] "" External_Tool
   SG_ KeepAliveChecksumFault : 2|1@1+ (1,0) [0|1] "" External_Tool
   SG_ ProgramCalibrationFlashChecksum : 1|1@1+ (1,0) [0|1] "" External_Tool
   SG_ ApplicationFlashChecksumFault : 0|1@1+ (1,0) [0|1] "" External_Tool

BO_ 500 XCP_MRR_DAQ_RESP: 8 MRR
   SG_ MRR_xcp_daq_resp_byte7 : 63|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_daq_resp_byte6 : 55|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_daq_resp_byte5 : 47|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_daq_resp_byte4 : 39|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_daq_resp_byte3 : 31|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_daq_resp_byte2 : 23|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_daq_resp_byte1 : 15|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_daq_resp_byte0 : 7|8@0+ (1,0) [0|255] "" External_Tool

BO_ 499 XCP_MRR_DTO_RESP: 8 MRR
   SG_ MRR_xcp_dto_resp_byte7 : 63|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_dto_resp_byte6 : 55|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_dto_resp_byte5 : 47|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_dto_resp_byte4 : 39|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_dto_resp_byte3 : 31|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_dto_resp_byte2 : 23|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_dto_resp_byte1 : 15|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_dto_resp_byte0 : 7|8@0+ (1,0) [0|255] "" External_Tool

BO_ 497 XCP_MRR_CTO_RESP: 8 MRR
   SG_ MRR_xcp_cto_resp_byte7 : 63|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_cto_resp_byte6 : 55|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_cto_resp_byte5 : 47|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_cto_resp_byte4 : 39|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_cto_resp_byte3 : 31|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_cto_resp_byte2 : 23|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_cto_resp_byte1 : 15|8@0+ (1,0) [0|255] "" External_Tool
   SG_ MRR_xcp_cto_resp_byte0 : 7|8@0+ (1,0) [0|255] "" External_Tool

BO_ 1900 Ford_Diag_Resp_Phys: 8 MRR
   SG_ TesterPhysicalResCCM : 7|64@0+ (1,0) [0|1.84467e+19] "" IFV_Host

BO_ 261 MRR_Status_SerialNumber: 8 MRR
   SG_ CAN_SEQUENCE_NUMBER : 55|16@0+ (1,0) [0|65535] "" External_Tool
   SG_ CAN_SERIAL_NUMBER : 7|40@0+ (1,0) [0|1.09951e+12] "" External_Tool

BO_ 264 MRR_Status_SwVersion: 8 MRR
   SG_ CAN_PBL_Field_Revision : 47|8@0+ (1,0) [0|255] "" External_Tool
   SG_ CAN_PBL_Promote_Revision : 39|8@0+ (1,0) [0|255] "" External_Tool
   SG_ CAN_SW_Field_Revision : 23|8@0+ (1,0) [0|255] "" External_Tool
   SG_ CAN_SW_Promote_Revision : 15|8@0+ (1,0) [0|255] "" External_Tool
   SG_ CAN_SW_Release_Revision : 7|8@0+ (1,0) [0|255] "" External_Tool
   SG_ CAN_PBL_Release_Revision : 31|8@0+ (1,0) [0|255] "" External_Tool

BO_ 373 MRR_Header_SensorPosition: 8 MRR
   SG_ CAN_SENSOR_POLARITY : 55|1@0+ (1,0) [0|1] "" External_Tool
   SG_ CAN_SENSOR_LAT_OFFSET : 39|16@0+ (0.01,0) [0|655.35] "cm" External_Tool
   SG_ CAN_SENSOR_LONG_OFFSET : 23|16@0+ (0.01,0) [0|655.35] "cm" External_Tool
   SG_ CAN_SENSOR_HANGLE_OFFSET : 7|8@0+ (0.0625,-8) [-8|7.9375] "deg" External_Tool

BO_ 372 MRR_Header_SensorCoverage: 8 MRR
   SG_ CAN_SENSOR_FOV_HOR : 39|8@0+ (1,0) [0|255] "deg" IFV_VFP
   SG_ CAN_DOPPLER_COVERAGE : 23|8@0+ (1,-128) [-128|127] "m/s" IFV_VFP
   SG_ CAN_RANGE_COVERAGE : 7|8@0+ (1,0) [0|255] "m" IFV_VFP

BO_ 371 MRR_Header_AlignmentState: 8 MRR
   SG_ CAN_AUTO_ALIGN_HANGLE_QF : 13|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_ALIGNMENT_STATUS : 51|4@0+ (1,0) [0|11] "" IFV_VFP
   SG_ CAN_ALIGNMENT_STATE : 55|3@0+ (1,0) [0|7] "" IFV_VFP
   SG_ CAN_AUTO_ALIGN_HANGLE_REF : 11|10@0+ (0.000341218,-0.174533) [-0.174533|0.174533] "rad" IFV_VFP
   SG_ CAN_AUTO_ALIGN_HANGLE : 7|10@0+ (0.000341218,-0.174533) [-0.174533|0.174533] "rad" IFV_VFP

BO_ 369 MRR_Header_Timestamps: 8 MRR
   SG_ CAN_DET_TIME_SINCE_MEAS : 39|11@0+ (0.1,0) [0|204.7] "ms" IFV_Host
   SG_ CAN_SENSOR_TIME_STAMP : 7|32@0+ (0.1,0) [0|4.29497e+08] "ms" IFV_VFP

BO_ 368 MRR_Header_InformationDetections: 8 MRR
   SG_ CAN_ALIGN_UPDATES_DONE : 55|16@0+ (1,0) [0|65535] "" IFV_VFP
   SG_ CAN_SCAN_INDEX : 31|16@0+ (1,0) [0|65535] "" IFV_VFP
   SG_ CAN_NUMBER_OF_DET : 47|8@0+ (1,0) [0|255] "" External_Tool
   SG_ CAN_LOOK_ID : 23|2@0+ (1,0) [0|3] "" External_Tool
   SG_ CAN_LOOK_INDEX : 7|16@0+ (1,0) [0|65535] "" External_Tool

BO_ 265 MRR_Status_Temp_Volt: 8 MRR
   SG_ CAN_BATT_VOLTS : 63|8@0+ (0.08,0) [0|20.4] "V" External_Tool
   SG_ CAN_1_25_V : 55|8@0+ (0.08,0) [0|20.4] "V" External_Tool
   SG_ CAN_5_V : 47|8@0+ (0.08,0) [0|20.4] "V" External_Tool
   SG_ CAN_3_3_V_RAW : 31|8@0+ (0.08,0) [0|20.4] "V" External_Tool
   SG_ CAN_3_3_V_DAC : 15|8@0+ (0.08,0) [0|20.4] "V" External_Tool
   SG_ CAN_MMIC_Temp1 : 39|8@0+ (1,-50) [-50|205] "C" External_Tool
   SG_ CAN_Processor_Thermistor : 23|8@0+ (1,-50) [-50|205] "C" External_Tool
   SG_ CAN_Processor_Temp1 : 7|8@0+ (1,-50) [-50|205] "C" External_Tool

EOF

build_bo "04"

cat <<EOF >> ${OUT_FILENAME}
BO_ 351 MRR_Detection_064: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_64 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_64 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_64 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_64 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_64 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_64 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_64 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_64 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_64 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_64 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 350 MRR_Detection_063: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_63 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_63 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_63 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_63 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_63 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_63 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_63 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_63 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_63 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_63 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 349 MRR_Detection_062: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_62 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_62 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_62 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_62 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_62 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_62 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_62 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_62 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_62 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_62 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 348 MRR_Detection_061: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_61 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_61 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_61 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_61 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_61 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_61 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_61 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_61 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_61 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_61 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 347 MRR_Detection_060: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_60 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_60 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_60 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_60 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_60 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_60 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_60 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_60 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_60 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_60 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 346 MRR_Detection_059: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_59 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_59 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_59 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_59 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_59 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_59 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_59 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_59 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_59 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_59 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 345 MRR_Detection_058: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_58 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_58 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_58 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_58 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_58 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_58 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_58 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_58 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_58 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_58 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 344 MRR_Detection_057: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_57 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_57 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_57 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_57 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_57 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_57 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_57 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_57 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_57 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_57 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 343 MRR_Detection_056: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_56 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_56 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_56 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_56 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_56 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_56 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_56 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_56 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_56 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_56 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 342 MRR_Detection_055: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_55 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_55 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_55 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_55 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_55 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_55 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_55 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_55 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_55 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_55 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 335 MRR_Detection_048: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_48 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_48 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_48 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_48 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_48 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_48 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_48 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_48 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_48 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_48 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 334 MRR_Detection_047: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_47 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_47 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_47 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_47 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_47 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_47 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_47 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_47 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_47 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_47 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 333 MRR_Detection_046: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_46 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_46 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_46 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_46 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_46 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_46 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_46 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_46 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_46 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_46 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 332 MRR_Detection_045: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_45 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_45 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_45 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_45 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_45 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_45 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_45 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_45 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_45 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_45 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 331 MRR_Detection_044: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_44 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_44 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_44 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_44 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_44 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_44 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_44 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_44 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_44 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_44 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 330 MRR_Detection_043: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_43 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_43 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_43 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_43 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_43 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_43 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_43 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_43 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_43 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_43 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 329 MRR_Detection_042: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_42 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_42 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_42 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_42 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_42 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_42 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_42 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_42 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_42 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_42 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 328 MRR_Detection_041: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_41 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_41 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_41 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_41 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_41 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_41 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_41 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_41 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_41 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_41 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 327 MRR_Detection_040: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_40 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_40 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_40 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_40 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_40 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_40 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_40 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_40 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_40 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_40 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 325 MRR_Detection_038: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_38 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_38 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_38 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_38 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_38 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_38 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_38 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_38 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_38 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_38 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 324 MRR_Detection_037: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_37 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_37 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_37 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_37 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_37 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_37 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_37 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_37 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_37 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_37 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 323 MRR_Detection_036: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_36 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_36 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_36 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_36 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_36 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_36 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_36 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_36 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_36 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_36 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 322 MRR_Detection_035: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_35 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_35 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_35 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_35 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_35 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_35 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_35 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_35 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_35 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_35 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 321 MRR_Detection_034: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_34 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_34 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_34 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_34 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_34 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_34 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_34 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_34 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_34 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_34 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 320 MRR_Detection_033: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_33 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_33 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_33 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_33 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_33 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_33 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_33 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_33 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_33 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_33 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 319 MRR_Detection_032: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_32 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_32 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_32 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_32 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_32 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_32 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_32 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_32 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_32 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_32 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 318 MRR_Detection_031: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_31 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_31 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_31 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_31 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_31 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_31 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_31 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_31 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_31 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_31 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 317 MRR_Detection_030: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_30 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_30 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_30 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_30 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_30 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_30 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_30 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_30 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_30 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_30 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 316 MRR_Detection_029: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_29 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_29 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_29 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_29 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_29 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_29 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_29 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_29 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_29 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_29 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 314 MRR_Detection_027: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_27 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_27 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_27 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_27 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_27 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_27 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_27 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_27 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_27 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_27 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 313 MRR_Detection_026: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_26 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_26 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_26 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_26 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_26 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_26 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_26 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_26 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_26 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_26 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 312 MRR_Detection_025: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_25 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_25 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_25 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_25 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_25 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_25 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_25 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_25 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_25 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_25 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 311 MRR_Detection_024: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_24 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_24 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_24 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_24 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_24 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_24 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_24 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_24 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_24 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_24 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 310 MRR_Detection_023: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_23 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_23 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_23 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_23 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_23 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_23 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_23 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_23 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_23 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_23 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

EOF

build_bo "22"
build_bo "21"
build_bo "20"
build_bo "19"
build_bo "18"

cat <<EOF >> ${OUT_FILENAME}
BO_ 341 MRR_Detection_054: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_54 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_54 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_54 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_54 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_54 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_54 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_54 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_54 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_54 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_54 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 340 MRR_Detection_053: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_53 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_53 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_53 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_53 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_53 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_53 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_53 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_53 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_53 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_53 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 339 MRR_Detection_052: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_52 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_52 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_52 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_52 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_52 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_52 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_52 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_52 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_52 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_52 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 338 MRR_Detection_051: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_51 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_51 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_51 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_51 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_51 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_51 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_51 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_51 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_51 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_51 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 337 MRR_Detection_050: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_50 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_50 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_50 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_50 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_50 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_50 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_50 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_50 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_50 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_50 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 336 MRR_Detection_049: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_49 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_49 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_49 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_49 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_49 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_49 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_49 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_49 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_49 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_49 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 326 MRR_Detection_039: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_39 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_39 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_39 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_39 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_39 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_39 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_39 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_39 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_39 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_39 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

BO_ 315 MRR_Detection_028: 8 MRR
   SG_ CAN_DET_CONFID_AZIMUTH_28 : 33|2@0+ (1,0) [0|3] "" IFV_VFP
   SG_ CAN_DET_SUPER_RES_TARGET_28 : 56|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_ND_TARGET_28 : 48|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_HOST_VEH_CLUTTER_28 : 49|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_VALID_LEVEL_28 : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_DET_AZIMUTH_28 : 47|14@0+ (0.0003834,-3.1416) [-3.1416|3.13964] "rad" IFV_VFP
   SG_ CAN_DET_RANGE_28 : 31|14@0+ (0.015625,0) [0|255.984] "m" IFV_VFP
   SG_ CAN_DET_RANGE_RATE_28 : 15|14@0+ (0.015625,-128) [-128|127.984] "m/s" IFV_VFP
   SG_ CAN_DET_AMPLITUDE_28 : 7|7@0+ (1,-64) [-64|63] "dBsm" IFV_VFP
   SG_ CAN_SCAN_INDEX_2LSB_28 : 17|2@0+ (1,0) [0|3] "" IFV_VFP

EOF

build_bo "17"
build_bo "16"
build_bo "15"
build_bo "14"
build_bo "13"
build_bo "12"
build_bo "11"
build_bo "10"
build_bo "09"
build_bo "08"
build_bo "07"
build_bo "06"
build_bo "05"
build_bo "03"
build_bo "02"

cat <<EOF >> ${OUT_FILENAME}
BO_ 256 MRR_Status_CANVersion: 8 MRR
   SG_ CAN_USC_SECTION_COMPATIBILITY : 23|16@0+ (1,0) [0|65535] "" External_Tool
   SG_ CAN_PCAN_MINOR_MRR : 7|8@0+ (1,0) [0|255] "" IFV_VFP
   SG_ CAN_PCAN_MAJOR_MRR : 15|8@0+ (1,0) [0|255] "" IFV_VFP

BO_ 257 MRR_Status_Radar: 8 MRR
   SG_ CAN_INTERFERENCE_TYPE : 11|2@0+ (1,0) [0|3] "" IFV_Host
   SG_ CAN_RECOMMEND_UNCONVERGE : 9|1@0+ (1,0) [0|1] "" IFV_Host
   SG_ CAN_BLOCKAGE_SIDELOBE_FILTER_VAL : 15|4@0+ (1,0) [0|15] "" IFV_Host
   SG_ CAN_RADAR_ALIGN_INCOMPLETE : 8|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_BLOCKAGE_SIDELOBE : 4|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_BLOCKAGE_MNR : 5|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_RADAR_EXT_COND_NOK : 1|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_RADAR_ALIGN_OUT_RANGE : 2|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_RADAR_ALIGN_NOT_START : 0|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_RADAR_OVERHEAT_ERROR : 3|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_RADAR_NOT_OP : 6|1@0+ (1,0) [0|1] "" IFV_VFP
   SG_ CAN_XCVR_OPERATIONAL : 7|1@0+ (1,0) [0|1] "" IFV_VFP

EOF

build_bo "01"

cat <<EOF >> ${OUT_FILENAME}
BA_DEF_ SG_ "CrossOver_InfoCAN" ENUM "No","Yes";
BA_DEF_ SG_ "CrossOver_LIN" ENUM "No","Yes","No","Yes";
BA_DEF_ SG_ "UsedOnPgmDBC" ENUM "No","Yes","No","Yes","No","Yes";
BA_DEF_ SG_ "ContentDependant" ENUM "No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ SG_ "GenSigTimeoutTime_RCM" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_GWM" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_OCS" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_ABS_ESC" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_CCM" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_IPMA" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_TSTR" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_SCCM" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_PSCM" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime__delete" INT 0 100000;
BA_DEF_ SG_ "GenSigTimeoutTime_Generic_BCM" INT 0 100000;
BA_DEF_ BO_ "NmMessage" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ BO_ "DiagResponse" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ BO_ "DiagRequest" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ BO_ "TpTxIndex" INT 0 255;
BA_DEF_ BO_ "DiagState" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ BO_ "TpApplType" STRING ;
BA_DEF_ BO_ "NmAsrMessage" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ BO_ "Mulitplexer" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ BO_ "ConfiguredTransmitter" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ BO_ "EventRateOfChange" INT 10 10000;
BA_DEF_ BO_ "GenMsgHandlingTypeDoc" STRING ;
BA_DEF_ BO_ "GenMsgHandlingTypeCode" STRING ;
BA_DEF_ BO_ "GenMsgMarked" STRING ;
BA_DEF_ SG_ "GenSigMarked" STRING ;
BA_DEF_ SG_ "GenSigVtIndex" STRING ;
BA_DEF_ SG_ "GenSigVtName" STRING ;
BA_DEF_ SG_ "GenSigVtEn" STRING ;
BA_DEF_ SG_ "GenSigSNA" STRING ;
BA_DEF_ SG_ "GenSigCmt" STRING ;
BA_DEF_ BO_ "GenMsgCmt" STRING ;
BA_DEF_ SG_ "GenSigSendType" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType";
BA_DEF_ SG_ "GenSigInactiveValue" INT 0 100000;
BA_DEF_ SG_ "GenSigMissingSourceValue" INT 0 1e+09;
BA_DEF_ SG_ "WakeupSignal" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes";
BA_DEF_ SG_ "GenSigStartValue" INT 0 1e+09;
BA_DEF_ BO_ "GenMsgILSupport" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes","No","Yes";
BA_DEF_ BO_ "NetworkInitializationCommand" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes","No","Yes","No","Yes";
BA_DEF_ BO_ "GenMsgSendType" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes","No","Yes","No","Yes","FixedPeriodic","Event","EnabledPeriodic","NotUsed","NotUsed","EventPeriodic","NotUsed","NotUsed","NoMsgSendType";
BA_DEF_ BO_ "GenMsgCycleTime" INT 0 100000;
BA_DEF_ BO_ "GenMsgCycleTimeFast" INT 0 100000;
BA_DEF_ BO_ "GenMsgDelayTime" INT 0 1000;
BA_DEF_ BO_ "GenMsgNrOfRepetition" INT 0 100;
BA_DEF_ BO_ "GenMsgStartDelayTime" INT 0 10000;
BA_DEF_ BO_ "NetworkInitialization" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes","No","Yes","No","Yes","FixedPeriodic","Event","EnabledPeriodic","NotUsed","NotUsed","EventPeriodic","NotUsed","NotUsed","NoMsgSendType","No","Yes";
BA_DEF_ BO_ "MessageGateway" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes","No","Yes","No","Yes","FixedPeriodic","Event","EnabledPeriodic","NotUsed","NotUsed","EventPeriodic","NotUsed","NotUsed","NoMsgSendType","No","Yes","No","Yes";
BA_DEF_ BU_ "ILUsed" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes","No","Yes","No","Yes","FixedPeriodic","Event","EnabledPeriodic","NotUsed","NotUsed","EventPeriodic","NotUsed","NotUsed","NoMsgSendType","No","Yes","No","Yes","No","Yes";
BA_DEF_ BU_ "NetworkInitializationUsed" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes","No","Yes","No","Yes","FixedPeriodic","Event","EnabledPeriodic","NotUsed","NotUsed","EventPeriodic","NotUsed","NotUsed","NoMsgSendType","No","Yes","No","Yes","No","Yes","No","Yes";
BA_DEF_ BU_ "PowerType" ENUM "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Cyclic","OnWrite","vector_leerstring","OnChange","vector_leerstring","IfActive","vector_leerstring","NoSigSendType","No","Yes","No","Yes","No","Yes","FixedPeriodic","Event","EnabledPeriodic","NotUsed","NotUsed","EventPeriodic","NotUsed","NotUsed","NoMsgSendType","No","Yes","No","Yes","No","Yes","No","Yes","Switched","Latched","Sleep","vector_leerstring","vector_leerstring";
BA_DEF_ BU_ "NodeStartUpTime" INT 0 10000;
BA_DEF_ BU_ "NodeWakeUpTime" INT 0 10000;
BA_DEF_ BO_ "GenMsgBackgroundColor" STRING ;
BA_DEF_ BO_ "GenMsgForegroundColor" STRING ;
BA_ "GenMsgCycleTime" BO_ 34 1000;
BA_ "GenMsgSendType" BO_ 34 0;
BA_ "GenSigVtEn" SG_ 34 IPMA_PCAN_DataRangeCheck "IPMA_PCAN_DataRangeCheck";
BA_ "GenSigVtName" SG_ 34 IPMA_PCAN_DataRangeCheck "IPMA_PCAN_DataRangeCheck";
BA_ "GenSigVtEn" SG_ 34 IPMA_PCAN_MissingMsg "IPMA_PCAN_MissingMsg";
BA_ "GenSigVtName" SG_ 34 IPMA_PCAN_MissingMsg "IPMA_PCAN_MissingMsg";
BA_ "GenSigVtEn" SG_ 34 VINSignalCompareFailure "VINSignalCompareFailure";
BA_ "GenSigVtName" SG_ 34 VINSignalCompareFailure "VINSignalCompareFailure";
BA_ "GenSigVtEn" SG_ 34 ModuleNotConfiguredError "ModuleNotConfiguredError";
BA_ "GenSigVtName" SG_ 34 ModuleNotConfiguredError "ModuleNotConfiguredError";
BA_ "GenSigVtEn" SG_ 34 CarCfgNotConfiguredError "CarCfgNotConfiguredError";
BA_ "GenSigVtName" SG_ 34 CarCfgNotConfiguredError "CarCfgNotConfiguredError";
BA_ "GenMsgCycleTime" BO_ 33 1000;
BA_ "GenMsgSendType" BO_ 33 0;
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte7_bit7 "Active_Flt_Latched_byte7_bit7";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte7_bit7 "Active_Flt_Latched_byte7_bit7";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte7_bit6 "Active_Flt_Latched_byte7_bit6";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte7_bit6 "Active_Flt_Latched_byte7_bit6";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte7_bit5 "Active_Flt_Latched_byte7_bit5";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte7_bit5 "Active_Flt_Latched_byte7_bit5";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte7_bit4 "Active_Flt_Latched_byte7_bit4";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte7_bit4 "Active_Flt_Latched_byte7_bit4";
BA_ "GenSigVtEn" SG_ 33 ARMtoDSPChksumFault "ARMtoDSPChksumFault";
BA_ "GenSigVtName" SG_ 33 ARMtoDSPChksumFault "ARMtoDSPChksumFault";
BA_ "GenSigVtEn" SG_ 33 DSPtoArmChksumFault "DSPtoArmChksumFault";
BA_ "GenSigVtName" SG_ 33 DSPtoArmChksumFault "DSPtoArmChksumFault";
BA_ "GenSigVtEn" SG_ 33 HostToArmChksumFault "HostToArmChksumFault";
BA_ "GenSigVtName" SG_ 33 HostToArmChksumFault "HostToArmChksumFault";
BA_ "GenSigVtEn" SG_ 33 ARMtoHostChksumFault "ARMtoHostChksumFault";
BA_ "GenSigVtName" SG_ 33 ARMtoHostChksumFault "ARMtoHostChksumFault";
BA_ "GenSigVtEn" SG_ 33 LoopBWOutOfRange "LoopBWOutOfRange";
BA_ "GenSigVtName" SG_ 33 LoopBWOutOfRange "LoopBWOutOfRange";
BA_ "GenSigVtEn" SG_ 33 DSPOverrunFault "DSPOverrunFault";
BA_ "GenSigVtName" SG_ 33 DSPOverrunFault "DSPOverrunFault";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte6_bit5 "Active_Flt_Latched_byte6_bit5";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte6_bit5 "Active_Flt_Latched_byte6_bit5";
BA_ "GenSigVtEn" SG_ 33 TuningSensitivityFault "TuningSensitivityFault";
BA_ "GenSigVtName" SG_ 33 TuningSensitivityFault "TuningSensitivityFault";
BA_ "GenSigVtEn" SG_ 33 SaturatedTuningFreqFault "SaturatedTuningFreqFault";
BA_ "GenSigVtName" SG_ 33 SaturatedTuningFreqFault "SaturatedTuningFreqFault";
BA_ "GenSigVtEn" SG_ 33 LocalOscPowerFault "LocalOscPowerFault";
BA_ "GenSigVtName" SG_ 33 LocalOscPowerFault "LocalOscPowerFault";
BA_ "GenSigVtEn" SG_ 33 TransmitterPowerFault "TransmitterPowerFault";
BA_ "GenSigVtName" SG_ 33 TransmitterPowerFault "TransmitterPowerFault";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte6_bit0 "Active_Flt_Latched_byte6_bit0";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte6_bit0 "Active_Flt_Latched_byte6_bit0";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte5_bit7 "Active_Flt_Latched_byte5_bit7";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte5_bit7 "Active_Flt_Latched_byte5_bit7";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte5_bit6 "Active_Flt_Latched_byte5_bit6";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte5_bit6 "Active_Flt_Latched_byte5_bit6";
BA_ "GenSigVtEn" SG_ 33 XCVRDeviceSPIFault "XCVRDeviceSPIFault";
BA_ "GenSigVtName" SG_ 33 XCVRDeviceSPIFault "XCVRDeviceSPIFault";
BA_ "GenSigVtEn" SG_ 33 FreqSynthesizerSPIFault "FreqSynthesizerSPIFault";
BA_ "GenSigVtName" SG_ 33 FreqSynthesizerSPIFault "FreqSynthesizerSPIFault";
BA_ "GenSigVtEn" SG_ 33 AnalogConverterDevicSPIFault "AnalogConverterDevicSPIFault";
BA_ "GenSigVtName" SG_ 33 AnalogConverterDevicSPIFault "AnalogConverterDevicSPIFault";
BA_ "GenSigVtEn" SG_ 33 SidelobeBlockage "SidelobeBlockage";
BA_ "GenSigVtName" SG_ 33 SidelobeBlockage "SidelobeBlockage";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte5_bit1 "Active_Flt_Latched_byte5_bit1";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte5_bit1 "Active_Flt_Latched_byte5_bit1";
BA_ "GenSigVtEn" SG_ 33 MNRBlocked "MNRBlocked";
BA_ "GenSigVtName" SG_ 33 MNRBlocked "MNRBlocked";
BA_ "GenSigVtEn" SG_ 33 ECUTempHighFault "ECUTempHighFault";
BA_ "GenSigVtName" SG_ 33 ECUTempHighFault "ECUTempHighFault";
BA_ "GenSigVtEn" SG_ 33 TransmitterTempHighFault "TransmitterTempHighFault";
BA_ "GenSigVtName" SG_ 33 TransmitterTempHighFault "TransmitterTempHighFault";
BA_ "GenSigVtEn" SG_ 33 AlignmentRoutineFailedFault "AlignmentRoutineFailedFault";
BA_ "GenSigVtName" SG_ 33 AlignmentRoutineFailedFault "AlignmentRoutineFailedFault";
BA_ "GenSigVtEn" SG_ 33 UnreasonableRadarData "UnreasonableRadarData";
BA_ "GenSigVtName" SG_ 33 UnreasonableRadarData "UnreasonableRadarData";
BA_ "GenSigVtEn" SG_ 33 MicroprocessorTempHighFault "MicroprocessorTempHighFault";
BA_ "GenSigVtName" SG_ 33 MicroprocessorTempHighFault "MicroprocessorTempHighFault";
BA_ "GenSigVtEn" SG_ 33 VerticalAlignmentOutOfRange "VerticalAlignmentOutOfRange";
BA_ "GenSigVtName" SG_ 33 VerticalAlignmentOutOfRange "VerticalAlignmentOutOfRange";
BA_ "GenSigVtEn" SG_ 33 HorizontalAlignmentOutOfRange "HorizontalAlignmentOutOfRange";
BA_ "GenSigVtName" SG_ 33 HorizontalAlignmentOutOfRange "HorizontalAlignmentOutOfRange";
BA_ "GenSigVtEn" SG_ 33 FactoryAlignmentMode "FactoryAlignmentMode";
BA_ "GenSigVtName" SG_ 33 FactoryAlignmentMode "FactoryAlignmentMode";
BA_ "GenSigVtEn" SG_ 33 BatteryLowFault "BatteryLowFault";
BA_ "GenSigVtName" SG_ 33 BatteryLowFault "BatteryLowFault";
BA_ "GenSigVtEn" SG_ 33 BatteryHighFault "BatteryHighFault";
BA_ "GenSigVtName" SG_ 33 BatteryHighFault "BatteryHighFault";
BA_ "GenSigVtEn" SG_ 33 v_1p25SupplyOutOfRange "v_1p25SupplyOutOfRange";
BA_ "GenSigVtName" SG_ 33 v_1p25SupplyOutOfRange "v_1p25SupplyOutOfRange";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte3_bit4 "Active_Flt_Latched_byte3_bit4";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte3_bit4 "Active_Flt_Latched_byte3_bit4";
BA_ "GenSigVtEn" SG_ 33 ThermistorOutOfRange "ThermistorOutOfRange";
BA_ "GenSigVtName" SG_ 33 ThermistorOutOfRange "ThermistorOutOfRange";
BA_ "GenSigVtEn" SG_ 33 v_3p3DACSupplyOutOfRange "v_3p3DACSupplyOutOfRange";
BA_ "GenSigVtName" SG_ 33 v_3p3DACSupplyOutOfRange "v_3p3DACSupplyOutOfRange";
BA_ "GenSigVtEn" SG_ 33 v_3p3RAWSupplyOutOfRange "v_3p3RAWSupplyOutOfRange";
BA_ "GenSigVtName" SG_ 33 v_3p3RAWSupplyOutOfRange "v_3p3RAWSupplyOutOfRange";
BA_ "GenSigVtEn" SG_ 33 v_5_SupplyOutOfRange "v_5_SupplyOutOfRange";
BA_ "GenSigVtName" SG_ 33 v_5_SupplyOutOfRange "v_5_SupplyOutOfRange";
BA_ "GenSigVtEn" SG_ 33 TransmitterIDFault "TransmitterIDFault";
BA_ "GenSigVtName" SG_ 33 TransmitterIDFault "TransmitterIDFault";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte2_bit6 "Active_Flt_Latched_byte2_bit6";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte2_bit6 "Active_Flt_Latched_byte2_bit6";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte2_bit5 "Active_Flt_Latched_byte2_bit5";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte2_bit5 "Active_Flt_Latched_byte2_bit5";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte2_bit4 "Active_Flt_Latched_byte2_bit4";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte2_bit4 "Active_Flt_Latched_byte2_bit4";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte2_bit3 "Active_Flt_Latched_byte2_bit3";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte2_bit3 "Active_Flt_Latched_byte2_bit3";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte2_bit2 "Active_Flt_Latched_byte2_bit2";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte2_bit2 "Active_Flt_Latched_byte2_bit2";
BA_ "GenSigVtEn" SG_ 33 PCANMissingMsgFault "PCANMissingMsgFault";
BA_ "GenSigVtName" SG_ 33 PCANMissingMsgFault "PCANMissingMsgFault";
BA_ "GenSigVtEn" SG_ 33 PCANBusOff "PCANBusOff";
BA_ "GenSigVtName" SG_ 33 PCANBusOff "PCANBusOff";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte1_bit7 "Active_Flt_Latched_byte1_bit7";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte1_bit7 "Active_Flt_Latched_byte1_bit7";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte1_bit6 "Active_Flt_Latched_byte1_bit6";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte1_bit6 "Active_Flt_Latched_byte1_bit6";
BA_ "GenSigVtEn" SG_ 33 InstructionSetCheckFault "InstructionSetCheckFault";
BA_ "GenSigVtName" SG_ 33 InstructionSetCheckFault "InstructionSetCheckFault";
BA_ "GenSigVtEn" SG_ 33 StackOverflowFault "StackOverflowFault";
BA_ "GenSigVtName" SG_ 33 StackOverflowFault "StackOverflowFault";
BA_ "GenSigVtEn" SG_ 33 WatchdogFault "WatchdogFault";
BA_ "GenSigVtName" SG_ 33 WatchdogFault "WatchdogFault";
BA_ "GenSigVtEn" SG_ 33 PLLLockFault "PLLLockFault";
BA_ "GenSigVtName" SG_ 33 PLLLockFault "PLLLockFault";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte1_bit1 "Active_Flt_Latched_byte1_bit1";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte1_bit1 "Active_Flt_Latched_byte1_bit1";
BA_ "GenSigVtEn" SG_ 33 RAMMemoryTestFault "RAMMemoryTestFault";
BA_ "GenSigVtName" SG_ 33 RAMMemoryTestFault "RAMMemoryTestFault";
BA_ "GenSigVtName" SG_ 33 USCValidationFault "USCValidationFault";
BA_ "GenSigVtEn" SG_ 33 USCValidationFault "USCValidationFault";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte0_bit6 "Active_Flt_Latched_byte0_bit6";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte0_bit6 "Active_Flt_Latched_byte0_bit6";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte0_bit5 "Active_Flt_Latched_byte0_bit5";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte0_bit5 "Active_Flt_Latched_byte0_bit5";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte0_bit4 "Active_Flt_Latched_byte0_bit4";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte0_bit4 "Active_Flt_Latched_byte0_bit4";
BA_ "GenSigVtEn" SG_ 33 Active_Flt_Latched_byte0_bit3 "Active_Flt_Latched_byte0_bit3";
BA_ "GenSigVtName" SG_ 33 Active_Flt_Latched_byte0_bit3 "Active_Flt_Latched_byte0_bit3";
BA_ "GenSigVtEn" SG_ 33 KeepAliveChecksumFault "KeepAliveChecksumFault";
BA_ "GenSigVtName" SG_ 33 KeepAliveChecksumFault "KeepAliveChecksumFault";
BA_ "GenSigVtEn" SG_ 33 ProgramCalibrationFlashChecksum "ProgramCalibrationFlashChecksum";
BA_ "GenSigVtName" SG_ 33 ProgramCalibrationFlashChecksum "ProgramCalibrationFlashChecksum";
BA_ "GenSigVtEn" SG_ 33 ApplicationFlashChecksumFault "ApplicationFlashChecksumFault";
BA_ "GenSigVtName" SG_ 33 ApplicationFlashChecksumFault "ApplicationFlashChecksumFault";
BA_ "GenMsgNrOfRepetition" BO_ 500 0;
BA_ "GenMsgSendType" BO_ 500 1;
BA_ "GenSigSendType" SG_ 500 MRR_xcp_daq_resp_byte7 0;
BA_ "GenSigCmt" SG_ 500 MRR_xcp_daq_resp_byte7 "MRR_xcp_daq_resp_byte7";
BA_ "GenSigSendType" SG_ 500 MRR_xcp_daq_resp_byte6 0;
BA_ "GenSigCmt" SG_ 500 MRR_xcp_daq_resp_byte6 "MRR_xcp_daq_resp_byte6";
BA_ "GenSigSendType" SG_ 500 MRR_xcp_daq_resp_byte5 0;
BA_ "GenSigCmt" SG_ 500 MRR_xcp_daq_resp_byte5 "MRR_xcp_daq_resp_byte5";
BA_ "GenSigSendType" SG_ 500 MRR_xcp_daq_resp_byte4 0;
BA_ "GenSigCmt" SG_ 500 MRR_xcp_daq_resp_byte4 "MRR_xcp_daq_resp_byte4";
BA_ "GenSigSendType" SG_ 500 MRR_xcp_daq_resp_byte3 0;
BA_ "GenSigCmt" SG_ 500 MRR_xcp_daq_resp_byte3 "MRR_xcp_daq_resp_byte3";
BA_ "GenSigSendType" SG_ 500 MRR_xcp_daq_resp_byte2 0;
BA_ "GenSigCmt" SG_ 500 MRR_xcp_daq_resp_byte2 "MRR_xcp_daq_resp_byte2";
BA_ "GenSigSendType" SG_ 500 MRR_xcp_daq_resp_byte1 0;
BA_ "GenSigCmt" SG_ 500 MRR_xcp_daq_resp_byte1 "MRR_xcp_daq_resp_byte1";
BA_ "GenSigSendType" SG_ 500 MRR_xcp_daq_resp_byte0 0;
BA_ "GenSigCmt" SG_ 500 MRR_xcp_daq_resp_byte0 "MRR_xcp_daq_resp_byte0";
BA_ "GenMsgNrOfRepetition" BO_ 499 0;
BA_ "GenMsgSendType" BO_ 499 1;
BA_ "GenSigSendType" SG_ 499 MRR_xcp_dto_resp_byte7 0;
BA_ "GenSigCmt" SG_ 499 MRR_xcp_dto_resp_byte7 "MRR_xcp_dto_resp_byte7";
BA_ "GenSigSendType" SG_ 499 MRR_xcp_dto_resp_byte6 0;
BA_ "GenSigCmt" SG_ 499 MRR_xcp_dto_resp_byte6 "MRR_xcp_dto_resp_byte6";
BA_ "GenSigSendType" SG_ 499 MRR_xcp_dto_resp_byte5 0;
BA_ "GenSigCmt" SG_ 499 MRR_xcp_dto_resp_byte5 "MRR_xcp_dto_resp_byte5";
BA_ "GenSigSendType" SG_ 499 MRR_xcp_dto_resp_byte4 0;
BA_ "GenSigCmt" SG_ 499 MRR_xcp_dto_resp_byte4 "MRR_xcp_dto_resp_byte4";
BA_ "GenSigSendType" SG_ 499 MRR_xcp_dto_resp_byte3 0;
BA_ "GenSigCmt" SG_ 499 MRR_xcp_dto_resp_byte3 "MRR_xcp_dto_resp_byte3";
BA_ "GenSigSendType" SG_ 499 MRR_xcp_dto_resp_byte2 0;
BA_ "GenSigCmt" SG_ 499 MRR_xcp_dto_resp_byte2 "MRR_xcp_dto_resp_byte2";
BA_ "GenSigSendType" SG_ 499 MRR_xcp_dto_resp_byte1 0;
BA_ "GenSigCmt" SG_ 499 MRR_xcp_dto_resp_byte1 "MRR_xcp_dto_resp_byte1";
BA_ "GenSigSendType" SG_ 499 MRR_xcp_dto_resp_byte0 0;
BA_ "GenSigCmt" SG_ 499 MRR_xcp_dto_resp_byte0 "MRR_xcp_dto_resp_byte0";
BA_ "GenMsgNrOfRepetition" BO_ 497 0;
BA_ "GenMsgSendType" BO_ 497 1;
BA_ "GenSigSendType" SG_ 497 MRR_xcp_cto_resp_byte7 0;
BA_ "GenSigCmt" SG_ 497 MRR_xcp_cto_resp_byte7 "MRR_xcp_cto_resp_byte7";
BA_ "GenSigSendType" SG_ 497 MRR_xcp_cto_resp_byte6 0;
BA_ "GenSigCmt" SG_ 497 MRR_xcp_cto_resp_byte6 "MRR_xcp_cto_resp_byte6";
BA_ "GenSigSendType" SG_ 497 MRR_xcp_cto_resp_byte5 0;
BA_ "GenSigCmt" SG_ 497 MRR_xcp_cto_resp_byte5 "MRR_xcp_cto_resp_byte5";
BA_ "GenSigSendType" SG_ 497 MRR_xcp_cto_resp_byte4 0;
BA_ "GenSigCmt" SG_ 497 MRR_xcp_cto_resp_byte4 "MRR_xcp_cto_resp_byte4";
BA_ "GenSigSendType" SG_ 497 MRR_xcp_cto_resp_byte3 0;
BA_ "GenSigCmt" SG_ 497 MRR_xcp_cto_resp_byte3 "MRR_xcp_cto_resp_byte3";
BA_ "GenSigSendType" SG_ 497 MRR_xcp_cto_resp_byte2 0;
BA_ "GenSigCmt" SG_ 497 MRR_xcp_cto_resp_byte2 "MRR_xcp_cto_resp_byte2";
BA_ "GenSigSendType" SG_ 497 MRR_xcp_cto_resp_byte1 0;
BA_ "GenSigCmt" SG_ 497 MRR_xcp_cto_resp_byte1 "MRR_xcp_cto_resp_byte1";
BA_ "GenSigSendType" SG_ 497 MRR_xcp_cto_resp_byte0 0;
BA_ "GenSigCmt" SG_ 497 MRR_xcp_cto_resp_byte0 "MRR_xcp_cto_resp_byte0";
BA_ "GenMsgSendType" BO_ 1900 1;
BA_ "GenMsgNrOfRepetition" BO_ 1900 0;
BA_ "DiagResponse" BO_ 1900 1;
BA_ "GenSigCmt" SG_ 1900 TesterPhysicalResCCM "TesterPhysicalResCCM";
BA_ "GenSigSendType" SG_ 1900 TesterPhysicalResCCM 0;
BA_ "GenMsgSendType" BO_ 261 0;
BA_ "GenMsgCycleTime" BO_ 261 1000;
BA_ "GenMsgNrOfRepetition" BO_ 261 0;
BA_ "GenSigCmt" SG_ 261 CAN_SEQUENCE_NUMBER "CAN_SEQUENCE_NUMBER";
BA_ "GenSigCmt" SG_ 261 CAN_SERIAL_NUMBER "CAN_SERIAL_NUMBER";
BA_ "GenSigSendType" SG_ 261 CAN_SERIAL_NUMBER 0;
BA_ "GenMsgSendType" BO_ 264 1;
BA_ "GenMsgNrOfRepetition" BO_ 264 0;
BA_ "GenSigSendType" SG_ 264 CAN_PBL_Field_Revision 0;
BA_ "GenSigSendType" SG_ 264 CAN_PBL_Promote_Revision 0;
BA_ "GenSigSendType" SG_ 264 CAN_SW_Field_Revision 0;
BA_ "GenSigSendType" SG_ 264 CAN_SW_Promote_Revision 0;
BA_ "GenSigSendType" SG_ 264 CAN_SW_Release_Revision 0;
BA_ "GenSigSendType" SG_ 264 CAN_PBL_Release_Revision 0;
BA_ "GenMsgSendType" BO_ 373 1;
BA_ "NetworkInitialization" BO_ 373 0;
BA_ "GenMsgNrOfRepetition" BO_ 373 0;
BA_ "GenSigSendType" SG_ 373 CAN_SENSOR_POLARITY 0;
BA_ "GenSigCmt" SG_ 373 CAN_SENSOR_POLARITY "CAN_SENSOR_POLARITY";
BA_ "GenSigSendType" SG_ 373 CAN_SENSOR_LAT_OFFSET 0;
BA_ "GenSigCmt" SG_ 373 CAN_SENSOR_LAT_OFFSET "CAN_SENSOR_LAT_OFFSET";
BA_ "GenSigSendType" SG_ 373 CAN_SENSOR_LONG_OFFSET 0;
BA_ "GenSigCmt" SG_ 373 CAN_SENSOR_LONG_OFFSET "CAN_SENSOR_LONG_OFFSET";
BA_ "GenSigSendType" SG_ 373 CAN_SENSOR_HANGLE_OFFSET 0;
BA_ "GenSigCmt" SG_ 373 CAN_SENSOR_HANGLE_OFFSET "CAN_SENSOR_HANGLE_OFFSET";
BA_ "GenSigStartValue" SG_ 373 CAN_SENSOR_HANGLE_OFFSET 0;
BA_ "GenMsgSendType" BO_ 372 1;
BA_ "NetworkInitialization" BO_ 372 0;
BA_ "GenMsgNrOfRepetition" BO_ 372 0;
BA_ "GenSigSendType" SG_ 372 CAN_SENSOR_FOV_HOR 0;
BA_ "GenSigCmt" SG_ 372 CAN_SENSOR_FOV_HOR "CAN_SENSOR_FOV_HOR";
BA_ "GenSigStartValue" SG_ 372 CAN_SENSOR_FOV_HOR 0;
BA_ "GenSigSendType" SG_ 372 CAN_DOPPLER_COVERAGE 0;
BA_ "GenSigCmt" SG_ 372 CAN_DOPPLER_COVERAGE "CAN_DOPPLER_COVERAGE";
BA_ "GenSigStartValue" SG_ 372 CAN_DOPPLER_COVERAGE 0;
BA_ "GenSigSendType" SG_ 372 CAN_RANGE_COVERAGE 0;
BA_ "GenSigCmt" SG_ 372 CAN_RANGE_COVERAGE "CAN_RANGE_COVERAGE";
BA_ "GenMsgSendType" BO_ 371 1;
BA_ "NetworkInitialization" BO_ 371 0;
BA_ "GenMsgNrOfRepetition" BO_ 371 0;
BA_ "GenSigVtEn" SG_ 371 CAN_AUTO_ALIGN_HANGLE_QF "CAN_AUTO_ALIGN_HANGLE_QF";
BA_ "GenSigVtName" SG_ 371 CAN_AUTO_ALIGN_HANGLE_QF "CAN_AUTO_ALIGN_HANGLE_QF";
BA_ "GenSigSendType" SG_ 371 CAN_AUTO_ALIGN_HANGLE_QF 0;
BA_ "GenSigCmt" SG_ 371 CAN_AUTO_ALIGN_HANGLE_QF "CAN_AUTO_ALIGN_HANGLE_QF";
BA_ "GenSigVtEn" SG_ 371 CAN_ALIGNMENT_STATUS "CAN_ALIGNMENT_STATUS";
BA_ "GenSigVtName" SG_ 371 CAN_ALIGNMENT_STATUS "CAN_ALIGNMENT_STATUS";
BA_ "GenSigSendType" SG_ 371 CAN_ALIGNMENT_STATUS 0;
BA_ "GenSigCmt" SG_ 371 CAN_ALIGNMENT_STATUS "CAN_ALIGNMENT_STATUS";
BA_ "GenSigVtEn" SG_ 371 CAN_ALIGNMENT_STATE "CAN_ALIGNMENT_STATE";
BA_ "GenSigVtName" SG_ 371 CAN_ALIGNMENT_STATE "CAN_ALIGNMENT_STATE";
BA_ "GenSigSendType" SG_ 371 CAN_ALIGNMENT_STATE 0;
BA_ "GenSigCmt" SG_ 371 CAN_ALIGNMENT_STATE "CAN_ALIGNMENT_STATE";
BA_ "GenSigSendType" SG_ 371 CAN_AUTO_ALIGN_HANGLE_REF 0;
BA_ "GenSigStartValue" SG_ 371 CAN_AUTO_ALIGN_HANGLE_REF 0;
BA_ "GenSigCmt" SG_ 371 CAN_AUTO_ALIGN_HANGLE_REF "CAN_AUTO_ALIGN_HANGLE_REF";
BA_ "GenSigStartValue" SG_ 371 CAN_AUTO_ALIGN_HANGLE 0;
BA_ "GenSigSendType" SG_ 371 CAN_AUTO_ALIGN_HANGLE 0;
BA_ "GenSigCmt" SG_ 371 CAN_AUTO_ALIGN_HANGLE "CAN_AUTO_ALIGN_HANGLE";
BA_ "GenMsgSendType" BO_ 369 1;
BA_ "NetworkInitialization" BO_ 369 0;
BA_ "GenMsgNrOfRepetition" BO_ 369 0;
BA_ "GenSigCmt" SG_ 369 CAN_DET_TIME_SINCE_MEAS "CAN_DET_TIME_SINCE_MEAS";
BA_ "GenSigSendType" SG_ 369 CAN_DET_TIME_SINCE_MEAS 0;
BA_ "GenSigSendType" SG_ 369 CAN_SENSOR_TIME_STAMP 0;
BA_ "GenSigCmt" SG_ 369 CAN_SENSOR_TIME_STAMP "CAN_SENSOR_TIME_STAMP";
BA_ "GenMsgSendType" BO_ 368 1;
BA_ "NetworkInitialization" BO_ 368 0;
BA_ "GenMsgNrOfRepetition" BO_ 368 0;
BA_ "GenSigSendType" SG_ 368 CAN_ALIGN_UPDATES_DONE 0;
BA_ "GenSigCmt" SG_ 368 CAN_ALIGN_UPDATES_DONE "CAN_ALIGN_UPDATES_DONE";
BA_ "GenSigSendType" SG_ 368 CAN_SCAN_INDEX 0;
BA_ "GenSigCmt" SG_ 368 CAN_SCAN_INDEX "CAN_SCAN_INDEX";
BA_ "GenSigSendType" SG_ 368 CAN_NUMBER_OF_DET 0;
BA_ "GenSigCmt" SG_ 368 CAN_NUMBER_OF_DET "CAN_NUMBER_OF_DET";
BA_ "GenSigSendType" SG_ 368 CAN_LOOK_ID 0;
BA_ "GenSigCmt" SG_ 368 CAN_LOOK_ID "CAN_LOOK_ID";
BA_ "GenSigSendType" SG_ 368 CAN_LOOK_INDEX 0;
BA_ "GenSigCmt" SG_ 368 CAN_LOOK_INDEX "CAN_LOOK_INDEX";
BA_ "GenMsgSendType" BO_ 265 1;
BA_ "NetworkInitialization" BO_ 265 0;
BA_ "GenMsgNrOfRepetition" BO_ 265 0;
BA_ "GenSigCmt" SG_ 265 CAN_BATT_VOLTS "CAN_BATT_VOLTS";
BA_ "GenSigCmt" SG_ 265 CAN_1_25_V "CAN_1_25_V";
BA_ "GenSigCmt" SG_ 265 CAN_5_V "CAN_5_V";
BA_ "GenSigCmt" SG_ 265 CAN_3_3_V_RAW "CAN_3_3_V_RAW";
BA_ "GenSigCmt" SG_ 265 CAN_3_3_V_DAC "CAN_3_3_V_DAC";
BA_ "GenSigSendType" SG_ 265 CAN_MMIC_Temp1 0;
BA_ "GenSigCmt" SG_ 265 CAN_MMIC_Temp1 "CAN_MMIC_Temp1";
BA_ "GenSigStartValue" SG_ 265 CAN_MMIC_Temp1 0;
BA_ "GenSigSendType" SG_ 265 CAN_Processor_Thermistor 0;
BA_ "GenSigCmt" SG_ 265 CAN_Processor_Thermistor "CAN_Processor_Thermistor";
BA_ "GenSigStartValue" SG_ 265 CAN_Processor_Thermistor 0;
BA_ "GenSigSendType" SG_ 265 CAN_Processor_Temp1 0;
BA_ "GenSigCmt" SG_ 265 CAN_Processor_Temp1 "CAN_Processor_Temp1";
BA_ "GenSigStartValue" SG_ 265 CAN_Processor_Temp1 0;
EOF

build_ba "04"

cat <<EOF >> ${OUT_FILENAME}
BA_ "GenMsgSendType" BO_ 351 1;
BA_ "GenMsgILSupport" BO_ 351 1;
BA_ "GenMsgNrOfRepetition" BO_ 351 0;
BA_ "GenMsgCycleTime" BO_ 351 0;
BA_ "NetworkInitialization" BO_ 351 0;
BA_ "GenMsgDelayTime" BO_ 351 0;
BA_ "GenSigVtEn" SG_ 351 CAN_DET_CONFID_AZIMUTH_64 "CAN_DET_CONFID_AZIMUTH_64";
BA_ "GenSigVtName" SG_ 351 CAN_DET_CONFID_AZIMUTH_64 "CAN_DET_CONFID_AZIMUTH_64";
BA_ "GenSigSendType" SG_ 351 CAN_DET_CONFID_AZIMUTH_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_CONFID_AZIMUTH_64 "CAN_DET_CONFID_AZIMUTH_64";
BA_ "GenSigSendType" SG_ 351 CAN_DET_SUPER_RES_TARGET_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_SUPER_RES_TARGET_64 "CAN_DET_SUPER_RES_TARGET_64";
BA_ "GenSigSendType" SG_ 351 CAN_DET_ND_TARGET_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_ND_TARGET_64 "CAN_DET_ND_TARGET_64";
BA_ "GenSigSendType" SG_ 351 CAN_DET_HOST_VEH_CLUTTER_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_HOST_VEH_CLUTTER_64 "CAN_DET_HOST_VEH_CLUTTER_64";
BA_ "GenSigSendType" SG_ 351 CAN_DET_VALID_LEVEL_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_VALID_LEVEL_64 "CAN_DET_VALID_LEVEL_64";
BA_ "GenSigStartValue" SG_ 351 CAN_DET_AZIMUTH_64 0;
BA_ "GenSigSendType" SG_ 351 CAN_DET_AZIMUTH_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_AZIMUTH_64 "CAN_DET_AZIMUTH_64";
BA_ "GenSigSendType" SG_ 351 CAN_DET_RANGE_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_RANGE_64 "CAN_DET_RANGE_64";
BA_ "GenSigStartValue" SG_ 351 CAN_DET_RANGE_RATE_64 0;
BA_ "GenSigSendType" SG_ 351 CAN_DET_RANGE_RATE_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_RANGE_RATE_64 "CAN_DET_RANGE_RATE_64";
BA_ "GenSigSendType" SG_ 351 CAN_DET_AMPLITUDE_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_DET_AMPLITUDE_64 "CAN_DET_AMPLITUDE_64";
BA_ "GenSigSendType" SG_ 351 CAN_SCAN_INDEX_2LSB_64 0;
BA_ "GenSigCmt" SG_ 351 CAN_SCAN_INDEX_2LSB_64 "CAN_SCAN_INDEX_2LSB_64";
BA_ "GenMsgSendType" BO_ 350 1;
BA_ "GenMsgILSupport" BO_ 350 1;
BA_ "GenMsgNrOfRepetition" BO_ 350 0;
BA_ "GenMsgCycleTime" BO_ 350 0;
BA_ "NetworkInitialization" BO_ 350 0;
BA_ "GenMsgDelayTime" BO_ 350 0;
BA_ "GenSigVtEn" SG_ 350 CAN_DET_CONFID_AZIMUTH_63 "CAN_DET_CONFID_AZIMUTH_63";
BA_ "GenSigVtName" SG_ 350 CAN_DET_CONFID_AZIMUTH_63 "CAN_DET_CONFID_AZIMUTH_63";
BA_ "GenSigSendType" SG_ 350 CAN_DET_CONFID_AZIMUTH_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_CONFID_AZIMUTH_63 "CAN_DET_CONFID_AZIMUTH_63";
BA_ "GenSigSendType" SG_ 350 CAN_DET_SUPER_RES_TARGET_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_SUPER_RES_TARGET_63 "CAN_DET_SUPER_RES_TARGET_63";
BA_ "GenSigSendType" SG_ 350 CAN_DET_ND_TARGET_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_ND_TARGET_63 "CAN_DET_ND_TARGET_63";
BA_ "GenSigSendType" SG_ 350 CAN_DET_HOST_VEH_CLUTTER_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_HOST_VEH_CLUTTER_63 "CAN_DET_HOST_VEH_CLUTTER_63";
BA_ "GenSigSendType" SG_ 350 CAN_DET_VALID_LEVEL_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_VALID_LEVEL_63 "CAN_DET_VALID_LEVEL_63";
BA_ "GenSigStartValue" SG_ 350 CAN_DET_AZIMUTH_63 0;
BA_ "GenSigSendType" SG_ 350 CAN_DET_AZIMUTH_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_AZIMUTH_63 "CAN_DET_AZIMUTH_63";
BA_ "GenSigSendType" SG_ 350 CAN_DET_RANGE_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_RANGE_63 "CAN_DET_RANGE_63";
BA_ "GenSigStartValue" SG_ 350 CAN_DET_RANGE_RATE_63 0;
BA_ "GenSigSendType" SG_ 350 CAN_DET_RANGE_RATE_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_RANGE_RATE_63 "CAN_DET_RANGE_RATE_63";
BA_ "GenSigSendType" SG_ 350 CAN_DET_AMPLITUDE_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_DET_AMPLITUDE_63 "CAN_DET_AMPLITUDE_63";
BA_ "GenSigSendType" SG_ 350 CAN_SCAN_INDEX_2LSB_63 0;
BA_ "GenSigCmt" SG_ 350 CAN_SCAN_INDEX_2LSB_63 "CAN_SCAN_INDEX_2LSB_63";
BA_ "GenMsgSendType" BO_ 349 1;
BA_ "GenMsgILSupport" BO_ 349 1;
BA_ "GenMsgNrOfRepetition" BO_ 349 0;
BA_ "GenMsgCycleTime" BO_ 349 0;
BA_ "NetworkInitialization" BO_ 349 0;
BA_ "GenMsgDelayTime" BO_ 349 0;
BA_ "GenSigVtEn" SG_ 349 CAN_DET_CONFID_AZIMUTH_62 "CAN_DET_CONFID_AZIMUTH_62";
BA_ "GenSigVtName" SG_ 349 CAN_DET_CONFID_AZIMUTH_62 "CAN_DET_CONFID_AZIMUTH_62";
BA_ "GenSigSendType" SG_ 349 CAN_DET_CONFID_AZIMUTH_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_CONFID_AZIMUTH_62 "CAN_DET_CONFID_AZIMUTH_62";
BA_ "GenSigSendType" SG_ 349 CAN_DET_SUPER_RES_TARGET_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_SUPER_RES_TARGET_62 "CAN_DET_SUPER_RES_TARGET_62";
BA_ "GenSigSendType" SG_ 349 CAN_DET_ND_TARGET_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_ND_TARGET_62 "CAN_DET_ND_TARGET_62";
BA_ "GenSigSendType" SG_ 349 CAN_DET_HOST_VEH_CLUTTER_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_HOST_VEH_CLUTTER_62 "CAN_DET_HOST_VEH_CLUTTER_62";
BA_ "GenSigSendType" SG_ 349 CAN_DET_VALID_LEVEL_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_VALID_LEVEL_62 "CAN_DET_VALID_LEVEL_62";
BA_ "GenSigStartValue" SG_ 349 CAN_DET_AZIMUTH_62 0;
BA_ "GenSigSendType" SG_ 349 CAN_DET_AZIMUTH_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_AZIMUTH_62 "CAN_DET_AZIMUTH_62";
BA_ "GenSigSendType" SG_ 349 CAN_DET_RANGE_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_RANGE_62 "CAN_DET_RANGE_62";
BA_ "GenSigStartValue" SG_ 349 CAN_DET_RANGE_RATE_62 0;
BA_ "GenSigSendType" SG_ 349 CAN_DET_RANGE_RATE_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_RANGE_RATE_62 "CAN_DET_RANGE_RATE_62";
BA_ "GenSigSendType" SG_ 349 CAN_DET_AMPLITUDE_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_DET_AMPLITUDE_62 "CAN_DET_AMPLITUDE_62";
BA_ "GenSigSendType" SG_ 349 CAN_SCAN_INDEX_2LSB_62 0;
BA_ "GenSigCmt" SG_ 349 CAN_SCAN_INDEX_2LSB_62 "CAN_SCAN_INDEX_2LSB_62";
BA_ "GenMsgSendType" BO_ 348 1;
BA_ "GenMsgILSupport" BO_ 348 1;
BA_ "GenMsgNrOfRepetition" BO_ 348 0;
BA_ "GenMsgCycleTime" BO_ 348 0;
BA_ "NetworkInitialization" BO_ 348 0;
BA_ "GenMsgDelayTime" BO_ 348 0;
BA_ "GenSigVtEn" SG_ 348 CAN_DET_CONFID_AZIMUTH_61 "CAN_DET_CONFID_AZIMUTH_61";
BA_ "GenSigVtName" SG_ 348 CAN_DET_CONFID_AZIMUTH_61 "CAN_DET_CONFID_AZIMUTH_61";
BA_ "GenSigSendType" SG_ 348 CAN_DET_CONFID_AZIMUTH_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_CONFID_AZIMUTH_61 "CAN_DET_CONFID_AZIMUTH_61";
BA_ "GenSigSendType" SG_ 348 CAN_DET_SUPER_RES_TARGET_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_SUPER_RES_TARGET_61 "CAN_DET_SUPER_RES_TARGET_61";
BA_ "GenSigSendType" SG_ 348 CAN_DET_ND_TARGET_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_ND_TARGET_61 "CAN_DET_ND_TARGET_61";
BA_ "GenSigSendType" SG_ 348 CAN_DET_HOST_VEH_CLUTTER_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_HOST_VEH_CLUTTER_61 "CAN_DET_HOST_VEH_CLUTTER_61";
BA_ "GenSigSendType" SG_ 348 CAN_DET_VALID_LEVEL_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_VALID_LEVEL_61 "CAN_DET_VALID_LEVEL_61";
BA_ "GenSigStartValue" SG_ 348 CAN_DET_AZIMUTH_61 0;
BA_ "GenSigSendType" SG_ 348 CAN_DET_AZIMUTH_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_AZIMUTH_61 "CAN_DET_AZIMUTH_61";
BA_ "GenSigSendType" SG_ 348 CAN_DET_RANGE_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_RANGE_61 "CAN_DET_RANGE_61";
BA_ "GenSigStartValue" SG_ 348 CAN_DET_RANGE_RATE_61 0;
BA_ "GenSigSendType" SG_ 348 CAN_DET_RANGE_RATE_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_RANGE_RATE_61 "CAN_DET_RANGE_RATE_61";
BA_ "GenSigSendType" SG_ 348 CAN_DET_AMPLITUDE_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_DET_AMPLITUDE_61 "CAN_DET_AMPLITUDE_61";
BA_ "GenSigSendType" SG_ 348 CAN_SCAN_INDEX_2LSB_61 0;
BA_ "GenSigCmt" SG_ 348 CAN_SCAN_INDEX_2LSB_61 "CAN_SCAN_INDEX_2LSB_61";
BA_ "GenMsgSendType" BO_ 347 1;
BA_ "GenMsgILSupport" BO_ 347 1;
BA_ "GenMsgNrOfRepetition" BO_ 347 0;
BA_ "GenMsgCycleTime" BO_ 347 0;
BA_ "NetworkInitialization" BO_ 347 0;
BA_ "GenMsgDelayTime" BO_ 347 0;
BA_ "GenSigVtEn" SG_ 347 CAN_DET_CONFID_AZIMUTH_60 "CAN_DET_CONFID_AZIMUTH_60";
BA_ "GenSigVtName" SG_ 347 CAN_DET_CONFID_AZIMUTH_60 "CAN_DET_CONFID_AZIMUTH_60";
BA_ "GenSigSendType" SG_ 347 CAN_DET_CONFID_AZIMUTH_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_CONFID_AZIMUTH_60 "CAN_DET_CONFID_AZIMUTH_60";
BA_ "GenSigSendType" SG_ 347 CAN_DET_SUPER_RES_TARGET_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_SUPER_RES_TARGET_60 "CAN_DET_SUPER_RES_TARGET_60";
BA_ "GenSigSendType" SG_ 347 CAN_DET_ND_TARGET_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_ND_TARGET_60 "CAN_DET_ND_TARGET_60";
BA_ "GenSigSendType" SG_ 347 CAN_DET_HOST_VEH_CLUTTER_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_HOST_VEH_CLUTTER_60 "CAN_DET_HOST_VEH_CLUTTER_60";
BA_ "GenSigSendType" SG_ 347 CAN_DET_VALID_LEVEL_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_VALID_LEVEL_60 "CAN_DET_VALID_LEVEL_60";
BA_ "GenSigStartValue" SG_ 347 CAN_DET_AZIMUTH_60 0;
BA_ "GenSigSendType" SG_ 347 CAN_DET_AZIMUTH_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_AZIMUTH_60 "CAN_DET_AZIMUTH_60";
BA_ "GenSigSendType" SG_ 347 CAN_DET_RANGE_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_RANGE_60 "CAN_DET_RANGE_60";
BA_ "GenSigStartValue" SG_ 347 CAN_DET_RANGE_RATE_60 0;
BA_ "GenSigSendType" SG_ 347 CAN_DET_RANGE_RATE_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_RANGE_RATE_60 "CAN_DET_RANGE_RATE_60";
BA_ "GenSigSendType" SG_ 347 CAN_DET_AMPLITUDE_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_DET_AMPLITUDE_60 "CAN_DET_AMPLITUDE_60";
BA_ "GenSigSendType" SG_ 347 CAN_SCAN_INDEX_2LSB_60 0;
BA_ "GenSigCmt" SG_ 347 CAN_SCAN_INDEX_2LSB_60 "CAN_SCAN_INDEX_2LSB_60";
BA_ "GenMsgSendType" BO_ 346 1;
BA_ "GenMsgILSupport" BO_ 346 1;
BA_ "GenMsgNrOfRepetition" BO_ 346 0;
BA_ "GenMsgCycleTime" BO_ 346 0;
BA_ "NetworkInitialization" BO_ 346 0;
BA_ "GenMsgDelayTime" BO_ 346 0;
BA_ "GenSigVtEn" SG_ 346 CAN_DET_CONFID_AZIMUTH_59 "CAN_DET_CONFID_AZIMUTH_59";
BA_ "GenSigVtName" SG_ 346 CAN_DET_CONFID_AZIMUTH_59 "CAN_DET_CONFID_AZIMUTH_59";
BA_ "GenSigSendType" SG_ 346 CAN_DET_CONFID_AZIMUTH_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_CONFID_AZIMUTH_59 "CAN_DET_CONFID_AZIMUTH_59";
BA_ "GenSigSendType" SG_ 346 CAN_DET_SUPER_RES_TARGET_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_SUPER_RES_TARGET_59 "CAN_DET_SUPER_RES_TARGET_59";
BA_ "GenSigSendType" SG_ 346 CAN_DET_ND_TARGET_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_ND_TARGET_59 "CAN_DET_ND_TARGET_59";
BA_ "GenSigSendType" SG_ 346 CAN_DET_HOST_VEH_CLUTTER_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_HOST_VEH_CLUTTER_59 "CAN_DET_HOST_VEH_CLUTTER_59";
BA_ "GenSigSendType" SG_ 346 CAN_DET_VALID_LEVEL_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_VALID_LEVEL_59 "CAN_DET_VALID_LEVEL_59";
BA_ "GenSigStartValue" SG_ 346 CAN_DET_AZIMUTH_59 0;
BA_ "GenSigSendType" SG_ 346 CAN_DET_AZIMUTH_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_AZIMUTH_59 "CAN_DET_AZIMUTH_59";
BA_ "GenSigSendType" SG_ 346 CAN_DET_RANGE_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_RANGE_59 "CAN_DET_RANGE_59";
BA_ "GenSigStartValue" SG_ 346 CAN_DET_RANGE_RATE_59 0;
BA_ "GenSigSendType" SG_ 346 CAN_DET_RANGE_RATE_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_RANGE_RATE_59 "CAN_DET_RANGE_RATE_59";
BA_ "GenSigSendType" SG_ 346 CAN_DET_AMPLITUDE_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_DET_AMPLITUDE_59 "CAN_DET_AMPLITUDE_59";
BA_ "GenSigSendType" SG_ 346 CAN_SCAN_INDEX_2LSB_59 0;
BA_ "GenSigCmt" SG_ 346 CAN_SCAN_INDEX_2LSB_59 "CAN_SCAN_INDEX_2LSB_59";
BA_ "GenMsgSendType" BO_ 345 1;
BA_ "GenMsgILSupport" BO_ 345 1;
BA_ "GenMsgNrOfRepetition" BO_ 345 0;
BA_ "GenMsgCycleTime" BO_ 345 0;
BA_ "NetworkInitialization" BO_ 345 0;
BA_ "GenMsgDelayTime" BO_ 345 0;
BA_ "GenSigVtEn" SG_ 345 CAN_DET_CONFID_AZIMUTH_58 "CAN_DET_CONFID_AZIMUTH_58";
BA_ "GenSigVtName" SG_ 345 CAN_DET_CONFID_AZIMUTH_58 "CAN_DET_CONFID_AZIMUTH_58";
BA_ "GenSigSendType" SG_ 345 CAN_DET_CONFID_AZIMUTH_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_CONFID_AZIMUTH_58 "CAN_DET_CONFID_AZIMUTH_58";
BA_ "GenSigSendType" SG_ 345 CAN_DET_SUPER_RES_TARGET_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_SUPER_RES_TARGET_58 "CAN_DET_SUPER_RES_TARGET_58";
BA_ "GenSigSendType" SG_ 345 CAN_DET_ND_TARGET_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_ND_TARGET_58 "CAN_DET_ND_TARGET_58";
BA_ "GenSigSendType" SG_ 345 CAN_DET_HOST_VEH_CLUTTER_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_HOST_VEH_CLUTTER_58 "CAN_DET_HOST_VEH_CLUTTER_58";
BA_ "GenSigSendType" SG_ 345 CAN_DET_VALID_LEVEL_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_VALID_LEVEL_58 "CAN_DET_VALID_LEVEL_58";
BA_ "GenSigStartValue" SG_ 345 CAN_DET_AZIMUTH_58 0;
BA_ "GenSigSendType" SG_ 345 CAN_DET_AZIMUTH_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_AZIMUTH_58 "CAN_DET_AZIMUTH_58";
BA_ "GenSigSendType" SG_ 345 CAN_DET_RANGE_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_RANGE_58 "CAN_DET_RANGE_58";
BA_ "GenSigStartValue" SG_ 345 CAN_DET_RANGE_RATE_58 0;
BA_ "GenSigSendType" SG_ 345 CAN_DET_RANGE_RATE_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_RANGE_RATE_58 "CAN_DET_RANGE_RATE_58";
BA_ "GenSigSendType" SG_ 345 CAN_DET_AMPLITUDE_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_DET_AMPLITUDE_58 "CAN_DET_AMPLITUDE_58";
BA_ "GenSigSendType" SG_ 345 CAN_SCAN_INDEX_2LSB_58 0;
BA_ "GenSigCmt" SG_ 345 CAN_SCAN_INDEX_2LSB_58 "CAN_SCAN_INDEX_2LSB_58";
BA_ "GenMsgSendType" BO_ 344 1;
BA_ "GenMsgILSupport" BO_ 344 1;
BA_ "GenMsgNrOfRepetition" BO_ 344 0;
BA_ "GenMsgCycleTime" BO_ 344 0;
BA_ "NetworkInitialization" BO_ 344 0;
BA_ "GenMsgDelayTime" BO_ 344 0;
BA_ "GenSigVtEn" SG_ 344 CAN_DET_CONFID_AZIMUTH_57 "CAN_DET_CONFID_AZIMUTH_57";
BA_ "GenSigVtName" SG_ 344 CAN_DET_CONFID_AZIMUTH_57 "CAN_DET_CONFID_AZIMUTH_57";
BA_ "GenSigSendType" SG_ 344 CAN_DET_CONFID_AZIMUTH_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_CONFID_AZIMUTH_57 "CAN_DET_CONFID_AZIMUTH_57";
BA_ "GenSigSendType" SG_ 344 CAN_DET_SUPER_RES_TARGET_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_SUPER_RES_TARGET_57 "CAN_DET_SUPER_RES_TARGET_57";
BA_ "GenSigSendType" SG_ 344 CAN_DET_ND_TARGET_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_ND_TARGET_57 "CAN_DET_ND_TARGET_57";
BA_ "GenSigSendType" SG_ 344 CAN_DET_HOST_VEH_CLUTTER_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_HOST_VEH_CLUTTER_57 "CAN_DET_HOST_VEH_CLUTTER_57";
BA_ "GenSigSendType" SG_ 344 CAN_DET_VALID_LEVEL_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_VALID_LEVEL_57 "CAN_DET_VALID_LEVEL_57";
BA_ "GenSigStartValue" SG_ 344 CAN_DET_AZIMUTH_57 0;
BA_ "GenSigSendType" SG_ 344 CAN_DET_AZIMUTH_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_AZIMUTH_57 "CAN_DET_AZIMUTH_57";
BA_ "GenSigSendType" SG_ 344 CAN_DET_RANGE_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_RANGE_57 "CAN_DET_RANGE_57";
BA_ "GenSigStartValue" SG_ 344 CAN_DET_RANGE_RATE_57 0;
BA_ "GenSigSendType" SG_ 344 CAN_DET_RANGE_RATE_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_RANGE_RATE_57 "CAN_DET_RANGE_RATE_57";
BA_ "GenSigSendType" SG_ 344 CAN_DET_AMPLITUDE_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_DET_AMPLITUDE_57 "CAN_DET_AMPLITUDE_57";
BA_ "GenSigSendType" SG_ 344 CAN_SCAN_INDEX_2LSB_57 0;
BA_ "GenSigCmt" SG_ 344 CAN_SCAN_INDEX_2LSB_57 "CAN_SCAN_INDEX_2LSB_57";
BA_ "GenMsgSendType" BO_ 343 1;
BA_ "GenMsgILSupport" BO_ 343 1;
BA_ "GenMsgNrOfRepetition" BO_ 343 0;
BA_ "GenMsgCycleTime" BO_ 343 0;
BA_ "NetworkInitialization" BO_ 343 0;
BA_ "GenMsgDelayTime" BO_ 343 0;
BA_ "GenSigVtEn" SG_ 343 CAN_DET_CONFID_AZIMUTH_56 "CAN_DET_CONFID_AZIMUTH_56";
BA_ "GenSigVtName" SG_ 343 CAN_DET_CONFID_AZIMUTH_56 "CAN_DET_CONFID_AZIMUTH_56";
BA_ "GenSigSendType" SG_ 343 CAN_DET_CONFID_AZIMUTH_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_CONFID_AZIMUTH_56 "CAN_DET_CONFID_AZIMUTH_56";
BA_ "GenSigSendType" SG_ 343 CAN_DET_SUPER_RES_TARGET_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_SUPER_RES_TARGET_56 "CAN_DET_SUPER_RES_TARGET_56";
BA_ "GenSigSendType" SG_ 343 CAN_DET_ND_TARGET_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_ND_TARGET_56 "CAN_DET_ND_TARGET_56";
BA_ "GenSigSendType" SG_ 343 CAN_DET_HOST_VEH_CLUTTER_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_HOST_VEH_CLUTTER_56 "CAN_DET_HOST_VEH_CLUTTER_56";
BA_ "GenSigSendType" SG_ 343 CAN_DET_VALID_LEVEL_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_VALID_LEVEL_56 "CAN_DET_VALID_LEVEL_56";
BA_ "GenSigStartValue" SG_ 343 CAN_DET_AZIMUTH_56 0;
BA_ "GenSigSendType" SG_ 343 CAN_DET_AZIMUTH_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_AZIMUTH_56 "CAN_DET_AZIMUTH_56";
BA_ "GenSigSendType" SG_ 343 CAN_DET_RANGE_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_RANGE_56 "CAN_DET_RANGE_56";
BA_ "GenSigStartValue" SG_ 343 CAN_DET_RANGE_RATE_56 0;
BA_ "GenSigSendType" SG_ 343 CAN_DET_RANGE_RATE_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_RANGE_RATE_56 "CAN_DET_RANGE_RATE_56";
BA_ "GenSigSendType" SG_ 343 CAN_DET_AMPLITUDE_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_DET_AMPLITUDE_56 "CAN_DET_AMPLITUDE_56";
BA_ "GenSigSendType" SG_ 343 CAN_SCAN_INDEX_2LSB_56 0;
BA_ "GenSigCmt" SG_ 343 CAN_SCAN_INDEX_2LSB_56 "CAN_SCAN_INDEX_2LSB_56";
BA_ "GenMsgSendType" BO_ 342 1;
BA_ "GenMsgILSupport" BO_ 342 1;
BA_ "GenMsgNrOfRepetition" BO_ 342 0;
BA_ "GenMsgCycleTime" BO_ 342 0;
BA_ "NetworkInitialization" BO_ 342 0;
BA_ "GenMsgDelayTime" BO_ 342 0;
BA_ "GenSigVtEn" SG_ 342 CAN_DET_CONFID_AZIMUTH_55 "CAN_DET_CONFID_AZIMUTH_55";
BA_ "GenSigVtName" SG_ 342 CAN_DET_CONFID_AZIMUTH_55 "CAN_DET_CONFID_AZIMUTH_55";
BA_ "GenSigSendType" SG_ 342 CAN_DET_CONFID_AZIMUTH_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_CONFID_AZIMUTH_55 "CAN_DET_CONFID_AZIMUTH_55";
BA_ "GenSigSendType" SG_ 342 CAN_DET_SUPER_RES_TARGET_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_SUPER_RES_TARGET_55 "CAN_DET_SUPER_RES_TARGET_55";
BA_ "GenSigSendType" SG_ 342 CAN_DET_ND_TARGET_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_ND_TARGET_55 "CAN_DET_ND_TARGET_55";
BA_ "GenSigSendType" SG_ 342 CAN_DET_HOST_VEH_CLUTTER_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_HOST_VEH_CLUTTER_55 "CAN_DET_HOST_VEH_CLUTTER_55";
BA_ "GenSigSendType" SG_ 342 CAN_DET_VALID_LEVEL_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_VALID_LEVEL_55 "CAN_DET_VALID_LEVEL_55";
BA_ "GenSigStartValue" SG_ 342 CAN_DET_AZIMUTH_55 0;
BA_ "GenSigSendType" SG_ 342 CAN_DET_AZIMUTH_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_AZIMUTH_55 "CAN_DET_AZIMUTH_55";
BA_ "GenSigSendType" SG_ 342 CAN_DET_RANGE_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_RANGE_55 "CAN_DET_RANGE_55";
BA_ "GenSigStartValue" SG_ 342 CAN_DET_RANGE_RATE_55 0;
BA_ "GenSigSendType" SG_ 342 CAN_DET_RANGE_RATE_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_RANGE_RATE_55 "CAN_DET_RANGE_RATE_55";
BA_ "GenSigSendType" SG_ 342 CAN_DET_AMPLITUDE_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_DET_AMPLITUDE_55 "CAN_DET_AMPLITUDE_55";
BA_ "GenSigSendType" SG_ 342 CAN_SCAN_INDEX_2LSB_55 0;
BA_ "GenSigCmt" SG_ 342 CAN_SCAN_INDEX_2LSB_55 "CAN_SCAN_INDEX_2LSB_55";
BA_ "GenMsgSendType" BO_ 335 1;
BA_ "GenMsgILSupport" BO_ 335 1;
BA_ "GenMsgNrOfRepetition" BO_ 335 0;
BA_ "GenMsgCycleTime" BO_ 335 0;
BA_ "NetworkInitialization" BO_ 335 0;
BA_ "GenMsgDelayTime" BO_ 335 0;
BA_ "GenSigVtEn" SG_ 335 CAN_DET_CONFID_AZIMUTH_48 "CAN_DET_CONFID_AZIMUTH_48";
BA_ "GenSigVtName" SG_ 335 CAN_DET_CONFID_AZIMUTH_48 "CAN_DET_CONFID_AZIMUTH_48";
BA_ "GenSigSendType" SG_ 335 CAN_DET_CONFID_AZIMUTH_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_CONFID_AZIMUTH_48 "CAN_DET_CONFID_AZIMUTH_48";
BA_ "GenSigSendType" SG_ 335 CAN_DET_SUPER_RES_TARGET_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_SUPER_RES_TARGET_48 "CAN_DET_SUPER_RES_TARGET_48";
BA_ "GenSigSendType" SG_ 335 CAN_DET_ND_TARGET_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_ND_TARGET_48 "CAN_DET_ND_TARGET_48";
BA_ "GenSigSendType" SG_ 335 CAN_DET_HOST_VEH_CLUTTER_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_HOST_VEH_CLUTTER_48 "CAN_DET_HOST_VEH_CLUTTER_48";
BA_ "GenSigSendType" SG_ 335 CAN_DET_VALID_LEVEL_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_VALID_LEVEL_48 "CAN_DET_VALID_LEVEL_48";
BA_ "GenSigStartValue" SG_ 335 CAN_DET_AZIMUTH_48 0;
BA_ "GenSigSendType" SG_ 335 CAN_DET_AZIMUTH_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_AZIMUTH_48 "CAN_DET_AZIMUTH_48";
BA_ "GenSigSendType" SG_ 335 CAN_DET_RANGE_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_RANGE_48 "CAN_DET_RANGE_48";
BA_ "GenSigStartValue" SG_ 335 CAN_DET_RANGE_RATE_48 0;
BA_ "GenSigSendType" SG_ 335 CAN_DET_RANGE_RATE_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_RANGE_RATE_48 "CAN_DET_RANGE_RATE_48";
BA_ "GenSigSendType" SG_ 335 CAN_DET_AMPLITUDE_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_DET_AMPLITUDE_48 "CAN_DET_AMPLITUDE_48";
BA_ "GenSigSendType" SG_ 335 CAN_SCAN_INDEX_2LSB_48 0;
BA_ "GenSigCmt" SG_ 335 CAN_SCAN_INDEX_2LSB_48 "CAN_SCAN_INDEX_2LSB_48";
BA_ "GenMsgSendType" BO_ 334 1;
BA_ "GenMsgILSupport" BO_ 334 1;
BA_ "GenMsgNrOfRepetition" BO_ 334 0;
BA_ "GenMsgCycleTime" BO_ 334 0;
BA_ "NetworkInitialization" BO_ 334 0;
BA_ "GenMsgDelayTime" BO_ 334 0;
BA_ "GenSigVtEn" SG_ 334 CAN_DET_CONFID_AZIMUTH_47 "CAN_DET_CONFID_AZIMUTH_47";
BA_ "GenSigVtName" SG_ 334 CAN_DET_CONFID_AZIMUTH_47 "CAN_DET_CONFID_AZIMUTH_47";
BA_ "GenSigSendType" SG_ 334 CAN_DET_CONFID_AZIMUTH_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_CONFID_AZIMUTH_47 "CAN_DET_CONFID_AZIMUTH_47";
BA_ "GenSigSendType" SG_ 334 CAN_DET_SUPER_RES_TARGET_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_SUPER_RES_TARGET_47 "CAN_DET_SUPER_RES_TARGET_47";
BA_ "GenSigSendType" SG_ 334 CAN_DET_ND_TARGET_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_ND_TARGET_47 "CAN_DET_ND_TARGET_47";
BA_ "GenSigSendType" SG_ 334 CAN_DET_HOST_VEH_CLUTTER_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_HOST_VEH_CLUTTER_47 "CAN_DET_HOST_VEH_CLUTTER_47";
BA_ "GenSigSendType" SG_ 334 CAN_DET_VALID_LEVEL_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_VALID_LEVEL_47 "CAN_DET_VALID_LEVEL_47";
BA_ "GenSigStartValue" SG_ 334 CAN_DET_AZIMUTH_47 0;
BA_ "GenSigSendType" SG_ 334 CAN_DET_AZIMUTH_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_AZIMUTH_47 "CAN_DET_AZIMUTH_47";
BA_ "GenSigSendType" SG_ 334 CAN_DET_RANGE_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_RANGE_47 "CAN_DET_RANGE_47";
BA_ "GenSigStartValue" SG_ 334 CAN_DET_RANGE_RATE_47 0;
BA_ "GenSigSendType" SG_ 334 CAN_DET_RANGE_RATE_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_RANGE_RATE_47 "CAN_DET_RANGE_RATE_47";
BA_ "GenSigSendType" SG_ 334 CAN_DET_AMPLITUDE_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_DET_AMPLITUDE_47 "CAN_DET_AMPLITUDE_47";
BA_ "GenSigSendType" SG_ 334 CAN_SCAN_INDEX_2LSB_47 0;
BA_ "GenSigCmt" SG_ 334 CAN_SCAN_INDEX_2LSB_47 "CAN_SCAN_INDEX_2LSB_47";
BA_ "GenMsgSendType" BO_ 333 1;
BA_ "GenMsgILSupport" BO_ 333 1;
BA_ "GenMsgNrOfRepetition" BO_ 333 0;
BA_ "GenMsgCycleTime" BO_ 333 0;
BA_ "NetworkInitialization" BO_ 333 0;
BA_ "GenMsgDelayTime" BO_ 333 0;
BA_ "GenSigVtEn" SG_ 333 CAN_DET_CONFID_AZIMUTH_46 "CAN_DET_CONFID_AZIMUTH_46";
BA_ "GenSigVtName" SG_ 333 CAN_DET_CONFID_AZIMUTH_46 "CAN_DET_CONFID_AZIMUTH_46";
BA_ "GenSigSendType" SG_ 333 CAN_DET_CONFID_AZIMUTH_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_CONFID_AZIMUTH_46 "CAN_DET_CONFID_AZIMUTH_46";
BA_ "GenSigSendType" SG_ 333 CAN_DET_SUPER_RES_TARGET_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_SUPER_RES_TARGET_46 "CAN_DET_SUPER_RES_TARGET_46";
BA_ "GenSigSendType" SG_ 333 CAN_DET_ND_TARGET_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_ND_TARGET_46 "CAN_DET_ND_TARGET_46";
BA_ "GenSigSendType" SG_ 333 CAN_DET_HOST_VEH_CLUTTER_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_HOST_VEH_CLUTTER_46 "CAN_DET_HOST_VEH_CLUTTER_46";
BA_ "GenSigSendType" SG_ 333 CAN_DET_VALID_LEVEL_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_VALID_LEVEL_46 "CAN_DET_VALID_LEVEL_46";
BA_ "GenSigStartValue" SG_ 333 CAN_DET_AZIMUTH_46 0;
BA_ "GenSigSendType" SG_ 333 CAN_DET_AZIMUTH_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_AZIMUTH_46 "CAN_DET_AZIMUTH_46";
BA_ "GenSigSendType" SG_ 333 CAN_DET_RANGE_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_RANGE_46 "CAN_DET_RANGE_46";
BA_ "GenSigStartValue" SG_ 333 CAN_DET_RANGE_RATE_46 0;
BA_ "GenSigSendType" SG_ 333 CAN_DET_RANGE_RATE_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_RANGE_RATE_46 "CAN_DET_RANGE_RATE_46";
BA_ "GenSigSendType" SG_ 333 CAN_DET_AMPLITUDE_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_DET_AMPLITUDE_46 "CAN_DET_AMPLITUDE_46";
BA_ "GenSigSendType" SG_ 333 CAN_SCAN_INDEX_2LSB_46 0;
BA_ "GenSigCmt" SG_ 333 CAN_SCAN_INDEX_2LSB_46 "CAN_SCAN_INDEX_2LSB_46";
BA_ "GenMsgSendType" BO_ 332 1;
BA_ "GenMsgILSupport" BO_ 332 1;
BA_ "GenMsgNrOfRepetition" BO_ 332 0;
BA_ "GenMsgCycleTime" BO_ 332 0;
BA_ "NetworkInitialization" BO_ 332 0;
BA_ "GenMsgDelayTime" BO_ 332 0;
BA_ "GenSigVtEn" SG_ 332 CAN_DET_CONFID_AZIMUTH_45 "CAN_DET_CONFID_AZIMUTH_45";
BA_ "GenSigVtName" SG_ 332 CAN_DET_CONFID_AZIMUTH_45 "CAN_DET_CONFID_AZIMUTH_45";
BA_ "GenSigSendType" SG_ 332 CAN_DET_CONFID_AZIMUTH_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_CONFID_AZIMUTH_45 "CAN_DET_CONFID_AZIMUTH_45";
BA_ "GenSigSendType" SG_ 332 CAN_DET_SUPER_RES_TARGET_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_SUPER_RES_TARGET_45 "CAN_DET_SUPER_RES_TARGET_45";
BA_ "GenSigSendType" SG_ 332 CAN_DET_ND_TARGET_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_ND_TARGET_45 "CAN_DET_ND_TARGET_45";
BA_ "GenSigSendType" SG_ 332 CAN_DET_HOST_VEH_CLUTTER_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_HOST_VEH_CLUTTER_45 "CAN_DET_HOST_VEH_CLUTTER_45";
BA_ "GenSigSendType" SG_ 332 CAN_DET_VALID_LEVEL_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_VALID_LEVEL_45 "CAN_DET_VALID_LEVEL_45";
BA_ "GenSigStartValue" SG_ 332 CAN_DET_AZIMUTH_45 0;
BA_ "GenSigSendType" SG_ 332 CAN_DET_AZIMUTH_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_AZIMUTH_45 "CAN_DET_AZIMUTH_45";
BA_ "GenSigSendType" SG_ 332 CAN_DET_RANGE_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_RANGE_45 "CAN_DET_RANGE_45";
BA_ "GenSigStartValue" SG_ 332 CAN_DET_RANGE_RATE_45 0;
BA_ "GenSigSendType" SG_ 332 CAN_DET_RANGE_RATE_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_RANGE_RATE_45 "CAN_DET_RANGE_RATE_45";
BA_ "GenSigSendType" SG_ 332 CAN_DET_AMPLITUDE_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_DET_AMPLITUDE_45 "CAN_DET_AMPLITUDE_45";
BA_ "GenSigSendType" SG_ 332 CAN_SCAN_INDEX_2LSB_45 0;
BA_ "GenSigCmt" SG_ 332 CAN_SCAN_INDEX_2LSB_45 "CAN_SCAN_INDEX_2LSB_45";
BA_ "GenMsgSendType" BO_ 331 1;
BA_ "GenMsgILSupport" BO_ 331 1;
BA_ "GenMsgNrOfRepetition" BO_ 331 0;
BA_ "GenMsgCycleTime" BO_ 331 0;
BA_ "NetworkInitialization" BO_ 331 0;
BA_ "GenMsgDelayTime" BO_ 331 0;
BA_ "GenSigVtEn" SG_ 331 CAN_DET_CONFID_AZIMUTH_44 "CAN_DET_CONFID_AZIMUTH_44";
BA_ "GenSigVtName" SG_ 331 CAN_DET_CONFID_AZIMUTH_44 "CAN_DET_CONFID_AZIMUTH_44";
BA_ "GenSigSendType" SG_ 331 CAN_DET_CONFID_AZIMUTH_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_CONFID_AZIMUTH_44 "CAN_DET_CONFID_AZIMUTH_44";
BA_ "GenSigSendType" SG_ 331 CAN_DET_SUPER_RES_TARGET_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_SUPER_RES_TARGET_44 "CAN_DET_SUPER_RES_TARGET_44";
BA_ "GenSigSendType" SG_ 331 CAN_DET_ND_TARGET_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_ND_TARGET_44 "CAN_DET_ND_TARGET_44";
BA_ "GenSigSendType" SG_ 331 CAN_DET_HOST_VEH_CLUTTER_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_HOST_VEH_CLUTTER_44 "CAN_DET_HOST_VEH_CLUTTER_44";
BA_ "GenSigSendType" SG_ 331 CAN_DET_VALID_LEVEL_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_VALID_LEVEL_44 "CAN_DET_VALID_LEVEL_44";
BA_ "GenSigStartValue" SG_ 331 CAN_DET_AZIMUTH_44 0;
BA_ "GenSigSendType" SG_ 331 CAN_DET_AZIMUTH_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_AZIMUTH_44 "CAN_DET_AZIMUTH_44";
BA_ "GenSigSendType" SG_ 331 CAN_DET_RANGE_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_RANGE_44 "CAN_DET_RANGE_44";
BA_ "GenSigStartValue" SG_ 331 CAN_DET_RANGE_RATE_44 0;
BA_ "GenSigSendType" SG_ 331 CAN_DET_RANGE_RATE_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_RANGE_RATE_44 "CAN_DET_RANGE_RATE_44";
BA_ "GenSigSendType" SG_ 331 CAN_DET_AMPLITUDE_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_DET_AMPLITUDE_44 "CAN_DET_AMPLITUDE_44";
BA_ "GenSigSendType" SG_ 331 CAN_SCAN_INDEX_2LSB_44 0;
BA_ "GenSigCmt" SG_ 331 CAN_SCAN_INDEX_2LSB_44 "CAN_SCAN_INDEX_2LSB_44";
BA_ "GenMsgSendType" BO_ 330 1;
BA_ "GenMsgILSupport" BO_ 330 1;
BA_ "GenMsgNrOfRepetition" BO_ 330 0;
BA_ "GenMsgCycleTime" BO_ 330 0;
BA_ "NetworkInitialization" BO_ 330 0;
BA_ "GenMsgDelayTime" BO_ 330 0;
BA_ "GenSigVtEn" SG_ 330 CAN_DET_CONFID_AZIMUTH_43 "CAN_DET_CONFID_AZIMUTH_43";
BA_ "GenSigVtName" SG_ 330 CAN_DET_CONFID_AZIMUTH_43 "CAN_DET_CONFID_AZIMUTH_43";
BA_ "GenSigSendType" SG_ 330 CAN_DET_CONFID_AZIMUTH_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_CONFID_AZIMUTH_43 "CAN_DET_CONFID_AZIMUTH_43";
BA_ "GenSigSendType" SG_ 330 CAN_DET_SUPER_RES_TARGET_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_SUPER_RES_TARGET_43 "CAN_DET_SUPER_RES_TARGET_43";
BA_ "GenSigSendType" SG_ 330 CAN_DET_ND_TARGET_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_ND_TARGET_43 "CAN_DET_ND_TARGET_43";
BA_ "GenSigSendType" SG_ 330 CAN_DET_HOST_VEH_CLUTTER_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_HOST_VEH_CLUTTER_43 "CAN_DET_HOST_VEH_CLUTTER_43";
BA_ "GenSigSendType" SG_ 330 CAN_DET_VALID_LEVEL_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_VALID_LEVEL_43 "CAN_DET_VALID_LEVEL_43";
BA_ "GenSigStartValue" SG_ 330 CAN_DET_AZIMUTH_43 0;
BA_ "GenSigSendType" SG_ 330 CAN_DET_AZIMUTH_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_AZIMUTH_43 "CAN_DET_AZIMUTH_43";
BA_ "GenSigSendType" SG_ 330 CAN_DET_RANGE_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_RANGE_43 "CAN_DET_RANGE_43";
BA_ "GenSigStartValue" SG_ 330 CAN_DET_RANGE_RATE_43 0;
BA_ "GenSigSendType" SG_ 330 CAN_DET_RANGE_RATE_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_RANGE_RATE_43 "CAN_DET_RANGE_RATE_43";
BA_ "GenSigSendType" SG_ 330 CAN_DET_AMPLITUDE_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_DET_AMPLITUDE_43 "CAN_DET_AMPLITUDE_43";
BA_ "GenSigSendType" SG_ 330 CAN_SCAN_INDEX_2LSB_43 0;
BA_ "GenSigCmt" SG_ 330 CAN_SCAN_INDEX_2LSB_43 "CAN_SCAN_INDEX_2LSB_43";
BA_ "GenMsgSendType" BO_ 329 1;
BA_ "GenMsgILSupport" BO_ 329 1;
BA_ "GenMsgNrOfRepetition" BO_ 329 0;
BA_ "GenMsgCycleTime" BO_ 329 0;
BA_ "NetworkInitialization" BO_ 329 0;
BA_ "GenMsgDelayTime" BO_ 329 0;
BA_ "GenSigVtEn" SG_ 329 CAN_DET_CONFID_AZIMUTH_42 "CAN_DET_CONFID_AZIMUTH_42";
BA_ "GenSigVtName" SG_ 329 CAN_DET_CONFID_AZIMUTH_42 "CAN_DET_CONFID_AZIMUTH_42";
BA_ "GenSigSendType" SG_ 329 CAN_DET_CONFID_AZIMUTH_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_CONFID_AZIMUTH_42 "CAN_DET_CONFID_AZIMUTH_42";
BA_ "GenSigSendType" SG_ 329 CAN_DET_SUPER_RES_TARGET_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_SUPER_RES_TARGET_42 "CAN_DET_SUPER_RES_TARGET_42";
BA_ "GenSigSendType" SG_ 329 CAN_DET_ND_TARGET_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_ND_TARGET_42 "CAN_DET_ND_TARGET_42";
BA_ "GenSigSendType" SG_ 329 CAN_DET_HOST_VEH_CLUTTER_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_HOST_VEH_CLUTTER_42 "CAN_DET_HOST_VEH_CLUTTER_42";
BA_ "GenSigSendType" SG_ 329 CAN_DET_VALID_LEVEL_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_VALID_LEVEL_42 "CAN_DET_VALID_LEVEL_42";
BA_ "GenSigStartValue" SG_ 329 CAN_DET_AZIMUTH_42 0;
BA_ "GenSigSendType" SG_ 329 CAN_DET_AZIMUTH_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_AZIMUTH_42 "CAN_DET_AZIMUTH_42";
BA_ "GenSigSendType" SG_ 329 CAN_DET_RANGE_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_RANGE_42 "CAN_DET_RANGE_42";
BA_ "GenSigStartValue" SG_ 329 CAN_DET_RANGE_RATE_42 0;
BA_ "GenSigSendType" SG_ 329 CAN_DET_RANGE_RATE_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_RANGE_RATE_42 "CAN_DET_RANGE_RATE_42";
BA_ "GenSigSendType" SG_ 329 CAN_DET_AMPLITUDE_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_DET_AMPLITUDE_42 "CAN_DET_AMPLITUDE_42";
BA_ "GenSigSendType" SG_ 329 CAN_SCAN_INDEX_2LSB_42 0;
BA_ "GenSigCmt" SG_ 329 CAN_SCAN_INDEX_2LSB_42 "CAN_SCAN_INDEX_2LSB_42";
BA_ "GenMsgSendType" BO_ 328 1;
BA_ "GenMsgILSupport" BO_ 328 1;
BA_ "GenMsgNrOfRepetition" BO_ 328 0;
BA_ "GenMsgCycleTime" BO_ 328 0;
BA_ "NetworkInitialization" BO_ 328 0;
BA_ "GenMsgDelayTime" BO_ 328 0;
BA_ "GenSigVtEn" SG_ 328 CAN_DET_CONFID_AZIMUTH_41 "CAN_DET_CONFID_AZIMUTH_41";
BA_ "GenSigVtName" SG_ 328 CAN_DET_CONFID_AZIMUTH_41 "CAN_DET_CONFID_AZIMUTH_41";
BA_ "GenSigSendType" SG_ 328 CAN_DET_CONFID_AZIMUTH_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_CONFID_AZIMUTH_41 "CAN_DET_CONFID_AZIMUTH_41";
BA_ "GenSigSendType" SG_ 328 CAN_DET_SUPER_RES_TARGET_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_SUPER_RES_TARGET_41 "CAN_DET_SUPER_RES_TARGET_41";
BA_ "GenSigSendType" SG_ 328 CAN_DET_ND_TARGET_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_ND_TARGET_41 "CAN_DET_ND_TARGET_41";
BA_ "GenSigSendType" SG_ 328 CAN_DET_HOST_VEH_CLUTTER_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_HOST_VEH_CLUTTER_41 "CAN_DET_HOST_VEH_CLUTTER_41";
BA_ "GenSigSendType" SG_ 328 CAN_DET_VALID_LEVEL_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_VALID_LEVEL_41 "CAN_DET_VALID_LEVEL_41";
BA_ "GenSigStartValue" SG_ 328 CAN_DET_AZIMUTH_41 0;
BA_ "GenSigSendType" SG_ 328 CAN_DET_AZIMUTH_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_AZIMUTH_41 "CAN_DET_AZIMUTH_41";
BA_ "GenSigSendType" SG_ 328 CAN_DET_RANGE_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_RANGE_41 "CAN_DET_RANGE_41";
BA_ "GenSigStartValue" SG_ 328 CAN_DET_RANGE_RATE_41 0;
BA_ "GenSigSendType" SG_ 328 CAN_DET_RANGE_RATE_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_RANGE_RATE_41 "CAN_DET_RANGE_RATE_41";
BA_ "GenSigSendType" SG_ 328 CAN_DET_AMPLITUDE_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_DET_AMPLITUDE_41 "CAN_DET_AMPLITUDE_41";
BA_ "GenSigSendType" SG_ 328 CAN_SCAN_INDEX_2LSB_41 0;
BA_ "GenSigCmt" SG_ 328 CAN_SCAN_INDEX_2LSB_41 "CAN_SCAN_INDEX_2LSB_41";
BA_ "GenMsgSendType" BO_ 327 1;
BA_ "GenMsgILSupport" BO_ 327 1;
BA_ "GenMsgNrOfRepetition" BO_ 327 0;
BA_ "GenMsgCycleTime" BO_ 327 0;
BA_ "NetworkInitialization" BO_ 327 0;
BA_ "GenMsgDelayTime" BO_ 327 0;
BA_ "GenSigVtEn" SG_ 327 CAN_DET_CONFID_AZIMUTH_40 "CAN_DET_CONFID_AZIMUTH_40";
BA_ "GenSigVtName" SG_ 327 CAN_DET_CONFID_AZIMUTH_40 "CAN_DET_CONFID_AZIMUTH_40";
BA_ "GenSigSendType" SG_ 327 CAN_DET_CONFID_AZIMUTH_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_CONFID_AZIMUTH_40 "CAN_DET_CONFID_AZIMUTH_40";
BA_ "GenSigSendType" SG_ 327 CAN_DET_SUPER_RES_TARGET_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_SUPER_RES_TARGET_40 "CAN_DET_SUPER_RES_TARGET_40";
BA_ "GenSigSendType" SG_ 327 CAN_DET_ND_TARGET_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_ND_TARGET_40 "CAN_DET_ND_TARGET_40";
BA_ "GenSigSendType" SG_ 327 CAN_DET_HOST_VEH_CLUTTER_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_HOST_VEH_CLUTTER_40 "CAN_DET_HOST_VEH_CLUTTER_40";
BA_ "GenSigSendType" SG_ 327 CAN_DET_VALID_LEVEL_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_VALID_LEVEL_40 "CAN_DET_VALID_LEVEL_40";
BA_ "GenSigStartValue" SG_ 327 CAN_DET_AZIMUTH_40 0;
BA_ "GenSigSendType" SG_ 327 CAN_DET_AZIMUTH_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_AZIMUTH_40 "CAN_DET_AZIMUTH_40";
BA_ "GenSigSendType" SG_ 327 CAN_DET_RANGE_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_RANGE_40 "CAN_DET_RANGE_40";
BA_ "GenSigStartValue" SG_ 327 CAN_DET_RANGE_RATE_40 0;
BA_ "GenSigSendType" SG_ 327 CAN_DET_RANGE_RATE_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_RANGE_RATE_40 "CAN_DET_RANGE_RATE_40";
BA_ "GenSigSendType" SG_ 327 CAN_DET_AMPLITUDE_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_DET_AMPLITUDE_40 "CAN_DET_AMPLITUDE_40";
BA_ "GenSigSendType" SG_ 327 CAN_SCAN_INDEX_2LSB_40 0;
BA_ "GenSigCmt" SG_ 327 CAN_SCAN_INDEX_2LSB_40 "CAN_SCAN_INDEX_2LSB_40";
BA_ "GenMsgSendType" BO_ 325 1;
BA_ "GenMsgILSupport" BO_ 325 1;
BA_ "GenMsgNrOfRepetition" BO_ 325 0;
BA_ "GenMsgCycleTime" BO_ 325 0;
BA_ "NetworkInitialization" BO_ 325 0;
BA_ "GenMsgDelayTime" BO_ 325 0;
BA_ "GenSigVtEn" SG_ 325 CAN_DET_CONFID_AZIMUTH_38 "CAN_DET_CONFID_AZIMUTH_38";
BA_ "GenSigVtName" SG_ 325 CAN_DET_CONFID_AZIMUTH_38 "CAN_DET_CONFID_AZIMUTH_38";
BA_ "GenSigSendType" SG_ 325 CAN_DET_CONFID_AZIMUTH_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_CONFID_AZIMUTH_38 "CAN_DET_CONFID_AZIMUTH_38";
BA_ "GenSigSendType" SG_ 325 CAN_DET_SUPER_RES_TARGET_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_SUPER_RES_TARGET_38 "CAN_DET_SUPER_RES_TARGET_38";
BA_ "GenSigSendType" SG_ 325 CAN_DET_ND_TARGET_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_ND_TARGET_38 "CAN_DET_ND_TARGET_38";
BA_ "GenSigSendType" SG_ 325 CAN_DET_HOST_VEH_CLUTTER_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_HOST_VEH_CLUTTER_38 "CAN_DET_HOST_VEH_CLUTTER_38";
BA_ "GenSigSendType" SG_ 325 CAN_DET_VALID_LEVEL_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_VALID_LEVEL_38 "CAN_DET_VALID_LEVEL_38";
BA_ "GenSigStartValue" SG_ 325 CAN_DET_AZIMUTH_38 0;
BA_ "GenSigSendType" SG_ 325 CAN_DET_AZIMUTH_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_AZIMUTH_38 "CAN_DET_AZIMUTH_38";
BA_ "GenSigSendType" SG_ 325 CAN_DET_RANGE_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_RANGE_38 "CAN_DET_RANGE_38";
BA_ "GenSigStartValue" SG_ 325 CAN_DET_RANGE_RATE_38 0;
BA_ "GenSigSendType" SG_ 325 CAN_DET_RANGE_RATE_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_RANGE_RATE_38 "CAN_DET_RANGE_RATE_38";
BA_ "GenSigSendType" SG_ 325 CAN_DET_AMPLITUDE_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_DET_AMPLITUDE_38 "CAN_DET_AMPLITUDE_38";
BA_ "GenSigSendType" SG_ 325 CAN_SCAN_INDEX_2LSB_38 0;
BA_ "GenSigCmt" SG_ 325 CAN_SCAN_INDEX_2LSB_38 "CAN_SCAN_INDEX_2LSB_38";
BA_ "GenMsgSendType" BO_ 324 1;
BA_ "GenMsgILSupport" BO_ 324 1;
BA_ "GenMsgNrOfRepetition" BO_ 324 0;
BA_ "GenMsgCycleTime" BO_ 324 0;
BA_ "NetworkInitialization" BO_ 324 0;
BA_ "GenMsgDelayTime" BO_ 324 0;
BA_ "GenSigVtEn" SG_ 324 CAN_DET_CONFID_AZIMUTH_37 "CAN_DET_CONFID_AZIMUTH_37";
BA_ "GenSigVtName" SG_ 324 CAN_DET_CONFID_AZIMUTH_37 "CAN_DET_CONFID_AZIMUTH_37";
BA_ "GenSigSendType" SG_ 324 CAN_DET_CONFID_AZIMUTH_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_CONFID_AZIMUTH_37 "CAN_DET_CONFID_AZIMUTH_37";
BA_ "GenSigSendType" SG_ 324 CAN_DET_SUPER_RES_TARGET_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_SUPER_RES_TARGET_37 "CAN_DET_SUPER_RES_TARGET_37";
BA_ "GenSigSendType" SG_ 324 CAN_DET_ND_TARGET_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_ND_TARGET_37 "CAN_DET_ND_TARGET_37";
BA_ "GenSigSendType" SG_ 324 CAN_DET_HOST_VEH_CLUTTER_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_HOST_VEH_CLUTTER_37 "CAN_DET_HOST_VEH_CLUTTER_37";
BA_ "GenSigSendType" SG_ 324 CAN_DET_VALID_LEVEL_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_VALID_LEVEL_37 "CAN_DET_VALID_LEVEL_37";
BA_ "GenSigStartValue" SG_ 324 CAN_DET_AZIMUTH_37 0;
BA_ "GenSigSendType" SG_ 324 CAN_DET_AZIMUTH_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_AZIMUTH_37 "CAN_DET_AZIMUTH_37";
BA_ "GenSigSendType" SG_ 324 CAN_DET_RANGE_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_RANGE_37 "CAN_DET_RANGE_37";
BA_ "GenSigStartValue" SG_ 324 CAN_DET_RANGE_RATE_37 0;
BA_ "GenSigSendType" SG_ 324 CAN_DET_RANGE_RATE_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_RANGE_RATE_37 "CAN_DET_RANGE_RATE_37";
BA_ "GenSigSendType" SG_ 324 CAN_DET_AMPLITUDE_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_DET_AMPLITUDE_37 "CAN_DET_AMPLITUDE_37";
BA_ "GenSigSendType" SG_ 324 CAN_SCAN_INDEX_2LSB_37 0;
BA_ "GenSigCmt" SG_ 324 CAN_SCAN_INDEX_2LSB_37 "CAN_SCAN_INDEX_2LSB_37";
BA_ "GenMsgSendType" BO_ 323 1;
BA_ "GenMsgILSupport" BO_ 323 1;
BA_ "GenMsgNrOfRepetition" BO_ 323 0;
BA_ "GenMsgCycleTime" BO_ 323 0;
BA_ "NetworkInitialization" BO_ 323 0;
BA_ "GenMsgDelayTime" BO_ 323 0;
BA_ "GenSigVtEn" SG_ 323 CAN_DET_CONFID_AZIMUTH_36 "CAN_DET_CONFID_AZIMUTH_36";
BA_ "GenSigVtName" SG_ 323 CAN_DET_CONFID_AZIMUTH_36 "CAN_DET_CONFID_AZIMUTH_36";
BA_ "GenSigSendType" SG_ 323 CAN_DET_CONFID_AZIMUTH_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_CONFID_AZIMUTH_36 "CAN_DET_CONFID_AZIMUTH_36";
BA_ "GenSigSendType" SG_ 323 CAN_DET_SUPER_RES_TARGET_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_SUPER_RES_TARGET_36 "CAN_DET_SUPER_RES_TARGET_36";
BA_ "GenSigSendType" SG_ 323 CAN_DET_ND_TARGET_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_ND_TARGET_36 "CAN_DET_ND_TARGET_36";
BA_ "GenSigSendType" SG_ 323 CAN_DET_HOST_VEH_CLUTTER_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_HOST_VEH_CLUTTER_36 "CAN_DET_HOST_VEH_CLUTTER_36";
BA_ "GenSigSendType" SG_ 323 CAN_DET_VALID_LEVEL_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_VALID_LEVEL_36 "CAN_DET_VALID_LEVEL_36";
BA_ "GenSigStartValue" SG_ 323 CAN_DET_AZIMUTH_36 0;
BA_ "GenSigSendType" SG_ 323 CAN_DET_AZIMUTH_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_AZIMUTH_36 "CAN_DET_AZIMUTH_36";
BA_ "GenSigSendType" SG_ 323 CAN_DET_RANGE_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_RANGE_36 "CAN_DET_RANGE_36";
BA_ "GenSigStartValue" SG_ 323 CAN_DET_RANGE_RATE_36 0;
BA_ "GenSigSendType" SG_ 323 CAN_DET_RANGE_RATE_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_RANGE_RATE_36 "CAN_DET_RANGE_RATE_36";
BA_ "GenSigSendType" SG_ 323 CAN_DET_AMPLITUDE_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_DET_AMPLITUDE_36 "CAN_DET_AMPLITUDE_36";
BA_ "GenSigSendType" SG_ 323 CAN_SCAN_INDEX_2LSB_36 0;
BA_ "GenSigCmt" SG_ 323 CAN_SCAN_INDEX_2LSB_36 "CAN_SCAN_INDEX_2LSB_36";
BA_ "GenMsgSendType" BO_ 322 1;
BA_ "GenMsgILSupport" BO_ 322 1;
BA_ "GenMsgNrOfRepetition" BO_ 322 0;
BA_ "GenMsgCycleTime" BO_ 322 0;
BA_ "NetworkInitialization" BO_ 322 0;
BA_ "GenMsgDelayTime" BO_ 322 0;
BA_ "GenSigVtEn" SG_ 322 CAN_DET_CONFID_AZIMUTH_35 "CAN_DET_CONFID_AZIMUTH_35";
BA_ "GenSigVtName" SG_ 322 CAN_DET_CONFID_AZIMUTH_35 "CAN_DET_CONFID_AZIMUTH_35";
BA_ "GenSigSendType" SG_ 322 CAN_DET_CONFID_AZIMUTH_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_CONFID_AZIMUTH_35 "CAN_DET_CONFID_AZIMUTH_35";
BA_ "GenSigSendType" SG_ 322 CAN_DET_SUPER_RES_TARGET_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_SUPER_RES_TARGET_35 "CAN_DET_SUPER_RES_TARGET_35";
BA_ "GenSigSendType" SG_ 322 CAN_DET_ND_TARGET_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_ND_TARGET_35 "CAN_DET_ND_TARGET_35";
BA_ "GenSigSendType" SG_ 322 CAN_DET_HOST_VEH_CLUTTER_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_HOST_VEH_CLUTTER_35 "CAN_DET_HOST_VEH_CLUTTER_35";
BA_ "GenSigSendType" SG_ 322 CAN_DET_VALID_LEVEL_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_VALID_LEVEL_35 "CAN_DET_VALID_LEVEL_35";
BA_ "GenSigStartValue" SG_ 322 CAN_DET_AZIMUTH_35 0;
BA_ "GenSigSendType" SG_ 322 CAN_DET_AZIMUTH_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_AZIMUTH_35 "CAN_DET_AZIMUTH_35";
BA_ "GenSigSendType" SG_ 322 CAN_DET_RANGE_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_RANGE_35 "CAN_DET_RANGE_35";
BA_ "GenSigStartValue" SG_ 322 CAN_DET_RANGE_RATE_35 0;
BA_ "GenSigSendType" SG_ 322 CAN_DET_RANGE_RATE_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_RANGE_RATE_35 "CAN_DET_RANGE_RATE_35";
BA_ "GenSigSendType" SG_ 322 CAN_DET_AMPLITUDE_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_DET_AMPLITUDE_35 "CAN_DET_AMPLITUDE_35";
BA_ "GenSigSendType" SG_ 322 CAN_SCAN_INDEX_2LSB_35 0;
BA_ "GenSigCmt" SG_ 322 CAN_SCAN_INDEX_2LSB_35 "CAN_SCAN_INDEX_2LSB_35";
BA_ "GenMsgSendType" BO_ 321 1;
BA_ "GenMsgILSupport" BO_ 321 1;
BA_ "GenMsgNrOfRepetition" BO_ 321 0;
BA_ "GenMsgCycleTime" BO_ 321 0;
BA_ "NetworkInitialization" BO_ 321 0;
BA_ "GenMsgDelayTime" BO_ 321 0;
BA_ "GenSigVtEn" SG_ 321 CAN_DET_CONFID_AZIMUTH_34 "CAN_DET_CONFID_AZIMUTH_34";
BA_ "GenSigVtName" SG_ 321 CAN_DET_CONFID_AZIMUTH_34 "CAN_DET_CONFID_AZIMUTH_34";
BA_ "GenSigSendType" SG_ 321 CAN_DET_CONFID_AZIMUTH_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_CONFID_AZIMUTH_34 "CAN_DET_CONFID_AZIMUTH_34";
BA_ "GenSigSendType" SG_ 321 CAN_DET_SUPER_RES_TARGET_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_SUPER_RES_TARGET_34 "CAN_DET_SUPER_RES_TARGET_34";
BA_ "GenSigSendType" SG_ 321 CAN_DET_ND_TARGET_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_ND_TARGET_34 "CAN_DET_ND_TARGET_34";
BA_ "GenSigSendType" SG_ 321 CAN_DET_HOST_VEH_CLUTTER_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_HOST_VEH_CLUTTER_34 "CAN_DET_HOST_VEH_CLUTTER_34";
BA_ "GenSigSendType" SG_ 321 CAN_DET_VALID_LEVEL_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_VALID_LEVEL_34 "CAN_DET_VALID_LEVEL_34";
BA_ "GenSigStartValue" SG_ 321 CAN_DET_AZIMUTH_34 0;
BA_ "GenSigSendType" SG_ 321 CAN_DET_AZIMUTH_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_AZIMUTH_34 "CAN_DET_AZIMUTH_34";
BA_ "GenSigSendType" SG_ 321 CAN_DET_RANGE_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_RANGE_34 "CAN_DET_RANGE_34";
BA_ "GenSigStartValue" SG_ 321 CAN_DET_RANGE_RATE_34 0;
BA_ "GenSigSendType" SG_ 321 CAN_DET_RANGE_RATE_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_RANGE_RATE_34 "CAN_DET_RANGE_RATE_34";
BA_ "GenSigSendType" SG_ 321 CAN_DET_AMPLITUDE_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_DET_AMPLITUDE_34 "CAN_DET_AMPLITUDE_34";
BA_ "GenSigSendType" SG_ 321 CAN_SCAN_INDEX_2LSB_34 0;
BA_ "GenSigCmt" SG_ 321 CAN_SCAN_INDEX_2LSB_34 "CAN_SCAN_INDEX_2LSB_34";
BA_ "GenMsgSendType" BO_ 320 1;
BA_ "GenMsgILSupport" BO_ 320 1;
BA_ "GenMsgNrOfRepetition" BO_ 320 0;
BA_ "GenMsgCycleTime" BO_ 320 0;
BA_ "NetworkInitialization" BO_ 320 0;
BA_ "GenMsgDelayTime" BO_ 320 0;
BA_ "GenSigVtEn" SG_ 320 CAN_DET_CONFID_AZIMUTH_33 "CAN_DET_CONFID_AZIMUTH_33";
BA_ "GenSigVtName" SG_ 320 CAN_DET_CONFID_AZIMUTH_33 "CAN_DET_CONFID_AZIMUTH_33";
BA_ "GenSigSendType" SG_ 320 CAN_DET_CONFID_AZIMUTH_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_CONFID_AZIMUTH_33 "CAN_DET_CONFID_AZIMUTH_33";
BA_ "GenSigSendType" SG_ 320 CAN_DET_SUPER_RES_TARGET_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_SUPER_RES_TARGET_33 "CAN_DET_SUPER_RES_TARGET_33";
BA_ "GenSigSendType" SG_ 320 CAN_DET_ND_TARGET_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_ND_TARGET_33 "CAN_DET_ND_TARGET_33";
BA_ "GenSigSendType" SG_ 320 CAN_DET_HOST_VEH_CLUTTER_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_HOST_VEH_CLUTTER_33 "CAN_DET_HOST_VEH_CLUTTER_33";
BA_ "GenSigSendType" SG_ 320 CAN_DET_VALID_LEVEL_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_VALID_LEVEL_33 "CAN_DET_VALID_LEVEL_33";
BA_ "GenSigStartValue" SG_ 320 CAN_DET_AZIMUTH_33 0;
BA_ "GenSigSendType" SG_ 320 CAN_DET_AZIMUTH_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_AZIMUTH_33 "CAN_DET_AZIMUTH_33";
BA_ "GenSigSendType" SG_ 320 CAN_DET_RANGE_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_RANGE_33 "CAN_DET_RANGE_33";
BA_ "GenSigStartValue" SG_ 320 CAN_DET_RANGE_RATE_33 0;
BA_ "GenSigSendType" SG_ 320 CAN_DET_RANGE_RATE_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_RANGE_RATE_33 "CAN_DET_RANGE_RATE_33";
BA_ "GenSigSendType" SG_ 320 CAN_DET_AMPLITUDE_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_DET_AMPLITUDE_33 "CAN_DET_AMPLITUDE_33";
BA_ "GenSigSendType" SG_ 320 CAN_SCAN_INDEX_2LSB_33 0;
BA_ "GenSigCmt" SG_ 320 CAN_SCAN_INDEX_2LSB_33 "CAN_SCAN_INDEX_2LSB_33";
BA_ "GenMsgSendType" BO_ 319 1;
BA_ "GenMsgILSupport" BO_ 319 1;
BA_ "GenMsgNrOfRepetition" BO_ 319 0;
BA_ "GenMsgCycleTime" BO_ 319 0;
BA_ "NetworkInitialization" BO_ 319 0;
BA_ "GenMsgDelayTime" BO_ 319 0;
BA_ "GenSigVtEn" SG_ 319 CAN_DET_CONFID_AZIMUTH_32 "CAN_DET_CONFID_AZIMUTH_32";
BA_ "GenSigVtName" SG_ 319 CAN_DET_CONFID_AZIMUTH_32 "CAN_DET_CONFID_AZIMUTH_32";
BA_ "GenSigSendType" SG_ 319 CAN_DET_CONFID_AZIMUTH_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_CONFID_AZIMUTH_32 "CAN_DET_CONFID_AZIMUTH_32";
BA_ "GenSigSendType" SG_ 319 CAN_DET_SUPER_RES_TARGET_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_SUPER_RES_TARGET_32 "CAN_DET_SUPER_RES_TARGET_32";
BA_ "GenSigSendType" SG_ 319 CAN_DET_ND_TARGET_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_ND_TARGET_32 "CAN_DET_ND_TARGET_32";
BA_ "GenSigSendType" SG_ 319 CAN_DET_HOST_VEH_CLUTTER_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_HOST_VEH_CLUTTER_32 "CAN_DET_HOST_VEH_CLUTTER_32";
BA_ "GenSigSendType" SG_ 319 CAN_DET_VALID_LEVEL_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_VALID_LEVEL_32 "CAN_DET_VALID_LEVEL_32";
BA_ "GenSigStartValue" SG_ 319 CAN_DET_AZIMUTH_32 0;
BA_ "GenSigSendType" SG_ 319 CAN_DET_AZIMUTH_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_AZIMUTH_32 "CAN_DET_AZIMUTH_32";
BA_ "GenSigSendType" SG_ 319 CAN_DET_RANGE_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_RANGE_32 "CAN_DET_RANGE_32";
BA_ "GenSigStartValue" SG_ 319 CAN_DET_RANGE_RATE_32 0;
BA_ "GenSigSendType" SG_ 319 CAN_DET_RANGE_RATE_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_RANGE_RATE_32 "CAN_DET_RANGE_RATE_32";
BA_ "GenSigSendType" SG_ 319 CAN_DET_AMPLITUDE_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_DET_AMPLITUDE_32 "CAN_DET_AMPLITUDE_32";
BA_ "GenSigSendType" SG_ 319 CAN_SCAN_INDEX_2LSB_32 0;
BA_ "GenSigCmt" SG_ 319 CAN_SCAN_INDEX_2LSB_32 "CAN_SCAN_INDEX_2LSB_32";
BA_ "GenMsgSendType" BO_ 318 1;
BA_ "GenMsgILSupport" BO_ 318 1;
BA_ "GenMsgNrOfRepetition" BO_ 318 0;
BA_ "GenMsgCycleTime" BO_ 318 0;
BA_ "NetworkInitialization" BO_ 318 0;
BA_ "GenMsgDelayTime" BO_ 318 0;
BA_ "GenSigVtEn" SG_ 318 CAN_DET_CONFID_AZIMUTH_31 "CAN_DET_CONFID_AZIMUTH_31";
BA_ "GenSigVtName" SG_ 318 CAN_DET_CONFID_AZIMUTH_31 "CAN_DET_CONFID_AZIMUTH_31";
BA_ "GenSigSendType" SG_ 318 CAN_DET_CONFID_AZIMUTH_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_CONFID_AZIMUTH_31 "CAN_DET_CONFID_AZIMUTH_31";
BA_ "GenSigSendType" SG_ 318 CAN_DET_SUPER_RES_TARGET_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_SUPER_RES_TARGET_31 "CAN_DET_SUPER_RES_TARGET_31";
BA_ "GenSigSendType" SG_ 318 CAN_DET_ND_TARGET_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_ND_TARGET_31 "CAN_DET_ND_TARGET_31";
BA_ "GenSigSendType" SG_ 318 CAN_DET_HOST_VEH_CLUTTER_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_HOST_VEH_CLUTTER_31 "CAN_DET_HOST_VEH_CLUTTER_31";
BA_ "GenSigSendType" SG_ 318 CAN_DET_VALID_LEVEL_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_VALID_LEVEL_31 "CAN_DET_VALID_LEVEL_31";
BA_ "GenSigStartValue" SG_ 318 CAN_DET_AZIMUTH_31 0;
BA_ "GenSigSendType" SG_ 318 CAN_DET_AZIMUTH_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_AZIMUTH_31 "CAN_DET_AZIMUTH_31";
BA_ "GenSigSendType" SG_ 318 CAN_DET_RANGE_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_RANGE_31 "CAN_DET_RANGE_31";
BA_ "GenSigStartValue" SG_ 318 CAN_DET_RANGE_RATE_31 0;
BA_ "GenSigSendType" SG_ 318 CAN_DET_RANGE_RATE_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_RANGE_RATE_31 "CAN_DET_RANGE_RATE_31";
BA_ "GenSigSendType" SG_ 318 CAN_DET_AMPLITUDE_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_DET_AMPLITUDE_31 "CAN_DET_AMPLITUDE_31";
BA_ "GenSigSendType" SG_ 318 CAN_SCAN_INDEX_2LSB_31 0;
BA_ "GenSigCmt" SG_ 318 CAN_SCAN_INDEX_2LSB_31 "CAN_SCAN_INDEX_2LSB_31";
BA_ "GenMsgSendType" BO_ 317 1;
BA_ "GenMsgILSupport" BO_ 317 1;
BA_ "GenMsgNrOfRepetition" BO_ 317 0;
BA_ "GenMsgCycleTime" BO_ 317 0;
BA_ "NetworkInitialization" BO_ 317 0;
BA_ "GenMsgDelayTime" BO_ 317 0;
BA_ "GenSigVtEn" SG_ 317 CAN_DET_CONFID_AZIMUTH_30 "CAN_DET_CONFID_AZIMUTH_30";
BA_ "GenSigVtName" SG_ 317 CAN_DET_CONFID_AZIMUTH_30 "CAN_DET_CONFID_AZIMUTH_30";
BA_ "GenSigSendType" SG_ 317 CAN_DET_CONFID_AZIMUTH_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_CONFID_AZIMUTH_30 "CAN_DET_CONFID_AZIMUTH_30";
BA_ "GenSigSendType" SG_ 317 CAN_DET_SUPER_RES_TARGET_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_SUPER_RES_TARGET_30 "CAN_DET_SUPER_RES_TARGET_30";
BA_ "GenSigSendType" SG_ 317 CAN_DET_ND_TARGET_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_ND_TARGET_30 "CAN_DET_ND_TARGET_30";
BA_ "GenSigSendType" SG_ 317 CAN_DET_HOST_VEH_CLUTTER_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_HOST_VEH_CLUTTER_30 "CAN_DET_HOST_VEH_CLUTTER_30";
BA_ "GenSigSendType" SG_ 317 CAN_DET_VALID_LEVEL_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_VALID_LEVEL_30 "CAN_DET_VALID_LEVEL_30";
BA_ "GenSigStartValue" SG_ 317 CAN_DET_AZIMUTH_30 0;
BA_ "GenSigSendType" SG_ 317 CAN_DET_AZIMUTH_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_AZIMUTH_30 "CAN_DET_AZIMUTH_30";
BA_ "GenSigSendType" SG_ 317 CAN_DET_RANGE_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_RANGE_30 "CAN_DET_RANGE_30";
BA_ "GenSigStartValue" SG_ 317 CAN_DET_RANGE_RATE_30 0;
BA_ "GenSigSendType" SG_ 317 CAN_DET_RANGE_RATE_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_RANGE_RATE_30 "CAN_DET_RANGE_RATE_30";
BA_ "GenSigSendType" SG_ 317 CAN_DET_AMPLITUDE_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_DET_AMPLITUDE_30 "CAN_DET_AMPLITUDE_30";
BA_ "GenSigSendType" SG_ 317 CAN_SCAN_INDEX_2LSB_30 0;
BA_ "GenSigCmt" SG_ 317 CAN_SCAN_INDEX_2LSB_30 "CAN_SCAN_INDEX_2LSB_30";
BA_ "GenMsgSendType" BO_ 316 1;
BA_ "GenMsgILSupport" BO_ 316 1;
BA_ "GenMsgNrOfRepetition" BO_ 316 0;
BA_ "GenMsgCycleTime" BO_ 316 0;
BA_ "NetworkInitialization" BO_ 316 0;
BA_ "GenMsgDelayTime" BO_ 316 0;
BA_ "GenSigVtEn" SG_ 316 CAN_DET_CONFID_AZIMUTH_29 "CAN_DET_CONFID_AZIMUTH_29";
BA_ "GenSigVtName" SG_ 316 CAN_DET_CONFID_AZIMUTH_29 "CAN_DET_CONFID_AZIMUTH_29";
BA_ "GenSigSendType" SG_ 316 CAN_DET_CONFID_AZIMUTH_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_CONFID_AZIMUTH_29 "CAN_DET_CONFID_AZIMUTH_29";
BA_ "GenSigSendType" SG_ 316 CAN_DET_SUPER_RES_TARGET_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_SUPER_RES_TARGET_29 "CAN_DET_SUPER_RES_TARGET_29";
BA_ "GenSigSendType" SG_ 316 CAN_DET_ND_TARGET_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_ND_TARGET_29 "CAN_DET_ND_TARGET_29";
BA_ "GenSigSendType" SG_ 316 CAN_DET_HOST_VEH_CLUTTER_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_HOST_VEH_CLUTTER_29 "CAN_DET_HOST_VEH_CLUTTER_29";
BA_ "GenSigSendType" SG_ 316 CAN_DET_VALID_LEVEL_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_VALID_LEVEL_29 "CAN_DET_VALID_LEVEL_29";
BA_ "GenSigStartValue" SG_ 316 CAN_DET_AZIMUTH_29 0;
BA_ "GenSigSendType" SG_ 316 CAN_DET_AZIMUTH_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_AZIMUTH_29 "CAN_DET_AZIMUTH_29";
BA_ "GenSigSendType" SG_ 316 CAN_DET_RANGE_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_RANGE_29 "CAN_DET_RANGE_29";
BA_ "GenSigStartValue" SG_ 316 CAN_DET_RANGE_RATE_29 0;
BA_ "GenSigSendType" SG_ 316 CAN_DET_RANGE_RATE_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_RANGE_RATE_29 "CAN_DET_RANGE_RATE_29";
BA_ "GenSigSendType" SG_ 316 CAN_DET_AMPLITUDE_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_DET_AMPLITUDE_29 "CAN_DET_AMPLITUDE_29";
BA_ "GenSigSendType" SG_ 316 CAN_SCAN_INDEX_2LSB_29 0;
BA_ "GenSigCmt" SG_ 316 CAN_SCAN_INDEX_2LSB_29 "CAN_SCAN_INDEX_2LSB_29";
BA_ "GenMsgSendType" BO_ 314 1;
BA_ "GenMsgILSupport" BO_ 314 1;
BA_ "GenMsgNrOfRepetition" BO_ 314 0;
BA_ "GenMsgCycleTime" BO_ 314 0;
BA_ "NetworkInitialization" BO_ 314 0;
BA_ "GenMsgDelayTime" BO_ 314 0;
BA_ "GenSigVtEn" SG_ 314 CAN_DET_CONFID_AZIMUTH_27 "CAN_DET_CONFID_AZIMUTH_27";
BA_ "GenSigVtName" SG_ 314 CAN_DET_CONFID_AZIMUTH_27 "CAN_DET_CONFID_AZIMUTH_27";
BA_ "GenSigSendType" SG_ 314 CAN_DET_CONFID_AZIMUTH_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_CONFID_AZIMUTH_27 "CAN_DET_CONFID_AZIMUTH_27";
BA_ "GenSigSendType" SG_ 314 CAN_DET_SUPER_RES_TARGET_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_SUPER_RES_TARGET_27 "CAN_DET_SUPER_RES_TARGET_27";
BA_ "GenSigSendType" SG_ 314 CAN_DET_ND_TARGET_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_ND_TARGET_27 "CAN_DET_ND_TARGET_27";
BA_ "GenSigSendType" SG_ 314 CAN_DET_HOST_VEH_CLUTTER_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_HOST_VEH_CLUTTER_27 "CAN_DET_HOST_VEH_CLUTTER_27";
BA_ "GenSigSendType" SG_ 314 CAN_DET_VALID_LEVEL_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_VALID_LEVEL_27 "CAN_DET_VALID_LEVEL_27";
BA_ "GenSigStartValue" SG_ 314 CAN_DET_AZIMUTH_27 0;
BA_ "GenSigSendType" SG_ 314 CAN_DET_AZIMUTH_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_AZIMUTH_27 "CAN_DET_AZIMUTH_27";
BA_ "GenSigSendType" SG_ 314 CAN_DET_RANGE_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_RANGE_27 "CAN_DET_RANGE_27";
BA_ "GenSigStartValue" SG_ 314 CAN_DET_RANGE_RATE_27 0;
BA_ "GenSigSendType" SG_ 314 CAN_DET_RANGE_RATE_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_RANGE_RATE_27 "CAN_DET_RANGE_RATE_27";
BA_ "GenSigSendType" SG_ 314 CAN_DET_AMPLITUDE_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_DET_AMPLITUDE_27 "CAN_DET_AMPLITUDE_27";
BA_ "GenSigSendType" SG_ 314 CAN_SCAN_INDEX_2LSB_27 0;
BA_ "GenSigCmt" SG_ 314 CAN_SCAN_INDEX_2LSB_27 "CAN_SCAN_INDEX_2LSB_27";
BA_ "GenMsgSendType" BO_ 313 1;
BA_ "GenMsgILSupport" BO_ 313 1;
BA_ "GenMsgNrOfRepetition" BO_ 313 0;
BA_ "GenMsgCycleTime" BO_ 313 0;
BA_ "NetworkInitialization" BO_ 313 0;
BA_ "GenMsgDelayTime" BO_ 313 0;
BA_ "GenSigVtEn" SG_ 313 CAN_DET_CONFID_AZIMUTH_26 "CAN_DET_CONFID_AZIMUTH_26";
BA_ "GenSigVtName" SG_ 313 CAN_DET_CONFID_AZIMUTH_26 "CAN_DET_CONFID_AZIMUTH_26";
BA_ "GenSigSendType" SG_ 313 CAN_DET_CONFID_AZIMUTH_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_CONFID_AZIMUTH_26 "CAN_DET_CONFID_AZIMUTH_26";
BA_ "GenSigSendType" SG_ 313 CAN_DET_SUPER_RES_TARGET_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_SUPER_RES_TARGET_26 "CAN_DET_SUPER_RES_TARGET_26";
BA_ "GenSigSendType" SG_ 313 CAN_DET_ND_TARGET_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_ND_TARGET_26 "CAN_DET_ND_TARGET_26";
BA_ "GenSigSendType" SG_ 313 CAN_DET_HOST_VEH_CLUTTER_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_HOST_VEH_CLUTTER_26 "CAN_DET_HOST_VEH_CLUTTER_26";
BA_ "GenSigSendType" SG_ 313 CAN_DET_VALID_LEVEL_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_VALID_LEVEL_26 "CAN_DET_VALID_LEVEL_26";
BA_ "GenSigStartValue" SG_ 313 CAN_DET_AZIMUTH_26 0;
BA_ "GenSigSendType" SG_ 313 CAN_DET_AZIMUTH_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_AZIMUTH_26 "CAN_DET_AZIMUTH_26";
BA_ "GenSigSendType" SG_ 313 CAN_DET_RANGE_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_RANGE_26 "CAN_DET_RANGE_26";
BA_ "GenSigStartValue" SG_ 313 CAN_DET_RANGE_RATE_26 0;
BA_ "GenSigSendType" SG_ 313 CAN_DET_RANGE_RATE_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_RANGE_RATE_26 "CAN_DET_RANGE_RATE_26";
BA_ "GenSigSendType" SG_ 313 CAN_DET_AMPLITUDE_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_DET_AMPLITUDE_26 "CAN_DET_AMPLITUDE_26";
BA_ "GenSigSendType" SG_ 313 CAN_SCAN_INDEX_2LSB_26 0;
BA_ "GenSigCmt" SG_ 313 CAN_SCAN_INDEX_2LSB_26 "CAN_SCAN_INDEX_2LSB_26";
BA_ "GenMsgSendType" BO_ 312 1;
BA_ "GenMsgILSupport" BO_ 312 1;
BA_ "GenMsgNrOfRepetition" BO_ 312 0;
BA_ "GenMsgCycleTime" BO_ 312 0;
BA_ "NetworkInitialization" BO_ 312 0;
BA_ "GenMsgDelayTime" BO_ 312 0;
BA_ "GenSigVtEn" SG_ 312 CAN_DET_CONFID_AZIMUTH_25 "CAN_DET_CONFID_AZIMUTH_25";
BA_ "GenSigVtName" SG_ 312 CAN_DET_CONFID_AZIMUTH_25 "CAN_DET_CONFID_AZIMUTH_25";
BA_ "GenSigSendType" SG_ 312 CAN_DET_CONFID_AZIMUTH_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_CONFID_AZIMUTH_25 "CAN_DET_CONFID_AZIMUTH_25";
BA_ "GenSigSendType" SG_ 312 CAN_DET_SUPER_RES_TARGET_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_SUPER_RES_TARGET_25 "CAN_DET_SUPER_RES_TARGET_25";
BA_ "GenSigSendType" SG_ 312 CAN_DET_ND_TARGET_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_ND_TARGET_25 "CAN_DET_ND_TARGET_25";
BA_ "GenSigSendType" SG_ 312 CAN_DET_HOST_VEH_CLUTTER_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_HOST_VEH_CLUTTER_25 "CAN_DET_HOST_VEH_CLUTTER_25";
BA_ "GenSigSendType" SG_ 312 CAN_DET_VALID_LEVEL_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_VALID_LEVEL_25 "CAN_DET_VALID_LEVEL_25";
BA_ "GenSigStartValue" SG_ 312 CAN_DET_AZIMUTH_25 0;
BA_ "GenSigSendType" SG_ 312 CAN_DET_AZIMUTH_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_AZIMUTH_25 "CAN_DET_AZIMUTH_25";
BA_ "GenSigSendType" SG_ 312 CAN_DET_RANGE_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_RANGE_25 "CAN_DET_RANGE_25";
BA_ "GenSigStartValue" SG_ 312 CAN_DET_RANGE_RATE_25 0;
BA_ "GenSigSendType" SG_ 312 CAN_DET_RANGE_RATE_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_RANGE_RATE_25 "CAN_DET_RANGE_RATE_25";
BA_ "GenSigSendType" SG_ 312 CAN_DET_AMPLITUDE_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_DET_AMPLITUDE_25 "CAN_DET_AMPLITUDE_25";
BA_ "GenSigSendType" SG_ 312 CAN_SCAN_INDEX_2LSB_25 0;
BA_ "GenSigCmt" SG_ 312 CAN_SCAN_INDEX_2LSB_25 "CAN_SCAN_INDEX_2LSB_25";
BA_ "GenMsgSendType" BO_ 311 1;
BA_ "GenMsgILSupport" BO_ 311 1;
BA_ "GenMsgNrOfRepetition" BO_ 311 0;
BA_ "GenMsgCycleTime" BO_ 311 0;
BA_ "NetworkInitialization" BO_ 311 0;
BA_ "GenMsgDelayTime" BO_ 311 0;
BA_ "GenSigVtEn" SG_ 311 CAN_DET_CONFID_AZIMUTH_24 "CAN_DET_CONFID_AZIMUTH_24";
BA_ "GenSigVtName" SG_ 311 CAN_DET_CONFID_AZIMUTH_24 "CAN_DET_CONFID_AZIMUTH_24";
BA_ "GenSigSendType" SG_ 311 CAN_DET_CONFID_AZIMUTH_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_CONFID_AZIMUTH_24 "CAN_DET_CONFID_AZIMUTH_24";
BA_ "GenSigSendType" SG_ 311 CAN_DET_SUPER_RES_TARGET_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_SUPER_RES_TARGET_24 "CAN_DET_SUPER_RES_TARGET_24";
BA_ "GenSigSendType" SG_ 311 CAN_DET_ND_TARGET_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_ND_TARGET_24 "CAN_DET_ND_TARGET_24";
BA_ "GenSigSendType" SG_ 311 CAN_DET_HOST_VEH_CLUTTER_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_HOST_VEH_CLUTTER_24 "CAN_DET_HOST_VEH_CLUTTER_24";
BA_ "GenSigSendType" SG_ 311 CAN_DET_VALID_LEVEL_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_VALID_LEVEL_24 "CAN_DET_VALID_LEVEL_24";
BA_ "GenSigStartValue" SG_ 311 CAN_DET_AZIMUTH_24 0;
BA_ "GenSigSendType" SG_ 311 CAN_DET_AZIMUTH_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_AZIMUTH_24 "CAN_DET_AZIMUTH_24";
BA_ "GenSigSendType" SG_ 311 CAN_DET_RANGE_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_RANGE_24 "CAN_DET_RANGE_24";
BA_ "GenSigStartValue" SG_ 311 CAN_DET_RANGE_RATE_24 0;
BA_ "GenSigSendType" SG_ 311 CAN_DET_RANGE_RATE_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_RANGE_RATE_24 "CAN_DET_RANGE_RATE_24";
BA_ "GenSigSendType" SG_ 311 CAN_DET_AMPLITUDE_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_DET_AMPLITUDE_24 "CAN_DET_AMPLITUDE_24";
BA_ "GenSigSendType" SG_ 311 CAN_SCAN_INDEX_2LSB_24 0;
BA_ "GenSigCmt" SG_ 311 CAN_SCAN_INDEX_2LSB_24 "CAN_SCAN_INDEX_2LSB_24";
BA_ "GenMsgSendType" BO_ 310 1;
BA_ "GenMsgILSupport" BO_ 310 1;
BA_ "GenMsgNrOfRepetition" BO_ 310 0;
BA_ "GenMsgCycleTime" BO_ 310 0;
BA_ "NetworkInitialization" BO_ 310 0;
BA_ "GenMsgDelayTime" BO_ 310 0;
BA_ "GenSigVtEn" SG_ 310 CAN_DET_CONFID_AZIMUTH_23 "CAN_DET_CONFID_AZIMUTH_23";
BA_ "GenSigVtName" SG_ 310 CAN_DET_CONFID_AZIMUTH_23 "CAN_DET_CONFID_AZIMUTH_23";
BA_ "GenSigSendType" SG_ 310 CAN_DET_CONFID_AZIMUTH_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_CONFID_AZIMUTH_23 "CAN_DET_CONFID_AZIMUTH_23";
BA_ "GenSigSendType" SG_ 310 CAN_DET_SUPER_RES_TARGET_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_SUPER_RES_TARGET_23 "CAN_DET_SUPER_RES_TARGET_23";
BA_ "GenSigSendType" SG_ 310 CAN_DET_ND_TARGET_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_ND_TARGET_23 "CAN_DET_ND_TARGET_23";
BA_ "GenSigSendType" SG_ 310 CAN_DET_HOST_VEH_CLUTTER_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_HOST_VEH_CLUTTER_23 "CAN_DET_HOST_VEH_CLUTTER_23";
BA_ "GenSigSendType" SG_ 310 CAN_DET_VALID_LEVEL_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_VALID_LEVEL_23 "CAN_DET_VALID_LEVEL_23";
BA_ "GenSigStartValue" SG_ 310 CAN_DET_AZIMUTH_23 0;
BA_ "GenSigSendType" SG_ 310 CAN_DET_AZIMUTH_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_AZIMUTH_23 "CAN_DET_AZIMUTH_23";
BA_ "GenSigSendType" SG_ 310 CAN_DET_RANGE_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_RANGE_23 "CAN_DET_RANGE_23";
BA_ "GenSigStartValue" SG_ 310 CAN_DET_RANGE_RATE_23 0;
BA_ "GenSigSendType" SG_ 310 CAN_DET_RANGE_RATE_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_RANGE_RATE_23 "CAN_DET_RANGE_RATE_23";
BA_ "GenSigSendType" SG_ 310 CAN_DET_AMPLITUDE_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_DET_AMPLITUDE_23 "CAN_DET_AMPLITUDE_23";
BA_ "GenSigSendType" SG_ 310 CAN_SCAN_INDEX_2LSB_23 0;
BA_ "GenSigCmt" SG_ 310 CAN_SCAN_INDEX_2LSB_23 "CAN_SCAN_INDEX_2LSB_23";
EOF

build_ba "22"
build_ba "21"
build_ba "20"
build_ba "19"
build_ba "18"

cat <<EOF >> ${OUT_FILENAME}
BA_ "GenMsgSendType" BO_ 341 1;
BA_ "GenMsgILSupport" BO_ 341 1;
BA_ "GenMsgNrOfRepetition" BO_ 341 0;
BA_ "GenMsgCycleTime" BO_ 341 0;
BA_ "NetworkInitialization" BO_ 341 0;
BA_ "GenMsgDelayTime" BO_ 341 0;
BA_ "GenSigVtEn" SG_ 341 CAN_DET_CONFID_AZIMUTH_54 "CAN_DET_CONFID_AZIMUTH_54";
BA_ "GenSigVtName" SG_ 341 CAN_DET_CONFID_AZIMUTH_54 "CAN_DET_CONFID_AZIMUTH_54";
BA_ "GenSigSendType" SG_ 341 CAN_DET_CONFID_AZIMUTH_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_CONFID_AZIMUTH_54 "CAN_DET_CONFID_AZIMUTH_54";
BA_ "GenSigSendType" SG_ 341 CAN_DET_SUPER_RES_TARGET_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_SUPER_RES_TARGET_54 "CAN_DET_SUPER_RES_TARGET_54";
BA_ "GenSigSendType" SG_ 341 CAN_DET_ND_TARGET_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_ND_TARGET_54 "CAN_DET_ND_TARGET_54";
BA_ "GenSigSendType" SG_ 341 CAN_DET_HOST_VEH_CLUTTER_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_HOST_VEH_CLUTTER_54 "CAN_DET_HOST_VEH_CLUTTER_54";
BA_ "GenSigSendType" SG_ 341 CAN_DET_VALID_LEVEL_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_VALID_LEVEL_54 "CAN_DET_VALID_LEVEL_54";
BA_ "GenSigStartValue" SG_ 341 CAN_DET_AZIMUTH_54 0;
BA_ "GenSigSendType" SG_ 341 CAN_DET_AZIMUTH_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_AZIMUTH_54 "CAN_DET_AZIMUTH_54";
BA_ "GenSigSendType" SG_ 341 CAN_DET_RANGE_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_RANGE_54 "CAN_DET_RANGE_54";
BA_ "GenSigStartValue" SG_ 341 CAN_DET_RANGE_RATE_54 0;
BA_ "GenSigSendType" SG_ 341 CAN_DET_RANGE_RATE_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_RANGE_RATE_54 "CAN_DET_RANGE_RATE_54";
BA_ "GenSigSendType" SG_ 341 CAN_DET_AMPLITUDE_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_DET_AMPLITUDE_54 "CAN_DET_AMPLITUDE_54";
BA_ "GenSigSendType" SG_ 341 CAN_SCAN_INDEX_2LSB_54 0;
BA_ "GenSigCmt" SG_ 341 CAN_SCAN_INDEX_2LSB_54 "CAN_SCAN_INDEX_2LSB_54";
BA_ "GenMsgSendType" BO_ 340 1;
BA_ "GenMsgILSupport" BO_ 340 1;
BA_ "GenMsgNrOfRepetition" BO_ 340 0;
BA_ "GenMsgCycleTime" BO_ 340 0;
BA_ "NetworkInitialization" BO_ 340 0;
BA_ "GenMsgDelayTime" BO_ 340 0;
BA_ "GenSigVtEn" SG_ 340 CAN_DET_CONFID_AZIMUTH_53 "CAN_DET_CONFID_AZIMUTH_53";
BA_ "GenSigVtName" SG_ 340 CAN_DET_CONFID_AZIMUTH_53 "CAN_DET_CONFID_AZIMUTH_53";
BA_ "GenSigSendType" SG_ 340 CAN_DET_CONFID_AZIMUTH_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_CONFID_AZIMUTH_53 "CAN_DET_CONFID_AZIMUTH_53";
BA_ "GenSigSendType" SG_ 340 CAN_DET_SUPER_RES_TARGET_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_SUPER_RES_TARGET_53 "CAN_DET_SUPER_RES_TARGET_53";
BA_ "GenSigSendType" SG_ 340 CAN_DET_ND_TARGET_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_ND_TARGET_53 "CAN_DET_ND_TARGET_53";
BA_ "GenSigSendType" SG_ 340 CAN_DET_HOST_VEH_CLUTTER_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_HOST_VEH_CLUTTER_53 "CAN_DET_HOST_VEH_CLUTTER_53";
BA_ "GenSigSendType" SG_ 340 CAN_DET_VALID_LEVEL_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_VALID_LEVEL_53 "CAN_DET_VALID_LEVEL_53";
BA_ "GenSigStartValue" SG_ 340 CAN_DET_AZIMUTH_53 0;
BA_ "GenSigSendType" SG_ 340 CAN_DET_AZIMUTH_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_AZIMUTH_53 "CAN_DET_AZIMUTH_53";
BA_ "GenSigSendType" SG_ 340 CAN_DET_RANGE_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_RANGE_53 "CAN_DET_RANGE_53";
BA_ "GenSigStartValue" SG_ 340 CAN_DET_RANGE_RATE_53 0;
BA_ "GenSigSendType" SG_ 340 CAN_DET_RANGE_RATE_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_RANGE_RATE_53 "CAN_DET_RANGE_RATE_53";
BA_ "GenSigSendType" SG_ 340 CAN_DET_AMPLITUDE_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_DET_AMPLITUDE_53 "CAN_DET_AMPLITUDE_53";
BA_ "GenSigSendType" SG_ 340 CAN_SCAN_INDEX_2LSB_53 0;
BA_ "GenSigCmt" SG_ 340 CAN_SCAN_INDEX_2LSB_53 "CAN_SCAN_INDEX_2LSB_53";
BA_ "GenMsgSendType" BO_ 339 1;
BA_ "GenMsgILSupport" BO_ 339 1;
BA_ "GenMsgNrOfRepetition" BO_ 339 0;
BA_ "GenMsgCycleTime" BO_ 339 0;
BA_ "NetworkInitialization" BO_ 339 0;
BA_ "GenMsgDelayTime" BO_ 339 0;
BA_ "GenSigVtEn" SG_ 339 CAN_DET_CONFID_AZIMUTH_52 "CAN_DET_CONFID_AZIMUTH_52";
BA_ "GenSigVtName" SG_ 339 CAN_DET_CONFID_AZIMUTH_52 "CAN_DET_CONFID_AZIMUTH_52";
BA_ "GenSigSendType" SG_ 339 CAN_DET_CONFID_AZIMUTH_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_CONFID_AZIMUTH_52 "CAN_DET_CONFID_AZIMUTH_52";
BA_ "GenSigSendType" SG_ 339 CAN_DET_SUPER_RES_TARGET_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_SUPER_RES_TARGET_52 "CAN_DET_SUPER_RES_TARGET_52";
BA_ "GenSigSendType" SG_ 339 CAN_DET_ND_TARGET_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_ND_TARGET_52 "CAN_DET_ND_TARGET_52";
BA_ "GenSigSendType" SG_ 339 CAN_DET_HOST_VEH_CLUTTER_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_HOST_VEH_CLUTTER_52 "CAN_DET_HOST_VEH_CLUTTER_52";
BA_ "GenSigSendType" SG_ 339 CAN_DET_VALID_LEVEL_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_VALID_LEVEL_52 "CAN_DET_VALID_LEVEL_52";
BA_ "GenSigStartValue" SG_ 339 CAN_DET_AZIMUTH_52 0;
BA_ "GenSigSendType" SG_ 339 CAN_DET_AZIMUTH_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_AZIMUTH_52 "CAN_DET_AZIMUTH_52";
BA_ "GenSigSendType" SG_ 339 CAN_DET_RANGE_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_RANGE_52 "CAN_DET_RANGE_52";
BA_ "GenSigStartValue" SG_ 339 CAN_DET_RANGE_RATE_52 0;
BA_ "GenSigSendType" SG_ 339 CAN_DET_RANGE_RATE_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_RANGE_RATE_52 "CAN_DET_RANGE_RATE_52";
BA_ "GenSigSendType" SG_ 339 CAN_DET_AMPLITUDE_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_DET_AMPLITUDE_52 "CAN_DET_AMPLITUDE_52";
BA_ "GenSigSendType" SG_ 339 CAN_SCAN_INDEX_2LSB_52 0;
BA_ "GenSigCmt" SG_ 339 CAN_SCAN_INDEX_2LSB_52 "CAN_SCAN_INDEX_2LSB_52";
BA_ "GenMsgSendType" BO_ 338 1;
BA_ "GenMsgILSupport" BO_ 338 1;
BA_ "GenMsgNrOfRepetition" BO_ 338 0;
BA_ "GenMsgCycleTime" BO_ 338 0;
BA_ "NetworkInitialization" BO_ 338 0;
BA_ "GenMsgDelayTime" BO_ 338 0;
BA_ "GenSigVtEn" SG_ 338 CAN_DET_CONFID_AZIMUTH_51 "CAN_DET_CONFID_AZIMUTH_51";
BA_ "GenSigVtName" SG_ 338 CAN_DET_CONFID_AZIMUTH_51 "CAN_DET_CONFID_AZIMUTH_51";
BA_ "GenSigSendType" SG_ 338 CAN_DET_CONFID_AZIMUTH_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_CONFID_AZIMUTH_51 "CAN_DET_CONFID_AZIMUTH_51";
BA_ "GenSigSendType" SG_ 338 CAN_DET_SUPER_RES_TARGET_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_SUPER_RES_TARGET_51 "CAN_DET_SUPER_RES_TARGET_51";
BA_ "GenSigSendType" SG_ 338 CAN_DET_ND_TARGET_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_ND_TARGET_51 "CAN_DET_ND_TARGET_51";
BA_ "GenSigSendType" SG_ 338 CAN_DET_HOST_VEH_CLUTTER_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_HOST_VEH_CLUTTER_51 "CAN_DET_HOST_VEH_CLUTTER_51";
BA_ "GenSigSendType" SG_ 338 CAN_DET_VALID_LEVEL_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_VALID_LEVEL_51 "CAN_DET_VALID_LEVEL_51";
BA_ "GenSigStartValue" SG_ 338 CAN_DET_AZIMUTH_51 0;
BA_ "GenSigSendType" SG_ 338 CAN_DET_AZIMUTH_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_AZIMUTH_51 "CAN_DET_AZIMUTH_51";
BA_ "GenSigSendType" SG_ 338 CAN_DET_RANGE_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_RANGE_51 "CAN_DET_RANGE_51";
BA_ "GenSigStartValue" SG_ 338 CAN_DET_RANGE_RATE_51 0;
BA_ "GenSigSendType" SG_ 338 CAN_DET_RANGE_RATE_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_RANGE_RATE_51 "CAN_DET_RANGE_RATE_51";
BA_ "GenSigSendType" SG_ 338 CAN_DET_AMPLITUDE_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_DET_AMPLITUDE_51 "CAN_DET_AMPLITUDE_51";
BA_ "GenSigSendType" SG_ 338 CAN_SCAN_INDEX_2LSB_51 0;
BA_ "GenSigCmt" SG_ 338 CAN_SCAN_INDEX_2LSB_51 "CAN_SCAN_INDEX_2LSB_51";
BA_ "GenMsgSendType" BO_ 337 1;
BA_ "GenMsgILSupport" BO_ 337 1;
BA_ "GenMsgNrOfRepetition" BO_ 337 0;
BA_ "GenMsgCycleTime" BO_ 337 0;
BA_ "NetworkInitialization" BO_ 337 0;
BA_ "GenMsgDelayTime" BO_ 337 0;
BA_ "GenSigVtEn" SG_ 337 CAN_DET_CONFID_AZIMUTH_50 "CAN_DET_CONFID_AZIMUTH_50";
BA_ "GenSigVtName" SG_ 337 CAN_DET_CONFID_AZIMUTH_50 "CAN_DET_CONFID_AZIMUTH_50";
BA_ "GenSigSendType" SG_ 337 CAN_DET_CONFID_AZIMUTH_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_CONFID_AZIMUTH_50 "CAN_DET_CONFID_AZIMUTH_50";
BA_ "GenSigSendType" SG_ 337 CAN_DET_SUPER_RES_TARGET_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_SUPER_RES_TARGET_50 "CAN_DET_SUPER_RES_TARGET_50";
BA_ "GenSigSendType" SG_ 337 CAN_DET_ND_TARGET_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_ND_TARGET_50 "CAN_DET_ND_TARGET_50";
BA_ "GenSigSendType" SG_ 337 CAN_DET_HOST_VEH_CLUTTER_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_HOST_VEH_CLUTTER_50 "CAN_DET_HOST_VEH_CLUTTER_50";
BA_ "GenSigSendType" SG_ 337 CAN_DET_VALID_LEVEL_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_VALID_LEVEL_50 "CAN_DET_VALID_LEVEL_50";
BA_ "GenSigStartValue" SG_ 337 CAN_DET_AZIMUTH_50 0;
BA_ "GenSigSendType" SG_ 337 CAN_DET_AZIMUTH_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_AZIMUTH_50 "CAN_DET_AZIMUTH_50";
BA_ "GenSigSendType" SG_ 337 CAN_DET_RANGE_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_RANGE_50 "CAN_DET_RANGE_50";
BA_ "GenSigStartValue" SG_ 337 CAN_DET_RANGE_RATE_50 0;
BA_ "GenSigSendType" SG_ 337 CAN_DET_RANGE_RATE_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_RANGE_RATE_50 "CAN_DET_RANGE_RATE_50";
BA_ "GenSigSendType" SG_ 337 CAN_DET_AMPLITUDE_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_DET_AMPLITUDE_50 "CAN_DET_AMPLITUDE_50";
BA_ "GenSigSendType" SG_ 337 CAN_SCAN_INDEX_2LSB_50 0;
BA_ "GenSigCmt" SG_ 337 CAN_SCAN_INDEX_2LSB_50 "CAN_SCAN_INDEX_2LSB_50";
BA_ "GenMsgSendType" BO_ 336 1;
BA_ "GenMsgILSupport" BO_ 336 1;
BA_ "GenMsgNrOfRepetition" BO_ 336 0;
BA_ "GenMsgCycleTime" BO_ 336 0;
BA_ "NetworkInitialization" BO_ 336 0;
BA_ "GenMsgDelayTime" BO_ 336 0;
BA_ "GenSigVtEn" SG_ 336 CAN_DET_CONFID_AZIMUTH_49 "CAN_DET_CONFID_AZIMUTH_49";
BA_ "GenSigVtName" SG_ 336 CAN_DET_CONFID_AZIMUTH_49 "CAN_DET_CONFID_AZIMUTH_49";
BA_ "GenSigSendType" SG_ 336 CAN_DET_CONFID_AZIMUTH_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_CONFID_AZIMUTH_49 "CAN_DET_CONFID_AZIMUTH_49";
BA_ "GenSigSendType" SG_ 336 CAN_DET_SUPER_RES_TARGET_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_SUPER_RES_TARGET_49 "CAN_DET_SUPER_RES_TARGET_49";
BA_ "GenSigSendType" SG_ 336 CAN_DET_ND_TARGET_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_ND_TARGET_49 "CAN_DET_ND_TARGET_49";
BA_ "GenSigSendType" SG_ 336 CAN_DET_HOST_VEH_CLUTTER_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_HOST_VEH_CLUTTER_49 "CAN_DET_HOST_VEH_CLUTTER_49";
BA_ "GenSigSendType" SG_ 336 CAN_DET_VALID_LEVEL_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_VALID_LEVEL_49 "CAN_DET_VALID_LEVEL_49";
BA_ "GenSigStartValue" SG_ 336 CAN_DET_AZIMUTH_49 0;
BA_ "GenSigSendType" SG_ 336 CAN_DET_AZIMUTH_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_AZIMUTH_49 "CAN_DET_AZIMUTH_49";
BA_ "GenSigSendType" SG_ 336 CAN_DET_RANGE_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_RANGE_49 "CAN_DET_RANGE_49";
BA_ "GenSigStartValue" SG_ 336 CAN_DET_RANGE_RATE_49 0;
BA_ "GenSigSendType" SG_ 336 CAN_DET_RANGE_RATE_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_RANGE_RATE_49 "CAN_DET_RANGE_RATE_49";
BA_ "GenSigSendType" SG_ 336 CAN_DET_AMPLITUDE_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_DET_AMPLITUDE_49 "CAN_DET_AMPLITUDE_49";
BA_ "GenSigSendType" SG_ 336 CAN_SCAN_INDEX_2LSB_49 0;
BA_ "GenSigCmt" SG_ 336 CAN_SCAN_INDEX_2LSB_49 "CAN_SCAN_INDEX_2LSB_49";
BA_ "GenMsgSendType" BO_ 326 1;
BA_ "GenMsgILSupport" BO_ 326 1;
BA_ "GenMsgNrOfRepetition" BO_ 326 0;
BA_ "GenMsgCycleTime" BO_ 326 0;
BA_ "NetworkInitialization" BO_ 326 0;
BA_ "GenMsgDelayTime" BO_ 326 0;
BA_ "GenSigVtEn" SG_ 326 CAN_DET_CONFID_AZIMUTH_39 "CAN_DET_CONFID_AZIMUTH_39";
BA_ "GenSigVtName" SG_ 326 CAN_DET_CONFID_AZIMUTH_39 "CAN_DET_CONFID_AZIMUTH_39";
BA_ "GenSigSendType" SG_ 326 CAN_DET_CONFID_AZIMUTH_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_CONFID_AZIMUTH_39 "CAN_DET_CONFID_AZIMUTH_39";
BA_ "GenSigSendType" SG_ 326 CAN_DET_SUPER_RES_TARGET_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_SUPER_RES_TARGET_39 "CAN_DET_SUPER_RES_TARGET_39";
BA_ "GenSigSendType" SG_ 326 CAN_DET_ND_TARGET_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_ND_TARGET_39 "CAN_DET_ND_TARGET_39";
BA_ "GenSigSendType" SG_ 326 CAN_DET_HOST_VEH_CLUTTER_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_HOST_VEH_CLUTTER_39 "CAN_DET_HOST_VEH_CLUTTER_39";
BA_ "GenSigSendType" SG_ 326 CAN_DET_VALID_LEVEL_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_VALID_LEVEL_39 "CAN_DET_VALID_LEVEL_39";
BA_ "GenSigStartValue" SG_ 326 CAN_DET_AZIMUTH_39 0;
BA_ "GenSigSendType" SG_ 326 CAN_DET_AZIMUTH_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_AZIMUTH_39 "CAN_DET_AZIMUTH_39";
BA_ "GenSigSendType" SG_ 326 CAN_DET_RANGE_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_RANGE_39 "CAN_DET_RANGE_39";
BA_ "GenSigStartValue" SG_ 326 CAN_DET_RANGE_RATE_39 0;
BA_ "GenSigSendType" SG_ 326 CAN_DET_RANGE_RATE_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_RANGE_RATE_39 "CAN_DET_RANGE_RATE_39";
BA_ "GenSigSendType" SG_ 326 CAN_DET_AMPLITUDE_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_DET_AMPLITUDE_39 "CAN_DET_AMPLITUDE_39";
BA_ "GenSigSendType" SG_ 326 CAN_SCAN_INDEX_2LSB_39 0;
BA_ "GenSigCmt" SG_ 326 CAN_SCAN_INDEX_2LSB_39 "CAN_SCAN_INDEX_2LSB_39";
BA_ "GenMsgSendType" BO_ 315 1;
BA_ "GenMsgILSupport" BO_ 315 1;
BA_ "GenMsgNrOfRepetition" BO_ 315 0;
BA_ "GenMsgCycleTime" BO_ 315 0;
BA_ "NetworkInitialization" BO_ 315 0;
BA_ "GenMsgDelayTime" BO_ 315 0;
BA_ "GenSigVtEn" SG_ 315 CAN_DET_CONFID_AZIMUTH_28 "CAN_DET_CONFID_AZIMUTH_28";
BA_ "GenSigVtName" SG_ 315 CAN_DET_CONFID_AZIMUTH_28 "CAN_DET_CONFID_AZIMUTH_28";
BA_ "GenSigSendType" SG_ 315 CAN_DET_CONFID_AZIMUTH_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_CONFID_AZIMUTH_28 "CAN_DET_CONFID_AZIMUTH_28";
BA_ "GenSigSendType" SG_ 315 CAN_DET_SUPER_RES_TARGET_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_SUPER_RES_TARGET_28 "CAN_DET_SUPER_RES_TARGET_28";
BA_ "GenSigSendType" SG_ 315 CAN_DET_ND_TARGET_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_ND_TARGET_28 "CAN_DET_ND_TARGET_28";
BA_ "GenSigSendType" SG_ 315 CAN_DET_HOST_VEH_CLUTTER_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_HOST_VEH_CLUTTER_28 "CAN_DET_HOST_VEH_CLUTTER_28";
BA_ "GenSigSendType" SG_ 315 CAN_DET_VALID_LEVEL_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_VALID_LEVEL_28 "CAN_DET_VALID_LEVEL_28";
BA_ "GenSigStartValue" SG_ 315 CAN_DET_AZIMUTH_28 0;
BA_ "GenSigSendType" SG_ 315 CAN_DET_AZIMUTH_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_AZIMUTH_28 "CAN_DET_AZIMUTH_28";
BA_ "GenSigSendType" SG_ 315 CAN_DET_RANGE_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_RANGE_28 "CAN_DET_RANGE_28";
BA_ "GenSigStartValue" SG_ 315 CAN_DET_RANGE_RATE_28 0;
BA_ "GenSigSendType" SG_ 315 CAN_DET_RANGE_RATE_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_RANGE_RATE_28 "CAN_DET_RANGE_RATE_28";
BA_ "GenSigSendType" SG_ 315 CAN_DET_AMPLITUDE_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_DET_AMPLITUDE_28 "CAN_DET_AMPLITUDE_28";
BA_ "GenSigSendType" SG_ 315 CAN_SCAN_INDEX_2LSB_28 0;
BA_ "GenSigCmt" SG_ 315 CAN_SCAN_INDEX_2LSB_28 "CAN_SCAN_INDEX_2LSB_28";
EOF

build_ba "17"
build_ba "16"
build_ba "15"
build_ba "14"
build_ba "13"
build_ba "12"
build_ba "11"
build_ba "10"
build_ba "09"
build_ba "08"
build_ba "07"
build_ba "06"
build_ba "05"
build_ba "03"
build_ba "02"

cat <<EOF >> ${OUT_FILENAME}
BA_ "GenMsgSendType" BO_ 256 1;
BA_ "GenMsgILSupport" BO_ 256 1;
BA_ "GenMsgNrOfRepetition" BO_ 256 0;
BA_ "NetworkInitialization" BO_ 256 0;
BA_ "GenSigCmt" SG_ 256 CAN_PCAN_MINOR_MRR "CAN_PCAN_MINOR_MRR";
BA_ "GenSigSendType" SG_ 256 CAN_PCAN_MINOR_MRR 0;
BA_ "GenSigCmt" SG_ 256 CAN_PCAN_MAJOR_MRR "CAN_PCAN_MAJOR_MRR";
BA_ "GenSigSendType" SG_ 256 CAN_PCAN_MAJOR_MRR 0;
BA_ "GenMsgCycleTime" BO_ 257 30;
BA_ "GenMsgSendType" BO_ 257 0;
BA_ "GenMsgILSupport" BO_ 257 1;
BA_ "GenMsgNrOfRepetition" BO_ 257 0;
BA_ "NetworkInitialization" BO_ 257 0;
BA_ "GenSigCmt" SG_ 257 CAN_INTERFERENCE_TYPE "CAN_INTERFERENCE_TYPE";
BA_ "GenSigVtEn" SG_ 257 CAN_INTERFERENCE_TYPE "CAN_INTERFERENCE_TYPE";
BA_ "GenSigVtName" SG_ 257 CAN_INTERFERENCE_TYPE "CAN_INTERFERENCE_TYPE";
BA_ "GenSigVtName" SG_ 257 CAN_RECOMMEND_UNCONVERGE "CAN_RECOMMEND_UNCONVERGE";
BA_ "GenSigVtEn" SG_ 257 CAN_RECOMMEND_UNCONVERGE "CAN_RECOMMEND_UNCONVERGE";
BA_ "GenSigCmt" SG_ 257 CAN_RECOMMEND_UNCONVERGE "CAN_RECOMMEND_UNCONVERGE";
BA_ "GenSigStartValue" SG_ 257 CAN_BLOCKAGE_SIDELOBE_FILTER_VAL 0;
BA_ "GenSigSendType" SG_ 257 CAN_BLOCKAGE_SIDELOBE_FILTER_VAL 0;
BA_ "GenSigCmt" SG_ 257 CAN_BLOCKAGE_SIDELOBE_FILTER_VAL "CAN_BLOCKAGE_SIDELOBE_FILTER_VAL";
BA_ "GenSigVtEn" SG_ 257 CAN_BLOCKAGE_SIDELOBE_FILTER_VAL "CAN_BLOCKAGE_SIDELOBE_FILTER_VAL";
BA_ "GenSigVtName" SG_ 257 CAN_BLOCKAGE_SIDELOBE_FILTER_VAL "CAN_BLOCKAGE_SIDELOBE_FILTER_VAL";
BA_ "GenSigCmt" SG_ 257 CAN_RADAR_ALIGN_INCOMPLETE "CAN_RADAR_ALIGN_INCOMPLETE";
BA_ "GenSigVtEn" SG_ 257 CAN_RADAR_ALIGN_INCOMPLETE "CAN_RADAR_ALIGN_INCOMPLETE";
BA_ "GenSigVtName" SG_ 257 CAN_RADAR_ALIGN_INCOMPLETE "CAN_RADAR_ALIGN_INCOMPLETE";
BA_ "GenSigCmt" SG_ 257 CAN_BLOCKAGE_SIDELOBE "CAN_BLOCKAGE_SIDELOBE";
BA_ "GenSigVtEn" SG_ 257 CAN_BLOCKAGE_SIDELOBE "CAN_BLOCKAGE_SIDELOBE";
BA_ "GenSigVtName" SG_ 257 CAN_BLOCKAGE_SIDELOBE "CAN_BLOCKAGE_SIDELOBE";
BA_ "GenSigSendType" SG_ 257 CAN_BLOCKAGE_SIDELOBE 0;
BA_ "GenSigCmt" SG_ 257 CAN_BLOCKAGE_MNR "CAN_BLOCKAGE_MNR";
BA_ "GenSigVtEn" SG_ 257 CAN_BLOCKAGE_MNR "CAN_BLOCKAGE_MNR";
BA_ "GenSigVtName" SG_ 257 CAN_BLOCKAGE_MNR "CAN_BLOCKAGE_MNR";
BA_ "GenSigSendType" SG_ 257 CAN_BLOCKAGE_MNR 0;
BA_ "GenSigCmt" SG_ 257 CAN_RADAR_EXT_COND_NOK "CAN_RADAR_EXT_COND_NOK";
BA_ "GenSigVtEn" SG_ 257 CAN_RADAR_EXT_COND_NOK "CAN_RADAR_EXT_COND_NOK";
BA_ "GenSigVtName" SG_ 257 CAN_RADAR_EXT_COND_NOK "CAN_RADAR_EXT_COND_NOK";
BA_ "GenSigSendType" SG_ 257 CAN_RADAR_EXT_COND_NOK 0;
BA_ "GenSigCmt" SG_ 257 CAN_RADAR_ALIGN_OUT_RANGE "CAN_RADAR_ALIGN_OUT_RANGE";
BA_ "GenSigVtEn" SG_ 257 CAN_RADAR_ALIGN_OUT_RANGE "CAN_RADAR_ALIGN_OUT_RANGE";
BA_ "GenSigVtName" SG_ 257 CAN_RADAR_ALIGN_OUT_RANGE "CAN_RADAR_ALIGN_OUT_RANGE";
BA_ "GenSigSendType" SG_ 257 CAN_RADAR_ALIGN_OUT_RANGE 0;
BA_ "GenSigCmt" SG_ 257 CAN_RADAR_ALIGN_NOT_START "CAN_RADAR_ALIGN_NOT_START";
BA_ "GenSigVtEn" SG_ 257 CAN_RADAR_ALIGN_NOT_START "CAN_RADAR_ALIGN_NOT_START";
BA_ "GenSigVtName" SG_ 257 CAN_RADAR_ALIGN_NOT_START "CAN_RADAR_ALIGN_NOT_START";
BA_ "GenSigSendType" SG_ 257 CAN_RADAR_ALIGN_NOT_START 0;
BA_ "GenSigCmt" SG_ 257 CAN_RADAR_OVERHEAT_ERROR "CAN_RADAR_OVERHEAT_ERROR";
BA_ "GenSigVtEn" SG_ 257 CAN_RADAR_OVERHEAT_ERROR "CAN_RADAR_OVERHEAT_ERROR";
BA_ "GenSigVtName" SG_ 257 CAN_RADAR_OVERHEAT_ERROR "CAN_RADAR_OVERHEAT_ERROR";
BA_ "GenSigSendType" SG_ 257 CAN_RADAR_OVERHEAT_ERROR 0;
BA_ "GenSigCmt" SG_ 257 CAN_RADAR_NOT_OP "CAN_RADAR_NOT_OP";
BA_ "GenSigVtEn" SG_ 257 CAN_RADAR_NOT_OP "CAN_RADAR_NOT_OP";
BA_ "GenSigVtName" SG_ 257 CAN_RADAR_NOT_OP "CAN_RADAR_NOT_OP";
BA_ "GenSigSendType" SG_ 257 CAN_RADAR_NOT_OP 0;
BA_ "GenSigCmt" SG_ 257 CAN_XCVR_OPERATIONAL "CAN_XCVR_OPERATIONAL";
BA_ "GenSigVtEn" SG_ 257 CAN_XCVR_OPERATIONAL "CAN_XCVR_OPERATIONAL";
BA_ "GenSigVtName" SG_ 257 CAN_XCVR_OPERATIONAL "CAN_XCVR_OPERATIONAL";
BA_ "GenSigSendType" SG_ 257 CAN_XCVR_OPERATIONAL 0;
EOF

build_ba "01"

cat <<EOF >> ${OUT_FILENAME}
BA_DEF_DEF_ "CrossOver_InfoCAN" "No";
BA_DEF_DEF_ "CrossOver_LIN" "No";
BA_DEF_DEF_ "UsedOnPgmDBC" "Yes";
BA_DEF_DEF_ "ContentDependant" "No";
BA_DEF_DEF_ "GenSigTimeoutTime_RCM" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_GWM" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_OCS" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_ABS_ESC" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_CCM" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_IPMA" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_TSTR" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_SCCM" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_PSCM" 0;
BA_DEF_DEF_ "GenSigTimeoutTime__delete" 0;
BA_DEF_DEF_ "GenSigTimeoutTime_Generic_BCM" 0;
BA_DEF_DEF_ "NmMessage" "No";
BA_DEF_DEF_ "DiagResponse" "No";
BA_DEF_DEF_ "DiagRequest" "No";
BA_DEF_DEF_ "TpTxIndex" 0;
BA_DEF_DEF_ "DiagState" "No";
BA_DEF_DEF_ "TpApplType" "";
BA_DEF_DEF_ "NmAsrMessage" "No";
BA_DEF_DEF_ "Mulitplexer" "No";
BA_DEF_DEF_ "ConfiguredTransmitter" "No";
BA_DEF_DEF_ "EventRateOfChange" 10000;
BA_DEF_DEF_ "GenMsgHandlingTypeDoc" "";
BA_DEF_DEF_ "GenMsgHandlingTypeCode" "";
BA_DEF_DEF_ "GenMsgMarked" "";
BA_DEF_DEF_ "GenSigMarked" "";
BA_DEF_DEF_ "GenSigVtIndex" "";
BA_DEF_DEF_ "GenSigVtName" "";
BA_DEF_DEF_ "GenSigVtEn" "";
BA_DEF_DEF_ "GenSigSNA" "";
BA_DEF_DEF_ "GenSigCmt" "";
BA_DEF_DEF_ "GenMsgCmt" "";
BA_DEF_DEF_ "GenSigSendType" "NoSigSendType";
BA_DEF_DEF_ "GenSigInactiveValue" 0;
BA_DEF_DEF_ "GenSigMissingSourceValue" 0;
BA_DEF_DEF_ "WakeupSignal" "No";
BA_DEF_DEF_ "GenSigStartValue" 0;
BA_DEF_DEF_ "GenMsgILSupport" "Yes";
BA_DEF_DEF_ "NetworkInitializationCommand" "No";
BA_DEF_DEF_ "GenMsgSendType" "NoMsgSendType";
BA_DEF_DEF_ "GenMsgCycleTime" 0;
BA_DEF_DEF_ "GenMsgCycleTimeFast" 0;
BA_DEF_DEF_ "GenMsgDelayTime" 0;
BA_DEF_DEF_ "GenMsgNrOfRepetition" 0;
BA_DEF_DEF_ "GenMsgStartDelayTime" 0;
BA_DEF_DEF_ "NetworkInitialization" "No";
BA_DEF_DEF_ "MessageGateway" "No";
BA_DEF_DEF_ "ILUsed" "Yes";
BA_DEF_DEF_ "NetworkInitializationUsed" "No";
BA_DEF_DEF_ "PowerType" "Switched";
BA_DEF_DEF_ "NodeStartUpTime" 250;
BA_DEF_DEF_ "NodeWakeUpTime" 10;
BA_DEF_DEF_ "GenMsgBackgroundColor" "#ffffff";
BA_DEF_DEF_ "GenMsgForegroundColor" "#000000";
VAL_ 34 IPMA_PCAN_DataRangeCheck 1 "Fault Present" 0 "No Fault";
VAL_ 34 IPMA_PCAN_MissingMsg 1 "Fault Present" 0 "No Fault ";
VAL_ 34 VINSignalCompareFailure 1 "Fault Present" 0 "No Fault";
VAL_ 34 ModuleNotConfiguredError 1 "Fault Present" 0 "No Fault";
VAL_ 34 CarCfgNotConfiguredError 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte7_bit7 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte7_bit6 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte7_bit5 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte7_bit4 1 "Fault Present" 0 "No Fault";
VAL_ 33 ARMtoDSPChksumFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 DSPtoArmChksumFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 HostToArmChksumFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 ARMtoHostChksumFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 LoopBWOutOfRange 1 "Fault Present" 0 "No Fault";
VAL_ 33 DSPOverrunFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte6_bit5 1 "Fault Present" 0 "No Fault";
VAL_ 33 TuningSensitivityFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 SaturatedTuningFreqFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 LocalOscPowerFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 TransmitterPowerFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte6_bit0 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte5_bit7 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte5_bit6 1 "Fault Present" 0 "No Fault";
VAL_ 33 XCVRDeviceSPIFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 FreqSynthesizerSPIFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 AnalogConverterDevicSPIFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 SidelobeBlockage 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte5_bit1 1 "Fault Present" 0 "No Fault";
VAL_ 33 MNRBlocked 1 "Fault Present" 0 "No Fault";
VAL_ 33 ECUTempHighFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 TransmitterTempHighFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 AlignmentRoutineFailedFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 UnreasonableRadarData 1 "Fault Present" 0 "No Fault";
VAL_ 33 MicroprocessorTempHighFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 VerticalAlignmentOutOfRange 1 "Fault Present" 0 "No Fault";
VAL_ 33 HorizontalAlignmentOutOfRange 1 "Fault Present" 0 "No Fault";
VAL_ 33 FactoryAlignmentMode 1 "Fault Present" 0 "No Fault";
VAL_ 33 BatteryLowFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 BatteryHighFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 v_1p25SupplyOutOfRange 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte3_bit4 1 "Fault Present" 0 "No Fault";
VAL_ 33 ThermistorOutOfRange 1 "Fault Present" 0 "No Fault";
VAL_ 33 v_3p3DACSupplyOutOfRange 1 "Fault Present" 0 "No Fault";
VAL_ 33 v_3p3RAWSupplyOutOfRange 1 "Fault Present" 0 "No Fault";
VAL_ 33 v_5_SupplyOutOfRange 1 "Fault Present" 0 "No Fault";
VAL_ 33 TransmitterIDFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte2_bit6 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte2_bit5 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte2_bit4 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte2_bit3 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte2_bit2 1 "Fault Present" 0 "No Fault";
VAL_ 33 PCANMissingMsgFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 PCANBusOff 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte1_bit7 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte1_bit6 1 "Fault Present" 0 "No Fault";
VAL_ 33 InstructionSetCheckFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 StackOverflowFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 WatchdogFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 PLLLockFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte1_bit1 1 "Fault Present" 0 "No Fault";
VAL_ 33 RAMMemoryTestFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 USCValidationFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte0_bit6 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte0_bit5 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte0_bit4 1 "Fault Present" 0 "No Fault";
VAL_ 33 Active_Flt_Latched_byte0_bit3 1 "Fault Present" 0 "No Fault";
VAL_ 33 KeepAliveChecksumFault 1 "Fault Present" 0 "No Fault";
VAL_ 33 ProgramCalibrationFlashChecksum 1 "Fault Present" 0 "No Fault";
VAL_ 33 ApplicationFlashChecksumFault 1 "Fault Present" 0 "No Fault";
VAL_ 371 CAN_AUTO_ALIGN_HANGLE_QF 3 "Accurate" 2 "Inaccurate" 1 "Temporarily undefined" 0 "Undefined";
VAL_ 371 CAN_ALIGNMENT_STATUS 15 "Undefined_2" 14 "Undefined_1" 13 "Low Amplitude (Flat-plate only)" 12 "No Peak (Flat-plate only)" 11 "Fail Ver and Hor OutOfRange" 10 "Fail Vertical Align OutOfRange" 9 "Fail Horizontal Align OutOfRange" 8 "Fail Time Out" 7 "Fail Only Right Target Found" 6 "Fail Only Left Target Found" 5 "Fail Variance Too Large" 4 "Fail Deviation Too Large" 3 "Fail No Target" 2 "Success" 1 "Busy" 0 "Off";
VAL_ 371 CAN_ALIGNMENT_STATE 6 "Static alignment flat-plate" 5 "Static alignment 2-target" 4 "Static alignment 1-target" 3 "Service alignment" 2 "Short track alignment" 1 "Auto alignment" 0 "Off";
EOF

build_val "04"

cat <<EOF >> ${OUT_FILENAME}
VAL_ 351 CAN_DET_CONFID_AZIMUTH_64 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 350 CAN_DET_CONFID_AZIMUTH_63 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 349 CAN_DET_CONFID_AZIMUTH_62 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 348 CAN_DET_CONFID_AZIMUTH_61 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 347 CAN_DET_CONFID_AZIMUTH_60 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 346 CAN_DET_CONFID_AZIMUTH_59 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 345 CAN_DET_CONFID_AZIMUTH_58 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 344 CAN_DET_CONFID_AZIMUTH_57 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 343 CAN_DET_CONFID_AZIMUTH_56 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 342 CAN_DET_CONFID_AZIMUTH_55 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 335 CAN_DET_CONFID_AZIMUTH_48 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 334 CAN_DET_CONFID_AZIMUTH_47 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 333 CAN_DET_CONFID_AZIMUTH_46 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 332 CAN_DET_CONFID_AZIMUTH_45 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 331 CAN_DET_CONFID_AZIMUTH_44 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 330 CAN_DET_CONFID_AZIMUTH_43 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 329 CAN_DET_CONFID_AZIMUTH_42 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 328 CAN_DET_CONFID_AZIMUTH_41 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 327 CAN_DET_CONFID_AZIMUTH_40 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 325 CAN_DET_CONFID_AZIMUTH_38 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 324 CAN_DET_CONFID_AZIMUTH_37 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 323 CAN_DET_CONFID_AZIMUTH_36 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 322 CAN_DET_CONFID_AZIMUTH_35 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 321 CAN_DET_CONFID_AZIMUTH_34 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 320 CAN_DET_CONFID_AZIMUTH_33 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 319 CAN_DET_CONFID_AZIMUTH_32 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 318 CAN_DET_CONFID_AZIMUTH_31 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 317 CAN_DET_CONFID_AZIMUTH_30 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 316 CAN_DET_CONFID_AZIMUTH_29 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 314 CAN_DET_CONFID_AZIMUTH_27 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 313 CAN_DET_CONFID_AZIMUTH_26 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 312 CAN_DET_CONFID_AZIMUTH_25 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 311 CAN_DET_CONFID_AZIMUTH_24 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 310 CAN_DET_CONFID_AZIMUTH_23 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
EOF

build_val "22"
build_val "21"
build_val "20"
build_val "19"
build_val "18"

cat <<EOF >> ${OUT_FILENAME}
VAL_ 341 CAN_DET_CONFID_AZIMUTH_54 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 340 CAN_DET_CONFID_AZIMUTH_53 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 339 CAN_DET_CONFID_AZIMUTH_52 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 338 CAN_DET_CONFID_AZIMUTH_51 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 337 CAN_DET_CONFID_AZIMUTH_50 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 336 CAN_DET_CONFID_AZIMUTH_49 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 326 CAN_DET_CONFID_AZIMUTH_39 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
VAL_ 315 CAN_DET_CONFID_AZIMUTH_28 3 "Low" 2 "Medium_Low" 1 "Medium_High" 0 "High";
EOF

build_val "17"
build_val "16"
build_val "15"
build_val "14"
build_val "13"
build_val "12"
build_val "11"
build_val "10"
build_val "09"
build_val "08"
build_val "07"
build_val "06"
build_val "05"
build_val "03"
build_val "02"

cat <<EOF >> ${OUT_FILENAME}
VAL_ 257 CAN_INTERFERENCE_TYPE 2 "Star PD-Like" 1 "Slow FMCW" 0 "No Interference";
VAL_ 257 CAN_RECOMMEND_UNCONVERGE 1 "Recommended" 0 "Not Recommended";
VAL_ 257 CAN_RADAR_ALIGN_INCOMPLETE 1 "Alignment Incomplete" 0 "Alignment Completed";
VAL_ 257 CAN_BLOCKAGE_SIDELOBE 1 "Radar Blockage" 0 "No Radar Blockage";
VAL_ 257 CAN_BLOCKAGE_MNR 1 "Radar Blockage" 0 "No Radar Blockage";
VAL_ 257 CAN_RADAR_EXT_COND_NOK 1 "Too high temp or insufficient pw" 0 "External conditions OK";
VAL_ 257 CAN_RADAR_ALIGN_OUT_RANGE 1 "Radar out of range" 0 "Radar within range";
VAL_ 257 CAN_RADAR_ALIGN_NOT_START 1 "Radar align not started" 0 "Radar align started";
VAL_ 257 CAN_RADAR_OVERHEAT_ERROR 1 "Radar overheat condition" 0 "No Overheat";
VAL_ 257 CAN_RADAR_NOT_OP 1 "Radar not operational" 0 "Radar operational";
VAL_ 257 CAN_XCVR_OPERATIONAL 1 "On" 0 "Off ";
EOF

build_val "01"