# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
PPSMC_Result: TypeAlias = ctypes.c_uint32
PPSMC_MSG: TypeAlias = ctypes.c_uint32
FEATURE_LIST_e: dict[int, str] = {(FEATURE_DATA_CALCULATION:=0): 'FEATURE_DATA_CALCULATION', (FEATURE_DPM_FCLK:=1): 'FEATURE_DPM_FCLK', (FEATURE_DPM_GFXCLK:=2): 'FEATURE_DPM_GFXCLK', (FEATURE_DPM_LCLK:=3): 'FEATURE_DPM_LCLK', (FEATURE_DPM_SOCCLK:=4): 'FEATURE_DPM_SOCCLK', (FEATURE_DPM_UCLK:=5): 'FEATURE_DPM_UCLK', (FEATURE_DPM_VCN:=6): 'FEATURE_DPM_VCN', (FEATURE_DPM_XGMI:=7): 'FEATURE_DPM_XGMI', (FEATURE_DS_FCLK:=8): 'FEATURE_DS_FCLK', (FEATURE_DS_GFXCLK:=9): 'FEATURE_DS_GFXCLK', (FEATURE_DS_LCLK:=10): 'FEATURE_DS_LCLK', (FEATURE_DS_MP0CLK:=11): 'FEATURE_DS_MP0CLK', (FEATURE_DS_MP1CLK:=12): 'FEATURE_DS_MP1CLK', (FEATURE_DS_MPIOCLK:=13): 'FEATURE_DS_MPIOCLK', (FEATURE_DS_SOCCLK:=14): 'FEATURE_DS_SOCCLK', (FEATURE_DS_VCN:=15): 'FEATURE_DS_VCN', (FEATURE_APCC_DFLL:=16): 'FEATURE_APCC_DFLL', (FEATURE_APCC_PLUS:=17): 'FEATURE_APCC_PLUS', (FEATURE_PPT:=18): 'FEATURE_PPT', (FEATURE_TDC:=19): 'FEATURE_TDC', (FEATURE_THERMAL:=20): 'FEATURE_THERMAL', (FEATURE_SOC_PCC:=21): 'FEATURE_SOC_PCC', (FEATURE_PROCHOT:=22): 'FEATURE_PROCHOT', (FEATURE_FDD_AID_HBM:=23): 'FEATURE_FDD_AID_HBM', (FEATURE_FDD_AID_SOC:=24): 'FEATURE_FDD_AID_SOC', (FEATURE_FDD_XCD_EDC:=25): 'FEATURE_FDD_XCD_EDC', (FEATURE_FDD_XCD_XVMIN:=26): 'FEATURE_FDD_XCD_XVMIN', (FEATURE_FW_CTF:=27): 'FEATURE_FW_CTF', (FEATURE_SMU_CG:=28): 'FEATURE_SMU_CG', (FEATURE_PSI7:=29): 'FEATURE_PSI7', (FEATURE_XGMI_PER_LINK_PWR_DOWN:=30): 'FEATURE_XGMI_PER_LINK_PWR_DOWN', (FEATURE_SOC_DC_RTC:=31): 'FEATURE_SOC_DC_RTC', (FEATURE_GFX_DC_RTC:=32): 'FEATURE_GFX_DC_RTC', (FEATURE_DVM_MIN_PSM:=33): 'FEATURE_DVM_MIN_PSM', (FEATURE_PRC:=34): 'FEATURE_PRC', (FEATURE_PSM_SQ_THROTTLER:=35): 'FEATURE_PSM_SQ_THROTTLER', (FEATURE_PIT:=36): 'FEATURE_PIT', (FEATURE_DVO:=37): 'FEATURE_DVO', (FEATURE_XVMINORPSM_CLKSTOP_DS:=38): 'FEATURE_XVMINORPSM_CLKSTOP_DS', (FEATURE_GLOBAL_DPM:=39): 'FEATURE_GLOBAL_DPM', (FEATURE_HROM_EN:=40): 'FEATURE_HROM_EN', (NUM_FEATURES:=41): 'NUM_FEATURES'}
PCIE_LINK_SPEED_INDEX_TABLE_e: dict[int, str] = {(PCIE_LINK_SPEED_INDEX_TABLE_RESERVED:=0): 'PCIE_LINK_SPEED_INDEX_TABLE_RESERVED', (PCIE_LINK_SPEED_INDEX_TABLE_GEN1:=1): 'PCIE_LINK_SPEED_INDEX_TABLE_GEN1', (PCIE_LINK_SPEED_INDEX_TABLE_GEN2:=2): 'PCIE_LINK_SPEED_INDEX_TABLE_GEN2', (PCIE_LINK_SPEED_INDEX_TABLE_GEN3:=3): 'PCIE_LINK_SPEED_INDEX_TABLE_GEN3', (PCIE_LINK_SPEED_INDEX_TABLE_GEN4:=4): 'PCIE_LINK_SPEED_INDEX_TABLE_GEN4', (PCIE_LINK_SPEED_INDEX_TABLE_GEN5:=5): 'PCIE_LINK_SPEED_INDEX_TABLE_GEN5', (PCIE_LINK_SPEED_INDEX_TABLE_COUNT:=6): 'PCIE_LINK_SPEED_INDEX_TABLE_COUNT'}
GFX_GUARDBAND_OFFSET_e: dict[int, str] = {(GFX_GUARDBAND_OFFSET_0:=0): 'GFX_GUARDBAND_OFFSET_0', (GFX_GUARDBAND_OFFSET_1:=1): 'GFX_GUARDBAND_OFFSET_1', (GFX_GUARDBAND_OFFSET_2:=2): 'GFX_GUARDBAND_OFFSET_2', (GFX_GUARDBAND_OFFSET_3:=3): 'GFX_GUARDBAND_OFFSET_3', (GFX_GUARDBAND_OFFSET_4:=4): 'GFX_GUARDBAND_OFFSET_4', (GFX_GUARDBAND_OFFSET_5:=5): 'GFX_GUARDBAND_OFFSET_5', (GFX_GUARDBAND_OFFSET_6:=6): 'GFX_GUARDBAND_OFFSET_6', (GFX_GUARDBAND_OFFSET_7:=7): 'GFX_GUARDBAND_OFFSET_7', (GFX_GUARDBAND_OFFSET_COUNT:=8): 'GFX_GUARDBAND_OFFSET_COUNT'}
GFX_DVM_MARGIN_e: dict[int, str] = {(GFX_DVM_MARGINHI_0:=0): 'GFX_DVM_MARGINHI_0', (GFX_DVM_MARGINHI_1:=1): 'GFX_DVM_MARGINHI_1', (GFX_DVM_MARGINHI_2:=2): 'GFX_DVM_MARGINHI_2', (GFX_DVM_MARGINHI_3:=3): 'GFX_DVM_MARGINHI_3', (GFX_DVM_MARGINHI_4:=4): 'GFX_DVM_MARGINHI_4', (GFX_DVM_MARGINHI_5:=5): 'GFX_DVM_MARGINHI_5', (GFX_DVM_MARGINHI_6:=6): 'GFX_DVM_MARGINHI_6', (GFX_DVM_MARGINHI_7:=7): 'GFX_DVM_MARGINHI_7', (GFX_DVM_MARGINLO_0:=8): 'GFX_DVM_MARGINLO_0', (GFX_DVM_MARGINLO_1:=9): 'GFX_DVM_MARGINLO_1', (GFX_DVM_MARGINLO_2:=10): 'GFX_DVM_MARGINLO_2', (GFX_DVM_MARGINLO_3:=11): 'GFX_DVM_MARGINLO_3', (GFX_DVM_MARGINLO_4:=12): 'GFX_DVM_MARGINLO_4', (GFX_DVM_MARGINLO_5:=13): 'GFX_DVM_MARGINLO_5', (GFX_DVM_MARGINLO_6:=14): 'GFX_DVM_MARGINLO_6', (GFX_DVM_MARGINLO_7:=15): 'GFX_DVM_MARGINLO_7', (GFX_DVM_MARGIN_COUNT:=16): 'GFX_DVM_MARGIN_COUNT'}
SYSTEM_TEMP_e: dict[int, str] = {(SYSTEM_TEMP_UBB_FPGA:=0): 'SYSTEM_TEMP_UBB_FPGA', (SYSTEM_TEMP_UBB_FRONT:=1): 'SYSTEM_TEMP_UBB_FRONT', (SYSTEM_TEMP_UBB_BACK:=2): 'SYSTEM_TEMP_UBB_BACK', (SYSTEM_TEMP_UBB_OAM7:=3): 'SYSTEM_TEMP_UBB_OAM7', (SYSTEM_TEMP_UBB_IBC:=4): 'SYSTEM_TEMP_UBB_IBC', (SYSTEM_TEMP_UBB_UFPGA:=5): 'SYSTEM_TEMP_UBB_UFPGA', (SYSTEM_TEMP_UBB_OAM1:=6): 'SYSTEM_TEMP_UBB_OAM1', (SYSTEM_TEMP_OAM_0_1_HSC:=7): 'SYSTEM_TEMP_OAM_0_1_HSC', (SYSTEM_TEMP_OAM_2_3_HSC:=8): 'SYSTEM_TEMP_OAM_2_3_HSC', (SYSTEM_TEMP_OAM_4_5_HSC:=9): 'SYSTEM_TEMP_OAM_4_5_HSC', (SYSTEM_TEMP_OAM_6_7_HSC:=10): 'SYSTEM_TEMP_OAM_6_7_HSC', (SYSTEM_TEMP_UBB_FPGA_0V72_VR:=11): 'SYSTEM_TEMP_UBB_FPGA_0V72_VR', (SYSTEM_TEMP_UBB_FPGA_3V3_VR:=12): 'SYSTEM_TEMP_UBB_FPGA_3V3_VR', (SYSTEM_TEMP_RETIMER_0_1_2_3_1V2_VR:=13): 'SYSTEM_TEMP_RETIMER_0_1_2_3_1V2_VR', (SYSTEM_TEMP_RETIMER_4_5_6_7_1V2_VR:=14): 'SYSTEM_TEMP_RETIMER_4_5_6_7_1V2_VR', (SYSTEM_TEMP_RETIMER_0_1_0V9_VR:=15): 'SYSTEM_TEMP_RETIMER_0_1_0V9_VR', (SYSTEM_TEMP_RETIMER_4_5_0V9_VR:=16): 'SYSTEM_TEMP_RETIMER_4_5_0V9_VR', (SYSTEM_TEMP_RETIMER_2_3_0V9_VR:=17): 'SYSTEM_TEMP_RETIMER_2_3_0V9_VR', (SYSTEM_TEMP_RETIMER_6_7_0V9_VR:=18): 'SYSTEM_TEMP_RETIMER_6_7_0V9_VR', (SYSTEM_TEMP_OAM_0_1_2_3_3V3_VR:=19): 'SYSTEM_TEMP_OAM_0_1_2_3_3V3_VR', (SYSTEM_TEMP_OAM_4_5_6_7_3V3_VR:=20): 'SYSTEM_TEMP_OAM_4_5_6_7_3V3_VR', (SYSTEM_TEMP_IBC_HSC:=21): 'SYSTEM_TEMP_IBC_HSC', (SYSTEM_TEMP_IBC:=22): 'SYSTEM_TEMP_IBC', (SYSTEM_TEMP_MAX_ENTRIES:=32): 'SYSTEM_TEMP_MAX_ENTRIES'}
NODE_TEMP_e: dict[int, str] = {(NODE_TEMP_RETIMER:=0): 'NODE_TEMP_RETIMER', (NODE_TEMP_IBC_TEMP:=1): 'NODE_TEMP_IBC_TEMP', (NODE_TEMP_IBC_2_TEMP:=2): 'NODE_TEMP_IBC_2_TEMP', (NODE_TEMP_VDD18_VR_TEMP:=3): 'NODE_TEMP_VDD18_VR_TEMP', (NODE_TEMP_04_HBM_B_VR_TEMP:=4): 'NODE_TEMP_04_HBM_B_VR_TEMP', (NODE_TEMP_04_HBM_D_VR_TEMP:=5): 'NODE_TEMP_04_HBM_D_VR_TEMP', (NODE_TEMP_MAX_TEMP_ENTRIES:=12): 'NODE_TEMP_MAX_TEMP_ENTRIES'}
SVI_TEMP_e: dict[int, str] = {(SVI_VDDCR_VDD0_TEMP:=0): 'SVI_VDDCR_VDD0_TEMP', (SVI_VDDCR_VDD1_TEMP:=1): 'SVI_VDDCR_VDD1_TEMP', (SVI_VDDCR_VDD2_TEMP:=2): 'SVI_VDDCR_VDD2_TEMP', (SVI_VDDCR_VDD3_TEMP:=3): 'SVI_VDDCR_VDD3_TEMP', (SVI_VDDCR_SOC_A_TEMP:=4): 'SVI_VDDCR_SOC_A_TEMP', (SVI_VDDCR_SOC_C_TEMP:=5): 'SVI_VDDCR_SOC_C_TEMP', (SVI_VDDCR_SOCIO_A_TEMP:=6): 'SVI_VDDCR_SOCIO_A_TEMP', (SVI_VDDCR_SOCIO_C_TEMP:=7): 'SVI_VDDCR_SOCIO_C_TEMP', (SVI_VDD_085_HBM_TEMP:=8): 'SVI_VDD_085_HBM_TEMP', (SVI_VDDCR_11_HBM_B_TEMP:=9): 'SVI_VDDCR_11_HBM_B_TEMP', (SVI_VDDCR_11_HBM_D_TEMP:=10): 'SVI_VDDCR_11_HBM_D_TEMP', (SVI_VDD_USR_TEMP:=11): 'SVI_VDD_USR_TEMP', (SVI_VDDIO_11_E32_TEMP:=12): 'SVI_VDDIO_11_E32_TEMP', (SVI_MAX_TEMP_ENTRIES:=13): 'SVI_MAX_TEMP_ENTRIES'}
@c.record
class MetricsTable_t(c.Struct):
  SIZE = 1284
  AccumulationCounter: int
  MaxSocketTemperature: int
  MaxVrTemperature: int
  MaxHbmTemperature: int
  MaxSocketTemperatureAcc: int
  MaxVrTemperatureAcc: int
  MaxHbmTemperatureAcc: int
  SocketPowerLimit: int
  SocketPower: int
  Timestamp: int
  SocketEnergyAcc: int
  XcdEnergyAcc: int
  AidEnergyAcc: int
  HbmEnergyAcc: int
  GfxclkFrequencyLimit: int
  FclkFrequency: int
  UclkFrequency: int
  SocclkFrequency: c.Array[ctypes.c_uint32, Literal[4]]
  VclkFrequency: c.Array[ctypes.c_uint32, Literal[4]]
  DclkFrequency: c.Array[ctypes.c_uint32, Literal[4]]
  LclkFrequency: c.Array[ctypes.c_uint32, Literal[4]]
  GfxclkFrequencyAcc: c.Array[ctypes.c_uint64, Literal[8]]
  MaxLclkDpmRange: int
  MinLclkDpmRange: int
  XgmiWidth: int
  XgmiBitrate: int
  XgmiReadBandwidthAcc: c.Array[ctypes.c_uint64, Literal[8]]
  XgmiWriteBandwidthAcc: c.Array[ctypes.c_uint64, Literal[8]]
  SocketGfxBusy: int
  DramBandwidthUtilization: int
  SocketGfxBusyAcc: int
  DramBandwidthAcc: int
  MaxDramBandwidth: int
  DramBandwidthUtilizationAcc: int
  PcieBandwidthAcc: c.Array[ctypes.c_uint64, Literal[4]]
  ProchotResidencyAcc: int
  PptResidencyAcc: int
  SocketThmResidencyAcc: int
  VrThmResidencyAcc: int
  HbmThmResidencyAcc: int
  GfxLockXCDMak: int
  GfxclkFrequency: c.Array[ctypes.c_uint32, Literal[8]]
  XgmiReadDataSizeAcc: c.Array[ctypes.c_uint64, Literal[8]]
  XgmiWriteDataSizeAcc: c.Array[ctypes.c_uint64, Literal[8]]
  PcieBandwidth: c.Array[ctypes.c_uint32, Literal[4]]
  PCIeL0ToRecoveryCountAcc: int
  PCIenReplayAAcc: int
  PCIenReplayARolloverCountAcc: int
  PCIeNAKSentCountAcc: int
  PCIeNAKReceivedCountAcc: int
  VcnBusy: c.Array[ctypes.c_uint32, Literal[4]]
  JpegBusy: c.Array[ctypes.c_uint32, Literal[40]]
  PCIeLinkSpeed: int
  PCIeLinkWidth: int
  GfxBusy: c.Array[ctypes.c_uint32, Literal[8]]
  GfxBusyAcc: c.Array[ctypes.c_uint64, Literal[8]]
  PCIeOtherEndRecoveryAcc: int
  GfxclkBelowHostLimitPptAcc: c.Array[ctypes.c_uint64, Literal[8]]
  GfxclkBelowHostLimitThmAcc: c.Array[ctypes.c_uint64, Literal[8]]
  GfxclkBelowHostLimitTotalAcc: c.Array[ctypes.c_uint64, Literal[8]]
  GfxclkLowUtilizationAcc: c.Array[ctypes.c_uint64, Literal[8]]
  AidTemperature: c.Array[ctypes.c_uint32, Literal[4]]
  XcdTemperature: c.Array[ctypes.c_uint32, Literal[8]]
  HbmTemperature: c.Array[ctypes.c_uint32, Literal[8]]
uint64_t: TypeAlias = ctypes.c_uint64
uint32_t: TypeAlias = ctypes.c_uint32
MetricsTable_t.register_fields([('AccumulationCounter', uint64_t, 0), ('MaxSocketTemperature', uint32_t, 8), ('MaxVrTemperature', uint32_t, 12), ('MaxHbmTemperature', uint32_t, 16), ('MaxSocketTemperatureAcc', uint64_t, 20), ('MaxVrTemperatureAcc', uint64_t, 28), ('MaxHbmTemperatureAcc', uint64_t, 36), ('SocketPowerLimit', uint32_t, 44), ('SocketPower', uint32_t, 48), ('Timestamp', uint64_t, 52), ('SocketEnergyAcc', uint64_t, 60), ('XcdEnergyAcc', uint64_t, 68), ('AidEnergyAcc', uint64_t, 76), ('HbmEnergyAcc', uint64_t, 84), ('GfxclkFrequencyLimit', uint32_t, 92), ('FclkFrequency', uint32_t, 96), ('UclkFrequency', uint32_t, 100), ('SocclkFrequency', c.Array[uint32_t, Literal[4]], 104), ('VclkFrequency', c.Array[uint32_t, Literal[4]], 120), ('DclkFrequency', c.Array[uint32_t, Literal[4]], 136), ('LclkFrequency', c.Array[uint32_t, Literal[4]], 152), ('GfxclkFrequencyAcc', c.Array[uint64_t, Literal[8]], 168), ('MaxLclkDpmRange', uint32_t, 232), ('MinLclkDpmRange', uint32_t, 236), ('XgmiWidth', uint32_t, 240), ('XgmiBitrate', uint32_t, 244), ('XgmiReadBandwidthAcc', c.Array[uint64_t, Literal[8]], 248), ('XgmiWriteBandwidthAcc', c.Array[uint64_t, Literal[8]], 312), ('SocketGfxBusy', uint32_t, 376), ('DramBandwidthUtilization', uint32_t, 380), ('SocketGfxBusyAcc', uint64_t, 384), ('DramBandwidthAcc', uint64_t, 392), ('MaxDramBandwidth', uint32_t, 400), ('DramBandwidthUtilizationAcc', uint64_t, 404), ('PcieBandwidthAcc', c.Array[uint64_t, Literal[4]], 412), ('ProchotResidencyAcc', uint32_t, 444), ('PptResidencyAcc', uint32_t, 448), ('SocketThmResidencyAcc', uint32_t, 452), ('VrThmResidencyAcc', uint32_t, 456), ('HbmThmResidencyAcc', uint32_t, 460), ('GfxLockXCDMak', uint32_t, 464), ('GfxclkFrequency', c.Array[uint32_t, Literal[8]], 468), ('XgmiReadDataSizeAcc', c.Array[uint64_t, Literal[8]], 500), ('XgmiWriteDataSizeAcc', c.Array[uint64_t, Literal[8]], 564), ('PcieBandwidth', c.Array[uint32_t, Literal[4]], 628), ('PCIeL0ToRecoveryCountAcc', uint32_t, 644), ('PCIenReplayAAcc', uint32_t, 648), ('PCIenReplayARolloverCountAcc', uint32_t, 652), ('PCIeNAKSentCountAcc', uint32_t, 656), ('PCIeNAKReceivedCountAcc', uint32_t, 660), ('VcnBusy', c.Array[uint32_t, Literal[4]], 664), ('JpegBusy', c.Array[uint32_t, Literal[40]], 680), ('PCIeLinkSpeed', uint32_t, 840), ('PCIeLinkWidth', uint32_t, 844), ('GfxBusy', c.Array[uint32_t, Literal[8]], 848), ('GfxBusyAcc', c.Array[uint64_t, Literal[8]], 880), ('PCIeOtherEndRecoveryAcc', uint32_t, 944), ('GfxclkBelowHostLimitPptAcc', c.Array[uint64_t, Literal[8]], 948), ('GfxclkBelowHostLimitThmAcc', c.Array[uint64_t, Literal[8]], 1012), ('GfxclkBelowHostLimitTotalAcc', c.Array[uint64_t, Literal[8]], 1076), ('GfxclkLowUtilizationAcc', c.Array[uint64_t, Literal[8]], 1140), ('AidTemperature', c.Array[uint32_t, Literal[4]], 1204), ('XcdTemperature', c.Array[uint32_t, Literal[8]], 1220), ('HbmTemperature', c.Array[uint32_t, Literal[8]], 1252)])
@c.record
class SystemMetricsTable_t(c.Struct):
  SIZE = 152
  AccumulationCounter: int
  LabelVersion: int
  NodeIdentifier: int
  SystemTemperatures: c.Array[ctypes.c_int16, Literal[32]]
  NodeTemperatures: c.Array[ctypes.c_int16, Literal[12]]
  VrTemperatures: c.Array[ctypes.c_int16, Literal[13]]
  spare: c.Array[ctypes.c_int16, Literal[7]]
  NodePowerLimit: int
  NodePower: int
  GlobalPPTResidencyAcc: int
uint16_t: TypeAlias = ctypes.c_uint16
int16_t: TypeAlias = ctypes.c_int16
SystemMetricsTable_t.register_fields([('AccumulationCounter', uint64_t, 0), ('LabelVersion', uint16_t, 8), ('NodeIdentifier', uint16_t, 10), ('SystemTemperatures', c.Array[int16_t, Literal[32]], 12), ('NodeTemperatures', c.Array[int16_t, Literal[12]], 76), ('VrTemperatures', c.Array[int16_t, Literal[13]], 100), ('spare', c.Array[int16_t, Literal[7]], 126), ('NodePowerLimit', uint32_t, 140), ('NodePower', uint32_t, 144), ('GlobalPPTResidencyAcc', uint32_t, 148)])
@c.record
class VfMetricsTable_t(c.Struct):
  SIZE = 56
  AccumulationCounter: int
  InstGfxclk_TargFreq: int
  AccGfxclk_TargFreq: int
  AccGfxRsmuDpm_Busy: int
  AccGfxclkBelowHostLimitPpt: int
  AccGfxclkBelowHostLimitThm: int
  AccGfxclkBelowHostLimitTotal: int
  AccGfxclkLowUtilization: int
VfMetricsTable_t.register_fields([('AccumulationCounter', uint32_t, 0), ('InstGfxclk_TargFreq', uint32_t, 4), ('AccGfxclk_TargFreq', uint64_t, 8), ('AccGfxRsmuDpm_Busy', uint64_t, 16), ('AccGfxclkBelowHostLimitPpt', uint64_t, 24), ('AccGfxclkBelowHostLimitThm', uint64_t, 32), ('AccGfxclkBelowHostLimitTotal', uint64_t, 40), ('AccGfxclkLowUtilization', uint64_t, 48)])
@c.record
class FRUProductInfo_t(c.Struct):
  SIZE = 168
  ModelNumber: c.Array[ctypes.c_ubyte, Literal[20]]
  Name: c.Array[ctypes.c_ubyte, Literal[64]]
  Serial: c.Array[ctypes.c_ubyte, Literal[20]]
  ManufacturerName: c.Array[ctypes.c_ubyte, Literal[32]]
  FruId: c.Array[ctypes.c_ubyte, Literal[32]]
uint8_t: TypeAlias = ctypes.c_ubyte
FRUProductInfo_t.register_fields([('ModelNumber', c.Array[uint8_t, Literal[20]], 0), ('Name', c.Array[uint8_t, Literal[64]], 20), ('Serial', c.Array[uint8_t, Literal[20]], 84), ('ManufacturerName', c.Array[uint8_t, Literal[32]], 104), ('FruId', c.Array[uint8_t, Literal[32]], 136)])
@c.record
class StaticMetricsTable_t(c.Struct):
  SIZE = 408
  ProductInfo: FRUProductInfo_t
  MaxSocketPowerLimit: int
  MaxGfxclkFrequency: int
  MinGfxclkFrequency: int
  FclkFrequencyTable: c.Array[ctypes.c_uint32, Literal[4]]
  UclkFrequencyTable: c.Array[ctypes.c_uint32, Literal[4]]
  SocclkFrequencyTable: c.Array[ctypes.c_uint32, Literal[4]]
  VclkFrequencyTable: c.Array[ctypes.c_uint32, Literal[4]]
  DclkFrequencyTable: c.Array[ctypes.c_uint32, Literal[4]]
  LclkFrequencyTable: c.Array[ctypes.c_uint32, Literal[4]]
  PublicSerialNumber_AID: c.Array[ctypes.c_uint64, Literal[4]]
  PublicSerialNumber_XCD: c.Array[ctypes.c_uint64, Literal[8]]
  MaxXgmiWidth: int
  MaxXgmiBitrate: int
  InputTelemetryVoltageInmV: int
  pldmVersion: c.Array[ctypes.c_uint32, Literal[2]]
  MaxNodePowerLimit: int
  PPT1Max: int
  PPT1Min: int
  PPT1Default: int
StaticMetricsTable_t.register_fields([('ProductInfo', FRUProductInfo_t, 0), ('MaxSocketPowerLimit', uint32_t, 168), ('MaxGfxclkFrequency', uint32_t, 172), ('MinGfxclkFrequency', uint32_t, 176), ('FclkFrequencyTable', c.Array[uint32_t, Literal[4]], 180), ('UclkFrequencyTable', c.Array[uint32_t, Literal[4]], 196), ('SocclkFrequencyTable', c.Array[uint32_t, Literal[4]], 212), ('VclkFrequencyTable', c.Array[uint32_t, Literal[4]], 228), ('DclkFrequencyTable', c.Array[uint32_t, Literal[4]], 244), ('LclkFrequencyTable', c.Array[uint32_t, Literal[4]], 260), ('PublicSerialNumber_AID', c.Array[uint64_t, Literal[4]], 276), ('PublicSerialNumber_XCD', c.Array[uint64_t, Literal[8]], 308), ('MaxXgmiWidth', uint32_t, 372), ('MaxXgmiBitrate', uint32_t, 376), ('InputTelemetryVoltageInmV', uint32_t, 380), ('pldmVersion', c.Array[uint32_t, Literal[2]], 384), ('MaxNodePowerLimit', uint32_t, 392), ('PPT1Max', uint32_t, 396), ('PPT1Min', uint32_t, 400), ('PPT1Default', uint32_t, 404)])
I2cControllerPort_e: dict[int, str] = {(I2C_CONTROLLER_PORT_0:=0): 'I2C_CONTROLLER_PORT_0', (I2C_CONTROLLER_PORT_1:=1): 'I2C_CONTROLLER_PORT_1', (I2C_CONTROLLER_PORT_COUNT:=2): 'I2C_CONTROLLER_PORT_COUNT'}
I2cSpeed_e: dict[int, str] = {(UNSUPPORTED_1:=0): 'UNSUPPORTED_1', (I2C_SPEED_STANDARD_100K:=1): 'I2C_SPEED_STANDARD_100K', (I2C_SPEED_FAST_400K:=2): 'I2C_SPEED_FAST_400K', (I2C_SPEED_FAST_PLUS_1M:=3): 'I2C_SPEED_FAST_PLUS_1M', (UNSUPPORTED_2:=4): 'UNSUPPORTED_2', (UNSUPPORTED_3:=5): 'UNSUPPORTED_3', (I2C_SPEED_COUNT:=6): 'I2C_SPEED_COUNT'}
I2cCmdType_e: dict[int, str] = {(I2C_CMD_READ:=0): 'I2C_CMD_READ', (I2C_CMD_WRITE:=1): 'I2C_CMD_WRITE', (I2C_CMD_COUNT:=2): 'I2C_CMD_COUNT'}
ERR_CODE_e: dict[int, str] = {(CODE_DAGB0:=0): 'CODE_DAGB0', (CODE_EA0:=5): 'CODE_EA0', (CODE_UTCL2_ROUTER:=10): 'CODE_UTCL2_ROUTER', (CODE_VML2:=11): 'CODE_VML2', (CODE_VML2_WALKER:=12): 'CODE_VML2_WALKER', (CODE_MMCANE:=13): 'CODE_MMCANE', (CODE_VIDD:=14): 'CODE_VIDD', (CODE_VIDV:=15): 'CODE_VIDV', (CODE_JPEG0S:=16): 'CODE_JPEG0S', (CODE_JPEG0D:=17): 'CODE_JPEG0D', (CODE_JPEG1S:=18): 'CODE_JPEG1S', (CODE_JPEG1D:=19): 'CODE_JPEG1D', (CODE_JPEG2S:=20): 'CODE_JPEG2S', (CODE_JPEG2D:=21): 'CODE_JPEG2D', (CODE_JPEG3S:=22): 'CODE_JPEG3S', (CODE_JPEG3D:=23): 'CODE_JPEG3D', (CODE_JPEG4S:=24): 'CODE_JPEG4S', (CODE_JPEG4D:=25): 'CODE_JPEG4D', (CODE_JPEG5S:=26): 'CODE_JPEG5S', (CODE_JPEG5D:=27): 'CODE_JPEG5D', (CODE_JPEG6S:=28): 'CODE_JPEG6S', (CODE_JPEG6D:=29): 'CODE_JPEG6D', (CODE_JPEG7S:=30): 'CODE_JPEG7S', (CODE_JPEG7D:=31): 'CODE_JPEG7D', (CODE_MMSCHD:=32): 'CODE_MMSCHD', (CODE_SDMA0:=33): 'CODE_SDMA0', (CODE_SDMA1:=34): 'CODE_SDMA1', (CODE_SDMA2:=35): 'CODE_SDMA2', (CODE_SDMA3:=36): 'CODE_SDMA3', (CODE_HDP:=37): 'CODE_HDP', (CODE_ATHUB:=38): 'CODE_ATHUB', (CODE_IH:=39): 'CODE_IH', (CODE_XHUB_POISON:=40): 'CODE_XHUB_POISON', (CODE_SMN_SLVERR:=40): 'CODE_SMN_SLVERR', (CODE_WDT:=41): 'CODE_WDT', (CODE_UNKNOWN:=42): 'CODE_UNKNOWN', (CODE_COUNT:=43): 'CODE_COUNT'}
GC_ERROR_CODE_e: dict[int, str] = {(SH_FED_CODE:=0): 'SH_FED_CODE', (GCEA_CODE:=1): 'GCEA_CODE', (SQ_CODE:=2): 'SQ_CODE', (LDS_CODE:=3): 'LDS_CODE', (GDS_CODE:=4): 'GDS_CODE', (SP0_CODE:=5): 'SP0_CODE', (SP1_CODE:=6): 'SP1_CODE', (TCC_CODE:=7): 'TCC_CODE', (TCA_CODE:=8): 'TCA_CODE', (TCX_CODE:=9): 'TCX_CODE', (CPC_CODE:=10): 'CPC_CODE', (CPF_CODE:=11): 'CPF_CODE', (CPG_CODE:=12): 'CPG_CODE', (SPI_CODE:=13): 'SPI_CODE', (RLC_CODE:=14): 'RLC_CODE', (SQC_CODE:=15): 'SQC_CODE', (TA_CODE:=16): 'TA_CODE', (TD_CODE:=17): 'TD_CODE', (TCP_CODE:=18): 'TCP_CODE', (TCI_CODE:=19): 'TCI_CODE', (GC_ROUTER_CODE:=20): 'GC_ROUTER_CODE', (VML2_CODE:=21): 'VML2_CODE', (VML2_WALKER_CODE:=22): 'VML2_WALKER_CODE', (ATCL2_CODE:=23): 'ATCL2_CODE', (GC_CANE_CODE:=24): 'GC_CANE_CODE', (MP5_CODE_SMN_SLVERR:=40): 'MP5_CODE_SMN_SLVERR', (MP5_CODE_UNKNOWN:=42): 'MP5_CODE_UNKNOWN'}
@c.record
class SwI2cCmd_t(c.Struct):
  SIZE = 2
  ReadWriteData: int
  CmdConfig: int
SwI2cCmd_t.register_fields([('ReadWriteData', uint8_t, 0), ('CmdConfig', uint8_t, 1)])
@c.record
class SwI2cRequest_t(c.Struct):
  SIZE = 52
  I2CcontrollerPort: int
  I2CSpeed: int
  SlaveAddress: int
  NumCmds: int
  SwI2cCmds: c.Array[SwI2cCmd_t, Literal[24]]
SwI2cRequest_t.register_fields([('I2CcontrollerPort', uint8_t, 0), ('I2CSpeed', uint8_t, 1), ('SlaveAddress', uint8_t, 2), ('NumCmds', uint8_t, 3), ('SwI2cCmds', c.Array[SwI2cCmd_t, Literal[24]], 4)])
@c.record
class SwI2cRequestExternal_t(c.Struct):
  SIZE = 116
  SwI2cRequest: SwI2cRequest_t
  Spare: c.Array[ctypes.c_uint32, Literal[8]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
SwI2cRequestExternal_t.register_fields([('SwI2cRequest', SwI2cRequest_t, 0), ('Spare', c.Array[uint32_t, Literal[8]], 52), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 84)])
PPCLK_e: dict[int, str] = {(PPCLK_VCLK:=0): 'PPCLK_VCLK', (PPCLK_DCLK:=1): 'PPCLK_DCLK', (PPCLK_SOCCLK:=2): 'PPCLK_SOCCLK', (PPCLK_UCLK:=3): 'PPCLK_UCLK', (PPCLK_FCLK:=4): 'PPCLK_FCLK', (PPCLK_LCLK:=5): 'PPCLK_LCLK', (PPCLK_COUNT:=6): 'PPCLK_COUNT'}
GpioIntPolarity_e: dict[int, str] = {(GPIO_INT_POLARITY_ACTIVE_LOW:=0): 'GPIO_INT_POLARITY_ACTIVE_LOW', (GPIO_INT_POLARITY_ACTIVE_HIGH:=1): 'GPIO_INT_POLARITY_ACTIVE_HIGH'}
UCLK_DPM_MODE_e: dict[int, str] = {(UCLK_DPM_MODE_BANDWIDTH:=0): 'UCLK_DPM_MODE_BANDWIDTH', (UCLK_DPM_MODE_LATENCY:=1): 'UCLK_DPM_MODE_LATENCY'}
@c.record
class AvfsDebugTableAid_t(c.Struct):
  SIZE = 360
  avgPsmCount: c.Array[ctypes.c_uint16, Literal[30]]
  minPsmCount: c.Array[ctypes.c_uint16, Literal[30]]
  avgPsmVoltage: c.Array[ctypes.c_float, Literal[30]]
  minPsmVoltage: c.Array[ctypes.c_float, Literal[30]]
AvfsDebugTableAid_t.register_fields([('avgPsmCount', c.Array[uint16_t, Literal[30]], 0), ('minPsmCount', c.Array[uint16_t, Literal[30]], 60), ('avgPsmVoltage', c.Array[ctypes.c_float, Literal[30]], 120), ('minPsmVoltage', c.Array[ctypes.c_float, Literal[30]], 240)])
@c.record
class AvfsDebugTableXcd_t(c.Struct):
  SIZE = 360
  avgPsmCount: c.Array[ctypes.c_uint16, Literal[30]]
  minPsmCount: c.Array[ctypes.c_uint16, Literal[30]]
  avgPsmVoltage: c.Array[ctypes.c_float, Literal[30]]
  minPsmVoltage: c.Array[ctypes.c_float, Literal[30]]
AvfsDebugTableXcd_t.register_fields([('avgPsmCount', c.Array[uint16_t, Literal[30]], 0), ('minPsmCount', c.Array[uint16_t, Literal[30]], 60), ('avgPsmVoltage', c.Array[ctypes.c_float, Literal[30]], 120), ('minPsmVoltage', c.Array[ctypes.c_float, Literal[30]], 240)])
@c.record
class struct_smu_hw_power_state(c.Struct):
  SIZE = 4
  magic: int
struct_smu_hw_power_state.register_fields([('magic', ctypes.c_uint32, 0)])
class struct_smu_power_state(c.Struct): pass
enum_smu_state_ui_label: dict[int, str] = {(SMU_STATE_UI_LABEL_NONE:=0): 'SMU_STATE_UI_LABEL_NONE', (SMU_STATE_UI_LABEL_BATTERY:=1): 'SMU_STATE_UI_LABEL_BATTERY', (SMU_STATE_UI_TABEL_MIDDLE_LOW:=2): 'SMU_STATE_UI_TABEL_MIDDLE_LOW', (SMU_STATE_UI_LABEL_BALLANCED:=3): 'SMU_STATE_UI_LABEL_BALLANCED', (SMU_STATE_UI_LABEL_MIDDLE_HIGHT:=4): 'SMU_STATE_UI_LABEL_MIDDLE_HIGHT', (SMU_STATE_UI_LABEL_PERFORMANCE:=5): 'SMU_STATE_UI_LABEL_PERFORMANCE', (SMU_STATE_UI_LABEL_BACO:=6): 'SMU_STATE_UI_LABEL_BACO'}
enum_smu_state_classification_flag: dict[int, str] = {(SMU_STATE_CLASSIFICATION_FLAG_BOOT:=1): 'SMU_STATE_CLASSIFICATION_FLAG_BOOT', (SMU_STATE_CLASSIFICATION_FLAG_THERMAL:=2): 'SMU_STATE_CLASSIFICATION_FLAG_THERMAL', (SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE:=4): 'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE', (SMU_STATE_CLASSIFICATION_FLAG_RESET:=8): 'SMU_STATE_CLASSIFICATION_FLAG_RESET', (SMU_STATE_CLASSIFICATION_FLAG_FORCED:=16): 'SMU_STATE_CLASSIFICATION_FLAG_FORCED', (SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE:=32): 'SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE', (SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE:=64): 'SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE', (SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE:=128): 'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE', (SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE:=256): 'SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE', (SMU_STATE_CLASSIFICATION_FLAG_UVD:=512): 'SMU_STATE_CLASSIFICATION_FLAG_UVD', (SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW:=1024): 'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW', (SMU_STATE_CLASSIFICATION_FLAG_ACPI:=2048): 'SMU_STATE_CLASSIFICATION_FLAG_ACPI', (SMU_STATE_CLASSIFICATION_FLAG_HD2:=4096): 'SMU_STATE_CLASSIFICATION_FLAG_HD2', (SMU_STATE_CLASSIFICATION_FLAG_UVD_HD:=8192): 'SMU_STATE_CLASSIFICATION_FLAG_UVD_HD', (SMU_STATE_CLASSIFICATION_FLAG_UVD_SD:=16384): 'SMU_STATE_CLASSIFICATION_FLAG_UVD_SD', (SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE:=32768): 'SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE', (SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE:=65536): 'SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE', (SMU_STATE_CLASSIFICATION_FLAG_BACO:=131072): 'SMU_STATE_CLASSIFICATION_FLAG_BACO', (SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2:=262144): 'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2', (SMU_STATE_CLASSIFICATION_FLAG_ULV:=524288): 'SMU_STATE_CLASSIFICATION_FLAG_ULV', (SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC:=1048576): 'SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC'}
@c.record
class struct_smu_state_classification_block(c.Struct):
  SIZE = 16
  ui_label: int
  flags: int
  bios_index: int
  temporary_state: bool
  to_be_deleted: bool
struct_smu_state_classification_block.register_fields([('ui_label', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('bios_index', ctypes.c_int32, 8), ('temporary_state', ctypes.c_bool, 12), ('to_be_deleted', ctypes.c_bool, 13)])
@c.record
class struct_smu_state_pcie_block(c.Struct):
  SIZE = 4
  lanes: int
struct_smu_state_pcie_block.register_fields([('lanes', ctypes.c_uint32, 0)])
enum_smu_refreshrate_source: dict[int, str] = {(SMU_REFRESHRATE_SOURCE_EDID:=0): 'SMU_REFRESHRATE_SOURCE_EDID', (SMU_REFRESHRATE_SOURCE_EXPLICIT:=1): 'SMU_REFRESHRATE_SOURCE_EXPLICIT'}
@c.record
class struct_smu_state_display_block(c.Struct):
  SIZE = 20
  disable_frame_modulation: bool
  limit_refreshrate: bool
  refreshrate_source: int
  explicit_refreshrate: int
  edid_refreshrate_index: int
  enable_vari_bright: bool
struct_smu_state_display_block.register_fields([('disable_frame_modulation', ctypes.c_bool, 0), ('limit_refreshrate', ctypes.c_bool, 1), ('refreshrate_source', ctypes.c_uint32, 4), ('explicit_refreshrate', ctypes.c_int32, 8), ('edid_refreshrate_index', ctypes.c_int32, 12), ('enable_vari_bright', ctypes.c_bool, 16)])
@c.record
class struct_smu_state_memory_block(c.Struct):
  SIZE = 5
  dll_off: bool
  m3arb: int
  unused: c.Array[ctypes.c_ubyte, Literal[3]]
struct_smu_state_memory_block.register_fields([('dll_off', ctypes.c_bool, 0), ('m3arb', uint8_t, 1), ('unused', c.Array[uint8_t, Literal[3]], 2)])
@c.record
class struct_smu_state_software_algorithm_block(c.Struct):
  SIZE = 2
  disable_load_balancing: bool
  enable_sleep_for_timestamps: bool
struct_smu_state_software_algorithm_block.register_fields([('disable_load_balancing', ctypes.c_bool, 0), ('enable_sleep_for_timestamps', ctypes.c_bool, 1)])
@c.record
class struct_smu_temperature_range(c.Struct):
  SIZE = 44
  min: int
  max: int
  edge_emergency_max: int
  hotspot_min: int
  hotspot_crit_max: int
  hotspot_emergency_max: int
  mem_min: int
  mem_crit_max: int
  mem_emergency_max: int
  software_shutdown_temp: int
  software_shutdown_temp_offset: int
struct_smu_temperature_range.register_fields([('min', ctypes.c_int32, 0), ('max', ctypes.c_int32, 4), ('edge_emergency_max', ctypes.c_int32, 8), ('hotspot_min', ctypes.c_int32, 12), ('hotspot_crit_max', ctypes.c_int32, 16), ('hotspot_emergency_max', ctypes.c_int32, 20), ('mem_min', ctypes.c_int32, 24), ('mem_crit_max', ctypes.c_int32, 28), ('mem_emergency_max', ctypes.c_int32, 32), ('software_shutdown_temp', ctypes.c_int32, 36), ('software_shutdown_temp_offset', ctypes.c_int32, 40)])
@c.record
class struct_smu_state_validation_block(c.Struct):
  SIZE = 3
  single_display_only: bool
  disallow_on_dc: bool
  supported_power_levels: int
struct_smu_state_validation_block.register_fields([('single_display_only', ctypes.c_bool, 0), ('disallow_on_dc', ctypes.c_bool, 1), ('supported_power_levels', uint8_t, 2)])
@c.record
class struct_smu_uvd_clocks(c.Struct):
  SIZE = 8
  vclk: int
  dclk: int
struct_smu_uvd_clocks.register_fields([('vclk', uint32_t, 0), ('dclk', uint32_t, 4)])
enum_smu_power_src_type: dict[int, str] = {(SMU_POWER_SOURCE_AC:=0): 'SMU_POWER_SOURCE_AC', (SMU_POWER_SOURCE_DC:=1): 'SMU_POWER_SOURCE_DC', (SMU_POWER_SOURCE_COUNT:=2): 'SMU_POWER_SOURCE_COUNT'}
enum_smu_ppt_limit_type: dict[int, str] = {(SMU_DEFAULT_PPT_LIMIT:=0): 'SMU_DEFAULT_PPT_LIMIT', (SMU_FAST_PPT_LIMIT:=1): 'SMU_FAST_PPT_LIMIT'}
enum_smu_ppt_limit_level: dict[int, str] = {(SMU_PPT_LIMIT_MIN:=-1): 'SMU_PPT_LIMIT_MIN', (SMU_PPT_LIMIT_CURRENT:=0): 'SMU_PPT_LIMIT_CURRENT', (SMU_PPT_LIMIT_DEFAULT:=1): 'SMU_PPT_LIMIT_DEFAULT', (SMU_PPT_LIMIT_MAX:=2): 'SMU_PPT_LIMIT_MAX'}
enum_smu_memory_pool_size: dict[int, str] = {(SMU_MEMORY_POOL_SIZE_ZERO:=0): 'SMU_MEMORY_POOL_SIZE_ZERO', (SMU_MEMORY_POOL_SIZE_256_MB:=268435456): 'SMU_MEMORY_POOL_SIZE_256_MB', (SMU_MEMORY_POOL_SIZE_512_MB:=536870912): 'SMU_MEMORY_POOL_SIZE_512_MB', (SMU_MEMORY_POOL_SIZE_1_GB:=1073741824): 'SMU_MEMORY_POOL_SIZE_1_GB', (SMU_MEMORY_POOL_SIZE_2_GB:=2147483648): 'SMU_MEMORY_POOL_SIZE_2_GB'}
enum_smu_clk_type: dict[int, str] = {(SMU_GFXCLK:=0): 'SMU_GFXCLK', (SMU_VCLK:=1): 'SMU_VCLK', (SMU_DCLK:=2): 'SMU_DCLK', (SMU_VCLK1:=3): 'SMU_VCLK1', (SMU_DCLK1:=4): 'SMU_DCLK1', (SMU_ECLK:=5): 'SMU_ECLK', (SMU_SOCCLK:=6): 'SMU_SOCCLK', (SMU_UCLK:=7): 'SMU_UCLK', (SMU_DCEFCLK:=8): 'SMU_DCEFCLK', (SMU_DISPCLK:=9): 'SMU_DISPCLK', (SMU_PIXCLK:=10): 'SMU_PIXCLK', (SMU_PHYCLK:=11): 'SMU_PHYCLK', (SMU_FCLK:=12): 'SMU_FCLK', (SMU_SCLK:=13): 'SMU_SCLK', (SMU_MCLK:=14): 'SMU_MCLK', (SMU_PCIE:=15): 'SMU_PCIE', (SMU_LCLK:=16): 'SMU_LCLK', (SMU_OD_CCLK:=17): 'SMU_OD_CCLK', (SMU_OD_SCLK:=18): 'SMU_OD_SCLK', (SMU_OD_MCLK:=19): 'SMU_OD_MCLK', (SMU_OD_VDDC_CURVE:=20): 'SMU_OD_VDDC_CURVE', (SMU_OD_RANGE:=21): 'SMU_OD_RANGE', (SMU_OD_VDDGFX_OFFSET:=22): 'SMU_OD_VDDGFX_OFFSET', (SMU_OD_FAN_CURVE:=23): 'SMU_OD_FAN_CURVE', (SMU_OD_ACOUSTIC_LIMIT:=24): 'SMU_OD_ACOUSTIC_LIMIT', (SMU_OD_ACOUSTIC_TARGET:=25): 'SMU_OD_ACOUSTIC_TARGET', (SMU_OD_FAN_TARGET_TEMPERATURE:=26): 'SMU_OD_FAN_TARGET_TEMPERATURE', (SMU_OD_FAN_MINIMUM_PWM:=27): 'SMU_OD_FAN_MINIMUM_PWM', (SMU_CLK_COUNT:=28): 'SMU_CLK_COUNT'}
@c.record
class struct_smu_user_dpm_profile(c.Struct):
  SIZE = 140
  fan_mode: int
  power_limit: int
  fan_speed_pwm: int
  fan_speed_rpm: int
  flags: int
  user_od: int
  clk_mask: c.Array[ctypes.c_uint32, Literal[28]]
  clk_dependency: int
struct_smu_user_dpm_profile.register_fields([('fan_mode', uint32_t, 0), ('power_limit', uint32_t, 4), ('fan_speed_pwm', uint32_t, 8), ('fan_speed_rpm', uint32_t, 12), ('flags', uint32_t, 16), ('user_od', uint32_t, 20), ('clk_mask', c.Array[uint32_t, Literal[28]], 24), ('clk_dependency', uint32_t, 136)])
@c.record
class struct_smu_table(c.Struct):
  SIZE = 48
  size: int
  align: int
  domain: int
  mc_address: int
  cpu_addr: ctypes.c_void_p
  bo: c.POINTER[struct_amdgpu_bo]
  version: int
class struct_amdgpu_bo(c.Struct): pass
struct_smu_table.register_fields([('size', uint64_t, 0), ('align', uint32_t, 8), ('domain', uint8_t, 12), ('mc_address', uint64_t, 16), ('cpu_addr', ctypes.c_void_p, 24), ('bo', c.POINTER[struct_amdgpu_bo], 32), ('version', uint32_t, 40)])
enum_smu_perf_level_designation: dict[int, str] = {(PERF_LEVEL_ACTIVITY:=0): 'PERF_LEVEL_ACTIVITY', (PERF_LEVEL_POWER_CONTAINMENT:=1): 'PERF_LEVEL_POWER_CONTAINMENT'}
@c.record
class struct_smu_performance_level(c.Struct):
  SIZE = 24
  core_clock: int
  memory_clock: int
  vddc: int
  vddci: int
  non_local_mem_freq: int
  non_local_mem_width: int
struct_smu_performance_level.register_fields([('core_clock', uint32_t, 0), ('memory_clock', uint32_t, 4), ('vddc', uint32_t, 8), ('vddci', uint32_t, 12), ('non_local_mem_freq', uint32_t, 16), ('non_local_mem_width', uint32_t, 20)])
@c.record
class struct_smu_clock_info(c.Struct):
  SIZE = 24
  min_mem_clk: int
  max_mem_clk: int
  min_eng_clk: int
  max_eng_clk: int
  min_bus_bandwidth: int
  max_bus_bandwidth: int
struct_smu_clock_info.register_fields([('min_mem_clk', uint32_t, 0), ('max_mem_clk', uint32_t, 4), ('min_eng_clk', uint32_t, 8), ('max_eng_clk', uint32_t, 12), ('min_bus_bandwidth', uint32_t, 16), ('max_bus_bandwidth', uint32_t, 20)])
@c.record
class struct_smu_bios_boot_up_values(c.Struct):
  SIZE = 68
  revision: int
  gfxclk: int
  uclk: int
  socclk: int
  dcefclk: int
  eclk: int
  vclk: int
  dclk: int
  vddc: int
  vddci: int
  mvddc: int
  vdd_gfx: int
  cooling_id: int
  pp_table_id: int
  format_revision: int
  content_revision: int
  fclk: int
  lclk: int
  firmware_caps: int
struct_smu_bios_boot_up_values.register_fields([('revision', uint32_t, 0), ('gfxclk', uint32_t, 4), ('uclk', uint32_t, 8), ('socclk', uint32_t, 12), ('dcefclk', uint32_t, 16), ('eclk', uint32_t, 20), ('vclk', uint32_t, 24), ('dclk', uint32_t, 28), ('vddc', uint16_t, 32), ('vddci', uint16_t, 34), ('mvddc', uint16_t, 36), ('vdd_gfx', uint16_t, 38), ('cooling_id', uint8_t, 40), ('pp_table_id', uint32_t, 44), ('format_revision', uint32_t, 48), ('content_revision', uint32_t, 52), ('fclk', uint32_t, 56), ('lclk', uint32_t, 60), ('firmware_caps', uint32_t, 64)])
enum_smu_table_id: dict[int, str] = {(SMU_TABLE_PPTABLE:=0): 'SMU_TABLE_PPTABLE', (SMU_TABLE_WATERMARKS:=1): 'SMU_TABLE_WATERMARKS', (SMU_TABLE_CUSTOM_DPM:=2): 'SMU_TABLE_CUSTOM_DPM', (SMU_TABLE_DPMCLOCKS:=3): 'SMU_TABLE_DPMCLOCKS', (SMU_TABLE_AVFS:=4): 'SMU_TABLE_AVFS', (SMU_TABLE_AVFS_PSM_DEBUG:=5): 'SMU_TABLE_AVFS_PSM_DEBUG', (SMU_TABLE_AVFS_FUSE_OVERRIDE:=6): 'SMU_TABLE_AVFS_FUSE_OVERRIDE', (SMU_TABLE_PMSTATUSLOG:=7): 'SMU_TABLE_PMSTATUSLOG', (SMU_TABLE_SMU_METRICS:=8): 'SMU_TABLE_SMU_METRICS', (SMU_TABLE_DRIVER_SMU_CONFIG:=9): 'SMU_TABLE_DRIVER_SMU_CONFIG', (SMU_TABLE_ACTIVITY_MONITOR_COEFF:=10): 'SMU_TABLE_ACTIVITY_MONITOR_COEFF', (SMU_TABLE_OVERDRIVE:=11): 'SMU_TABLE_OVERDRIVE', (SMU_TABLE_I2C_COMMANDS:=12): 'SMU_TABLE_I2C_COMMANDS', (SMU_TABLE_PACE:=13): 'SMU_TABLE_PACE', (SMU_TABLE_ECCINFO:=14): 'SMU_TABLE_ECCINFO', (SMU_TABLE_COMBO_PPTABLE:=15): 'SMU_TABLE_COMBO_PPTABLE', (SMU_TABLE_WIFIBAND:=16): 'SMU_TABLE_WIFIBAND', (SMU_TABLE_COUNT:=17): 'SMU_TABLE_COUNT'}
PPSMC_Result_OK = 0x1
PPSMC_Result_Failed = 0xFF
PPSMC_Result_UnknownCmd = 0xFE
PPSMC_Result_CmdRejectedPrereq = 0xFD
PPSMC_Result_CmdRejectedBusy = 0xFC
PPSMC_MSG_TestMessage = 0x1
PPSMC_MSG_GetSmuVersion = 0x2
PPSMC_MSG_GfxDriverReset = 0x3
PPSMC_MSG_GetDriverIfVersion = 0x4
PPSMC_MSG_EnableAllSmuFeatures = 0x5
PPSMC_MSG_DisableAllSmuFeatures = 0x6
PPSMC_MSG_RequestI2cTransaction = 0x7
PPSMC_MSG_GetMetricsVersion = 0x8
PPSMC_MSG_GetMetricsTable = 0x9
PPSMC_MSG_GetEccInfoTable = 0xA
PPSMC_MSG_GetEnabledSmuFeaturesLow = 0xB
PPSMC_MSG_GetEnabledSmuFeaturesHigh = 0xC
PPSMC_MSG_SetDriverDramAddrHigh = 0xD
PPSMC_MSG_SetDriverDramAddrLow = 0xE
PPSMC_MSG_SetToolsDramAddrHigh = 0xF
PPSMC_MSG_SetToolsDramAddrLow = 0x10
PPSMC_MSG_SetSystemVirtualDramAddrHigh = 0x11
PPSMC_MSG_SetSystemVirtualDramAddrLow = 0x12
PPSMC_MSG_SetSoftMinByFreq = 0x13
PPSMC_MSG_SetSoftMaxByFreq = 0x14
PPSMC_MSG_GetMinDpmFreq = 0x15
PPSMC_MSG_GetMaxDpmFreq = 0x16
PPSMC_MSG_GetDpmFreqByIndex = 0x17
PPSMC_MSG_SetPptLimit = 0x18
PPSMC_MSG_GetPptLimit = 0x19
PPSMC_MSG_DramLogSetDramAddrHigh = 0x1A
PPSMC_MSG_DramLogSetDramAddrLow = 0x1B
PPSMC_MSG_DramLogSetDramSize = 0x1C
PPSMC_MSG_GetDebugData = 0x1D
PPSMC_MSG_HeavySBR = 0x1E
PPSMC_MSG_SetNumBadHbmPagesRetired = 0x1F
PPSMC_MSG_DFCstateControl = 0x20
PPSMC_MSG_GetGmiPwrDnHyst = 0x21
PPSMC_MSG_SetGmiPwrDnHyst = 0x22
PPSMC_MSG_GmiPwrDnControl = 0x23
PPSMC_MSG_EnterGfxoff = 0x24
PPSMC_MSG_ExitGfxoff = 0x25
PPSMC_MSG_EnableDeterminism = 0x26
PPSMC_MSG_DisableDeterminism = 0x27
PPSMC_MSG_DumpSTBtoDram = 0x28
PPSMC_MSG_STBtoDramLogSetDramAddrHigh = 0x29
PPSMC_MSG_STBtoDramLogSetDramAddrLow = 0x2A
PPSMC_MSG_STBtoDramLogSetDramSize = 0x2B
PPSMC_MSG_SetSystemVirtualSTBtoDramAddrHigh = 0x2C
PPSMC_MSG_SetSystemVirtualSTBtoDramAddrLow = 0x2D
PPSMC_MSG_GfxDriverResetRecovery = 0x2E
PPSMC_MSG_TriggerVFFLR = 0x2F
PPSMC_MSG_SetSoftMinGfxClk = 0x30
PPSMC_MSG_SetSoftMaxGfxClk = 0x31
PPSMC_MSG_GetMinGfxDpmFreq = 0x32
PPSMC_MSG_GetMaxGfxDpmFreq = 0x33
PPSMC_MSG_PrepareForDriverUnload = 0x34
PPSMC_MSG_ReadThrottlerLimit = 0x35
PPSMC_MSG_QueryValidMcaCount = 0x36
PPSMC_MSG_McaBankDumpDW = 0x37
PPSMC_MSG_GetCTFLimit = 0x38
PPSMC_MSG_ClearMcaOnRead = 0x39
PPSMC_MSG_QueryValidMcaCeCount = 0x3A
PPSMC_MSG_McaBankCeDumpDW = 0x3B
PPSMC_MSG_SelectPLPDMode = 0x40
PPSMC_MSG_PmLogReadSample = 0x41
PPSMC_MSG_PmLogGetTableVersion = 0x42
PPSMC_MSG_RmaDueToBadPageThreshold = 0x43
PPSMC_MSG_SetThrottlingPolicy = 0x44
PPSMC_MSG_SetPhaseDetectCSBWThreshold = 0x45
PPSMC_MSG_SetPhaseDetectFreqHigh = 0x46
PPSMC_MSG_SetPhaseDetectFreqLow = 0x47
PPSMC_MSG_SetPhaseDetectDownHysterisis = 0x48
PPSMC_MSG_SetPhaseDetectAlphaX1e6 = 0x49
PPSMC_MSG_SetPhaseDetectOnOff = 0x4A
PPSMC_MSG_GetPhaseDetectResidency = 0x4B
PPSMC_MSG_UpdatePccWaitDecMaxStr = 0x4C
PPSMC_MSG_ResetSDMA = 0x4D
PPSMC_MSG_GetRasTableVersion = 0x4E
PPSMC_MSG_GetBadPageCount = 0x50
PPSMC_MSG_GetBadPageMcaAddress = 0x51
PPSMC_MSG_SetTimestamp = 0x53
PPSMC_MSG_SetTimestampHi = 0x54
PPSMC_MSG_GetTimestamp = 0x55
PPSMC_MSG_GetBadPageIpIdLoHi = 0x57
PPSMC_MSG_EraseRasTable = 0x58
PPSMC_MSG_GetStaticMetricsTable = 0x59
PPSMC_MSG_ResetVfArbitersByIndex = 0x5A
PPSMC_MSG_GetSystemMetricsTable = 0x5C
PPSMC_MSG_GetSystemMetricsVersion = 0x5D
PPSMC_MSG_ResetVCN = 0x5E
PPSMC_MSG_SetFastPptLimit = 0x5F
PPSMC_MSG_GetFastPptLimit = 0x60
PPSMC_Message_Count = 0x61
PPSMC_RESET_TYPE_DRIVER_MODE_1_RESET = 0x1
PPSMC_RESET_TYPE_DRIVER_MODE_2_RESET = 0x2
PPSMC_RESET_TYPE_DRIVER_MODE_3_RESET = 0x3
PPSMC_THROTTLING_LIMIT_TYPE_SOCKET = 0x1
PPSMC_THROTTLING_LIMIT_TYPE_HBM = 0x2
PPSMC_AID_THM_TYPE = 0x1
PPSMC_CCD_THM_TYPE = 0x2
PPSMC_XCD_THM_TYPE = 0x3
PPSMC_HBM_THM_TYPE = 0x4
PPSMC_PLPD_MODE_DEFAULT = 0x1
PPSMC_PLPD_MODE_OPTIMIZED = 0x2
NUM_VCLK_DPM_LEVELS = 4
NUM_DCLK_DPM_LEVELS = 4
NUM_SOCCLK_DPM_LEVELS = 4
NUM_LCLK_DPM_LEVELS = 4
NUM_UCLK_DPM_LEVELS = 4
NUM_FCLK_DPM_LEVELS = 4
NUM_XGMI_DPM_LEVELS = 2
NUM_CXL_BITRATES = 4
NUM_PCIE_BITRATES = 4
NUM_XGMI_BITRATES = 4
NUM_XGMI_WIDTHS = 3
NUM_TDP_GROUPS = 4
NUM_SOC_P2S_TABLES = 6
NUM_GFX_P2S_TABLES = 8
NUM_PSM_DIDT_THRESHOLDS = 3
NUM_XVMIN_VMIN_THRESHOLDS = 3
PRODUCT_MODEL_NUMBER_LEN = 20
PRODUCT_NAME_LEN = 64
PRODUCT_SERIAL_LEN = 20
PRODUCT_MANUFACTURER_NAME_LEN = 32
PRODUCT_FRU_ID_LEN = 32
SMU_METRICS_TABLE_VERSION = 0x15
SMU_SYSTEM_METRICS_TABLE_VERSION = 0x1
SMU_VF_METRICS_TABLE_MASK = (1 << 31)
SMU_VF_METRICS_TABLE_VERSION = (0x6 | SMU_VF_METRICS_TABLE_MASK)
SMU13_0_6_DRIVER_IF_VERSION = 0x08042024
NUM_I2C_CONTROLLERS = 8
I2C_CONTROLLER_ENABLED = 1
I2C_CONTROLLER_DISABLED = 0
MAX_SW_I2C_COMMANDS = 24
CMDCONFIG_STOP_BIT = 0
CMDCONFIG_RESTART_BIT = 1
CMDCONFIG_READWRITE_BIT = 2
CMDCONFIG_STOP_MASK = (1 << CMDCONFIG_STOP_BIT)
CMDCONFIG_RESTART_MASK = (1 << CMDCONFIG_RESTART_BIT)
CMDCONFIG_READWRITE_MASK = (1 << CMDCONFIG_READWRITE_BIT)
IH_INTERRUPT_ID_TO_DRIVER = 0xFE
IH_INTERRUPT_CONTEXT_ID_THERMAL_THROTTLING = 0x7
THROTTLER_PROCHOT_BIT = 0
THROTTLER_PPT_BIT = 1
THROTTLER_THERMAL_SOCKET_BIT = 2
THROTTLER_THERMAL_VR_BIT = 3
THROTTLER_THERMAL_HBM_BIT = 4
ClearMcaOnRead_UE_FLAG_MASK = 0x1
ClearMcaOnRead_CE_POLL_MASK = 0x2
SMU_THERMAL_MINIMUM_ALERT_TEMP = 0
SMU_THERMAL_MAXIMUM_ALERT_TEMP = 255
SMU_TEMPERATURE_UNITS_PER_CENTIGRADES = 1000
SMU_FW_NAME_LEN = 0x24
SMU_DPM_USER_PROFILE_RESTORE = (1 << 0)
SMU_CUSTOM_FAN_SPEED_RPM = (1 << 1)
SMU_CUSTOM_FAN_SPEED_PWM = (1 << 2)
SMU_THROTTLER_PPT0_BIT = 0
SMU_THROTTLER_PPT1_BIT = 1
SMU_THROTTLER_PPT2_BIT = 2
SMU_THROTTLER_PPT3_BIT = 3
SMU_THROTTLER_SPL_BIT = 4
SMU_THROTTLER_FPPT_BIT = 5
SMU_THROTTLER_SPPT_BIT = 6
SMU_THROTTLER_SPPT_APU_BIT = 7
SMU_THROTTLER_TDC_GFX_BIT = 16
SMU_THROTTLER_TDC_SOC_BIT = 17
SMU_THROTTLER_TDC_MEM_BIT = 18
SMU_THROTTLER_TDC_VDD_BIT = 19
SMU_THROTTLER_TDC_CVIP_BIT = 20
SMU_THROTTLER_EDC_CPU_BIT = 21
SMU_THROTTLER_EDC_GFX_BIT = 22
SMU_THROTTLER_APCC_BIT = 23
SMU_THROTTLER_TEMP_GPU_BIT = 32
SMU_THROTTLER_TEMP_CORE_BIT = 33
SMU_THROTTLER_TEMP_MEM_BIT = 34
SMU_THROTTLER_TEMP_EDGE_BIT = 35
SMU_THROTTLER_TEMP_HOTSPOT_BIT = 36
SMU_THROTTLER_TEMP_SOC_BIT = 37
SMU_THROTTLER_TEMP_VR_GFX_BIT = 38
SMU_THROTTLER_TEMP_VR_SOC_BIT = 39
SMU_THROTTLER_TEMP_VR_MEM0_BIT = 40
SMU_THROTTLER_TEMP_VR_MEM1_BIT = 41
SMU_THROTTLER_TEMP_LIQUID0_BIT = 42
SMU_THROTTLER_TEMP_LIQUID1_BIT = 43
SMU_THROTTLER_VRHOT0_BIT = 44
SMU_THROTTLER_VRHOT1_BIT = 45
SMU_THROTTLER_PROCHOT_CPU_BIT = 46
SMU_THROTTLER_PROCHOT_GFX_BIT = 47
SMU_THROTTLER_PPM_BIT = 56
SMU_THROTTLER_FIT_BIT = 57