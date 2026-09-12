# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_SMU14_Firmware_Footer(c.Struct):
  SIZE = 4
  Signature: int
uint32_t: TypeAlias = ctypes.c_uint32
struct_SMU14_Firmware_Footer.register_fields([('Signature', uint32_t, 0)])
SMU14_Firmware_Footer: TypeAlias = struct_SMU14_Firmware_Footer
@c.record
class SMU_Firmware_Header(c.Struct):
  SIZE = 256
  ImageVersion: int
  ImageVersion2: int
  Padding0: c.Array[ctypes.c_uint32, Literal[3]]
  SizeFWSigned: int
  Padding1: c.Array[ctypes.c_uint32, Literal[25]]
  FirmwareType: int
  Filler: c.Array[ctypes.c_uint32, Literal[32]]
SMU_Firmware_Header.register_fields([('ImageVersion', uint32_t, 0), ('ImageVersion2', uint32_t, 4), ('Padding0', c.Array[uint32_t, Literal[3]], 8), ('SizeFWSigned', uint32_t, 20), ('Padding1', c.Array[uint32_t, Literal[25]], 24), ('FirmwareType', uint32_t, 124), ('Filler', c.Array[uint32_t, Literal[32]], 128)])
@c.record
class FwStatus_t(c.Struct):
  SIZE = 24
  DpmHandlerID: int
  ActivityMonitorID: int
  DpmTimerID: int
  DpmHubID: int
  DpmHubTask: int
  CclkSyncStatus: int
  Ccx0CpuOff: int
  Ccx1CpuOff: int
  GfxOffStatus: int
  VddOff: int
  InWhisperMode: int
  ZstateStatus: int
  spare0: int
  DstateFun: int
  DstateDev: int
  P2JobHandler: int
  RsmuPmiP2PendingCnt: int
  PostCode: int
  MsgPortBusy: int
  RsmuPmiP1Pending: int
  DfCstateExitPending: int
  Ccx0Pc6ExitPending: int
  Ccx1Pc6ExitPending: int
  WarmResetPending: int
  spare1: int
  IdleMask: int
FwStatus_t.register_fields([('DpmHandlerID', uint32_t, 0, 8, 0), ('ActivityMonitorID', uint32_t, 1, 8, 0), ('DpmTimerID', uint32_t, 2, 8, 0), ('DpmHubID', uint32_t, 3, 4, 0), ('DpmHubTask', uint32_t, 3, 4, 4), ('CclkSyncStatus', uint32_t, 4, 8, 0), ('Ccx0CpuOff', uint32_t, 5, 2, 0), ('Ccx1CpuOff', uint32_t, 5, 2, 2), ('GfxOffStatus', uint32_t, 5, 2, 4), ('VddOff', uint32_t, 5, 1, 6), ('InWhisperMode', uint32_t, 5, 1, 7), ('ZstateStatus', uint32_t, 6, 4, 0), ('spare0', uint32_t, 6, 4, 4), ('DstateFun', uint32_t, 7, 4, 0), ('DstateDev', uint32_t, 7, 4, 4), ('P2JobHandler', uint32_t, 8, 24, 0), ('RsmuPmiP2PendingCnt', uint32_t, 11, 8, 0), ('PostCode', uint32_t, 12, 32, 0), ('MsgPortBusy', uint32_t, 16, 24, 0), ('RsmuPmiP1Pending', uint32_t, 19, 1, 0), ('DfCstateExitPending', uint32_t, 19, 1, 1), ('Ccx0Pc6ExitPending', uint32_t, 19, 1, 2), ('Ccx1Pc6ExitPending', uint32_t, 19, 1, 3), ('WarmResetPending', uint32_t, 19, 1, 4), ('spare1', uint32_t, 19, 3, 5), ('IdleMask', uint32_t, 20, 32, 0)])
@c.record
class FwStatus_t_v14_0_1(c.Struct):
  SIZE = 24
  DpmHandlerID: int
  ActivityMonitorID: int
  DpmTimerID: int
  DpmHubID: int
  DpmHubTask: int
  CclkSyncStatus: int
  ZstateStatus: int
  Cpu1VddOff: int
  DstateFun: int
  DstateDev: int
  GfxOffStatus: int
  Cpu0Off: int
  Cpu1Off: int
  Cpu0VddOff: int
  P2JobHandler: int
  PostCode: int
  MsgPortBusy: int
  RsmuPmiP1Pending: int
  RsmuPmiP2PendingCnt: int
  DfCstateExitPending: int
  Pc6EntryPending: int
  Pc6ExitPending: int
  WarmResetPending: int
  Mp0ClkPending: int
  InWhisperMode: int
  spare2: int
  IdleMask: int
FwStatus_t_v14_0_1.register_fields([('DpmHandlerID', uint32_t, 0, 8, 0), ('ActivityMonitorID', uint32_t, 1, 8, 0), ('DpmTimerID', uint32_t, 2, 8, 0), ('DpmHubID', uint32_t, 3, 4, 0), ('DpmHubTask', uint32_t, 3, 4, 4), ('CclkSyncStatus', uint32_t, 4, 8, 0), ('ZstateStatus', uint32_t, 5, 4, 0), ('Cpu1VddOff', uint32_t, 5, 4, 4), ('DstateFun', uint32_t, 6, 4, 0), ('DstateDev', uint32_t, 6, 4, 4), ('GfxOffStatus', uint32_t, 7, 2, 0), ('Cpu0Off', uint32_t, 7, 2, 2), ('Cpu1Off', uint32_t, 7, 2, 4), ('Cpu0VddOff', uint32_t, 7, 2, 6), ('P2JobHandler', uint32_t, 8, 32, 0), ('PostCode', uint32_t, 12, 32, 0), ('MsgPortBusy', uint32_t, 16, 15, 0), ('RsmuPmiP1Pending', uint32_t, 17, 1, 7), ('RsmuPmiP2PendingCnt', uint32_t, 18, 8, 0), ('DfCstateExitPending', uint32_t, 19, 1, 0), ('Pc6EntryPending', uint32_t, 19, 1, 1), ('Pc6ExitPending', uint32_t, 19, 1, 2), ('WarmResetPending', uint32_t, 19, 1, 3), ('Mp0ClkPending', uint32_t, 19, 1, 4), ('InWhisperMode', uint32_t, 19, 1, 5), ('spare2', uint32_t, 19, 2, 6), ('IdleMask', uint32_t, 20, 32, 0)])
FEATURE_PWR_DOMAIN_e: dict[int, str] = {(FEATURE_PWR_ALL:=0): 'FEATURE_PWR_ALL', (FEATURE_PWR_S5:=1): 'FEATURE_PWR_S5', (FEATURE_PWR_BACO:=2): 'FEATURE_PWR_BACO', (FEATURE_PWR_SOC:=3): 'FEATURE_PWR_SOC', (FEATURE_PWR_GFX:=4): 'FEATURE_PWR_GFX', (FEATURE_PWR_DOMAIN_COUNT:=5): 'FEATURE_PWR_DOMAIN_COUNT'}
FEATURE_BTC_e: dict[int, str] = {(FEATURE_BTC_NOP:=0): 'FEATURE_BTC_NOP', (FEATURE_BTC_SAVE:=1): 'FEATURE_BTC_SAVE', (FEATURE_BTC_RESTORE:=2): 'FEATURE_BTC_RESTORE', (FEATURE_BTC_COUNT:=3): 'FEATURE_BTC_COUNT'}
SVI_PSI_e: dict[int, str] = {(SVI_PSI_0:=0): 'SVI_PSI_0', (SVI_PSI_1:=1): 'SVI_PSI_1', (SVI_PSI_2:=2): 'SVI_PSI_2', (SVI_PSI_3:=3): 'SVI_PSI_3', (SVI_PSI_4:=4): 'SVI_PSI_4', (SVI_PSI_5:=5): 'SVI_PSI_5', (SVI_PSI_6:=6): 'SVI_PSI_6', (SVI_PSI_7:=7): 'SVI_PSI_7'}
SMARTSHIFT_VERSION_e: dict[int, str] = {(SMARTSHIFT_VERSION_1:=0): 'SMARTSHIFT_VERSION_1', (SMARTSHIFT_VERSION_2:=1): 'SMARTSHIFT_VERSION_2', (SMARTSHIFT_VERSION_3:=2): 'SMARTSHIFT_VERSION_3'}
FOPT_CALC_e: dict[int, str] = {(FOPT_CALC_AC_CALC_DC:=0): 'FOPT_CALC_AC_CALC_DC', (FOPT_PPTABLE_AC_CALC_DC:=1): 'FOPT_PPTABLE_AC_CALC_DC', (FOPT_CALC_AC_PPTABLE_DC:=2): 'FOPT_CALC_AC_PPTABLE_DC', (FOPT_PPTABLE_AC_PPTABLE_DC:=3): 'FOPT_PPTABLE_AC_PPTABLE_DC'}
DRAM_BIT_WIDTH_TYPE_e: dict[int, str] = {(DRAM_BIT_WIDTH_DISABLED:=0): 'DRAM_BIT_WIDTH_DISABLED', (DRAM_BIT_WIDTH_X_8:=8): 'DRAM_BIT_WIDTH_X_8', (DRAM_BIT_WIDTH_X_16:=16): 'DRAM_BIT_WIDTH_X_16', (DRAM_BIT_WIDTH_X_32:=32): 'DRAM_BIT_WIDTH_X_32', (DRAM_BIT_WIDTH_X_64:=64): 'DRAM_BIT_WIDTH_X_64', (DRAM_BIT_WIDTH_X_128:=128): 'DRAM_BIT_WIDTH_X_128', (DRAM_BIT_WIDTH_COUNT:=129): 'DRAM_BIT_WIDTH_COUNT'}
I2cControllerPort_e: dict[int, str] = {(I2C_CONTROLLER_PORT_0:=0): 'I2C_CONTROLLER_PORT_0', (I2C_CONTROLLER_PORT_1:=1): 'I2C_CONTROLLER_PORT_1', (I2C_CONTROLLER_PORT_COUNT:=2): 'I2C_CONTROLLER_PORT_COUNT'}
I2cControllerName_e: dict[int, str] = {(I2C_CONTROLLER_NAME_VR_GFX:=0): 'I2C_CONTROLLER_NAME_VR_GFX', (I2C_CONTROLLER_NAME_VR_SOC:=1): 'I2C_CONTROLLER_NAME_VR_SOC', (I2C_CONTROLLER_NAME_VR_VMEMP:=2): 'I2C_CONTROLLER_NAME_VR_VMEMP', (I2C_CONTROLLER_NAME_VR_VDDIO:=3): 'I2C_CONTROLLER_NAME_VR_VDDIO', (I2C_CONTROLLER_NAME_LIQUID0:=4): 'I2C_CONTROLLER_NAME_LIQUID0', (I2C_CONTROLLER_NAME_LIQUID1:=5): 'I2C_CONTROLLER_NAME_LIQUID1', (I2C_CONTROLLER_NAME_PLX:=6): 'I2C_CONTROLLER_NAME_PLX', (I2C_CONTROLLER_NAME_FAN_INTAKE:=7): 'I2C_CONTROLLER_NAME_FAN_INTAKE', (I2C_CONTROLLER_NAME_COUNT:=8): 'I2C_CONTROLLER_NAME_COUNT'}
I2cControllerThrottler_e: dict[int, str] = {(I2C_CONTROLLER_THROTTLER_TYPE_NONE:=0): 'I2C_CONTROLLER_THROTTLER_TYPE_NONE', (I2C_CONTROLLER_THROTTLER_VR_GFX:=1): 'I2C_CONTROLLER_THROTTLER_VR_GFX', (I2C_CONTROLLER_THROTTLER_VR_SOC:=2): 'I2C_CONTROLLER_THROTTLER_VR_SOC', (I2C_CONTROLLER_THROTTLER_VR_VMEMP:=3): 'I2C_CONTROLLER_THROTTLER_VR_VMEMP', (I2C_CONTROLLER_THROTTLER_VR_VDDIO:=4): 'I2C_CONTROLLER_THROTTLER_VR_VDDIO', (I2C_CONTROLLER_THROTTLER_LIQUID0:=5): 'I2C_CONTROLLER_THROTTLER_LIQUID0', (I2C_CONTROLLER_THROTTLER_LIQUID1:=6): 'I2C_CONTROLLER_THROTTLER_LIQUID1', (I2C_CONTROLLER_THROTTLER_PLX:=7): 'I2C_CONTROLLER_THROTTLER_PLX', (I2C_CONTROLLER_THROTTLER_FAN_INTAKE:=8): 'I2C_CONTROLLER_THROTTLER_FAN_INTAKE', (I2C_CONTROLLER_THROTTLER_INA3221:=9): 'I2C_CONTROLLER_THROTTLER_INA3221', (I2C_CONTROLLER_THROTTLER_COUNT:=10): 'I2C_CONTROLLER_THROTTLER_COUNT'}
I2cControllerProtocol_e: dict[int, str] = {(I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5:=0): 'I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5', (I2C_CONTROLLER_PROTOCOL_VR_IR35217:=1): 'I2C_CONTROLLER_PROTOCOL_VR_IR35217', (I2C_CONTROLLER_PROTOCOL_TMP_MAX31875:=2): 'I2C_CONTROLLER_PROTOCOL_TMP_MAX31875', (I2C_CONTROLLER_PROTOCOL_INA3221:=3): 'I2C_CONTROLLER_PROTOCOL_INA3221', (I2C_CONTROLLER_PROTOCOL_TMP_MAX6604:=4): 'I2C_CONTROLLER_PROTOCOL_TMP_MAX6604', (I2C_CONTROLLER_PROTOCOL_COUNT:=5): 'I2C_CONTROLLER_PROTOCOL_COUNT'}
@c.record
class I2cControllerConfig_t(c.Struct):
  SIZE = 8
  Enabled: int
  Speed: int
  SlaveAddress: int
  ControllerPort: int
  ControllerName: int
  ThermalThrotter: int
  I2cProtocol: int
  PaddingConfig: int
uint8_t: TypeAlias = ctypes.c_ubyte
I2cControllerConfig_t.register_fields([('Enabled', uint8_t, 0), ('Speed', uint8_t, 1), ('SlaveAddress', uint8_t, 2), ('ControllerPort', uint8_t, 3), ('ControllerName', uint8_t, 4), ('ThermalThrotter', uint8_t, 5), ('I2cProtocol', uint8_t, 6), ('PaddingConfig', uint8_t, 7)])
I2cPort_e: dict[int, str] = {(I2C_PORT_SVD_SCL:=0): 'I2C_PORT_SVD_SCL', (I2C_PORT_GPIO:=1): 'I2C_PORT_GPIO'}
I2cSpeed_e: dict[int, str] = {(I2C_SPEED_FAST_50K:=0): 'I2C_SPEED_FAST_50K', (I2C_SPEED_FAST_100K:=1): 'I2C_SPEED_FAST_100K', (I2C_SPEED_FAST_400K:=2): 'I2C_SPEED_FAST_400K', (I2C_SPEED_FAST_PLUS_1M:=3): 'I2C_SPEED_FAST_PLUS_1M', (I2C_SPEED_HIGH_1M:=4): 'I2C_SPEED_HIGH_1M', (I2C_SPEED_HIGH_2M:=5): 'I2C_SPEED_HIGH_2M', (I2C_SPEED_COUNT:=6): 'I2C_SPEED_COUNT'}
I2cCmdType_e: dict[int, str] = {(I2C_CMD_READ:=0): 'I2C_CMD_READ', (I2C_CMD_WRITE:=1): 'I2C_CMD_WRITE', (I2C_CMD_COUNT:=2): 'I2C_CMD_COUNT'}
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
@c.record
class EccInfo_t(c.Struct):
  SIZE = 24
  mca_umc_status: int
  mca_umc_addr: int
  ce_count_lo_chip: int
  ce_count_hi_chip: int
  eccPadding: int
uint64_t: TypeAlias = ctypes.c_uint64
uint16_t: TypeAlias = ctypes.c_uint16
EccInfo_t.register_fields([('mca_umc_status', uint64_t, 0), ('mca_umc_addr', uint64_t, 8), ('ce_count_lo_chip', uint16_t, 16), ('ce_count_hi_chip', uint16_t, 18), ('eccPadding', uint32_t, 20)])
@c.record
class EccInfoTable_t(c.Struct):
  SIZE = 576
  EccInfo: c.Array[EccInfo_t, Literal[24]]
EccInfoTable_t.register_fields([('EccInfo', c.Array[EccInfo_t, Literal[24]], 0)])
EPCS_STATUS_e: dict[int, str] = {(EPCS_SHORTED_LIMIT:=0): 'EPCS_SHORTED_LIMIT', (EPCS_LOW_POWER_LIMIT:=1): 'EPCS_LOW_POWER_LIMIT', (EPCS_NORMAL_POWER_LIMIT:=2): 'EPCS_NORMAL_POWER_LIMIT', (EPCS_HIGH_POWER_LIMIT:=3): 'EPCS_HIGH_POWER_LIMIT', (EPCS_NOT_CONFIGURED:=4): 'EPCS_NOT_CONFIGURED', (EPCS_STATUS_COUNT:=5): 'EPCS_STATUS_COUNT'}
D3HOTSequence_e: dict[int, str] = {(BACO_SEQUENCE:=0): 'BACO_SEQUENCE', (MSR_SEQUENCE:=1): 'MSR_SEQUENCE', (BAMACO_SEQUENCE:=2): 'BAMACO_SEQUENCE', (ULPS_SEQUENCE:=3): 'ULPS_SEQUENCE', (D3HOT_SEQUENCE_COUNT:=4): 'D3HOT_SEQUENCE_COUNT'}
PowerGatingMode_e: dict[int, str] = {(PG_DYNAMIC_MODE:=0): 'PG_DYNAMIC_MODE', (PG_STATIC_MODE:=1): 'PG_STATIC_MODE'}
PowerGatingSettings_e: dict[int, str] = {(PG_POWER_DOWN:=0): 'PG_POWER_DOWN', (PG_POWER_UP:=1): 'PG_POWER_UP'}
@c.record
class QuadraticInt_t(c.Struct):
  SIZE = 12
  a: int
  b: int
  c: int
QuadraticInt_t.register_fields([('a', uint32_t, 0), ('b', uint32_t, 4), ('c', uint32_t, 8)])
@c.record
class LinearInt_t(c.Struct):
  SIZE = 8
  m: int
  b: int
LinearInt_t.register_fields([('m', uint32_t, 0), ('b', uint32_t, 4)])
@c.record
class DroopInt_t(c.Struct):
  SIZE = 12
  a: int
  b: int
  c: int
DroopInt_t.register_fields([('a', uint32_t, 0), ('b', uint32_t, 4), ('c', uint32_t, 8)])
DCS_ARCH_e: dict[int, str] = {(DCS_ARCH_DISABLED:=0): 'DCS_ARCH_DISABLED', (DCS_ARCH_FADCS:=1): 'DCS_ARCH_FADCS', (DCS_ARCH_ASYNC:=2): 'DCS_ARCH_ASYNC'}
PPCLK_e: dict[int, str] = {(PPCLK_GFXCLK:=0): 'PPCLK_GFXCLK', (PPCLK_SOCCLK:=1): 'PPCLK_SOCCLK', (PPCLK_UCLK:=2): 'PPCLK_UCLK', (PPCLK_FCLK:=3): 'PPCLK_FCLK', (PPCLK_DCLK_0:=4): 'PPCLK_DCLK_0', (PPCLK_VCLK_0:=5): 'PPCLK_VCLK_0', (PPCLK_DISPCLK:=6): 'PPCLK_DISPCLK', (PPCLK_DPPCLK:=7): 'PPCLK_DPPCLK', (PPCLK_DPREFCLK:=8): 'PPCLK_DPREFCLK', (PPCLK_DCFCLK:=9): 'PPCLK_DCFCLK', (PPCLK_DTBCLK:=10): 'PPCLK_DTBCLK', (PPCLK_COUNT:=11): 'PPCLK_COUNT'}
VOLTAGE_MODE_e: dict[int, str] = {(VOLTAGE_MODE_PPTABLE:=0): 'VOLTAGE_MODE_PPTABLE', (VOLTAGE_MODE_FUSES:=1): 'VOLTAGE_MODE_FUSES', (VOLTAGE_MODE_COUNT:=2): 'VOLTAGE_MODE_COUNT'}
AVFS_VOLTAGE_TYPE_e: dict[int, str] = {(AVFS_VOLTAGE_GFX:=0): 'AVFS_VOLTAGE_GFX', (AVFS_VOLTAGE_SOC:=1): 'AVFS_VOLTAGE_SOC', (AVFS_VOLTAGE_COUNT:=2): 'AVFS_VOLTAGE_COUNT'}
AVFS_TEMP_e: dict[int, str] = {(AVFS_TEMP_COLD:=0): 'AVFS_TEMP_COLD', (AVFS_TEMP_HOT:=1): 'AVFS_TEMP_HOT', (AVFS_TEMP_COUNT:=2): 'AVFS_TEMP_COUNT'}
AVFS_D_e: dict[int, str] = {(AVFS_D_G:=0): 'AVFS_D_G', (AVFS_D_COUNT:=1): 'AVFS_D_COUNT'}
UCLK_DIV_e: dict[int, str] = {(UCLK_DIV_BY_1:=0): 'UCLK_DIV_BY_1', (UCLK_DIV_BY_2:=1): 'UCLK_DIV_BY_2', (UCLK_DIV_BY_4:=2): 'UCLK_DIV_BY_4', (UCLK_DIV_BY_8:=3): 'UCLK_DIV_BY_8'}
GpioIntPolarity_e: dict[int, str] = {(GPIO_INT_POLARITY_ACTIVE_LOW:=0): 'GPIO_INT_POLARITY_ACTIVE_LOW', (GPIO_INT_POLARITY_ACTIVE_HIGH:=1): 'GPIO_INT_POLARITY_ACTIVE_HIGH'}
PwrConfig_e: dict[int, str] = {(PWR_CONFIG_TDP:=0): 'PWR_CONFIG_TDP', (PWR_CONFIG_TGP:=1): 'PWR_CONFIG_TGP', (PWR_CONFIG_TCP_ESTIMATED:=2): 'PWR_CONFIG_TCP_ESTIMATED', (PWR_CONFIG_TCP_MEASURED:=3): 'PWR_CONFIG_TCP_MEASURED', (PWR_CONFIG_TBP_DESKTOP:=4): 'PWR_CONFIG_TBP_DESKTOP', (PWR_CONFIG_TBP_MOBILE:=5): 'PWR_CONFIG_TBP_MOBILE'}
@c.record
class DpmDescriptor_t(c.Struct):
  SIZE = 32
  Padding: int
  SnapToDiscrete: int
  NumDiscreteLevels: int
  CalculateFopt: int
  ConversionToAvfsClk: LinearInt_t
  Padding3: c.Array[ctypes.c_uint32, Literal[3]]
  Padding4: int
  FoptimalDc: int
  FoptimalAc: int
  Padding2: int
DpmDescriptor_t.register_fields([('Padding', uint8_t, 0), ('SnapToDiscrete', uint8_t, 1), ('NumDiscreteLevels', uint8_t, 2), ('CalculateFopt', uint8_t, 3), ('ConversionToAvfsClk', LinearInt_t, 4), ('Padding3', c.Array[uint32_t, Literal[3]], 12), ('Padding4', uint16_t, 24), ('FoptimalDc', uint16_t, 26), ('FoptimalAc', uint16_t, 28), ('Padding2', uint16_t, 30)])
PPT_THROTTLER_e: dict[int, str] = {(PPT_THROTTLER_PPT0:=0): 'PPT_THROTTLER_PPT0', (PPT_THROTTLER_PPT1:=1): 'PPT_THROTTLER_PPT1', (PPT_THROTTLER_PPT2:=2): 'PPT_THROTTLER_PPT2', (PPT_THROTTLER_PPT3:=3): 'PPT_THROTTLER_PPT3', (PPT_THROTTLER_COUNT:=4): 'PPT_THROTTLER_COUNT'}
TEMP_e: dict[int, str] = {(TEMP_EDGE:=0): 'TEMP_EDGE', (TEMP_HOTSPOT:=1): 'TEMP_HOTSPOT', (TEMP_HOTSPOT_GFX:=2): 'TEMP_HOTSPOT_GFX', (TEMP_HOTSPOT_SOC:=3): 'TEMP_HOTSPOT_SOC', (TEMP_MEM:=4): 'TEMP_MEM', (TEMP_VR_GFX:=5): 'TEMP_VR_GFX', (TEMP_VR_SOC:=6): 'TEMP_VR_SOC', (TEMP_VR_MEM0:=7): 'TEMP_VR_MEM0', (TEMP_VR_MEM1:=8): 'TEMP_VR_MEM1', (TEMP_LIQUID0:=9): 'TEMP_LIQUID0', (TEMP_LIQUID1:=10): 'TEMP_LIQUID1', (TEMP_PLX:=11): 'TEMP_PLX', (TEMP_COUNT:=12): 'TEMP_COUNT'}
TDC_THROTTLER_e: dict[int, str] = {(TDC_THROTTLER_GFX:=0): 'TDC_THROTTLER_GFX', (TDC_THROTTLER_SOC:=1): 'TDC_THROTTLER_SOC', (TDC_THROTTLER_COUNT:=2): 'TDC_THROTTLER_COUNT'}
SVI_PLANE_e: dict[int, str] = {(SVI_PLANE_VDD_GFX:=0): 'SVI_PLANE_VDD_GFX', (SVI_PLANE_VDD_SOC:=1): 'SVI_PLANE_VDD_SOC', (SVI_PLANE_VDDCI_MEM:=2): 'SVI_PLANE_VDDCI_MEM', (SVI_PLANE_VDDIO_MEM:=3): 'SVI_PLANE_VDDIO_MEM', (SVI_PLANE_COUNT:=4): 'SVI_PLANE_COUNT'}
PMFW_VOLT_PLANE_e: dict[int, str] = {(PMFW_VOLT_PLANE_GFX:=0): 'PMFW_VOLT_PLANE_GFX', (PMFW_VOLT_PLANE_SOC:=1): 'PMFW_VOLT_PLANE_SOC', (PMFW_VOLT_PLANE_COUNT:=2): 'PMFW_VOLT_PLANE_COUNT'}
CUSTOMER_VARIANT_e: dict[int, str] = {(CUSTOMER_VARIANT_ROW:=0): 'CUSTOMER_VARIANT_ROW', (CUSTOMER_VARIANT_FALCON:=1): 'CUSTOMER_VARIANT_FALCON', (CUSTOMER_VARIANT_COUNT:=2): 'CUSTOMER_VARIANT_COUNT'}
POWER_SOURCE_e: dict[int, str] = {(POWER_SOURCE_AC:=0): 'POWER_SOURCE_AC', (POWER_SOURCE_DC:=1): 'POWER_SOURCE_DC', (POWER_SOURCE_COUNT:=2): 'POWER_SOURCE_COUNT'}
MEM_VENDOR_e: dict[int, str] = {(MEM_VENDOR_PLACEHOLDER0:=0): 'MEM_VENDOR_PLACEHOLDER0', (MEM_VENDOR_SAMSUNG:=1): 'MEM_VENDOR_SAMSUNG', (MEM_VENDOR_INFINEON:=2): 'MEM_VENDOR_INFINEON', (MEM_VENDOR_ELPIDA:=3): 'MEM_VENDOR_ELPIDA', (MEM_VENDOR_ETRON:=4): 'MEM_VENDOR_ETRON', (MEM_VENDOR_NANYA:=5): 'MEM_VENDOR_NANYA', (MEM_VENDOR_HYNIX:=6): 'MEM_VENDOR_HYNIX', (MEM_VENDOR_MOSEL:=7): 'MEM_VENDOR_MOSEL', (MEM_VENDOR_WINBOND:=8): 'MEM_VENDOR_WINBOND', (MEM_VENDOR_ESMT:=9): 'MEM_VENDOR_ESMT', (MEM_VENDOR_PLACEHOLDER1:=10): 'MEM_VENDOR_PLACEHOLDER1', (MEM_VENDOR_PLACEHOLDER2:=11): 'MEM_VENDOR_PLACEHOLDER2', (MEM_VENDOR_PLACEHOLDER3:=12): 'MEM_VENDOR_PLACEHOLDER3', (MEM_VENDOR_PLACEHOLDER4:=13): 'MEM_VENDOR_PLACEHOLDER4', (MEM_VENDOR_PLACEHOLDER5:=14): 'MEM_VENDOR_PLACEHOLDER5', (MEM_VENDOR_MICRON:=15): 'MEM_VENDOR_MICRON', (MEM_VENDOR_COUNT:=16): 'MEM_VENDOR_COUNT'}
PP_GRTAVFS_HW_FUSE_e: dict[int, str] = {(PP_GRTAVFS_HW_CPO_CTL_ZONE0:=0): 'PP_GRTAVFS_HW_CPO_CTL_ZONE0', (PP_GRTAVFS_HW_CPO_CTL_ZONE1:=1): 'PP_GRTAVFS_HW_CPO_CTL_ZONE1', (PP_GRTAVFS_HW_CPO_CTL_ZONE2:=2): 'PP_GRTAVFS_HW_CPO_CTL_ZONE2', (PP_GRTAVFS_HW_CPO_CTL_ZONE3:=3): 'PP_GRTAVFS_HW_CPO_CTL_ZONE3', (PP_GRTAVFS_HW_CPO_CTL_ZONE4:=4): 'PP_GRTAVFS_HW_CPO_CTL_ZONE4', (PP_GRTAVFS_HW_CPO_EN_0_31_ZONE0:=5): 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE0', (PP_GRTAVFS_HW_CPO_EN_32_63_ZONE0:=6): 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE0', (PP_GRTAVFS_HW_CPO_EN_0_31_ZONE1:=7): 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE1', (PP_GRTAVFS_HW_CPO_EN_32_63_ZONE1:=8): 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE1', (PP_GRTAVFS_HW_CPO_EN_0_31_ZONE2:=9): 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE2', (PP_GRTAVFS_HW_CPO_EN_32_63_ZONE2:=10): 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE2', (PP_GRTAVFS_HW_CPO_EN_0_31_ZONE3:=11): 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE3', (PP_GRTAVFS_HW_CPO_EN_32_63_ZONE3:=12): 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE3', (PP_GRTAVFS_HW_CPO_EN_0_31_ZONE4:=13): 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE4', (PP_GRTAVFS_HW_CPO_EN_32_63_ZONE4:=14): 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE4', (PP_GRTAVFS_HW_ZONE0_VF:=15): 'PP_GRTAVFS_HW_ZONE0_VF', (PP_GRTAVFS_HW_ZONE1_VF1:=16): 'PP_GRTAVFS_HW_ZONE1_VF1', (PP_GRTAVFS_HW_ZONE2_VF2:=17): 'PP_GRTAVFS_HW_ZONE2_VF2', (PP_GRTAVFS_HW_ZONE3_VF3:=18): 'PP_GRTAVFS_HW_ZONE3_VF3', (PP_GRTAVFS_HW_VOLTAGE_GB:=19): 'PP_GRTAVFS_HW_VOLTAGE_GB', (PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE0:=20): 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE0', (PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE1:=21): 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE1', (PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE2:=22): 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE2', (PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE3:=23): 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE3', (PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE4:=24): 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE4', (PP_GRTAVFS_HW_RESERVED_0:=25): 'PP_GRTAVFS_HW_RESERVED_0', (PP_GRTAVFS_HW_RESERVED_1:=26): 'PP_GRTAVFS_HW_RESERVED_1', (PP_GRTAVFS_HW_RESERVED_2:=27): 'PP_GRTAVFS_HW_RESERVED_2', (PP_GRTAVFS_HW_RESERVED_3:=28): 'PP_GRTAVFS_HW_RESERVED_3', (PP_GRTAVFS_HW_RESERVED_4:=29): 'PP_GRTAVFS_HW_RESERVED_4', (PP_GRTAVFS_HW_RESERVED_5:=30): 'PP_GRTAVFS_HW_RESERVED_5', (PP_GRTAVFS_HW_RESERVED_6:=31): 'PP_GRTAVFS_HW_RESERVED_6', (PP_GRTAVFS_HW_FUSE_COUNT:=32): 'PP_GRTAVFS_HW_FUSE_COUNT'}
PP_GRTAVFS_FW_COMMON_FUSE_e: dict[int, str] = {(PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_HOT_T0:=0): 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_HOT_T0', (PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_COLD_T0:=1): 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_COLD_T0', (PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_HOT_T0:=2): 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_HOT_T0', (PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_COLD_T0:=3): 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_COLD_T0', (PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_HOT_T0:=4): 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_HOT_T0', (PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_COLD_T0:=5): 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_COLD_T0', (PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_HOT_T0:=6): 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_HOT_T0', (PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_COLD_T0:=7): 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_COLD_T0', (PP_GRTAVFS_FW_COMMON_SRAM_RM_Z0:=8): 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z0', (PP_GRTAVFS_FW_COMMON_SRAM_RM_Z1:=9): 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z1', (PP_GRTAVFS_FW_COMMON_SRAM_RM_Z2:=10): 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z2', (PP_GRTAVFS_FW_COMMON_SRAM_RM_Z3:=11): 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z3', (PP_GRTAVFS_FW_COMMON_SRAM_RM_Z4:=12): 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z4', (PP_GRTAVFS_FW_COMMON_FUSE_COUNT:=13): 'PP_GRTAVFS_FW_COMMON_FUSE_COUNT'}
PP_GRTAVFS_FW_SEP_FUSE_e: dict[int, str] = {(PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_NEG_1:=0): 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_NEG_1', (PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_0:=1): 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_0', (PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_1:=2): 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_1', (PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_2:=3): 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_2', (PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_3:=4): 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_3', (PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_4:=5): 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_4', (PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_NEG_1:=6): 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_NEG_1', (PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_0:=7): 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_0', (PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_1:=8): 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_1', (PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_2:=9): 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_2', (PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_3:=10): 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_3', (PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_4:=11): 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_4', (PP_GRTAVFS_FW_SEP_FUSE_VF_NEG_1_FREQUENCY:=12): 'PP_GRTAVFS_FW_SEP_FUSE_VF_NEG_1_FREQUENCY', (PP_GRTAVFS_FW_SEP_FUSE_VF4_FREQUENCY:=13): 'PP_GRTAVFS_FW_SEP_FUSE_VF4_FREQUENCY', (PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_0:=14): 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_0', (PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_1:=15): 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_1', (PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_2:=16): 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_2', (PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_3:=17): 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_3', (PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_4:=18): 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_4', (PP_GRTAVFS_FW_SEP_FUSE_COUNT:=19): 'PP_GRTAVFS_FW_SEP_FUSE_COUNT'}
@c.record
class SviTelemetryScale_t(c.Struct):
  SIZE = 4
  Offset: int
  Padding: int
  MaxCurrent: int
int8_t: TypeAlias = ctypes.c_byte
SviTelemetryScale_t.register_fields([('Offset', int8_t, 0), ('Padding', uint8_t, 1), ('MaxCurrent', uint16_t, 2)])
PP_OD_POWER_FEATURE_e: dict[int, str] = {(PP_OD_POWER_FEATURE_ALWAYS_ENABLED:=0): 'PP_OD_POWER_FEATURE_ALWAYS_ENABLED', (PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING:=1): 'PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING', (PP_OD_POWER_FEATURE_ALWAYS_DISABLED:=2): 'PP_OD_POWER_FEATURE_ALWAYS_DISABLED'}
FanMode_e: dict[int, str] = {(FAN_MODE_AUTO:=0): 'FAN_MODE_AUTO', (FAN_MODE_MANUAL_LINEAR:=1): 'FAN_MODE_MANUAL_LINEAR'}
OD_FAIL_e: dict[int, str] = {(OD_NO_ERROR:=0): 'OD_NO_ERROR', (OD_REQUEST_ADVANCED_NOT_SUPPORTED:=1): 'OD_REQUEST_ADVANCED_NOT_SUPPORTED', (OD_UNSUPPORTED_FEATURE:=2): 'OD_UNSUPPORTED_FEATURE', (OD_INVALID_FEATURE_COMBO_ERROR:=3): 'OD_INVALID_FEATURE_COMBO_ERROR', (OD_GFXCLK_VF_CURVE_OFFSET_ERROR:=4): 'OD_GFXCLK_VF_CURVE_OFFSET_ERROR', (OD_VDD_GFX_VMAX_ERROR:=5): 'OD_VDD_GFX_VMAX_ERROR', (OD_VDD_SOC_VMAX_ERROR:=6): 'OD_VDD_SOC_VMAX_ERROR', (OD_PPT_ERROR:=7): 'OD_PPT_ERROR', (OD_FAN_MIN_PWM_ERROR:=8): 'OD_FAN_MIN_PWM_ERROR', (OD_FAN_ACOUSTIC_TARGET_ERROR:=9): 'OD_FAN_ACOUSTIC_TARGET_ERROR', (OD_FAN_ACOUSTIC_LIMIT_ERROR:=10): 'OD_FAN_ACOUSTIC_LIMIT_ERROR', (OD_FAN_TARGET_TEMP_ERROR:=11): 'OD_FAN_TARGET_TEMP_ERROR', (OD_FAN_ZERO_RPM_STOP_TEMP_ERROR:=12): 'OD_FAN_ZERO_RPM_STOP_TEMP_ERROR', (OD_FAN_CURVE_PWM_ERROR:=13): 'OD_FAN_CURVE_PWM_ERROR', (OD_FAN_CURVE_TEMP_ERROR:=14): 'OD_FAN_CURVE_TEMP_ERROR', (OD_FULL_CTRL_GFXCLK_ERROR:=15): 'OD_FULL_CTRL_GFXCLK_ERROR', (OD_FULL_CTRL_UCLK_ERROR:=16): 'OD_FULL_CTRL_UCLK_ERROR', (OD_FULL_CTRL_FCLK_ERROR:=17): 'OD_FULL_CTRL_FCLK_ERROR', (OD_FULL_CTRL_VDD_GFX_ERROR:=18): 'OD_FULL_CTRL_VDD_GFX_ERROR', (OD_FULL_CTRL_VDD_SOC_ERROR:=19): 'OD_FULL_CTRL_VDD_SOC_ERROR', (OD_TDC_ERROR:=20): 'OD_TDC_ERROR', (OD_GFXCLK_ERROR:=21): 'OD_GFXCLK_ERROR', (OD_UCLK_ERROR:=22): 'OD_UCLK_ERROR', (OD_FCLK_ERROR:=23): 'OD_FCLK_ERROR', (OD_OP_TEMP_ERROR:=24): 'OD_OP_TEMP_ERROR', (OD_OP_GFX_EDC_ERROR:=25): 'OD_OP_GFX_EDC_ERROR', (OD_OP_GFX_PCC_ERROR:=26): 'OD_OP_GFX_PCC_ERROR', (OD_POWER_FEATURE_CTRL_ERROR:=27): 'OD_POWER_FEATURE_CTRL_ERROR'}
@c.record
class OverDriveTable_t(c.Struct):
  SIZE = 156
  FeatureCtrlMask: int
  VoltageOffsetPerZoneBoundary: c.Array[ctypes.c_int16, Literal[6]]
  VddGfxVmax: int
  VddSocVmax: int
  IdlePwrSavingFeaturesCtrl: int
  RuntimePwrSavingFeaturesCtrl: int
  Padding: int
  GfxclkFoffset: int
  Padding1: int
  UclkFmin: int
  UclkFmax: int
  FclkFmin: int
  FclkFmax: int
  Ppt: int
  Tdc: int
  FanLinearPwmPoints: c.Array[ctypes.c_ubyte, Literal[6]]
  FanLinearTempPoints: c.Array[ctypes.c_ubyte, Literal[6]]
  FanMinimumPwm: int
  AcousticTargetRpmThreshold: int
  AcousticLimitRpmThreshold: int
  FanTargetTemperature: int
  FanZeroRpmEnable: int
  FanZeroRpmStopTemp: int
  FanMode: int
  MaxOpTemp: int
  AdvancedOdModeEnabled: int
  Padding2: c.Array[ctypes.c_ubyte, Literal[3]]
  GfxVoltageFullCtrlMode: int
  SocVoltageFullCtrlMode: int
  GfxclkFullCtrlMode: int
  UclkFullCtrlMode: int
  FclkFullCtrlMode: int
  Padding3: int
  GfxEdc: int
  GfxPccLimitControl: int
  GfxclkFmaxVmax: int
  GfxclkFmaxVmaxTemperature: int
  Padding4: c.Array[ctypes.c_ubyte, Literal[1]]
  Spare: c.Array[ctypes.c_uint32, Literal[9]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
int16_t: TypeAlias = ctypes.c_int16
OverDriveTable_t.register_fields([('FeatureCtrlMask', uint32_t, 0), ('VoltageOffsetPerZoneBoundary', c.Array[int16_t, Literal[6]], 4), ('VddGfxVmax', uint16_t, 16), ('VddSocVmax', uint16_t, 18), ('IdlePwrSavingFeaturesCtrl', uint8_t, 20), ('RuntimePwrSavingFeaturesCtrl', uint8_t, 21), ('Padding', uint16_t, 22), ('GfxclkFoffset', int16_t, 24), ('Padding1', uint16_t, 26), ('UclkFmin', uint16_t, 28), ('UclkFmax', uint16_t, 30), ('FclkFmin', uint16_t, 32), ('FclkFmax', uint16_t, 34), ('Ppt', int16_t, 36), ('Tdc', int16_t, 38), ('FanLinearPwmPoints', c.Array[uint8_t, Literal[6]], 40), ('FanLinearTempPoints', c.Array[uint8_t, Literal[6]], 46), ('FanMinimumPwm', uint16_t, 52), ('AcousticTargetRpmThreshold', uint16_t, 54), ('AcousticLimitRpmThreshold', uint16_t, 56), ('FanTargetTemperature', uint16_t, 58), ('FanZeroRpmEnable', uint8_t, 60), ('FanZeroRpmStopTemp', uint8_t, 61), ('FanMode', uint8_t, 62), ('MaxOpTemp', uint8_t, 63), ('AdvancedOdModeEnabled', uint8_t, 64), ('Padding2', c.Array[uint8_t, Literal[3]], 65), ('GfxVoltageFullCtrlMode', uint16_t, 68), ('SocVoltageFullCtrlMode', uint16_t, 70), ('GfxclkFullCtrlMode', uint16_t, 72), ('UclkFullCtrlMode', uint16_t, 74), ('FclkFullCtrlMode', uint16_t, 76), ('Padding3', uint16_t, 78), ('GfxEdc', int16_t, 80), ('GfxPccLimitControl', int16_t, 82), ('GfxclkFmaxVmax', uint16_t, 84), ('GfxclkFmaxVmaxTemperature', uint8_t, 86), ('Padding4', c.Array[uint8_t, Literal[1]], 87), ('Spare', c.Array[uint32_t, Literal[9]], 88), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 124)])
@c.record
class OverDriveTableExternal_t(c.Struct):
  SIZE = 156
  OverDriveTable: OverDriveTable_t
OverDriveTableExternal_t.register_fields([('OverDriveTable', OverDriveTable_t, 0)])
@c.record
class OverDriveLimits_t(c.Struct):
  SIZE = 96
  FeatureCtrlMask: int
  VoltageOffsetPerZoneBoundary: c.Array[ctypes.c_int16, Literal[6]]
  VddGfxVmax: int
  VddSocVmax: int
  GfxclkFoffset: int
  Padding: int
  UclkFmin: int
  UclkFmax: int
  FclkFmin: int
  FclkFmax: int
  Ppt: int
  Tdc: int
  FanLinearPwmPoints: c.Array[ctypes.c_ubyte, Literal[6]]
  FanLinearTempPoints: c.Array[ctypes.c_ubyte, Literal[6]]
  FanMinimumPwm: int
  AcousticTargetRpmThreshold: int
  AcousticLimitRpmThreshold: int
  FanTargetTemperature: int
  FanZeroRpmEnable: int
  MaxOpTemp: int
  Padding1: c.Array[ctypes.c_ubyte, Literal[2]]
  GfxVoltageFullCtrlMode: int
  SocVoltageFullCtrlMode: int
  GfxclkFullCtrlMode: int
  UclkFullCtrlMode: int
  FclkFullCtrlMode: int
  GfxEdc: int
  GfxPccLimitControl: int
  Padding2: int
  Spare: c.Array[ctypes.c_uint32, Literal[5]]
OverDriveLimits_t.register_fields([('FeatureCtrlMask', uint32_t, 0), ('VoltageOffsetPerZoneBoundary', c.Array[int16_t, Literal[6]], 4), ('VddGfxVmax', uint16_t, 16), ('VddSocVmax', uint16_t, 18), ('GfxclkFoffset', int16_t, 20), ('Padding', uint16_t, 22), ('UclkFmin', uint16_t, 24), ('UclkFmax', uint16_t, 26), ('FclkFmin', uint16_t, 28), ('FclkFmax', uint16_t, 30), ('Ppt', int16_t, 32), ('Tdc', int16_t, 34), ('FanLinearPwmPoints', c.Array[uint8_t, Literal[6]], 36), ('FanLinearTempPoints', c.Array[uint8_t, Literal[6]], 42), ('FanMinimumPwm', uint16_t, 48), ('AcousticTargetRpmThreshold', uint16_t, 50), ('AcousticLimitRpmThreshold', uint16_t, 52), ('FanTargetTemperature', uint16_t, 54), ('FanZeroRpmEnable', uint8_t, 56), ('MaxOpTemp', uint8_t, 57), ('Padding1', c.Array[uint8_t, Literal[2]], 58), ('GfxVoltageFullCtrlMode', uint16_t, 60), ('SocVoltageFullCtrlMode', uint16_t, 62), ('GfxclkFullCtrlMode', uint16_t, 64), ('UclkFullCtrlMode', uint16_t, 66), ('FclkFullCtrlMode', uint16_t, 68), ('GfxEdc', int16_t, 70), ('GfxPccLimitControl', int16_t, 72), ('Padding2', int16_t, 74), ('Spare', c.Array[uint32_t, Literal[5]], 76)])
BOARD_GPIO_TYPE_e: dict[int, str] = {(BOARD_GPIO_SMUIO_0:=0): 'BOARD_GPIO_SMUIO_0', (BOARD_GPIO_SMUIO_1:=1): 'BOARD_GPIO_SMUIO_1', (BOARD_GPIO_SMUIO_2:=2): 'BOARD_GPIO_SMUIO_2', (BOARD_GPIO_SMUIO_3:=3): 'BOARD_GPIO_SMUIO_3', (BOARD_GPIO_SMUIO_4:=4): 'BOARD_GPIO_SMUIO_4', (BOARD_GPIO_SMUIO_5:=5): 'BOARD_GPIO_SMUIO_5', (BOARD_GPIO_SMUIO_6:=6): 'BOARD_GPIO_SMUIO_6', (BOARD_GPIO_SMUIO_7:=7): 'BOARD_GPIO_SMUIO_7', (BOARD_GPIO_SMUIO_8:=8): 'BOARD_GPIO_SMUIO_8', (BOARD_GPIO_SMUIO_9:=9): 'BOARD_GPIO_SMUIO_9', (BOARD_GPIO_SMUIO_10:=10): 'BOARD_GPIO_SMUIO_10', (BOARD_GPIO_SMUIO_11:=11): 'BOARD_GPIO_SMUIO_11', (BOARD_GPIO_SMUIO_12:=12): 'BOARD_GPIO_SMUIO_12', (BOARD_GPIO_SMUIO_13:=13): 'BOARD_GPIO_SMUIO_13', (BOARD_GPIO_SMUIO_14:=14): 'BOARD_GPIO_SMUIO_14', (BOARD_GPIO_SMUIO_15:=15): 'BOARD_GPIO_SMUIO_15', (BOARD_GPIO_SMUIO_16:=16): 'BOARD_GPIO_SMUIO_16', (BOARD_GPIO_SMUIO_17:=17): 'BOARD_GPIO_SMUIO_17', (BOARD_GPIO_SMUIO_18:=18): 'BOARD_GPIO_SMUIO_18', (BOARD_GPIO_SMUIO_19:=19): 'BOARD_GPIO_SMUIO_19', (BOARD_GPIO_SMUIO_20:=20): 'BOARD_GPIO_SMUIO_20', (BOARD_GPIO_SMUIO_21:=21): 'BOARD_GPIO_SMUIO_21', (BOARD_GPIO_SMUIO_22:=22): 'BOARD_GPIO_SMUIO_22', (BOARD_GPIO_SMUIO_23:=23): 'BOARD_GPIO_SMUIO_23', (BOARD_GPIO_SMUIO_24:=24): 'BOARD_GPIO_SMUIO_24', (BOARD_GPIO_SMUIO_25:=25): 'BOARD_GPIO_SMUIO_25', (BOARD_GPIO_SMUIO_26:=26): 'BOARD_GPIO_SMUIO_26', (BOARD_GPIO_SMUIO_27:=27): 'BOARD_GPIO_SMUIO_27', (BOARD_GPIO_SMUIO_28:=28): 'BOARD_GPIO_SMUIO_28', (BOARD_GPIO_SMUIO_29:=29): 'BOARD_GPIO_SMUIO_29', (BOARD_GPIO_SMUIO_30:=30): 'BOARD_GPIO_SMUIO_30', (BOARD_GPIO_SMUIO_31:=31): 'BOARD_GPIO_SMUIO_31', (MAX_BOARD_GPIO_SMUIO_NUM:=32): 'MAX_BOARD_GPIO_SMUIO_NUM', (BOARD_GPIO_DC_GEN_A:=33): 'BOARD_GPIO_DC_GEN_A', (BOARD_GPIO_DC_GEN_B:=34): 'BOARD_GPIO_DC_GEN_B', (BOARD_GPIO_DC_GEN_C:=35): 'BOARD_GPIO_DC_GEN_C', (BOARD_GPIO_DC_GEN_D:=36): 'BOARD_GPIO_DC_GEN_D', (BOARD_GPIO_DC_GEN_E:=37): 'BOARD_GPIO_DC_GEN_E', (BOARD_GPIO_DC_GEN_F:=38): 'BOARD_GPIO_DC_GEN_F', (BOARD_GPIO_DC_GEN_G:=39): 'BOARD_GPIO_DC_GEN_G', (BOARD_GPIO_DC_GENLK_CLK:=40): 'BOARD_GPIO_DC_GENLK_CLK', (BOARD_GPIO_DC_GENLK_VSYNC:=41): 'BOARD_GPIO_DC_GENLK_VSYNC', (BOARD_GPIO_DC_SWAPLOCK_A:=42): 'BOARD_GPIO_DC_SWAPLOCK_A', (BOARD_GPIO_DC_SWAPLOCK_B:=43): 'BOARD_GPIO_DC_SWAPLOCK_B', (MAX_BOARD_DC_GPIO_NUM:=44): 'MAX_BOARD_DC_GPIO_NUM', (BOARD_GPIO_LV_EN:=45): 'BOARD_GPIO_LV_EN'}
@c.record
class BootValues_t(c.Struct):
  SIZE = 124
  InitImuClk: int
  InitSocclk: int
  InitMpioclk: int
  InitSmnclk: int
  InitDispClk: int
  InitDppClk: int
  InitDprefclk: int
  InitDcfclk: int
  InitDtbclk: int
  InitDbguSocClk: int
  InitGfxclk_bypass: int
  InitMp1clk: int
  InitLclk: int
  InitDbguBacoClk: int
  InitBaco400clk: int
  InitBaco1200clk_bypass: int
  InitBaco700clk_bypass: int
  InitBaco500clk: int
  InitDclk0: int
  InitVclk0: int
  InitFclk: int
  Padding1: int
  InitUclkLevel: int
  Padding: c.Array[ctypes.c_ubyte, Literal[3]]
  InitVcoFreqPll0: int
  InitVcoFreqPll1: int
  InitVcoFreqPll2: int
  InitVcoFreqPll3: int
  InitVcoFreqPll4: int
  InitVcoFreqPll5: int
  InitVcoFreqPll6: int
  InitVcoFreqPll7: int
  InitVcoFreqPll8: int
  InitGfx: int
  InitSoc: int
  InitVddIoMem: int
  InitVddCiMem: int
  Spare: c.Array[ctypes.c_uint32, Literal[8]]
BootValues_t.register_fields([('InitImuClk', uint16_t, 0), ('InitSocclk', uint16_t, 2), ('InitMpioclk', uint16_t, 4), ('InitSmnclk', uint16_t, 6), ('InitDispClk', uint16_t, 8), ('InitDppClk', uint16_t, 10), ('InitDprefclk', uint16_t, 12), ('InitDcfclk', uint16_t, 14), ('InitDtbclk', uint16_t, 16), ('InitDbguSocClk', uint16_t, 18), ('InitGfxclk_bypass', uint16_t, 20), ('InitMp1clk', uint16_t, 22), ('InitLclk', uint16_t, 24), ('InitDbguBacoClk', uint16_t, 26), ('InitBaco400clk', uint16_t, 28), ('InitBaco1200clk_bypass', uint16_t, 30), ('InitBaco700clk_bypass', uint16_t, 32), ('InitBaco500clk', uint16_t, 34), ('InitDclk0', uint16_t, 36), ('InitVclk0', uint16_t, 38), ('InitFclk', uint16_t, 40), ('Padding1', uint16_t, 42), ('InitUclkLevel', uint8_t, 44), ('Padding', c.Array[uint8_t, Literal[3]], 45), ('InitVcoFreqPll0', uint32_t, 48), ('InitVcoFreqPll1', uint32_t, 52), ('InitVcoFreqPll2', uint32_t, 56), ('InitVcoFreqPll3', uint32_t, 60), ('InitVcoFreqPll4', uint32_t, 64), ('InitVcoFreqPll5', uint32_t, 68), ('InitVcoFreqPll6', uint32_t, 72), ('InitVcoFreqPll7', uint32_t, 76), ('InitVcoFreqPll8', uint32_t, 80), ('InitGfx', uint16_t, 84), ('InitSoc', uint16_t, 86), ('InitVddIoMem', uint16_t, 88), ('InitVddCiMem', uint16_t, 90), ('Spare', c.Array[uint32_t, Literal[8]], 92)])
@c.record
class MsgLimits_t(c.Struct):
  SIZE = 116
  Power: c.Array[c.Array[ctypes.c_uint16, Literal[2]], Literal[4]]
  Tdc: c.Array[ctypes.c_uint16, Literal[2]]
  Temperature: c.Array[ctypes.c_uint16, Literal[12]]
  PwmLimitMin: int
  PwmLimitMax: int
  FanTargetTemperature: int
  Spare1: c.Array[ctypes.c_ubyte, Literal[1]]
  AcousticTargetRpmThresholdMin: int
  AcousticTargetRpmThresholdMax: int
  AcousticLimitRpmThresholdMin: int
  AcousticLimitRpmThresholdMax: int
  PccLimitMin: int
  PccLimitMax: int
  FanStopTempMin: int
  FanStopTempMax: int
  FanStartTempMin: int
  FanStartTempMax: int
  PowerMinPpt0: c.Array[ctypes.c_uint16, Literal[2]]
  Spare: c.Array[ctypes.c_uint32, Literal[11]]
MsgLimits_t.register_fields([('Power', c.Array[c.Array[uint16_t, Literal[2]], Literal[4]], 0), ('Tdc', c.Array[uint16_t, Literal[2]], 16), ('Temperature', c.Array[uint16_t, Literal[12]], 20), ('PwmLimitMin', uint8_t, 44), ('PwmLimitMax', uint8_t, 45), ('FanTargetTemperature', uint8_t, 46), ('Spare1', c.Array[uint8_t, Literal[1]], 47), ('AcousticTargetRpmThresholdMin', uint16_t, 48), ('AcousticTargetRpmThresholdMax', uint16_t, 50), ('AcousticLimitRpmThresholdMin', uint16_t, 52), ('AcousticLimitRpmThresholdMax', uint16_t, 54), ('PccLimitMin', uint16_t, 56), ('PccLimitMax', uint16_t, 58), ('FanStopTempMin', uint16_t, 60), ('FanStopTempMax', uint16_t, 62), ('FanStartTempMin', uint16_t, 64), ('FanStartTempMax', uint16_t, 66), ('PowerMinPpt0', c.Array[uint16_t, Literal[2]], 68), ('Spare', c.Array[uint32_t, Literal[11]], 72)])
@c.record
class DriverReportedClocks_t(c.Struct):
  SIZE = 28
  BaseClockAc: int
  GameClockAc: int
  BoostClockAc: int
  BaseClockDc: int
  GameClockDc: int
  BoostClockDc: int
  MaxReportedClock: int
  Padding: int
  Reserved: c.Array[ctypes.c_uint32, Literal[3]]
DriverReportedClocks_t.register_fields([('BaseClockAc', uint16_t, 0), ('GameClockAc', uint16_t, 2), ('BoostClockAc', uint16_t, 4), ('BaseClockDc', uint16_t, 6), ('GameClockDc', uint16_t, 8), ('BoostClockDc', uint16_t, 10), ('MaxReportedClock', uint16_t, 12), ('Padding', uint16_t, 14), ('Reserved', c.Array[uint32_t, Literal[3]], 16)])
@c.record
class AvfsDcBtcParams_t(c.Struct):
  SIZE = 20
  DcBtcEnabled: int
  Padding: c.Array[ctypes.c_ubyte, Literal[3]]
  DcTol: int
  DcBtcGb: int
  DcBtcMin: int
  DcBtcMax: int
  DcBtcGbScalar: LinearInt_t
AvfsDcBtcParams_t.register_fields([('DcBtcEnabled', uint8_t, 0), ('Padding', c.Array[uint8_t, Literal[3]], 1), ('DcTol', uint16_t, 4), ('DcBtcGb', uint16_t, 6), ('DcBtcMin', uint16_t, 8), ('DcBtcMax', uint16_t, 10), ('DcBtcGbScalar', LinearInt_t, 12)])
@c.record
class AvfsFuseOverride_t(c.Struct):
  SIZE = 56
  AvfsTemp: c.Array[ctypes.c_uint16, Literal[2]]
  VftFMin: int
  VInversion: int
  qVft: c.Array[QuadraticInt_t, Literal[2]]
  qAvfsGb: QuadraticInt_t
  qAvfsGb2: QuadraticInt_t
AvfsFuseOverride_t.register_fields([('AvfsTemp', c.Array[uint16_t, Literal[2]], 0), ('VftFMin', uint16_t, 4), ('VInversion', uint16_t, 6), ('qVft', c.Array[QuadraticInt_t, Literal[2]], 8), ('qAvfsGb', QuadraticInt_t, 32), ('qAvfsGb2', QuadraticInt_t, 44)])
@c.record
class PFE_Settings_t(c.Struct):
  SIZE = 28
  Version: int
  Spare8: c.Array[ctypes.c_ubyte, Literal[3]]
  FeaturesToRun: c.Array[ctypes.c_uint32, Literal[2]]
  FwDStateMask: int
  DebugOverrides: int
  Spare: c.Array[ctypes.c_uint32, Literal[2]]
PFE_Settings_t.register_fields([('Version', uint8_t, 0), ('Spare8', c.Array[uint8_t, Literal[3]], 1), ('FeaturesToRun', c.Array[uint32_t, Literal[2]], 4), ('FwDStateMask', uint32_t, 12), ('DebugOverrides', uint32_t, 16), ('Spare', c.Array[uint32_t, Literal[2]], 20)])
@c.record
class SkuTable_t(c.Struct):
  SIZE = 3552
  Version: int
  TotalPowerConfig: int
  CustomerVariant: int
  MemoryTemperatureTypeMask: int
  SmartShiftVersion: int
  SocketPowerLimitSpare: c.Array[ctypes.c_ubyte, Literal[10]]
  EnableLegacyPptLimit: int
  UseInputTelemetry: int
  SmartShiftMinReportedPptinDcs: int
  PaddingPpt: c.Array[ctypes.c_ubyte, Literal[7]]
  HwCtfTempLimit: int
  PaddingInfra: int
  FitControllerFailureRateLimit: int
  FitControllerGfxDutyCycle: int
  FitControllerSocDutyCycle: int
  FitControllerSocOffset: int
  GfxApccPlusResidencyLimit: int
  ThrottlerControlMask: int
  UlvVoltageOffset: c.Array[ctypes.c_uint16, Literal[2]]
  Padding: c.Array[ctypes.c_ubyte, Literal[2]]
  DeepUlvVoltageOffsetSoc: int
  DefaultMaxVoltage: c.Array[ctypes.c_uint16, Literal[2]]
  BoostMaxVoltage: c.Array[ctypes.c_uint16, Literal[2]]
  VminTempHystersis: c.Array[ctypes.c_int16, Literal[2]]
  VminTempThreshold: c.Array[ctypes.c_int16, Literal[2]]
  Vmin_Hot_T0: c.Array[ctypes.c_uint16, Literal[2]]
  Vmin_Cold_T0: c.Array[ctypes.c_uint16, Literal[2]]
  Vmin_Hot_Eol: c.Array[ctypes.c_uint16, Literal[2]]
  Vmin_Cold_Eol: c.Array[ctypes.c_uint16, Literal[2]]
  Vmin_Aging_Offset: c.Array[ctypes.c_uint16, Literal[2]]
  Spare_Vmin_Plat_Offset_Hot: c.Array[ctypes.c_uint16, Literal[2]]
  Spare_Vmin_Plat_Offset_Cold: c.Array[ctypes.c_uint16, Literal[2]]
  VcBtcFixedVminAgingOffset: c.Array[ctypes.c_uint16, Literal[2]]
  VcBtcVmin2PsmDegrationGb: c.Array[ctypes.c_uint16, Literal[2]]
  VcBtcPsmA: c.Array[ctypes.c_uint32, Literal[2]]
  VcBtcPsmB: c.Array[ctypes.c_uint32, Literal[2]]
  VcBtcVminA: c.Array[ctypes.c_uint32, Literal[2]]
  VcBtcVminB: c.Array[ctypes.c_uint32, Literal[2]]
  PerPartVminEnabled: c.Array[ctypes.c_ubyte, Literal[2]]
  VcBtcEnabled: c.Array[ctypes.c_ubyte, Literal[2]]
  SocketPowerLimitAcTau: c.Array[ctypes.c_uint16, Literal[4]]
  SocketPowerLimitDcTau: c.Array[ctypes.c_uint16, Literal[4]]
  Gfx_Vmin_droop: QuadraticInt_t
  Soc_Vmin_droop: QuadraticInt_t
  SpareVmin: c.Array[ctypes.c_uint32, Literal[6]]
  DpmDescriptor: c.Array[DpmDescriptor_t, Literal[11]]
  FreqTableGfx: c.Array[ctypes.c_uint16, Literal[16]]
  FreqTableVclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableSocclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableUclk: c.Array[ctypes.c_uint16, Literal[6]]
  FreqTableShadowUclk: c.Array[ctypes.c_uint16, Literal[6]]
  FreqTableDispclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDppClk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDprefclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDcfclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDtbclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableFclk: c.Array[ctypes.c_uint16, Literal[8]]
  DcModeMaxFreq: c.Array[ctypes.c_uint32, Literal[11]]
  GfxclkAibFmax: int
  GfxDpmPadding: int
  GfxclkFgfxoffEntry: int
  GfxclkFgfxoffExitImu: int
  GfxclkFgfxoffExitRlc: int
  GfxclkThrottleClock: int
  EnableGfxPowerStagesGpio: int
  GfxIdlePadding: int
  SmsRepairWRCKClkDivEn: int
  SmsRepairWRCKClkDivVal: int
  GfxOffEntryEarlyMGCGEn: int
  GfxOffEntryForceCGCGEn: int
  GfxOffEntryForceCGCGDelayEn: int
  GfxOffEntryForceCGCGDelayVal: int
  GfxclkFreqGfxUlv: int
  GfxIdlePadding2: c.Array[ctypes.c_ubyte, Literal[2]]
  GfxOffEntryHysteresis: int
  GfxoffSpare: c.Array[ctypes.c_uint32, Literal[15]]
  DfllMstrOscConfigA: int
  DfllSlvOscConfigA: int
  DfllBtcMasterScalerM: int
  DfllBtcMasterScalerB: int
  DfllBtcSlaveScalerM: int
  DfllBtcSlaveScalerB: int
  DfllPccAsWaitCtrl: int
  DfllPccAsStepCtrl: int
  GfxDfllSpare: c.Array[ctypes.c_uint32, Literal[9]]
  DvoPsmDownThresholdVoltage: int
  DvoPsmUpThresholdVoltage: int
  DvoFmaxLowScaler: int
  PaddingDcs: int
  DcsMinGfxOffTime: int
  DcsMaxGfxOffTime: int
  DcsMinCreditAccum: int
  DcsExitHysteresis: int
  DcsTimeout: int
  DcsPfGfxFopt: int
  DcsPfUclkFopt: int
  FoptEnabled: int
  DcsSpare2: c.Array[ctypes.c_ubyte, Literal[3]]
  DcsFoptM: int
  DcsFoptB: int
  DcsSpare: c.Array[ctypes.c_uint32, Literal[9]]
  UseStrobeModeOptimizations: int
  PaddingMem: c.Array[ctypes.c_ubyte, Literal[3]]
  UclkDpmPstates: c.Array[ctypes.c_ubyte, Literal[6]]
  UclkDpmShadowPstates: c.Array[ctypes.c_ubyte, Literal[6]]
  FreqTableUclkDiv: c.Array[ctypes.c_ubyte, Literal[6]]
  FreqTableShadowUclkDiv: c.Array[ctypes.c_ubyte, Literal[6]]
  MemVmempVoltage: c.Array[ctypes.c_uint16, Literal[6]]
  MemVddioVoltage: c.Array[ctypes.c_uint16, Literal[6]]
  DalDcModeMaxUclkFreq: int
  PaddingsMem: c.Array[ctypes.c_ubyte, Literal[2]]
  PaddingFclk: int
  PcieGenSpeed: c.Array[ctypes.c_ubyte, Literal[3]]
  PcieLaneCount: c.Array[ctypes.c_ubyte, Literal[3]]
  LclkFreq: c.Array[ctypes.c_uint16, Literal[3]]
  OverrideGfxAvfsFuses: int
  GfxAvfsPadding: c.Array[ctypes.c_ubyte, Literal[1]]
  DroopGBStDev: int
  SocHwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[32]]
  GfxL2HwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[32]]
  PsmDidt_Vcross: c.Array[ctypes.c_uint16, Literal[2]]
  PsmDidt_StaticDroop_A: c.Array[ctypes.c_uint32, Literal[3]]
  PsmDidt_StaticDroop_B: c.Array[ctypes.c_uint32, Literal[3]]
  PsmDidt_DynDroop_A: c.Array[ctypes.c_uint32, Literal[3]]
  PsmDidt_DynDroop_B: c.Array[ctypes.c_uint32, Literal[3]]
  spare_HwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[19]]
  SocCommonRtAvfs: c.Array[ctypes.c_uint32, Literal[13]]
  GfxCommonRtAvfs: c.Array[ctypes.c_uint32, Literal[13]]
  SocFwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[19]]
  GfxL2FwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[19]]
  spare_FwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[19]]
  Soc_Droop_PWL_F: c.Array[ctypes.c_uint32, Literal[5]]
  Soc_Droop_PWL_a: c.Array[ctypes.c_uint32, Literal[5]]
  Soc_Droop_PWL_b: c.Array[ctypes.c_uint32, Literal[5]]
  Soc_Droop_PWL_c: c.Array[ctypes.c_uint32, Literal[5]]
  Gfx_Droop_PWL_F: c.Array[ctypes.c_uint32, Literal[5]]
  Gfx_Droop_PWL_a: c.Array[ctypes.c_uint32, Literal[5]]
  Gfx_Droop_PWL_b: c.Array[ctypes.c_uint32, Literal[5]]
  Gfx_Droop_PWL_c: c.Array[ctypes.c_uint32, Literal[5]]
  Gfx_Static_PWL_Offset: c.Array[ctypes.c_uint32, Literal[5]]
  Soc_Static_PWL_Offset: c.Array[ctypes.c_uint32, Literal[5]]
  dGbV_dT_vmin: int
  dGbV_dT_vmax: int
  PaddingV2F: c.Array[ctypes.c_uint32, Literal[4]]
  DcBtcGfxParams: AvfsDcBtcParams_t
  SSCurve_GFX: QuadraticInt_t
  GfxAvfsSpare: c.Array[ctypes.c_uint32, Literal[29]]
  OverrideSocAvfsFuses: int
  MinSocAvfsRevision: int
  SocAvfsPadding: c.Array[ctypes.c_ubyte, Literal[2]]
  SocAvfsFuseOverride: c.Array[AvfsFuseOverride_t, Literal[1]]
  dBtcGbSoc: c.Array[DroopInt_t, Literal[1]]
  qAgingGb: c.Array[LinearInt_t, Literal[1]]
  qStaticVoltageOffset: c.Array[QuadraticInt_t, Literal[1]]
  DcBtcSocParams: c.Array[AvfsDcBtcParams_t, Literal[1]]
  SSCurve_SOC: QuadraticInt_t
  SocAvfsSpare: c.Array[ctypes.c_uint32, Literal[29]]
  BootValues: BootValues_t
  DriverReportedClocks: DriverReportedClocks_t
  MsgLimits: MsgLimits_t
  OverDriveLimitsBasicMin: OverDriveLimits_t
  OverDriveLimitsBasicMax: OverDriveLimits_t
  OverDriveLimitsAdvancedMin: OverDriveLimits_t
  OverDriveLimitsAdvancedMax: OverDriveLimits_t
  TotalBoardPowerSupport: int
  TotalBoardPowerPadding: c.Array[ctypes.c_ubyte, Literal[1]]
  TotalBoardPowerRoc: int
  qFeffCoeffGameClock: c.Array[QuadraticInt_t, Literal[2]]
  qFeffCoeffBaseClock: c.Array[QuadraticInt_t, Literal[2]]
  qFeffCoeffBoostClock: c.Array[QuadraticInt_t, Literal[2]]
  AptUclkGfxclkLookup: c.Array[c.Array[ctypes.c_int32, Literal[6]], Literal[2]]
  AptUclkGfxclkLookupHyst: c.Array[c.Array[ctypes.c_uint32, Literal[6]], Literal[2]]
  AptPadding: int
  GfxXvminDidtDroopThresh: QuadraticInt_t
  GfxXvminDidtResetDDWait: int
  GfxXvminDidtClkStopWait: int
  GfxXvminDidtFcsStepCtrl: int
  GfxXvminDidtFcsWaitCtrl: int
  PsmModeEnabled: int
  P2v_a: int
  P2v_b: int
  P2v_c: int
  T2p_a: int
  T2p_b: int
  T2p_c: int
  P2vTemp: int
  PsmDidtStaticSettings: QuadraticInt_t
  PsmDidtDynamicSettings: QuadraticInt_t
  PsmDidtAvgDiv: int
  PsmDidtForceStall: int
  PsmDidtReleaseTimer: int
  PsmDidtStallPattern: int
  CacEdcCacLeakageC0: int
  CacEdcCacLeakageC1: int
  CacEdcCacLeakageC2: int
  CacEdcCacLeakageC3: int
  CacEdcCacLeakageC4: int
  CacEdcCacLeakageC5: int
  CacEdcGfxClkScalar: int
  CacEdcGfxClkIntercept: int
  CacEdcCac_m: int
  CacEdcCac_b: int
  CacEdcCurrLimitGuardband: int
  CacEdcDynToTotalCacRatio: int
  XVmin_Gfx_EdcThreshScalar: int
  XVmin_Gfx_EdcEnableFreq: int
  XVmin_Gfx_EdcPccAsStepCtrl: int
  XVmin_Gfx_EdcPccAsWaitCtrl: int
  XVmin_Gfx_EdcThreshold: int
  XVmin_Gfx_EdcFiltHysWaitCtrl: int
  XVmin_Soc_EdcThreshScalar: int
  XVmin_Soc_EdcEnableFreq: int
  XVmin_Soc_EdcThreshold: int
  XVmin_Soc_EdcStepUpTime: int
  XVmin_Soc_EdcStepDownTime: int
  XVmin_Soc_EdcInitPccStep: int
  PaddingSocEdc: c.Array[ctypes.c_ubyte, Literal[3]]
  GfxXvminFuseOverride: int
  SocXvminFuseOverride: int
  PaddingXvminFuseOverride: c.Array[ctypes.c_ubyte, Literal[2]]
  GfxXvminFddTempLow: int
  GfxXvminFddTempHigh: int
  SocXvminFddTempLow: int
  SocXvminFddTempHigh: int
  GfxXvminFddVolt0: int
  GfxXvminFddVolt1: int
  GfxXvminFddVolt2: int
  SocXvminFddVolt0: int
  SocXvminFddVolt1: int
  SocXvminFddVolt2: int
  GfxXvminDsFddDsm: c.Array[ctypes.c_uint16, Literal[6]]
  GfxXvminEdcFddDsm: c.Array[ctypes.c_uint16, Literal[6]]
  SocXvminEdcFddDsm: c.Array[ctypes.c_uint16, Literal[6]]
  Spare: int
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
int32_t: TypeAlias = ctypes.c_int32
SkuTable_t.register_fields([('Version', uint32_t, 0), ('TotalPowerConfig', uint8_t, 4), ('CustomerVariant', uint8_t, 5), ('MemoryTemperatureTypeMask', uint8_t, 6), ('SmartShiftVersion', uint8_t, 7), ('SocketPowerLimitSpare', c.Array[uint8_t, Literal[10]], 8), ('EnableLegacyPptLimit', uint8_t, 18), ('UseInputTelemetry', uint8_t, 19), ('SmartShiftMinReportedPptinDcs', uint8_t, 20), ('PaddingPpt', c.Array[uint8_t, Literal[7]], 21), ('HwCtfTempLimit', uint16_t, 28), ('PaddingInfra', uint16_t, 30), ('FitControllerFailureRateLimit', uint32_t, 32), ('FitControllerGfxDutyCycle', uint32_t, 36), ('FitControllerSocDutyCycle', uint32_t, 40), ('FitControllerSocOffset', uint32_t, 44), ('GfxApccPlusResidencyLimit', uint32_t, 48), ('ThrottlerControlMask', uint32_t, 52), ('UlvVoltageOffset', c.Array[uint16_t, Literal[2]], 56), ('Padding', c.Array[uint8_t, Literal[2]], 60), ('DeepUlvVoltageOffsetSoc', uint16_t, 62), ('DefaultMaxVoltage', c.Array[uint16_t, Literal[2]], 64), ('BoostMaxVoltage', c.Array[uint16_t, Literal[2]], 68), ('VminTempHystersis', c.Array[int16_t, Literal[2]], 72), ('VminTempThreshold', c.Array[int16_t, Literal[2]], 76), ('Vmin_Hot_T0', c.Array[uint16_t, Literal[2]], 80), ('Vmin_Cold_T0', c.Array[uint16_t, Literal[2]], 84), ('Vmin_Hot_Eol', c.Array[uint16_t, Literal[2]], 88), ('Vmin_Cold_Eol', c.Array[uint16_t, Literal[2]], 92), ('Vmin_Aging_Offset', c.Array[uint16_t, Literal[2]], 96), ('Spare_Vmin_Plat_Offset_Hot', c.Array[uint16_t, Literal[2]], 100), ('Spare_Vmin_Plat_Offset_Cold', c.Array[uint16_t, Literal[2]], 104), ('VcBtcFixedVminAgingOffset', c.Array[uint16_t, Literal[2]], 108), ('VcBtcVmin2PsmDegrationGb', c.Array[uint16_t, Literal[2]], 112), ('VcBtcPsmA', c.Array[uint32_t, Literal[2]], 116), ('VcBtcPsmB', c.Array[uint32_t, Literal[2]], 124), ('VcBtcVminA', c.Array[uint32_t, Literal[2]], 132), ('VcBtcVminB', c.Array[uint32_t, Literal[2]], 140), ('PerPartVminEnabled', c.Array[uint8_t, Literal[2]], 148), ('VcBtcEnabled', c.Array[uint8_t, Literal[2]], 150), ('SocketPowerLimitAcTau', c.Array[uint16_t, Literal[4]], 152), ('SocketPowerLimitDcTau', c.Array[uint16_t, Literal[4]], 160), ('Gfx_Vmin_droop', QuadraticInt_t, 168), ('Soc_Vmin_droop', QuadraticInt_t, 180), ('SpareVmin', c.Array[uint32_t, Literal[6]], 192), ('DpmDescriptor', c.Array[DpmDescriptor_t, Literal[11]], 216), ('FreqTableGfx', c.Array[uint16_t, Literal[16]], 568), ('FreqTableVclk', c.Array[uint16_t, Literal[8]], 600), ('FreqTableDclk', c.Array[uint16_t, Literal[8]], 616), ('FreqTableSocclk', c.Array[uint16_t, Literal[8]], 632), ('FreqTableUclk', c.Array[uint16_t, Literal[6]], 648), ('FreqTableShadowUclk', c.Array[uint16_t, Literal[6]], 660), ('FreqTableDispclk', c.Array[uint16_t, Literal[8]], 672), ('FreqTableDppClk', c.Array[uint16_t, Literal[8]], 688), ('FreqTableDprefclk', c.Array[uint16_t, Literal[8]], 704), ('FreqTableDcfclk', c.Array[uint16_t, Literal[8]], 720), ('FreqTableDtbclk', c.Array[uint16_t, Literal[8]], 736), ('FreqTableFclk', c.Array[uint16_t, Literal[8]], 752), ('DcModeMaxFreq', c.Array[uint32_t, Literal[11]], 768), ('GfxclkAibFmax', uint16_t, 812), ('GfxDpmPadding', uint16_t, 814), ('GfxclkFgfxoffEntry', uint16_t, 816), ('GfxclkFgfxoffExitImu', uint16_t, 818), ('GfxclkFgfxoffExitRlc', uint16_t, 820), ('GfxclkThrottleClock', uint16_t, 822), ('EnableGfxPowerStagesGpio', uint8_t, 824), ('GfxIdlePadding', uint8_t, 825), ('SmsRepairWRCKClkDivEn', uint8_t, 826), ('SmsRepairWRCKClkDivVal', uint8_t, 827), ('GfxOffEntryEarlyMGCGEn', uint8_t, 828), ('GfxOffEntryForceCGCGEn', uint8_t, 829), ('GfxOffEntryForceCGCGDelayEn', uint8_t, 830), ('GfxOffEntryForceCGCGDelayVal', uint8_t, 831), ('GfxclkFreqGfxUlv', uint16_t, 832), ('GfxIdlePadding2', c.Array[uint8_t, Literal[2]], 834), ('GfxOffEntryHysteresis', uint32_t, 836), ('GfxoffSpare', c.Array[uint32_t, Literal[15]], 840), ('DfllMstrOscConfigA', uint16_t, 900), ('DfllSlvOscConfigA', uint16_t, 902), ('DfllBtcMasterScalerM', uint32_t, 904), ('DfllBtcMasterScalerB', int32_t, 908), ('DfllBtcSlaveScalerM', uint32_t, 912), ('DfllBtcSlaveScalerB', int32_t, 916), ('DfllPccAsWaitCtrl', uint32_t, 920), ('DfllPccAsStepCtrl', uint32_t, 924), ('GfxDfllSpare', c.Array[uint32_t, Literal[9]], 928), ('DvoPsmDownThresholdVoltage', uint32_t, 964), ('DvoPsmUpThresholdVoltage', uint32_t, 968), ('DvoFmaxLowScaler', uint32_t, 972), ('PaddingDcs', uint32_t, 976), ('DcsMinGfxOffTime', uint16_t, 980), ('DcsMaxGfxOffTime', uint16_t, 982), ('DcsMinCreditAccum', uint32_t, 984), ('DcsExitHysteresis', uint16_t, 988), ('DcsTimeout', uint16_t, 990), ('DcsPfGfxFopt', uint32_t, 992), ('DcsPfUclkFopt', uint32_t, 996), ('FoptEnabled', uint8_t, 1000), ('DcsSpare2', c.Array[uint8_t, Literal[3]], 1001), ('DcsFoptM', uint32_t, 1004), ('DcsFoptB', uint32_t, 1008), ('DcsSpare', c.Array[uint32_t, Literal[9]], 1012), ('UseStrobeModeOptimizations', uint8_t, 1048), ('PaddingMem', c.Array[uint8_t, Literal[3]], 1049), ('UclkDpmPstates', c.Array[uint8_t, Literal[6]], 1052), ('UclkDpmShadowPstates', c.Array[uint8_t, Literal[6]], 1058), ('FreqTableUclkDiv', c.Array[uint8_t, Literal[6]], 1064), ('FreqTableShadowUclkDiv', c.Array[uint8_t, Literal[6]], 1070), ('MemVmempVoltage', c.Array[uint16_t, Literal[6]], 1076), ('MemVddioVoltage', c.Array[uint16_t, Literal[6]], 1088), ('DalDcModeMaxUclkFreq', uint16_t, 1100), ('PaddingsMem', c.Array[uint8_t, Literal[2]], 1102), ('PaddingFclk', uint32_t, 1104), ('PcieGenSpeed', c.Array[uint8_t, Literal[3]], 1108), ('PcieLaneCount', c.Array[uint8_t, Literal[3]], 1111), ('LclkFreq', c.Array[uint16_t, Literal[3]], 1114), ('OverrideGfxAvfsFuses', uint8_t, 1120), ('GfxAvfsPadding', c.Array[uint8_t, Literal[1]], 1121), ('DroopGBStDev', uint16_t, 1122), ('SocHwRtAvfsFuses', c.Array[uint32_t, Literal[32]], 1124), ('GfxL2HwRtAvfsFuses', c.Array[uint32_t, Literal[32]], 1252), ('PsmDidt_Vcross', c.Array[uint16_t, Literal[2]], 1380), ('PsmDidt_StaticDroop_A', c.Array[uint32_t, Literal[3]], 1384), ('PsmDidt_StaticDroop_B', c.Array[uint32_t, Literal[3]], 1396), ('PsmDidt_DynDroop_A', c.Array[uint32_t, Literal[3]], 1408), ('PsmDidt_DynDroop_B', c.Array[uint32_t, Literal[3]], 1420), ('spare_HwRtAvfsFuses', c.Array[uint32_t, Literal[19]], 1432), ('SocCommonRtAvfs', c.Array[uint32_t, Literal[13]], 1508), ('GfxCommonRtAvfs', c.Array[uint32_t, Literal[13]], 1560), ('SocFwRtAvfsFuses', c.Array[uint32_t, Literal[19]], 1612), ('GfxL2FwRtAvfsFuses', c.Array[uint32_t, Literal[19]], 1688), ('spare_FwRtAvfsFuses', c.Array[uint32_t, Literal[19]], 1764), ('Soc_Droop_PWL_F', c.Array[uint32_t, Literal[5]], 1840), ('Soc_Droop_PWL_a', c.Array[uint32_t, Literal[5]], 1860), ('Soc_Droop_PWL_b', c.Array[uint32_t, Literal[5]], 1880), ('Soc_Droop_PWL_c', c.Array[uint32_t, Literal[5]], 1900), ('Gfx_Droop_PWL_F', c.Array[uint32_t, Literal[5]], 1920), ('Gfx_Droop_PWL_a', c.Array[uint32_t, Literal[5]], 1940), ('Gfx_Droop_PWL_b', c.Array[uint32_t, Literal[5]], 1960), ('Gfx_Droop_PWL_c', c.Array[uint32_t, Literal[5]], 1980), ('Gfx_Static_PWL_Offset', c.Array[uint32_t, Literal[5]], 2000), ('Soc_Static_PWL_Offset', c.Array[uint32_t, Literal[5]], 2020), ('dGbV_dT_vmin', uint32_t, 2040), ('dGbV_dT_vmax', uint32_t, 2044), ('PaddingV2F', c.Array[uint32_t, Literal[4]], 2048), ('DcBtcGfxParams', AvfsDcBtcParams_t, 2064), ('SSCurve_GFX', QuadraticInt_t, 2084), ('GfxAvfsSpare', c.Array[uint32_t, Literal[29]], 2096), ('OverrideSocAvfsFuses', uint8_t, 2212), ('MinSocAvfsRevision', uint8_t, 2213), ('SocAvfsPadding', c.Array[uint8_t, Literal[2]], 2214), ('SocAvfsFuseOverride', c.Array[AvfsFuseOverride_t, Literal[1]], 2216), ('dBtcGbSoc', c.Array[DroopInt_t, Literal[1]], 2272), ('qAgingGb', c.Array[LinearInt_t, Literal[1]], 2284), ('qStaticVoltageOffset', c.Array[QuadraticInt_t, Literal[1]], 2292), ('DcBtcSocParams', c.Array[AvfsDcBtcParams_t, Literal[1]], 2304), ('SSCurve_SOC', QuadraticInt_t, 2324), ('SocAvfsSpare', c.Array[uint32_t, Literal[29]], 2336), ('BootValues', BootValues_t, 2452), ('DriverReportedClocks', DriverReportedClocks_t, 2576), ('MsgLimits', MsgLimits_t, 2604), ('OverDriveLimitsBasicMin', OverDriveLimits_t, 2720), ('OverDriveLimitsBasicMax', OverDriveLimits_t, 2816), ('OverDriveLimitsAdvancedMin', OverDriveLimits_t, 2912), ('OverDriveLimitsAdvancedMax', OverDriveLimits_t, 3008), ('TotalBoardPowerSupport', uint8_t, 3104), ('TotalBoardPowerPadding', c.Array[uint8_t, Literal[1]], 3105), ('TotalBoardPowerRoc', uint16_t, 3106), ('qFeffCoeffGameClock', c.Array[QuadraticInt_t, Literal[2]], 3108), ('qFeffCoeffBaseClock', c.Array[QuadraticInt_t, Literal[2]], 3132), ('qFeffCoeffBoostClock', c.Array[QuadraticInt_t, Literal[2]], 3156), ('AptUclkGfxclkLookup', c.Array[c.Array[int32_t, Literal[6]], Literal[2]], 3180), ('AptUclkGfxclkLookupHyst', c.Array[c.Array[uint32_t, Literal[6]], Literal[2]], 3228), ('AptPadding', uint32_t, 3276), ('GfxXvminDidtDroopThresh', QuadraticInt_t, 3280), ('GfxXvminDidtResetDDWait', uint32_t, 3292), ('GfxXvminDidtClkStopWait', uint32_t, 3296), ('GfxXvminDidtFcsStepCtrl', uint32_t, 3300), ('GfxXvminDidtFcsWaitCtrl', uint32_t, 3304), ('PsmModeEnabled', uint32_t, 3308), ('P2v_a', uint32_t, 3312), ('P2v_b', uint32_t, 3316), ('P2v_c', uint32_t, 3320), ('T2p_a', uint32_t, 3324), ('T2p_b', uint32_t, 3328), ('T2p_c', uint32_t, 3332), ('P2vTemp', uint32_t, 3336), ('PsmDidtStaticSettings', QuadraticInt_t, 3340), ('PsmDidtDynamicSettings', QuadraticInt_t, 3352), ('PsmDidtAvgDiv', uint8_t, 3364), ('PsmDidtForceStall', uint8_t, 3365), ('PsmDidtReleaseTimer', uint16_t, 3366), ('PsmDidtStallPattern', uint32_t, 3368), ('CacEdcCacLeakageC0', uint32_t, 3372), ('CacEdcCacLeakageC1', uint32_t, 3376), ('CacEdcCacLeakageC2', uint32_t, 3380), ('CacEdcCacLeakageC3', uint32_t, 3384), ('CacEdcCacLeakageC4', uint32_t, 3388), ('CacEdcCacLeakageC5', uint32_t, 3392), ('CacEdcGfxClkScalar', uint32_t, 3396), ('CacEdcGfxClkIntercept', uint32_t, 3400), ('CacEdcCac_m', uint32_t, 3404), ('CacEdcCac_b', uint32_t, 3408), ('CacEdcCurrLimitGuardband', uint32_t, 3412), ('CacEdcDynToTotalCacRatio', uint32_t, 3416), ('XVmin_Gfx_EdcThreshScalar', uint32_t, 3420), ('XVmin_Gfx_EdcEnableFreq', uint32_t, 3424), ('XVmin_Gfx_EdcPccAsStepCtrl', uint32_t, 3428), ('XVmin_Gfx_EdcPccAsWaitCtrl', uint32_t, 3432), ('XVmin_Gfx_EdcThreshold', uint16_t, 3436), ('XVmin_Gfx_EdcFiltHysWaitCtrl', uint16_t, 3438), ('XVmin_Soc_EdcThreshScalar', uint32_t, 3440), ('XVmin_Soc_EdcEnableFreq', uint32_t, 3444), ('XVmin_Soc_EdcThreshold', uint32_t, 3448), ('XVmin_Soc_EdcStepUpTime', uint16_t, 3452), ('XVmin_Soc_EdcStepDownTime', uint16_t, 3454), ('XVmin_Soc_EdcInitPccStep', uint8_t, 3456), ('PaddingSocEdc', c.Array[uint8_t, Literal[3]], 3457), ('GfxXvminFuseOverride', uint8_t, 3460), ('SocXvminFuseOverride', uint8_t, 3461), ('PaddingXvminFuseOverride', c.Array[uint8_t, Literal[2]], 3462), ('GfxXvminFddTempLow', uint8_t, 3464), ('GfxXvminFddTempHigh', uint8_t, 3465), ('SocXvminFddTempLow', uint8_t, 3466), ('SocXvminFddTempHigh', uint8_t, 3467), ('GfxXvminFddVolt0', uint16_t, 3468), ('GfxXvminFddVolt1', uint16_t, 3470), ('GfxXvminFddVolt2', uint16_t, 3472), ('SocXvminFddVolt0', uint16_t, 3474), ('SocXvminFddVolt1', uint16_t, 3476), ('SocXvminFddVolt2', uint16_t, 3478), ('GfxXvminDsFddDsm', c.Array[uint16_t, Literal[6]], 3480), ('GfxXvminEdcFddDsm', c.Array[uint16_t, Literal[6]], 3492), ('SocXvminEdcFddDsm', c.Array[uint16_t, Literal[6]], 3504), ('Spare', uint32_t, 3516), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 3520)])
@c.record
class Svi3RegulatorSettings_t(c.Struct):
  SIZE = 28
  SlewRateConditions: int
  LoadLineAdjust: int
  VoutOffset: int
  VidMax: int
  VidMin: int
  TenBitTelEn: int
  SixteenBitTelEn: int
  OcpThresh: int
  OcpWarnThresh: int
  OcpSettings: int
  VrhotThresh: int
  OtpThresh: int
  UvpOvpDeltaRef: int
  PhaseShed: int
  Padding: c.Array[ctypes.c_ubyte, Literal[10]]
  SettingOverrideMask: int
Svi3RegulatorSettings_t.register_fields([('SlewRateConditions', uint8_t, 0), ('LoadLineAdjust', uint8_t, 1), ('VoutOffset', uint8_t, 2), ('VidMax', uint8_t, 3), ('VidMin', uint8_t, 4), ('TenBitTelEn', uint8_t, 5), ('SixteenBitTelEn', uint8_t, 6), ('OcpThresh', uint8_t, 7), ('OcpWarnThresh', uint8_t, 8), ('OcpSettings', uint8_t, 9), ('VrhotThresh', uint8_t, 10), ('OtpThresh', uint8_t, 11), ('UvpOvpDeltaRef', uint8_t, 12), ('PhaseShed', uint8_t, 13), ('Padding', c.Array[uint8_t, Literal[10]], 14), ('SettingOverrideMask', uint32_t, 24)])
@c.record
class BoardTable_t(c.Struct):
  SIZE = 528
  Version: int
  I2cControllers: c.Array[I2cControllerConfig_t, Literal[8]]
  SlaveAddrMapping: c.Array[ctypes.c_ubyte, Literal[4]]
  VrPsiSupport: c.Array[ctypes.c_ubyte, Literal[4]]
  Svi3SvcSpeed: int
  EnablePsi6: c.Array[ctypes.c_ubyte, Literal[4]]
  Svi3RegSettings: c.Array[Svi3RegulatorSettings_t, Literal[4]]
  LedOffGpio: int
  FanOffGpio: int
  GfxVrPowerStageOffGpio: int
  AcDcGpio: int
  AcDcPolarity: int
  VR0HotGpio: int
  VR0HotPolarity: int
  GthrGpio: int
  GthrPolarity: int
  LedPin0: int
  LedPin1: int
  LedPin2: int
  LedEnableMask: int
  LedPcie: int
  LedError: int
  PaddingLed: int
  UclkTrainingModeSpreadPercent: int
  UclkSpreadPadding: int
  UclkSpreadFreq: int
  UclkSpreadPercent: c.Array[ctypes.c_ubyte, Literal[16]]
  GfxclkSpreadEnable: int
  FclkSpreadPercent: int
  FclkSpreadFreq: int
  DramWidth: int
  PaddingMem1: c.Array[ctypes.c_ubyte, Literal[7]]
  HsrEnabled: int
  VddqOffEnabled: int
  PaddingUmcFlags: c.Array[ctypes.c_ubyte, Literal[2]]
  Paddign1: int
  BacoEntryDelay: int
  FuseWritePowerMuxPresent: int
  FuseWritePadding: c.Array[ctypes.c_ubyte, Literal[3]]
  LoadlineGfx: int
  LoadlineSoc: int
  GfxEdcLimit: int
  SocEdcLimit: int
  RestBoardPower: int
  ConnectorsImpedance: int
  EpcsSens0: int
  EpcsSens1: int
  PaddingEpcs: c.Array[ctypes.c_ubyte, Literal[2]]
  BoardSpare: c.Array[ctypes.c_uint32, Literal[52]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
BoardTable_t.register_fields([('Version', uint32_t, 0), ('I2cControllers', c.Array[I2cControllerConfig_t, Literal[8]], 4), ('SlaveAddrMapping', c.Array[uint8_t, Literal[4]], 68), ('VrPsiSupport', c.Array[uint8_t, Literal[4]], 72), ('Svi3SvcSpeed', uint32_t, 76), ('EnablePsi6', c.Array[uint8_t, Literal[4]], 80), ('Svi3RegSettings', c.Array[Svi3RegulatorSettings_t, Literal[4]], 84), ('LedOffGpio', uint8_t, 196), ('FanOffGpio', uint8_t, 197), ('GfxVrPowerStageOffGpio', uint8_t, 198), ('AcDcGpio', uint8_t, 199), ('AcDcPolarity', uint8_t, 200), ('VR0HotGpio', uint8_t, 201), ('VR0HotPolarity', uint8_t, 202), ('GthrGpio', uint8_t, 203), ('GthrPolarity', uint8_t, 204), ('LedPin0', uint8_t, 205), ('LedPin1', uint8_t, 206), ('LedPin2', uint8_t, 207), ('LedEnableMask', uint8_t, 208), ('LedPcie', uint8_t, 209), ('LedError', uint8_t, 210), ('PaddingLed', uint8_t, 211), ('UclkTrainingModeSpreadPercent', uint8_t, 212), ('UclkSpreadPadding', uint8_t, 213), ('UclkSpreadFreq', uint16_t, 214), ('UclkSpreadPercent', c.Array[uint8_t, Literal[16]], 216), ('GfxclkSpreadEnable', uint8_t, 232), ('FclkSpreadPercent', uint8_t, 233), ('FclkSpreadFreq', uint16_t, 234), ('DramWidth', uint8_t, 236), ('PaddingMem1', c.Array[uint8_t, Literal[7]], 237), ('HsrEnabled', uint8_t, 244), ('VddqOffEnabled', uint8_t, 245), ('PaddingUmcFlags', c.Array[uint8_t, Literal[2]], 246), ('Paddign1', uint32_t, 248), ('BacoEntryDelay', uint32_t, 252), ('FuseWritePowerMuxPresent', uint8_t, 256), ('FuseWritePadding', c.Array[uint8_t, Literal[3]], 257), ('LoadlineGfx', uint32_t, 260), ('LoadlineSoc', uint32_t, 264), ('GfxEdcLimit', uint32_t, 268), ('SocEdcLimit', uint32_t, 272), ('RestBoardPower', uint32_t, 276), ('ConnectorsImpedance', uint32_t, 280), ('EpcsSens0', uint8_t, 284), ('EpcsSens1', uint8_t, 285), ('PaddingEpcs', c.Array[uint8_t, Literal[2]], 286), ('BoardSpare', c.Array[uint32_t, Literal[52]], 288), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 496)])
@c.record
class CustomSkuTable_t(c.Struct):
  SIZE = 360
  SocketPowerLimitAc: c.Array[ctypes.c_uint16, Literal[4]]
  VrTdcLimit: c.Array[ctypes.c_uint16, Literal[2]]
  TotalIdleBoardPowerM: int
  TotalIdleBoardPowerB: int
  TotalBoardPowerM: int
  TotalBoardPowerB: int
  TemperatureLimit: c.Array[ctypes.c_uint16, Literal[12]]
  FanStopTemp: c.Array[ctypes.c_uint16, Literal[12]]
  FanStartTemp: c.Array[ctypes.c_uint16, Literal[12]]
  FanGain: c.Array[ctypes.c_uint16, Literal[12]]
  FanPwmMin: int
  AcousticTargetRpmThreshold: int
  AcousticLimitRpmThreshold: int
  FanMaximumRpm: int
  MGpuAcousticLimitRpmThreshold: int
  FanTargetGfxclk: int
  TempInputSelectMask: int
  FanZeroRpmEnable: int
  FanTachEdgePerRev: int
  FanPadding: int
  FanTargetTemperature: c.Array[ctypes.c_uint16, Literal[12]]
  FuzzyFan_ErrorSetDelta: int
  FuzzyFan_ErrorRateSetDelta: int
  FuzzyFan_PwmSetDelta: int
  FanPadding2: int
  FwCtfLimit: c.Array[ctypes.c_uint16, Literal[12]]
  IntakeTempEnableRPM: int
  IntakeTempOffsetTemp: int
  IntakeTempReleaseTemp: int
  IntakeTempHighIntakeAcousticLimit: int
  IntakeTempAcouticLimitReleaseRate: int
  FanAbnormalTempLimitOffset: int
  FanStalledTriggerRpm: int
  FanAbnormalTriggerRpmCoeff: int
  FanSpare: c.Array[ctypes.c_uint16, Literal[1]]
  FanIntakeSensorSupport: int
  FanIntakePadding: int
  FanSpare2: c.Array[ctypes.c_uint32, Literal[12]]
  ODFeatureCtrlMask: int
  TemperatureLimit_Hynix: int
  TemperatureLimit_Micron: int
  TemperatureFwCtfLimit_Hynix: int
  TemperatureFwCtfLimit_Micron: int
  PlatformTdcLimit: c.Array[ctypes.c_uint16, Literal[2]]
  SocketPowerLimitDc: c.Array[ctypes.c_uint16, Literal[4]]
  SocketPowerLimitSmartShift2: int
  CustomSkuSpare16b: int
  CustomSkuSpare32b: c.Array[ctypes.c_uint32, Literal[10]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
CustomSkuTable_t.register_fields([('SocketPowerLimitAc', c.Array[uint16_t, Literal[4]], 0), ('VrTdcLimit', c.Array[uint16_t, Literal[2]], 8), ('TotalIdleBoardPowerM', int16_t, 12), ('TotalIdleBoardPowerB', int16_t, 14), ('TotalBoardPowerM', int16_t, 16), ('TotalBoardPowerB', int16_t, 18), ('TemperatureLimit', c.Array[uint16_t, Literal[12]], 20), ('FanStopTemp', c.Array[uint16_t, Literal[12]], 44), ('FanStartTemp', c.Array[uint16_t, Literal[12]], 68), ('FanGain', c.Array[uint16_t, Literal[12]], 92), ('FanPwmMin', uint16_t, 116), ('AcousticTargetRpmThreshold', uint16_t, 118), ('AcousticLimitRpmThreshold', uint16_t, 120), ('FanMaximumRpm', uint16_t, 122), ('MGpuAcousticLimitRpmThreshold', uint16_t, 124), ('FanTargetGfxclk', uint16_t, 126), ('TempInputSelectMask', uint32_t, 128), ('FanZeroRpmEnable', uint8_t, 132), ('FanTachEdgePerRev', uint8_t, 133), ('FanPadding', uint16_t, 134), ('FanTargetTemperature', c.Array[uint16_t, Literal[12]], 136), ('FuzzyFan_ErrorSetDelta', int16_t, 160), ('FuzzyFan_ErrorRateSetDelta', int16_t, 162), ('FuzzyFan_PwmSetDelta', int16_t, 164), ('FanPadding2', uint16_t, 166), ('FwCtfLimit', c.Array[uint16_t, Literal[12]], 168), ('IntakeTempEnableRPM', uint16_t, 192), ('IntakeTempOffsetTemp', int16_t, 194), ('IntakeTempReleaseTemp', uint16_t, 196), ('IntakeTempHighIntakeAcousticLimit', uint16_t, 198), ('IntakeTempAcouticLimitReleaseRate', uint16_t, 200), ('FanAbnormalTempLimitOffset', int16_t, 202), ('FanStalledTriggerRpm', uint16_t, 204), ('FanAbnormalTriggerRpmCoeff', uint16_t, 206), ('FanSpare', c.Array[uint16_t, Literal[1]], 208), ('FanIntakeSensorSupport', uint8_t, 210), ('FanIntakePadding', uint8_t, 211), ('FanSpare2', c.Array[uint32_t, Literal[12]], 212), ('ODFeatureCtrlMask', uint32_t, 260), ('TemperatureLimit_Hynix', uint16_t, 264), ('TemperatureLimit_Micron', uint16_t, 266), ('TemperatureFwCtfLimit_Hynix', uint16_t, 268), ('TemperatureFwCtfLimit_Micron', uint16_t, 270), ('PlatformTdcLimit', c.Array[uint16_t, Literal[2]], 272), ('SocketPowerLimitDc', c.Array[uint16_t, Literal[4]], 276), ('SocketPowerLimitSmartShift2', uint16_t, 284), ('CustomSkuSpare16b', uint16_t, 286), ('CustomSkuSpare32b', c.Array[uint32_t, Literal[10]], 288), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 328)])
@c.record
class PPTable_t(c.Struct):
  SIZE = 4468
  PFE_Settings: PFE_Settings_t
  SkuTable: SkuTable_t
  CustomSkuTable: CustomSkuTable_t
  BoardTable: BoardTable_t
PPTable_t.register_fields([('PFE_Settings', PFE_Settings_t, 0), ('SkuTable', SkuTable_t, 28), ('CustomSkuTable', CustomSkuTable_t, 3580), ('BoardTable', BoardTable_t, 3940)])
@c.record
class DriverSmuConfig_t(c.Struct):
  SIZE = 20
  GfxclkAverageLpfTau: int
  FclkAverageLpfTau: int
  UclkAverageLpfTau: int
  GfxActivityLpfTau: int
  UclkActivityLpfTau: int
  UclkMaxActivityLpfTau: int
  SocketPowerLpfTau: int
  VcnClkAverageLpfTau: int
  VcnUsageAverageLpfTau: int
  PcieActivityLpTau: int
DriverSmuConfig_t.register_fields([('GfxclkAverageLpfTau', uint16_t, 0), ('FclkAverageLpfTau', uint16_t, 2), ('UclkAverageLpfTau', uint16_t, 4), ('GfxActivityLpfTau', uint16_t, 6), ('UclkActivityLpfTau', uint16_t, 8), ('UclkMaxActivityLpfTau', uint16_t, 10), ('SocketPowerLpfTau', uint16_t, 12), ('VcnClkAverageLpfTau', uint16_t, 14), ('VcnUsageAverageLpfTau', uint16_t, 16), ('PcieActivityLpTau', uint16_t, 18)])
@c.record
class DriverSmuConfigExternal_t(c.Struct):
  SIZE = 84
  DriverSmuConfig: DriverSmuConfig_t
  Spare: c.Array[ctypes.c_uint32, Literal[8]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
DriverSmuConfigExternal_t.register_fields([('DriverSmuConfig', DriverSmuConfig_t, 0), ('Spare', c.Array[uint32_t, Literal[8]], 20), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 52)])
@c.record
class DriverInfoTable_t(c.Struct):
  SIZE = 372
  FreqTableGfx: c.Array[ctypes.c_uint16, Literal[16]]
  FreqTableVclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableSocclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableUclk: c.Array[ctypes.c_uint16, Literal[6]]
  FreqTableDispclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDppClk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDprefclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDcfclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDtbclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableFclk: c.Array[ctypes.c_uint16, Literal[8]]
  DcModeMaxFreq: c.Array[ctypes.c_uint16, Literal[11]]
  Padding: int
  Spare: c.Array[ctypes.c_uint32, Literal[32]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
DriverInfoTable_t.register_fields([('FreqTableGfx', c.Array[uint16_t, Literal[16]], 0), ('FreqTableVclk', c.Array[uint16_t, Literal[8]], 32), ('FreqTableDclk', c.Array[uint16_t, Literal[8]], 48), ('FreqTableSocclk', c.Array[uint16_t, Literal[8]], 64), ('FreqTableUclk', c.Array[uint16_t, Literal[6]], 80), ('FreqTableDispclk', c.Array[uint16_t, Literal[8]], 92), ('FreqTableDppClk', c.Array[uint16_t, Literal[8]], 108), ('FreqTableDprefclk', c.Array[uint16_t, Literal[8]], 124), ('FreqTableDcfclk', c.Array[uint16_t, Literal[8]], 140), ('FreqTableDtbclk', c.Array[uint16_t, Literal[8]], 156), ('FreqTableFclk', c.Array[uint16_t, Literal[8]], 172), ('DcModeMaxFreq', c.Array[uint16_t, Literal[11]], 188), ('Padding', uint16_t, 210), ('Spare', c.Array[uint32_t, Literal[32]], 212), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 340)])
@c.record
class SmuMetrics_t(c.Struct):
  SIZE = 260
  CurrClock: c.Array[ctypes.c_uint32, Literal[11]]
  AverageGfxclkFrequencyTarget: int
  AverageGfxclkFrequencyPreDs: int
  AverageGfxclkFrequencyPostDs: int
  AverageFclkFrequencyPreDs: int
  AverageFclkFrequencyPostDs: int
  AverageMemclkFrequencyPreDs: int
  AverageMemclkFrequencyPostDs: int
  AverageVclk0Frequency: int
  AverageDclk0Frequency: int
  AverageVclk1Frequency: int
  AverageDclk1Frequency: int
  AveragePCIeBusy: int
  dGPU_W_MAX: int
  padding: int
  MovingAverageGfxclkFrequencyTarget: int
  MovingAverageGfxclkFrequencyPreDs: int
  MovingAverageGfxclkFrequencyPostDs: int
  MovingAverageFclkFrequencyPreDs: int
  MovingAverageFclkFrequencyPostDs: int
  MovingAverageMemclkFrequencyPreDs: int
  MovingAverageMemclkFrequencyPostDs: int
  MovingAverageVclk0Frequency: int
  MovingAverageDclk0Frequency: int
  MovingAverageGfxActivity: int
  MovingAverageUclkActivity: int
  MovingAverageVcn0ActivityPercentage: int
  MovingAveragePCIeBusy: int
  MovingAverageUclkActivity_MAX: int
  MovingAverageSocketPower: int
  MovingAveragePadding: int
  MetricsCounter: int
  AvgVoltage: c.Array[ctypes.c_uint16, Literal[4]]
  AvgCurrent: c.Array[ctypes.c_uint16, Literal[4]]
  AverageGfxActivity: int
  AverageUclkActivity: int
  AverageVcn0ActivityPercentage: int
  Vcn1ActivityPercentage: int
  EnergyAccumulator: int
  AverageSocketPower: int
  AverageTotalBoardPower: int
  AvgTemperature: c.Array[ctypes.c_uint16, Literal[12]]
  AvgTemperatureFanIntake: int
  PcieRate: int
  PcieWidth: int
  AvgFanPwm: int
  Padding: c.Array[ctypes.c_ubyte, Literal[1]]
  AvgFanRpm: int
  ThrottlingPercentage: c.Array[ctypes.c_ubyte, Literal[21]]
  VmaxThrottlingPercentage: int
  padding1: c.Array[ctypes.c_ubyte, Literal[2]]
  D3HotEntryCountPerMode: c.Array[ctypes.c_uint32, Literal[4]]
  D3HotExitCountPerMode: c.Array[ctypes.c_uint32, Literal[4]]
  ArmMsgReceivedCountPerMode: c.Array[ctypes.c_uint32, Literal[4]]
  ApuSTAPMSmartShiftLimit: int
  ApuSTAPMLimit: int
  AvgApuSocketPower: int
  AverageUclkActivity_MAX: int
  PublicSerialNumberLower: int
  PublicSerialNumberUpper: int
SmuMetrics_t.register_fields([('CurrClock', c.Array[uint32_t, Literal[11]], 0), ('AverageGfxclkFrequencyTarget', uint16_t, 44), ('AverageGfxclkFrequencyPreDs', uint16_t, 46), ('AverageGfxclkFrequencyPostDs', uint16_t, 48), ('AverageFclkFrequencyPreDs', uint16_t, 50), ('AverageFclkFrequencyPostDs', uint16_t, 52), ('AverageMemclkFrequencyPreDs', uint16_t, 54), ('AverageMemclkFrequencyPostDs', uint16_t, 56), ('AverageVclk0Frequency', uint16_t, 58), ('AverageDclk0Frequency', uint16_t, 60), ('AverageVclk1Frequency', uint16_t, 62), ('AverageDclk1Frequency', uint16_t, 64), ('AveragePCIeBusy', uint16_t, 66), ('dGPU_W_MAX', uint16_t, 68), ('padding', uint16_t, 70), ('MovingAverageGfxclkFrequencyTarget', uint16_t, 72), ('MovingAverageGfxclkFrequencyPreDs', uint16_t, 74), ('MovingAverageGfxclkFrequencyPostDs', uint16_t, 76), ('MovingAverageFclkFrequencyPreDs', uint16_t, 78), ('MovingAverageFclkFrequencyPostDs', uint16_t, 80), ('MovingAverageMemclkFrequencyPreDs', uint16_t, 82), ('MovingAverageMemclkFrequencyPostDs', uint16_t, 84), ('MovingAverageVclk0Frequency', uint16_t, 86), ('MovingAverageDclk0Frequency', uint16_t, 88), ('MovingAverageGfxActivity', uint16_t, 90), ('MovingAverageUclkActivity', uint16_t, 92), ('MovingAverageVcn0ActivityPercentage', uint16_t, 94), ('MovingAveragePCIeBusy', uint16_t, 96), ('MovingAverageUclkActivity_MAX', uint16_t, 98), ('MovingAverageSocketPower', uint16_t, 100), ('MovingAveragePadding', uint16_t, 102), ('MetricsCounter', uint32_t, 104), ('AvgVoltage', c.Array[uint16_t, Literal[4]], 108), ('AvgCurrent', c.Array[uint16_t, Literal[4]], 116), ('AverageGfxActivity', uint16_t, 124), ('AverageUclkActivity', uint16_t, 126), ('AverageVcn0ActivityPercentage', uint16_t, 128), ('Vcn1ActivityPercentage', uint16_t, 130), ('EnergyAccumulator', uint32_t, 132), ('AverageSocketPower', uint16_t, 136), ('AverageTotalBoardPower', uint16_t, 138), ('AvgTemperature', c.Array[uint16_t, Literal[12]], 140), ('AvgTemperatureFanIntake', uint16_t, 164), ('PcieRate', uint8_t, 166), ('PcieWidth', uint8_t, 167), ('AvgFanPwm', uint8_t, 168), ('Padding', c.Array[uint8_t, Literal[1]], 169), ('AvgFanRpm', uint16_t, 170), ('ThrottlingPercentage', c.Array[uint8_t, Literal[21]], 172), ('VmaxThrottlingPercentage', uint8_t, 193), ('padding1', c.Array[uint8_t, Literal[2]], 194), ('D3HotEntryCountPerMode', c.Array[uint32_t, Literal[4]], 196), ('D3HotExitCountPerMode', c.Array[uint32_t, Literal[4]], 212), ('ArmMsgReceivedCountPerMode', c.Array[uint32_t, Literal[4]], 228), ('ApuSTAPMSmartShiftLimit', uint16_t, 244), ('ApuSTAPMLimit', uint16_t, 246), ('AvgApuSocketPower', uint16_t, 248), ('AverageUclkActivity_MAX', uint16_t, 250), ('PublicSerialNumberLower', uint32_t, 252), ('PublicSerialNumberUpper', uint32_t, 256)])
@c.record
class SmuMetricsExternal_t(c.Struct):
  SIZE = 412
  SmuMetrics: SmuMetrics_t
  Spare: c.Array[ctypes.c_uint32, Literal[30]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
SmuMetricsExternal_t.register_fields([('SmuMetrics', SmuMetrics_t, 0), ('Spare', c.Array[uint32_t, Literal[30]], 260), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 380)])
@c.record
class WatermarkRowGeneric_t(c.Struct):
  SIZE = 4
  WmSetting: int
  Flags: int
  Padding: c.Array[ctypes.c_ubyte, Literal[2]]
WatermarkRowGeneric_t.register_fields([('WmSetting', uint8_t, 0), ('Flags', uint8_t, 1), ('Padding', c.Array[uint8_t, Literal[2]], 2)])
WATERMARKS_FLAGS_e: dict[int, str] = {(WATERMARKS_CLOCK_RANGE:=0): 'WATERMARKS_CLOCK_RANGE', (WATERMARKS_DUMMY_PSTATE:=1): 'WATERMARKS_DUMMY_PSTATE', (WATERMARKS_MALL:=2): 'WATERMARKS_MALL', (WATERMARKS_COUNT:=3): 'WATERMARKS_COUNT'}
@c.record
class Watermarks_t(c.Struct):
  SIZE = 16
  WatermarkRow: c.Array[WatermarkRowGeneric_t, Literal[4]]
Watermarks_t.register_fields([('WatermarkRow', c.Array[WatermarkRowGeneric_t, Literal[4]], 0)])
@c.record
class WatermarksExternal_t(c.Struct):
  SIZE = 112
  Watermarks: Watermarks_t
  Spare: c.Array[ctypes.c_uint32, Literal[16]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
WatermarksExternal_t.register_fields([('Watermarks', Watermarks_t, 0), ('Spare', c.Array[uint32_t, Literal[16]], 16), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 80)])
@c.record
class AvfsDebugTable_t(c.Struct):
  SIZE = 1368
  avgPsmCount: c.Array[ctypes.c_uint16, Literal[76]]
  minPsmCount: c.Array[ctypes.c_uint16, Literal[76]]
  maxPsmCount: c.Array[ctypes.c_uint16, Literal[76]]
  avgPsmVoltage: c.Array[ctypes.c_float, Literal[76]]
  minPsmVoltage: c.Array[ctypes.c_float, Literal[76]]
  maxPsmVoltage: c.Array[ctypes.c_float, Literal[76]]
AvfsDebugTable_t.register_fields([('avgPsmCount', c.Array[uint16_t, Literal[76]], 0), ('minPsmCount', c.Array[uint16_t, Literal[76]], 152), ('maxPsmCount', c.Array[uint16_t, Literal[76]], 304), ('avgPsmVoltage', c.Array[ctypes.c_float, Literal[76]], 456), ('minPsmVoltage', c.Array[ctypes.c_float, Literal[76]], 760), ('maxPsmVoltage', c.Array[ctypes.c_float, Literal[76]], 1064)])
@c.record
class AvfsDebugTableExternal_t(c.Struct):
  SIZE = 1400
  AvfsDebugTable: AvfsDebugTable_t
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
AvfsDebugTableExternal_t.register_fields([('AvfsDebugTable', AvfsDebugTable_t, 0), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 1368)])
@c.record
class DpmActivityMonitorCoeffInt_t(c.Struct):
  SIZE = 108
  Gfx_ActiveHystLimit: int
  Gfx_IdleHystLimit: int
  Gfx_FPS: int
  Gfx_MinActiveFreqType: int
  Gfx_BoosterFreqType: int
  PaddingGfx: int
  Gfx_MinActiveFreq: int
  Gfx_BoosterFreq: int
  Gfx_PD_Data_time_constant: int
  Gfx_PD_Data_limit_a: int
  Gfx_PD_Data_limit_b: int
  Gfx_PD_Data_limit_c: int
  Gfx_PD_Data_error_coeff: int
  Gfx_PD_Data_error_rate_coeff: int
  Fclk_ActiveHystLimit: int
  Fclk_IdleHystLimit: int
  Fclk_FPS: int
  Fclk_MinActiveFreqType: int
  Fclk_BoosterFreqType: int
  PaddingFclk: int
  Fclk_MinActiveFreq: int
  Fclk_BoosterFreq: int
  Fclk_PD_Data_time_constant: int
  Fclk_PD_Data_limit_a: int
  Fclk_PD_Data_limit_b: int
  Fclk_PD_Data_limit_c: int
  Fclk_PD_Data_error_coeff: int
  Fclk_PD_Data_error_rate_coeff: int
  Mem_UpThreshold_Limit: c.Array[ctypes.c_uint32, Literal[6]]
  Mem_UpHystLimit: c.Array[ctypes.c_ubyte, Literal[6]]
  Mem_DownHystLimit: c.Array[ctypes.c_uint16, Literal[6]]
  Mem_Fps: int
DpmActivityMonitorCoeffInt_t.register_fields([('Gfx_ActiveHystLimit', uint8_t, 0), ('Gfx_IdleHystLimit', uint8_t, 1), ('Gfx_FPS', uint8_t, 2), ('Gfx_MinActiveFreqType', uint8_t, 3), ('Gfx_BoosterFreqType', uint8_t, 4), ('PaddingGfx', uint8_t, 5), ('Gfx_MinActiveFreq', uint16_t, 6), ('Gfx_BoosterFreq', uint16_t, 8), ('Gfx_PD_Data_time_constant', uint16_t, 10), ('Gfx_PD_Data_limit_a', uint32_t, 12), ('Gfx_PD_Data_limit_b', uint32_t, 16), ('Gfx_PD_Data_limit_c', uint32_t, 20), ('Gfx_PD_Data_error_coeff', uint32_t, 24), ('Gfx_PD_Data_error_rate_coeff', uint32_t, 28), ('Fclk_ActiveHystLimit', uint8_t, 32), ('Fclk_IdleHystLimit', uint8_t, 33), ('Fclk_FPS', uint8_t, 34), ('Fclk_MinActiveFreqType', uint8_t, 35), ('Fclk_BoosterFreqType', uint8_t, 36), ('PaddingFclk', uint8_t, 37), ('Fclk_MinActiveFreq', uint16_t, 38), ('Fclk_BoosterFreq', uint16_t, 40), ('Fclk_PD_Data_time_constant', uint16_t, 42), ('Fclk_PD_Data_limit_a', uint32_t, 44), ('Fclk_PD_Data_limit_b', uint32_t, 48), ('Fclk_PD_Data_limit_c', uint32_t, 52), ('Fclk_PD_Data_error_coeff', uint32_t, 56), ('Fclk_PD_Data_error_rate_coeff', uint32_t, 60), ('Mem_UpThreshold_Limit', c.Array[uint32_t, Literal[6]], 64), ('Mem_UpHystLimit', c.Array[uint8_t, Literal[6]], 88), ('Mem_DownHystLimit', c.Array[uint16_t, Literal[6]], 94), ('Mem_Fps', uint16_t, 106)])
@c.record
class DpmActivityMonitorCoeffIntExternal_t(c.Struct):
  SIZE = 140
  DpmActivityMonitorCoeffInt: DpmActivityMonitorCoeffInt_t
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
DpmActivityMonitorCoeffIntExternal_t.register_fields([('DpmActivityMonitorCoeffInt', DpmActivityMonitorCoeffInt_t, 0), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 108)])
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
FEATURE_CCLK_DPM_BIT = 0
FEATURE_FAN_CONTROLLER_BIT = 1
FEATURE_DATA_CALCULATION_BIT = 2
FEATURE_PPT_BIT = 3
FEATURE_TDC_BIT = 4
FEATURE_THERMAL_BIT = 5
FEATURE_FIT_BIT = 6
FEATURE_EDC_BIT = 7
FEATURE_PLL_POWER_DOWN_BIT = 8
FEATURE_VDDOFF_BIT = 9
FEATURE_VCN_DPM_BIT = 10
FEATURE_DS_MPM_BIT = 11
FEATURE_FCLK_DPM_BIT = 12
FEATURE_SOCCLK_DPM_BIT = 13
FEATURE_DS_MPIO_BIT = 14
FEATURE_LCLK_DPM_BIT = 15
FEATURE_SHUBCLK_DPM_BIT = 16
FEATURE_DCFCLK_DPM_BIT = 17
FEATURE_ISP_DPM_BIT = 18
FEATURE_IPU_DPM_BIT = 19
FEATURE_GFX_DPM_BIT = 20
FEATURE_DS_GFXCLK_BIT = 21
FEATURE_DS_SOCCLK_BIT = 22
FEATURE_DS_LCLK_BIT = 23
FEATURE_LOW_POWER_DCNCLKS_BIT = 24
FEATURE_DS_SHUBCLK_BIT = 25
FEATURE_RESERVED0_BIT = 26
FEATURE_ZSTATES_BIT = 27
FEATURE_IOMMUL2_PG_BIT = 28
FEATURE_DS_FCLK_BIT = 29
FEATURE_DS_SMNCLK_BIT = 30
FEATURE_DS_MP1CLK_BIT = 31
FEATURE_WHISPER_MODE_BIT = 32
FEATURE_SMU_LOW_POWER_BIT = 33
FEATURE_RESERVED1_BIT = 34
FEATURE_GFX_DEM_BIT = 35
FEATURE_PSI_BIT = 36
FEATURE_PROCHOT_BIT = 37
FEATURE_CPUOFF_BIT = 38
FEATURE_STAPM_BIT = 39
FEATURE_S0I3_BIT = 40
FEATURE_DF_LIGHT_CSTATE = 41
FEATURE_PERF_LIMIT_BIT = 42
FEATURE_CORE_DLDO_BIT = 43
FEATURE_DVO_BIT = 44
FEATURE_DS_VCN_BIT = 45
FEATURE_CPPC_BIT = 46
FEATURE_CPPC_PREFERRED_CORES = 47
FEATURE_DF_CSTATES_BIT = 48
FEATURE_FAST_PSTATE_CLDO_BIT = 49
FEATURE_ATHUB_PG_BIT = 50
FEATURE_VDDOFF_ECO_BIT = 51
FEATURE_ZSTATES_ECO_BIT = 52
FEATURE_CC6_BIT = 53
FEATURE_DS_UMCCLK_BIT = 54
FEATURE_DS_ISPCLK_BIT = 55
FEATURE_DS_HSPCLK_BIT = 56
FEATURE_P3T_BIT = 57
FEATURE_DS_IPUCLK_BIT = 58
FEATURE_DS_VPECLK_BIT = 59
FEATURE_VPE_DPM_BIT = 60
FEATURE_SMART_L3_RINSER_BIT = 61
FEATURE_PCC_BIT = 62
NUM_FEATURES = 63
PPSMC_VERSION = 0x1
PPSMC_Result_OK = 0x1
PPSMC_Result_Failed = 0xFF
PPSMC_Result_UnknownCmd = 0xFE
PPSMC_Result_CmdRejectedPrereq = 0xFD
PPSMC_Result_CmdRejectedBusy = 0xFC
PPSMC_MSG_TestMessage = 0x1
PPSMC_MSG_GetSmuVersion = 0x2
PPSMC_MSG_GetDriverIfVersion = 0x3
PPSMC_MSG_SetAllowedFeaturesMaskLow = 0x4
PPSMC_MSG_SetAllowedFeaturesMaskHigh = 0x5
PPSMC_MSG_EnableAllSmuFeatures = 0x6
PPSMC_MSG_DisableAllSmuFeatures = 0x7
PPSMC_MSG_EnableSmuFeaturesLow = 0x8
PPSMC_MSG_EnableSmuFeaturesHigh = 0x9
PPSMC_MSG_DisableSmuFeaturesLow = 0xA
PPSMC_MSG_DisableSmuFeaturesHigh = 0xB
PPSMC_MSG_GetRunningSmuFeaturesLow = 0xC
PPSMC_MSG_GetRunningSmuFeaturesHigh = 0xD
PPSMC_MSG_SetDriverDramAddrHigh = 0xE
PPSMC_MSG_SetDriverDramAddrLow = 0xF
PPSMC_MSG_SetToolsDramAddrHigh = 0x10
PPSMC_MSG_SetToolsDramAddrLow = 0x11
PPSMC_MSG_TransferTableSmu2Dram = 0x12
PPSMC_MSG_TransferTableDram2Smu = 0x13
PPSMC_MSG_UseDefaultPPTable = 0x14
PPSMC_MSG_EnterBaco = 0x15
PPSMC_MSG_ExitBaco = 0x16
PPSMC_MSG_ArmD3 = 0x17
PPSMC_MSG_BacoAudioD3PME = 0x18
PPSMC_MSG_SetSoftMinByFreq = 0x19
PPSMC_MSG_SetSoftMaxByFreq = 0x1A
PPSMC_MSG_SetHardMinByFreq = 0x1B
PPSMC_MSG_SetHardMaxByFreq = 0x1C
PPSMC_MSG_GetMinDpmFreq = 0x1D
PPSMC_MSG_GetMaxDpmFreq = 0x1E
PPSMC_MSG_GetDpmFreqByIndex = 0x1F
PPSMC_MSG_OverridePcieParameters = 0x20
PPSMC_MSG_DramLogSetDramAddrHigh = 0x21
PPSMC_MSG_DramLogSetDramAddrLow = 0x22
PPSMC_MSG_DramLogSetDramSize = 0x23
PPSMC_MSG_SetWorkloadMask = 0x24
PPSMC_MSG_GetVoltageByDpm = 0x25
PPSMC_MSG_SetVideoFps = 0x26
PPSMC_MSG_GetDcModeMaxDpmFreq = 0x27
PPSMC_MSG_AllowGfxOff = 0x28
PPSMC_MSG_DisallowGfxOff = 0x29
PPSMC_MSG_PowerUpVcn = 0x2A
PPSMC_MSG_PowerDownVcn = 0x2B
PPSMC_MSG_PowerUpJpeg = 0x2C
PPSMC_MSG_PowerDownJpeg = 0x2D
PPSMC_MSG_PrepareMp1ForUnload = 0x2E
PPSMC_MSG_SetSystemVirtualDramAddrHigh = 0x30
PPSMC_MSG_SetSystemVirtualDramAddrLow = 0x31
PPSMC_MSG_SetPptLimit = 0x32
PPSMC_MSG_GetPptLimit = 0x33
PPSMC_MSG_ReenableAcDcInterrupt = 0x34
PPSMC_MSG_NotifyPowerSource = 0x35
PPSMC_MSG_RunDcBtc = 0x36
PPSMC_MSG_SetTemperatureInputSelect = 0x38
PPSMC_MSG_SetFwDstatesMask = 0x39
PPSMC_MSG_SetThrottlerMask = 0x3A
PPSMC_MSG_SetExternalClientDfCstateAllow = 0x3B
PPSMC_MSG_SetMGpuFanBoostLimitRpm = 0x3C
PPSMC_MSG_DumpSTBtoDram = 0x3D
PPSMC_MSG_STBtoDramLogSetDramAddress = 0x3E
PPSMC_MSG_DummyUndefined = 0x3F
PPSMC_MSG_STBtoDramLogSetDramSize = 0x40
PPSMC_MSG_SetOBMTraceBufferLogging = 0x41
PPSMC_MSG_UseProfilingMode = 0x42
PPSMC_MSG_AllowGfxDcs = 0x43
PPSMC_MSG_DisallowGfxDcs = 0x44
PPSMC_MSG_EnableAudioStutterWA = 0x45
PPSMC_MSG_PowerUpUmsch = 0x46
PPSMC_MSG_PowerDownUmsch = 0x47
PPSMC_MSG_SetDcsArch = 0x48
PPSMC_MSG_TriggerVFFLR = 0x49
PPSMC_MSG_SetNumBadMemoryPagesRetired = 0x4A
PPSMC_MSG_SetBadMemoryPagesRetiredFlagsPerChannel = 0x4B
PPSMC_MSG_SetPriorityDeltaGain = 0x4C
PPSMC_MSG_AllowIHHostInterrupt = 0x4D
PPSMC_MSG_EnableShadowDpm = 0x4E
PPSMC_MSG_Mode3Reset = 0x4F
PPSMC_MSG_SetDriverDramAddr = 0x50
PPSMC_MSG_SetToolsDramAddr = 0x51
PPSMC_MSG_TransferTableSmu2DramWithAddr = 0x52
PPSMC_MSG_TransferTableDram2SmuWithAddr = 0x53
PPSMC_MSG_GetAllRunningSmuFeatures = 0x54
PPSMC_MSG_GetSvi3Voltage = 0x55
PPSMC_MSG_UpdatePolicy = 0x56
PPSMC_MSG_ExtPwrConnSupport = 0x57
PPSMC_MSG_PreloadSwPstateForUclkOverDrive = 0x58
PPSMC_Message_Count = 0x59
PPTABLE_VERSION = 0x1B
NUM_GFXCLK_DPM_LEVELS = 16
NUM_SOCCLK_DPM_LEVELS = 8
NUM_MP0CLK_DPM_LEVELS = 2
NUM_DCLK_DPM_LEVELS = 8
NUM_VCLK_DPM_LEVELS = 8
NUM_DISPCLK_DPM_LEVELS = 8
NUM_DPPCLK_DPM_LEVELS = 8
NUM_DPREFCLK_DPM_LEVELS = 8
NUM_DCFCLK_DPM_LEVELS = 8
NUM_DTBCLK_DPM_LEVELS = 8
NUM_UCLK_DPM_LEVELS = 6
NUM_LINK_LEVELS = 3
NUM_FCLK_DPM_LEVELS = 8
NUM_OD_FAN_MAX_POINTS = 6
FEATURE_FW_DATA_READ_BIT = 0
FEATURE_DPM_GFXCLK_BIT = 1
FEATURE_DPM_GFX_POWER_OPTIMIZER_BIT = 2
FEATURE_DPM_UCLK_BIT = 3
FEATURE_DPM_FCLK_BIT = 4
FEATURE_DPM_SOCCLK_BIT = 5
FEATURE_DPM_LINK_BIT = 6
FEATURE_DPM_DCN_BIT = 7
FEATURE_VMEMP_SCALING_BIT = 8
FEATURE_VDDIO_MEM_SCALING_BIT = 9
FEATURE_DS_GFXCLK_BIT = 10
FEATURE_DS_SOCCLK_BIT = 11
FEATURE_DS_FCLK_BIT = 12
FEATURE_DS_LCLK_BIT = 13
FEATURE_DS_DCFCLK_BIT = 14
FEATURE_DS_UCLK_BIT = 15
FEATURE_GFX_ULV_BIT = 16
FEATURE_FW_DSTATE_BIT = 17
FEATURE_GFXOFF_BIT = 18
FEATURE_BACO_BIT = 19
FEATURE_MM_DPM_BIT = 20
FEATURE_SOC_MPCLK_DS_BIT = 21
FEATURE_BACO_MPCLK_DS_BIT = 22
FEATURE_THROTTLERS_BIT = 23
FEATURE_SMARTSHIFT_BIT = 24
FEATURE_GTHR_BIT = 25
FEATURE_ACDC_BIT = 26
FEATURE_VR0HOT_BIT = 27
FEATURE_FW_CTF_BIT = 28
FEATURE_FAN_CONTROL_BIT = 29
FEATURE_GFX_DCS_BIT = 30
FEATURE_GFX_READ_MARGIN_BIT = 31
FEATURE_LED_DISPLAY_BIT = 32
FEATURE_GFXCLK_SPREAD_SPECTRUM_BIT = 33
FEATURE_OUT_OF_BAND_MONITOR_BIT = 34
FEATURE_OPTIMIZED_VMIN_BIT = 35
FEATURE_GFX_IMU_BIT = 36
FEATURE_BOOT_TIME_CAL_BIT = 37
FEATURE_GFX_PCC_DFLL_BIT = 38
FEATURE_SOC_CG_BIT = 39
FEATURE_DF_CSTATE_BIT = 40
FEATURE_GFX_EDC_BIT = 41
FEATURE_BOOT_POWER_OPT_BIT = 42
FEATURE_CLOCK_POWER_DOWN_BYPASS_BIT = 43
FEATURE_DS_VCN_BIT = 44
FEATURE_BACO_CG_BIT = 45
FEATURE_MEM_TEMP_READ_BIT = 46
FEATURE_ATHUB_MMHUB_PG_BIT = 47
FEATURE_SOC_PCC_BIT = 48
FEATURE_EDC_PWRBRK_BIT = 49
FEATURE_SOC_EDC_XVMIN_BIT = 50
FEATURE_GFX_PSM_DIDT_BIT = 51
FEATURE_APT_ALL_ENABLE_BIT = 52
FEATURE_APT_SQ_THROTTLE_BIT = 53
FEATURE_APT_PF_DCS_BIT = 54
FEATURE_GFX_EDC_XVMIN_BIT = 55
FEATURE_GFX_DIDT_XVMIN_BIT = 56
FEATURE_FAN_ABNORMAL_BIT = 57
FEATURE_CLOCK_STRETCH_COMPENSATOR = 58
FEATURE_SPARE_59_BIT = 59
FEATURE_SPARE_60_BIT = 60
FEATURE_SPARE_61_BIT = 61
FEATURE_SPARE_62_BIT = 62
FEATURE_SPARE_63_BIT = 63
NUM_FEATURES = 64
ALLOWED_FEATURE_CTRL_DEFAULT = 0xFFFFFFFFFFFFFFFF
ALLOWED_FEATURE_CTRL_SCPM = (1 << FEATURE_DPM_GFXCLK_BIT) | (1 << FEATURE_DPM_GFX_POWER_OPTIMIZER_BIT) | (1 << FEATURE_DPM_UCLK_BIT) | (1 << FEATURE_DPM_FCLK_BIT) | (1 << FEATURE_DPM_SOCCLK_BIT) | (1 << FEATURE_DPM_LINK_BIT) | (1 << FEATURE_DPM_DCN_BIT) | (1 << FEATURE_DS_GFXCLK_BIT) | (1 << FEATURE_DS_SOCCLK_BIT) | (1 << FEATURE_DS_FCLK_BIT) | (1 << FEATURE_DS_LCLK_BIT) | (1 << FEATURE_DS_DCFCLK_BIT) | (1 << FEATURE_DS_UCLK_BIT) | (1 << FEATURE_DS_VCN_BIT)
DEBUG_OVERRIDE_NOT_USE = 0x00000001
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_DCN_FCLK = 0x00000002
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_MP0_FCLK = 0x00000004
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_VCN_DCFCLK = 0x00000008
DEBUG_OVERRIDE_DISABLE_FAST_FCLK_TIMER = 0x00000010
DEBUG_OVERRIDE_DISABLE_VCN_PG = 0x00000020
DEBUG_OVERRIDE_DISABLE_FMAX_VMAX = 0x00000040
DEBUG_OVERRIDE_DISABLE_IMU_FW_CHECKS = 0x00000080
DEBUG_OVERRIDE_DISABLE_D0i2_REENTRY_HSR_TIMER_CHECK = 0x00000100
DEBUG_OVERRIDE_DISABLE_DFLL = 0x00000200
DEBUG_OVERRIDE_ENABLE_RLC_VF_BRINGUP_MODE = 0x00000400
DEBUG_OVERRIDE_DFLL_MASTER_MODE = 0x00000800
DEBUG_OVERRIDE_ENABLE_PROFILING_MODE = 0x00001000
DEBUG_OVERRIDE_ENABLE_SOC_VF_BRINGUP_MODE = 0x00002000
DEBUG_OVERRIDE_ENABLE_PER_WGP_RESIENCY = 0x00004000
DEBUG_OVERRIDE_DISABLE_MEMORY_VOLTAGE_SCALING = 0x00008000
DEBUG_OVERRIDE_DFLL_BTC_FCW_LOG = 0x00010000
VR_MAPPING_VR_SELECT_MASK = 0x01
VR_MAPPING_VR_SELECT_SHIFT = 0x00
VR_MAPPING_PLANE_SELECT_MASK = 0x02
VR_MAPPING_PLANE_SELECT_SHIFT = 0x01
PSI_SEL_VR0_PLANE0_PSI0 = 0x01
PSI_SEL_VR0_PLANE0_PSI1 = 0x02
PSI_SEL_VR0_PLANE1_PSI0 = 0x04
PSI_SEL_VR0_PLANE1_PSI1 = 0x08
PSI_SEL_VR1_PLANE0_PSI0 = 0x10
PSI_SEL_VR1_PLANE0_PSI1 = 0x20
PSI_SEL_VR1_PLANE1_PSI0 = 0x40
PSI_SEL_VR1_PLANE1_PSI1 = 0x80
THROTTLER_TEMP_EDGE_BIT = 0
THROTTLER_TEMP_HOTSPOT_BIT = 1
THROTTLER_TEMP_HOTSPOT_GFX_BIT = 2
THROTTLER_TEMP_HOTSPOT_SOC_BIT = 3
THROTTLER_TEMP_MEM_BIT = 4
THROTTLER_TEMP_VR_GFX_BIT = 5
THROTTLER_TEMP_VR_SOC_BIT = 6
THROTTLER_TEMP_VR_MEM0_BIT = 7
THROTTLER_TEMP_VR_MEM1_BIT = 8
THROTTLER_TEMP_LIQUID0_BIT = 9
THROTTLER_TEMP_LIQUID1_BIT = 10
THROTTLER_TEMP_PLX_BIT = 11
THROTTLER_TDC_GFX_BIT = 12
THROTTLER_TDC_SOC_BIT = 13
THROTTLER_PPT0_BIT = 14
THROTTLER_PPT1_BIT = 15
THROTTLER_PPT2_BIT = 16
THROTTLER_PPT3_BIT = 17
THROTTLER_FIT_BIT = 18
THROTTLER_GFX_APCC_PLUS_BIT = 19
THROTTLER_GFX_DVO_BIT = 20
THROTTLER_COUNT = 21
FW_DSTATE_SOC_ULV_BIT = 0
FW_DSTATE_G6_HSR_BIT = 1
FW_DSTATE_G6_PHY_VMEMP_OFF_BIT = 2
FW_DSTATE_SMN_DS_BIT = 3
FW_DSTATE_MP1_WHISPER_MODE_BIT = 4
FW_DSTATE_SOC_LIV_MIN_BIT = 5
FW_DSTATE_SOC_PLL_PWRDN_BIT = 6
FW_DSTATE_MEM_PLL_PWRDN_BIT = 7
FW_DSTATE_MALL_ALLOC_BIT = 8
FW_DSTATE_MEM_PSI_BIT = 9
FW_DSTATE_HSR_NON_STROBE_BIT = 10
FW_DSTATE_MP0_ENTER_WFI_BIT = 11
FW_DSTATE_MALL_FLUSH_BIT = 12
FW_DSTATE_SOC_PSI_BIT = 13
FW_DSTATE_MMHUB_INTERLOCK_BIT = 14
FW_DSTATE_D0i3_2_QUIET_FW_BIT = 15
FW_DSTATE_CLDO_PRG_BIT = 16
FW_DSTATE_DF_PLL_PWRDN_BIT = 17
LED_DISPLAY_GFX_DPM_BIT = 0
LED_DISPLAY_PCIE_BIT = 1
LED_DISPLAY_ERROR_BIT = 2
MEM_TEMP_READ_OUT_OF_BAND_BIT = 0
MEM_TEMP_READ_IN_BAND_REFRESH_BIT = 1
MEM_TEMP_READ_IN_BAND_DUMMY_PSTATE_BIT = 2
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
EPCS_HIGH_POWER = 600
EPCS_NORMAL_POWER = 450
EPCS_LOW_POWER = 300
EPCS_SHORTED_POWER = 150
EPCS_NO_BOOTUP = 0
PP_NUM_RTAVFS_PWL_ZONES = 5
PP_NUM_PSM_DIDT_PWL_ZONES = 3
PP_NUM_OD_VF_CURVE_POINTS = PP_NUM_RTAVFS_PWL_ZONES + 1
PP_OD_FEATURE_GFX_VF_CURVE_BIT = 0
PP_OD_FEATURE_GFX_VMAX_BIT = 1
PP_OD_FEATURE_SOC_VMAX_BIT = 2
PP_OD_FEATURE_PPT_BIT = 3
PP_OD_FEATURE_FAN_CURVE_BIT = 4
PP_OD_FEATURE_FAN_LEGACY_BIT = 5
PP_OD_FEATURE_FULL_CTRL_BIT = 6
PP_OD_FEATURE_TDC_BIT = 7
PP_OD_FEATURE_GFXCLK_BIT = 8
PP_OD_FEATURE_UCLK_BIT = 9
PP_OD_FEATURE_FCLK_BIT = 10
PP_OD_FEATURE_ZERO_FAN_BIT = 11
PP_OD_FEATURE_TEMPERATURE_BIT = 12
PP_OD_FEATURE_EDC_BIT = 13
PP_OD_FEATURE_COUNT = 14
INVALID_BOARD_GPIO = 0xFF
NUM_WM_RANGES = 4
WORKLOAD_PPLIB_DEFAULT_BIT = 0
WORKLOAD_PPLIB_FULL_SCREEN_3D_BIT = 1
WORKLOAD_PPLIB_POWER_SAVING_BIT = 2
WORKLOAD_PPLIB_VIDEO_BIT = 3
WORKLOAD_PPLIB_VR_BIT = 4
WORKLOAD_PPLIB_COMPUTE_BIT = 5
WORKLOAD_PPLIB_CUSTOM_BIT = 6
WORKLOAD_PPLIB_WINDOW_3D_BIT = 7
WORKLOAD_PPLIB_DIRECT_ML_BIT = 8
WORKLOAD_PPLIB_CGVDI_BIT = 9
WORKLOAD_PPLIB_COUNT = 10
TABLE_TRANSFER_OK = 0x0
TABLE_TRANSFER_FAILED = 0xFF
TABLE_TRANSFER_PENDING = 0xAB
TABLE_PPT_FAILED = 0x100
TABLE_TDC_FAILED = 0x200
TABLE_TEMP_FAILED = 0x400
TABLE_FAN_TARGET_TEMP_FAILED = 0x800
TABLE_FAN_STOP_TEMP_FAILED = 0x1000
TABLE_FAN_START_TEMP_FAILED = 0x2000
TABLE_FAN_PWM_MIN_FAILED = 0x4000
TABLE_ACOUSTIC_TARGET_RPM_FAILED = 0x8000
TABLE_ACOUSTIC_LIMIT_RPM_FAILED = 0x10000
TABLE_MGPU_ACOUSTIC_TARGET_RPM_FAILED = 0x20000
TABLE_PPTABLE = 0
TABLE_COMBO_PPTABLE = 1
TABLE_WATERMARKS = 2
TABLE_AVFS_PSM_DEBUG = 3
TABLE_PMSTATUSLOG = 4
TABLE_SMU_METRICS = 5
TABLE_DRIVER_SMU_CONFIG = 6
TABLE_ACTIVITY_MONITOR_COEFF = 7
TABLE_OVERDRIVE = 8
TABLE_I2C_COMMANDS = 9
TABLE_DRIVER_INFO = 10
TABLE_ECCINFO = 11
TABLE_CUSTOM_SKUTABLE = 12
TABLE_COUNT = 13
IH_INTERRUPT_ID_TO_DRIVER = 0xFE
IH_INTERRUPT_CONTEXT_ID_BACO = 0x2
IH_INTERRUPT_CONTEXT_ID_AC = 0x3
IH_INTERRUPT_CONTEXT_ID_DC = 0x4
IH_INTERRUPT_CONTEXT_ID_AUDIO_D0 = 0x5
IH_INTERRUPT_CONTEXT_ID_AUDIO_D3 = 0x6
IH_INTERRUPT_CONTEXT_ID_THERMAL_THROTTLING = 0x7
IH_INTERRUPT_CONTEXT_ID_FAN_ABNORMAL = 0x8
IH_INTERRUPT_CONTEXT_ID_FAN_RECOVERY = 0x9
IH_INTERRUPT_CONTEXT_ID_DYNAMIC_TABLE = 0xA
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