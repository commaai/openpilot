# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_SMU14_Firmware_Footer(c.Struct):
  SIZE = 4
  Signature: Annotated[uint32_t, 0]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
SMU14_Firmware_Footer: TypeAlias = struct_SMU14_Firmware_Footer
@c.record
class SMU_Firmware_Header(c.Struct):
  SIZE = 256
  ImageVersion: Annotated[uint32_t, 0]
  ImageVersion2: Annotated[uint32_t, 4]
  Padding0: Annotated[c.Array[uint32_t, Literal[3]], 8]
  SizeFWSigned: Annotated[uint32_t, 20]
  Padding1: Annotated[c.Array[uint32_t, Literal[25]], 24]
  FirmwareType: Annotated[uint32_t, 124]
  Filler: Annotated[c.Array[uint32_t, Literal[32]], 128]
@c.record
class FwStatus_t(c.Struct):
  SIZE = 24
  DpmHandlerID: Annotated[uint32_t, 0, 8, 0]
  ActivityMonitorID: Annotated[uint32_t, 1, 8, 0]
  DpmTimerID: Annotated[uint32_t, 2, 8, 0]
  DpmHubID: Annotated[uint32_t, 3, 4, 0]
  DpmHubTask: Annotated[uint32_t, 3, 4, 4]
  CclkSyncStatus: Annotated[uint32_t, 4, 8, 0]
  Ccx0CpuOff: Annotated[uint32_t, 5, 2, 0]
  Ccx1CpuOff: Annotated[uint32_t, 5, 2, 2]
  GfxOffStatus: Annotated[uint32_t, 5, 2, 4]
  VddOff: Annotated[uint32_t, 5, 1, 6]
  InWhisperMode: Annotated[uint32_t, 5, 1, 7]
  ZstateStatus: Annotated[uint32_t, 6, 4, 0]
  spare0: Annotated[uint32_t, 6, 4, 4]
  DstateFun: Annotated[uint32_t, 7, 4, 0]
  DstateDev: Annotated[uint32_t, 7, 4, 4]
  P2JobHandler: Annotated[uint32_t, 8, 24, 0]
  RsmuPmiP2PendingCnt: Annotated[uint32_t, 11, 8, 0]
  PostCode: Annotated[uint32_t, 12, 32, 0]
  MsgPortBusy: Annotated[uint32_t, 16, 24, 0]
  RsmuPmiP1Pending: Annotated[uint32_t, 19, 1, 0]
  DfCstateExitPending: Annotated[uint32_t, 19, 1, 1]
  Ccx0Pc6ExitPending: Annotated[uint32_t, 19, 1, 2]
  Ccx1Pc6ExitPending: Annotated[uint32_t, 19, 1, 3]
  WarmResetPending: Annotated[uint32_t, 19, 1, 4]
  spare1: Annotated[uint32_t, 19, 3, 5]
  IdleMask: Annotated[uint32_t, 20, 32, 0]
@c.record
class FwStatus_t_v14_0_1(c.Struct):
  SIZE = 24
  DpmHandlerID: Annotated[uint32_t, 0, 8, 0]
  ActivityMonitorID: Annotated[uint32_t, 1, 8, 0]
  DpmTimerID: Annotated[uint32_t, 2, 8, 0]
  DpmHubID: Annotated[uint32_t, 3, 4, 0]
  DpmHubTask: Annotated[uint32_t, 3, 4, 4]
  CclkSyncStatus: Annotated[uint32_t, 4, 8, 0]
  ZstateStatus: Annotated[uint32_t, 5, 4, 0]
  Cpu1VddOff: Annotated[uint32_t, 5, 4, 4]
  DstateFun: Annotated[uint32_t, 6, 4, 0]
  DstateDev: Annotated[uint32_t, 6, 4, 4]
  GfxOffStatus: Annotated[uint32_t, 7, 2, 0]
  Cpu0Off: Annotated[uint32_t, 7, 2, 2]
  Cpu1Off: Annotated[uint32_t, 7, 2, 4]
  Cpu0VddOff: Annotated[uint32_t, 7, 2, 6]
  P2JobHandler: Annotated[uint32_t, 8, 32, 0]
  PostCode: Annotated[uint32_t, 12, 32, 0]
  MsgPortBusy: Annotated[uint32_t, 16, 15, 0]
  RsmuPmiP1Pending: Annotated[uint32_t, 17, 1, 7]
  RsmuPmiP2PendingCnt: Annotated[uint32_t, 18, 8, 0]
  DfCstateExitPending: Annotated[uint32_t, 19, 1, 0]
  Pc6EntryPending: Annotated[uint32_t, 19, 1, 1]
  Pc6ExitPending: Annotated[uint32_t, 19, 1, 2]
  WarmResetPending: Annotated[uint32_t, 19, 1, 3]
  Mp0ClkPending: Annotated[uint32_t, 19, 1, 4]
  InWhisperMode: Annotated[uint32_t, 19, 1, 5]
  spare2: Annotated[uint32_t, 19, 2, 6]
  IdleMask: Annotated[uint32_t, 20, 32, 0]
class FEATURE_PWR_DOMAIN_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
FEATURE_PWR_ALL = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_ALL', 0)
FEATURE_PWR_S5 = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_S5', 1)
FEATURE_PWR_BACO = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_BACO', 2)
FEATURE_PWR_SOC = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_SOC', 3)
FEATURE_PWR_GFX = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_GFX', 4)
FEATURE_PWR_DOMAIN_COUNT = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_DOMAIN_COUNT', 5)

class FEATURE_BTC_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
FEATURE_BTC_NOP = FEATURE_BTC_e.define('FEATURE_BTC_NOP', 0)
FEATURE_BTC_SAVE = FEATURE_BTC_e.define('FEATURE_BTC_SAVE', 1)
FEATURE_BTC_RESTORE = FEATURE_BTC_e.define('FEATURE_BTC_RESTORE', 2)
FEATURE_BTC_COUNT = FEATURE_BTC_e.define('FEATURE_BTC_COUNT', 3)

class SVI_PSI_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
SVI_PSI_0 = SVI_PSI_e.define('SVI_PSI_0', 0)
SVI_PSI_1 = SVI_PSI_e.define('SVI_PSI_1', 1)
SVI_PSI_2 = SVI_PSI_e.define('SVI_PSI_2', 2)
SVI_PSI_3 = SVI_PSI_e.define('SVI_PSI_3', 3)
SVI_PSI_4 = SVI_PSI_e.define('SVI_PSI_4', 4)
SVI_PSI_5 = SVI_PSI_e.define('SVI_PSI_5', 5)
SVI_PSI_6 = SVI_PSI_e.define('SVI_PSI_6', 6)
SVI_PSI_7 = SVI_PSI_e.define('SVI_PSI_7', 7)

class SMARTSHIFT_VERSION_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMARTSHIFT_VERSION_1 = SMARTSHIFT_VERSION_e.define('SMARTSHIFT_VERSION_1', 0)
SMARTSHIFT_VERSION_2 = SMARTSHIFT_VERSION_e.define('SMARTSHIFT_VERSION_2', 1)
SMARTSHIFT_VERSION_3 = SMARTSHIFT_VERSION_e.define('SMARTSHIFT_VERSION_3', 2)

class FOPT_CALC_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
FOPT_CALC_AC_CALC_DC = FOPT_CALC_e.define('FOPT_CALC_AC_CALC_DC', 0)
FOPT_PPTABLE_AC_CALC_DC = FOPT_CALC_e.define('FOPT_PPTABLE_AC_CALC_DC', 1)
FOPT_CALC_AC_PPTABLE_DC = FOPT_CALC_e.define('FOPT_CALC_AC_PPTABLE_DC', 2)
FOPT_PPTABLE_AC_PPTABLE_DC = FOPT_CALC_e.define('FOPT_PPTABLE_AC_PPTABLE_DC', 3)

class DRAM_BIT_WIDTH_TYPE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
DRAM_BIT_WIDTH_DISABLED = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_DISABLED', 0)
DRAM_BIT_WIDTH_X_8 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_8', 8)
DRAM_BIT_WIDTH_X_16 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_16', 16)
DRAM_BIT_WIDTH_X_32 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_32', 32)
DRAM_BIT_WIDTH_X_64 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_64', 64)
DRAM_BIT_WIDTH_X_128 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_128', 128)
DRAM_BIT_WIDTH_COUNT = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_COUNT', 129)

class I2cControllerPort_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_CONTROLLER_PORT_0 = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_0', 0)
I2C_CONTROLLER_PORT_1 = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_1', 1)
I2C_CONTROLLER_PORT_COUNT = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_COUNT', 2)

class I2cControllerName_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_CONTROLLER_NAME_VR_GFX = I2cControllerName_e.define('I2C_CONTROLLER_NAME_VR_GFX', 0)
I2C_CONTROLLER_NAME_VR_SOC = I2cControllerName_e.define('I2C_CONTROLLER_NAME_VR_SOC', 1)
I2C_CONTROLLER_NAME_VR_VMEMP = I2cControllerName_e.define('I2C_CONTROLLER_NAME_VR_VMEMP', 2)
I2C_CONTROLLER_NAME_VR_VDDIO = I2cControllerName_e.define('I2C_CONTROLLER_NAME_VR_VDDIO', 3)
I2C_CONTROLLER_NAME_LIQUID0 = I2cControllerName_e.define('I2C_CONTROLLER_NAME_LIQUID0', 4)
I2C_CONTROLLER_NAME_LIQUID1 = I2cControllerName_e.define('I2C_CONTROLLER_NAME_LIQUID1', 5)
I2C_CONTROLLER_NAME_PLX = I2cControllerName_e.define('I2C_CONTROLLER_NAME_PLX', 6)
I2C_CONTROLLER_NAME_FAN_INTAKE = I2cControllerName_e.define('I2C_CONTROLLER_NAME_FAN_INTAKE', 7)
I2C_CONTROLLER_NAME_COUNT = I2cControllerName_e.define('I2C_CONTROLLER_NAME_COUNT', 8)

class I2cControllerThrottler_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_CONTROLLER_THROTTLER_TYPE_NONE = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_TYPE_NONE', 0)
I2C_CONTROLLER_THROTTLER_VR_GFX = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_VR_GFX', 1)
I2C_CONTROLLER_THROTTLER_VR_SOC = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_VR_SOC', 2)
I2C_CONTROLLER_THROTTLER_VR_VMEMP = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_VR_VMEMP', 3)
I2C_CONTROLLER_THROTTLER_VR_VDDIO = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_VR_VDDIO', 4)
I2C_CONTROLLER_THROTTLER_LIQUID0 = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_LIQUID0', 5)
I2C_CONTROLLER_THROTTLER_LIQUID1 = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_LIQUID1', 6)
I2C_CONTROLLER_THROTTLER_PLX = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_PLX', 7)
I2C_CONTROLLER_THROTTLER_FAN_INTAKE = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_FAN_INTAKE', 8)
I2C_CONTROLLER_THROTTLER_INA3221 = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_INA3221', 9)
I2C_CONTROLLER_THROTTLER_COUNT = I2cControllerThrottler_e.define('I2C_CONTROLLER_THROTTLER_COUNT', 10)

class I2cControllerProtocol_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5', 0)
I2C_CONTROLLER_PROTOCOL_VR_IR35217 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_VR_IR35217', 1)
I2C_CONTROLLER_PROTOCOL_TMP_MAX31875 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_TMP_MAX31875', 2)
I2C_CONTROLLER_PROTOCOL_INA3221 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_INA3221', 3)
I2C_CONTROLLER_PROTOCOL_TMP_MAX6604 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_TMP_MAX6604', 4)
I2C_CONTROLLER_PROTOCOL_COUNT = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_COUNT', 5)

@c.record
class I2cControllerConfig_t(c.Struct):
  SIZE = 8
  Enabled: Annotated[uint8_t, 0]
  Speed: Annotated[uint8_t, 1]
  SlaveAddress: Annotated[uint8_t, 2]
  ControllerPort: Annotated[uint8_t, 3]
  ControllerName: Annotated[uint8_t, 4]
  ThermalThrotter: Annotated[uint8_t, 5]
  I2cProtocol: Annotated[uint8_t, 6]
  PaddingConfig: Annotated[uint8_t, 7]
uint8_t: TypeAlias = Annotated[int, ctypes.c_ubyte]
class I2cPort_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_PORT_SVD_SCL = I2cPort_e.define('I2C_PORT_SVD_SCL', 0)
I2C_PORT_GPIO = I2cPort_e.define('I2C_PORT_GPIO', 1)

class I2cSpeed_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_SPEED_FAST_50K = I2cSpeed_e.define('I2C_SPEED_FAST_50K', 0)
I2C_SPEED_FAST_100K = I2cSpeed_e.define('I2C_SPEED_FAST_100K', 1)
I2C_SPEED_FAST_400K = I2cSpeed_e.define('I2C_SPEED_FAST_400K', 2)
I2C_SPEED_FAST_PLUS_1M = I2cSpeed_e.define('I2C_SPEED_FAST_PLUS_1M', 3)
I2C_SPEED_HIGH_1M = I2cSpeed_e.define('I2C_SPEED_HIGH_1M', 4)
I2C_SPEED_HIGH_2M = I2cSpeed_e.define('I2C_SPEED_HIGH_2M', 5)
I2C_SPEED_COUNT = I2cSpeed_e.define('I2C_SPEED_COUNT', 6)

class I2cCmdType_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_CMD_READ = I2cCmdType_e.define('I2C_CMD_READ', 0)
I2C_CMD_WRITE = I2cCmdType_e.define('I2C_CMD_WRITE', 1)
I2C_CMD_COUNT = I2cCmdType_e.define('I2C_CMD_COUNT', 2)

@c.record
class SwI2cCmd_t(c.Struct):
  SIZE = 2
  ReadWriteData: Annotated[uint8_t, 0]
  CmdConfig: Annotated[uint8_t, 1]
@c.record
class SwI2cRequest_t(c.Struct):
  SIZE = 52
  I2CcontrollerPort: Annotated[uint8_t, 0]
  I2CSpeed: Annotated[uint8_t, 1]
  SlaveAddress: Annotated[uint8_t, 2]
  NumCmds: Annotated[uint8_t, 3]
  SwI2cCmds: Annotated[c.Array[SwI2cCmd_t, Literal[24]], 4]
@c.record
class SwI2cRequestExternal_t(c.Struct):
  SIZE = 116
  SwI2cRequest: Annotated[SwI2cRequest_t, 0]
  Spare: Annotated[c.Array[uint32_t, Literal[8]], 52]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 84]
@c.record
class EccInfo_t(c.Struct):
  SIZE = 24
  mca_umc_status: Annotated[uint64_t, 0]
  mca_umc_addr: Annotated[uint64_t, 8]
  ce_count_lo_chip: Annotated[uint16_t, 16]
  ce_count_hi_chip: Annotated[uint16_t, 18]
  eccPadding: Annotated[uint32_t, 20]
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
uint16_t: TypeAlias = Annotated[int, ctypes.c_uint16]
@c.record
class EccInfoTable_t(c.Struct):
  SIZE = 576
  EccInfo: Annotated[c.Array[EccInfo_t, Literal[24]], 0]
class EPCS_STATUS_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
EPCS_SHORTED_LIMIT = EPCS_STATUS_e.define('EPCS_SHORTED_LIMIT', 0)
EPCS_LOW_POWER_LIMIT = EPCS_STATUS_e.define('EPCS_LOW_POWER_LIMIT', 1)
EPCS_NORMAL_POWER_LIMIT = EPCS_STATUS_e.define('EPCS_NORMAL_POWER_LIMIT', 2)
EPCS_HIGH_POWER_LIMIT = EPCS_STATUS_e.define('EPCS_HIGH_POWER_LIMIT', 3)
EPCS_NOT_CONFIGURED = EPCS_STATUS_e.define('EPCS_NOT_CONFIGURED', 4)
EPCS_STATUS_COUNT = EPCS_STATUS_e.define('EPCS_STATUS_COUNT', 5)

class D3HOTSequence_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
BACO_SEQUENCE = D3HOTSequence_e.define('BACO_SEQUENCE', 0)
MSR_SEQUENCE = D3HOTSequence_e.define('MSR_SEQUENCE', 1)
BAMACO_SEQUENCE = D3HOTSequence_e.define('BAMACO_SEQUENCE', 2)
ULPS_SEQUENCE = D3HOTSequence_e.define('ULPS_SEQUENCE', 3)
D3HOT_SEQUENCE_COUNT = D3HOTSequence_e.define('D3HOT_SEQUENCE_COUNT', 4)

class PowerGatingMode_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PG_DYNAMIC_MODE = PowerGatingMode_e.define('PG_DYNAMIC_MODE', 0)
PG_STATIC_MODE = PowerGatingMode_e.define('PG_STATIC_MODE', 1)

class PowerGatingSettings_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PG_POWER_DOWN = PowerGatingSettings_e.define('PG_POWER_DOWN', 0)
PG_POWER_UP = PowerGatingSettings_e.define('PG_POWER_UP', 1)

@c.record
class QuadraticInt_t(c.Struct):
  SIZE = 12
  a: Annotated[uint32_t, 0]
  b: Annotated[uint32_t, 4]
  c: Annotated[uint32_t, 8]
@c.record
class LinearInt_t(c.Struct):
  SIZE = 8
  m: Annotated[uint32_t, 0]
  b: Annotated[uint32_t, 4]
@c.record
class DroopInt_t(c.Struct):
  SIZE = 12
  a: Annotated[uint32_t, 0]
  b: Annotated[uint32_t, 4]
  c: Annotated[uint32_t, 8]
class DCS_ARCH_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
DCS_ARCH_DISABLED = DCS_ARCH_e.define('DCS_ARCH_DISABLED', 0)
DCS_ARCH_FADCS = DCS_ARCH_e.define('DCS_ARCH_FADCS', 1)
DCS_ARCH_ASYNC = DCS_ARCH_e.define('DCS_ARCH_ASYNC', 2)

class PPCLK_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PPCLK_GFXCLK = PPCLK_e.define('PPCLK_GFXCLK', 0)
PPCLK_SOCCLK = PPCLK_e.define('PPCLK_SOCCLK', 1)
PPCLK_UCLK = PPCLK_e.define('PPCLK_UCLK', 2)
PPCLK_FCLK = PPCLK_e.define('PPCLK_FCLK', 3)
PPCLK_DCLK_0 = PPCLK_e.define('PPCLK_DCLK_0', 4)
PPCLK_VCLK_0 = PPCLK_e.define('PPCLK_VCLK_0', 5)
PPCLK_DISPCLK = PPCLK_e.define('PPCLK_DISPCLK', 6)
PPCLK_DPPCLK = PPCLK_e.define('PPCLK_DPPCLK', 7)
PPCLK_DPREFCLK = PPCLK_e.define('PPCLK_DPREFCLK', 8)
PPCLK_DCFCLK = PPCLK_e.define('PPCLK_DCFCLK', 9)
PPCLK_DTBCLK = PPCLK_e.define('PPCLK_DTBCLK', 10)
PPCLK_COUNT = PPCLK_e.define('PPCLK_COUNT', 11)

class VOLTAGE_MODE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
VOLTAGE_MODE_PPTABLE = VOLTAGE_MODE_e.define('VOLTAGE_MODE_PPTABLE', 0)
VOLTAGE_MODE_FUSES = VOLTAGE_MODE_e.define('VOLTAGE_MODE_FUSES', 1)
VOLTAGE_MODE_COUNT = VOLTAGE_MODE_e.define('VOLTAGE_MODE_COUNT', 2)

class AVFS_VOLTAGE_TYPE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
AVFS_VOLTAGE_GFX = AVFS_VOLTAGE_TYPE_e.define('AVFS_VOLTAGE_GFX', 0)
AVFS_VOLTAGE_SOC = AVFS_VOLTAGE_TYPE_e.define('AVFS_VOLTAGE_SOC', 1)
AVFS_VOLTAGE_COUNT = AVFS_VOLTAGE_TYPE_e.define('AVFS_VOLTAGE_COUNT', 2)

class AVFS_TEMP_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
AVFS_TEMP_COLD = AVFS_TEMP_e.define('AVFS_TEMP_COLD', 0)
AVFS_TEMP_HOT = AVFS_TEMP_e.define('AVFS_TEMP_HOT', 1)
AVFS_TEMP_COUNT = AVFS_TEMP_e.define('AVFS_TEMP_COUNT', 2)

class AVFS_D_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
AVFS_D_G = AVFS_D_e.define('AVFS_D_G', 0)
AVFS_D_COUNT = AVFS_D_e.define('AVFS_D_COUNT', 1)

class UCLK_DIV_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
UCLK_DIV_BY_1 = UCLK_DIV_e.define('UCLK_DIV_BY_1', 0)
UCLK_DIV_BY_2 = UCLK_DIV_e.define('UCLK_DIV_BY_2', 1)
UCLK_DIV_BY_4 = UCLK_DIV_e.define('UCLK_DIV_BY_4', 2)
UCLK_DIV_BY_8 = UCLK_DIV_e.define('UCLK_DIV_BY_8', 3)

class GpioIntPolarity_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
GPIO_INT_POLARITY_ACTIVE_LOW = GpioIntPolarity_e.define('GPIO_INT_POLARITY_ACTIVE_LOW', 0)
GPIO_INT_POLARITY_ACTIVE_HIGH = GpioIntPolarity_e.define('GPIO_INT_POLARITY_ACTIVE_HIGH', 1)

class PwrConfig_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PWR_CONFIG_TDP = PwrConfig_e.define('PWR_CONFIG_TDP', 0)
PWR_CONFIG_TGP = PwrConfig_e.define('PWR_CONFIG_TGP', 1)
PWR_CONFIG_TCP_ESTIMATED = PwrConfig_e.define('PWR_CONFIG_TCP_ESTIMATED', 2)
PWR_CONFIG_TCP_MEASURED = PwrConfig_e.define('PWR_CONFIG_TCP_MEASURED', 3)
PWR_CONFIG_TBP_DESKTOP = PwrConfig_e.define('PWR_CONFIG_TBP_DESKTOP', 4)
PWR_CONFIG_TBP_MOBILE = PwrConfig_e.define('PWR_CONFIG_TBP_MOBILE', 5)

@c.record
class DpmDescriptor_t(c.Struct):
  SIZE = 32
  Padding: Annotated[uint8_t, 0]
  SnapToDiscrete: Annotated[uint8_t, 1]
  NumDiscreteLevels: Annotated[uint8_t, 2]
  CalculateFopt: Annotated[uint8_t, 3]
  ConversionToAvfsClk: Annotated[LinearInt_t, 4]
  Padding3: Annotated[c.Array[uint32_t, Literal[3]], 12]
  Padding4: Annotated[uint16_t, 24]
  FoptimalDc: Annotated[uint16_t, 26]
  FoptimalAc: Annotated[uint16_t, 28]
  Padding2: Annotated[uint16_t, 30]
class PPT_THROTTLER_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PPT_THROTTLER_PPT0 = PPT_THROTTLER_e.define('PPT_THROTTLER_PPT0', 0)
PPT_THROTTLER_PPT1 = PPT_THROTTLER_e.define('PPT_THROTTLER_PPT1', 1)
PPT_THROTTLER_PPT2 = PPT_THROTTLER_e.define('PPT_THROTTLER_PPT2', 2)
PPT_THROTTLER_PPT3 = PPT_THROTTLER_e.define('PPT_THROTTLER_PPT3', 3)
PPT_THROTTLER_COUNT = PPT_THROTTLER_e.define('PPT_THROTTLER_COUNT', 4)

class TEMP_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
TEMP_EDGE = TEMP_e.define('TEMP_EDGE', 0)
TEMP_HOTSPOT = TEMP_e.define('TEMP_HOTSPOT', 1)
TEMP_HOTSPOT_GFX = TEMP_e.define('TEMP_HOTSPOT_GFX', 2)
TEMP_HOTSPOT_SOC = TEMP_e.define('TEMP_HOTSPOT_SOC', 3)
TEMP_MEM = TEMP_e.define('TEMP_MEM', 4)
TEMP_VR_GFX = TEMP_e.define('TEMP_VR_GFX', 5)
TEMP_VR_SOC = TEMP_e.define('TEMP_VR_SOC', 6)
TEMP_VR_MEM0 = TEMP_e.define('TEMP_VR_MEM0', 7)
TEMP_VR_MEM1 = TEMP_e.define('TEMP_VR_MEM1', 8)
TEMP_LIQUID0 = TEMP_e.define('TEMP_LIQUID0', 9)
TEMP_LIQUID1 = TEMP_e.define('TEMP_LIQUID1', 10)
TEMP_PLX = TEMP_e.define('TEMP_PLX', 11)
TEMP_COUNT = TEMP_e.define('TEMP_COUNT', 12)

class TDC_THROTTLER_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
TDC_THROTTLER_GFX = TDC_THROTTLER_e.define('TDC_THROTTLER_GFX', 0)
TDC_THROTTLER_SOC = TDC_THROTTLER_e.define('TDC_THROTTLER_SOC', 1)
TDC_THROTTLER_COUNT = TDC_THROTTLER_e.define('TDC_THROTTLER_COUNT', 2)

class SVI_PLANE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
SVI_PLANE_VDD_GFX = SVI_PLANE_e.define('SVI_PLANE_VDD_GFX', 0)
SVI_PLANE_VDD_SOC = SVI_PLANE_e.define('SVI_PLANE_VDD_SOC', 1)
SVI_PLANE_VDDCI_MEM = SVI_PLANE_e.define('SVI_PLANE_VDDCI_MEM', 2)
SVI_PLANE_VDDIO_MEM = SVI_PLANE_e.define('SVI_PLANE_VDDIO_MEM', 3)
SVI_PLANE_COUNT = SVI_PLANE_e.define('SVI_PLANE_COUNT', 4)

class PMFW_VOLT_PLANE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PMFW_VOLT_PLANE_GFX = PMFW_VOLT_PLANE_e.define('PMFW_VOLT_PLANE_GFX', 0)
PMFW_VOLT_PLANE_SOC = PMFW_VOLT_PLANE_e.define('PMFW_VOLT_PLANE_SOC', 1)
PMFW_VOLT_PLANE_COUNT = PMFW_VOLT_PLANE_e.define('PMFW_VOLT_PLANE_COUNT', 2)

class CUSTOMER_VARIANT_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
CUSTOMER_VARIANT_ROW = CUSTOMER_VARIANT_e.define('CUSTOMER_VARIANT_ROW', 0)
CUSTOMER_VARIANT_FALCON = CUSTOMER_VARIANT_e.define('CUSTOMER_VARIANT_FALCON', 1)
CUSTOMER_VARIANT_COUNT = CUSTOMER_VARIANT_e.define('CUSTOMER_VARIANT_COUNT', 2)

class POWER_SOURCE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
POWER_SOURCE_AC = POWER_SOURCE_e.define('POWER_SOURCE_AC', 0)
POWER_SOURCE_DC = POWER_SOURCE_e.define('POWER_SOURCE_DC', 1)
POWER_SOURCE_COUNT = POWER_SOURCE_e.define('POWER_SOURCE_COUNT', 2)

class MEM_VENDOR_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
MEM_VENDOR_PLACEHOLDER0 = MEM_VENDOR_e.define('MEM_VENDOR_PLACEHOLDER0', 0)
MEM_VENDOR_SAMSUNG = MEM_VENDOR_e.define('MEM_VENDOR_SAMSUNG', 1)
MEM_VENDOR_INFINEON = MEM_VENDOR_e.define('MEM_VENDOR_INFINEON', 2)
MEM_VENDOR_ELPIDA = MEM_VENDOR_e.define('MEM_VENDOR_ELPIDA', 3)
MEM_VENDOR_ETRON = MEM_VENDOR_e.define('MEM_VENDOR_ETRON', 4)
MEM_VENDOR_NANYA = MEM_VENDOR_e.define('MEM_VENDOR_NANYA', 5)
MEM_VENDOR_HYNIX = MEM_VENDOR_e.define('MEM_VENDOR_HYNIX', 6)
MEM_VENDOR_MOSEL = MEM_VENDOR_e.define('MEM_VENDOR_MOSEL', 7)
MEM_VENDOR_WINBOND = MEM_VENDOR_e.define('MEM_VENDOR_WINBOND', 8)
MEM_VENDOR_ESMT = MEM_VENDOR_e.define('MEM_VENDOR_ESMT', 9)
MEM_VENDOR_PLACEHOLDER1 = MEM_VENDOR_e.define('MEM_VENDOR_PLACEHOLDER1', 10)
MEM_VENDOR_PLACEHOLDER2 = MEM_VENDOR_e.define('MEM_VENDOR_PLACEHOLDER2', 11)
MEM_VENDOR_PLACEHOLDER3 = MEM_VENDOR_e.define('MEM_VENDOR_PLACEHOLDER3', 12)
MEM_VENDOR_PLACEHOLDER4 = MEM_VENDOR_e.define('MEM_VENDOR_PLACEHOLDER4', 13)
MEM_VENDOR_PLACEHOLDER5 = MEM_VENDOR_e.define('MEM_VENDOR_PLACEHOLDER5', 14)
MEM_VENDOR_MICRON = MEM_VENDOR_e.define('MEM_VENDOR_MICRON', 15)
MEM_VENDOR_COUNT = MEM_VENDOR_e.define('MEM_VENDOR_COUNT', 16)

class PP_GRTAVFS_HW_FUSE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PP_GRTAVFS_HW_CPO_CTL_ZONE0 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_CTL_ZONE0', 0)
PP_GRTAVFS_HW_CPO_CTL_ZONE1 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_CTL_ZONE1', 1)
PP_GRTAVFS_HW_CPO_CTL_ZONE2 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_CTL_ZONE2', 2)
PP_GRTAVFS_HW_CPO_CTL_ZONE3 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_CTL_ZONE3', 3)
PP_GRTAVFS_HW_CPO_CTL_ZONE4 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_CTL_ZONE4', 4)
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE0 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_0_31_ZONE0', 5)
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE0 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_32_63_ZONE0', 6)
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE1 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_0_31_ZONE1', 7)
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE1 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_32_63_ZONE1', 8)
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE2 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_0_31_ZONE2', 9)
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE2 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_32_63_ZONE2', 10)
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE3 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_0_31_ZONE3', 11)
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE3 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_32_63_ZONE3', 12)
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE4 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_0_31_ZONE4', 13)
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE4 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPO_EN_32_63_ZONE4', 14)
PP_GRTAVFS_HW_ZONE0_VF = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_ZONE0_VF', 15)
PP_GRTAVFS_HW_ZONE1_VF1 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_ZONE1_VF1', 16)
PP_GRTAVFS_HW_ZONE2_VF2 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_ZONE2_VF2', 17)
PP_GRTAVFS_HW_ZONE3_VF3 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_ZONE3_VF3', 18)
PP_GRTAVFS_HW_VOLTAGE_GB = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_VOLTAGE_GB', 19)
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE0 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE0', 20)
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE1 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE1', 21)
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE2 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE2', 22)
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE3 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE3', 23)
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE4 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE4', 24)
PP_GRTAVFS_HW_RESERVED_0 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_RESERVED_0', 25)
PP_GRTAVFS_HW_RESERVED_1 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_RESERVED_1', 26)
PP_GRTAVFS_HW_RESERVED_2 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_RESERVED_2', 27)
PP_GRTAVFS_HW_RESERVED_3 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_RESERVED_3', 28)
PP_GRTAVFS_HW_RESERVED_4 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_RESERVED_4', 29)
PP_GRTAVFS_HW_RESERVED_5 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_RESERVED_5', 30)
PP_GRTAVFS_HW_RESERVED_6 = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_RESERVED_6', 31)
PP_GRTAVFS_HW_FUSE_COUNT = PP_GRTAVFS_HW_FUSE_e.define('PP_GRTAVFS_HW_FUSE_COUNT', 32)

class PP_GRTAVFS_FW_COMMON_FUSE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_HOT_T0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_HOT_T0', 0)
PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_COLD_T0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_COLD_T0', 1)
PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_HOT_T0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_HOT_T0', 2)
PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_COLD_T0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_COLD_T0', 3)
PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_HOT_T0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_HOT_T0', 4)
PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_COLD_T0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_COLD_T0', 5)
PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_HOT_T0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_HOT_T0', 6)
PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_COLD_T0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_COLD_T0', 7)
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z0 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_SRAM_RM_Z0', 8)
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z1 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_SRAM_RM_Z1', 9)
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z2 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_SRAM_RM_Z2', 10)
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z3 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_SRAM_RM_Z3', 11)
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z4 = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_SRAM_RM_Z4', 12)
PP_GRTAVFS_FW_COMMON_FUSE_COUNT = PP_GRTAVFS_FW_COMMON_FUSE_e.define('PP_GRTAVFS_FW_COMMON_FUSE_COUNT', 13)

class PP_GRTAVFS_FW_SEP_FUSE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_NEG_1 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_NEG_1', 0)
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_0 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_0', 1)
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_1 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_1', 2)
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_2 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_2', 3)
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_3 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_3', 4)
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_4 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_4', 5)
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_NEG_1 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_NEG_1', 6)
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_0 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_0', 7)
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_1 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_1', 8)
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_2 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_2', 9)
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_3 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_3', 10)
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_4 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_4', 11)
PP_GRTAVFS_FW_SEP_FUSE_VF_NEG_1_FREQUENCY = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_VF_NEG_1_FREQUENCY', 12)
PP_GRTAVFS_FW_SEP_FUSE_VF4_FREQUENCY = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_VF4_FREQUENCY', 13)
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_0 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_0', 14)
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_1 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_1', 15)
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_2 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_2', 16)
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_3 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_3', 17)
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_4 = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_4', 18)
PP_GRTAVFS_FW_SEP_FUSE_COUNT = PP_GRTAVFS_FW_SEP_FUSE_e.define('PP_GRTAVFS_FW_SEP_FUSE_COUNT', 19)

@c.record
class SviTelemetryScale_t(c.Struct):
  SIZE = 4
  Offset: Annotated[int8_t, 0]
  Padding: Annotated[uint8_t, 1]
  MaxCurrent: Annotated[uint16_t, 2]
int8_t: TypeAlias = Annotated[int, ctypes.c_byte]
class PP_OD_POWER_FEATURE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PP_OD_POWER_FEATURE_ALWAYS_ENABLED = PP_OD_POWER_FEATURE_e.define('PP_OD_POWER_FEATURE_ALWAYS_ENABLED', 0)
PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING = PP_OD_POWER_FEATURE_e.define('PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING', 1)
PP_OD_POWER_FEATURE_ALWAYS_DISABLED = PP_OD_POWER_FEATURE_e.define('PP_OD_POWER_FEATURE_ALWAYS_DISABLED', 2)

class FanMode_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
FAN_MODE_AUTO = FanMode_e.define('FAN_MODE_AUTO', 0)
FAN_MODE_MANUAL_LINEAR = FanMode_e.define('FAN_MODE_MANUAL_LINEAR', 1)

class OD_FAIL_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
OD_NO_ERROR = OD_FAIL_e.define('OD_NO_ERROR', 0)
OD_REQUEST_ADVANCED_NOT_SUPPORTED = OD_FAIL_e.define('OD_REQUEST_ADVANCED_NOT_SUPPORTED', 1)
OD_UNSUPPORTED_FEATURE = OD_FAIL_e.define('OD_UNSUPPORTED_FEATURE', 2)
OD_INVALID_FEATURE_COMBO_ERROR = OD_FAIL_e.define('OD_INVALID_FEATURE_COMBO_ERROR', 3)
OD_GFXCLK_VF_CURVE_OFFSET_ERROR = OD_FAIL_e.define('OD_GFXCLK_VF_CURVE_OFFSET_ERROR', 4)
OD_VDD_GFX_VMAX_ERROR = OD_FAIL_e.define('OD_VDD_GFX_VMAX_ERROR', 5)
OD_VDD_SOC_VMAX_ERROR = OD_FAIL_e.define('OD_VDD_SOC_VMAX_ERROR', 6)
OD_PPT_ERROR = OD_FAIL_e.define('OD_PPT_ERROR', 7)
OD_FAN_MIN_PWM_ERROR = OD_FAIL_e.define('OD_FAN_MIN_PWM_ERROR', 8)
OD_FAN_ACOUSTIC_TARGET_ERROR = OD_FAIL_e.define('OD_FAN_ACOUSTIC_TARGET_ERROR', 9)
OD_FAN_ACOUSTIC_LIMIT_ERROR = OD_FAIL_e.define('OD_FAN_ACOUSTIC_LIMIT_ERROR', 10)
OD_FAN_TARGET_TEMP_ERROR = OD_FAIL_e.define('OD_FAN_TARGET_TEMP_ERROR', 11)
OD_FAN_ZERO_RPM_STOP_TEMP_ERROR = OD_FAIL_e.define('OD_FAN_ZERO_RPM_STOP_TEMP_ERROR', 12)
OD_FAN_CURVE_PWM_ERROR = OD_FAIL_e.define('OD_FAN_CURVE_PWM_ERROR', 13)
OD_FAN_CURVE_TEMP_ERROR = OD_FAIL_e.define('OD_FAN_CURVE_TEMP_ERROR', 14)
OD_FULL_CTRL_GFXCLK_ERROR = OD_FAIL_e.define('OD_FULL_CTRL_GFXCLK_ERROR', 15)
OD_FULL_CTRL_UCLK_ERROR = OD_FAIL_e.define('OD_FULL_CTRL_UCLK_ERROR', 16)
OD_FULL_CTRL_FCLK_ERROR = OD_FAIL_e.define('OD_FULL_CTRL_FCLK_ERROR', 17)
OD_FULL_CTRL_VDD_GFX_ERROR = OD_FAIL_e.define('OD_FULL_CTRL_VDD_GFX_ERROR', 18)
OD_FULL_CTRL_VDD_SOC_ERROR = OD_FAIL_e.define('OD_FULL_CTRL_VDD_SOC_ERROR', 19)
OD_TDC_ERROR = OD_FAIL_e.define('OD_TDC_ERROR', 20)
OD_GFXCLK_ERROR = OD_FAIL_e.define('OD_GFXCLK_ERROR', 21)
OD_UCLK_ERROR = OD_FAIL_e.define('OD_UCLK_ERROR', 22)
OD_FCLK_ERROR = OD_FAIL_e.define('OD_FCLK_ERROR', 23)
OD_OP_TEMP_ERROR = OD_FAIL_e.define('OD_OP_TEMP_ERROR', 24)
OD_OP_GFX_EDC_ERROR = OD_FAIL_e.define('OD_OP_GFX_EDC_ERROR', 25)
OD_OP_GFX_PCC_ERROR = OD_FAIL_e.define('OD_OP_GFX_PCC_ERROR', 26)
OD_POWER_FEATURE_CTRL_ERROR = OD_FAIL_e.define('OD_POWER_FEATURE_CTRL_ERROR', 27)

@c.record
class OverDriveTable_t(c.Struct):
  SIZE = 156
  FeatureCtrlMask: Annotated[uint32_t, 0]
  VoltageOffsetPerZoneBoundary: Annotated[c.Array[int16_t, Literal[6]], 4]
  VddGfxVmax: Annotated[uint16_t, 16]
  VddSocVmax: Annotated[uint16_t, 18]
  IdlePwrSavingFeaturesCtrl: Annotated[uint8_t, 20]
  RuntimePwrSavingFeaturesCtrl: Annotated[uint8_t, 21]
  Padding: Annotated[uint16_t, 22]
  GfxclkFoffset: Annotated[int16_t, 24]
  Padding1: Annotated[uint16_t, 26]
  UclkFmin: Annotated[uint16_t, 28]
  UclkFmax: Annotated[uint16_t, 30]
  FclkFmin: Annotated[uint16_t, 32]
  FclkFmax: Annotated[uint16_t, 34]
  Ppt: Annotated[int16_t, 36]
  Tdc: Annotated[int16_t, 38]
  FanLinearPwmPoints: Annotated[c.Array[uint8_t, Literal[6]], 40]
  FanLinearTempPoints: Annotated[c.Array[uint8_t, Literal[6]], 46]
  FanMinimumPwm: Annotated[uint16_t, 52]
  AcousticTargetRpmThreshold: Annotated[uint16_t, 54]
  AcousticLimitRpmThreshold: Annotated[uint16_t, 56]
  FanTargetTemperature: Annotated[uint16_t, 58]
  FanZeroRpmEnable: Annotated[uint8_t, 60]
  FanZeroRpmStopTemp: Annotated[uint8_t, 61]
  FanMode: Annotated[uint8_t, 62]
  MaxOpTemp: Annotated[uint8_t, 63]
  AdvancedOdModeEnabled: Annotated[uint8_t, 64]
  Padding2: Annotated[c.Array[uint8_t, Literal[3]], 65]
  GfxVoltageFullCtrlMode: Annotated[uint16_t, 68]
  SocVoltageFullCtrlMode: Annotated[uint16_t, 70]
  GfxclkFullCtrlMode: Annotated[uint16_t, 72]
  UclkFullCtrlMode: Annotated[uint16_t, 74]
  FclkFullCtrlMode: Annotated[uint16_t, 76]
  Padding3: Annotated[uint16_t, 78]
  GfxEdc: Annotated[int16_t, 80]
  GfxPccLimitControl: Annotated[int16_t, 82]
  GfxclkFmaxVmax: Annotated[uint16_t, 84]
  GfxclkFmaxVmaxTemperature: Annotated[uint8_t, 86]
  Padding4: Annotated[c.Array[uint8_t, Literal[1]], 87]
  Spare: Annotated[c.Array[uint32_t, Literal[9]], 88]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 124]
int16_t: TypeAlias = Annotated[int, ctypes.c_int16]
@c.record
class OverDriveTableExternal_t(c.Struct):
  SIZE = 156
  OverDriveTable: Annotated[OverDriveTable_t, 0]
@c.record
class OverDriveLimits_t(c.Struct):
  SIZE = 96
  FeatureCtrlMask: Annotated[uint32_t, 0]
  VoltageOffsetPerZoneBoundary: Annotated[c.Array[int16_t, Literal[6]], 4]
  VddGfxVmax: Annotated[uint16_t, 16]
  VddSocVmax: Annotated[uint16_t, 18]
  GfxclkFoffset: Annotated[int16_t, 20]
  Padding: Annotated[uint16_t, 22]
  UclkFmin: Annotated[uint16_t, 24]
  UclkFmax: Annotated[uint16_t, 26]
  FclkFmin: Annotated[uint16_t, 28]
  FclkFmax: Annotated[uint16_t, 30]
  Ppt: Annotated[int16_t, 32]
  Tdc: Annotated[int16_t, 34]
  FanLinearPwmPoints: Annotated[c.Array[uint8_t, Literal[6]], 36]
  FanLinearTempPoints: Annotated[c.Array[uint8_t, Literal[6]], 42]
  FanMinimumPwm: Annotated[uint16_t, 48]
  AcousticTargetRpmThreshold: Annotated[uint16_t, 50]
  AcousticLimitRpmThreshold: Annotated[uint16_t, 52]
  FanTargetTemperature: Annotated[uint16_t, 54]
  FanZeroRpmEnable: Annotated[uint8_t, 56]
  MaxOpTemp: Annotated[uint8_t, 57]
  Padding1: Annotated[c.Array[uint8_t, Literal[2]], 58]
  GfxVoltageFullCtrlMode: Annotated[uint16_t, 60]
  SocVoltageFullCtrlMode: Annotated[uint16_t, 62]
  GfxclkFullCtrlMode: Annotated[uint16_t, 64]
  UclkFullCtrlMode: Annotated[uint16_t, 66]
  FclkFullCtrlMode: Annotated[uint16_t, 68]
  GfxEdc: Annotated[int16_t, 70]
  GfxPccLimitControl: Annotated[int16_t, 72]
  Padding2: Annotated[int16_t, 74]
  Spare: Annotated[c.Array[uint32_t, Literal[5]], 76]
class BOARD_GPIO_TYPE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
BOARD_GPIO_SMUIO_0 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_0', 0)
BOARD_GPIO_SMUIO_1 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_1', 1)
BOARD_GPIO_SMUIO_2 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_2', 2)
BOARD_GPIO_SMUIO_3 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_3', 3)
BOARD_GPIO_SMUIO_4 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_4', 4)
BOARD_GPIO_SMUIO_5 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_5', 5)
BOARD_GPIO_SMUIO_6 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_6', 6)
BOARD_GPIO_SMUIO_7 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_7', 7)
BOARD_GPIO_SMUIO_8 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_8', 8)
BOARD_GPIO_SMUIO_9 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_9', 9)
BOARD_GPIO_SMUIO_10 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_10', 10)
BOARD_GPIO_SMUIO_11 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_11', 11)
BOARD_GPIO_SMUIO_12 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_12', 12)
BOARD_GPIO_SMUIO_13 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_13', 13)
BOARD_GPIO_SMUIO_14 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_14', 14)
BOARD_GPIO_SMUIO_15 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_15', 15)
BOARD_GPIO_SMUIO_16 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_16', 16)
BOARD_GPIO_SMUIO_17 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_17', 17)
BOARD_GPIO_SMUIO_18 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_18', 18)
BOARD_GPIO_SMUIO_19 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_19', 19)
BOARD_GPIO_SMUIO_20 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_20', 20)
BOARD_GPIO_SMUIO_21 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_21', 21)
BOARD_GPIO_SMUIO_22 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_22', 22)
BOARD_GPIO_SMUIO_23 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_23', 23)
BOARD_GPIO_SMUIO_24 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_24', 24)
BOARD_GPIO_SMUIO_25 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_25', 25)
BOARD_GPIO_SMUIO_26 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_26', 26)
BOARD_GPIO_SMUIO_27 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_27', 27)
BOARD_GPIO_SMUIO_28 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_28', 28)
BOARD_GPIO_SMUIO_29 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_29', 29)
BOARD_GPIO_SMUIO_30 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_30', 30)
BOARD_GPIO_SMUIO_31 = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_SMUIO_31', 31)
MAX_BOARD_GPIO_SMUIO_NUM = BOARD_GPIO_TYPE_e.define('MAX_BOARD_GPIO_SMUIO_NUM', 32)
BOARD_GPIO_DC_GEN_A = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GEN_A', 33)
BOARD_GPIO_DC_GEN_B = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GEN_B', 34)
BOARD_GPIO_DC_GEN_C = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GEN_C', 35)
BOARD_GPIO_DC_GEN_D = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GEN_D', 36)
BOARD_GPIO_DC_GEN_E = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GEN_E', 37)
BOARD_GPIO_DC_GEN_F = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GEN_F', 38)
BOARD_GPIO_DC_GEN_G = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GEN_G', 39)
BOARD_GPIO_DC_GENLK_CLK = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GENLK_CLK', 40)
BOARD_GPIO_DC_GENLK_VSYNC = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_GENLK_VSYNC', 41)
BOARD_GPIO_DC_SWAPLOCK_A = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_SWAPLOCK_A', 42)
BOARD_GPIO_DC_SWAPLOCK_B = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_DC_SWAPLOCK_B', 43)
MAX_BOARD_DC_GPIO_NUM = BOARD_GPIO_TYPE_e.define('MAX_BOARD_DC_GPIO_NUM', 44)
BOARD_GPIO_LV_EN = BOARD_GPIO_TYPE_e.define('BOARD_GPIO_LV_EN', 45)

@c.record
class BootValues_t(c.Struct):
  SIZE = 124
  InitImuClk: Annotated[uint16_t, 0]
  InitSocclk: Annotated[uint16_t, 2]
  InitMpioclk: Annotated[uint16_t, 4]
  InitSmnclk: Annotated[uint16_t, 6]
  InitDispClk: Annotated[uint16_t, 8]
  InitDppClk: Annotated[uint16_t, 10]
  InitDprefclk: Annotated[uint16_t, 12]
  InitDcfclk: Annotated[uint16_t, 14]
  InitDtbclk: Annotated[uint16_t, 16]
  InitDbguSocClk: Annotated[uint16_t, 18]
  InitGfxclk_bypass: Annotated[uint16_t, 20]
  InitMp1clk: Annotated[uint16_t, 22]
  InitLclk: Annotated[uint16_t, 24]
  InitDbguBacoClk: Annotated[uint16_t, 26]
  InitBaco400clk: Annotated[uint16_t, 28]
  InitBaco1200clk_bypass: Annotated[uint16_t, 30]
  InitBaco700clk_bypass: Annotated[uint16_t, 32]
  InitBaco500clk: Annotated[uint16_t, 34]
  InitDclk0: Annotated[uint16_t, 36]
  InitVclk0: Annotated[uint16_t, 38]
  InitFclk: Annotated[uint16_t, 40]
  Padding1: Annotated[uint16_t, 42]
  InitUclkLevel: Annotated[uint8_t, 44]
  Padding: Annotated[c.Array[uint8_t, Literal[3]], 45]
  InitVcoFreqPll0: Annotated[uint32_t, 48]
  InitVcoFreqPll1: Annotated[uint32_t, 52]
  InitVcoFreqPll2: Annotated[uint32_t, 56]
  InitVcoFreqPll3: Annotated[uint32_t, 60]
  InitVcoFreqPll4: Annotated[uint32_t, 64]
  InitVcoFreqPll5: Annotated[uint32_t, 68]
  InitVcoFreqPll6: Annotated[uint32_t, 72]
  InitVcoFreqPll7: Annotated[uint32_t, 76]
  InitVcoFreqPll8: Annotated[uint32_t, 80]
  InitGfx: Annotated[uint16_t, 84]
  InitSoc: Annotated[uint16_t, 86]
  InitVddIoMem: Annotated[uint16_t, 88]
  InitVddCiMem: Annotated[uint16_t, 90]
  Spare: Annotated[c.Array[uint32_t, Literal[8]], 92]
@c.record
class MsgLimits_t(c.Struct):
  SIZE = 116
  Power: Annotated[c.Array[c.Array[uint16_t, Literal[2]], Literal[4]], 0]
  Tdc: Annotated[c.Array[uint16_t, Literal[2]], 16]
  Temperature: Annotated[c.Array[uint16_t, Literal[12]], 20]
  PwmLimitMin: Annotated[uint8_t, 44]
  PwmLimitMax: Annotated[uint8_t, 45]
  FanTargetTemperature: Annotated[uint8_t, 46]
  Spare1: Annotated[c.Array[uint8_t, Literal[1]], 47]
  AcousticTargetRpmThresholdMin: Annotated[uint16_t, 48]
  AcousticTargetRpmThresholdMax: Annotated[uint16_t, 50]
  AcousticLimitRpmThresholdMin: Annotated[uint16_t, 52]
  AcousticLimitRpmThresholdMax: Annotated[uint16_t, 54]
  PccLimitMin: Annotated[uint16_t, 56]
  PccLimitMax: Annotated[uint16_t, 58]
  FanStopTempMin: Annotated[uint16_t, 60]
  FanStopTempMax: Annotated[uint16_t, 62]
  FanStartTempMin: Annotated[uint16_t, 64]
  FanStartTempMax: Annotated[uint16_t, 66]
  PowerMinPpt0: Annotated[c.Array[uint16_t, Literal[2]], 68]
  Spare: Annotated[c.Array[uint32_t, Literal[11]], 72]
@c.record
class DriverReportedClocks_t(c.Struct):
  SIZE = 28
  BaseClockAc: Annotated[uint16_t, 0]
  GameClockAc: Annotated[uint16_t, 2]
  BoostClockAc: Annotated[uint16_t, 4]
  BaseClockDc: Annotated[uint16_t, 6]
  GameClockDc: Annotated[uint16_t, 8]
  BoostClockDc: Annotated[uint16_t, 10]
  MaxReportedClock: Annotated[uint16_t, 12]
  Padding: Annotated[uint16_t, 14]
  Reserved: Annotated[c.Array[uint32_t, Literal[3]], 16]
@c.record
class AvfsDcBtcParams_t(c.Struct):
  SIZE = 20
  DcBtcEnabled: Annotated[uint8_t, 0]
  Padding: Annotated[c.Array[uint8_t, Literal[3]], 1]
  DcTol: Annotated[uint16_t, 4]
  DcBtcGb: Annotated[uint16_t, 6]
  DcBtcMin: Annotated[uint16_t, 8]
  DcBtcMax: Annotated[uint16_t, 10]
  DcBtcGbScalar: Annotated[LinearInt_t, 12]
@c.record
class AvfsFuseOverride_t(c.Struct):
  SIZE = 56
  AvfsTemp: Annotated[c.Array[uint16_t, Literal[2]], 0]
  VftFMin: Annotated[uint16_t, 4]
  VInversion: Annotated[uint16_t, 6]
  qVft: Annotated[c.Array[QuadraticInt_t, Literal[2]], 8]
  qAvfsGb: Annotated[QuadraticInt_t, 32]
  qAvfsGb2: Annotated[QuadraticInt_t, 44]
@c.record
class PFE_Settings_t(c.Struct):
  SIZE = 28
  Version: Annotated[uint8_t, 0]
  Spare8: Annotated[c.Array[uint8_t, Literal[3]], 1]
  FeaturesToRun: Annotated[c.Array[uint32_t, Literal[2]], 4]
  FwDStateMask: Annotated[uint32_t, 12]
  DebugOverrides: Annotated[uint32_t, 16]
  Spare: Annotated[c.Array[uint32_t, Literal[2]], 20]
@c.record
class SkuTable_t(c.Struct):
  SIZE = 3552
  Version: Annotated[uint32_t, 0]
  TotalPowerConfig: Annotated[uint8_t, 4]
  CustomerVariant: Annotated[uint8_t, 5]
  MemoryTemperatureTypeMask: Annotated[uint8_t, 6]
  SmartShiftVersion: Annotated[uint8_t, 7]
  SocketPowerLimitSpare: Annotated[c.Array[uint8_t, Literal[10]], 8]
  EnableLegacyPptLimit: Annotated[uint8_t, 18]
  UseInputTelemetry: Annotated[uint8_t, 19]
  SmartShiftMinReportedPptinDcs: Annotated[uint8_t, 20]
  PaddingPpt: Annotated[c.Array[uint8_t, Literal[7]], 21]
  HwCtfTempLimit: Annotated[uint16_t, 28]
  PaddingInfra: Annotated[uint16_t, 30]
  FitControllerFailureRateLimit: Annotated[uint32_t, 32]
  FitControllerGfxDutyCycle: Annotated[uint32_t, 36]
  FitControllerSocDutyCycle: Annotated[uint32_t, 40]
  FitControllerSocOffset: Annotated[uint32_t, 44]
  GfxApccPlusResidencyLimit: Annotated[uint32_t, 48]
  ThrottlerControlMask: Annotated[uint32_t, 52]
  UlvVoltageOffset: Annotated[c.Array[uint16_t, Literal[2]], 56]
  Padding: Annotated[c.Array[uint8_t, Literal[2]], 60]
  DeepUlvVoltageOffsetSoc: Annotated[uint16_t, 62]
  DefaultMaxVoltage: Annotated[c.Array[uint16_t, Literal[2]], 64]
  BoostMaxVoltage: Annotated[c.Array[uint16_t, Literal[2]], 68]
  VminTempHystersis: Annotated[c.Array[int16_t, Literal[2]], 72]
  VminTempThreshold: Annotated[c.Array[int16_t, Literal[2]], 76]
  Vmin_Hot_T0: Annotated[c.Array[uint16_t, Literal[2]], 80]
  Vmin_Cold_T0: Annotated[c.Array[uint16_t, Literal[2]], 84]
  Vmin_Hot_Eol: Annotated[c.Array[uint16_t, Literal[2]], 88]
  Vmin_Cold_Eol: Annotated[c.Array[uint16_t, Literal[2]], 92]
  Vmin_Aging_Offset: Annotated[c.Array[uint16_t, Literal[2]], 96]
  Spare_Vmin_Plat_Offset_Hot: Annotated[c.Array[uint16_t, Literal[2]], 100]
  Spare_Vmin_Plat_Offset_Cold: Annotated[c.Array[uint16_t, Literal[2]], 104]
  VcBtcFixedVminAgingOffset: Annotated[c.Array[uint16_t, Literal[2]], 108]
  VcBtcVmin2PsmDegrationGb: Annotated[c.Array[uint16_t, Literal[2]], 112]
  VcBtcPsmA: Annotated[c.Array[uint32_t, Literal[2]], 116]
  VcBtcPsmB: Annotated[c.Array[uint32_t, Literal[2]], 124]
  VcBtcVminA: Annotated[c.Array[uint32_t, Literal[2]], 132]
  VcBtcVminB: Annotated[c.Array[uint32_t, Literal[2]], 140]
  PerPartVminEnabled: Annotated[c.Array[uint8_t, Literal[2]], 148]
  VcBtcEnabled: Annotated[c.Array[uint8_t, Literal[2]], 150]
  SocketPowerLimitAcTau: Annotated[c.Array[uint16_t, Literal[4]], 152]
  SocketPowerLimitDcTau: Annotated[c.Array[uint16_t, Literal[4]], 160]
  Gfx_Vmin_droop: Annotated[QuadraticInt_t, 168]
  Soc_Vmin_droop: Annotated[QuadraticInt_t, 180]
  SpareVmin: Annotated[c.Array[uint32_t, Literal[6]], 192]
  DpmDescriptor: Annotated[c.Array[DpmDescriptor_t, Literal[11]], 216]
  FreqTableGfx: Annotated[c.Array[uint16_t, Literal[16]], 568]
  FreqTableVclk: Annotated[c.Array[uint16_t, Literal[8]], 600]
  FreqTableDclk: Annotated[c.Array[uint16_t, Literal[8]], 616]
  FreqTableSocclk: Annotated[c.Array[uint16_t, Literal[8]], 632]
  FreqTableUclk: Annotated[c.Array[uint16_t, Literal[6]], 648]
  FreqTableShadowUclk: Annotated[c.Array[uint16_t, Literal[6]], 660]
  FreqTableDispclk: Annotated[c.Array[uint16_t, Literal[8]], 672]
  FreqTableDppClk: Annotated[c.Array[uint16_t, Literal[8]], 688]
  FreqTableDprefclk: Annotated[c.Array[uint16_t, Literal[8]], 704]
  FreqTableDcfclk: Annotated[c.Array[uint16_t, Literal[8]], 720]
  FreqTableDtbclk: Annotated[c.Array[uint16_t, Literal[8]], 736]
  FreqTableFclk: Annotated[c.Array[uint16_t, Literal[8]], 752]
  DcModeMaxFreq: Annotated[c.Array[uint32_t, Literal[11]], 768]
  GfxclkAibFmax: Annotated[uint16_t, 812]
  GfxDpmPadding: Annotated[uint16_t, 814]
  GfxclkFgfxoffEntry: Annotated[uint16_t, 816]
  GfxclkFgfxoffExitImu: Annotated[uint16_t, 818]
  GfxclkFgfxoffExitRlc: Annotated[uint16_t, 820]
  GfxclkThrottleClock: Annotated[uint16_t, 822]
  EnableGfxPowerStagesGpio: Annotated[uint8_t, 824]
  GfxIdlePadding: Annotated[uint8_t, 825]
  SmsRepairWRCKClkDivEn: Annotated[uint8_t, 826]
  SmsRepairWRCKClkDivVal: Annotated[uint8_t, 827]
  GfxOffEntryEarlyMGCGEn: Annotated[uint8_t, 828]
  GfxOffEntryForceCGCGEn: Annotated[uint8_t, 829]
  GfxOffEntryForceCGCGDelayEn: Annotated[uint8_t, 830]
  GfxOffEntryForceCGCGDelayVal: Annotated[uint8_t, 831]
  GfxclkFreqGfxUlv: Annotated[uint16_t, 832]
  GfxIdlePadding2: Annotated[c.Array[uint8_t, Literal[2]], 834]
  GfxOffEntryHysteresis: Annotated[uint32_t, 836]
  GfxoffSpare: Annotated[c.Array[uint32_t, Literal[15]], 840]
  DfllMstrOscConfigA: Annotated[uint16_t, 900]
  DfllSlvOscConfigA: Annotated[uint16_t, 902]
  DfllBtcMasterScalerM: Annotated[uint32_t, 904]
  DfllBtcMasterScalerB: Annotated[int32_t, 908]
  DfllBtcSlaveScalerM: Annotated[uint32_t, 912]
  DfllBtcSlaveScalerB: Annotated[int32_t, 916]
  DfllPccAsWaitCtrl: Annotated[uint32_t, 920]
  DfllPccAsStepCtrl: Annotated[uint32_t, 924]
  GfxDfllSpare: Annotated[c.Array[uint32_t, Literal[9]], 928]
  DvoPsmDownThresholdVoltage: Annotated[uint32_t, 964]
  DvoPsmUpThresholdVoltage: Annotated[uint32_t, 968]
  DvoFmaxLowScaler: Annotated[uint32_t, 972]
  PaddingDcs: Annotated[uint32_t, 976]
  DcsMinGfxOffTime: Annotated[uint16_t, 980]
  DcsMaxGfxOffTime: Annotated[uint16_t, 982]
  DcsMinCreditAccum: Annotated[uint32_t, 984]
  DcsExitHysteresis: Annotated[uint16_t, 988]
  DcsTimeout: Annotated[uint16_t, 990]
  DcsPfGfxFopt: Annotated[uint32_t, 992]
  DcsPfUclkFopt: Annotated[uint32_t, 996]
  FoptEnabled: Annotated[uint8_t, 1000]
  DcsSpare2: Annotated[c.Array[uint8_t, Literal[3]], 1001]
  DcsFoptM: Annotated[uint32_t, 1004]
  DcsFoptB: Annotated[uint32_t, 1008]
  DcsSpare: Annotated[c.Array[uint32_t, Literal[9]], 1012]
  UseStrobeModeOptimizations: Annotated[uint8_t, 1048]
  PaddingMem: Annotated[c.Array[uint8_t, Literal[3]], 1049]
  UclkDpmPstates: Annotated[c.Array[uint8_t, Literal[6]], 1052]
  UclkDpmShadowPstates: Annotated[c.Array[uint8_t, Literal[6]], 1058]
  FreqTableUclkDiv: Annotated[c.Array[uint8_t, Literal[6]], 1064]
  FreqTableShadowUclkDiv: Annotated[c.Array[uint8_t, Literal[6]], 1070]
  MemVmempVoltage: Annotated[c.Array[uint16_t, Literal[6]], 1076]
  MemVddioVoltage: Annotated[c.Array[uint16_t, Literal[6]], 1088]
  DalDcModeMaxUclkFreq: Annotated[uint16_t, 1100]
  PaddingsMem: Annotated[c.Array[uint8_t, Literal[2]], 1102]
  PaddingFclk: Annotated[uint32_t, 1104]
  PcieGenSpeed: Annotated[c.Array[uint8_t, Literal[3]], 1108]
  PcieLaneCount: Annotated[c.Array[uint8_t, Literal[3]], 1111]
  LclkFreq: Annotated[c.Array[uint16_t, Literal[3]], 1114]
  OverrideGfxAvfsFuses: Annotated[uint8_t, 1120]
  GfxAvfsPadding: Annotated[c.Array[uint8_t, Literal[1]], 1121]
  DroopGBStDev: Annotated[uint16_t, 1122]
  SocHwRtAvfsFuses: Annotated[c.Array[uint32_t, Literal[32]], 1124]
  GfxL2HwRtAvfsFuses: Annotated[c.Array[uint32_t, Literal[32]], 1252]
  PsmDidt_Vcross: Annotated[c.Array[uint16_t, Literal[2]], 1380]
  PsmDidt_StaticDroop_A: Annotated[c.Array[uint32_t, Literal[3]], 1384]
  PsmDidt_StaticDroop_B: Annotated[c.Array[uint32_t, Literal[3]], 1396]
  PsmDidt_DynDroop_A: Annotated[c.Array[uint32_t, Literal[3]], 1408]
  PsmDidt_DynDroop_B: Annotated[c.Array[uint32_t, Literal[3]], 1420]
  spare_HwRtAvfsFuses: Annotated[c.Array[uint32_t, Literal[19]], 1432]
  SocCommonRtAvfs: Annotated[c.Array[uint32_t, Literal[13]], 1508]
  GfxCommonRtAvfs: Annotated[c.Array[uint32_t, Literal[13]], 1560]
  SocFwRtAvfsFuses: Annotated[c.Array[uint32_t, Literal[19]], 1612]
  GfxL2FwRtAvfsFuses: Annotated[c.Array[uint32_t, Literal[19]], 1688]
  spare_FwRtAvfsFuses: Annotated[c.Array[uint32_t, Literal[19]], 1764]
  Soc_Droop_PWL_F: Annotated[c.Array[uint32_t, Literal[5]], 1840]
  Soc_Droop_PWL_a: Annotated[c.Array[uint32_t, Literal[5]], 1860]
  Soc_Droop_PWL_b: Annotated[c.Array[uint32_t, Literal[5]], 1880]
  Soc_Droop_PWL_c: Annotated[c.Array[uint32_t, Literal[5]], 1900]
  Gfx_Droop_PWL_F: Annotated[c.Array[uint32_t, Literal[5]], 1920]
  Gfx_Droop_PWL_a: Annotated[c.Array[uint32_t, Literal[5]], 1940]
  Gfx_Droop_PWL_b: Annotated[c.Array[uint32_t, Literal[5]], 1960]
  Gfx_Droop_PWL_c: Annotated[c.Array[uint32_t, Literal[5]], 1980]
  Gfx_Static_PWL_Offset: Annotated[c.Array[uint32_t, Literal[5]], 2000]
  Soc_Static_PWL_Offset: Annotated[c.Array[uint32_t, Literal[5]], 2020]
  dGbV_dT_vmin: Annotated[uint32_t, 2040]
  dGbV_dT_vmax: Annotated[uint32_t, 2044]
  PaddingV2F: Annotated[c.Array[uint32_t, Literal[4]], 2048]
  DcBtcGfxParams: Annotated[AvfsDcBtcParams_t, 2064]
  SSCurve_GFX: Annotated[QuadraticInt_t, 2084]
  GfxAvfsSpare: Annotated[c.Array[uint32_t, Literal[29]], 2096]
  OverrideSocAvfsFuses: Annotated[uint8_t, 2212]
  MinSocAvfsRevision: Annotated[uint8_t, 2213]
  SocAvfsPadding: Annotated[c.Array[uint8_t, Literal[2]], 2214]
  SocAvfsFuseOverride: Annotated[c.Array[AvfsFuseOverride_t, Literal[1]], 2216]
  dBtcGbSoc: Annotated[c.Array[DroopInt_t, Literal[1]], 2272]
  qAgingGb: Annotated[c.Array[LinearInt_t, Literal[1]], 2284]
  qStaticVoltageOffset: Annotated[c.Array[QuadraticInt_t, Literal[1]], 2292]
  DcBtcSocParams: Annotated[c.Array[AvfsDcBtcParams_t, Literal[1]], 2304]
  SSCurve_SOC: Annotated[QuadraticInt_t, 2324]
  SocAvfsSpare: Annotated[c.Array[uint32_t, Literal[29]], 2336]
  BootValues: Annotated[BootValues_t, 2452]
  DriverReportedClocks: Annotated[DriverReportedClocks_t, 2576]
  MsgLimits: Annotated[MsgLimits_t, 2604]
  OverDriveLimitsBasicMin: Annotated[OverDriveLimits_t, 2720]
  OverDriveLimitsBasicMax: Annotated[OverDriveLimits_t, 2816]
  OverDriveLimitsAdvancedMin: Annotated[OverDriveLimits_t, 2912]
  OverDriveLimitsAdvancedMax: Annotated[OverDriveLimits_t, 3008]
  TotalBoardPowerSupport: Annotated[uint8_t, 3104]
  TotalBoardPowerPadding: Annotated[c.Array[uint8_t, Literal[1]], 3105]
  TotalBoardPowerRoc: Annotated[uint16_t, 3106]
  qFeffCoeffGameClock: Annotated[c.Array[QuadraticInt_t, Literal[2]], 3108]
  qFeffCoeffBaseClock: Annotated[c.Array[QuadraticInt_t, Literal[2]], 3132]
  qFeffCoeffBoostClock: Annotated[c.Array[QuadraticInt_t, Literal[2]], 3156]
  AptUclkGfxclkLookup: Annotated[c.Array[c.Array[int32_t, Literal[6]], Literal[2]], 3180]
  AptUclkGfxclkLookupHyst: Annotated[c.Array[c.Array[uint32_t, Literal[6]], Literal[2]], 3228]
  AptPadding: Annotated[uint32_t, 3276]
  GfxXvminDidtDroopThresh: Annotated[QuadraticInt_t, 3280]
  GfxXvminDidtResetDDWait: Annotated[uint32_t, 3292]
  GfxXvminDidtClkStopWait: Annotated[uint32_t, 3296]
  GfxXvminDidtFcsStepCtrl: Annotated[uint32_t, 3300]
  GfxXvminDidtFcsWaitCtrl: Annotated[uint32_t, 3304]
  PsmModeEnabled: Annotated[uint32_t, 3308]
  P2v_a: Annotated[uint32_t, 3312]
  P2v_b: Annotated[uint32_t, 3316]
  P2v_c: Annotated[uint32_t, 3320]
  T2p_a: Annotated[uint32_t, 3324]
  T2p_b: Annotated[uint32_t, 3328]
  T2p_c: Annotated[uint32_t, 3332]
  P2vTemp: Annotated[uint32_t, 3336]
  PsmDidtStaticSettings: Annotated[QuadraticInt_t, 3340]
  PsmDidtDynamicSettings: Annotated[QuadraticInt_t, 3352]
  PsmDidtAvgDiv: Annotated[uint8_t, 3364]
  PsmDidtForceStall: Annotated[uint8_t, 3365]
  PsmDidtReleaseTimer: Annotated[uint16_t, 3366]
  PsmDidtStallPattern: Annotated[uint32_t, 3368]
  CacEdcCacLeakageC0: Annotated[uint32_t, 3372]
  CacEdcCacLeakageC1: Annotated[uint32_t, 3376]
  CacEdcCacLeakageC2: Annotated[uint32_t, 3380]
  CacEdcCacLeakageC3: Annotated[uint32_t, 3384]
  CacEdcCacLeakageC4: Annotated[uint32_t, 3388]
  CacEdcCacLeakageC5: Annotated[uint32_t, 3392]
  CacEdcGfxClkScalar: Annotated[uint32_t, 3396]
  CacEdcGfxClkIntercept: Annotated[uint32_t, 3400]
  CacEdcCac_m: Annotated[uint32_t, 3404]
  CacEdcCac_b: Annotated[uint32_t, 3408]
  CacEdcCurrLimitGuardband: Annotated[uint32_t, 3412]
  CacEdcDynToTotalCacRatio: Annotated[uint32_t, 3416]
  XVmin_Gfx_EdcThreshScalar: Annotated[uint32_t, 3420]
  XVmin_Gfx_EdcEnableFreq: Annotated[uint32_t, 3424]
  XVmin_Gfx_EdcPccAsStepCtrl: Annotated[uint32_t, 3428]
  XVmin_Gfx_EdcPccAsWaitCtrl: Annotated[uint32_t, 3432]
  XVmin_Gfx_EdcThreshold: Annotated[uint16_t, 3436]
  XVmin_Gfx_EdcFiltHysWaitCtrl: Annotated[uint16_t, 3438]
  XVmin_Soc_EdcThreshScalar: Annotated[uint32_t, 3440]
  XVmin_Soc_EdcEnableFreq: Annotated[uint32_t, 3444]
  XVmin_Soc_EdcThreshold: Annotated[uint32_t, 3448]
  XVmin_Soc_EdcStepUpTime: Annotated[uint16_t, 3452]
  XVmin_Soc_EdcStepDownTime: Annotated[uint16_t, 3454]
  XVmin_Soc_EdcInitPccStep: Annotated[uint8_t, 3456]
  PaddingSocEdc: Annotated[c.Array[uint8_t, Literal[3]], 3457]
  GfxXvminFuseOverride: Annotated[uint8_t, 3460]
  SocXvminFuseOverride: Annotated[uint8_t, 3461]
  PaddingXvminFuseOverride: Annotated[c.Array[uint8_t, Literal[2]], 3462]
  GfxXvminFddTempLow: Annotated[uint8_t, 3464]
  GfxXvminFddTempHigh: Annotated[uint8_t, 3465]
  SocXvminFddTempLow: Annotated[uint8_t, 3466]
  SocXvminFddTempHigh: Annotated[uint8_t, 3467]
  GfxXvminFddVolt0: Annotated[uint16_t, 3468]
  GfxXvminFddVolt1: Annotated[uint16_t, 3470]
  GfxXvminFddVolt2: Annotated[uint16_t, 3472]
  SocXvminFddVolt0: Annotated[uint16_t, 3474]
  SocXvminFddVolt1: Annotated[uint16_t, 3476]
  SocXvminFddVolt2: Annotated[uint16_t, 3478]
  GfxXvminDsFddDsm: Annotated[c.Array[uint16_t, Literal[6]], 3480]
  GfxXvminEdcFddDsm: Annotated[c.Array[uint16_t, Literal[6]], 3492]
  SocXvminEdcFddDsm: Annotated[c.Array[uint16_t, Literal[6]], 3504]
  Spare: Annotated[uint32_t, 3516]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 3520]
int32_t: TypeAlias = Annotated[int, ctypes.c_int32]
@c.record
class Svi3RegulatorSettings_t(c.Struct):
  SIZE = 28
  SlewRateConditions: Annotated[uint8_t, 0]
  LoadLineAdjust: Annotated[uint8_t, 1]
  VoutOffset: Annotated[uint8_t, 2]
  VidMax: Annotated[uint8_t, 3]
  VidMin: Annotated[uint8_t, 4]
  TenBitTelEn: Annotated[uint8_t, 5]
  SixteenBitTelEn: Annotated[uint8_t, 6]
  OcpThresh: Annotated[uint8_t, 7]
  OcpWarnThresh: Annotated[uint8_t, 8]
  OcpSettings: Annotated[uint8_t, 9]
  VrhotThresh: Annotated[uint8_t, 10]
  OtpThresh: Annotated[uint8_t, 11]
  UvpOvpDeltaRef: Annotated[uint8_t, 12]
  PhaseShed: Annotated[uint8_t, 13]
  Padding: Annotated[c.Array[uint8_t, Literal[10]], 14]
  SettingOverrideMask: Annotated[uint32_t, 24]
@c.record
class BoardTable_t(c.Struct):
  SIZE = 528
  Version: Annotated[uint32_t, 0]
  I2cControllers: Annotated[c.Array[I2cControllerConfig_t, Literal[8]], 4]
  SlaveAddrMapping: Annotated[c.Array[uint8_t, Literal[4]], 68]
  VrPsiSupport: Annotated[c.Array[uint8_t, Literal[4]], 72]
  Svi3SvcSpeed: Annotated[uint32_t, 76]
  EnablePsi6: Annotated[c.Array[uint8_t, Literal[4]], 80]
  Svi3RegSettings: Annotated[c.Array[Svi3RegulatorSettings_t, Literal[4]], 84]
  LedOffGpio: Annotated[uint8_t, 196]
  FanOffGpio: Annotated[uint8_t, 197]
  GfxVrPowerStageOffGpio: Annotated[uint8_t, 198]
  AcDcGpio: Annotated[uint8_t, 199]
  AcDcPolarity: Annotated[uint8_t, 200]
  VR0HotGpio: Annotated[uint8_t, 201]
  VR0HotPolarity: Annotated[uint8_t, 202]
  GthrGpio: Annotated[uint8_t, 203]
  GthrPolarity: Annotated[uint8_t, 204]
  LedPin0: Annotated[uint8_t, 205]
  LedPin1: Annotated[uint8_t, 206]
  LedPin2: Annotated[uint8_t, 207]
  LedEnableMask: Annotated[uint8_t, 208]
  LedPcie: Annotated[uint8_t, 209]
  LedError: Annotated[uint8_t, 210]
  PaddingLed: Annotated[uint8_t, 211]
  UclkTrainingModeSpreadPercent: Annotated[uint8_t, 212]
  UclkSpreadPadding: Annotated[uint8_t, 213]
  UclkSpreadFreq: Annotated[uint16_t, 214]
  UclkSpreadPercent: Annotated[c.Array[uint8_t, Literal[16]], 216]
  GfxclkSpreadEnable: Annotated[uint8_t, 232]
  FclkSpreadPercent: Annotated[uint8_t, 233]
  FclkSpreadFreq: Annotated[uint16_t, 234]
  DramWidth: Annotated[uint8_t, 236]
  PaddingMem1: Annotated[c.Array[uint8_t, Literal[7]], 237]
  HsrEnabled: Annotated[uint8_t, 244]
  VddqOffEnabled: Annotated[uint8_t, 245]
  PaddingUmcFlags: Annotated[c.Array[uint8_t, Literal[2]], 246]
  Paddign1: Annotated[uint32_t, 248]
  BacoEntryDelay: Annotated[uint32_t, 252]
  FuseWritePowerMuxPresent: Annotated[uint8_t, 256]
  FuseWritePadding: Annotated[c.Array[uint8_t, Literal[3]], 257]
  LoadlineGfx: Annotated[uint32_t, 260]
  LoadlineSoc: Annotated[uint32_t, 264]
  GfxEdcLimit: Annotated[uint32_t, 268]
  SocEdcLimit: Annotated[uint32_t, 272]
  RestBoardPower: Annotated[uint32_t, 276]
  ConnectorsImpedance: Annotated[uint32_t, 280]
  EpcsSens0: Annotated[uint8_t, 284]
  EpcsSens1: Annotated[uint8_t, 285]
  PaddingEpcs: Annotated[c.Array[uint8_t, Literal[2]], 286]
  BoardSpare: Annotated[c.Array[uint32_t, Literal[52]], 288]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 496]
@c.record
class CustomSkuTable_t(c.Struct):
  SIZE = 360
  SocketPowerLimitAc: Annotated[c.Array[uint16_t, Literal[4]], 0]
  VrTdcLimit: Annotated[c.Array[uint16_t, Literal[2]], 8]
  TotalIdleBoardPowerM: Annotated[int16_t, 12]
  TotalIdleBoardPowerB: Annotated[int16_t, 14]
  TotalBoardPowerM: Annotated[int16_t, 16]
  TotalBoardPowerB: Annotated[int16_t, 18]
  TemperatureLimit: Annotated[c.Array[uint16_t, Literal[12]], 20]
  FanStopTemp: Annotated[c.Array[uint16_t, Literal[12]], 44]
  FanStartTemp: Annotated[c.Array[uint16_t, Literal[12]], 68]
  FanGain: Annotated[c.Array[uint16_t, Literal[12]], 92]
  FanPwmMin: Annotated[uint16_t, 116]
  AcousticTargetRpmThreshold: Annotated[uint16_t, 118]
  AcousticLimitRpmThreshold: Annotated[uint16_t, 120]
  FanMaximumRpm: Annotated[uint16_t, 122]
  MGpuAcousticLimitRpmThreshold: Annotated[uint16_t, 124]
  FanTargetGfxclk: Annotated[uint16_t, 126]
  TempInputSelectMask: Annotated[uint32_t, 128]
  FanZeroRpmEnable: Annotated[uint8_t, 132]
  FanTachEdgePerRev: Annotated[uint8_t, 133]
  FanPadding: Annotated[uint16_t, 134]
  FanTargetTemperature: Annotated[c.Array[uint16_t, Literal[12]], 136]
  FuzzyFan_ErrorSetDelta: Annotated[int16_t, 160]
  FuzzyFan_ErrorRateSetDelta: Annotated[int16_t, 162]
  FuzzyFan_PwmSetDelta: Annotated[int16_t, 164]
  FanPadding2: Annotated[uint16_t, 166]
  FwCtfLimit: Annotated[c.Array[uint16_t, Literal[12]], 168]
  IntakeTempEnableRPM: Annotated[uint16_t, 192]
  IntakeTempOffsetTemp: Annotated[int16_t, 194]
  IntakeTempReleaseTemp: Annotated[uint16_t, 196]
  IntakeTempHighIntakeAcousticLimit: Annotated[uint16_t, 198]
  IntakeTempAcouticLimitReleaseRate: Annotated[uint16_t, 200]
  FanAbnormalTempLimitOffset: Annotated[int16_t, 202]
  FanStalledTriggerRpm: Annotated[uint16_t, 204]
  FanAbnormalTriggerRpmCoeff: Annotated[uint16_t, 206]
  FanSpare: Annotated[c.Array[uint16_t, Literal[1]], 208]
  FanIntakeSensorSupport: Annotated[uint8_t, 210]
  FanIntakePadding: Annotated[uint8_t, 211]
  FanSpare2: Annotated[c.Array[uint32_t, Literal[12]], 212]
  ODFeatureCtrlMask: Annotated[uint32_t, 260]
  TemperatureLimit_Hynix: Annotated[uint16_t, 264]
  TemperatureLimit_Micron: Annotated[uint16_t, 266]
  TemperatureFwCtfLimit_Hynix: Annotated[uint16_t, 268]
  TemperatureFwCtfLimit_Micron: Annotated[uint16_t, 270]
  PlatformTdcLimit: Annotated[c.Array[uint16_t, Literal[2]], 272]
  SocketPowerLimitDc: Annotated[c.Array[uint16_t, Literal[4]], 276]
  SocketPowerLimitSmartShift2: Annotated[uint16_t, 284]
  CustomSkuSpare16b: Annotated[uint16_t, 286]
  CustomSkuSpare32b: Annotated[c.Array[uint32_t, Literal[10]], 288]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 328]
@c.record
class PPTable_t(c.Struct):
  SIZE = 4468
  PFE_Settings: Annotated[PFE_Settings_t, 0]
  SkuTable: Annotated[SkuTable_t, 28]
  CustomSkuTable: Annotated[CustomSkuTable_t, 3580]
  BoardTable: Annotated[BoardTable_t, 3940]
@c.record
class DriverSmuConfig_t(c.Struct):
  SIZE = 20
  GfxclkAverageLpfTau: Annotated[uint16_t, 0]
  FclkAverageLpfTau: Annotated[uint16_t, 2]
  UclkAverageLpfTau: Annotated[uint16_t, 4]
  GfxActivityLpfTau: Annotated[uint16_t, 6]
  UclkActivityLpfTau: Annotated[uint16_t, 8]
  UclkMaxActivityLpfTau: Annotated[uint16_t, 10]
  SocketPowerLpfTau: Annotated[uint16_t, 12]
  VcnClkAverageLpfTau: Annotated[uint16_t, 14]
  VcnUsageAverageLpfTau: Annotated[uint16_t, 16]
  PcieActivityLpTau: Annotated[uint16_t, 18]
@c.record
class DriverSmuConfigExternal_t(c.Struct):
  SIZE = 84
  DriverSmuConfig: Annotated[DriverSmuConfig_t, 0]
  Spare: Annotated[c.Array[uint32_t, Literal[8]], 20]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 52]
@c.record
class DriverInfoTable_t(c.Struct):
  SIZE = 372
  FreqTableGfx: Annotated[c.Array[uint16_t, Literal[16]], 0]
  FreqTableVclk: Annotated[c.Array[uint16_t, Literal[8]], 32]
  FreqTableDclk: Annotated[c.Array[uint16_t, Literal[8]], 48]
  FreqTableSocclk: Annotated[c.Array[uint16_t, Literal[8]], 64]
  FreqTableUclk: Annotated[c.Array[uint16_t, Literal[6]], 80]
  FreqTableDispclk: Annotated[c.Array[uint16_t, Literal[8]], 92]
  FreqTableDppClk: Annotated[c.Array[uint16_t, Literal[8]], 108]
  FreqTableDprefclk: Annotated[c.Array[uint16_t, Literal[8]], 124]
  FreqTableDcfclk: Annotated[c.Array[uint16_t, Literal[8]], 140]
  FreqTableDtbclk: Annotated[c.Array[uint16_t, Literal[8]], 156]
  FreqTableFclk: Annotated[c.Array[uint16_t, Literal[8]], 172]
  DcModeMaxFreq: Annotated[c.Array[uint16_t, Literal[11]], 188]
  Padding: Annotated[uint16_t, 210]
  Spare: Annotated[c.Array[uint32_t, Literal[32]], 212]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 340]
@c.record
class SmuMetrics_t(c.Struct):
  SIZE = 260
  CurrClock: Annotated[c.Array[uint32_t, Literal[11]], 0]
  AverageGfxclkFrequencyTarget: Annotated[uint16_t, 44]
  AverageGfxclkFrequencyPreDs: Annotated[uint16_t, 46]
  AverageGfxclkFrequencyPostDs: Annotated[uint16_t, 48]
  AverageFclkFrequencyPreDs: Annotated[uint16_t, 50]
  AverageFclkFrequencyPostDs: Annotated[uint16_t, 52]
  AverageMemclkFrequencyPreDs: Annotated[uint16_t, 54]
  AverageMemclkFrequencyPostDs: Annotated[uint16_t, 56]
  AverageVclk0Frequency: Annotated[uint16_t, 58]
  AverageDclk0Frequency: Annotated[uint16_t, 60]
  AverageVclk1Frequency: Annotated[uint16_t, 62]
  AverageDclk1Frequency: Annotated[uint16_t, 64]
  AveragePCIeBusy: Annotated[uint16_t, 66]
  dGPU_W_MAX: Annotated[uint16_t, 68]
  padding: Annotated[uint16_t, 70]
  MovingAverageGfxclkFrequencyTarget: Annotated[uint16_t, 72]
  MovingAverageGfxclkFrequencyPreDs: Annotated[uint16_t, 74]
  MovingAverageGfxclkFrequencyPostDs: Annotated[uint16_t, 76]
  MovingAverageFclkFrequencyPreDs: Annotated[uint16_t, 78]
  MovingAverageFclkFrequencyPostDs: Annotated[uint16_t, 80]
  MovingAverageMemclkFrequencyPreDs: Annotated[uint16_t, 82]
  MovingAverageMemclkFrequencyPostDs: Annotated[uint16_t, 84]
  MovingAverageVclk0Frequency: Annotated[uint16_t, 86]
  MovingAverageDclk0Frequency: Annotated[uint16_t, 88]
  MovingAverageGfxActivity: Annotated[uint16_t, 90]
  MovingAverageUclkActivity: Annotated[uint16_t, 92]
  MovingAverageVcn0ActivityPercentage: Annotated[uint16_t, 94]
  MovingAveragePCIeBusy: Annotated[uint16_t, 96]
  MovingAverageUclkActivity_MAX: Annotated[uint16_t, 98]
  MovingAverageSocketPower: Annotated[uint16_t, 100]
  MovingAveragePadding: Annotated[uint16_t, 102]
  MetricsCounter: Annotated[uint32_t, 104]
  AvgVoltage: Annotated[c.Array[uint16_t, Literal[4]], 108]
  AvgCurrent: Annotated[c.Array[uint16_t, Literal[4]], 116]
  AverageGfxActivity: Annotated[uint16_t, 124]
  AverageUclkActivity: Annotated[uint16_t, 126]
  AverageVcn0ActivityPercentage: Annotated[uint16_t, 128]
  Vcn1ActivityPercentage: Annotated[uint16_t, 130]
  EnergyAccumulator: Annotated[uint32_t, 132]
  AverageSocketPower: Annotated[uint16_t, 136]
  AverageTotalBoardPower: Annotated[uint16_t, 138]
  AvgTemperature: Annotated[c.Array[uint16_t, Literal[12]], 140]
  AvgTemperatureFanIntake: Annotated[uint16_t, 164]
  PcieRate: Annotated[uint8_t, 166]
  PcieWidth: Annotated[uint8_t, 167]
  AvgFanPwm: Annotated[uint8_t, 168]
  Padding: Annotated[c.Array[uint8_t, Literal[1]], 169]
  AvgFanRpm: Annotated[uint16_t, 170]
  ThrottlingPercentage: Annotated[c.Array[uint8_t, Literal[21]], 172]
  VmaxThrottlingPercentage: Annotated[uint8_t, 193]
  padding1: Annotated[c.Array[uint8_t, Literal[2]], 194]
  D3HotEntryCountPerMode: Annotated[c.Array[uint32_t, Literal[4]], 196]
  D3HotExitCountPerMode: Annotated[c.Array[uint32_t, Literal[4]], 212]
  ArmMsgReceivedCountPerMode: Annotated[c.Array[uint32_t, Literal[4]], 228]
  ApuSTAPMSmartShiftLimit: Annotated[uint16_t, 244]
  ApuSTAPMLimit: Annotated[uint16_t, 246]
  AvgApuSocketPower: Annotated[uint16_t, 248]
  AverageUclkActivity_MAX: Annotated[uint16_t, 250]
  PublicSerialNumberLower: Annotated[uint32_t, 252]
  PublicSerialNumberUpper: Annotated[uint32_t, 256]
@c.record
class SmuMetricsExternal_t(c.Struct):
  SIZE = 412
  SmuMetrics: Annotated[SmuMetrics_t, 0]
  Spare: Annotated[c.Array[uint32_t, Literal[30]], 260]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 380]
@c.record
class WatermarkRowGeneric_t(c.Struct):
  SIZE = 4
  WmSetting: Annotated[uint8_t, 0]
  Flags: Annotated[uint8_t, 1]
  Padding: Annotated[c.Array[uint8_t, Literal[2]], 2]
class WATERMARKS_FLAGS_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
WATERMARKS_CLOCK_RANGE = WATERMARKS_FLAGS_e.define('WATERMARKS_CLOCK_RANGE', 0)
WATERMARKS_DUMMY_PSTATE = WATERMARKS_FLAGS_e.define('WATERMARKS_DUMMY_PSTATE', 1)
WATERMARKS_MALL = WATERMARKS_FLAGS_e.define('WATERMARKS_MALL', 2)
WATERMARKS_COUNT = WATERMARKS_FLAGS_e.define('WATERMARKS_COUNT', 3)

@c.record
class Watermarks_t(c.Struct):
  SIZE = 16
  WatermarkRow: Annotated[c.Array[WatermarkRowGeneric_t, Literal[4]], 0]
@c.record
class WatermarksExternal_t(c.Struct):
  SIZE = 112
  Watermarks: Annotated[Watermarks_t, 0]
  Spare: Annotated[c.Array[uint32_t, Literal[16]], 16]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 80]
@c.record
class AvfsDebugTable_t(c.Struct):
  SIZE = 1368
  avgPsmCount: Annotated[c.Array[uint16_t, Literal[76]], 0]
  minPsmCount: Annotated[c.Array[uint16_t, Literal[76]], 152]
  maxPsmCount: Annotated[c.Array[uint16_t, Literal[76]], 304]
  avgPsmVoltage: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[76]], 456]
  minPsmVoltage: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[76]], 760]
  maxPsmVoltage: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[76]], 1064]
@c.record
class AvfsDebugTableExternal_t(c.Struct):
  SIZE = 1400
  AvfsDebugTable: Annotated[AvfsDebugTable_t, 0]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 1368]
@c.record
class DpmActivityMonitorCoeffInt_t(c.Struct):
  SIZE = 108
  Gfx_ActiveHystLimit: Annotated[uint8_t, 0]
  Gfx_IdleHystLimit: Annotated[uint8_t, 1]
  Gfx_FPS: Annotated[uint8_t, 2]
  Gfx_MinActiveFreqType: Annotated[uint8_t, 3]
  Gfx_BoosterFreqType: Annotated[uint8_t, 4]
  PaddingGfx: Annotated[uint8_t, 5]
  Gfx_MinActiveFreq: Annotated[uint16_t, 6]
  Gfx_BoosterFreq: Annotated[uint16_t, 8]
  Gfx_PD_Data_time_constant: Annotated[uint16_t, 10]
  Gfx_PD_Data_limit_a: Annotated[uint32_t, 12]
  Gfx_PD_Data_limit_b: Annotated[uint32_t, 16]
  Gfx_PD_Data_limit_c: Annotated[uint32_t, 20]
  Gfx_PD_Data_error_coeff: Annotated[uint32_t, 24]
  Gfx_PD_Data_error_rate_coeff: Annotated[uint32_t, 28]
  Fclk_ActiveHystLimit: Annotated[uint8_t, 32]
  Fclk_IdleHystLimit: Annotated[uint8_t, 33]
  Fclk_FPS: Annotated[uint8_t, 34]
  Fclk_MinActiveFreqType: Annotated[uint8_t, 35]
  Fclk_BoosterFreqType: Annotated[uint8_t, 36]
  PaddingFclk: Annotated[uint8_t, 37]
  Fclk_MinActiveFreq: Annotated[uint16_t, 38]
  Fclk_BoosterFreq: Annotated[uint16_t, 40]
  Fclk_PD_Data_time_constant: Annotated[uint16_t, 42]
  Fclk_PD_Data_limit_a: Annotated[uint32_t, 44]
  Fclk_PD_Data_limit_b: Annotated[uint32_t, 48]
  Fclk_PD_Data_limit_c: Annotated[uint32_t, 52]
  Fclk_PD_Data_error_coeff: Annotated[uint32_t, 56]
  Fclk_PD_Data_error_rate_coeff: Annotated[uint32_t, 60]
  Mem_UpThreshold_Limit: Annotated[c.Array[uint32_t, Literal[6]], 64]
  Mem_UpHystLimit: Annotated[c.Array[uint8_t, Literal[6]], 88]
  Mem_DownHystLimit: Annotated[c.Array[uint16_t, Literal[6]], 94]
  Mem_Fps: Annotated[uint16_t, 106]
@c.record
class DpmActivityMonitorCoeffIntExternal_t(c.Struct):
  SIZE = 140
  DpmActivityMonitorCoeffInt: Annotated[DpmActivityMonitorCoeffInt_t, 0]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 108]
@c.record
class struct_smu_hw_power_state(c.Struct):
  SIZE = 4
  magic: Annotated[Annotated[int, ctypes.c_uint32], 0]
class struct_smu_power_state(ctypes.Structure): pass
class enum_smu_state_ui_label(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_STATE_UI_LABEL_NONE = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_NONE', 0)
SMU_STATE_UI_LABEL_BATTERY = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BATTERY', 1)
SMU_STATE_UI_TABEL_MIDDLE_LOW = enum_smu_state_ui_label.define('SMU_STATE_UI_TABEL_MIDDLE_LOW', 2)
SMU_STATE_UI_LABEL_BALLANCED = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BALLANCED', 3)
SMU_STATE_UI_LABEL_MIDDLE_HIGHT = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_MIDDLE_HIGHT', 4)
SMU_STATE_UI_LABEL_PERFORMANCE = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_PERFORMANCE', 5)
SMU_STATE_UI_LABEL_BACO = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BACO', 6)

class enum_smu_state_classification_flag(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_STATE_CLASSIFICATION_FLAG_BOOT = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_BOOT', 1)
SMU_STATE_CLASSIFICATION_FLAG_THERMAL = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_THERMAL', 2)
SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE', 4)
SMU_STATE_CLASSIFICATION_FLAG_RESET = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_RESET', 8)
SMU_STATE_CLASSIFICATION_FLAG_FORCED = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_FORCED', 16)
SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE', 32)
SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE', 64)
SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE', 128)
SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE', 256)
SMU_STATE_CLASSIFICATION_FLAG_UVD = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_UVD', 512)
SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW', 1024)
SMU_STATE_CLASSIFICATION_FLAG_ACPI = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_ACPI', 2048)
SMU_STATE_CLASSIFICATION_FLAG_HD2 = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_HD2', 4096)
SMU_STATE_CLASSIFICATION_FLAG_UVD_HD = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_UVD_HD', 8192)
SMU_STATE_CLASSIFICATION_FLAG_UVD_SD = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_UVD_SD', 16384)
SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE', 32768)
SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE', 65536)
SMU_STATE_CLASSIFICATION_FLAG_BACO = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_BACO', 131072)
SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2 = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2', 262144)
SMU_STATE_CLASSIFICATION_FLAG_ULV = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_ULV', 524288)
SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC', 1048576)

@c.record
class struct_smu_state_classification_block(c.Struct):
  SIZE = 16
  ui_label: Annotated[enum_smu_state_ui_label, 0]
  flags: Annotated[enum_smu_state_classification_flag, 4]
  bios_index: Annotated[Annotated[int, ctypes.c_int32], 8]
  temporary_state: Annotated[Annotated[bool, ctypes.c_bool], 12]
  to_be_deleted: Annotated[Annotated[bool, ctypes.c_bool], 13]
@c.record
class struct_smu_state_pcie_block(c.Struct):
  SIZE = 4
  lanes: Annotated[Annotated[int, ctypes.c_uint32], 0]
class enum_smu_refreshrate_source(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_REFRESHRATE_SOURCE_EDID = enum_smu_refreshrate_source.define('SMU_REFRESHRATE_SOURCE_EDID', 0)
SMU_REFRESHRATE_SOURCE_EXPLICIT = enum_smu_refreshrate_source.define('SMU_REFRESHRATE_SOURCE_EXPLICIT', 1)

@c.record
class struct_smu_state_display_block(c.Struct):
  SIZE = 20
  disable_frame_modulation: Annotated[Annotated[bool, ctypes.c_bool], 0]
  limit_refreshrate: Annotated[Annotated[bool, ctypes.c_bool], 1]
  refreshrate_source: Annotated[enum_smu_refreshrate_source, 4]
  explicit_refreshrate: Annotated[Annotated[int, ctypes.c_int32], 8]
  edid_refreshrate_index: Annotated[Annotated[int, ctypes.c_int32], 12]
  enable_vari_bright: Annotated[Annotated[bool, ctypes.c_bool], 16]
@c.record
class struct_smu_state_memory_block(c.Struct):
  SIZE = 5
  dll_off: Annotated[Annotated[bool, ctypes.c_bool], 0]
  m3arb: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  unused: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 2]
@c.record
class struct_smu_state_software_algorithm_block(c.Struct):
  SIZE = 2
  disable_load_balancing: Annotated[Annotated[bool, ctypes.c_bool], 0]
  enable_sleep_for_timestamps: Annotated[Annotated[bool, ctypes.c_bool], 1]
@c.record
class struct_smu_temperature_range(c.Struct):
  SIZE = 44
  min: Annotated[Annotated[int, ctypes.c_int32], 0]
  max: Annotated[Annotated[int, ctypes.c_int32], 4]
  edge_emergency_max: Annotated[Annotated[int, ctypes.c_int32], 8]
  hotspot_min: Annotated[Annotated[int, ctypes.c_int32], 12]
  hotspot_crit_max: Annotated[Annotated[int, ctypes.c_int32], 16]
  hotspot_emergency_max: Annotated[Annotated[int, ctypes.c_int32], 20]
  mem_min: Annotated[Annotated[int, ctypes.c_int32], 24]
  mem_crit_max: Annotated[Annotated[int, ctypes.c_int32], 28]
  mem_emergency_max: Annotated[Annotated[int, ctypes.c_int32], 32]
  software_shutdown_temp: Annotated[Annotated[int, ctypes.c_int32], 36]
  software_shutdown_temp_offset: Annotated[Annotated[int, ctypes.c_int32], 40]
@c.record
class struct_smu_state_validation_block(c.Struct):
  SIZE = 3
  single_display_only: Annotated[Annotated[bool, ctypes.c_bool], 0]
  disallow_on_dc: Annotated[Annotated[bool, ctypes.c_bool], 1]
  supported_power_levels: Annotated[Annotated[int, ctypes.c_ubyte], 2]
@c.record
class struct_smu_uvd_clocks(c.Struct):
  SIZE = 8
  vclk: Annotated[Annotated[int, ctypes.c_uint32], 0]
  dclk: Annotated[Annotated[int, ctypes.c_uint32], 4]
class enum_smu_power_src_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_POWER_SOURCE_AC = enum_smu_power_src_type.define('SMU_POWER_SOURCE_AC', 0)
SMU_POWER_SOURCE_DC = enum_smu_power_src_type.define('SMU_POWER_SOURCE_DC', 1)
SMU_POWER_SOURCE_COUNT = enum_smu_power_src_type.define('SMU_POWER_SOURCE_COUNT', 2)

class enum_smu_ppt_limit_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_DEFAULT_PPT_LIMIT = enum_smu_ppt_limit_type.define('SMU_DEFAULT_PPT_LIMIT', 0)
SMU_FAST_PPT_LIMIT = enum_smu_ppt_limit_type.define('SMU_FAST_PPT_LIMIT', 1)

class enum_smu_ppt_limit_level(Annotated[int, ctypes.c_int32], c.Enum): pass
SMU_PPT_LIMIT_MIN = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_MIN', -1)
SMU_PPT_LIMIT_CURRENT = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_CURRENT', 0)
SMU_PPT_LIMIT_DEFAULT = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_DEFAULT', 1)
SMU_PPT_LIMIT_MAX = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_MAX', 2)

class enum_smu_memory_pool_size(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_MEMORY_POOL_SIZE_ZERO = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_ZERO', 0)
SMU_MEMORY_POOL_SIZE_256_MB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_256_MB', 268435456)
SMU_MEMORY_POOL_SIZE_512_MB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_512_MB', 536870912)
SMU_MEMORY_POOL_SIZE_1_GB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_1_GB', 1073741824)
SMU_MEMORY_POOL_SIZE_2_GB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_2_GB', 2147483648)

class enum_smu_clk_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_GFXCLK = enum_smu_clk_type.define('SMU_GFXCLK', 0)
SMU_VCLK = enum_smu_clk_type.define('SMU_VCLK', 1)
SMU_DCLK = enum_smu_clk_type.define('SMU_DCLK', 2)
SMU_VCLK1 = enum_smu_clk_type.define('SMU_VCLK1', 3)
SMU_DCLK1 = enum_smu_clk_type.define('SMU_DCLK1', 4)
SMU_ECLK = enum_smu_clk_type.define('SMU_ECLK', 5)
SMU_SOCCLK = enum_smu_clk_type.define('SMU_SOCCLK', 6)
SMU_UCLK = enum_smu_clk_type.define('SMU_UCLK', 7)
SMU_DCEFCLK = enum_smu_clk_type.define('SMU_DCEFCLK', 8)
SMU_DISPCLK = enum_smu_clk_type.define('SMU_DISPCLK', 9)
SMU_PIXCLK = enum_smu_clk_type.define('SMU_PIXCLK', 10)
SMU_PHYCLK = enum_smu_clk_type.define('SMU_PHYCLK', 11)
SMU_FCLK = enum_smu_clk_type.define('SMU_FCLK', 12)
SMU_SCLK = enum_smu_clk_type.define('SMU_SCLK', 13)
SMU_MCLK = enum_smu_clk_type.define('SMU_MCLK', 14)
SMU_PCIE = enum_smu_clk_type.define('SMU_PCIE', 15)
SMU_LCLK = enum_smu_clk_type.define('SMU_LCLK', 16)
SMU_OD_CCLK = enum_smu_clk_type.define('SMU_OD_CCLK', 17)
SMU_OD_SCLK = enum_smu_clk_type.define('SMU_OD_SCLK', 18)
SMU_OD_MCLK = enum_smu_clk_type.define('SMU_OD_MCLK', 19)
SMU_OD_VDDC_CURVE = enum_smu_clk_type.define('SMU_OD_VDDC_CURVE', 20)
SMU_OD_RANGE = enum_smu_clk_type.define('SMU_OD_RANGE', 21)
SMU_OD_VDDGFX_OFFSET = enum_smu_clk_type.define('SMU_OD_VDDGFX_OFFSET', 22)
SMU_OD_FAN_CURVE = enum_smu_clk_type.define('SMU_OD_FAN_CURVE', 23)
SMU_OD_ACOUSTIC_LIMIT = enum_smu_clk_type.define('SMU_OD_ACOUSTIC_LIMIT', 24)
SMU_OD_ACOUSTIC_TARGET = enum_smu_clk_type.define('SMU_OD_ACOUSTIC_TARGET', 25)
SMU_OD_FAN_TARGET_TEMPERATURE = enum_smu_clk_type.define('SMU_OD_FAN_TARGET_TEMPERATURE', 26)
SMU_OD_FAN_MINIMUM_PWM = enum_smu_clk_type.define('SMU_OD_FAN_MINIMUM_PWM', 27)
SMU_CLK_COUNT = enum_smu_clk_type.define('SMU_CLK_COUNT', 28)

@c.record
class struct_smu_user_dpm_profile(c.Struct):
  SIZE = 140
  fan_mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
  power_limit: Annotated[Annotated[int, ctypes.c_uint32], 4]
  fan_speed_pwm: Annotated[Annotated[int, ctypes.c_uint32], 8]
  fan_speed_rpm: Annotated[Annotated[int, ctypes.c_uint32], 12]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  user_od: Annotated[Annotated[int, ctypes.c_uint32], 20]
  clk_mask: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[28]], 24]
  clk_dependency: Annotated[Annotated[int, ctypes.c_uint32], 136]
@c.record
class struct_smu_table(c.Struct):
  SIZE = 48
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  align: Annotated[Annotated[int, ctypes.c_uint32], 8]
  domain: Annotated[Annotated[int, ctypes.c_ubyte], 12]
  mc_address: Annotated[Annotated[int, ctypes.c_uint64], 16]
  cpu_addr: Annotated[ctypes.c_void_p, 24]
  bo: Annotated[c.POINTER[struct_amdgpu_bo], 32]
  version: Annotated[Annotated[int, ctypes.c_uint32], 40]
class struct_amdgpu_bo(ctypes.Structure): pass
class enum_smu_perf_level_designation(Annotated[int, ctypes.c_uint32], c.Enum): pass
PERF_LEVEL_ACTIVITY = enum_smu_perf_level_designation.define('PERF_LEVEL_ACTIVITY', 0)
PERF_LEVEL_POWER_CONTAINMENT = enum_smu_perf_level_designation.define('PERF_LEVEL_POWER_CONTAINMENT', 1)

@c.record
class struct_smu_performance_level(c.Struct):
  SIZE = 24
  core_clock: Annotated[Annotated[int, ctypes.c_uint32], 0]
  memory_clock: Annotated[Annotated[int, ctypes.c_uint32], 4]
  vddc: Annotated[Annotated[int, ctypes.c_uint32], 8]
  vddci: Annotated[Annotated[int, ctypes.c_uint32], 12]
  non_local_mem_freq: Annotated[Annotated[int, ctypes.c_uint32], 16]
  non_local_mem_width: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_smu_clock_info(c.Struct):
  SIZE = 24
  min_mem_clk: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_mem_clk: Annotated[Annotated[int, ctypes.c_uint32], 4]
  min_eng_clk: Annotated[Annotated[int, ctypes.c_uint32], 8]
  max_eng_clk: Annotated[Annotated[int, ctypes.c_uint32], 12]
  min_bus_bandwidth: Annotated[Annotated[int, ctypes.c_uint32], 16]
  max_bus_bandwidth: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_smu_bios_boot_up_values(c.Struct):
  SIZE = 68
  revision: Annotated[Annotated[int, ctypes.c_uint32], 0]
  gfxclk: Annotated[Annotated[int, ctypes.c_uint32], 4]
  uclk: Annotated[Annotated[int, ctypes.c_uint32], 8]
  socclk: Annotated[Annotated[int, ctypes.c_uint32], 12]
  dcefclk: Annotated[Annotated[int, ctypes.c_uint32], 16]
  eclk: Annotated[Annotated[int, ctypes.c_uint32], 20]
  vclk: Annotated[Annotated[int, ctypes.c_uint32], 24]
  dclk: Annotated[Annotated[int, ctypes.c_uint32], 28]
  vddc: Annotated[Annotated[int, ctypes.c_uint16], 32]
  vddci: Annotated[Annotated[int, ctypes.c_uint16], 34]
  mvddc: Annotated[Annotated[int, ctypes.c_uint16], 36]
  vdd_gfx: Annotated[Annotated[int, ctypes.c_uint16], 38]
  cooling_id: Annotated[Annotated[int, ctypes.c_ubyte], 40]
  pp_table_id: Annotated[Annotated[int, ctypes.c_uint32], 44]
  format_revision: Annotated[Annotated[int, ctypes.c_uint32], 48]
  content_revision: Annotated[Annotated[int, ctypes.c_uint32], 52]
  fclk: Annotated[Annotated[int, ctypes.c_uint32], 56]
  lclk: Annotated[Annotated[int, ctypes.c_uint32], 60]
  firmware_caps: Annotated[Annotated[int, ctypes.c_uint32], 64]
class enum_smu_table_id(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_TABLE_PPTABLE = enum_smu_table_id.define('SMU_TABLE_PPTABLE', 0)
SMU_TABLE_WATERMARKS = enum_smu_table_id.define('SMU_TABLE_WATERMARKS', 1)
SMU_TABLE_CUSTOM_DPM = enum_smu_table_id.define('SMU_TABLE_CUSTOM_DPM', 2)
SMU_TABLE_DPMCLOCKS = enum_smu_table_id.define('SMU_TABLE_DPMCLOCKS', 3)
SMU_TABLE_AVFS = enum_smu_table_id.define('SMU_TABLE_AVFS', 4)
SMU_TABLE_AVFS_PSM_DEBUG = enum_smu_table_id.define('SMU_TABLE_AVFS_PSM_DEBUG', 5)
SMU_TABLE_AVFS_FUSE_OVERRIDE = enum_smu_table_id.define('SMU_TABLE_AVFS_FUSE_OVERRIDE', 6)
SMU_TABLE_PMSTATUSLOG = enum_smu_table_id.define('SMU_TABLE_PMSTATUSLOG', 7)
SMU_TABLE_SMU_METRICS = enum_smu_table_id.define('SMU_TABLE_SMU_METRICS', 8)
SMU_TABLE_DRIVER_SMU_CONFIG = enum_smu_table_id.define('SMU_TABLE_DRIVER_SMU_CONFIG', 9)
SMU_TABLE_ACTIVITY_MONITOR_COEFF = enum_smu_table_id.define('SMU_TABLE_ACTIVITY_MONITOR_COEFF', 10)
SMU_TABLE_OVERDRIVE = enum_smu_table_id.define('SMU_TABLE_OVERDRIVE', 11)
SMU_TABLE_I2C_COMMANDS = enum_smu_table_id.define('SMU_TABLE_I2C_COMMANDS', 12)
SMU_TABLE_PACE = enum_smu_table_id.define('SMU_TABLE_PACE', 13)
SMU_TABLE_ECCINFO = enum_smu_table_id.define('SMU_TABLE_ECCINFO', 14)
SMU_TABLE_COMBO_PPTABLE = enum_smu_table_id.define('SMU_TABLE_COMBO_PPTABLE', 15)
SMU_TABLE_WIFIBAND = enum_smu_table_id.define('SMU_TABLE_WIFIBAND', 16)
SMU_TABLE_COUNT = enum_smu_table_id.define('SMU_TABLE_COUNT', 17)

c.init_records()
FEATURE_CCLK_DPM_BIT = 0 # type: ignore
FEATURE_FAN_CONTROLLER_BIT = 1 # type: ignore
FEATURE_DATA_CALCULATION_BIT = 2 # type: ignore
FEATURE_PPT_BIT = 3 # type: ignore
FEATURE_TDC_BIT = 4 # type: ignore
FEATURE_THERMAL_BIT = 5 # type: ignore
FEATURE_FIT_BIT = 6 # type: ignore
FEATURE_EDC_BIT = 7 # type: ignore
FEATURE_PLL_POWER_DOWN_BIT = 8 # type: ignore
FEATURE_VDDOFF_BIT = 9 # type: ignore
FEATURE_VCN_DPM_BIT = 10 # type: ignore
FEATURE_DS_MPM_BIT = 11 # type: ignore
FEATURE_FCLK_DPM_BIT = 12 # type: ignore
FEATURE_SOCCLK_DPM_BIT = 13 # type: ignore
FEATURE_DS_MPIO_BIT = 14 # type: ignore
FEATURE_LCLK_DPM_BIT = 15 # type: ignore
FEATURE_SHUBCLK_DPM_BIT = 16 # type: ignore
FEATURE_DCFCLK_DPM_BIT = 17 # type: ignore
FEATURE_ISP_DPM_BIT = 18 # type: ignore
FEATURE_IPU_DPM_BIT = 19 # type: ignore
FEATURE_GFX_DPM_BIT = 20 # type: ignore
FEATURE_DS_GFXCLK_BIT = 21 # type: ignore
FEATURE_DS_SOCCLK_BIT = 22 # type: ignore
FEATURE_DS_LCLK_BIT = 23 # type: ignore
FEATURE_LOW_POWER_DCNCLKS_BIT = 24 # type: ignore
FEATURE_DS_SHUBCLK_BIT = 25 # type: ignore
FEATURE_RESERVED0_BIT = 26 # type: ignore
FEATURE_ZSTATES_BIT = 27 # type: ignore
FEATURE_IOMMUL2_PG_BIT = 28 # type: ignore
FEATURE_DS_FCLK_BIT = 29 # type: ignore
FEATURE_DS_SMNCLK_BIT = 30 # type: ignore
FEATURE_DS_MP1CLK_BIT = 31 # type: ignore
FEATURE_WHISPER_MODE_BIT = 32 # type: ignore
FEATURE_SMU_LOW_POWER_BIT = 33 # type: ignore
FEATURE_RESERVED1_BIT = 34 # type: ignore
FEATURE_GFX_DEM_BIT = 35 # type: ignore
FEATURE_PSI_BIT = 36 # type: ignore
FEATURE_PROCHOT_BIT = 37 # type: ignore
FEATURE_CPUOFF_BIT = 38 # type: ignore
FEATURE_STAPM_BIT = 39 # type: ignore
FEATURE_S0I3_BIT = 40 # type: ignore
FEATURE_DF_LIGHT_CSTATE = 41 # type: ignore
FEATURE_PERF_LIMIT_BIT = 42 # type: ignore
FEATURE_CORE_DLDO_BIT = 43 # type: ignore
FEATURE_DVO_BIT = 44 # type: ignore
FEATURE_DS_VCN_BIT = 45 # type: ignore
FEATURE_CPPC_BIT = 46 # type: ignore
FEATURE_CPPC_PREFERRED_CORES = 47 # type: ignore
FEATURE_DF_CSTATES_BIT = 48 # type: ignore
FEATURE_FAST_PSTATE_CLDO_BIT = 49 # type: ignore
FEATURE_ATHUB_PG_BIT = 50 # type: ignore
FEATURE_VDDOFF_ECO_BIT = 51 # type: ignore
FEATURE_ZSTATES_ECO_BIT = 52 # type: ignore
FEATURE_CC6_BIT = 53 # type: ignore
FEATURE_DS_UMCCLK_BIT = 54 # type: ignore
FEATURE_DS_ISPCLK_BIT = 55 # type: ignore
FEATURE_DS_HSPCLK_BIT = 56 # type: ignore
FEATURE_P3T_BIT = 57 # type: ignore
FEATURE_DS_IPUCLK_BIT = 58 # type: ignore
FEATURE_DS_VPECLK_BIT = 59 # type: ignore
FEATURE_VPE_DPM_BIT = 60 # type: ignore
FEATURE_SMART_L3_RINSER_BIT = 61 # type: ignore
FEATURE_PCC_BIT = 62 # type: ignore
NUM_FEATURES = 63 # type: ignore
PPSMC_VERSION = 0x1 # type: ignore
PPSMC_Result_OK = 0x1 # type: ignore
PPSMC_Result_Failed = 0xFF # type: ignore
PPSMC_Result_UnknownCmd = 0xFE # type: ignore
PPSMC_Result_CmdRejectedPrereq = 0xFD # type: ignore
PPSMC_Result_CmdRejectedBusy = 0xFC # type: ignore
PPSMC_MSG_TestMessage = 0x1 # type: ignore
PPSMC_MSG_GetSmuVersion = 0x2 # type: ignore
PPSMC_MSG_GetDriverIfVersion = 0x3 # type: ignore
PPSMC_MSG_SetAllowedFeaturesMaskLow = 0x4 # type: ignore
PPSMC_MSG_SetAllowedFeaturesMaskHigh = 0x5 # type: ignore
PPSMC_MSG_EnableAllSmuFeatures = 0x6 # type: ignore
PPSMC_MSG_DisableAllSmuFeatures = 0x7 # type: ignore
PPSMC_MSG_EnableSmuFeaturesLow = 0x8 # type: ignore
PPSMC_MSG_EnableSmuFeaturesHigh = 0x9 # type: ignore
PPSMC_MSG_DisableSmuFeaturesLow = 0xA # type: ignore
PPSMC_MSG_DisableSmuFeaturesHigh = 0xB # type: ignore
PPSMC_MSG_GetRunningSmuFeaturesLow = 0xC # type: ignore
PPSMC_MSG_GetRunningSmuFeaturesHigh = 0xD # type: ignore
PPSMC_MSG_SetDriverDramAddrHigh = 0xE # type: ignore
PPSMC_MSG_SetDriverDramAddrLow = 0xF # type: ignore
PPSMC_MSG_SetToolsDramAddrHigh = 0x10 # type: ignore
PPSMC_MSG_SetToolsDramAddrLow = 0x11 # type: ignore
PPSMC_MSG_TransferTableSmu2Dram = 0x12 # type: ignore
PPSMC_MSG_TransferTableDram2Smu = 0x13 # type: ignore
PPSMC_MSG_UseDefaultPPTable = 0x14 # type: ignore
PPSMC_MSG_EnterBaco = 0x15 # type: ignore
PPSMC_MSG_ExitBaco = 0x16 # type: ignore
PPSMC_MSG_ArmD3 = 0x17 # type: ignore
PPSMC_MSG_BacoAudioD3PME = 0x18 # type: ignore
PPSMC_MSG_SetSoftMinByFreq = 0x19 # type: ignore
PPSMC_MSG_SetSoftMaxByFreq = 0x1A # type: ignore
PPSMC_MSG_SetHardMinByFreq = 0x1B # type: ignore
PPSMC_MSG_SetHardMaxByFreq = 0x1C # type: ignore
PPSMC_MSG_GetMinDpmFreq = 0x1D # type: ignore
PPSMC_MSG_GetMaxDpmFreq = 0x1E # type: ignore
PPSMC_MSG_GetDpmFreqByIndex = 0x1F # type: ignore
PPSMC_MSG_OverridePcieParameters = 0x20 # type: ignore
PPSMC_MSG_DramLogSetDramAddrHigh = 0x21 # type: ignore
PPSMC_MSG_DramLogSetDramAddrLow = 0x22 # type: ignore
PPSMC_MSG_DramLogSetDramSize = 0x23 # type: ignore
PPSMC_MSG_SetWorkloadMask = 0x24 # type: ignore
PPSMC_MSG_GetVoltageByDpm = 0x25 # type: ignore
PPSMC_MSG_SetVideoFps = 0x26 # type: ignore
PPSMC_MSG_GetDcModeMaxDpmFreq = 0x27 # type: ignore
PPSMC_MSG_AllowGfxOff = 0x28 # type: ignore
PPSMC_MSG_DisallowGfxOff = 0x29 # type: ignore
PPSMC_MSG_PowerUpVcn = 0x2A # type: ignore
PPSMC_MSG_PowerDownVcn = 0x2B # type: ignore
PPSMC_MSG_PowerUpJpeg = 0x2C # type: ignore
PPSMC_MSG_PowerDownJpeg = 0x2D # type: ignore
PPSMC_MSG_PrepareMp1ForUnload = 0x2E # type: ignore
PPSMC_MSG_SetSystemVirtualDramAddrHigh = 0x30 # type: ignore
PPSMC_MSG_SetSystemVirtualDramAddrLow = 0x31 # type: ignore
PPSMC_MSG_SetPptLimit = 0x32 # type: ignore
PPSMC_MSG_GetPptLimit = 0x33 # type: ignore
PPSMC_MSG_ReenableAcDcInterrupt = 0x34 # type: ignore
PPSMC_MSG_NotifyPowerSource = 0x35 # type: ignore
PPSMC_MSG_RunDcBtc = 0x36 # type: ignore
PPSMC_MSG_SetTemperatureInputSelect = 0x38 # type: ignore
PPSMC_MSG_SetFwDstatesMask = 0x39 # type: ignore
PPSMC_MSG_SetThrottlerMask = 0x3A # type: ignore
PPSMC_MSG_SetExternalClientDfCstateAllow = 0x3B # type: ignore
PPSMC_MSG_SetMGpuFanBoostLimitRpm = 0x3C # type: ignore
PPSMC_MSG_DumpSTBtoDram = 0x3D # type: ignore
PPSMC_MSG_STBtoDramLogSetDramAddress = 0x3E # type: ignore
PPSMC_MSG_DummyUndefined = 0x3F # type: ignore
PPSMC_MSG_STBtoDramLogSetDramSize = 0x40 # type: ignore
PPSMC_MSG_SetOBMTraceBufferLogging = 0x41 # type: ignore
PPSMC_MSG_UseProfilingMode = 0x42 # type: ignore
PPSMC_MSG_AllowGfxDcs = 0x43 # type: ignore
PPSMC_MSG_DisallowGfxDcs = 0x44 # type: ignore
PPSMC_MSG_EnableAudioStutterWA = 0x45 # type: ignore
PPSMC_MSG_PowerUpUmsch = 0x46 # type: ignore
PPSMC_MSG_PowerDownUmsch = 0x47 # type: ignore
PPSMC_MSG_SetDcsArch = 0x48 # type: ignore
PPSMC_MSG_TriggerVFFLR = 0x49 # type: ignore
PPSMC_MSG_SetNumBadMemoryPagesRetired = 0x4A # type: ignore
PPSMC_MSG_SetBadMemoryPagesRetiredFlagsPerChannel = 0x4B # type: ignore
PPSMC_MSG_SetPriorityDeltaGain = 0x4C # type: ignore
PPSMC_MSG_AllowIHHostInterrupt = 0x4D # type: ignore
PPSMC_MSG_EnableShadowDpm = 0x4E # type: ignore
PPSMC_MSG_Mode3Reset = 0x4F # type: ignore
PPSMC_MSG_SetDriverDramAddr = 0x50 # type: ignore
PPSMC_MSG_SetToolsDramAddr = 0x51 # type: ignore
PPSMC_MSG_TransferTableSmu2DramWithAddr = 0x52 # type: ignore
PPSMC_MSG_TransferTableDram2SmuWithAddr = 0x53 # type: ignore
PPSMC_MSG_GetAllRunningSmuFeatures = 0x54 # type: ignore
PPSMC_MSG_GetSvi3Voltage = 0x55 # type: ignore
PPSMC_MSG_UpdatePolicy = 0x56 # type: ignore
PPSMC_MSG_ExtPwrConnSupport = 0x57 # type: ignore
PPSMC_MSG_PreloadSwPstateForUclkOverDrive = 0x58 # type: ignore
PPSMC_Message_Count = 0x59 # type: ignore
PPTABLE_VERSION = 0x1B # type: ignore
NUM_GFXCLK_DPM_LEVELS = 16 # type: ignore
NUM_SOCCLK_DPM_LEVELS = 8 # type: ignore
NUM_MP0CLK_DPM_LEVELS = 2 # type: ignore
NUM_DCLK_DPM_LEVELS = 8 # type: ignore
NUM_VCLK_DPM_LEVELS = 8 # type: ignore
NUM_DISPCLK_DPM_LEVELS = 8 # type: ignore
NUM_DPPCLK_DPM_LEVELS = 8 # type: ignore
NUM_DPREFCLK_DPM_LEVELS = 8 # type: ignore
NUM_DCFCLK_DPM_LEVELS = 8 # type: ignore
NUM_DTBCLK_DPM_LEVELS = 8 # type: ignore
NUM_UCLK_DPM_LEVELS = 6 # type: ignore
NUM_LINK_LEVELS = 3 # type: ignore
NUM_FCLK_DPM_LEVELS = 8 # type: ignore
NUM_OD_FAN_MAX_POINTS = 6 # type: ignore
FEATURE_FW_DATA_READ_BIT = 0 # type: ignore
FEATURE_DPM_GFXCLK_BIT = 1 # type: ignore
FEATURE_DPM_GFX_POWER_OPTIMIZER_BIT = 2 # type: ignore
FEATURE_DPM_UCLK_BIT = 3 # type: ignore
FEATURE_DPM_FCLK_BIT = 4 # type: ignore
FEATURE_DPM_SOCCLK_BIT = 5 # type: ignore
FEATURE_DPM_LINK_BIT = 6 # type: ignore
FEATURE_DPM_DCN_BIT = 7 # type: ignore
FEATURE_VMEMP_SCALING_BIT = 8 # type: ignore
FEATURE_VDDIO_MEM_SCALING_BIT = 9 # type: ignore
FEATURE_DS_GFXCLK_BIT = 10 # type: ignore
FEATURE_DS_SOCCLK_BIT = 11 # type: ignore
FEATURE_DS_FCLK_BIT = 12 # type: ignore
FEATURE_DS_LCLK_BIT = 13 # type: ignore
FEATURE_DS_DCFCLK_BIT = 14 # type: ignore
FEATURE_DS_UCLK_BIT = 15 # type: ignore
FEATURE_GFX_ULV_BIT = 16 # type: ignore
FEATURE_FW_DSTATE_BIT = 17 # type: ignore
FEATURE_GFXOFF_BIT = 18 # type: ignore
FEATURE_BACO_BIT = 19 # type: ignore
FEATURE_MM_DPM_BIT = 20 # type: ignore
FEATURE_SOC_MPCLK_DS_BIT = 21 # type: ignore
FEATURE_BACO_MPCLK_DS_BIT = 22 # type: ignore
FEATURE_THROTTLERS_BIT = 23 # type: ignore
FEATURE_SMARTSHIFT_BIT = 24 # type: ignore
FEATURE_GTHR_BIT = 25 # type: ignore
FEATURE_ACDC_BIT = 26 # type: ignore
FEATURE_VR0HOT_BIT = 27 # type: ignore
FEATURE_FW_CTF_BIT = 28 # type: ignore
FEATURE_FAN_CONTROL_BIT = 29 # type: ignore
FEATURE_GFX_DCS_BIT = 30 # type: ignore
FEATURE_GFX_READ_MARGIN_BIT = 31 # type: ignore
FEATURE_LED_DISPLAY_BIT = 32 # type: ignore
FEATURE_GFXCLK_SPREAD_SPECTRUM_BIT = 33 # type: ignore
FEATURE_OUT_OF_BAND_MONITOR_BIT = 34 # type: ignore
FEATURE_OPTIMIZED_VMIN_BIT = 35 # type: ignore
FEATURE_GFX_IMU_BIT = 36 # type: ignore
FEATURE_BOOT_TIME_CAL_BIT = 37 # type: ignore
FEATURE_GFX_PCC_DFLL_BIT = 38 # type: ignore
FEATURE_SOC_CG_BIT = 39 # type: ignore
FEATURE_DF_CSTATE_BIT = 40 # type: ignore
FEATURE_GFX_EDC_BIT = 41 # type: ignore
FEATURE_BOOT_POWER_OPT_BIT = 42 # type: ignore
FEATURE_CLOCK_POWER_DOWN_BYPASS_BIT = 43 # type: ignore
FEATURE_DS_VCN_BIT = 44 # type: ignore
FEATURE_BACO_CG_BIT = 45 # type: ignore
FEATURE_MEM_TEMP_READ_BIT = 46 # type: ignore
FEATURE_ATHUB_MMHUB_PG_BIT = 47 # type: ignore
FEATURE_SOC_PCC_BIT = 48 # type: ignore
FEATURE_EDC_PWRBRK_BIT = 49 # type: ignore
FEATURE_SOC_EDC_XVMIN_BIT = 50 # type: ignore
FEATURE_GFX_PSM_DIDT_BIT = 51 # type: ignore
FEATURE_APT_ALL_ENABLE_BIT = 52 # type: ignore
FEATURE_APT_SQ_THROTTLE_BIT = 53 # type: ignore
FEATURE_APT_PF_DCS_BIT = 54 # type: ignore
FEATURE_GFX_EDC_XVMIN_BIT = 55 # type: ignore
FEATURE_GFX_DIDT_XVMIN_BIT = 56 # type: ignore
FEATURE_FAN_ABNORMAL_BIT = 57 # type: ignore
FEATURE_CLOCK_STRETCH_COMPENSATOR = 58 # type: ignore
FEATURE_SPARE_59_BIT = 59 # type: ignore
FEATURE_SPARE_60_BIT = 60 # type: ignore
FEATURE_SPARE_61_BIT = 61 # type: ignore
FEATURE_SPARE_62_BIT = 62 # type: ignore
FEATURE_SPARE_63_BIT = 63 # type: ignore
NUM_FEATURES = 64 # type: ignore
ALLOWED_FEATURE_CTRL_DEFAULT = 0xFFFFFFFFFFFFFFFF # type: ignore
ALLOWED_FEATURE_CTRL_SCPM = (1 << FEATURE_DPM_GFXCLK_BIT) | (1 << FEATURE_DPM_GFX_POWER_OPTIMIZER_BIT) | (1 << FEATURE_DPM_UCLK_BIT) | (1 << FEATURE_DPM_FCLK_BIT) | (1 << FEATURE_DPM_SOCCLK_BIT) | (1 << FEATURE_DPM_LINK_BIT) | (1 << FEATURE_DPM_DCN_BIT) | (1 << FEATURE_DS_GFXCLK_BIT) | (1 << FEATURE_DS_SOCCLK_BIT) | (1 << FEATURE_DS_FCLK_BIT) | (1 << FEATURE_DS_LCLK_BIT) | (1 << FEATURE_DS_DCFCLK_BIT) | (1 << FEATURE_DS_UCLK_BIT) | (1 << FEATURE_DS_VCN_BIT) # type: ignore
DEBUG_OVERRIDE_NOT_USE = 0x00000001 # type: ignore
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_DCN_FCLK = 0x00000002 # type: ignore
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_MP0_FCLK = 0x00000004 # type: ignore
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_VCN_DCFCLK = 0x00000008 # type: ignore
DEBUG_OVERRIDE_DISABLE_FAST_FCLK_TIMER = 0x00000010 # type: ignore
DEBUG_OVERRIDE_DISABLE_VCN_PG = 0x00000020 # type: ignore
DEBUG_OVERRIDE_DISABLE_FMAX_VMAX = 0x00000040 # type: ignore
DEBUG_OVERRIDE_DISABLE_IMU_FW_CHECKS = 0x00000080 # type: ignore
DEBUG_OVERRIDE_DISABLE_D0i2_REENTRY_HSR_TIMER_CHECK = 0x00000100 # type: ignore
DEBUG_OVERRIDE_DISABLE_DFLL = 0x00000200 # type: ignore
DEBUG_OVERRIDE_ENABLE_RLC_VF_BRINGUP_MODE = 0x00000400 # type: ignore
DEBUG_OVERRIDE_DFLL_MASTER_MODE = 0x00000800 # type: ignore
DEBUG_OVERRIDE_ENABLE_PROFILING_MODE = 0x00001000 # type: ignore
DEBUG_OVERRIDE_ENABLE_SOC_VF_BRINGUP_MODE = 0x00002000 # type: ignore
DEBUG_OVERRIDE_ENABLE_PER_WGP_RESIENCY = 0x00004000 # type: ignore
DEBUG_OVERRIDE_DISABLE_MEMORY_VOLTAGE_SCALING = 0x00008000 # type: ignore
DEBUG_OVERRIDE_DFLL_BTC_FCW_LOG = 0x00010000 # type: ignore
VR_MAPPING_VR_SELECT_MASK = 0x01 # type: ignore
VR_MAPPING_VR_SELECT_SHIFT = 0x00 # type: ignore
VR_MAPPING_PLANE_SELECT_MASK = 0x02 # type: ignore
VR_MAPPING_PLANE_SELECT_SHIFT = 0x01 # type: ignore
PSI_SEL_VR0_PLANE0_PSI0 = 0x01 # type: ignore
PSI_SEL_VR0_PLANE0_PSI1 = 0x02 # type: ignore
PSI_SEL_VR0_PLANE1_PSI0 = 0x04 # type: ignore
PSI_SEL_VR0_PLANE1_PSI1 = 0x08 # type: ignore
PSI_SEL_VR1_PLANE0_PSI0 = 0x10 # type: ignore
PSI_SEL_VR1_PLANE0_PSI1 = 0x20 # type: ignore
PSI_SEL_VR1_PLANE1_PSI0 = 0x40 # type: ignore
PSI_SEL_VR1_PLANE1_PSI1 = 0x80 # type: ignore
THROTTLER_TEMP_EDGE_BIT = 0 # type: ignore
THROTTLER_TEMP_HOTSPOT_BIT = 1 # type: ignore
THROTTLER_TEMP_HOTSPOT_GFX_BIT = 2 # type: ignore
THROTTLER_TEMP_HOTSPOT_SOC_BIT = 3 # type: ignore
THROTTLER_TEMP_MEM_BIT = 4 # type: ignore
THROTTLER_TEMP_VR_GFX_BIT = 5 # type: ignore
THROTTLER_TEMP_VR_SOC_BIT = 6 # type: ignore
THROTTLER_TEMP_VR_MEM0_BIT = 7 # type: ignore
THROTTLER_TEMP_VR_MEM1_BIT = 8 # type: ignore
THROTTLER_TEMP_LIQUID0_BIT = 9 # type: ignore
THROTTLER_TEMP_LIQUID1_BIT = 10 # type: ignore
THROTTLER_TEMP_PLX_BIT = 11 # type: ignore
THROTTLER_TDC_GFX_BIT = 12 # type: ignore
THROTTLER_TDC_SOC_BIT = 13 # type: ignore
THROTTLER_PPT0_BIT = 14 # type: ignore
THROTTLER_PPT1_BIT = 15 # type: ignore
THROTTLER_PPT2_BIT = 16 # type: ignore
THROTTLER_PPT3_BIT = 17 # type: ignore
THROTTLER_FIT_BIT = 18 # type: ignore
THROTTLER_GFX_APCC_PLUS_BIT = 19 # type: ignore
THROTTLER_GFX_DVO_BIT = 20 # type: ignore
THROTTLER_COUNT = 21 # type: ignore
FW_DSTATE_SOC_ULV_BIT = 0 # type: ignore
FW_DSTATE_G6_HSR_BIT = 1 # type: ignore
FW_DSTATE_G6_PHY_VMEMP_OFF_BIT = 2 # type: ignore
FW_DSTATE_SMN_DS_BIT = 3 # type: ignore
FW_DSTATE_MP1_WHISPER_MODE_BIT = 4 # type: ignore
FW_DSTATE_SOC_LIV_MIN_BIT = 5 # type: ignore
FW_DSTATE_SOC_PLL_PWRDN_BIT = 6 # type: ignore
FW_DSTATE_MEM_PLL_PWRDN_BIT = 7 # type: ignore
FW_DSTATE_MALL_ALLOC_BIT = 8 # type: ignore
FW_DSTATE_MEM_PSI_BIT = 9 # type: ignore
FW_DSTATE_HSR_NON_STROBE_BIT = 10 # type: ignore
FW_DSTATE_MP0_ENTER_WFI_BIT = 11 # type: ignore
FW_DSTATE_MALL_FLUSH_BIT = 12 # type: ignore
FW_DSTATE_SOC_PSI_BIT = 13 # type: ignore
FW_DSTATE_MMHUB_INTERLOCK_BIT = 14 # type: ignore
FW_DSTATE_D0i3_2_QUIET_FW_BIT = 15 # type: ignore
FW_DSTATE_CLDO_PRG_BIT = 16 # type: ignore
FW_DSTATE_DF_PLL_PWRDN_BIT = 17 # type: ignore
LED_DISPLAY_GFX_DPM_BIT = 0 # type: ignore
LED_DISPLAY_PCIE_BIT = 1 # type: ignore
LED_DISPLAY_ERROR_BIT = 2 # type: ignore
MEM_TEMP_READ_OUT_OF_BAND_BIT = 0 # type: ignore
MEM_TEMP_READ_IN_BAND_REFRESH_BIT = 1 # type: ignore
MEM_TEMP_READ_IN_BAND_DUMMY_PSTATE_BIT = 2 # type: ignore
NUM_I2C_CONTROLLERS = 8 # type: ignore
I2C_CONTROLLER_ENABLED = 1 # type: ignore
I2C_CONTROLLER_DISABLED = 0 # type: ignore
MAX_SW_I2C_COMMANDS = 24 # type: ignore
CMDCONFIG_STOP_BIT = 0 # type: ignore
CMDCONFIG_RESTART_BIT = 1 # type: ignore
CMDCONFIG_READWRITE_BIT = 2 # type: ignore
CMDCONFIG_STOP_MASK = (1 << CMDCONFIG_STOP_BIT) # type: ignore
CMDCONFIG_RESTART_MASK = (1 << CMDCONFIG_RESTART_BIT) # type: ignore
CMDCONFIG_READWRITE_MASK = (1 << CMDCONFIG_READWRITE_BIT) # type: ignore
EPCS_HIGH_POWER = 600 # type: ignore
EPCS_NORMAL_POWER = 450 # type: ignore
EPCS_LOW_POWER = 300 # type: ignore
EPCS_SHORTED_POWER = 150 # type: ignore
EPCS_NO_BOOTUP = 0 # type: ignore
PP_NUM_RTAVFS_PWL_ZONES = 5 # type: ignore
PP_NUM_PSM_DIDT_PWL_ZONES = 3 # type: ignore
PP_NUM_OD_VF_CURVE_POINTS = PP_NUM_RTAVFS_PWL_ZONES + 1 # type: ignore
PP_OD_FEATURE_GFX_VF_CURVE_BIT = 0 # type: ignore
PP_OD_FEATURE_GFX_VMAX_BIT = 1 # type: ignore
PP_OD_FEATURE_SOC_VMAX_BIT = 2 # type: ignore
PP_OD_FEATURE_PPT_BIT = 3 # type: ignore
PP_OD_FEATURE_FAN_CURVE_BIT = 4 # type: ignore
PP_OD_FEATURE_FAN_LEGACY_BIT = 5 # type: ignore
PP_OD_FEATURE_FULL_CTRL_BIT = 6 # type: ignore
PP_OD_FEATURE_TDC_BIT = 7 # type: ignore
PP_OD_FEATURE_GFXCLK_BIT = 8 # type: ignore
PP_OD_FEATURE_UCLK_BIT = 9 # type: ignore
PP_OD_FEATURE_FCLK_BIT = 10 # type: ignore
PP_OD_FEATURE_ZERO_FAN_BIT = 11 # type: ignore
PP_OD_FEATURE_TEMPERATURE_BIT = 12 # type: ignore
PP_OD_FEATURE_EDC_BIT = 13 # type: ignore
PP_OD_FEATURE_COUNT = 14 # type: ignore
INVALID_BOARD_GPIO = 0xFF # type: ignore
NUM_WM_RANGES = 4 # type: ignore
WORKLOAD_PPLIB_DEFAULT_BIT = 0 # type: ignore
WORKLOAD_PPLIB_FULL_SCREEN_3D_BIT = 1 # type: ignore
WORKLOAD_PPLIB_POWER_SAVING_BIT = 2 # type: ignore
WORKLOAD_PPLIB_VIDEO_BIT = 3 # type: ignore
WORKLOAD_PPLIB_VR_BIT = 4 # type: ignore
WORKLOAD_PPLIB_COMPUTE_BIT = 5 # type: ignore
WORKLOAD_PPLIB_CUSTOM_BIT = 6 # type: ignore
WORKLOAD_PPLIB_WINDOW_3D_BIT = 7 # type: ignore
WORKLOAD_PPLIB_DIRECT_ML_BIT = 8 # type: ignore
WORKLOAD_PPLIB_CGVDI_BIT = 9 # type: ignore
WORKLOAD_PPLIB_COUNT = 10 # type: ignore
TABLE_TRANSFER_OK = 0x0 # type: ignore
TABLE_TRANSFER_FAILED = 0xFF # type: ignore
TABLE_TRANSFER_PENDING = 0xAB # type: ignore
TABLE_PPT_FAILED = 0x100 # type: ignore
TABLE_TDC_FAILED = 0x200 # type: ignore
TABLE_TEMP_FAILED = 0x400 # type: ignore
TABLE_FAN_TARGET_TEMP_FAILED = 0x800 # type: ignore
TABLE_FAN_STOP_TEMP_FAILED = 0x1000 # type: ignore
TABLE_FAN_START_TEMP_FAILED = 0x2000 # type: ignore
TABLE_FAN_PWM_MIN_FAILED = 0x4000 # type: ignore
TABLE_ACOUSTIC_TARGET_RPM_FAILED = 0x8000 # type: ignore
TABLE_ACOUSTIC_LIMIT_RPM_FAILED = 0x10000 # type: ignore
TABLE_MGPU_ACOUSTIC_TARGET_RPM_FAILED = 0x20000 # type: ignore
TABLE_PPTABLE = 0 # type: ignore
TABLE_COMBO_PPTABLE = 1 # type: ignore
TABLE_WATERMARKS = 2 # type: ignore
TABLE_AVFS_PSM_DEBUG = 3 # type: ignore
TABLE_PMSTATUSLOG = 4 # type: ignore
TABLE_SMU_METRICS = 5 # type: ignore
TABLE_DRIVER_SMU_CONFIG = 6 # type: ignore
TABLE_ACTIVITY_MONITOR_COEFF = 7 # type: ignore
TABLE_OVERDRIVE = 8 # type: ignore
TABLE_I2C_COMMANDS = 9 # type: ignore
TABLE_DRIVER_INFO = 10 # type: ignore
TABLE_ECCINFO = 11 # type: ignore
TABLE_CUSTOM_SKUTABLE = 12 # type: ignore
TABLE_COUNT = 13 # type: ignore
IH_INTERRUPT_ID_TO_DRIVER = 0xFE # type: ignore
IH_INTERRUPT_CONTEXT_ID_BACO = 0x2 # type: ignore
IH_INTERRUPT_CONTEXT_ID_AC = 0x3 # type: ignore
IH_INTERRUPT_CONTEXT_ID_DC = 0x4 # type: ignore
IH_INTERRUPT_CONTEXT_ID_AUDIO_D0 = 0x5 # type: ignore
IH_INTERRUPT_CONTEXT_ID_AUDIO_D3 = 0x6 # type: ignore
IH_INTERRUPT_CONTEXT_ID_THERMAL_THROTTLING = 0x7 # type: ignore
IH_INTERRUPT_CONTEXT_ID_FAN_ABNORMAL = 0x8 # type: ignore
IH_INTERRUPT_CONTEXT_ID_FAN_RECOVERY = 0x9 # type: ignore
IH_INTERRUPT_CONTEXT_ID_DYNAMIC_TABLE = 0xA # type: ignore
int32_t = int # type: ignore
SMU_THERMAL_MINIMUM_ALERT_TEMP = 0 # type: ignore
SMU_THERMAL_MAXIMUM_ALERT_TEMP = 255 # type: ignore
SMU_TEMPERATURE_UNITS_PER_CENTIGRADES = 1000 # type: ignore
SMU_FW_NAME_LEN = 0x24 # type: ignore
SMU_DPM_USER_PROFILE_RESTORE = (1 << 0) # type: ignore
SMU_CUSTOM_FAN_SPEED_RPM = (1 << 1) # type: ignore
SMU_CUSTOM_FAN_SPEED_PWM = (1 << 2) # type: ignore
SMU_THROTTLER_PPT0_BIT = 0 # type: ignore
SMU_THROTTLER_PPT1_BIT = 1 # type: ignore
SMU_THROTTLER_PPT2_BIT = 2 # type: ignore
SMU_THROTTLER_PPT3_BIT = 3 # type: ignore
SMU_THROTTLER_SPL_BIT = 4 # type: ignore
SMU_THROTTLER_FPPT_BIT = 5 # type: ignore
SMU_THROTTLER_SPPT_BIT = 6 # type: ignore
SMU_THROTTLER_SPPT_APU_BIT = 7 # type: ignore
SMU_THROTTLER_TDC_GFX_BIT = 16 # type: ignore
SMU_THROTTLER_TDC_SOC_BIT = 17 # type: ignore
SMU_THROTTLER_TDC_MEM_BIT = 18 # type: ignore
SMU_THROTTLER_TDC_VDD_BIT = 19 # type: ignore
SMU_THROTTLER_TDC_CVIP_BIT = 20 # type: ignore
SMU_THROTTLER_EDC_CPU_BIT = 21 # type: ignore
SMU_THROTTLER_EDC_GFX_BIT = 22 # type: ignore
SMU_THROTTLER_APCC_BIT = 23 # type: ignore
SMU_THROTTLER_TEMP_GPU_BIT = 32 # type: ignore
SMU_THROTTLER_TEMP_CORE_BIT = 33 # type: ignore
SMU_THROTTLER_TEMP_MEM_BIT = 34 # type: ignore
SMU_THROTTLER_TEMP_EDGE_BIT = 35 # type: ignore
SMU_THROTTLER_TEMP_HOTSPOT_BIT = 36 # type: ignore
SMU_THROTTLER_TEMP_SOC_BIT = 37 # type: ignore
SMU_THROTTLER_TEMP_VR_GFX_BIT = 38 # type: ignore
SMU_THROTTLER_TEMP_VR_SOC_BIT = 39 # type: ignore
SMU_THROTTLER_TEMP_VR_MEM0_BIT = 40 # type: ignore
SMU_THROTTLER_TEMP_VR_MEM1_BIT = 41 # type: ignore
SMU_THROTTLER_TEMP_LIQUID0_BIT = 42 # type: ignore
SMU_THROTTLER_TEMP_LIQUID1_BIT = 43 # type: ignore
SMU_THROTTLER_VRHOT0_BIT = 44 # type: ignore
SMU_THROTTLER_VRHOT1_BIT = 45 # type: ignore
SMU_THROTTLER_PROCHOT_CPU_BIT = 46 # type: ignore
SMU_THROTTLER_PROCHOT_GFX_BIT = 47 # type: ignore
SMU_THROTTLER_PPM_BIT = 56 # type: ignore
SMU_THROTTLER_FIT_BIT = 57 # type: ignore