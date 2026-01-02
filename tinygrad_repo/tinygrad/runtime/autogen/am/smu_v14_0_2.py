# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
class struct_SMU14_Firmware_Footer(Struct): pass
uint32_t = ctypes.c_uint32
struct_SMU14_Firmware_Footer._packed_ = True
struct_SMU14_Firmware_Footer._fields_ = [
  ('Signature', uint32_t),
]
SMU14_Firmware_Footer = struct_SMU14_Firmware_Footer
class SMU_Firmware_Header(Struct): pass
SMU_Firmware_Header._packed_ = True
SMU_Firmware_Header._fields_ = [
  ('ImageVersion', uint32_t),
  ('ImageVersion2', uint32_t),
  ('Padding0', (uint32_t * 3)),
  ('SizeFWSigned', uint32_t),
  ('Padding1', (uint32_t * 25)),
  ('FirmwareType', uint32_t),
  ('Filler', (uint32_t * 32)),
]
class FwStatus_t(Struct): pass
FwStatus_t._packed_ = True
FwStatus_t._fields_ = [
  ('DpmHandlerID', uint32_t,8),
  ('ActivityMonitorID', uint32_t,8),
  ('DpmTimerID', uint32_t,8),
  ('DpmHubID', uint32_t,4),
  ('DpmHubTask', uint32_t,4),
  ('CclkSyncStatus', uint32_t,8),
  ('Ccx0CpuOff', uint32_t,2),
  ('Ccx1CpuOff', uint32_t,2),
  ('GfxOffStatus', uint32_t,2),
  ('VddOff', uint32_t,1),
  ('InWhisperMode', uint32_t,1),
  ('ZstateStatus', uint32_t,4),
  ('spare0', uint32_t,4),
  ('DstateFun', uint32_t,4),
  ('DstateDev', uint32_t,4),
  ('P2JobHandler', uint32_t,24),
  ('RsmuPmiP2PendingCnt', uint32_t,8),
  ('PostCode', uint32_t,32),
  ('MsgPortBusy', uint32_t,24),
  ('RsmuPmiP1Pending', uint32_t,1),
  ('DfCstateExitPending', uint32_t,1),
  ('Ccx0Pc6ExitPending', uint32_t,1),
  ('Ccx1Pc6ExitPending', uint32_t,1),
  ('WarmResetPending', uint32_t,1),
  ('spare1', uint32_t,3),
  ('IdleMask', uint32_t,32),
]
class FwStatus_t_v14_0_1(Struct): pass
FwStatus_t_v14_0_1._packed_ = True
FwStatus_t_v14_0_1._fields_ = [
  ('DpmHandlerID', uint32_t,8),
  ('ActivityMonitorID', uint32_t,8),
  ('DpmTimerID', uint32_t,8),
  ('DpmHubID', uint32_t,4),
  ('DpmHubTask', uint32_t,4),
  ('CclkSyncStatus', uint32_t,8),
  ('ZstateStatus', uint32_t,4),
  ('Cpu1VddOff', uint32_t,4),
  ('DstateFun', uint32_t,4),
  ('DstateDev', uint32_t,4),
  ('GfxOffStatus', uint32_t,2),
  ('Cpu0Off', uint32_t,2),
  ('Cpu1Off', uint32_t,2),
  ('Cpu0VddOff', uint32_t,2),
  ('P2JobHandler', uint32_t,32),
  ('PostCode', uint32_t,32),
  ('MsgPortBusy', uint32_t,15),
  ('RsmuPmiP1Pending', uint32_t,1),
  ('RsmuPmiP2PendingCnt', uint32_t,8),
  ('DfCstateExitPending', uint32_t,1),
  ('Pc6EntryPending', uint32_t,1),
  ('Pc6ExitPending', uint32_t,1),
  ('WarmResetPending', uint32_t,1),
  ('Mp0ClkPending', uint32_t,1),
  ('InWhisperMode', uint32_t,1),
  ('spare2', uint32_t,2),
  ('IdleMask', uint32_t,32),
]
FEATURE_PWR_DOMAIN_e = CEnum(ctypes.c_uint32)
FEATURE_PWR_ALL = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_ALL', 0)
FEATURE_PWR_S5 = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_S5', 1)
FEATURE_PWR_BACO = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_BACO', 2)
FEATURE_PWR_SOC = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_SOC', 3)
FEATURE_PWR_GFX = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_GFX', 4)
FEATURE_PWR_DOMAIN_COUNT = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_DOMAIN_COUNT', 5)

FEATURE_BTC_e = CEnum(ctypes.c_uint32)
FEATURE_BTC_NOP = FEATURE_BTC_e.define('FEATURE_BTC_NOP', 0)
FEATURE_BTC_SAVE = FEATURE_BTC_e.define('FEATURE_BTC_SAVE', 1)
FEATURE_BTC_RESTORE = FEATURE_BTC_e.define('FEATURE_BTC_RESTORE', 2)
FEATURE_BTC_COUNT = FEATURE_BTC_e.define('FEATURE_BTC_COUNT', 3)

SVI_PSI_e = CEnum(ctypes.c_uint32)
SVI_PSI_0 = SVI_PSI_e.define('SVI_PSI_0', 0)
SVI_PSI_1 = SVI_PSI_e.define('SVI_PSI_1', 1)
SVI_PSI_2 = SVI_PSI_e.define('SVI_PSI_2', 2)
SVI_PSI_3 = SVI_PSI_e.define('SVI_PSI_3', 3)
SVI_PSI_4 = SVI_PSI_e.define('SVI_PSI_4', 4)
SVI_PSI_5 = SVI_PSI_e.define('SVI_PSI_5', 5)
SVI_PSI_6 = SVI_PSI_e.define('SVI_PSI_6', 6)
SVI_PSI_7 = SVI_PSI_e.define('SVI_PSI_7', 7)

SMARTSHIFT_VERSION_e = CEnum(ctypes.c_uint32)
SMARTSHIFT_VERSION_1 = SMARTSHIFT_VERSION_e.define('SMARTSHIFT_VERSION_1', 0)
SMARTSHIFT_VERSION_2 = SMARTSHIFT_VERSION_e.define('SMARTSHIFT_VERSION_2', 1)
SMARTSHIFT_VERSION_3 = SMARTSHIFT_VERSION_e.define('SMARTSHIFT_VERSION_3', 2)

FOPT_CALC_e = CEnum(ctypes.c_uint32)
FOPT_CALC_AC_CALC_DC = FOPT_CALC_e.define('FOPT_CALC_AC_CALC_DC', 0)
FOPT_PPTABLE_AC_CALC_DC = FOPT_CALC_e.define('FOPT_PPTABLE_AC_CALC_DC', 1)
FOPT_CALC_AC_PPTABLE_DC = FOPT_CALC_e.define('FOPT_CALC_AC_PPTABLE_DC', 2)
FOPT_PPTABLE_AC_PPTABLE_DC = FOPT_CALC_e.define('FOPT_PPTABLE_AC_PPTABLE_DC', 3)

DRAM_BIT_WIDTH_TYPE_e = CEnum(ctypes.c_uint32)
DRAM_BIT_WIDTH_DISABLED = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_DISABLED', 0)
DRAM_BIT_WIDTH_X_8 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_8', 8)
DRAM_BIT_WIDTH_X_16 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_16', 16)
DRAM_BIT_WIDTH_X_32 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_32', 32)
DRAM_BIT_WIDTH_X_64 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_64', 64)
DRAM_BIT_WIDTH_X_128 = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_X_128', 128)
DRAM_BIT_WIDTH_COUNT = DRAM_BIT_WIDTH_TYPE_e.define('DRAM_BIT_WIDTH_COUNT', 129)

I2cControllerPort_e = CEnum(ctypes.c_uint32)
I2C_CONTROLLER_PORT_0 = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_0', 0)
I2C_CONTROLLER_PORT_1 = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_1', 1)
I2C_CONTROLLER_PORT_COUNT = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_COUNT', 2)

I2cControllerName_e = CEnum(ctypes.c_uint32)
I2C_CONTROLLER_NAME_VR_GFX = I2cControllerName_e.define('I2C_CONTROLLER_NAME_VR_GFX', 0)
I2C_CONTROLLER_NAME_VR_SOC = I2cControllerName_e.define('I2C_CONTROLLER_NAME_VR_SOC', 1)
I2C_CONTROLLER_NAME_VR_VMEMP = I2cControllerName_e.define('I2C_CONTROLLER_NAME_VR_VMEMP', 2)
I2C_CONTROLLER_NAME_VR_VDDIO = I2cControllerName_e.define('I2C_CONTROLLER_NAME_VR_VDDIO', 3)
I2C_CONTROLLER_NAME_LIQUID0 = I2cControllerName_e.define('I2C_CONTROLLER_NAME_LIQUID0', 4)
I2C_CONTROLLER_NAME_LIQUID1 = I2cControllerName_e.define('I2C_CONTROLLER_NAME_LIQUID1', 5)
I2C_CONTROLLER_NAME_PLX = I2cControllerName_e.define('I2C_CONTROLLER_NAME_PLX', 6)
I2C_CONTROLLER_NAME_FAN_INTAKE = I2cControllerName_e.define('I2C_CONTROLLER_NAME_FAN_INTAKE', 7)
I2C_CONTROLLER_NAME_COUNT = I2cControllerName_e.define('I2C_CONTROLLER_NAME_COUNT', 8)

I2cControllerThrottler_e = CEnum(ctypes.c_uint32)
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

I2cControllerProtocol_e = CEnum(ctypes.c_uint32)
I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5', 0)
I2C_CONTROLLER_PROTOCOL_VR_IR35217 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_VR_IR35217', 1)
I2C_CONTROLLER_PROTOCOL_TMP_MAX31875 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_TMP_MAX31875', 2)
I2C_CONTROLLER_PROTOCOL_INA3221 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_INA3221', 3)
I2C_CONTROLLER_PROTOCOL_TMP_MAX6604 = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_TMP_MAX6604', 4)
I2C_CONTROLLER_PROTOCOL_COUNT = I2cControllerProtocol_e.define('I2C_CONTROLLER_PROTOCOL_COUNT', 5)

class I2cControllerConfig_t(Struct): pass
uint8_t = ctypes.c_ubyte
I2cControllerConfig_t._fields_ = [
  ('Enabled', uint8_t),
  ('Speed', uint8_t),
  ('SlaveAddress', uint8_t),
  ('ControllerPort', uint8_t),
  ('ControllerName', uint8_t),
  ('ThermalThrotter', uint8_t),
  ('I2cProtocol', uint8_t),
  ('PaddingConfig', uint8_t),
]
I2cPort_e = CEnum(ctypes.c_uint32)
I2C_PORT_SVD_SCL = I2cPort_e.define('I2C_PORT_SVD_SCL', 0)
I2C_PORT_GPIO = I2cPort_e.define('I2C_PORT_GPIO', 1)

I2cSpeed_e = CEnum(ctypes.c_uint32)
I2C_SPEED_FAST_50K = I2cSpeed_e.define('I2C_SPEED_FAST_50K', 0)
I2C_SPEED_FAST_100K = I2cSpeed_e.define('I2C_SPEED_FAST_100K', 1)
I2C_SPEED_FAST_400K = I2cSpeed_e.define('I2C_SPEED_FAST_400K', 2)
I2C_SPEED_FAST_PLUS_1M = I2cSpeed_e.define('I2C_SPEED_FAST_PLUS_1M', 3)
I2C_SPEED_HIGH_1M = I2cSpeed_e.define('I2C_SPEED_HIGH_1M', 4)
I2C_SPEED_HIGH_2M = I2cSpeed_e.define('I2C_SPEED_HIGH_2M', 5)
I2C_SPEED_COUNT = I2cSpeed_e.define('I2C_SPEED_COUNT', 6)

I2cCmdType_e = CEnum(ctypes.c_uint32)
I2C_CMD_READ = I2cCmdType_e.define('I2C_CMD_READ', 0)
I2C_CMD_WRITE = I2cCmdType_e.define('I2C_CMD_WRITE', 1)
I2C_CMD_COUNT = I2cCmdType_e.define('I2C_CMD_COUNT', 2)

class SwI2cCmd_t(Struct): pass
SwI2cCmd_t._fields_ = [
  ('ReadWriteData', uint8_t),
  ('CmdConfig', uint8_t),
]
class SwI2cRequest_t(Struct): pass
SwI2cRequest_t._fields_ = [
  ('I2CcontrollerPort', uint8_t),
  ('I2CSpeed', uint8_t),
  ('SlaveAddress', uint8_t),
  ('NumCmds', uint8_t),
  ('SwI2cCmds', (SwI2cCmd_t * 24)),
]
class SwI2cRequestExternal_t(Struct): pass
SwI2cRequestExternal_t._fields_ = [
  ('SwI2cRequest', SwI2cRequest_t),
  ('Spare', (uint32_t * 8)),
  ('MmHubPadding', (uint32_t * 8)),
]
class EccInfo_t(Struct): pass
uint64_t = ctypes.c_uint64
uint16_t = ctypes.c_uint16
EccInfo_t._fields_ = [
  ('mca_umc_status', uint64_t),
  ('mca_umc_addr', uint64_t),
  ('ce_count_lo_chip', uint16_t),
  ('ce_count_hi_chip', uint16_t),
  ('eccPadding', uint32_t),
]
class EccInfoTable_t(Struct): pass
EccInfoTable_t._fields_ = [
  ('EccInfo', (EccInfo_t * 24)),
]
EPCS_STATUS_e = CEnum(ctypes.c_uint32)
EPCS_SHORTED_LIMIT = EPCS_STATUS_e.define('EPCS_SHORTED_LIMIT', 0)
EPCS_LOW_POWER_LIMIT = EPCS_STATUS_e.define('EPCS_LOW_POWER_LIMIT', 1)
EPCS_NORMAL_POWER_LIMIT = EPCS_STATUS_e.define('EPCS_NORMAL_POWER_LIMIT', 2)
EPCS_HIGH_POWER_LIMIT = EPCS_STATUS_e.define('EPCS_HIGH_POWER_LIMIT', 3)
EPCS_NOT_CONFIGURED = EPCS_STATUS_e.define('EPCS_NOT_CONFIGURED', 4)
EPCS_STATUS_COUNT = EPCS_STATUS_e.define('EPCS_STATUS_COUNT', 5)

D3HOTSequence_e = CEnum(ctypes.c_uint32)
BACO_SEQUENCE = D3HOTSequence_e.define('BACO_SEQUENCE', 0)
MSR_SEQUENCE = D3HOTSequence_e.define('MSR_SEQUENCE', 1)
BAMACO_SEQUENCE = D3HOTSequence_e.define('BAMACO_SEQUENCE', 2)
ULPS_SEQUENCE = D3HOTSequence_e.define('ULPS_SEQUENCE', 3)
D3HOT_SEQUENCE_COUNT = D3HOTSequence_e.define('D3HOT_SEQUENCE_COUNT', 4)

PowerGatingMode_e = CEnum(ctypes.c_uint32)
PG_DYNAMIC_MODE = PowerGatingMode_e.define('PG_DYNAMIC_MODE', 0)
PG_STATIC_MODE = PowerGatingMode_e.define('PG_STATIC_MODE', 1)

PowerGatingSettings_e = CEnum(ctypes.c_uint32)
PG_POWER_DOWN = PowerGatingSettings_e.define('PG_POWER_DOWN', 0)
PG_POWER_UP = PowerGatingSettings_e.define('PG_POWER_UP', 1)

class QuadraticInt_t(Struct): pass
QuadraticInt_t._fields_ = [
  ('a', uint32_t),
  ('b', uint32_t),
  ('c', uint32_t),
]
class LinearInt_t(Struct): pass
LinearInt_t._fields_ = [
  ('m', uint32_t),
  ('b', uint32_t),
]
class DroopInt_t(Struct): pass
DroopInt_t._fields_ = [
  ('a', uint32_t),
  ('b', uint32_t),
  ('c', uint32_t),
]
DCS_ARCH_e = CEnum(ctypes.c_uint32)
DCS_ARCH_DISABLED = DCS_ARCH_e.define('DCS_ARCH_DISABLED', 0)
DCS_ARCH_FADCS = DCS_ARCH_e.define('DCS_ARCH_FADCS', 1)
DCS_ARCH_ASYNC = DCS_ARCH_e.define('DCS_ARCH_ASYNC', 2)

PPCLK_e = CEnum(ctypes.c_uint32)
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

VOLTAGE_MODE_e = CEnum(ctypes.c_uint32)
VOLTAGE_MODE_PPTABLE = VOLTAGE_MODE_e.define('VOLTAGE_MODE_PPTABLE', 0)
VOLTAGE_MODE_FUSES = VOLTAGE_MODE_e.define('VOLTAGE_MODE_FUSES', 1)
VOLTAGE_MODE_COUNT = VOLTAGE_MODE_e.define('VOLTAGE_MODE_COUNT', 2)

AVFS_VOLTAGE_TYPE_e = CEnum(ctypes.c_uint32)
AVFS_VOLTAGE_GFX = AVFS_VOLTAGE_TYPE_e.define('AVFS_VOLTAGE_GFX', 0)
AVFS_VOLTAGE_SOC = AVFS_VOLTAGE_TYPE_e.define('AVFS_VOLTAGE_SOC', 1)
AVFS_VOLTAGE_COUNT = AVFS_VOLTAGE_TYPE_e.define('AVFS_VOLTAGE_COUNT', 2)

AVFS_TEMP_e = CEnum(ctypes.c_uint32)
AVFS_TEMP_COLD = AVFS_TEMP_e.define('AVFS_TEMP_COLD', 0)
AVFS_TEMP_HOT = AVFS_TEMP_e.define('AVFS_TEMP_HOT', 1)
AVFS_TEMP_COUNT = AVFS_TEMP_e.define('AVFS_TEMP_COUNT', 2)

AVFS_D_e = CEnum(ctypes.c_uint32)
AVFS_D_G = AVFS_D_e.define('AVFS_D_G', 0)
AVFS_D_COUNT = AVFS_D_e.define('AVFS_D_COUNT', 1)

UCLK_DIV_e = CEnum(ctypes.c_uint32)
UCLK_DIV_BY_1 = UCLK_DIV_e.define('UCLK_DIV_BY_1', 0)
UCLK_DIV_BY_2 = UCLK_DIV_e.define('UCLK_DIV_BY_2', 1)
UCLK_DIV_BY_4 = UCLK_DIV_e.define('UCLK_DIV_BY_4', 2)
UCLK_DIV_BY_8 = UCLK_DIV_e.define('UCLK_DIV_BY_8', 3)

GpioIntPolarity_e = CEnum(ctypes.c_uint32)
GPIO_INT_POLARITY_ACTIVE_LOW = GpioIntPolarity_e.define('GPIO_INT_POLARITY_ACTIVE_LOW', 0)
GPIO_INT_POLARITY_ACTIVE_HIGH = GpioIntPolarity_e.define('GPIO_INT_POLARITY_ACTIVE_HIGH', 1)

PwrConfig_e = CEnum(ctypes.c_uint32)
PWR_CONFIG_TDP = PwrConfig_e.define('PWR_CONFIG_TDP', 0)
PWR_CONFIG_TGP = PwrConfig_e.define('PWR_CONFIG_TGP', 1)
PWR_CONFIG_TCP_ESTIMATED = PwrConfig_e.define('PWR_CONFIG_TCP_ESTIMATED', 2)
PWR_CONFIG_TCP_MEASURED = PwrConfig_e.define('PWR_CONFIG_TCP_MEASURED', 3)
PWR_CONFIG_TBP_DESKTOP = PwrConfig_e.define('PWR_CONFIG_TBP_DESKTOP', 4)
PWR_CONFIG_TBP_MOBILE = PwrConfig_e.define('PWR_CONFIG_TBP_MOBILE', 5)

class DpmDescriptor_t(Struct): pass
DpmDescriptor_t._fields_ = [
  ('Padding', uint8_t),
  ('SnapToDiscrete', uint8_t),
  ('NumDiscreteLevels', uint8_t),
  ('CalculateFopt', uint8_t),
  ('ConversionToAvfsClk', LinearInt_t),
  ('Padding3', (uint32_t * 3)),
  ('Padding4', uint16_t),
  ('FoptimalDc', uint16_t),
  ('FoptimalAc', uint16_t),
  ('Padding2', uint16_t),
]
PPT_THROTTLER_e = CEnum(ctypes.c_uint32)
PPT_THROTTLER_PPT0 = PPT_THROTTLER_e.define('PPT_THROTTLER_PPT0', 0)
PPT_THROTTLER_PPT1 = PPT_THROTTLER_e.define('PPT_THROTTLER_PPT1', 1)
PPT_THROTTLER_PPT2 = PPT_THROTTLER_e.define('PPT_THROTTLER_PPT2', 2)
PPT_THROTTLER_PPT3 = PPT_THROTTLER_e.define('PPT_THROTTLER_PPT3', 3)
PPT_THROTTLER_COUNT = PPT_THROTTLER_e.define('PPT_THROTTLER_COUNT', 4)

TEMP_e = CEnum(ctypes.c_uint32)
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

TDC_THROTTLER_e = CEnum(ctypes.c_uint32)
TDC_THROTTLER_GFX = TDC_THROTTLER_e.define('TDC_THROTTLER_GFX', 0)
TDC_THROTTLER_SOC = TDC_THROTTLER_e.define('TDC_THROTTLER_SOC', 1)
TDC_THROTTLER_COUNT = TDC_THROTTLER_e.define('TDC_THROTTLER_COUNT', 2)

SVI_PLANE_e = CEnum(ctypes.c_uint32)
SVI_PLANE_VDD_GFX = SVI_PLANE_e.define('SVI_PLANE_VDD_GFX', 0)
SVI_PLANE_VDD_SOC = SVI_PLANE_e.define('SVI_PLANE_VDD_SOC', 1)
SVI_PLANE_VDDCI_MEM = SVI_PLANE_e.define('SVI_PLANE_VDDCI_MEM', 2)
SVI_PLANE_VDDIO_MEM = SVI_PLANE_e.define('SVI_PLANE_VDDIO_MEM', 3)
SVI_PLANE_COUNT = SVI_PLANE_e.define('SVI_PLANE_COUNT', 4)

PMFW_VOLT_PLANE_e = CEnum(ctypes.c_uint32)
PMFW_VOLT_PLANE_GFX = PMFW_VOLT_PLANE_e.define('PMFW_VOLT_PLANE_GFX', 0)
PMFW_VOLT_PLANE_SOC = PMFW_VOLT_PLANE_e.define('PMFW_VOLT_PLANE_SOC', 1)
PMFW_VOLT_PLANE_COUNT = PMFW_VOLT_PLANE_e.define('PMFW_VOLT_PLANE_COUNT', 2)

CUSTOMER_VARIANT_e = CEnum(ctypes.c_uint32)
CUSTOMER_VARIANT_ROW = CUSTOMER_VARIANT_e.define('CUSTOMER_VARIANT_ROW', 0)
CUSTOMER_VARIANT_FALCON = CUSTOMER_VARIANT_e.define('CUSTOMER_VARIANT_FALCON', 1)
CUSTOMER_VARIANT_COUNT = CUSTOMER_VARIANT_e.define('CUSTOMER_VARIANT_COUNT', 2)

POWER_SOURCE_e = CEnum(ctypes.c_uint32)
POWER_SOURCE_AC = POWER_SOURCE_e.define('POWER_SOURCE_AC', 0)
POWER_SOURCE_DC = POWER_SOURCE_e.define('POWER_SOURCE_DC', 1)
POWER_SOURCE_COUNT = POWER_SOURCE_e.define('POWER_SOURCE_COUNT', 2)

MEM_VENDOR_e = CEnum(ctypes.c_uint32)
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

PP_GRTAVFS_HW_FUSE_e = CEnum(ctypes.c_uint32)
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

PP_GRTAVFS_FW_COMMON_FUSE_e = CEnum(ctypes.c_uint32)
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

PP_GRTAVFS_FW_SEP_FUSE_e = CEnum(ctypes.c_uint32)
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

class SviTelemetryScale_t(Struct): pass
int8_t = ctypes.c_byte
SviTelemetryScale_t._fields_ = [
  ('Offset', int8_t),
  ('Padding', uint8_t),
  ('MaxCurrent', uint16_t),
]
PP_OD_POWER_FEATURE_e = CEnum(ctypes.c_uint32)
PP_OD_POWER_FEATURE_ALWAYS_ENABLED = PP_OD_POWER_FEATURE_e.define('PP_OD_POWER_FEATURE_ALWAYS_ENABLED', 0)
PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING = PP_OD_POWER_FEATURE_e.define('PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING', 1)
PP_OD_POWER_FEATURE_ALWAYS_DISABLED = PP_OD_POWER_FEATURE_e.define('PP_OD_POWER_FEATURE_ALWAYS_DISABLED', 2)

FanMode_e = CEnum(ctypes.c_uint32)
FAN_MODE_AUTO = FanMode_e.define('FAN_MODE_AUTO', 0)
FAN_MODE_MANUAL_LINEAR = FanMode_e.define('FAN_MODE_MANUAL_LINEAR', 1)

OD_FAIL_e = CEnum(ctypes.c_uint32)
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

class OverDriveTable_t(Struct): pass
int16_t = ctypes.c_int16
OverDriveTable_t._fields_ = [
  ('FeatureCtrlMask', uint32_t),
  ('VoltageOffsetPerZoneBoundary', (int16_t * 6)),
  ('VddGfxVmax', uint16_t),
  ('VddSocVmax', uint16_t),
  ('IdlePwrSavingFeaturesCtrl', uint8_t),
  ('RuntimePwrSavingFeaturesCtrl', uint8_t),
  ('Padding', uint16_t),
  ('GfxclkFoffset', int16_t),
  ('Padding1', uint16_t),
  ('UclkFmin', uint16_t),
  ('UclkFmax', uint16_t),
  ('FclkFmin', uint16_t),
  ('FclkFmax', uint16_t),
  ('Ppt', int16_t),
  ('Tdc', int16_t),
  ('FanLinearPwmPoints', (uint8_t * 6)),
  ('FanLinearTempPoints', (uint8_t * 6)),
  ('FanMinimumPwm', uint16_t),
  ('AcousticTargetRpmThreshold', uint16_t),
  ('AcousticLimitRpmThreshold', uint16_t),
  ('FanTargetTemperature', uint16_t),
  ('FanZeroRpmEnable', uint8_t),
  ('FanZeroRpmStopTemp', uint8_t),
  ('FanMode', uint8_t),
  ('MaxOpTemp', uint8_t),
  ('AdvancedOdModeEnabled', uint8_t),
  ('Padding2', (uint8_t * 3)),
  ('GfxVoltageFullCtrlMode', uint16_t),
  ('SocVoltageFullCtrlMode', uint16_t),
  ('GfxclkFullCtrlMode', uint16_t),
  ('UclkFullCtrlMode', uint16_t),
  ('FclkFullCtrlMode', uint16_t),
  ('Padding3', uint16_t),
  ('GfxEdc', int16_t),
  ('GfxPccLimitControl', int16_t),
  ('GfxclkFmaxVmax', uint16_t),
  ('GfxclkFmaxVmaxTemperature', uint8_t),
  ('Padding4', (uint8_t * 1)),
  ('Spare', (uint32_t * 9)),
  ('MmHubPadding', (uint32_t * 8)),
]
class OverDriveTableExternal_t(Struct): pass
OverDriveTableExternal_t._fields_ = [
  ('OverDriveTable', OverDriveTable_t),
]
class OverDriveLimits_t(Struct): pass
OverDriveLimits_t._fields_ = [
  ('FeatureCtrlMask', uint32_t),
  ('VoltageOffsetPerZoneBoundary', (int16_t * 6)),
  ('VddGfxVmax', uint16_t),
  ('VddSocVmax', uint16_t),
  ('GfxclkFoffset', int16_t),
  ('Padding', uint16_t),
  ('UclkFmin', uint16_t),
  ('UclkFmax', uint16_t),
  ('FclkFmin', uint16_t),
  ('FclkFmax', uint16_t),
  ('Ppt', int16_t),
  ('Tdc', int16_t),
  ('FanLinearPwmPoints', (uint8_t * 6)),
  ('FanLinearTempPoints', (uint8_t * 6)),
  ('FanMinimumPwm', uint16_t),
  ('AcousticTargetRpmThreshold', uint16_t),
  ('AcousticLimitRpmThreshold', uint16_t),
  ('FanTargetTemperature', uint16_t),
  ('FanZeroRpmEnable', uint8_t),
  ('MaxOpTemp', uint8_t),
  ('Padding1', (uint8_t * 2)),
  ('GfxVoltageFullCtrlMode', uint16_t),
  ('SocVoltageFullCtrlMode', uint16_t),
  ('GfxclkFullCtrlMode', uint16_t),
  ('UclkFullCtrlMode', uint16_t),
  ('FclkFullCtrlMode', uint16_t),
  ('GfxEdc', int16_t),
  ('GfxPccLimitControl', int16_t),
  ('Padding2', int16_t),
  ('Spare', (uint32_t * 5)),
]
BOARD_GPIO_TYPE_e = CEnum(ctypes.c_uint32)
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

class BootValues_t(Struct): pass
BootValues_t._fields_ = [
  ('InitImuClk', uint16_t),
  ('InitSocclk', uint16_t),
  ('InitMpioclk', uint16_t),
  ('InitSmnclk', uint16_t),
  ('InitDispClk', uint16_t),
  ('InitDppClk', uint16_t),
  ('InitDprefclk', uint16_t),
  ('InitDcfclk', uint16_t),
  ('InitDtbclk', uint16_t),
  ('InitDbguSocClk', uint16_t),
  ('InitGfxclk_bypass', uint16_t),
  ('InitMp1clk', uint16_t),
  ('InitLclk', uint16_t),
  ('InitDbguBacoClk', uint16_t),
  ('InitBaco400clk', uint16_t),
  ('InitBaco1200clk_bypass', uint16_t),
  ('InitBaco700clk_bypass', uint16_t),
  ('InitBaco500clk', uint16_t),
  ('InitDclk0', uint16_t),
  ('InitVclk0', uint16_t),
  ('InitFclk', uint16_t),
  ('Padding1', uint16_t),
  ('InitUclkLevel', uint8_t),
  ('Padding', (uint8_t * 3)),
  ('InitVcoFreqPll0', uint32_t),
  ('InitVcoFreqPll1', uint32_t),
  ('InitVcoFreqPll2', uint32_t),
  ('InitVcoFreqPll3', uint32_t),
  ('InitVcoFreqPll4', uint32_t),
  ('InitVcoFreqPll5', uint32_t),
  ('InitVcoFreqPll6', uint32_t),
  ('InitVcoFreqPll7', uint32_t),
  ('InitVcoFreqPll8', uint32_t),
  ('InitGfx', uint16_t),
  ('InitSoc', uint16_t),
  ('InitVddIoMem', uint16_t),
  ('InitVddCiMem', uint16_t),
  ('Spare', (uint32_t * 8)),
]
class MsgLimits_t(Struct): pass
MsgLimits_t._fields_ = [
  ('Power', ((uint16_t * 2) * 4)),
  ('Tdc', (uint16_t * 2)),
  ('Temperature', (uint16_t * 12)),
  ('PwmLimitMin', uint8_t),
  ('PwmLimitMax', uint8_t),
  ('FanTargetTemperature', uint8_t),
  ('Spare1', (uint8_t * 1)),
  ('AcousticTargetRpmThresholdMin', uint16_t),
  ('AcousticTargetRpmThresholdMax', uint16_t),
  ('AcousticLimitRpmThresholdMin', uint16_t),
  ('AcousticLimitRpmThresholdMax', uint16_t),
  ('PccLimitMin', uint16_t),
  ('PccLimitMax', uint16_t),
  ('FanStopTempMin', uint16_t),
  ('FanStopTempMax', uint16_t),
  ('FanStartTempMin', uint16_t),
  ('FanStartTempMax', uint16_t),
  ('PowerMinPpt0', (uint16_t * 2)),
  ('Spare', (uint32_t * 11)),
]
class DriverReportedClocks_t(Struct): pass
DriverReportedClocks_t._fields_ = [
  ('BaseClockAc', uint16_t),
  ('GameClockAc', uint16_t),
  ('BoostClockAc', uint16_t),
  ('BaseClockDc', uint16_t),
  ('GameClockDc', uint16_t),
  ('BoostClockDc', uint16_t),
  ('MaxReportedClock', uint16_t),
  ('Padding', uint16_t),
  ('Reserved', (uint32_t * 3)),
]
class AvfsDcBtcParams_t(Struct): pass
AvfsDcBtcParams_t._fields_ = [
  ('DcBtcEnabled', uint8_t),
  ('Padding', (uint8_t * 3)),
  ('DcTol', uint16_t),
  ('DcBtcGb', uint16_t),
  ('DcBtcMin', uint16_t),
  ('DcBtcMax', uint16_t),
  ('DcBtcGbScalar', LinearInt_t),
]
class AvfsFuseOverride_t(Struct): pass
AvfsFuseOverride_t._fields_ = [
  ('AvfsTemp', (uint16_t * 2)),
  ('VftFMin', uint16_t),
  ('VInversion', uint16_t),
  ('qVft', (QuadraticInt_t * 2)),
  ('qAvfsGb', QuadraticInt_t),
  ('qAvfsGb2', QuadraticInt_t),
]
class PFE_Settings_t(Struct): pass
PFE_Settings_t._fields_ = [
  ('Version', uint8_t),
  ('Spare8', (uint8_t * 3)),
  ('FeaturesToRun', (uint32_t * 2)),
  ('FwDStateMask', uint32_t),
  ('DebugOverrides', uint32_t),
  ('Spare', (uint32_t * 2)),
]
class SkuTable_t(Struct): pass
int32_t = ctypes.c_int32
SkuTable_t._fields_ = [
  ('Version', uint32_t),
  ('TotalPowerConfig', uint8_t),
  ('CustomerVariant', uint8_t),
  ('MemoryTemperatureTypeMask', uint8_t),
  ('SmartShiftVersion', uint8_t),
  ('SocketPowerLimitSpare', (uint8_t * 10)),
  ('EnableLegacyPptLimit', uint8_t),
  ('UseInputTelemetry', uint8_t),
  ('SmartShiftMinReportedPptinDcs', uint8_t),
  ('PaddingPpt', (uint8_t * 7)),
  ('HwCtfTempLimit', uint16_t),
  ('PaddingInfra', uint16_t),
  ('FitControllerFailureRateLimit', uint32_t),
  ('FitControllerGfxDutyCycle', uint32_t),
  ('FitControllerSocDutyCycle', uint32_t),
  ('FitControllerSocOffset', uint32_t),
  ('GfxApccPlusResidencyLimit', uint32_t),
  ('ThrottlerControlMask', uint32_t),
  ('UlvVoltageOffset', (uint16_t * 2)),
  ('Padding', (uint8_t * 2)),
  ('DeepUlvVoltageOffsetSoc', uint16_t),
  ('DefaultMaxVoltage', (uint16_t * 2)),
  ('BoostMaxVoltage', (uint16_t * 2)),
  ('VminTempHystersis', (int16_t * 2)),
  ('VminTempThreshold', (int16_t * 2)),
  ('Vmin_Hot_T0', (uint16_t * 2)),
  ('Vmin_Cold_T0', (uint16_t * 2)),
  ('Vmin_Hot_Eol', (uint16_t * 2)),
  ('Vmin_Cold_Eol', (uint16_t * 2)),
  ('Vmin_Aging_Offset', (uint16_t * 2)),
  ('Spare_Vmin_Plat_Offset_Hot', (uint16_t * 2)),
  ('Spare_Vmin_Plat_Offset_Cold', (uint16_t * 2)),
  ('VcBtcFixedVminAgingOffset', (uint16_t * 2)),
  ('VcBtcVmin2PsmDegrationGb', (uint16_t * 2)),
  ('VcBtcPsmA', (uint32_t * 2)),
  ('VcBtcPsmB', (uint32_t * 2)),
  ('VcBtcVminA', (uint32_t * 2)),
  ('VcBtcVminB', (uint32_t * 2)),
  ('PerPartVminEnabled', (uint8_t * 2)),
  ('VcBtcEnabled', (uint8_t * 2)),
  ('SocketPowerLimitAcTau', (uint16_t * 4)),
  ('SocketPowerLimitDcTau', (uint16_t * 4)),
  ('Gfx_Vmin_droop', QuadraticInt_t),
  ('Soc_Vmin_droop', QuadraticInt_t),
  ('SpareVmin', (uint32_t * 6)),
  ('DpmDescriptor', (DpmDescriptor_t * 11)),
  ('FreqTableGfx', (uint16_t * 16)),
  ('FreqTableVclk', (uint16_t * 8)),
  ('FreqTableDclk', (uint16_t * 8)),
  ('FreqTableSocclk', (uint16_t * 8)),
  ('FreqTableUclk', (uint16_t * 6)),
  ('FreqTableShadowUclk', (uint16_t * 6)),
  ('FreqTableDispclk', (uint16_t * 8)),
  ('FreqTableDppClk', (uint16_t * 8)),
  ('FreqTableDprefclk', (uint16_t * 8)),
  ('FreqTableDcfclk', (uint16_t * 8)),
  ('FreqTableDtbclk', (uint16_t * 8)),
  ('FreqTableFclk', (uint16_t * 8)),
  ('DcModeMaxFreq', (uint32_t * 11)),
  ('GfxclkAibFmax', uint16_t),
  ('GfxDpmPadding', uint16_t),
  ('GfxclkFgfxoffEntry', uint16_t),
  ('GfxclkFgfxoffExitImu', uint16_t),
  ('GfxclkFgfxoffExitRlc', uint16_t),
  ('GfxclkThrottleClock', uint16_t),
  ('EnableGfxPowerStagesGpio', uint8_t),
  ('GfxIdlePadding', uint8_t),
  ('SmsRepairWRCKClkDivEn', uint8_t),
  ('SmsRepairWRCKClkDivVal', uint8_t),
  ('GfxOffEntryEarlyMGCGEn', uint8_t),
  ('GfxOffEntryForceCGCGEn', uint8_t),
  ('GfxOffEntryForceCGCGDelayEn', uint8_t),
  ('GfxOffEntryForceCGCGDelayVal', uint8_t),
  ('GfxclkFreqGfxUlv', uint16_t),
  ('GfxIdlePadding2', (uint8_t * 2)),
  ('GfxOffEntryHysteresis', uint32_t),
  ('GfxoffSpare', (uint32_t * 15)),
  ('DfllMstrOscConfigA', uint16_t),
  ('DfllSlvOscConfigA', uint16_t),
  ('DfllBtcMasterScalerM', uint32_t),
  ('DfllBtcMasterScalerB', int32_t),
  ('DfllBtcSlaveScalerM', uint32_t),
  ('DfllBtcSlaveScalerB', int32_t),
  ('DfllPccAsWaitCtrl', uint32_t),
  ('DfllPccAsStepCtrl', uint32_t),
  ('GfxDfllSpare', (uint32_t * 9)),
  ('DvoPsmDownThresholdVoltage', uint32_t),
  ('DvoPsmUpThresholdVoltage', uint32_t),
  ('DvoFmaxLowScaler', uint32_t),
  ('PaddingDcs', uint32_t),
  ('DcsMinGfxOffTime', uint16_t),
  ('DcsMaxGfxOffTime', uint16_t),
  ('DcsMinCreditAccum', uint32_t),
  ('DcsExitHysteresis', uint16_t),
  ('DcsTimeout', uint16_t),
  ('DcsPfGfxFopt', uint32_t),
  ('DcsPfUclkFopt', uint32_t),
  ('FoptEnabled', uint8_t),
  ('DcsSpare2', (uint8_t * 3)),
  ('DcsFoptM', uint32_t),
  ('DcsFoptB', uint32_t),
  ('DcsSpare', (uint32_t * 9)),
  ('UseStrobeModeOptimizations', uint8_t),
  ('PaddingMem', (uint8_t * 3)),
  ('UclkDpmPstates', (uint8_t * 6)),
  ('UclkDpmShadowPstates', (uint8_t * 6)),
  ('FreqTableUclkDiv', (uint8_t * 6)),
  ('FreqTableShadowUclkDiv', (uint8_t * 6)),
  ('MemVmempVoltage', (uint16_t * 6)),
  ('MemVddioVoltage', (uint16_t * 6)),
  ('DalDcModeMaxUclkFreq', uint16_t),
  ('PaddingsMem', (uint8_t * 2)),
  ('PaddingFclk', uint32_t),
  ('PcieGenSpeed', (uint8_t * 3)),
  ('PcieLaneCount', (uint8_t * 3)),
  ('LclkFreq', (uint16_t * 3)),
  ('OverrideGfxAvfsFuses', uint8_t),
  ('GfxAvfsPadding', (uint8_t * 1)),
  ('DroopGBStDev', uint16_t),
  ('SocHwRtAvfsFuses', (uint32_t * 32)),
  ('GfxL2HwRtAvfsFuses', (uint32_t * 32)),
  ('PsmDidt_Vcross', (uint16_t * 2)),
  ('PsmDidt_StaticDroop_A', (uint32_t * 3)),
  ('PsmDidt_StaticDroop_B', (uint32_t * 3)),
  ('PsmDidt_DynDroop_A', (uint32_t * 3)),
  ('PsmDidt_DynDroop_B', (uint32_t * 3)),
  ('spare_HwRtAvfsFuses', (uint32_t * 19)),
  ('SocCommonRtAvfs', (uint32_t * 13)),
  ('GfxCommonRtAvfs', (uint32_t * 13)),
  ('SocFwRtAvfsFuses', (uint32_t * 19)),
  ('GfxL2FwRtAvfsFuses', (uint32_t * 19)),
  ('spare_FwRtAvfsFuses', (uint32_t * 19)),
  ('Soc_Droop_PWL_F', (uint32_t * 5)),
  ('Soc_Droop_PWL_a', (uint32_t * 5)),
  ('Soc_Droop_PWL_b', (uint32_t * 5)),
  ('Soc_Droop_PWL_c', (uint32_t * 5)),
  ('Gfx_Droop_PWL_F', (uint32_t * 5)),
  ('Gfx_Droop_PWL_a', (uint32_t * 5)),
  ('Gfx_Droop_PWL_b', (uint32_t * 5)),
  ('Gfx_Droop_PWL_c', (uint32_t * 5)),
  ('Gfx_Static_PWL_Offset', (uint32_t * 5)),
  ('Soc_Static_PWL_Offset', (uint32_t * 5)),
  ('dGbV_dT_vmin', uint32_t),
  ('dGbV_dT_vmax', uint32_t),
  ('PaddingV2F', (uint32_t * 4)),
  ('DcBtcGfxParams', AvfsDcBtcParams_t),
  ('SSCurve_GFX', QuadraticInt_t),
  ('GfxAvfsSpare', (uint32_t * 29)),
  ('OverrideSocAvfsFuses', uint8_t),
  ('MinSocAvfsRevision', uint8_t),
  ('SocAvfsPadding', (uint8_t * 2)),
  ('SocAvfsFuseOverride', (AvfsFuseOverride_t * 1)),
  ('dBtcGbSoc', (DroopInt_t * 1)),
  ('qAgingGb', (LinearInt_t * 1)),
  ('qStaticVoltageOffset', (QuadraticInt_t * 1)),
  ('DcBtcSocParams', (AvfsDcBtcParams_t * 1)),
  ('SSCurve_SOC', QuadraticInt_t),
  ('SocAvfsSpare', (uint32_t * 29)),
  ('BootValues', BootValues_t),
  ('DriverReportedClocks', DriverReportedClocks_t),
  ('MsgLimits', MsgLimits_t),
  ('OverDriveLimitsBasicMin', OverDriveLimits_t),
  ('OverDriveLimitsBasicMax', OverDriveLimits_t),
  ('OverDriveLimitsAdvancedMin', OverDriveLimits_t),
  ('OverDriveLimitsAdvancedMax', OverDriveLimits_t),
  ('TotalBoardPowerSupport', uint8_t),
  ('TotalBoardPowerPadding', (uint8_t * 1)),
  ('TotalBoardPowerRoc', uint16_t),
  ('qFeffCoeffGameClock', (QuadraticInt_t * 2)),
  ('qFeffCoeffBaseClock', (QuadraticInt_t * 2)),
  ('qFeffCoeffBoostClock', (QuadraticInt_t * 2)),
  ('AptUclkGfxclkLookup', ((int32_t * 6) * 2)),
  ('AptUclkGfxclkLookupHyst', ((uint32_t * 6) * 2)),
  ('AptPadding', uint32_t),
  ('GfxXvminDidtDroopThresh', QuadraticInt_t),
  ('GfxXvminDidtResetDDWait', uint32_t),
  ('GfxXvminDidtClkStopWait', uint32_t),
  ('GfxXvminDidtFcsStepCtrl', uint32_t),
  ('GfxXvminDidtFcsWaitCtrl', uint32_t),
  ('PsmModeEnabled', uint32_t),
  ('P2v_a', uint32_t),
  ('P2v_b', uint32_t),
  ('P2v_c', uint32_t),
  ('T2p_a', uint32_t),
  ('T2p_b', uint32_t),
  ('T2p_c', uint32_t),
  ('P2vTemp', uint32_t),
  ('PsmDidtStaticSettings', QuadraticInt_t),
  ('PsmDidtDynamicSettings', QuadraticInt_t),
  ('PsmDidtAvgDiv', uint8_t),
  ('PsmDidtForceStall', uint8_t),
  ('PsmDidtReleaseTimer', uint16_t),
  ('PsmDidtStallPattern', uint32_t),
  ('CacEdcCacLeakageC0', uint32_t),
  ('CacEdcCacLeakageC1', uint32_t),
  ('CacEdcCacLeakageC2', uint32_t),
  ('CacEdcCacLeakageC3', uint32_t),
  ('CacEdcCacLeakageC4', uint32_t),
  ('CacEdcCacLeakageC5', uint32_t),
  ('CacEdcGfxClkScalar', uint32_t),
  ('CacEdcGfxClkIntercept', uint32_t),
  ('CacEdcCac_m', uint32_t),
  ('CacEdcCac_b', uint32_t),
  ('CacEdcCurrLimitGuardband', uint32_t),
  ('CacEdcDynToTotalCacRatio', uint32_t),
  ('XVmin_Gfx_EdcThreshScalar', uint32_t),
  ('XVmin_Gfx_EdcEnableFreq', uint32_t),
  ('XVmin_Gfx_EdcPccAsStepCtrl', uint32_t),
  ('XVmin_Gfx_EdcPccAsWaitCtrl', uint32_t),
  ('XVmin_Gfx_EdcThreshold', uint16_t),
  ('XVmin_Gfx_EdcFiltHysWaitCtrl', uint16_t),
  ('XVmin_Soc_EdcThreshScalar', uint32_t),
  ('XVmin_Soc_EdcEnableFreq', uint32_t),
  ('XVmin_Soc_EdcThreshold', uint32_t),
  ('XVmin_Soc_EdcStepUpTime', uint16_t),
  ('XVmin_Soc_EdcStepDownTime', uint16_t),
  ('XVmin_Soc_EdcInitPccStep', uint8_t),
  ('PaddingSocEdc', (uint8_t * 3)),
  ('GfxXvminFuseOverride', uint8_t),
  ('SocXvminFuseOverride', uint8_t),
  ('PaddingXvminFuseOverride', (uint8_t * 2)),
  ('GfxXvminFddTempLow', uint8_t),
  ('GfxXvminFddTempHigh', uint8_t),
  ('SocXvminFddTempLow', uint8_t),
  ('SocXvminFddTempHigh', uint8_t),
  ('GfxXvminFddVolt0', uint16_t),
  ('GfxXvminFddVolt1', uint16_t),
  ('GfxXvminFddVolt2', uint16_t),
  ('SocXvminFddVolt0', uint16_t),
  ('SocXvminFddVolt1', uint16_t),
  ('SocXvminFddVolt2', uint16_t),
  ('GfxXvminDsFddDsm', (uint16_t * 6)),
  ('GfxXvminEdcFddDsm', (uint16_t * 6)),
  ('SocXvminEdcFddDsm', (uint16_t * 6)),
  ('Spare', uint32_t),
  ('MmHubPadding', (uint32_t * 8)),
]
class Svi3RegulatorSettings_t(Struct): pass
Svi3RegulatorSettings_t._fields_ = [
  ('SlewRateConditions', uint8_t),
  ('LoadLineAdjust', uint8_t),
  ('VoutOffset', uint8_t),
  ('VidMax', uint8_t),
  ('VidMin', uint8_t),
  ('TenBitTelEn', uint8_t),
  ('SixteenBitTelEn', uint8_t),
  ('OcpThresh', uint8_t),
  ('OcpWarnThresh', uint8_t),
  ('OcpSettings', uint8_t),
  ('VrhotThresh', uint8_t),
  ('OtpThresh', uint8_t),
  ('UvpOvpDeltaRef', uint8_t),
  ('PhaseShed', uint8_t),
  ('Padding', (uint8_t * 10)),
  ('SettingOverrideMask', uint32_t),
]
class BoardTable_t(Struct): pass
BoardTable_t._fields_ = [
  ('Version', uint32_t),
  ('I2cControllers', (I2cControllerConfig_t * 8)),
  ('SlaveAddrMapping', (uint8_t * 4)),
  ('VrPsiSupport', (uint8_t * 4)),
  ('Svi3SvcSpeed', uint32_t),
  ('EnablePsi6', (uint8_t * 4)),
  ('Svi3RegSettings', (Svi3RegulatorSettings_t * 4)),
  ('LedOffGpio', uint8_t),
  ('FanOffGpio', uint8_t),
  ('GfxVrPowerStageOffGpio', uint8_t),
  ('AcDcGpio', uint8_t),
  ('AcDcPolarity', uint8_t),
  ('VR0HotGpio', uint8_t),
  ('VR0HotPolarity', uint8_t),
  ('GthrGpio', uint8_t),
  ('GthrPolarity', uint8_t),
  ('LedPin0', uint8_t),
  ('LedPin1', uint8_t),
  ('LedPin2', uint8_t),
  ('LedEnableMask', uint8_t),
  ('LedPcie', uint8_t),
  ('LedError', uint8_t),
  ('PaddingLed', uint8_t),
  ('UclkTrainingModeSpreadPercent', uint8_t),
  ('UclkSpreadPadding', uint8_t),
  ('UclkSpreadFreq', uint16_t),
  ('UclkSpreadPercent', (uint8_t * 16)),
  ('GfxclkSpreadEnable', uint8_t),
  ('FclkSpreadPercent', uint8_t),
  ('FclkSpreadFreq', uint16_t),
  ('DramWidth', uint8_t),
  ('PaddingMem1', (uint8_t * 7)),
  ('HsrEnabled', uint8_t),
  ('VddqOffEnabled', uint8_t),
  ('PaddingUmcFlags', (uint8_t * 2)),
  ('Paddign1', uint32_t),
  ('BacoEntryDelay', uint32_t),
  ('FuseWritePowerMuxPresent', uint8_t),
  ('FuseWritePadding', (uint8_t * 3)),
  ('LoadlineGfx', uint32_t),
  ('LoadlineSoc', uint32_t),
  ('GfxEdcLimit', uint32_t),
  ('SocEdcLimit', uint32_t),
  ('RestBoardPower', uint32_t),
  ('ConnectorsImpedance', uint32_t),
  ('EpcsSens0', uint8_t),
  ('EpcsSens1', uint8_t),
  ('PaddingEpcs', (uint8_t * 2)),
  ('BoardSpare', (uint32_t * 52)),
  ('MmHubPadding', (uint32_t * 8)),
]
class CustomSkuTable_t(Struct): pass
CustomSkuTable_t._fields_ = [
  ('SocketPowerLimitAc', (uint16_t * 4)),
  ('VrTdcLimit', (uint16_t * 2)),
  ('TotalIdleBoardPowerM', int16_t),
  ('TotalIdleBoardPowerB', int16_t),
  ('TotalBoardPowerM', int16_t),
  ('TotalBoardPowerB', int16_t),
  ('TemperatureLimit', (uint16_t * 12)),
  ('FanStopTemp', (uint16_t * 12)),
  ('FanStartTemp', (uint16_t * 12)),
  ('FanGain', (uint16_t * 12)),
  ('FanPwmMin', uint16_t),
  ('AcousticTargetRpmThreshold', uint16_t),
  ('AcousticLimitRpmThreshold', uint16_t),
  ('FanMaximumRpm', uint16_t),
  ('MGpuAcousticLimitRpmThreshold', uint16_t),
  ('FanTargetGfxclk', uint16_t),
  ('TempInputSelectMask', uint32_t),
  ('FanZeroRpmEnable', uint8_t),
  ('FanTachEdgePerRev', uint8_t),
  ('FanPadding', uint16_t),
  ('FanTargetTemperature', (uint16_t * 12)),
  ('FuzzyFan_ErrorSetDelta', int16_t),
  ('FuzzyFan_ErrorRateSetDelta', int16_t),
  ('FuzzyFan_PwmSetDelta', int16_t),
  ('FanPadding2', uint16_t),
  ('FwCtfLimit', (uint16_t * 12)),
  ('IntakeTempEnableRPM', uint16_t),
  ('IntakeTempOffsetTemp', int16_t),
  ('IntakeTempReleaseTemp', uint16_t),
  ('IntakeTempHighIntakeAcousticLimit', uint16_t),
  ('IntakeTempAcouticLimitReleaseRate', uint16_t),
  ('FanAbnormalTempLimitOffset', int16_t),
  ('FanStalledTriggerRpm', uint16_t),
  ('FanAbnormalTriggerRpmCoeff', uint16_t),
  ('FanSpare', (uint16_t * 1)),
  ('FanIntakeSensorSupport', uint8_t),
  ('FanIntakePadding', uint8_t),
  ('FanSpare2', (uint32_t * 12)),
  ('ODFeatureCtrlMask', uint32_t),
  ('TemperatureLimit_Hynix', uint16_t),
  ('TemperatureLimit_Micron', uint16_t),
  ('TemperatureFwCtfLimit_Hynix', uint16_t),
  ('TemperatureFwCtfLimit_Micron', uint16_t),
  ('PlatformTdcLimit', (uint16_t * 2)),
  ('SocketPowerLimitDc', (uint16_t * 4)),
  ('SocketPowerLimitSmartShift2', uint16_t),
  ('CustomSkuSpare16b', uint16_t),
  ('CustomSkuSpare32b', (uint32_t * 10)),
  ('MmHubPadding', (uint32_t * 8)),
]
class PPTable_t(Struct): pass
PPTable_t._fields_ = [
  ('PFE_Settings', PFE_Settings_t),
  ('SkuTable', SkuTable_t),
  ('CustomSkuTable', CustomSkuTable_t),
  ('BoardTable', BoardTable_t),
]
class DriverSmuConfig_t(Struct): pass
DriverSmuConfig_t._fields_ = [
  ('GfxclkAverageLpfTau', uint16_t),
  ('FclkAverageLpfTau', uint16_t),
  ('UclkAverageLpfTau', uint16_t),
  ('GfxActivityLpfTau', uint16_t),
  ('UclkActivityLpfTau', uint16_t),
  ('UclkMaxActivityLpfTau', uint16_t),
  ('SocketPowerLpfTau', uint16_t),
  ('VcnClkAverageLpfTau', uint16_t),
  ('VcnUsageAverageLpfTau', uint16_t),
  ('PcieActivityLpTau', uint16_t),
]
class DriverSmuConfigExternal_t(Struct): pass
DriverSmuConfigExternal_t._fields_ = [
  ('DriverSmuConfig', DriverSmuConfig_t),
  ('Spare', (uint32_t * 8)),
  ('MmHubPadding', (uint32_t * 8)),
]
class DriverInfoTable_t(Struct): pass
DriverInfoTable_t._fields_ = [
  ('FreqTableGfx', (uint16_t * 16)),
  ('FreqTableVclk', (uint16_t * 8)),
  ('FreqTableDclk', (uint16_t * 8)),
  ('FreqTableSocclk', (uint16_t * 8)),
  ('FreqTableUclk', (uint16_t * 6)),
  ('FreqTableDispclk', (uint16_t * 8)),
  ('FreqTableDppClk', (uint16_t * 8)),
  ('FreqTableDprefclk', (uint16_t * 8)),
  ('FreqTableDcfclk', (uint16_t * 8)),
  ('FreqTableDtbclk', (uint16_t * 8)),
  ('FreqTableFclk', (uint16_t * 8)),
  ('DcModeMaxFreq', (uint16_t * 11)),
  ('Padding', uint16_t),
  ('Spare', (uint32_t * 32)),
  ('MmHubPadding', (uint32_t * 8)),
]
class SmuMetrics_t(Struct): pass
SmuMetrics_t._fields_ = [
  ('CurrClock', (uint32_t * 11)),
  ('AverageGfxclkFrequencyTarget', uint16_t),
  ('AverageGfxclkFrequencyPreDs', uint16_t),
  ('AverageGfxclkFrequencyPostDs', uint16_t),
  ('AverageFclkFrequencyPreDs', uint16_t),
  ('AverageFclkFrequencyPostDs', uint16_t),
  ('AverageMemclkFrequencyPreDs', uint16_t),
  ('AverageMemclkFrequencyPostDs', uint16_t),
  ('AverageVclk0Frequency', uint16_t),
  ('AverageDclk0Frequency', uint16_t),
  ('AverageVclk1Frequency', uint16_t),
  ('AverageDclk1Frequency', uint16_t),
  ('AveragePCIeBusy', uint16_t),
  ('dGPU_W_MAX', uint16_t),
  ('padding', uint16_t),
  ('MovingAverageGfxclkFrequencyTarget', uint16_t),
  ('MovingAverageGfxclkFrequencyPreDs', uint16_t),
  ('MovingAverageGfxclkFrequencyPostDs', uint16_t),
  ('MovingAverageFclkFrequencyPreDs', uint16_t),
  ('MovingAverageFclkFrequencyPostDs', uint16_t),
  ('MovingAverageMemclkFrequencyPreDs', uint16_t),
  ('MovingAverageMemclkFrequencyPostDs', uint16_t),
  ('MovingAverageVclk0Frequency', uint16_t),
  ('MovingAverageDclk0Frequency', uint16_t),
  ('MovingAverageGfxActivity', uint16_t),
  ('MovingAverageUclkActivity', uint16_t),
  ('MovingAverageVcn0ActivityPercentage', uint16_t),
  ('MovingAveragePCIeBusy', uint16_t),
  ('MovingAverageUclkActivity_MAX', uint16_t),
  ('MovingAverageSocketPower', uint16_t),
  ('MovingAveragePadding', uint16_t),
  ('MetricsCounter', uint32_t),
  ('AvgVoltage', (uint16_t * 4)),
  ('AvgCurrent', (uint16_t * 4)),
  ('AverageGfxActivity', uint16_t),
  ('AverageUclkActivity', uint16_t),
  ('AverageVcn0ActivityPercentage', uint16_t),
  ('Vcn1ActivityPercentage', uint16_t),
  ('EnergyAccumulator', uint32_t),
  ('AverageSocketPower', uint16_t),
  ('AverageTotalBoardPower', uint16_t),
  ('AvgTemperature', (uint16_t * 12)),
  ('AvgTemperatureFanIntake', uint16_t),
  ('PcieRate', uint8_t),
  ('PcieWidth', uint8_t),
  ('AvgFanPwm', uint8_t),
  ('Padding', (uint8_t * 1)),
  ('AvgFanRpm', uint16_t),
  ('ThrottlingPercentage', (uint8_t * 21)),
  ('VmaxThrottlingPercentage', uint8_t),
  ('padding1', (uint8_t * 2)),
  ('D3HotEntryCountPerMode', (uint32_t * 4)),
  ('D3HotExitCountPerMode', (uint32_t * 4)),
  ('ArmMsgReceivedCountPerMode', (uint32_t * 4)),
  ('ApuSTAPMSmartShiftLimit', uint16_t),
  ('ApuSTAPMLimit', uint16_t),
  ('AvgApuSocketPower', uint16_t),
  ('AverageUclkActivity_MAX', uint16_t),
  ('PublicSerialNumberLower', uint32_t),
  ('PublicSerialNumberUpper', uint32_t),
]
class SmuMetricsExternal_t(Struct): pass
SmuMetricsExternal_t._fields_ = [
  ('SmuMetrics', SmuMetrics_t),
  ('Spare', (uint32_t * 30)),
  ('MmHubPadding', (uint32_t * 8)),
]
class WatermarkRowGeneric_t(Struct): pass
WatermarkRowGeneric_t._fields_ = [
  ('WmSetting', uint8_t),
  ('Flags', uint8_t),
  ('Padding', (uint8_t * 2)),
]
WATERMARKS_FLAGS_e = CEnum(ctypes.c_uint32)
WATERMARKS_CLOCK_RANGE = WATERMARKS_FLAGS_e.define('WATERMARKS_CLOCK_RANGE', 0)
WATERMARKS_DUMMY_PSTATE = WATERMARKS_FLAGS_e.define('WATERMARKS_DUMMY_PSTATE', 1)
WATERMARKS_MALL = WATERMARKS_FLAGS_e.define('WATERMARKS_MALL', 2)
WATERMARKS_COUNT = WATERMARKS_FLAGS_e.define('WATERMARKS_COUNT', 3)

class Watermarks_t(Struct): pass
Watermarks_t._fields_ = [
  ('WatermarkRow', (WatermarkRowGeneric_t * 4)),
]
class WatermarksExternal_t(Struct): pass
WatermarksExternal_t._fields_ = [
  ('Watermarks', Watermarks_t),
  ('Spare', (uint32_t * 16)),
  ('MmHubPadding', (uint32_t * 8)),
]
class AvfsDebugTable_t(Struct): pass
AvfsDebugTable_t._fields_ = [
  ('avgPsmCount', (uint16_t * 76)),
  ('minPsmCount', (uint16_t * 76)),
  ('maxPsmCount', (uint16_t * 76)),
  ('avgPsmVoltage', (ctypes.c_float * 76)),
  ('minPsmVoltage', (ctypes.c_float * 76)),
  ('maxPsmVoltage', (ctypes.c_float * 76)),
]
class AvfsDebugTableExternal_t(Struct): pass
AvfsDebugTableExternal_t._fields_ = [
  ('AvfsDebugTable', AvfsDebugTable_t),
  ('MmHubPadding', (uint32_t * 8)),
]
class DpmActivityMonitorCoeffInt_t(Struct): pass
DpmActivityMonitorCoeffInt_t._fields_ = [
  ('Gfx_ActiveHystLimit', uint8_t),
  ('Gfx_IdleHystLimit', uint8_t),
  ('Gfx_FPS', uint8_t),
  ('Gfx_MinActiveFreqType', uint8_t),
  ('Gfx_BoosterFreqType', uint8_t),
  ('PaddingGfx', uint8_t),
  ('Gfx_MinActiveFreq', uint16_t),
  ('Gfx_BoosterFreq', uint16_t),
  ('Gfx_PD_Data_time_constant', uint16_t),
  ('Gfx_PD_Data_limit_a', uint32_t),
  ('Gfx_PD_Data_limit_b', uint32_t),
  ('Gfx_PD_Data_limit_c', uint32_t),
  ('Gfx_PD_Data_error_coeff', uint32_t),
  ('Gfx_PD_Data_error_rate_coeff', uint32_t),
  ('Fclk_ActiveHystLimit', uint8_t),
  ('Fclk_IdleHystLimit', uint8_t),
  ('Fclk_FPS', uint8_t),
  ('Fclk_MinActiveFreqType', uint8_t),
  ('Fclk_BoosterFreqType', uint8_t),
  ('PaddingFclk', uint8_t),
  ('Fclk_MinActiveFreq', uint16_t),
  ('Fclk_BoosterFreq', uint16_t),
  ('Fclk_PD_Data_time_constant', uint16_t),
  ('Fclk_PD_Data_limit_a', uint32_t),
  ('Fclk_PD_Data_limit_b', uint32_t),
  ('Fclk_PD_Data_limit_c', uint32_t),
  ('Fclk_PD_Data_error_coeff', uint32_t),
  ('Fclk_PD_Data_error_rate_coeff', uint32_t),
  ('Mem_UpThreshold_Limit', (uint32_t * 6)),
  ('Mem_UpHystLimit', (uint8_t * 6)),
  ('Mem_DownHystLimit', (uint16_t * 6)),
  ('Mem_Fps', uint16_t),
]
class DpmActivityMonitorCoeffIntExternal_t(Struct): pass
DpmActivityMonitorCoeffIntExternal_t._fields_ = [
  ('DpmActivityMonitorCoeffInt', DpmActivityMonitorCoeffInt_t),
  ('MmHubPadding', (uint32_t * 8)),
]
class struct_smu_hw_power_state(Struct): pass
struct_smu_hw_power_state._fields_ = [
  ('magic', ctypes.c_uint32),
]
class struct_smu_power_state(Struct): pass
enum_smu_state_ui_label = CEnum(ctypes.c_uint32)
SMU_STATE_UI_LABEL_NONE = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_NONE', 0)
SMU_STATE_UI_LABEL_BATTERY = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BATTERY', 1)
SMU_STATE_UI_TABEL_MIDDLE_LOW = enum_smu_state_ui_label.define('SMU_STATE_UI_TABEL_MIDDLE_LOW', 2)
SMU_STATE_UI_LABEL_BALLANCED = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BALLANCED', 3)
SMU_STATE_UI_LABEL_MIDDLE_HIGHT = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_MIDDLE_HIGHT', 4)
SMU_STATE_UI_LABEL_PERFORMANCE = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_PERFORMANCE', 5)
SMU_STATE_UI_LABEL_BACO = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BACO', 6)

enum_smu_state_classification_flag = CEnum(ctypes.c_uint32)
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

class struct_smu_state_classification_block(Struct): pass
struct_smu_state_classification_block._fields_ = [
  ('ui_label', enum_smu_state_ui_label),
  ('flags', enum_smu_state_classification_flag),
  ('bios_index', ctypes.c_int32),
  ('temporary_state', ctypes.c_bool),
  ('to_be_deleted', ctypes.c_bool),
]
class struct_smu_state_pcie_block(Struct): pass
struct_smu_state_pcie_block._fields_ = [
  ('lanes', ctypes.c_uint32),
]
enum_smu_refreshrate_source = CEnum(ctypes.c_uint32)
SMU_REFRESHRATE_SOURCE_EDID = enum_smu_refreshrate_source.define('SMU_REFRESHRATE_SOURCE_EDID', 0)
SMU_REFRESHRATE_SOURCE_EXPLICIT = enum_smu_refreshrate_source.define('SMU_REFRESHRATE_SOURCE_EXPLICIT', 1)

class struct_smu_state_display_block(Struct): pass
struct_smu_state_display_block._fields_ = [
  ('disable_frame_modulation', ctypes.c_bool),
  ('limit_refreshrate', ctypes.c_bool),
  ('refreshrate_source', enum_smu_refreshrate_source),
  ('explicit_refreshrate', ctypes.c_int32),
  ('edid_refreshrate_index', ctypes.c_int32),
  ('enable_vari_bright', ctypes.c_bool),
]
class struct_smu_state_memory_block(Struct): pass
struct_smu_state_memory_block._fields_ = [
  ('dll_off', ctypes.c_bool),
  ('m3arb', ctypes.c_ubyte),
  ('unused', (ctypes.c_ubyte * 3)),
]
class struct_smu_state_software_algorithm_block(Struct): pass
struct_smu_state_software_algorithm_block._fields_ = [
  ('disable_load_balancing', ctypes.c_bool),
  ('enable_sleep_for_timestamps', ctypes.c_bool),
]
class struct_smu_temperature_range(Struct): pass
struct_smu_temperature_range._fields_ = [
  ('min', ctypes.c_int32),
  ('max', ctypes.c_int32),
  ('edge_emergency_max', ctypes.c_int32),
  ('hotspot_min', ctypes.c_int32),
  ('hotspot_crit_max', ctypes.c_int32),
  ('hotspot_emergency_max', ctypes.c_int32),
  ('mem_min', ctypes.c_int32),
  ('mem_crit_max', ctypes.c_int32),
  ('mem_emergency_max', ctypes.c_int32),
  ('software_shutdown_temp', ctypes.c_int32),
  ('software_shutdown_temp_offset', ctypes.c_int32),
]
class struct_smu_state_validation_block(Struct): pass
struct_smu_state_validation_block._fields_ = [
  ('single_display_only', ctypes.c_bool),
  ('disallow_on_dc', ctypes.c_bool),
  ('supported_power_levels', ctypes.c_ubyte),
]
class struct_smu_uvd_clocks(Struct): pass
struct_smu_uvd_clocks._fields_ = [
  ('vclk', ctypes.c_uint32),
  ('dclk', ctypes.c_uint32),
]
enum_smu_power_src_type = CEnum(ctypes.c_uint32)
SMU_POWER_SOURCE_AC = enum_smu_power_src_type.define('SMU_POWER_SOURCE_AC', 0)
SMU_POWER_SOURCE_DC = enum_smu_power_src_type.define('SMU_POWER_SOURCE_DC', 1)
SMU_POWER_SOURCE_COUNT = enum_smu_power_src_type.define('SMU_POWER_SOURCE_COUNT', 2)

enum_smu_ppt_limit_type = CEnum(ctypes.c_uint32)
SMU_DEFAULT_PPT_LIMIT = enum_smu_ppt_limit_type.define('SMU_DEFAULT_PPT_LIMIT', 0)
SMU_FAST_PPT_LIMIT = enum_smu_ppt_limit_type.define('SMU_FAST_PPT_LIMIT', 1)

enum_smu_ppt_limit_level = CEnum(ctypes.c_int32)
SMU_PPT_LIMIT_MIN = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_MIN', -1)
SMU_PPT_LIMIT_CURRENT = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_CURRENT', 0)
SMU_PPT_LIMIT_DEFAULT = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_DEFAULT', 1)
SMU_PPT_LIMIT_MAX = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_MAX', 2)

enum_smu_memory_pool_size = CEnum(ctypes.c_uint32)
SMU_MEMORY_POOL_SIZE_ZERO = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_ZERO', 0)
SMU_MEMORY_POOL_SIZE_256_MB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_256_MB', 268435456)
SMU_MEMORY_POOL_SIZE_512_MB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_512_MB', 536870912)
SMU_MEMORY_POOL_SIZE_1_GB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_1_GB', 1073741824)
SMU_MEMORY_POOL_SIZE_2_GB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_2_GB', 2147483648)

enum_smu_clk_type = CEnum(ctypes.c_uint32)
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

class struct_smu_user_dpm_profile(Struct): pass
struct_smu_user_dpm_profile._fields_ = [
  ('fan_mode', ctypes.c_uint32),
  ('power_limit', ctypes.c_uint32),
  ('fan_speed_pwm', ctypes.c_uint32),
  ('fan_speed_rpm', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('user_od', ctypes.c_uint32),
  ('clk_mask', (ctypes.c_uint32 * 28)),
  ('clk_dependency', ctypes.c_uint32),
]
class struct_smu_table(Struct): pass
class struct_amdgpu_bo(Struct): pass
struct_smu_table._fields_ = [
  ('size', ctypes.c_uint64),
  ('align', ctypes.c_uint32),
  ('domain', ctypes.c_ubyte),
  ('mc_address', ctypes.c_uint64),
  ('cpu_addr', ctypes.c_void_p),
  ('bo', ctypes.POINTER(struct_amdgpu_bo)),
  ('version', ctypes.c_uint32),
]
enum_smu_perf_level_designation = CEnum(ctypes.c_uint32)
PERF_LEVEL_ACTIVITY = enum_smu_perf_level_designation.define('PERF_LEVEL_ACTIVITY', 0)
PERF_LEVEL_POWER_CONTAINMENT = enum_smu_perf_level_designation.define('PERF_LEVEL_POWER_CONTAINMENT', 1)

class struct_smu_performance_level(Struct): pass
struct_smu_performance_level._fields_ = [
  ('core_clock', ctypes.c_uint32),
  ('memory_clock', ctypes.c_uint32),
  ('vddc', ctypes.c_uint32),
  ('vddci', ctypes.c_uint32),
  ('non_local_mem_freq', ctypes.c_uint32),
  ('non_local_mem_width', ctypes.c_uint32),
]
class struct_smu_clock_info(Struct): pass
struct_smu_clock_info._fields_ = [
  ('min_mem_clk', ctypes.c_uint32),
  ('max_mem_clk', ctypes.c_uint32),
  ('min_eng_clk', ctypes.c_uint32),
  ('max_eng_clk', ctypes.c_uint32),
  ('min_bus_bandwidth', ctypes.c_uint32),
  ('max_bus_bandwidth', ctypes.c_uint32),
]
class struct_smu_bios_boot_up_values(Struct): pass
struct_smu_bios_boot_up_values._fields_ = [
  ('revision', ctypes.c_uint32),
  ('gfxclk', ctypes.c_uint32),
  ('uclk', ctypes.c_uint32),
  ('socclk', ctypes.c_uint32),
  ('dcefclk', ctypes.c_uint32),
  ('eclk', ctypes.c_uint32),
  ('vclk', ctypes.c_uint32),
  ('dclk', ctypes.c_uint32),
  ('vddc', ctypes.c_uint16),
  ('vddci', ctypes.c_uint16),
  ('mvddc', ctypes.c_uint16),
  ('vdd_gfx', ctypes.c_uint16),
  ('cooling_id', ctypes.c_ubyte),
  ('pp_table_id', ctypes.c_uint32),
  ('format_revision', ctypes.c_uint32),
  ('content_revision', ctypes.c_uint32),
  ('fclk', ctypes.c_uint32),
  ('lclk', ctypes.c_uint32),
  ('firmware_caps', ctypes.c_uint32),
]
enum_smu_table_id = CEnum(ctypes.c_uint32)
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
int32_t = int
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