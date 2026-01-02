# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
FEATURE_PWR_DOMAIN_e = CEnum(ctypes.c_uint32)
FEATURE_PWR_ALL = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_ALL', 0)
FEATURE_PWR_S5 = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_S5', 1)
FEATURE_PWR_BACO = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_BACO', 2)
FEATURE_PWR_SOC = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_SOC', 3)
FEATURE_PWR_GFX = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_GFX', 4)
FEATURE_PWR_DOMAIN_COUNT = FEATURE_PWR_DOMAIN_e.define('FEATURE_PWR_DOMAIN_COUNT', 5)

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
uint32_t = ctypes.c_uint32
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
PPCLK_DCLK_1 = PPCLK_e.define('PPCLK_DCLK_1', 6)
PPCLK_VCLK_1 = PPCLK_e.define('PPCLK_VCLK_1', 7)
PPCLK_DISPCLK = PPCLK_e.define('PPCLK_DISPCLK', 8)
PPCLK_DPPCLK = PPCLK_e.define('PPCLK_DPPCLK', 9)
PPCLK_DPREFCLK = PPCLK_e.define('PPCLK_DPREFCLK', 10)
PPCLK_DCFCLK = PPCLK_e.define('PPCLK_DCFCLK', 11)
PPCLK_DTBCLK = PPCLK_e.define('PPCLK_DTBCLK', 12)
PPCLK_COUNT = PPCLK_e.define('PPCLK_COUNT', 13)

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
AVFS_D_M_B = AVFS_D_e.define('AVFS_D_M_B', 1)
AVFS_D_M_S = AVFS_D_e.define('AVFS_D_M_S', 2)
AVFS_D_COUNT = AVFS_D_e.define('AVFS_D_COUNT', 3)

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
TEMP_HOTSPOT_G = TEMP_e.define('TEMP_HOTSPOT_G', 2)
TEMP_HOTSPOT_M = TEMP_e.define('TEMP_HOTSPOT_M', 3)
TEMP_MEM = TEMP_e.define('TEMP_MEM', 4)
TEMP_VR_GFX = TEMP_e.define('TEMP_VR_GFX', 5)
TEMP_VR_MEM0 = TEMP_e.define('TEMP_VR_MEM0', 6)
TEMP_VR_MEM1 = TEMP_e.define('TEMP_VR_MEM1', 7)
TEMP_VR_SOC = TEMP_e.define('TEMP_VR_SOC', 8)
TEMP_VR_U = TEMP_e.define('TEMP_VR_U', 9)
TEMP_LIQUID0 = TEMP_e.define('TEMP_LIQUID0', 10)
TEMP_LIQUID1 = TEMP_e.define('TEMP_LIQUID1', 11)
TEMP_PLX = TEMP_e.define('TEMP_PLX', 12)
TEMP_COUNT = TEMP_e.define('TEMP_COUNT', 13)

TDC_THROTTLER_e = CEnum(ctypes.c_uint32)
TDC_THROTTLER_GFX = TDC_THROTTLER_e.define('TDC_THROTTLER_GFX', 0)
TDC_THROTTLER_SOC = TDC_THROTTLER_e.define('TDC_THROTTLER_SOC', 1)
TDC_THROTTLER_U = TDC_THROTTLER_e.define('TDC_THROTTLER_U', 2)
TDC_THROTTLER_COUNT = TDC_THROTTLER_e.define('TDC_THROTTLER_COUNT', 3)

SVI_PLANE_e = CEnum(ctypes.c_uint32)
SVI_PLANE_GFX = SVI_PLANE_e.define('SVI_PLANE_GFX', 0)
SVI_PLANE_SOC = SVI_PLANE_e.define('SVI_PLANE_SOC', 1)
SVI_PLANE_VMEMP = SVI_PLANE_e.define('SVI_PLANE_VMEMP', 2)
SVI_PLANE_VDDIO_MEM = SVI_PLANE_e.define('SVI_PLANE_VDDIO_MEM', 3)
SVI_PLANE_U = SVI_PLANE_e.define('SVI_PLANE_U', 4)
SVI_PLANE_COUNT = SVI_PLANE_e.define('SVI_PLANE_COUNT', 5)

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
FanMode_e = CEnum(ctypes.c_uint32)
FAN_MODE_AUTO = FanMode_e.define('FAN_MODE_AUTO', 0)
FAN_MODE_MANUAL_LINEAR = FanMode_e.define('FAN_MODE_MANUAL_LINEAR', 1)

class OverDriveTable_t(Struct): pass
int16_t = ctypes.c_int16
OverDriveTable_t._fields_ = [
  ('FeatureCtrlMask', uint32_t),
  ('VoltageOffsetPerZoneBoundary', (int16_t * 6)),
  ('Reserved', uint32_t),
  ('GfxclkFmin', int16_t),
  ('GfxclkFmax', int16_t),
  ('UclkFmin', uint16_t),
  ('UclkFmax', uint16_t),
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
  ('Spare', (uint32_t * 13)),
  ('MmHubPadding', (uint32_t * 8)),
]
class OverDriveTableExternal_t(Struct): pass
OverDriveTableExternal_t._fields_ = [
  ('OverDriveTable', OverDriveTable_t),
]
class OverDriveLimits_t(Struct): pass
OverDriveLimits_t._fields_ = [
  ('FeatureCtrlMask', uint32_t),
  ('VoltageOffsetPerZoneBoundary', int16_t),
  ('Reserved1', uint16_t),
  ('Reserved2', uint16_t),
  ('GfxclkFmin', int16_t),
  ('GfxclkFmax', int16_t),
  ('UclkFmin', uint16_t),
  ('UclkFmax', uint16_t),
  ('Ppt', int16_t),
  ('Tdc', int16_t),
  ('FanLinearPwmPoints', uint8_t),
  ('FanLinearTempPoints', uint8_t),
  ('FanMinimumPwm', uint16_t),
  ('AcousticTargetRpmThreshold', uint16_t),
  ('AcousticLimitRpmThreshold', uint16_t),
  ('FanTargetTemperature', uint16_t),
  ('FanZeroRpmEnable', uint8_t),
  ('FanZeroRpmStopTemp', uint8_t),
  ('FanMode', uint8_t),
  ('MaxOpTemp', uint8_t),
  ('Spare', (uint32_t * 13)),
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

class BootValues_t(Struct): pass
BootValues_t._fields_ = [
  ('InitGfxclk_bypass', uint16_t),
  ('InitSocclk', uint16_t),
  ('InitMp0clk', uint16_t),
  ('InitMpioclk', uint16_t),
  ('InitSmnclk', uint16_t),
  ('InitUcpclk', uint16_t),
  ('InitCsrclk', uint16_t),
  ('InitDprefclk', uint16_t),
  ('InitDcfclk', uint16_t),
  ('InitDtbclk', uint16_t),
  ('InitDclk', uint16_t),
  ('InitVclk', uint16_t),
  ('InitUsbdfsclk', uint16_t),
  ('InitMp1clk', uint16_t),
  ('InitLclk', uint16_t),
  ('InitBaco400clk_bypass', uint16_t),
  ('InitBaco1200clk_bypass', uint16_t),
  ('InitBaco700clk_bypass', uint16_t),
  ('InitFclk', uint16_t),
  ('InitGfxclk_clkb', uint16_t),
  ('InitUclkDPMState', uint8_t),
  ('Padding', (uint8_t * 3)),
  ('InitVcoFreqPll0', uint32_t),
  ('InitVcoFreqPll1', uint32_t),
  ('InitVcoFreqPll2', uint32_t),
  ('InitVcoFreqPll3', uint32_t),
  ('InitVcoFreqPll4', uint32_t),
  ('InitVcoFreqPll5', uint32_t),
  ('InitVcoFreqPll6', uint32_t),
  ('InitGfx', uint16_t),
  ('InitSoc', uint16_t),
  ('InitU', uint16_t),
  ('Padding2', uint16_t),
  ('Spare', (uint32_t * 8)),
]
class MsgLimits_t(Struct): pass
MsgLimits_t._fields_ = [
  ('Power', ((uint16_t * 2) * 4)),
  ('Tdc', (uint16_t * 3)),
  ('Temperature', (uint16_t * 13)),
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
  ('Reserved', (uint32_t * 4)),
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
class SkuTable_t(Struct): pass
int32_t = ctypes.c_int32
SkuTable_t._fields_ = [
  ('Version', uint32_t),
  ('FeaturesToRun', (uint32_t * 2)),
  ('TotalPowerConfig', uint8_t),
  ('CustomerVariant', uint8_t),
  ('MemoryTemperatureTypeMask', uint8_t),
  ('SmartShiftVersion', uint8_t),
  ('SocketPowerLimitAc', (uint16_t * 4)),
  ('SocketPowerLimitDc', (uint16_t * 4)),
  ('SocketPowerLimitSmartShift2', uint16_t),
  ('EnableLegacyPptLimit', uint8_t),
  ('UseInputTelemetry', uint8_t),
  ('SmartShiftMinReportedPptinDcs', uint8_t),
  ('PaddingPpt', (uint8_t * 1)),
  ('VrTdcLimit', (uint16_t * 3)),
  ('PlatformTdcLimit', (uint16_t * 3)),
  ('TemperatureLimit', (uint16_t * 13)),
  ('HwCtfTempLimit', uint16_t),
  ('PaddingInfra', uint16_t),
  ('FitControllerFailureRateLimit', uint32_t),
  ('FitControllerGfxDutyCycle', uint32_t),
  ('FitControllerSocDutyCycle', uint32_t),
  ('FitControllerSocOffset', uint32_t),
  ('GfxApccPlusResidencyLimit', uint32_t),
  ('ThrottlerControlMask', uint32_t),
  ('FwDStateMask', uint32_t),
  ('UlvVoltageOffset', (uint16_t * 2)),
  ('UlvVoltageOffsetU', uint16_t),
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
  ('Vmin_droop', QuadraticInt_t),
  ('SpareVmin', (uint32_t * 9)),
  ('DpmDescriptor', (DpmDescriptor_t * 13)),
  ('FreqTableGfx', (uint16_t * 16)),
  ('FreqTableVclk', (uint16_t * 8)),
  ('FreqTableDclk', (uint16_t * 8)),
  ('FreqTableSocclk', (uint16_t * 8)),
  ('FreqTableUclk', (uint16_t * 4)),
  ('FreqTableDispclk', (uint16_t * 8)),
  ('FreqTableDppClk', (uint16_t * 8)),
  ('FreqTableDprefclk', (uint16_t * 8)),
  ('FreqTableDcfclk', (uint16_t * 8)),
  ('FreqTableDtbclk', (uint16_t * 8)),
  ('FreqTableFclk', (uint16_t * 8)),
  ('DcModeMaxFreq', (uint32_t * 13)),
  ('Mp0clkFreq', (uint16_t * 2)),
  ('Mp0DpmVoltage', (uint16_t * 2)),
  ('GfxclkSpare', (uint8_t * 2)),
  ('GfxclkFreqCap', uint16_t),
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
  ('DfllBtcMasterScalerM', uint32_t),
  ('DfllBtcMasterScalerB', int32_t),
  ('DfllBtcSlaveScalerM', uint32_t),
  ('DfllBtcSlaveScalerB', int32_t),
  ('DfllPccAsWaitCtrl', uint32_t),
  ('DfllPccAsStepCtrl', uint32_t),
  ('DfllL2FrequencyBoostM', uint32_t),
  ('DfllL2FrequencyBoostB', uint32_t),
  ('GfxGpoSpare', (uint32_t * 8)),
  ('DcsGfxOffVoltage', uint16_t),
  ('PaddingDcs', uint16_t),
  ('DcsMinGfxOffTime', uint16_t),
  ('DcsMaxGfxOffTime', uint16_t),
  ('DcsMinCreditAccum', uint32_t),
  ('DcsExitHysteresis', uint16_t),
  ('DcsTimeout', uint16_t),
  ('FoptEnabled', uint8_t),
  ('DcsSpare2', (uint8_t * 3)),
  ('DcsFoptM', uint32_t),
  ('DcsFoptB', uint32_t),
  ('DcsSpare', (uint32_t * 11)),
  ('ShadowFreqTableUclk', (uint16_t * 4)),
  ('UseStrobeModeOptimizations', uint8_t),
  ('PaddingMem', (uint8_t * 3)),
  ('UclkDpmPstates', (uint8_t * 4)),
  ('FreqTableUclkDiv', (uint8_t * 4)),
  ('MemVmempVoltage', (uint16_t * 4)),
  ('MemVddioVoltage', (uint16_t * 4)),
  ('FclkDpmUPstates', (uint8_t * 8)),
  ('FclkDpmVddU', (uint16_t * 8)),
  ('FclkDpmUSpeed', (uint16_t * 8)),
  ('FclkDpmDisallowPstateFreq', uint16_t),
  ('PaddingFclk', uint16_t),
  ('PcieGenSpeed', (uint8_t * 3)),
  ('PcieLaneCount', (uint8_t * 3)),
  ('LclkFreq', (uint16_t * 3)),
  ('FanStopTemp', (uint16_t * 13)),
  ('FanStartTemp', (uint16_t * 13)),
  ('FanGain', (uint16_t * 13)),
  ('FanGainPadding', uint16_t),
  ('FanPwmMin', uint16_t),
  ('AcousticTargetRpmThreshold', uint16_t),
  ('AcousticLimitRpmThreshold', uint16_t),
  ('FanMaximumRpm', uint16_t),
  ('MGpuAcousticLimitRpmThreshold', uint16_t),
  ('FanTargetGfxclk', uint16_t),
  ('TempInputSelectMask', uint32_t),
  ('FanZeroRpmEnable', uint8_t),
  ('FanTachEdgePerRev', uint8_t),
  ('FanTargetTemperature', (uint16_t * 13)),
  ('FuzzyFan_ErrorSetDelta', int16_t),
  ('FuzzyFan_ErrorRateSetDelta', int16_t),
  ('FuzzyFan_PwmSetDelta', int16_t),
  ('FuzzyFan_Reserved', uint16_t),
  ('FwCtfLimit', (uint16_t * 13)),
  ('IntakeTempEnableRPM', uint16_t),
  ('IntakeTempOffsetTemp', int16_t),
  ('IntakeTempReleaseTemp', uint16_t),
  ('IntakeTempHighIntakeAcousticLimit', uint16_t),
  ('IntakeTempAcouticLimitReleaseRate', uint16_t),
  ('FanAbnormalTempLimitOffset', int16_t),
  ('FanStalledTriggerRpm', uint16_t),
  ('FanAbnormalTriggerRpmCoeff', uint16_t),
  ('FanAbnormalDetectionEnable', uint16_t),
  ('FanIntakeSensorSupport', uint8_t),
  ('FanIntakePadding', (uint8_t * 3)),
  ('FanSpare', (uint32_t * 13)),
  ('OverrideGfxAvfsFuses', uint8_t),
  ('GfxAvfsPadding', (uint8_t * 3)),
  ('L2HwRtAvfsFuses', (uint32_t * 32)),
  ('SeHwRtAvfsFuses', (uint32_t * 32)),
  ('CommonRtAvfs', (uint32_t * 13)),
  ('L2FwRtAvfsFuses', (uint32_t * 19)),
  ('SeFwRtAvfsFuses', (uint32_t * 19)),
  ('Droop_PWL_F', (uint32_t * 5)),
  ('Droop_PWL_a', (uint32_t * 5)),
  ('Droop_PWL_b', (uint32_t * 5)),
  ('Droop_PWL_c', (uint32_t * 5)),
  ('Static_PWL_Offset', (uint32_t * 5)),
  ('dGbV_dT_vmin', uint32_t),
  ('dGbV_dT_vmax', uint32_t),
  ('V2F_vmin_range_low', uint32_t),
  ('V2F_vmin_range_high', uint32_t),
  ('V2F_vmax_range_low', uint32_t),
  ('V2F_vmax_range_high', uint32_t),
  ('DcBtcGfxParams', AvfsDcBtcParams_t),
  ('GfxAvfsSpare', (uint32_t * 32)),
  ('OverrideSocAvfsFuses', uint8_t),
  ('MinSocAvfsRevision', uint8_t),
  ('SocAvfsPadding', (uint8_t * 2)),
  ('SocAvfsFuseOverride', (AvfsFuseOverride_t * 3)),
  ('dBtcGbSoc', (DroopInt_t * 3)),
  ('qAgingGb', (LinearInt_t * 3)),
  ('qStaticVoltageOffset', (QuadraticInt_t * 3)),
  ('DcBtcSocParams', (AvfsDcBtcParams_t * 3)),
  ('SocAvfsSpare', (uint32_t * 32)),
  ('BootValues', BootValues_t),
  ('DriverReportedClocks', DriverReportedClocks_t),
  ('MsgLimits', MsgLimits_t),
  ('OverDriveLimitsMin', OverDriveLimits_t),
  ('OverDriveLimitsBasicMax', OverDriveLimits_t),
  ('reserved', (uint32_t * 22)),
  ('DebugOverrides', uint32_t),
  ('TotalBoardPowerSupport', uint8_t),
  ('TotalBoardPowerPadding', (uint8_t * 3)),
  ('TotalIdleBoardPowerM', int16_t),
  ('TotalIdleBoardPowerB', int16_t),
  ('TotalBoardPowerM', int16_t),
  ('TotalBoardPowerB', int16_t),
  ('qFeffCoeffGameClock', (QuadraticInt_t * 2)),
  ('qFeffCoeffBaseClock', (QuadraticInt_t * 2)),
  ('qFeffCoeffBoostClock', (QuadraticInt_t * 2)),
  ('TemperatureLimit_Hynix', uint16_t),
  ('TemperatureLimit_Micron', uint16_t),
  ('TemperatureFwCtfLimit_Hynix', uint16_t),
  ('TemperatureFwCtfLimit_Micron', uint16_t),
  ('Spare', (uint32_t * 41)),
  ('MmHubPadding', (uint32_t * 8)),
]
class BoardTable_t(Struct): pass
BoardTable_t._fields_ = [
  ('Version', uint32_t),
  ('I2cControllers', (I2cControllerConfig_t * 8)),
  ('VddGfxVrMapping', uint8_t),
  ('VddSocVrMapping', uint8_t),
  ('VddMem0VrMapping', uint8_t),
  ('VddMem1VrMapping', uint8_t),
  ('GfxUlvPhaseSheddingMask', uint8_t),
  ('SocUlvPhaseSheddingMask', uint8_t),
  ('VmempUlvPhaseSheddingMask', uint8_t),
  ('VddioUlvPhaseSheddingMask', uint8_t),
  ('SlaveAddrMapping', (uint8_t * 5)),
  ('VrPsiSupport', (uint8_t * 5)),
  ('PaddingPsi', (uint8_t * 5)),
  ('EnablePsi6', (uint8_t * 5)),
  ('SviTelemetryScale', (SviTelemetryScale_t * 5)),
  ('VoltageTelemetryRatio', (uint32_t * 5)),
  ('DownSlewRateVr', (uint8_t * 5)),
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
  ('PostVoltageSetBacoDelay', uint32_t),
  ('BacoEntryDelay', uint32_t),
  ('FuseWritePowerMuxPresent', uint8_t),
  ('FuseWritePadding', (uint8_t * 3)),
  ('BoardSpare', (uint32_t * 63)),
  ('MmHubPadding', (uint32_t * 8)),
]
class PPTable_t(Struct): pass
PPTable_t._packed_ = True
PPTable_t._fields_ = [
  ('SkuTable', SkuTable_t),
  ('BoardTable', BoardTable_t),
]
class DriverSmuConfig_t(Struct): pass
DriverSmuConfig_t._fields_ = [
  ('GfxclkAverageLpfTau', uint16_t),
  ('FclkAverageLpfTau', uint16_t),
  ('UclkAverageLpfTau', uint16_t),
  ('GfxActivityLpfTau', uint16_t),
  ('UclkActivityLpfTau', uint16_t),
  ('SocketPowerLpfTau', uint16_t),
  ('VcnClkAverageLpfTau', uint16_t),
  ('VcnUsageAverageLpfTau', uint16_t),
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
  ('FreqTableUclk', (uint16_t * 4)),
  ('FreqTableDispclk', (uint16_t * 8)),
  ('FreqTableDppClk', (uint16_t * 8)),
  ('FreqTableDprefclk', (uint16_t * 8)),
  ('FreqTableDcfclk', (uint16_t * 8)),
  ('FreqTableDtbclk', (uint16_t * 8)),
  ('FreqTableFclk', (uint16_t * 8)),
  ('DcModeMaxFreq', (uint16_t * 13)),
  ('Padding', uint16_t),
  ('Spare', (uint32_t * 32)),
  ('MmHubPadding', (uint32_t * 8)),
]
class SmuMetrics_t(Struct): pass
SmuMetrics_t._fields_ = [
  ('CurrClock', (uint32_t * 13)),
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
  ('PCIeBusy', uint16_t),
  ('dGPU_W_MAX', uint16_t),
  ('padding', uint16_t),
  ('MetricsCounter', uint32_t),
  ('AvgVoltage', (uint16_t * 5)),
  ('AvgCurrent', (uint16_t * 5)),
  ('AverageGfxActivity', uint16_t),
  ('AverageUclkActivity', uint16_t),
  ('Vcn0ActivityPercentage', uint16_t),
  ('Vcn1ActivityPercentage', uint16_t),
  ('EnergyAccumulator', uint32_t),
  ('AverageSocketPower', uint16_t),
  ('AverageTotalBoardPower', uint16_t),
  ('AvgTemperature', (uint16_t * 13)),
  ('AvgTemperatureFanIntake', uint16_t),
  ('PcieRate', uint8_t),
  ('PcieWidth', uint8_t),
  ('AvgFanPwm', uint8_t),
  ('Padding', (uint8_t * 1)),
  ('AvgFanRpm', uint16_t),
  ('ThrottlingPercentage', (uint8_t * 22)),
  ('VmaxThrottlingPercentage', uint8_t),
  ('Padding1', (uint8_t * 3)),
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
  ('Spare', (uint32_t * 29)),
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
  ('avgPsmCount', (uint16_t * 214)),
  ('minPsmCount', (uint16_t * 214)),
  ('avgPsmVoltage', (ctypes.c_float * 214)),
  ('minPsmVoltage', (ctypes.c_float * 214)),
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
  ('Mem_UpThreshold_Limit', (uint32_t * 4)),
  ('Mem_UpHystLimit', (uint8_t * 4)),
  ('Mem_DownHystLimit', (uint8_t * 4)),
  ('Mem_Fps', uint16_t),
  ('padding', (uint8_t * 2)),
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

PPSMC_VERSION = 0x1
DEBUGSMC_VERSION = 0x1
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
PPSMC_MSG_Mode1Reset = 0x2F
PPSMC_MSG_Mode2Reset = 0x4F
PPSMC_MSG_SetSystemVirtualDramAddrHigh = 0x30
PPSMC_MSG_SetSystemVirtualDramAddrLow = 0x31
PPSMC_MSG_SetPptLimit = 0x32
PPSMC_MSG_GetPptLimit = 0x33
PPSMC_MSG_ReenableAcDcInterrupt = 0x34
PPSMC_MSG_NotifyPowerSource = 0x35
PPSMC_MSG_RunDcBtc = 0x36
PPSMC_MSG_GetDebugData = 0x37
PPSMC_MSG_SetTemperatureInputSelect = 0x38
PPSMC_MSG_SetFwDstatesMask = 0x39
PPSMC_MSG_SetThrottlerMask = 0x3A
PPSMC_MSG_SetExternalClientDfCstateAllow = 0x3B
PPSMC_MSG_SetMGpuFanBoostLimitRpm = 0x3C
PPSMC_MSG_DumpSTBtoDram = 0x3D
PPSMC_MSG_STBtoDramLogSetDramAddrHigh = 0x3E
PPSMC_MSG_STBtoDramLogSetDramAddrLow = 0x3F
PPSMC_MSG_STBtoDramLogSetDramSize = 0x40
PPSMC_MSG_SetGpoAllow = 0x41
PPSMC_MSG_AllowGfxDcs = 0x42
PPSMC_MSG_DisallowGfxDcs = 0x43
PPSMC_MSG_EnableAudioStutterWA = 0x44
PPSMC_MSG_PowerUpUmsch = 0x45
PPSMC_MSG_PowerDownUmsch = 0x46
PPSMC_MSG_SetDcsArch = 0x47
PPSMC_MSG_TriggerVFFLR = 0x48
PPSMC_MSG_SetNumBadMemoryPagesRetired = 0x49
PPSMC_MSG_SetBadMemoryPagesRetiredFlagsPerChannel = 0x4A
PPSMC_MSG_SetPriorityDeltaGain = 0x4B
PPSMC_MSG_AllowIHHostInterrupt = 0x4C
PPSMC_MSG_DALNotPresent = 0x4E
PPSMC_MSG_EnableUCLKShadow = 0x51
PPSMC_Message_Count = 0x52
DEBUGSMC_MSG_TestMessage = 0x1
DEBUGSMC_MSG_GetDebugData = 0x2
DEBUGSMC_MSG_DebugDumpExit = 0x3
DEBUGSMC_Message_Count = 0x4
SMU13_0_0_DRIVER_IF_VERSION = 0x3D
PPTABLE_VERSION = 0x2B
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
NUM_UCLK_DPM_LEVELS = 4
NUM_LINK_LEVELS = 3
NUM_FCLK_DPM_LEVELS = 8
NUM_OD_FAN_MAX_POINTS = 6
FEATURE_FW_DATA_READ_BIT = 0
FEATURE_DPM_GFXCLK_BIT = 1
FEATURE_DPM_GFX_POWER_OPTIMIZER_BIT = 2
FEATURE_DPM_UCLK_BIT = 3
FEATURE_DPM_FCLK_BIT = 4
FEATURE_DPM_SOCCLK_BIT = 5
FEATURE_DPM_MP0CLK_BIT = 6
FEATURE_DPM_LINK_BIT = 7
FEATURE_DPM_DCN_BIT = 8
FEATURE_VMEMP_SCALING_BIT = 9
FEATURE_VDDIO_MEM_SCALING_BIT = 10
FEATURE_DS_GFXCLK_BIT = 11
FEATURE_DS_SOCCLK_BIT = 12
FEATURE_DS_FCLK_BIT = 13
FEATURE_DS_LCLK_BIT = 14
FEATURE_DS_DCFCLK_BIT = 15
FEATURE_DS_UCLK_BIT = 16
FEATURE_GFX_ULV_BIT = 17
FEATURE_FW_DSTATE_BIT = 18
FEATURE_GFXOFF_BIT = 19
FEATURE_BACO_BIT = 20
FEATURE_MM_DPM_BIT = 21
FEATURE_SOC_MPCLK_DS_BIT = 22
FEATURE_BACO_MPCLK_DS_BIT = 23
FEATURE_THROTTLERS_BIT = 24
FEATURE_SMARTSHIFT_BIT = 25
FEATURE_GTHR_BIT = 26
FEATURE_ACDC_BIT = 27
FEATURE_VR0HOT_BIT = 28
FEATURE_FW_CTF_BIT = 29
FEATURE_FAN_CONTROL_BIT = 30
FEATURE_GFX_DCS_BIT = 31
FEATURE_GFX_READ_MARGIN_BIT = 32
FEATURE_LED_DISPLAY_BIT = 33
FEATURE_GFXCLK_SPREAD_SPECTRUM_BIT = 34
FEATURE_OUT_OF_BAND_MONITOR_BIT = 35
FEATURE_OPTIMIZED_VMIN_BIT = 36
FEATURE_GFX_IMU_BIT = 37
FEATURE_BOOT_TIME_CAL_BIT = 38
FEATURE_GFX_PCC_DFLL_BIT = 39
FEATURE_SOC_CG_BIT = 40
FEATURE_DF_CSTATE_BIT = 41
FEATURE_GFX_EDC_BIT = 42
FEATURE_BOOT_POWER_OPT_BIT = 43
FEATURE_CLOCK_POWER_DOWN_BYPASS_BIT = 44
FEATURE_DS_VCN_BIT = 45
FEATURE_BACO_CG_BIT = 46
FEATURE_MEM_TEMP_READ_BIT = 47
FEATURE_ATHUB_MMHUB_PG_BIT = 48
FEATURE_SOC_PCC_BIT = 49
FEATURE_EDC_PWRBRK_BIT = 50
FEATURE_BOMXCO_SVI3_PROG_BIT = 51
FEATURE_SPARE_52_BIT = 52
FEATURE_SPARE_53_BIT = 53
FEATURE_SPARE_54_BIT = 54
FEATURE_SPARE_55_BIT = 55
FEATURE_SPARE_56_BIT = 56
FEATURE_SPARE_57_BIT = 57
FEATURE_SPARE_58_BIT = 58
FEATURE_SPARE_59_BIT = 59
FEATURE_SPARE_60_BIT = 60
FEATURE_SPARE_61_BIT = 61
FEATURE_SPARE_62_BIT = 62
FEATURE_SPARE_63_BIT = 63
NUM_FEATURES = 64
ALLOWED_FEATURE_CTRL_DEFAULT = 0xFFFFFFFFFFFFFFFF
ALLOWED_FEATURE_CTRL_SCPM = ((1 << FEATURE_DPM_GFXCLK_BIT) | (1 << FEATURE_DPM_GFX_POWER_OPTIMIZER_BIT) | (1 << FEATURE_DPM_UCLK_BIT) | (1 << FEATURE_DPM_FCLK_BIT) | (1 << FEATURE_DPM_SOCCLK_BIT) | (1 << FEATURE_DPM_MP0CLK_BIT) | (1 << FEATURE_DPM_LINK_BIT) | (1 << FEATURE_DPM_DCN_BIT) | (1 << FEATURE_DS_GFXCLK_BIT) | (1 << FEATURE_DS_SOCCLK_BIT) | (1 << FEATURE_DS_FCLK_BIT) | (1 << FEATURE_DS_LCLK_BIT) | (1 << FEATURE_DS_DCFCLK_BIT) | (1 << FEATURE_DS_UCLK_BIT) | (1 << FEATURE_DS_VCN_BIT))
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_VCN_FCLK = 0x00000001
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
THROTTLER_TEMP_HOTSPOT_G_BIT = 2
THROTTLER_TEMP_HOTSPOT_M_BIT = 3
THROTTLER_TEMP_MEM_BIT = 4
THROTTLER_TEMP_VR_GFX_BIT = 5
THROTTLER_TEMP_VR_MEM0_BIT = 6
THROTTLER_TEMP_VR_MEM1_BIT = 7
THROTTLER_TEMP_VR_SOC_BIT = 8
THROTTLER_TEMP_VR_U_BIT = 9
THROTTLER_TEMP_LIQUID0_BIT = 10
THROTTLER_TEMP_LIQUID1_BIT = 11
THROTTLER_TEMP_PLX_BIT = 12
THROTTLER_TDC_GFX_BIT = 13
THROTTLER_TDC_SOC_BIT = 14
THROTTLER_TDC_U_BIT = 15
THROTTLER_PPT0_BIT = 16
THROTTLER_PPT1_BIT = 17
THROTTLER_PPT2_BIT = 18
THROTTLER_PPT3_BIT = 19
THROTTLER_FIT_BIT = 20
THROTTLER_GFX_APCC_PLUS_BIT = 21
THROTTLER_COUNT = 22
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
FW_DSTATE_U_ULV_BIT = 12
FW_DSTATE_MALL_FLUSH_BIT = 13
FW_DSTATE_SOC_PSI_BIT = 14
FW_DSTATE_U_PSI_BIT = 15
FW_DSTATE_UCP_DS_BIT = 16
FW_DSTATE_CSRCLK_DS_BIT = 17
FW_DSTATE_MMHUB_INTERLOCK_BIT = 18
FW_DSTATE_D0i3_2_QUIET_FW_BIT = 19
FW_DSTATE_CLDO_PRG_BIT = 20
FW_DSTATE_DF_PLL_PWRDN_BIT = 21
FW_DSTATE_U_LOW_PWR_MODE_EN_BIT = 22
FW_DSTATE_GFX_PSI6_BIT = 23
FW_DSTATE_GFX_VR_PWR_STAGE_BIT = 24
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
PP_NUM_RTAVFS_PWL_ZONES = 5
PP_OD_FEATURE_GFX_VF_CURVE_BIT = 0
PP_OD_FEATURE_PPT_BIT = 2
PP_OD_FEATURE_FAN_CURVE_BIT = 3
PP_OD_FEATURE_GFXCLK_BIT = 7
PP_OD_FEATURE_UCLK_BIT = 8
PP_OD_FEATURE_ZERO_FAN_BIT = 9
PP_OD_FEATURE_TEMPERATURE_BIT = 10
PP_OD_FEATURE_COUNT = 13
PP_NUM_OD_VF_CURVE_POINTS = PP_NUM_RTAVFS_PWL_ZONES + 1
INVALID_BOARD_GPIO = 0xFF
MARKETING_BASE_CLOCKS = 0
MARKETING_GAME_CLOCKS = 1
MARKETING_BOOST_CLOCKS = 2
NUM_WM_RANGES = 4
WORKLOAD_PPLIB_DEFAULT_BIT = 0
WORKLOAD_PPLIB_FULL_SCREEN_3D_BIT = 1
WORKLOAD_PPLIB_POWER_SAVING_BIT = 2
WORKLOAD_PPLIB_VIDEO_BIT = 3
WORKLOAD_PPLIB_VR_BIT = 4
WORKLOAD_PPLIB_COMPUTE_BIT = 5
WORKLOAD_PPLIB_CUSTOM_BIT = 6
WORKLOAD_PPLIB_WINDOW_3D_BIT = 7
WORKLOAD_PPLIB_COUNT = 8
TABLE_TRANSFER_OK = 0x0
TABLE_TRANSFER_FAILED = 0xFF
TABLE_TRANSFER_PENDING = 0xAB
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
TABLE_WIFIBAND = 12
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