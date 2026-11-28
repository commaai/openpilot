# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
FEATURE_PWR_DOMAIN_e: dict[int, str] = {(FEATURE_PWR_ALL:=0): 'FEATURE_PWR_ALL', (FEATURE_PWR_S5:=1): 'FEATURE_PWR_S5', (FEATURE_PWR_BACO:=2): 'FEATURE_PWR_BACO', (FEATURE_PWR_SOC:=3): 'FEATURE_PWR_SOC', (FEATURE_PWR_GFX:=4): 'FEATURE_PWR_GFX', (FEATURE_PWR_DOMAIN_COUNT:=5): 'FEATURE_PWR_DOMAIN_COUNT'}
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
uint32_t: TypeAlias = ctypes.c_uint32
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
PPCLK_e: dict[int, str] = {(PPCLK_GFXCLK:=0): 'PPCLK_GFXCLK', (PPCLK_SOCCLK:=1): 'PPCLK_SOCCLK', (PPCLK_UCLK:=2): 'PPCLK_UCLK', (PPCLK_FCLK:=3): 'PPCLK_FCLK', (PPCLK_DCLK_0:=4): 'PPCLK_DCLK_0', (PPCLK_VCLK_0:=5): 'PPCLK_VCLK_0', (PPCLK_DCLK_1:=6): 'PPCLK_DCLK_1', (PPCLK_VCLK_1:=7): 'PPCLK_VCLK_1', (PPCLK_DISPCLK:=8): 'PPCLK_DISPCLK', (PPCLK_DPPCLK:=9): 'PPCLK_DPPCLK', (PPCLK_DPREFCLK:=10): 'PPCLK_DPREFCLK', (PPCLK_DCFCLK:=11): 'PPCLK_DCFCLK', (PPCLK_DTBCLK:=12): 'PPCLK_DTBCLK', (PPCLK_COUNT:=13): 'PPCLK_COUNT'}
VOLTAGE_MODE_e: dict[int, str] = {(VOLTAGE_MODE_PPTABLE:=0): 'VOLTAGE_MODE_PPTABLE', (VOLTAGE_MODE_FUSES:=1): 'VOLTAGE_MODE_FUSES', (VOLTAGE_MODE_COUNT:=2): 'VOLTAGE_MODE_COUNT'}
AVFS_VOLTAGE_TYPE_e: dict[int, str] = {(AVFS_VOLTAGE_GFX:=0): 'AVFS_VOLTAGE_GFX', (AVFS_VOLTAGE_SOC:=1): 'AVFS_VOLTAGE_SOC', (AVFS_VOLTAGE_COUNT:=2): 'AVFS_VOLTAGE_COUNT'}
AVFS_TEMP_e: dict[int, str] = {(AVFS_TEMP_COLD:=0): 'AVFS_TEMP_COLD', (AVFS_TEMP_HOT:=1): 'AVFS_TEMP_HOT', (AVFS_TEMP_COUNT:=2): 'AVFS_TEMP_COUNT'}
AVFS_D_e: dict[int, str] = {(AVFS_D_G:=0): 'AVFS_D_G', (AVFS_D_M_B:=1): 'AVFS_D_M_B', (AVFS_D_M_S:=2): 'AVFS_D_M_S', (AVFS_D_COUNT:=3): 'AVFS_D_COUNT'}
UCLK_DIV_e: dict[int, str] = {(UCLK_DIV_BY_1:=0): 'UCLK_DIV_BY_1', (UCLK_DIV_BY_2:=1): 'UCLK_DIV_BY_2', (UCLK_DIV_BY_4:=2): 'UCLK_DIV_BY_4', (UCLK_DIV_BY_8:=3): 'UCLK_DIV_BY_8'}
GpioIntPolarity_e: dict[int, str] = {(GPIO_INT_POLARITY_ACTIVE_LOW:=0): 'GPIO_INT_POLARITY_ACTIVE_LOW', (GPIO_INT_POLARITY_ACTIVE_HIGH:=1): 'GPIO_INT_POLARITY_ACTIVE_HIGH'}
PwrConfig_e: dict[int, str] = {(PWR_CONFIG_TDP:=0): 'PWR_CONFIG_TDP', (PWR_CONFIG_TGP:=1): 'PWR_CONFIG_TGP', (PWR_CONFIG_TCP_ESTIMATED:=2): 'PWR_CONFIG_TCP_ESTIMATED', (PWR_CONFIG_TCP_MEASURED:=3): 'PWR_CONFIG_TCP_MEASURED'}
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
TEMP_e: dict[int, str] = {(TEMP_EDGE:=0): 'TEMP_EDGE', (TEMP_HOTSPOT:=1): 'TEMP_HOTSPOT', (TEMP_HOTSPOT_G:=2): 'TEMP_HOTSPOT_G', (TEMP_HOTSPOT_M:=3): 'TEMP_HOTSPOT_M', (TEMP_MEM:=4): 'TEMP_MEM', (TEMP_VR_GFX:=5): 'TEMP_VR_GFX', (TEMP_VR_MEM0:=6): 'TEMP_VR_MEM0', (TEMP_VR_MEM1:=7): 'TEMP_VR_MEM1', (TEMP_VR_SOC:=8): 'TEMP_VR_SOC', (TEMP_VR_U:=9): 'TEMP_VR_U', (TEMP_LIQUID0:=10): 'TEMP_LIQUID0', (TEMP_LIQUID1:=11): 'TEMP_LIQUID1', (TEMP_PLX:=12): 'TEMP_PLX', (TEMP_COUNT:=13): 'TEMP_COUNT'}
TDC_THROTTLER_e: dict[int, str] = {(TDC_THROTTLER_GFX:=0): 'TDC_THROTTLER_GFX', (TDC_THROTTLER_SOC:=1): 'TDC_THROTTLER_SOC', (TDC_THROTTLER_U:=2): 'TDC_THROTTLER_U', (TDC_THROTTLER_COUNT:=3): 'TDC_THROTTLER_COUNT'}
SVI_PLANE_e: dict[int, str] = {(SVI_PLANE_GFX:=0): 'SVI_PLANE_GFX', (SVI_PLANE_SOC:=1): 'SVI_PLANE_SOC', (SVI_PLANE_VMEMP:=2): 'SVI_PLANE_VMEMP', (SVI_PLANE_VDDIO_MEM:=3): 'SVI_PLANE_VDDIO_MEM', (SVI_PLANE_U:=4): 'SVI_PLANE_U', (SVI_PLANE_COUNT:=5): 'SVI_PLANE_COUNT'}
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
FanMode_e: dict[int, str] = {(FAN_MODE_AUTO:=0): 'FAN_MODE_AUTO', (FAN_MODE_MANUAL_LINEAR:=1): 'FAN_MODE_MANUAL_LINEAR'}
@c.record
class OverDriveTable_t(c.Struct):
  SIZE = 140
  FeatureCtrlMask: int
  VoltageOffsetPerZoneBoundary: c.Array[ctypes.c_int16, Literal[6]]
  Reserved: int
  GfxclkFmin: int
  GfxclkFmax: int
  UclkFmin: int
  UclkFmax: int
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
  Spare: c.Array[ctypes.c_uint32, Literal[13]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
int16_t: TypeAlias = ctypes.c_int16
OverDriveTable_t.register_fields([('FeatureCtrlMask', uint32_t, 0), ('VoltageOffsetPerZoneBoundary', c.Array[int16_t, Literal[6]], 4), ('Reserved', uint32_t, 16), ('GfxclkFmin', int16_t, 20), ('GfxclkFmax', int16_t, 22), ('UclkFmin', uint16_t, 24), ('UclkFmax', uint16_t, 26), ('Ppt', int16_t, 28), ('Tdc', int16_t, 30), ('FanLinearPwmPoints', c.Array[uint8_t, Literal[6]], 32), ('FanLinearTempPoints', c.Array[uint8_t, Literal[6]], 38), ('FanMinimumPwm', uint16_t, 44), ('AcousticTargetRpmThreshold', uint16_t, 46), ('AcousticLimitRpmThreshold', uint16_t, 48), ('FanTargetTemperature', uint16_t, 50), ('FanZeroRpmEnable', uint8_t, 52), ('FanZeroRpmStopTemp', uint8_t, 53), ('FanMode', uint8_t, 54), ('MaxOpTemp', uint8_t, 55), ('Spare', c.Array[uint32_t, Literal[13]], 56), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 108)])
@c.record
class OverDriveTableExternal_t(c.Struct):
  SIZE = 140
  OverDriveTable: OverDriveTable_t
OverDriveTableExternal_t.register_fields([('OverDriveTable', OverDriveTable_t, 0)])
@c.record
class OverDriveLimits_t(c.Struct):
  SIZE = 88
  FeatureCtrlMask: int
  VoltageOffsetPerZoneBoundary: int
  Reserved1: int
  Reserved2: int
  GfxclkFmin: int
  GfxclkFmax: int
  UclkFmin: int
  UclkFmax: int
  Ppt: int
  Tdc: int
  FanLinearPwmPoints: int
  FanLinearTempPoints: int
  FanMinimumPwm: int
  AcousticTargetRpmThreshold: int
  AcousticLimitRpmThreshold: int
  FanTargetTemperature: int
  FanZeroRpmEnable: int
  FanZeroRpmStopTemp: int
  FanMode: int
  MaxOpTemp: int
  Spare: c.Array[ctypes.c_uint32, Literal[13]]
OverDriveLimits_t.register_fields([('FeatureCtrlMask', uint32_t, 0), ('VoltageOffsetPerZoneBoundary', int16_t, 4), ('Reserved1', uint16_t, 6), ('Reserved2', uint16_t, 8), ('GfxclkFmin', int16_t, 10), ('GfxclkFmax', int16_t, 12), ('UclkFmin', uint16_t, 14), ('UclkFmax', uint16_t, 16), ('Ppt', int16_t, 18), ('Tdc', int16_t, 20), ('FanLinearPwmPoints', uint8_t, 22), ('FanLinearTempPoints', uint8_t, 23), ('FanMinimumPwm', uint16_t, 24), ('AcousticTargetRpmThreshold', uint16_t, 26), ('AcousticLimitRpmThreshold', uint16_t, 28), ('FanTargetTemperature', uint16_t, 30), ('FanZeroRpmEnable', uint8_t, 32), ('FanZeroRpmStopTemp', uint8_t, 33), ('FanMode', uint8_t, 34), ('MaxOpTemp', uint8_t, 35), ('Spare', c.Array[uint32_t, Literal[13]], 36)])
BOARD_GPIO_TYPE_e: dict[int, str] = {(BOARD_GPIO_SMUIO_0:=0): 'BOARD_GPIO_SMUIO_0', (BOARD_GPIO_SMUIO_1:=1): 'BOARD_GPIO_SMUIO_1', (BOARD_GPIO_SMUIO_2:=2): 'BOARD_GPIO_SMUIO_2', (BOARD_GPIO_SMUIO_3:=3): 'BOARD_GPIO_SMUIO_3', (BOARD_GPIO_SMUIO_4:=4): 'BOARD_GPIO_SMUIO_4', (BOARD_GPIO_SMUIO_5:=5): 'BOARD_GPIO_SMUIO_5', (BOARD_GPIO_SMUIO_6:=6): 'BOARD_GPIO_SMUIO_6', (BOARD_GPIO_SMUIO_7:=7): 'BOARD_GPIO_SMUIO_7', (BOARD_GPIO_SMUIO_8:=8): 'BOARD_GPIO_SMUIO_8', (BOARD_GPIO_SMUIO_9:=9): 'BOARD_GPIO_SMUIO_9', (BOARD_GPIO_SMUIO_10:=10): 'BOARD_GPIO_SMUIO_10', (BOARD_GPIO_SMUIO_11:=11): 'BOARD_GPIO_SMUIO_11', (BOARD_GPIO_SMUIO_12:=12): 'BOARD_GPIO_SMUIO_12', (BOARD_GPIO_SMUIO_13:=13): 'BOARD_GPIO_SMUIO_13', (BOARD_GPIO_SMUIO_14:=14): 'BOARD_GPIO_SMUIO_14', (BOARD_GPIO_SMUIO_15:=15): 'BOARD_GPIO_SMUIO_15', (BOARD_GPIO_SMUIO_16:=16): 'BOARD_GPIO_SMUIO_16', (BOARD_GPIO_SMUIO_17:=17): 'BOARD_GPIO_SMUIO_17', (BOARD_GPIO_SMUIO_18:=18): 'BOARD_GPIO_SMUIO_18', (BOARD_GPIO_SMUIO_19:=19): 'BOARD_GPIO_SMUIO_19', (BOARD_GPIO_SMUIO_20:=20): 'BOARD_GPIO_SMUIO_20', (BOARD_GPIO_SMUIO_21:=21): 'BOARD_GPIO_SMUIO_21', (BOARD_GPIO_SMUIO_22:=22): 'BOARD_GPIO_SMUIO_22', (BOARD_GPIO_SMUIO_23:=23): 'BOARD_GPIO_SMUIO_23', (BOARD_GPIO_SMUIO_24:=24): 'BOARD_GPIO_SMUIO_24', (BOARD_GPIO_SMUIO_25:=25): 'BOARD_GPIO_SMUIO_25', (BOARD_GPIO_SMUIO_26:=26): 'BOARD_GPIO_SMUIO_26', (BOARD_GPIO_SMUIO_27:=27): 'BOARD_GPIO_SMUIO_27', (BOARD_GPIO_SMUIO_28:=28): 'BOARD_GPIO_SMUIO_28', (BOARD_GPIO_SMUIO_29:=29): 'BOARD_GPIO_SMUIO_29', (BOARD_GPIO_SMUIO_30:=30): 'BOARD_GPIO_SMUIO_30', (BOARD_GPIO_SMUIO_31:=31): 'BOARD_GPIO_SMUIO_31', (MAX_BOARD_GPIO_SMUIO_NUM:=32): 'MAX_BOARD_GPIO_SMUIO_NUM', (BOARD_GPIO_DC_GEN_A:=33): 'BOARD_GPIO_DC_GEN_A', (BOARD_GPIO_DC_GEN_B:=34): 'BOARD_GPIO_DC_GEN_B', (BOARD_GPIO_DC_GEN_C:=35): 'BOARD_GPIO_DC_GEN_C', (BOARD_GPIO_DC_GEN_D:=36): 'BOARD_GPIO_DC_GEN_D', (BOARD_GPIO_DC_GEN_E:=37): 'BOARD_GPIO_DC_GEN_E', (BOARD_GPIO_DC_GEN_F:=38): 'BOARD_GPIO_DC_GEN_F', (BOARD_GPIO_DC_GEN_G:=39): 'BOARD_GPIO_DC_GEN_G', (BOARD_GPIO_DC_GENLK_CLK:=40): 'BOARD_GPIO_DC_GENLK_CLK', (BOARD_GPIO_DC_GENLK_VSYNC:=41): 'BOARD_GPIO_DC_GENLK_VSYNC', (BOARD_GPIO_DC_SWAPLOCK_A:=42): 'BOARD_GPIO_DC_SWAPLOCK_A', (BOARD_GPIO_DC_SWAPLOCK_B:=43): 'BOARD_GPIO_DC_SWAPLOCK_B'}
@c.record
class BootValues_t(c.Struct):
  SIZE = 112
  InitGfxclk_bypass: int
  InitSocclk: int
  InitMp0clk: int
  InitMpioclk: int
  InitSmnclk: int
  InitUcpclk: int
  InitCsrclk: int
  InitDprefclk: int
  InitDcfclk: int
  InitDtbclk: int
  InitDclk: int
  InitVclk: int
  InitUsbdfsclk: int
  InitMp1clk: int
  InitLclk: int
  InitBaco400clk_bypass: int
  InitBaco1200clk_bypass: int
  InitBaco700clk_bypass: int
  InitFclk: int
  InitGfxclk_clkb: int
  InitUclkDPMState: int
  Padding: c.Array[ctypes.c_ubyte, Literal[3]]
  InitVcoFreqPll0: int
  InitVcoFreqPll1: int
  InitVcoFreqPll2: int
  InitVcoFreqPll3: int
  InitVcoFreqPll4: int
  InitVcoFreqPll5: int
  InitVcoFreqPll6: int
  InitGfx: int
  InitSoc: int
  InitU: int
  Padding2: int
  Spare: c.Array[ctypes.c_uint32, Literal[8]]
BootValues_t.register_fields([('InitGfxclk_bypass', uint16_t, 0), ('InitSocclk', uint16_t, 2), ('InitMp0clk', uint16_t, 4), ('InitMpioclk', uint16_t, 6), ('InitSmnclk', uint16_t, 8), ('InitUcpclk', uint16_t, 10), ('InitCsrclk', uint16_t, 12), ('InitDprefclk', uint16_t, 14), ('InitDcfclk', uint16_t, 16), ('InitDtbclk', uint16_t, 18), ('InitDclk', uint16_t, 20), ('InitVclk', uint16_t, 22), ('InitUsbdfsclk', uint16_t, 24), ('InitMp1clk', uint16_t, 26), ('InitLclk', uint16_t, 28), ('InitBaco400clk_bypass', uint16_t, 30), ('InitBaco1200clk_bypass', uint16_t, 32), ('InitBaco700clk_bypass', uint16_t, 34), ('InitFclk', uint16_t, 36), ('InitGfxclk_clkb', uint16_t, 38), ('InitUclkDPMState', uint8_t, 40), ('Padding', c.Array[uint8_t, Literal[3]], 41), ('InitVcoFreqPll0', uint32_t, 44), ('InitVcoFreqPll1', uint32_t, 48), ('InitVcoFreqPll2', uint32_t, 52), ('InitVcoFreqPll3', uint32_t, 56), ('InitVcoFreqPll4', uint32_t, 60), ('InitVcoFreqPll5', uint32_t, 64), ('InitVcoFreqPll6', uint32_t, 68), ('InitGfx', uint16_t, 72), ('InitSoc', uint16_t, 74), ('InitU', uint16_t, 76), ('Padding2', uint16_t, 78), ('Spare', c.Array[uint32_t, Literal[8]], 80)])
@c.record
class MsgLimits_t(c.Struct):
  SIZE = 120
  Power: c.Array[c.Array[ctypes.c_uint16, Literal[2]], Literal[4]]
  Tdc: c.Array[ctypes.c_uint16, Literal[3]]
  Temperature: c.Array[ctypes.c_uint16, Literal[13]]
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
MsgLimits_t.register_fields([('Power', c.Array[c.Array[uint16_t, Literal[2]], Literal[4]], 0), ('Tdc', c.Array[uint16_t, Literal[3]], 16), ('Temperature', c.Array[uint16_t, Literal[13]], 22), ('PwmLimitMin', uint8_t, 48), ('PwmLimitMax', uint8_t, 49), ('FanTargetTemperature', uint8_t, 50), ('Spare1', c.Array[uint8_t, Literal[1]], 51), ('AcousticTargetRpmThresholdMin', uint16_t, 52), ('AcousticTargetRpmThresholdMax', uint16_t, 54), ('AcousticLimitRpmThresholdMin', uint16_t, 56), ('AcousticLimitRpmThresholdMax', uint16_t, 58), ('PccLimitMin', uint16_t, 60), ('PccLimitMax', uint16_t, 62), ('FanStopTempMin', uint16_t, 64), ('FanStopTempMax', uint16_t, 66), ('FanStartTempMin', uint16_t, 68), ('FanStartTempMax', uint16_t, 70), ('PowerMinPpt0', c.Array[uint16_t, Literal[2]], 72), ('Spare', c.Array[uint32_t, Literal[11]], 76)])
@c.record
class DriverReportedClocks_t(c.Struct):
  SIZE = 28
  BaseClockAc: int
  GameClockAc: int
  BoostClockAc: int
  BaseClockDc: int
  GameClockDc: int
  BoostClockDc: int
  Reserved: c.Array[ctypes.c_uint32, Literal[4]]
DriverReportedClocks_t.register_fields([('BaseClockAc', uint16_t, 0), ('GameClockAc', uint16_t, 2), ('BoostClockAc', uint16_t, 4), ('BaseClockDc', uint16_t, 6), ('GameClockDc', uint16_t, 8), ('BoostClockDc', uint16_t, 10), ('Reserved', c.Array[uint32_t, Literal[4]], 12)])
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
class SkuTable_t(c.Struct):
  SIZE = 3484
  Version: int
  FeaturesToRun: c.Array[ctypes.c_uint32, Literal[2]]
  TotalPowerConfig: int
  CustomerVariant: int
  MemoryTemperatureTypeMask: int
  SmartShiftVersion: int
  SocketPowerLimitAc: c.Array[ctypes.c_uint16, Literal[4]]
  SocketPowerLimitDc: c.Array[ctypes.c_uint16, Literal[4]]
  SocketPowerLimitSmartShift2: int
  EnableLegacyPptLimit: int
  UseInputTelemetry: int
  SmartShiftMinReportedPptinDcs: int
  PaddingPpt: c.Array[ctypes.c_ubyte, Literal[1]]
  VrTdcLimit: c.Array[ctypes.c_uint16, Literal[3]]
  PlatformTdcLimit: c.Array[ctypes.c_uint16, Literal[3]]
  TemperatureLimit: c.Array[ctypes.c_uint16, Literal[13]]
  HwCtfTempLimit: int
  PaddingInfra: int
  FitControllerFailureRateLimit: int
  FitControllerGfxDutyCycle: int
  FitControllerSocDutyCycle: int
  FitControllerSocOffset: int
  GfxApccPlusResidencyLimit: int
  ThrottlerControlMask: int
  FwDStateMask: int
  UlvVoltageOffset: c.Array[ctypes.c_uint16, Literal[2]]
  UlvVoltageOffsetU: int
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
  Vmin_droop: QuadraticInt_t
  SpareVmin: c.Array[ctypes.c_uint32, Literal[9]]
  DpmDescriptor: c.Array[DpmDescriptor_t, Literal[13]]
  FreqTableGfx: c.Array[ctypes.c_uint16, Literal[16]]
  FreqTableVclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableSocclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableUclk: c.Array[ctypes.c_uint16, Literal[4]]
  FreqTableDispclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDppClk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDprefclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDcfclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDtbclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableFclk: c.Array[ctypes.c_uint16, Literal[8]]
  DcModeMaxFreq: c.Array[ctypes.c_uint32, Literal[13]]
  Mp0clkFreq: c.Array[ctypes.c_uint16, Literal[2]]
  Mp0DpmVoltage: c.Array[ctypes.c_uint16, Literal[2]]
  GfxclkSpare: c.Array[ctypes.c_ubyte, Literal[2]]
  GfxclkFreqCap: int
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
  DfllBtcMasterScalerM: int
  DfllBtcMasterScalerB: int
  DfllBtcSlaveScalerM: int
  DfllBtcSlaveScalerB: int
  DfllPccAsWaitCtrl: int
  DfllPccAsStepCtrl: int
  DfllL2FrequencyBoostM: int
  DfllL2FrequencyBoostB: int
  GfxGpoSpare: c.Array[ctypes.c_uint32, Literal[8]]
  DcsGfxOffVoltage: int
  PaddingDcs: int
  DcsMinGfxOffTime: int
  DcsMaxGfxOffTime: int
  DcsMinCreditAccum: int
  DcsExitHysteresis: int
  DcsTimeout: int
  FoptEnabled: int
  DcsSpare2: c.Array[ctypes.c_ubyte, Literal[3]]
  DcsFoptM: int
  DcsFoptB: int
  DcsSpare: c.Array[ctypes.c_uint32, Literal[11]]
  ShadowFreqTableUclk: c.Array[ctypes.c_uint16, Literal[4]]
  UseStrobeModeOptimizations: int
  PaddingMem: c.Array[ctypes.c_ubyte, Literal[3]]
  UclkDpmPstates: c.Array[ctypes.c_ubyte, Literal[4]]
  FreqTableUclkDiv: c.Array[ctypes.c_ubyte, Literal[4]]
  MemVmempVoltage: c.Array[ctypes.c_uint16, Literal[4]]
  MemVddioVoltage: c.Array[ctypes.c_uint16, Literal[4]]
  FclkDpmUPstates: c.Array[ctypes.c_ubyte, Literal[8]]
  FclkDpmVddU: c.Array[ctypes.c_uint16, Literal[8]]
  FclkDpmUSpeed: c.Array[ctypes.c_uint16, Literal[8]]
  FclkDpmDisallowPstateFreq: int
  PaddingFclk: int
  PcieGenSpeed: c.Array[ctypes.c_ubyte, Literal[3]]
  PcieLaneCount: c.Array[ctypes.c_ubyte, Literal[3]]
  LclkFreq: c.Array[ctypes.c_uint16, Literal[3]]
  FanStopTemp: c.Array[ctypes.c_uint16, Literal[13]]
  FanStartTemp: c.Array[ctypes.c_uint16, Literal[13]]
  FanGain: c.Array[ctypes.c_uint16, Literal[13]]
  FanGainPadding: int
  FanPwmMin: int
  AcousticTargetRpmThreshold: int
  AcousticLimitRpmThreshold: int
  FanMaximumRpm: int
  MGpuAcousticLimitRpmThreshold: int
  FanTargetGfxclk: int
  TempInputSelectMask: int
  FanZeroRpmEnable: int
  FanTachEdgePerRev: int
  FanTargetTemperature: c.Array[ctypes.c_uint16, Literal[13]]
  FuzzyFan_ErrorSetDelta: int
  FuzzyFan_ErrorRateSetDelta: int
  FuzzyFan_PwmSetDelta: int
  FuzzyFan_Reserved: int
  FwCtfLimit: c.Array[ctypes.c_uint16, Literal[13]]
  IntakeTempEnableRPM: int
  IntakeTempOffsetTemp: int
  IntakeTempReleaseTemp: int
  IntakeTempHighIntakeAcousticLimit: int
  IntakeTempAcouticLimitReleaseRate: int
  FanAbnormalTempLimitOffset: int
  FanStalledTriggerRpm: int
  FanAbnormalTriggerRpmCoeff: int
  FanAbnormalDetectionEnable: int
  FanIntakeSensorSupport: int
  FanIntakePadding: c.Array[ctypes.c_ubyte, Literal[3]]
  FanSpare: c.Array[ctypes.c_uint32, Literal[13]]
  OverrideGfxAvfsFuses: int
  GfxAvfsPadding: c.Array[ctypes.c_ubyte, Literal[3]]
  L2HwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[32]]
  SeHwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[32]]
  CommonRtAvfs: c.Array[ctypes.c_uint32, Literal[13]]
  L2FwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[19]]
  SeFwRtAvfsFuses: c.Array[ctypes.c_uint32, Literal[19]]
  Droop_PWL_F: c.Array[ctypes.c_uint32, Literal[5]]
  Droop_PWL_a: c.Array[ctypes.c_uint32, Literal[5]]
  Droop_PWL_b: c.Array[ctypes.c_uint32, Literal[5]]
  Droop_PWL_c: c.Array[ctypes.c_uint32, Literal[5]]
  Static_PWL_Offset: c.Array[ctypes.c_uint32, Literal[5]]
  dGbV_dT_vmin: int
  dGbV_dT_vmax: int
  V2F_vmin_range_low: int
  V2F_vmin_range_high: int
  V2F_vmax_range_low: int
  V2F_vmax_range_high: int
  DcBtcGfxParams: AvfsDcBtcParams_t
  GfxAvfsSpare: c.Array[ctypes.c_uint32, Literal[32]]
  OverrideSocAvfsFuses: int
  MinSocAvfsRevision: int
  SocAvfsPadding: c.Array[ctypes.c_ubyte, Literal[2]]
  SocAvfsFuseOverride: c.Array[AvfsFuseOverride_t, Literal[3]]
  dBtcGbSoc: c.Array[DroopInt_t, Literal[3]]
  qAgingGb: c.Array[LinearInt_t, Literal[3]]
  qStaticVoltageOffset: c.Array[QuadraticInt_t, Literal[3]]
  DcBtcSocParams: c.Array[AvfsDcBtcParams_t, Literal[3]]
  SocAvfsSpare: c.Array[ctypes.c_uint32, Literal[32]]
  BootValues: BootValues_t
  DriverReportedClocks: DriverReportedClocks_t
  MsgLimits: MsgLimits_t
  OverDriveLimitsMin: OverDriveLimits_t
  OverDriveLimitsBasicMax: OverDriveLimits_t
  reserved: c.Array[ctypes.c_uint32, Literal[22]]
  DebugOverrides: int
  TotalBoardPowerSupport: int
  TotalBoardPowerPadding: c.Array[ctypes.c_ubyte, Literal[3]]
  TotalIdleBoardPowerM: int
  TotalIdleBoardPowerB: int
  TotalBoardPowerM: int
  TotalBoardPowerB: int
  qFeffCoeffGameClock: c.Array[QuadraticInt_t, Literal[2]]
  qFeffCoeffBaseClock: c.Array[QuadraticInt_t, Literal[2]]
  qFeffCoeffBoostClock: c.Array[QuadraticInt_t, Literal[2]]
  TemperatureLimit_Hynix: int
  TemperatureLimit_Micron: int
  TemperatureFwCtfLimit_Hynix: int
  TemperatureFwCtfLimit_Micron: int
  Spare: c.Array[ctypes.c_uint32, Literal[41]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
int32_t: TypeAlias = ctypes.c_int32
SkuTable_t.register_fields([('Version', uint32_t, 0), ('FeaturesToRun', c.Array[uint32_t, Literal[2]], 4), ('TotalPowerConfig', uint8_t, 12), ('CustomerVariant', uint8_t, 13), ('MemoryTemperatureTypeMask', uint8_t, 14), ('SmartShiftVersion', uint8_t, 15), ('SocketPowerLimitAc', c.Array[uint16_t, Literal[4]], 16), ('SocketPowerLimitDc', c.Array[uint16_t, Literal[4]], 24), ('SocketPowerLimitSmartShift2', uint16_t, 32), ('EnableLegacyPptLimit', uint8_t, 34), ('UseInputTelemetry', uint8_t, 35), ('SmartShiftMinReportedPptinDcs', uint8_t, 36), ('PaddingPpt', c.Array[uint8_t, Literal[1]], 37), ('VrTdcLimit', c.Array[uint16_t, Literal[3]], 38), ('PlatformTdcLimit', c.Array[uint16_t, Literal[3]], 44), ('TemperatureLimit', c.Array[uint16_t, Literal[13]], 50), ('HwCtfTempLimit', uint16_t, 76), ('PaddingInfra', uint16_t, 78), ('FitControllerFailureRateLimit', uint32_t, 80), ('FitControllerGfxDutyCycle', uint32_t, 84), ('FitControllerSocDutyCycle', uint32_t, 88), ('FitControllerSocOffset', uint32_t, 92), ('GfxApccPlusResidencyLimit', uint32_t, 96), ('ThrottlerControlMask', uint32_t, 100), ('FwDStateMask', uint32_t, 104), ('UlvVoltageOffset', c.Array[uint16_t, Literal[2]], 108), ('UlvVoltageOffsetU', uint16_t, 112), ('DeepUlvVoltageOffsetSoc', uint16_t, 114), ('DefaultMaxVoltage', c.Array[uint16_t, Literal[2]], 116), ('BoostMaxVoltage', c.Array[uint16_t, Literal[2]], 120), ('VminTempHystersis', c.Array[int16_t, Literal[2]], 124), ('VminTempThreshold', c.Array[int16_t, Literal[2]], 128), ('Vmin_Hot_T0', c.Array[uint16_t, Literal[2]], 132), ('Vmin_Cold_T0', c.Array[uint16_t, Literal[2]], 136), ('Vmin_Hot_Eol', c.Array[uint16_t, Literal[2]], 140), ('Vmin_Cold_Eol', c.Array[uint16_t, Literal[2]], 144), ('Vmin_Aging_Offset', c.Array[uint16_t, Literal[2]], 148), ('Spare_Vmin_Plat_Offset_Hot', c.Array[uint16_t, Literal[2]], 152), ('Spare_Vmin_Plat_Offset_Cold', c.Array[uint16_t, Literal[2]], 156), ('VcBtcFixedVminAgingOffset', c.Array[uint16_t, Literal[2]], 160), ('VcBtcVmin2PsmDegrationGb', c.Array[uint16_t, Literal[2]], 164), ('VcBtcPsmA', c.Array[uint32_t, Literal[2]], 168), ('VcBtcPsmB', c.Array[uint32_t, Literal[2]], 176), ('VcBtcVminA', c.Array[uint32_t, Literal[2]], 184), ('VcBtcVminB', c.Array[uint32_t, Literal[2]], 192), ('PerPartVminEnabled', c.Array[uint8_t, Literal[2]], 200), ('VcBtcEnabled', c.Array[uint8_t, Literal[2]], 202), ('SocketPowerLimitAcTau', c.Array[uint16_t, Literal[4]], 204), ('SocketPowerLimitDcTau', c.Array[uint16_t, Literal[4]], 212), ('Vmin_droop', QuadraticInt_t, 220), ('SpareVmin', c.Array[uint32_t, Literal[9]], 232), ('DpmDescriptor', c.Array[DpmDescriptor_t, Literal[13]], 268), ('FreqTableGfx', c.Array[uint16_t, Literal[16]], 684), ('FreqTableVclk', c.Array[uint16_t, Literal[8]], 716), ('FreqTableDclk', c.Array[uint16_t, Literal[8]], 732), ('FreqTableSocclk', c.Array[uint16_t, Literal[8]], 748), ('FreqTableUclk', c.Array[uint16_t, Literal[4]], 764), ('FreqTableDispclk', c.Array[uint16_t, Literal[8]], 772), ('FreqTableDppClk', c.Array[uint16_t, Literal[8]], 788), ('FreqTableDprefclk', c.Array[uint16_t, Literal[8]], 804), ('FreqTableDcfclk', c.Array[uint16_t, Literal[8]], 820), ('FreqTableDtbclk', c.Array[uint16_t, Literal[8]], 836), ('FreqTableFclk', c.Array[uint16_t, Literal[8]], 852), ('DcModeMaxFreq', c.Array[uint32_t, Literal[13]], 868), ('Mp0clkFreq', c.Array[uint16_t, Literal[2]], 920), ('Mp0DpmVoltage', c.Array[uint16_t, Literal[2]], 924), ('GfxclkSpare', c.Array[uint8_t, Literal[2]], 928), ('GfxclkFreqCap', uint16_t, 930), ('GfxclkFgfxoffEntry', uint16_t, 932), ('GfxclkFgfxoffExitImu', uint16_t, 934), ('GfxclkFgfxoffExitRlc', uint16_t, 936), ('GfxclkThrottleClock', uint16_t, 938), ('EnableGfxPowerStagesGpio', uint8_t, 940), ('GfxIdlePadding', uint8_t, 941), ('SmsRepairWRCKClkDivEn', uint8_t, 942), ('SmsRepairWRCKClkDivVal', uint8_t, 943), ('GfxOffEntryEarlyMGCGEn', uint8_t, 944), ('GfxOffEntryForceCGCGEn', uint8_t, 945), ('GfxOffEntryForceCGCGDelayEn', uint8_t, 946), ('GfxOffEntryForceCGCGDelayVal', uint8_t, 947), ('GfxclkFreqGfxUlv', uint16_t, 948), ('GfxIdlePadding2', c.Array[uint8_t, Literal[2]], 950), ('GfxOffEntryHysteresis', uint32_t, 952), ('GfxoffSpare', c.Array[uint32_t, Literal[15]], 956), ('DfllBtcMasterScalerM', uint32_t, 1016), ('DfllBtcMasterScalerB', int32_t, 1020), ('DfllBtcSlaveScalerM', uint32_t, 1024), ('DfllBtcSlaveScalerB', int32_t, 1028), ('DfllPccAsWaitCtrl', uint32_t, 1032), ('DfllPccAsStepCtrl', uint32_t, 1036), ('DfllL2FrequencyBoostM', uint32_t, 1040), ('DfllL2FrequencyBoostB', uint32_t, 1044), ('GfxGpoSpare', c.Array[uint32_t, Literal[8]], 1048), ('DcsGfxOffVoltage', uint16_t, 1080), ('PaddingDcs', uint16_t, 1082), ('DcsMinGfxOffTime', uint16_t, 1084), ('DcsMaxGfxOffTime', uint16_t, 1086), ('DcsMinCreditAccum', uint32_t, 1088), ('DcsExitHysteresis', uint16_t, 1092), ('DcsTimeout', uint16_t, 1094), ('FoptEnabled', uint8_t, 1096), ('DcsSpare2', c.Array[uint8_t, Literal[3]], 1097), ('DcsFoptM', uint32_t, 1100), ('DcsFoptB', uint32_t, 1104), ('DcsSpare', c.Array[uint32_t, Literal[11]], 1108), ('ShadowFreqTableUclk', c.Array[uint16_t, Literal[4]], 1152), ('UseStrobeModeOptimizations', uint8_t, 1160), ('PaddingMem', c.Array[uint8_t, Literal[3]], 1161), ('UclkDpmPstates', c.Array[uint8_t, Literal[4]], 1164), ('FreqTableUclkDiv', c.Array[uint8_t, Literal[4]], 1168), ('MemVmempVoltage', c.Array[uint16_t, Literal[4]], 1172), ('MemVddioVoltage', c.Array[uint16_t, Literal[4]], 1180), ('FclkDpmUPstates', c.Array[uint8_t, Literal[8]], 1188), ('FclkDpmVddU', c.Array[uint16_t, Literal[8]], 1196), ('FclkDpmUSpeed', c.Array[uint16_t, Literal[8]], 1212), ('FclkDpmDisallowPstateFreq', uint16_t, 1228), ('PaddingFclk', uint16_t, 1230), ('PcieGenSpeed', c.Array[uint8_t, Literal[3]], 1232), ('PcieLaneCount', c.Array[uint8_t, Literal[3]], 1235), ('LclkFreq', c.Array[uint16_t, Literal[3]], 1238), ('FanStopTemp', c.Array[uint16_t, Literal[13]], 1244), ('FanStartTemp', c.Array[uint16_t, Literal[13]], 1270), ('FanGain', c.Array[uint16_t, Literal[13]], 1296), ('FanGainPadding', uint16_t, 1322), ('FanPwmMin', uint16_t, 1324), ('AcousticTargetRpmThreshold', uint16_t, 1326), ('AcousticLimitRpmThreshold', uint16_t, 1328), ('FanMaximumRpm', uint16_t, 1330), ('MGpuAcousticLimitRpmThreshold', uint16_t, 1332), ('FanTargetGfxclk', uint16_t, 1334), ('TempInputSelectMask', uint32_t, 1336), ('FanZeroRpmEnable', uint8_t, 1340), ('FanTachEdgePerRev', uint8_t, 1341), ('FanTargetTemperature', c.Array[uint16_t, Literal[13]], 1342), ('FuzzyFan_ErrorSetDelta', int16_t, 1368), ('FuzzyFan_ErrorRateSetDelta', int16_t, 1370), ('FuzzyFan_PwmSetDelta', int16_t, 1372), ('FuzzyFan_Reserved', uint16_t, 1374), ('FwCtfLimit', c.Array[uint16_t, Literal[13]], 1376), ('IntakeTempEnableRPM', uint16_t, 1402), ('IntakeTempOffsetTemp', int16_t, 1404), ('IntakeTempReleaseTemp', uint16_t, 1406), ('IntakeTempHighIntakeAcousticLimit', uint16_t, 1408), ('IntakeTempAcouticLimitReleaseRate', uint16_t, 1410), ('FanAbnormalTempLimitOffset', int16_t, 1412), ('FanStalledTriggerRpm', uint16_t, 1414), ('FanAbnormalTriggerRpmCoeff', uint16_t, 1416), ('FanAbnormalDetectionEnable', uint16_t, 1418), ('FanIntakeSensorSupport', uint8_t, 1420), ('FanIntakePadding', c.Array[uint8_t, Literal[3]], 1421), ('FanSpare', c.Array[uint32_t, Literal[13]], 1424), ('OverrideGfxAvfsFuses', uint8_t, 1476), ('GfxAvfsPadding', c.Array[uint8_t, Literal[3]], 1477), ('L2HwRtAvfsFuses', c.Array[uint32_t, Literal[32]], 1480), ('SeHwRtAvfsFuses', c.Array[uint32_t, Literal[32]], 1608), ('CommonRtAvfs', c.Array[uint32_t, Literal[13]], 1736), ('L2FwRtAvfsFuses', c.Array[uint32_t, Literal[19]], 1788), ('SeFwRtAvfsFuses', c.Array[uint32_t, Literal[19]], 1864), ('Droop_PWL_F', c.Array[uint32_t, Literal[5]], 1940), ('Droop_PWL_a', c.Array[uint32_t, Literal[5]], 1960), ('Droop_PWL_b', c.Array[uint32_t, Literal[5]], 1980), ('Droop_PWL_c', c.Array[uint32_t, Literal[5]], 2000), ('Static_PWL_Offset', c.Array[uint32_t, Literal[5]], 2020), ('dGbV_dT_vmin', uint32_t, 2040), ('dGbV_dT_vmax', uint32_t, 2044), ('V2F_vmin_range_low', uint32_t, 2048), ('V2F_vmin_range_high', uint32_t, 2052), ('V2F_vmax_range_low', uint32_t, 2056), ('V2F_vmax_range_high', uint32_t, 2060), ('DcBtcGfxParams', AvfsDcBtcParams_t, 2064), ('GfxAvfsSpare', c.Array[uint32_t, Literal[32]], 2084), ('OverrideSocAvfsFuses', uint8_t, 2212), ('MinSocAvfsRevision', uint8_t, 2213), ('SocAvfsPadding', c.Array[uint8_t, Literal[2]], 2214), ('SocAvfsFuseOverride', c.Array[AvfsFuseOverride_t, Literal[3]], 2216), ('dBtcGbSoc', c.Array[DroopInt_t, Literal[3]], 2384), ('qAgingGb', c.Array[LinearInt_t, Literal[3]], 2420), ('qStaticVoltageOffset', c.Array[QuadraticInt_t, Literal[3]], 2444), ('DcBtcSocParams', c.Array[AvfsDcBtcParams_t, Literal[3]], 2480), ('SocAvfsSpare', c.Array[uint32_t, Literal[32]], 2540), ('BootValues', BootValues_t, 2668), ('DriverReportedClocks', DriverReportedClocks_t, 2780), ('MsgLimits', MsgLimits_t, 2808), ('OverDriveLimitsMin', OverDriveLimits_t, 2928), ('OverDriveLimitsBasicMax', OverDriveLimits_t, 3016), ('reserved', c.Array[uint32_t, Literal[22]], 3104), ('DebugOverrides', uint32_t, 3192), ('TotalBoardPowerSupport', uint8_t, 3196), ('TotalBoardPowerPadding', c.Array[uint8_t, Literal[3]], 3197), ('TotalIdleBoardPowerM', int16_t, 3200), ('TotalIdleBoardPowerB', int16_t, 3202), ('TotalBoardPowerM', int16_t, 3204), ('TotalBoardPowerB', int16_t, 3206), ('qFeffCoeffGameClock', c.Array[QuadraticInt_t, Literal[2]], 3208), ('qFeffCoeffBaseClock', c.Array[QuadraticInt_t, Literal[2]], 3232), ('qFeffCoeffBoostClock', c.Array[QuadraticInt_t, Literal[2]], 3256), ('TemperatureLimit_Hynix', uint16_t, 3280), ('TemperatureLimit_Micron', uint16_t, 3282), ('TemperatureFwCtfLimit_Hynix', uint16_t, 3284), ('TemperatureFwCtfLimit_Micron', uint16_t, 3286), ('Spare', c.Array[uint32_t, Literal[41]], 3288), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 3452)])
@c.record
class BoardTable_t(c.Struct):
  SIZE = 488
  Version: int
  I2cControllers: c.Array[I2cControllerConfig_t, Literal[8]]
  VddGfxVrMapping: int
  VddSocVrMapping: int
  VddMem0VrMapping: int
  VddMem1VrMapping: int
  GfxUlvPhaseSheddingMask: int
  SocUlvPhaseSheddingMask: int
  VmempUlvPhaseSheddingMask: int
  VddioUlvPhaseSheddingMask: int
  SlaveAddrMapping: c.Array[ctypes.c_ubyte, Literal[5]]
  VrPsiSupport: c.Array[ctypes.c_ubyte, Literal[5]]
  PaddingPsi: c.Array[ctypes.c_ubyte, Literal[5]]
  EnablePsi6: c.Array[ctypes.c_ubyte, Literal[5]]
  SviTelemetryScale: c.Array[SviTelemetryScale_t, Literal[5]]
  VoltageTelemetryRatio: c.Array[ctypes.c_uint32, Literal[5]]
  DownSlewRateVr: c.Array[ctypes.c_ubyte, Literal[5]]
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
  PostVoltageSetBacoDelay: int
  BacoEntryDelay: int
  FuseWritePowerMuxPresent: int
  FuseWritePadding: c.Array[ctypes.c_ubyte, Literal[3]]
  BoardSpare: c.Array[ctypes.c_uint32, Literal[63]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
BoardTable_t.register_fields([('Version', uint32_t, 0), ('I2cControllers', c.Array[I2cControllerConfig_t, Literal[8]], 4), ('VddGfxVrMapping', uint8_t, 68), ('VddSocVrMapping', uint8_t, 69), ('VddMem0VrMapping', uint8_t, 70), ('VddMem1VrMapping', uint8_t, 71), ('GfxUlvPhaseSheddingMask', uint8_t, 72), ('SocUlvPhaseSheddingMask', uint8_t, 73), ('VmempUlvPhaseSheddingMask', uint8_t, 74), ('VddioUlvPhaseSheddingMask', uint8_t, 75), ('SlaveAddrMapping', c.Array[uint8_t, Literal[5]], 76), ('VrPsiSupport', c.Array[uint8_t, Literal[5]], 81), ('PaddingPsi', c.Array[uint8_t, Literal[5]], 86), ('EnablePsi6', c.Array[uint8_t, Literal[5]], 91), ('SviTelemetryScale', c.Array[SviTelemetryScale_t, Literal[5]], 96), ('VoltageTelemetryRatio', c.Array[uint32_t, Literal[5]], 116), ('DownSlewRateVr', c.Array[uint8_t, Literal[5]], 136), ('LedOffGpio', uint8_t, 141), ('FanOffGpio', uint8_t, 142), ('GfxVrPowerStageOffGpio', uint8_t, 143), ('AcDcGpio', uint8_t, 144), ('AcDcPolarity', uint8_t, 145), ('VR0HotGpio', uint8_t, 146), ('VR0HotPolarity', uint8_t, 147), ('GthrGpio', uint8_t, 148), ('GthrPolarity', uint8_t, 149), ('LedPin0', uint8_t, 150), ('LedPin1', uint8_t, 151), ('LedPin2', uint8_t, 152), ('LedEnableMask', uint8_t, 153), ('LedPcie', uint8_t, 154), ('LedError', uint8_t, 155), ('UclkTrainingModeSpreadPercent', uint8_t, 156), ('UclkSpreadPadding', uint8_t, 157), ('UclkSpreadFreq', uint16_t, 158), ('UclkSpreadPercent', c.Array[uint8_t, Literal[16]], 160), ('GfxclkSpreadEnable', uint8_t, 176), ('FclkSpreadPercent', uint8_t, 177), ('FclkSpreadFreq', uint16_t, 178), ('DramWidth', uint8_t, 180), ('PaddingMem1', c.Array[uint8_t, Literal[7]], 181), ('HsrEnabled', uint8_t, 188), ('VddqOffEnabled', uint8_t, 189), ('PaddingUmcFlags', c.Array[uint8_t, Literal[2]], 190), ('PostVoltageSetBacoDelay', uint32_t, 192), ('BacoEntryDelay', uint32_t, 196), ('FuseWritePowerMuxPresent', uint8_t, 200), ('FuseWritePadding', c.Array[uint8_t, Literal[3]], 201), ('BoardSpare', c.Array[uint32_t, Literal[63]], 204), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 456)])
@c.record
class PPTable_t(c.Struct):
  SIZE = 3972
  SkuTable: SkuTable_t
  BoardTable: BoardTable_t
PPTable_t.register_fields([('SkuTable', SkuTable_t, 0), ('BoardTable', BoardTable_t, 3484)])
@c.record
class DriverSmuConfig_t(c.Struct):
  SIZE = 16
  GfxclkAverageLpfTau: int
  FclkAverageLpfTau: int
  UclkAverageLpfTau: int
  GfxActivityLpfTau: int
  UclkActivityLpfTau: int
  SocketPowerLpfTau: int
  VcnClkAverageLpfTau: int
  VcnUsageAverageLpfTau: int
DriverSmuConfig_t.register_fields([('GfxclkAverageLpfTau', uint16_t, 0), ('FclkAverageLpfTau', uint16_t, 2), ('UclkAverageLpfTau', uint16_t, 4), ('GfxActivityLpfTau', uint16_t, 6), ('UclkActivityLpfTau', uint16_t, 8), ('SocketPowerLpfTau', uint16_t, 10), ('VcnClkAverageLpfTau', uint16_t, 12), ('VcnUsageAverageLpfTau', uint16_t, 14)])
@c.record
class DriverSmuConfigExternal_t(c.Struct):
  SIZE = 80
  DriverSmuConfig: DriverSmuConfig_t
  Spare: c.Array[ctypes.c_uint32, Literal[8]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
DriverSmuConfigExternal_t.register_fields([('DriverSmuConfig', DriverSmuConfig_t, 0), ('Spare', c.Array[uint32_t, Literal[8]], 16), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 48)])
@c.record
class DriverInfoTable_t(c.Struct):
  SIZE = 372
  FreqTableGfx: c.Array[ctypes.c_uint16, Literal[16]]
  FreqTableVclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableSocclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableUclk: c.Array[ctypes.c_uint16, Literal[4]]
  FreqTableDispclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDppClk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDprefclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDcfclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableDtbclk: c.Array[ctypes.c_uint16, Literal[8]]
  FreqTableFclk: c.Array[ctypes.c_uint16, Literal[8]]
  DcModeMaxFreq: c.Array[ctypes.c_uint16, Literal[13]]
  Padding: int
  Spare: c.Array[ctypes.c_uint32, Literal[32]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
DriverInfoTable_t.register_fields([('FreqTableGfx', c.Array[uint16_t, Literal[16]], 0), ('FreqTableVclk', c.Array[uint16_t, Literal[8]], 32), ('FreqTableDclk', c.Array[uint16_t, Literal[8]], 48), ('FreqTableSocclk', c.Array[uint16_t, Literal[8]], 64), ('FreqTableUclk', c.Array[uint16_t, Literal[4]], 80), ('FreqTableDispclk', c.Array[uint16_t, Literal[8]], 88), ('FreqTableDppClk', c.Array[uint16_t, Literal[8]], 104), ('FreqTableDprefclk', c.Array[uint16_t, Literal[8]], 120), ('FreqTableDcfclk', c.Array[uint16_t, Literal[8]], 136), ('FreqTableDtbclk', c.Array[uint16_t, Literal[8]], 152), ('FreqTableFclk', c.Array[uint16_t, Literal[8]], 168), ('DcModeMaxFreq', c.Array[uint16_t, Literal[13]], 184), ('Padding', uint16_t, 210), ('Spare', c.Array[uint32_t, Literal[32]], 212), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 340)])
@c.record
class SmuMetrics_t(c.Struct):
  SIZE = 244
  CurrClock: c.Array[ctypes.c_uint32, Literal[13]]
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
  PCIeBusy: int
  dGPU_W_MAX: int
  padding: int
  MetricsCounter: int
  AvgVoltage: c.Array[ctypes.c_uint16, Literal[5]]
  AvgCurrent: c.Array[ctypes.c_uint16, Literal[5]]
  AverageGfxActivity: int
  AverageUclkActivity: int
  Vcn0ActivityPercentage: int
  Vcn1ActivityPercentage: int
  EnergyAccumulator: int
  AverageSocketPower: int
  AverageTotalBoardPower: int
  AvgTemperature: c.Array[ctypes.c_uint16, Literal[13]]
  AvgTemperatureFanIntake: int
  PcieRate: int
  PcieWidth: int
  AvgFanPwm: int
  Padding: c.Array[ctypes.c_ubyte, Literal[1]]
  AvgFanRpm: int
  ThrottlingPercentage: c.Array[ctypes.c_ubyte, Literal[22]]
  VmaxThrottlingPercentage: int
  Padding1: c.Array[ctypes.c_ubyte, Literal[3]]
  D3HotEntryCountPerMode: c.Array[ctypes.c_uint32, Literal[4]]
  D3HotExitCountPerMode: c.Array[ctypes.c_uint32, Literal[4]]
  ArmMsgReceivedCountPerMode: c.Array[ctypes.c_uint32, Literal[4]]
  ApuSTAPMSmartShiftLimit: int
  ApuSTAPMLimit: int
  AvgApuSocketPower: int
  AverageUclkActivity_MAX: int
  PublicSerialNumberLower: int
  PublicSerialNumberUpper: int
SmuMetrics_t.register_fields([('CurrClock', c.Array[uint32_t, Literal[13]], 0), ('AverageGfxclkFrequencyTarget', uint16_t, 52), ('AverageGfxclkFrequencyPreDs', uint16_t, 54), ('AverageGfxclkFrequencyPostDs', uint16_t, 56), ('AverageFclkFrequencyPreDs', uint16_t, 58), ('AverageFclkFrequencyPostDs', uint16_t, 60), ('AverageMemclkFrequencyPreDs', uint16_t, 62), ('AverageMemclkFrequencyPostDs', uint16_t, 64), ('AverageVclk0Frequency', uint16_t, 66), ('AverageDclk0Frequency', uint16_t, 68), ('AverageVclk1Frequency', uint16_t, 70), ('AverageDclk1Frequency', uint16_t, 72), ('PCIeBusy', uint16_t, 74), ('dGPU_W_MAX', uint16_t, 76), ('padding', uint16_t, 78), ('MetricsCounter', uint32_t, 80), ('AvgVoltage', c.Array[uint16_t, Literal[5]], 84), ('AvgCurrent', c.Array[uint16_t, Literal[5]], 94), ('AverageGfxActivity', uint16_t, 104), ('AverageUclkActivity', uint16_t, 106), ('Vcn0ActivityPercentage', uint16_t, 108), ('Vcn1ActivityPercentage', uint16_t, 110), ('EnergyAccumulator', uint32_t, 112), ('AverageSocketPower', uint16_t, 116), ('AverageTotalBoardPower', uint16_t, 118), ('AvgTemperature', c.Array[uint16_t, Literal[13]], 120), ('AvgTemperatureFanIntake', uint16_t, 146), ('PcieRate', uint8_t, 148), ('PcieWidth', uint8_t, 149), ('AvgFanPwm', uint8_t, 150), ('Padding', c.Array[uint8_t, Literal[1]], 151), ('AvgFanRpm', uint16_t, 152), ('ThrottlingPercentage', c.Array[uint8_t, Literal[22]], 154), ('VmaxThrottlingPercentage', uint8_t, 176), ('Padding1', c.Array[uint8_t, Literal[3]], 177), ('D3HotEntryCountPerMode', c.Array[uint32_t, Literal[4]], 180), ('D3HotExitCountPerMode', c.Array[uint32_t, Literal[4]], 196), ('ArmMsgReceivedCountPerMode', c.Array[uint32_t, Literal[4]], 212), ('ApuSTAPMSmartShiftLimit', uint16_t, 228), ('ApuSTAPMLimit', uint16_t, 230), ('AvgApuSocketPower', uint16_t, 232), ('AverageUclkActivity_MAX', uint16_t, 234), ('PublicSerialNumberLower', uint32_t, 236), ('PublicSerialNumberUpper', uint32_t, 240)])
@c.record
class SmuMetricsExternal_t(c.Struct):
  SIZE = 392
  SmuMetrics: SmuMetrics_t
  Spare: c.Array[ctypes.c_uint32, Literal[29]]
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
SmuMetricsExternal_t.register_fields([('SmuMetrics', SmuMetrics_t, 0), ('Spare', c.Array[uint32_t, Literal[29]], 244), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 360)])
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
  SIZE = 2568
  avgPsmCount: c.Array[ctypes.c_uint16, Literal[214]]
  minPsmCount: c.Array[ctypes.c_uint16, Literal[214]]
  avgPsmVoltage: c.Array[ctypes.c_float, Literal[214]]
  minPsmVoltage: c.Array[ctypes.c_float, Literal[214]]
AvfsDebugTable_t.register_fields([('avgPsmCount', c.Array[uint16_t, Literal[214]], 0), ('minPsmCount', c.Array[uint16_t, Literal[214]], 428), ('avgPsmVoltage', c.Array[ctypes.c_float, Literal[214]], 856), ('minPsmVoltage', c.Array[ctypes.c_float, Literal[214]], 1712)])
@c.record
class AvfsDebugTableExternal_t(c.Struct):
  SIZE = 2600
  AvfsDebugTable: AvfsDebugTable_t
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
AvfsDebugTableExternal_t.register_fields([('AvfsDebugTable', AvfsDebugTable_t, 0), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 2568)])
@c.record
class DpmActivityMonitorCoeffInt_t(c.Struct):
  SIZE = 92
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
  Mem_UpThreshold_Limit: c.Array[ctypes.c_uint32, Literal[4]]
  Mem_UpHystLimit: c.Array[ctypes.c_ubyte, Literal[4]]
  Mem_DownHystLimit: c.Array[ctypes.c_ubyte, Literal[4]]
  Mem_Fps: int
  padding: c.Array[ctypes.c_ubyte, Literal[2]]
DpmActivityMonitorCoeffInt_t.register_fields([('Gfx_ActiveHystLimit', uint8_t, 0), ('Gfx_IdleHystLimit', uint8_t, 1), ('Gfx_FPS', uint8_t, 2), ('Gfx_MinActiveFreqType', uint8_t, 3), ('Gfx_BoosterFreqType', uint8_t, 4), ('PaddingGfx', uint8_t, 5), ('Gfx_MinActiveFreq', uint16_t, 6), ('Gfx_BoosterFreq', uint16_t, 8), ('Gfx_PD_Data_time_constant', uint16_t, 10), ('Gfx_PD_Data_limit_a', uint32_t, 12), ('Gfx_PD_Data_limit_b', uint32_t, 16), ('Gfx_PD_Data_limit_c', uint32_t, 20), ('Gfx_PD_Data_error_coeff', uint32_t, 24), ('Gfx_PD_Data_error_rate_coeff', uint32_t, 28), ('Fclk_ActiveHystLimit', uint8_t, 32), ('Fclk_IdleHystLimit', uint8_t, 33), ('Fclk_FPS', uint8_t, 34), ('Fclk_MinActiveFreqType', uint8_t, 35), ('Fclk_BoosterFreqType', uint8_t, 36), ('PaddingFclk', uint8_t, 37), ('Fclk_MinActiveFreq', uint16_t, 38), ('Fclk_BoosterFreq', uint16_t, 40), ('Fclk_PD_Data_time_constant', uint16_t, 42), ('Fclk_PD_Data_limit_a', uint32_t, 44), ('Fclk_PD_Data_limit_b', uint32_t, 48), ('Fclk_PD_Data_limit_c', uint32_t, 52), ('Fclk_PD_Data_error_coeff', uint32_t, 56), ('Fclk_PD_Data_error_rate_coeff', uint32_t, 60), ('Mem_UpThreshold_Limit', c.Array[uint32_t, Literal[4]], 64), ('Mem_UpHystLimit', c.Array[uint8_t, Literal[4]], 80), ('Mem_DownHystLimit', c.Array[uint8_t, Literal[4]], 84), ('Mem_Fps', uint16_t, 88), ('padding', c.Array[uint8_t, Literal[2]], 90)])
@c.record
class DpmActivityMonitorCoeffIntExternal_t(c.Struct):
  SIZE = 124
  DpmActivityMonitorCoeffInt: DpmActivityMonitorCoeffInt_t
  MmHubPadding: c.Array[ctypes.c_uint32, Literal[8]]
DpmActivityMonitorCoeffIntExternal_t.register_fields([('DpmActivityMonitorCoeffInt', DpmActivityMonitorCoeffInt_t, 0), ('MmHubPadding', c.Array[uint32_t, Literal[8]], 92)])
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