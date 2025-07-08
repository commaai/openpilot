# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-include', 'stdint.h']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



__SMU_V14_0_0_PMFW_H__ = True # macro
ENABLE_DEBUG_FEATURES = True # macro
FEATURE_CCLK_DPM_BIT = 0 # macro
FEATURE_FAN_CONTROLLER_BIT = 1 # macro
FEATURE_DATA_CALCULATION_BIT = 2 # macro
FEATURE_PPT_BIT = 3 # macro
FEATURE_TDC_BIT = 4 # macro
FEATURE_THERMAL_BIT = 5 # macro
FEATURE_FIT_BIT = 6 # macro
FEATURE_EDC_BIT = 7 # macro
FEATURE_PLL_POWER_DOWN_BIT = 8 # macro
FEATURE_VDDOFF_BIT = 9 # macro
FEATURE_VCN_DPM_BIT = 10 # macro
FEATURE_DS_MPM_BIT = 11 # macro
FEATURE_FCLK_DPM_BIT = 12 # macro
FEATURE_SOCCLK_DPM_BIT = 13 # macro
FEATURE_DS_MPIO_BIT = 14 # macro
FEATURE_LCLK_DPM_BIT = 15 # macro
FEATURE_SHUBCLK_DPM_BIT = 16 # macro
FEATURE_DCFCLK_DPM_BIT = 17 # macro
FEATURE_ISP_DPM_BIT = 18 # macro
FEATURE_IPU_DPM_BIT = 19 # macro
FEATURE_GFX_DPM_BIT = 20 # macro
FEATURE_DS_GFXCLK_BIT = 10 # macro
FEATURE_DS_SOCCLK_BIT = 11 # macro
FEATURE_DS_LCLK_BIT = 13 # macro
FEATURE_LOW_POWER_DCNCLKS_BIT = 24 # macro
FEATURE_DS_SHUBCLK_BIT = 25 # macro
FEATURE_RESERVED0_BIT = 26 # macro
FEATURE_ZSTATES_BIT = 27 # macro
FEATURE_IOMMUL2_PG_BIT = 28 # macro
FEATURE_DS_FCLK_BIT = 12 # macro
FEATURE_DS_SMNCLK_BIT = 30 # macro
FEATURE_DS_MP1CLK_BIT = 31 # macro
FEATURE_WHISPER_MODE_BIT = 32 # macro
FEATURE_SMU_LOW_POWER_BIT = 33 # macro
FEATURE_RESERVED1_BIT = 34 # macro
FEATURE_GFX_DEM_BIT = 35 # macro
FEATURE_PSI_BIT = 36 # macro
FEATURE_PROCHOT_BIT = 37 # macro
FEATURE_CPUOFF_BIT = 38 # macro
FEATURE_STAPM_BIT = 39 # macro
FEATURE_S0I3_BIT = 40 # macro
FEATURE_DF_LIGHT_CSTATE = 41 # macro
FEATURE_PERF_LIMIT_BIT = 42 # macro
FEATURE_CORE_DLDO_BIT = 43 # macro
FEATURE_DVO_BIT = 44 # macro
FEATURE_DS_VCN_BIT = 44 # macro
FEATURE_CPPC_BIT = 46 # macro
FEATURE_CPPC_PREFERRED_CORES = 47 # macro
FEATURE_DF_CSTATES_BIT = 48 # macro
FEATURE_FAST_PSTATE_CLDO_BIT = 49 # macro
FEATURE_ATHUB_PG_BIT = 50 # macro
FEATURE_VDDOFF_ECO_BIT = 51 # macro
FEATURE_ZSTATES_ECO_BIT = 52 # macro
FEATURE_CC6_BIT = 53 # macro
FEATURE_DS_UMCCLK_BIT = 54 # macro
FEATURE_DS_ISPCLK_BIT = 55 # macro
FEATURE_DS_HSPCLK_BIT = 56 # macro
FEATURE_P3T_BIT = 57 # macro
FEATURE_DS_IPUCLK_BIT = 58 # macro
FEATURE_DS_VPECLK_BIT = 59 # macro
FEATURE_VPE_DPM_BIT = 60 # macro
FEATURE_SMART_L3_RINSER_BIT = 61 # macro
FEATURE_PCC_BIT = 62 # macro
NUM_FEATURES = 64 # macro
class struct_SMU14_Firmware_Footer(Structure):
    pass

struct_SMU14_Firmware_Footer._pack_ = 1 # source:False
struct_SMU14_Firmware_Footer._fields_ = [
    ('Signature', ctypes.c_uint32),
]

SMU14_Firmware_Footer = struct_SMU14_Firmware_Footer
class struct_c__SA_SMU_Firmware_Header(Structure):
    pass

struct_c__SA_SMU_Firmware_Header._pack_ = 1 # source:False
struct_c__SA_SMU_Firmware_Header._fields_ = [
    ('ImageVersion', ctypes.c_uint32),
    ('ImageVersion2', ctypes.c_uint32),
    ('Padding0', ctypes.c_uint32 * 3),
    ('SizeFWSigned', ctypes.c_uint32),
    ('Padding1', ctypes.c_uint32 * 25),
    ('FirmwareType', ctypes.c_uint32),
    ('Filler', ctypes.c_uint32 * 32),
]

SMU_Firmware_Header = struct_c__SA_SMU_Firmware_Header
class struct_c__SA_FwStatus_t(Structure):
    pass

struct_c__SA_FwStatus_t._pack_ = 1 # source:False
struct_c__SA_FwStatus_t._fields_ = [
    ('DpmHandlerID', ctypes.c_uint64, 8),
    ('ActivityMonitorID', ctypes.c_uint64, 8),
    ('DpmTimerID', ctypes.c_uint64, 8),
    ('DpmHubID', ctypes.c_uint64, 4),
    ('DpmHubTask', ctypes.c_uint64, 4),
    ('CclkSyncStatus', ctypes.c_uint64, 8),
    ('Ccx0CpuOff', ctypes.c_uint64, 2),
    ('Ccx1CpuOff', ctypes.c_uint64, 2),
    ('GfxOffStatus', ctypes.c_uint64, 2),
    ('VddOff', ctypes.c_uint64, 1),
    ('InWhisperMode', ctypes.c_uint64, 1),
    ('ZstateStatus', ctypes.c_uint64, 4),
    ('spare0', ctypes.c_uint64, 4),
    ('DstateFun', ctypes.c_uint64, 4),
    ('DstateDev', ctypes.c_uint64, 4),
    ('P2JobHandler', ctypes.c_uint64, 24),
    ('RsmuPmiP2PendingCnt', ctypes.c_uint64, 8),
    ('PostCode', ctypes.c_uint64, 32),
    ('MsgPortBusy', ctypes.c_uint64, 24),
    ('RsmuPmiP1Pending', ctypes.c_uint64, 1),
    ('DfCstateExitPending', ctypes.c_uint64, 1),
    ('Ccx0Pc6ExitPending', ctypes.c_uint64, 1),
    ('Ccx1Pc6ExitPending', ctypes.c_uint64, 1),
    ('WarmResetPending', ctypes.c_uint64, 1),
    ('spare1', ctypes.c_uint64, 3),
    ('IdleMask', ctypes.c_uint64, 32),
]

FwStatus_t = struct_c__SA_FwStatus_t
class struct_c__SA_FwStatus_t_v14_0_1(Structure):
    pass

struct_c__SA_FwStatus_t_v14_0_1._pack_ = 1 # source:False
struct_c__SA_FwStatus_t_v14_0_1._fields_ = [
    ('DpmHandlerID', ctypes.c_uint64, 8),
    ('ActivityMonitorID', ctypes.c_uint64, 8),
    ('DpmTimerID', ctypes.c_uint64, 8),
    ('DpmHubID', ctypes.c_uint64, 4),
    ('DpmHubTask', ctypes.c_uint64, 4),
    ('CclkSyncStatus', ctypes.c_uint64, 8),
    ('ZstateStatus', ctypes.c_uint64, 4),
    ('Cpu1VddOff', ctypes.c_uint64, 4),
    ('DstateFun', ctypes.c_uint64, 4),
    ('DstateDev', ctypes.c_uint64, 4),
    ('GfxOffStatus', ctypes.c_uint64, 2),
    ('Cpu0Off', ctypes.c_uint64, 2),
    ('Cpu1Off', ctypes.c_uint64, 2),
    ('Cpu0VddOff', ctypes.c_uint64, 2),
    ('P2JobHandler', ctypes.c_uint64, 32),
    ('PostCode', ctypes.c_uint64, 32),
    ('MsgPortBusy', ctypes.c_uint64, 15),
    ('RsmuPmiP1Pending', ctypes.c_uint64, 1),
    ('RsmuPmiP2PendingCnt', ctypes.c_uint64, 8),
    ('DfCstateExitPending', ctypes.c_uint64, 1),
    ('Pc6EntryPending', ctypes.c_uint64, 1),
    ('Pc6ExitPending', ctypes.c_uint64, 1),
    ('WarmResetPending', ctypes.c_uint64, 1),
    ('Mp0ClkPending', ctypes.c_uint64, 1),
    ('InWhisperMode', ctypes.c_uint64, 1),
    ('spare2', ctypes.c_uint64, 2),
    ('IdleMask', ctypes.c_uint64, 32),
]

FwStatus_t_v14_0_1 = struct_c__SA_FwStatus_t_v14_0_1
SMU_V14_0_2_PPSMC_H = True # macro
PPSMC_VERSION = 0x1 # macro
PPSMC_Result_OK = 0x1 # macro
PPSMC_Result_Failed = 0xFF # macro
PPSMC_Result_UnknownCmd = 0xFE # macro
PPSMC_Result_CmdRejectedPrereq = 0xFD # macro
PPSMC_Result_CmdRejectedBusy = 0xFC # macro
PPSMC_MSG_TestMessage = 0x1 # macro
PPSMC_MSG_GetSmuVersion = 0x2 # macro
PPSMC_MSG_GetDriverIfVersion = 0x3 # macro
PPSMC_MSG_SetAllowedFeaturesMaskLow = 0x4 # macro
PPSMC_MSG_SetAllowedFeaturesMaskHigh = 0x5 # macro
PPSMC_MSG_EnableAllSmuFeatures = 0x6 # macro
PPSMC_MSG_DisableAllSmuFeatures = 0x7 # macro
PPSMC_MSG_EnableSmuFeaturesLow = 0x8 # macro
PPSMC_MSG_EnableSmuFeaturesHigh = 0x9 # macro
PPSMC_MSG_DisableSmuFeaturesLow = 0xA # macro
PPSMC_MSG_DisableSmuFeaturesHigh = 0xB # macro
PPSMC_MSG_GetRunningSmuFeaturesLow = 0xC # macro
PPSMC_MSG_GetRunningSmuFeaturesHigh = 0xD # macro
PPSMC_MSG_SetDriverDramAddrHigh = 0xE # macro
PPSMC_MSG_SetDriverDramAddrLow = 0xF # macro
PPSMC_MSG_SetToolsDramAddrHigh = 0x10 # macro
PPSMC_MSG_SetToolsDramAddrLow = 0x11 # macro
PPSMC_MSG_TransferTableSmu2Dram = 0x12 # macro
PPSMC_MSG_TransferTableDram2Smu = 0x13 # macro
PPSMC_MSG_UseDefaultPPTable = 0x14 # macro
PPSMC_MSG_EnterBaco = 0x15 # macro
PPSMC_MSG_ExitBaco = 0x16 # macro
PPSMC_MSG_ArmD3 = 0x17 # macro
PPSMC_MSG_BacoAudioD3PME = 0x18 # macro
PPSMC_MSG_SetSoftMinByFreq = 0x19 # macro
PPSMC_MSG_SetSoftMaxByFreq = 0x1A # macro
PPSMC_MSG_SetHardMinByFreq = 0x1B # macro
PPSMC_MSG_SetHardMaxByFreq = 0x1C # macro
PPSMC_MSG_GetMinDpmFreq = 0x1D # macro
PPSMC_MSG_GetMaxDpmFreq = 0x1E # macro
PPSMC_MSG_GetDpmFreqByIndex = 0x1F # macro
PPSMC_MSG_OverridePcieParameters = 0x20 # macro
PPSMC_MSG_DramLogSetDramAddrHigh = 0x21 # macro
PPSMC_MSG_DramLogSetDramAddrLow = 0x22 # macro
PPSMC_MSG_DramLogSetDramSize = 0x23 # macro
PPSMC_MSG_SetWorkloadMask = 0x24 # macro
PPSMC_MSG_GetVoltageByDpm = 0x25 # macro
PPSMC_MSG_SetVideoFps = 0x26 # macro
PPSMC_MSG_GetDcModeMaxDpmFreq = 0x27 # macro
PPSMC_MSG_AllowGfxOff = 0x28 # macro
PPSMC_MSG_DisallowGfxOff = 0x29 # macro
PPSMC_MSG_PowerUpVcn = 0x2A # macro
PPSMC_MSG_PowerDownVcn = 0x2B # macro
PPSMC_MSG_PowerUpJpeg = 0x2C # macro
PPSMC_MSG_PowerDownJpeg = 0x2D # macro
PPSMC_MSG_PrepareMp1ForUnload = 0x2E # macro
PPSMC_MSG_SetSystemVirtualDramAddrHigh = 0x30 # macro
PPSMC_MSG_SetSystemVirtualDramAddrLow = 0x31 # macro
PPSMC_MSG_SetPptLimit = 0x32 # macro
PPSMC_MSG_GetPptLimit = 0x33 # macro
PPSMC_MSG_ReenableAcDcInterrupt = 0x34 # macro
PPSMC_MSG_NotifyPowerSource = 0x35 # macro
PPSMC_MSG_RunDcBtc = 0x36 # macro
PPSMC_MSG_SetTemperatureInputSelect = 0x38 # macro
PPSMC_MSG_SetFwDstatesMask = 0x39 # macro
PPSMC_MSG_SetThrottlerMask = 0x3A # macro
PPSMC_MSG_SetExternalClientDfCstateAllow = 0x3B # macro
PPSMC_MSG_SetMGpuFanBoostLimitRpm = 0x3C # macro
PPSMC_MSG_DumpSTBtoDram = 0x3D # macro
PPSMC_MSG_STBtoDramLogSetDramAddress = 0x3E # macro
PPSMC_MSG_DummyUndefined = 0x3F # macro
PPSMC_MSG_STBtoDramLogSetDramSize = 0x40 # macro
PPSMC_MSG_SetOBMTraceBufferLogging = 0x41 # macro
PPSMC_MSG_UseProfilingMode = 0x42 # macro
PPSMC_MSG_AllowGfxDcs = 0x43 # macro
PPSMC_MSG_DisallowGfxDcs = 0x44 # macro
PPSMC_MSG_EnableAudioStutterWA = 0x45 # macro
PPSMC_MSG_PowerUpUmsch = 0x46 # macro
PPSMC_MSG_PowerDownUmsch = 0x47 # macro
PPSMC_MSG_SetDcsArch = 0x48 # macro
PPSMC_MSG_TriggerVFFLR = 0x49 # macro
PPSMC_MSG_SetNumBadMemoryPagesRetired = 0x4A # macro
PPSMC_MSG_SetBadMemoryPagesRetiredFlagsPerChannel = 0x4B # macro
PPSMC_MSG_SetPriorityDeltaGain = 0x4C # macro
PPSMC_MSG_AllowIHHostInterrupt = 0x4D # macro
PPSMC_MSG_EnableShadowDpm = 0x4E # macro
PPSMC_MSG_Mode3Reset = 0x4F # macro
PPSMC_MSG_SetDriverDramAddr = 0x50 # macro
PPSMC_MSG_SetToolsDramAddr = 0x51 # macro
PPSMC_MSG_TransferTableSmu2DramWithAddr = 0x52 # macro
PPSMC_MSG_TransferTableDram2SmuWithAddr = 0x53 # macro
PPSMC_MSG_GetAllRunningSmuFeatures = 0x54 # macro
PPSMC_MSG_GetSvi3Voltage = 0x55 # macro
PPSMC_MSG_UpdatePolicy = 0x56 # macro
PPSMC_MSG_ExtPwrConnSupport = 0x57 # macro
PPSMC_MSG_PreloadSwPstateForUclkOverDrive = 0x58 # macro
PPSMC_Message_Count = 0x59 # macro
SMU14_DRIVER_IF_V14_0_H = True # macro
PPTABLE_VERSION = 0x1B # macro
NUM_GFXCLK_DPM_LEVELS = 16 # macro
NUM_SOCCLK_DPM_LEVELS = 8 # macro
NUM_MP0CLK_DPM_LEVELS = 2 # macro
NUM_DCLK_DPM_LEVELS = 8 # macro
NUM_VCLK_DPM_LEVELS = 8 # macro
NUM_DISPCLK_DPM_LEVELS = 8 # macro
NUM_DPPCLK_DPM_LEVELS = 8 # macro
NUM_DPREFCLK_DPM_LEVELS = 8 # macro
NUM_DCFCLK_DPM_LEVELS = 8 # macro
NUM_DTBCLK_DPM_LEVELS = 8 # macro
NUM_UCLK_DPM_LEVELS = 6 # macro
NUM_LINK_LEVELS = 3 # macro
NUM_FCLK_DPM_LEVELS = 8 # macro
NUM_OD_FAN_MAX_POINTS = 6 # macro
FEATURE_FW_DATA_READ_BIT = 0 # macro
FEATURE_DPM_GFXCLK_BIT = 1 # macro
FEATURE_DPM_GFX_POWER_OPTIMIZER_BIT = 2 # macro
FEATURE_DPM_UCLK_BIT = 3 # macro
FEATURE_DPM_FCLK_BIT = 4 # macro
FEATURE_DPM_SOCCLK_BIT = 5 # macro
FEATURE_DPM_LINK_BIT = 6 # macro
FEATURE_DPM_DCN_BIT = 7 # macro
FEATURE_VMEMP_SCALING_BIT = 8 # macro
FEATURE_VDDIO_MEM_SCALING_BIT = 9 # macro
FEATURE_DS_DCFCLK_BIT = 14 # macro
FEATURE_DS_UCLK_BIT = 15 # macro
FEATURE_GFX_ULV_BIT = 16 # macro
FEATURE_FW_DSTATE_BIT = 17 # macro
FEATURE_GFXOFF_BIT = 18 # macro
FEATURE_BACO_BIT = 19 # macro
FEATURE_MM_DPM_BIT = 20 # macro
FEATURE_SOC_MPCLK_DS_BIT = 21 # macro
FEATURE_BACO_MPCLK_DS_BIT = 22 # macro
FEATURE_THROTTLERS_BIT = 23 # macro
FEATURE_SMARTSHIFT_BIT = 24 # macro
FEATURE_GTHR_BIT = 25 # macro
FEATURE_ACDC_BIT = 26 # macro
FEATURE_VR0HOT_BIT = 27 # macro
FEATURE_FW_CTF_BIT = 28 # macro
FEATURE_FAN_CONTROL_BIT = 29 # macro
FEATURE_GFX_DCS_BIT = 30 # macro
FEATURE_GFX_READ_MARGIN_BIT = 31 # macro
FEATURE_LED_DISPLAY_BIT = 32 # macro
FEATURE_GFXCLK_SPREAD_SPECTRUM_BIT = 33 # macro
FEATURE_OUT_OF_BAND_MONITOR_BIT = 34 # macro
FEATURE_OPTIMIZED_VMIN_BIT = 35 # macro
FEATURE_GFX_IMU_BIT = 36 # macro
FEATURE_BOOT_TIME_CAL_BIT = 37 # macro
FEATURE_GFX_PCC_DFLL_BIT = 38 # macro
FEATURE_SOC_CG_BIT = 39 # macro
FEATURE_DF_CSTATE_BIT = 40 # macro
FEATURE_GFX_EDC_BIT = 41 # macro
FEATURE_BOOT_POWER_OPT_BIT = 42 # macro
FEATURE_CLOCK_POWER_DOWN_BYPASS_BIT = 43 # macro
FEATURE_BACO_CG_BIT = 45 # macro
FEATURE_MEM_TEMP_READ_BIT = 46 # macro
FEATURE_ATHUB_MMHUB_PG_BIT = 47 # macro
FEATURE_SOC_PCC_BIT = 48 # macro
FEATURE_EDC_PWRBRK_BIT = 49 # macro
FEATURE_SOC_EDC_XVMIN_BIT = 50 # macro
FEATURE_GFX_PSM_DIDT_BIT = 51 # macro
FEATURE_APT_ALL_ENABLE_BIT = 52 # macro
FEATURE_APT_SQ_THROTTLE_BIT = 53 # macro
FEATURE_APT_PF_DCS_BIT = 54 # macro
FEATURE_GFX_EDC_XVMIN_BIT = 55 # macro
FEATURE_GFX_DIDT_XVMIN_BIT = 56 # macro
FEATURE_FAN_ABNORMAL_BIT = 57 # macro
FEATURE_CLOCK_STRETCH_COMPENSATOR = 58 # macro
FEATURE_SPARE_59_BIT = 59 # macro
FEATURE_SPARE_60_BIT = 60 # macro
FEATURE_SPARE_61_BIT = 61 # macro
FEATURE_SPARE_62_BIT = 62 # macro
FEATURE_SPARE_63_BIT = 63 # macro
ALLOWED_FEATURE_CTRL_DEFAULT = 0xFFFFFFFFFFFFFFFF # macro
ALLOWED_FEATURE_CTRL_SCPM = (1<<1) # macro
DEBUG_OVERRIDE_NOT_USE = 0x00000001 # macro
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_DCN_FCLK = 0x00000002 # macro
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_MP0_FCLK = 0x00000004 # macro
DEBUG_OVERRIDE_DISABLE_VOLT_LINK_VCN_DCFCLK = 0x00000008 # macro
DEBUG_OVERRIDE_DISABLE_FAST_FCLK_TIMER = 0x00000010 # macro
DEBUG_OVERRIDE_DISABLE_VCN_PG = 0x00000020 # macro
DEBUG_OVERRIDE_DISABLE_FMAX_VMAX = 0x00000040 # macro
DEBUG_OVERRIDE_DISABLE_IMU_FW_CHECKS = 0x00000080 # macro
DEBUG_OVERRIDE_DISABLE_D0i2_REENTRY_HSR_TIMER_CHECK = 0x00000100 # macro
DEBUG_OVERRIDE_DISABLE_DFLL = 0x00000200 # macro
DEBUG_OVERRIDE_ENABLE_RLC_VF_BRINGUP_MODE = 0x00000400 # macro
DEBUG_OVERRIDE_DFLL_MASTER_MODE = 0x00000800 # macro
DEBUG_OVERRIDE_ENABLE_PROFILING_MODE = 0x00001000 # macro
DEBUG_OVERRIDE_ENABLE_SOC_VF_BRINGUP_MODE = 0x00002000 # macro
DEBUG_OVERRIDE_ENABLE_PER_WGP_RESIENCY = 0x00004000 # macro
DEBUG_OVERRIDE_DISABLE_MEMORY_VOLTAGE_SCALING = 0x00008000 # macro
DEBUG_OVERRIDE_DFLL_BTC_FCW_LOG = 0x00010000 # macro
VR_MAPPING_VR_SELECT_MASK = 0x01 # macro
VR_MAPPING_VR_SELECT_SHIFT = 0x00 # macro
VR_MAPPING_PLANE_SELECT_MASK = 0x02 # macro
VR_MAPPING_PLANE_SELECT_SHIFT = 0x01 # macro
PSI_SEL_VR0_PLANE0_PSI0 = 0x01 # macro
PSI_SEL_VR0_PLANE0_PSI1 = 0x02 # macro
PSI_SEL_VR0_PLANE1_PSI0 = 0x04 # macro
PSI_SEL_VR0_PLANE1_PSI1 = 0x08 # macro
PSI_SEL_VR1_PLANE0_PSI0 = 0x10 # macro
PSI_SEL_VR1_PLANE0_PSI1 = 0x20 # macro
PSI_SEL_VR1_PLANE1_PSI0 = 0x40 # macro
PSI_SEL_VR1_PLANE1_PSI1 = 0x80 # macro
THROTTLER_TEMP_EDGE_BIT = 0 # macro
THROTTLER_TEMP_HOTSPOT_BIT = 1 # macro
THROTTLER_TEMP_HOTSPOT_GFX_BIT = 2 # macro
THROTTLER_TEMP_HOTSPOT_SOC_BIT = 3 # macro
THROTTLER_TEMP_MEM_BIT = 4 # macro
THROTTLER_TEMP_VR_GFX_BIT = 5 # macro
THROTTLER_TEMP_VR_SOC_BIT = 6 # macro
THROTTLER_TEMP_VR_MEM0_BIT = 7 # macro
THROTTLER_TEMP_VR_MEM1_BIT = 8 # macro
THROTTLER_TEMP_LIQUID0_BIT = 9 # macro
THROTTLER_TEMP_LIQUID1_BIT = 10 # macro
THROTTLER_TEMP_PLX_BIT = 11 # macro
THROTTLER_TDC_GFX_BIT = 12 # macro
THROTTLER_TDC_SOC_BIT = 13 # macro
THROTTLER_PPT0_BIT = 14 # macro
THROTTLER_PPT1_BIT = 15 # macro
THROTTLER_PPT2_BIT = 16 # macro
THROTTLER_PPT3_BIT = 17 # macro
THROTTLER_FIT_BIT = 18 # macro
THROTTLER_GFX_APCC_PLUS_BIT = 19 # macro
THROTTLER_GFX_DVO_BIT = 20 # macro
THROTTLER_COUNT = 21 # macro
FW_DSTATE_SOC_ULV_BIT = 0 # macro
FW_DSTATE_G6_HSR_BIT = 1 # macro
FW_DSTATE_G6_PHY_VMEMP_OFF_BIT = 2 # macro
FW_DSTATE_SMN_DS_BIT = 3 # macro
FW_DSTATE_MP1_WHISPER_MODE_BIT = 4 # macro
FW_DSTATE_SOC_LIV_MIN_BIT = 5 # macro
FW_DSTATE_SOC_PLL_PWRDN_BIT = 6 # macro
FW_DSTATE_MEM_PLL_PWRDN_BIT = 7 # macro
FW_DSTATE_MALL_ALLOC_BIT = 8 # macro
FW_DSTATE_MEM_PSI_BIT = 9 # macro
FW_DSTATE_HSR_NON_STROBE_BIT = 10 # macro
FW_DSTATE_MP0_ENTER_WFI_BIT = 11 # macro
FW_DSTATE_MALL_FLUSH_BIT = 12 # macro
FW_DSTATE_SOC_PSI_BIT = 13 # macro
FW_DSTATE_MMHUB_INTERLOCK_BIT = 14 # macro
FW_DSTATE_D0i3_2_QUIET_FW_BIT = 15 # macro
FW_DSTATE_CLDO_PRG_BIT = 16 # macro
FW_DSTATE_DF_PLL_PWRDN_BIT = 17 # macro
LED_DISPLAY_GFX_DPM_BIT = 0 # macro
LED_DISPLAY_PCIE_BIT = 1 # macro
LED_DISPLAY_ERROR_BIT = 2 # macro
MEM_TEMP_READ_OUT_OF_BAND_BIT = 0 # macro
MEM_TEMP_READ_IN_BAND_REFRESH_BIT = 1 # macro
MEM_TEMP_READ_IN_BAND_DUMMY_PSTATE_BIT = 2 # macro
NUM_I2C_CONTROLLERS = 8 # macro
I2C_CONTROLLER_ENABLED = 1 # macro
I2C_CONTROLLER_DISABLED = 0 # macro
MAX_SW_I2C_COMMANDS = 24 # macro
CMDCONFIG_STOP_BIT = 0 # macro
CMDCONFIG_RESTART_BIT = 1 # macro
CMDCONFIG_READWRITE_BIT = 2 # macro
CMDCONFIG_STOP_MASK = (1<<0) # macro
CMDCONFIG_RESTART_MASK = (1<<1) # macro
CMDCONFIG_READWRITE_MASK = (1<<2) # macro
EPCS_HIGH_POWER = 600 # macro
EPCS_NORMAL_POWER = 450 # macro
EPCS_LOW_POWER = 300 # macro
EPCS_SHORTED_POWER = 150 # macro
EPCS_NO_BOOTUP = 0 # macro
PP_NUM_RTAVFS_PWL_ZONES = 5 # macro
PP_NUM_PSM_DIDT_PWL_ZONES = 3 # macro
PP_NUM_OD_VF_CURVE_POINTS = 5 + 1 # macro
PP_OD_FEATURE_GFX_VF_CURVE_BIT = 0 # macro
PP_OD_FEATURE_GFX_VMAX_BIT = 1 # macro
PP_OD_FEATURE_SOC_VMAX_BIT = 2 # macro
PP_OD_FEATURE_PPT_BIT = 3 # macro
PP_OD_FEATURE_FAN_CURVE_BIT = 4 # macro
PP_OD_FEATURE_FAN_LEGACY_BIT = 5 # macro
PP_OD_FEATURE_FULL_CTRL_BIT = 6 # macro
PP_OD_FEATURE_TDC_BIT = 7 # macro
PP_OD_FEATURE_GFXCLK_BIT = 8 # macro
PP_OD_FEATURE_UCLK_BIT = 9 # macro
PP_OD_FEATURE_FCLK_BIT = 10 # macro
PP_OD_FEATURE_ZERO_FAN_BIT = 11 # macro
PP_OD_FEATURE_TEMPERATURE_BIT = 12 # macro
PP_OD_FEATURE_EDC_BIT = 13 # macro
PP_OD_FEATURE_COUNT = 14 # macro
INVALID_BOARD_GPIO = 0xFF # macro
NUM_WM_RANGES = 4 # macro
WORKLOAD_PPLIB_DEFAULT_BIT = 0 # macro
WORKLOAD_PPLIB_FULL_SCREEN_3D_BIT = 1 # macro
WORKLOAD_PPLIB_POWER_SAVING_BIT = 2 # macro
WORKLOAD_PPLIB_VIDEO_BIT = 3 # macro
WORKLOAD_PPLIB_VR_BIT = 4 # macro
WORKLOAD_PPLIB_COMPUTE_BIT = 5 # macro
WORKLOAD_PPLIB_CUSTOM_BIT = 6 # macro
WORKLOAD_PPLIB_WINDOW_3D_BIT = 7 # macro
WORKLOAD_PPLIB_DIRECT_ML_BIT = 8 # macro
WORKLOAD_PPLIB_CGVDI_BIT = 9 # macro
WORKLOAD_PPLIB_COUNT = 10 # macro
TABLE_TRANSFER_OK = 0x0 # macro
TABLE_TRANSFER_FAILED = 0xFF # macro
TABLE_TRANSFER_PENDING = 0xAB # macro
TABLE_PPT_FAILED = 0x100 # macro
TABLE_TDC_FAILED = 0x200 # macro
TABLE_TEMP_FAILED = 0x400 # macro
TABLE_FAN_TARGET_TEMP_FAILED = 0x800 # macro
TABLE_FAN_STOP_TEMP_FAILED = 0x1000 # macro
TABLE_FAN_START_TEMP_FAILED = 0x2000 # macro
TABLE_FAN_PWM_MIN_FAILED = 0x4000 # macro
TABLE_ACOUSTIC_TARGET_RPM_FAILED = 0x8000 # macro
TABLE_ACOUSTIC_LIMIT_RPM_FAILED = 0x10000 # macro
TABLE_MGPU_ACOUSTIC_TARGET_RPM_FAILED = 0x20000 # macro
TABLE_PPTABLE = 0 # macro
TABLE_COMBO_PPTABLE = 1 # macro
TABLE_WATERMARKS = 2 # macro
TABLE_AVFS_PSM_DEBUG = 3 # macro
TABLE_PMSTATUSLOG = 4 # macro
TABLE_SMU_METRICS = 5 # macro
TABLE_DRIVER_SMU_CONFIG = 6 # macro
TABLE_ACTIVITY_MONITOR_COEFF = 7 # macro
TABLE_OVERDRIVE = 8 # macro
TABLE_I2C_COMMANDS = 9 # macro
TABLE_DRIVER_INFO = 10 # macro
TABLE_ECCINFO = 11 # macro
TABLE_CUSTOM_SKUTABLE = 12 # macro
TABLE_COUNT = 13 # macro
IH_INTERRUPT_ID_TO_DRIVER = 0xFE # macro
IH_INTERRUPT_CONTEXT_ID_BACO = 0x2 # macro
IH_INTERRUPT_CONTEXT_ID_AC = 0x3 # macro
IH_INTERRUPT_CONTEXT_ID_DC = 0x4 # macro
IH_INTERRUPT_CONTEXT_ID_AUDIO_D0 = 0x5 # macro
IH_INTERRUPT_CONTEXT_ID_AUDIO_D3 = 0x6 # macro
IH_INTERRUPT_CONTEXT_ID_THERMAL_THROTTLING = 0x7 # macro
IH_INTERRUPT_CONTEXT_ID_FAN_ABNORMAL = 0x8 # macro
IH_INTERRUPT_CONTEXT_ID_FAN_RECOVERY = 0x9 # macro
IH_INTERRUPT_CONTEXT_ID_DYNAMIC_TABLE = 0xA # macro

# values for enumeration 'c__EA_FEATURE_PWR_DOMAIN_e'
c__EA_FEATURE_PWR_DOMAIN_e__enumvalues = {
    0: 'FEATURE_PWR_ALL',
    1: 'FEATURE_PWR_S5',
    2: 'FEATURE_PWR_BACO',
    3: 'FEATURE_PWR_SOC',
    4: 'FEATURE_PWR_GFX',
    5: 'FEATURE_PWR_DOMAIN_COUNT',
}
FEATURE_PWR_ALL = 0
FEATURE_PWR_S5 = 1
FEATURE_PWR_BACO = 2
FEATURE_PWR_SOC = 3
FEATURE_PWR_GFX = 4
FEATURE_PWR_DOMAIN_COUNT = 5
c__EA_FEATURE_PWR_DOMAIN_e = ctypes.c_uint32 # enum
FEATURE_PWR_DOMAIN_e = c__EA_FEATURE_PWR_DOMAIN_e
FEATURE_PWR_DOMAIN_e__enumvalues = c__EA_FEATURE_PWR_DOMAIN_e__enumvalues

# values for enumeration 'c__EA_FEATURE_BTC_e'
c__EA_FEATURE_BTC_e__enumvalues = {
    0: 'FEATURE_BTC_NOP',
    1: 'FEATURE_BTC_SAVE',
    2: 'FEATURE_BTC_RESTORE',
    3: 'FEATURE_BTC_COUNT',
}
FEATURE_BTC_NOP = 0
FEATURE_BTC_SAVE = 1
FEATURE_BTC_RESTORE = 2
FEATURE_BTC_COUNT = 3
c__EA_FEATURE_BTC_e = ctypes.c_uint32 # enum
FEATURE_BTC_e = c__EA_FEATURE_BTC_e
FEATURE_BTC_e__enumvalues = c__EA_FEATURE_BTC_e__enumvalues

# values for enumeration 'c__EA_SVI_PSI_e'
c__EA_SVI_PSI_e__enumvalues = {
    0: 'SVI_PSI_0',
    1: 'SVI_PSI_1',
    2: 'SVI_PSI_2',
    3: 'SVI_PSI_3',
    4: 'SVI_PSI_4',
    5: 'SVI_PSI_5',
    6: 'SVI_PSI_6',
    7: 'SVI_PSI_7',
}
SVI_PSI_0 = 0
SVI_PSI_1 = 1
SVI_PSI_2 = 2
SVI_PSI_3 = 3
SVI_PSI_4 = 4
SVI_PSI_5 = 5
SVI_PSI_6 = 6
SVI_PSI_7 = 7
c__EA_SVI_PSI_e = ctypes.c_uint32 # enum
SVI_PSI_e = c__EA_SVI_PSI_e
SVI_PSI_e__enumvalues = c__EA_SVI_PSI_e__enumvalues

# values for enumeration 'c__EA_SMARTSHIFT_VERSION_e'
c__EA_SMARTSHIFT_VERSION_e__enumvalues = {
    0: 'SMARTSHIFT_VERSION_1',
    1: 'SMARTSHIFT_VERSION_2',
    2: 'SMARTSHIFT_VERSION_3',
}
SMARTSHIFT_VERSION_1 = 0
SMARTSHIFT_VERSION_2 = 1
SMARTSHIFT_VERSION_3 = 2
c__EA_SMARTSHIFT_VERSION_e = ctypes.c_uint32 # enum
SMARTSHIFT_VERSION_e = c__EA_SMARTSHIFT_VERSION_e
SMARTSHIFT_VERSION_e__enumvalues = c__EA_SMARTSHIFT_VERSION_e__enumvalues

# values for enumeration 'c__EA_FOPT_CALC_e'
c__EA_FOPT_CALC_e__enumvalues = {
    0: 'FOPT_CALC_AC_CALC_DC',
    1: 'FOPT_PPTABLE_AC_CALC_DC',
    2: 'FOPT_CALC_AC_PPTABLE_DC',
    3: 'FOPT_PPTABLE_AC_PPTABLE_DC',
}
FOPT_CALC_AC_CALC_DC = 0
FOPT_PPTABLE_AC_CALC_DC = 1
FOPT_CALC_AC_PPTABLE_DC = 2
FOPT_PPTABLE_AC_PPTABLE_DC = 3
c__EA_FOPT_CALC_e = ctypes.c_uint32 # enum
FOPT_CALC_e = c__EA_FOPT_CALC_e
FOPT_CALC_e__enumvalues = c__EA_FOPT_CALC_e__enumvalues

# values for enumeration 'c__EA_DRAM_BIT_WIDTH_TYPE_e'
c__EA_DRAM_BIT_WIDTH_TYPE_e__enumvalues = {
    0: 'DRAM_BIT_WIDTH_DISABLED',
    8: 'DRAM_BIT_WIDTH_X_8',
    16: 'DRAM_BIT_WIDTH_X_16',
    32: 'DRAM_BIT_WIDTH_X_32',
    64: 'DRAM_BIT_WIDTH_X_64',
    128: 'DRAM_BIT_WIDTH_X_128',
    129: 'DRAM_BIT_WIDTH_COUNT',
}
DRAM_BIT_WIDTH_DISABLED = 0
DRAM_BIT_WIDTH_X_8 = 8
DRAM_BIT_WIDTH_X_16 = 16
DRAM_BIT_WIDTH_X_32 = 32
DRAM_BIT_WIDTH_X_64 = 64
DRAM_BIT_WIDTH_X_128 = 128
DRAM_BIT_WIDTH_COUNT = 129
c__EA_DRAM_BIT_WIDTH_TYPE_e = ctypes.c_uint32 # enum
DRAM_BIT_WIDTH_TYPE_e = c__EA_DRAM_BIT_WIDTH_TYPE_e
DRAM_BIT_WIDTH_TYPE_e__enumvalues = c__EA_DRAM_BIT_WIDTH_TYPE_e__enumvalues

# values for enumeration 'c__EA_I2cControllerPort_e'
c__EA_I2cControllerPort_e__enumvalues = {
    0: 'I2C_CONTROLLER_PORT_0',
    1: 'I2C_CONTROLLER_PORT_1',
    2: 'I2C_CONTROLLER_PORT_COUNT',
}
I2C_CONTROLLER_PORT_0 = 0
I2C_CONTROLLER_PORT_1 = 1
I2C_CONTROLLER_PORT_COUNT = 2
c__EA_I2cControllerPort_e = ctypes.c_uint32 # enum
I2cControllerPort_e = c__EA_I2cControllerPort_e
I2cControllerPort_e__enumvalues = c__EA_I2cControllerPort_e__enumvalues

# values for enumeration 'c__EA_I2cControllerName_e'
c__EA_I2cControllerName_e__enumvalues = {
    0: 'I2C_CONTROLLER_NAME_VR_GFX',
    1: 'I2C_CONTROLLER_NAME_VR_SOC',
    2: 'I2C_CONTROLLER_NAME_VR_VMEMP',
    3: 'I2C_CONTROLLER_NAME_VR_VDDIO',
    4: 'I2C_CONTROLLER_NAME_LIQUID0',
    5: 'I2C_CONTROLLER_NAME_LIQUID1',
    6: 'I2C_CONTROLLER_NAME_PLX',
    7: 'I2C_CONTROLLER_NAME_FAN_INTAKE',
    8: 'I2C_CONTROLLER_NAME_COUNT',
}
I2C_CONTROLLER_NAME_VR_GFX = 0
I2C_CONTROLLER_NAME_VR_SOC = 1
I2C_CONTROLLER_NAME_VR_VMEMP = 2
I2C_CONTROLLER_NAME_VR_VDDIO = 3
I2C_CONTROLLER_NAME_LIQUID0 = 4
I2C_CONTROLLER_NAME_LIQUID1 = 5
I2C_CONTROLLER_NAME_PLX = 6
I2C_CONTROLLER_NAME_FAN_INTAKE = 7
I2C_CONTROLLER_NAME_COUNT = 8
c__EA_I2cControllerName_e = ctypes.c_uint32 # enum
I2cControllerName_e = c__EA_I2cControllerName_e
I2cControllerName_e__enumvalues = c__EA_I2cControllerName_e__enumvalues

# values for enumeration 'c__EA_I2cControllerThrottler_e'
c__EA_I2cControllerThrottler_e__enumvalues = {
    0: 'I2C_CONTROLLER_THROTTLER_TYPE_NONE',
    1: 'I2C_CONTROLLER_THROTTLER_VR_GFX',
    2: 'I2C_CONTROLLER_THROTTLER_VR_SOC',
    3: 'I2C_CONTROLLER_THROTTLER_VR_VMEMP',
    4: 'I2C_CONTROLLER_THROTTLER_VR_VDDIO',
    5: 'I2C_CONTROLLER_THROTTLER_LIQUID0',
    6: 'I2C_CONTROLLER_THROTTLER_LIQUID1',
    7: 'I2C_CONTROLLER_THROTTLER_PLX',
    8: 'I2C_CONTROLLER_THROTTLER_FAN_INTAKE',
    9: 'I2C_CONTROLLER_THROTTLER_INA3221',
    10: 'I2C_CONTROLLER_THROTTLER_COUNT',
}
I2C_CONTROLLER_THROTTLER_TYPE_NONE = 0
I2C_CONTROLLER_THROTTLER_VR_GFX = 1
I2C_CONTROLLER_THROTTLER_VR_SOC = 2
I2C_CONTROLLER_THROTTLER_VR_VMEMP = 3
I2C_CONTROLLER_THROTTLER_VR_VDDIO = 4
I2C_CONTROLLER_THROTTLER_LIQUID0 = 5
I2C_CONTROLLER_THROTTLER_LIQUID1 = 6
I2C_CONTROLLER_THROTTLER_PLX = 7
I2C_CONTROLLER_THROTTLER_FAN_INTAKE = 8
I2C_CONTROLLER_THROTTLER_INA3221 = 9
I2C_CONTROLLER_THROTTLER_COUNT = 10
c__EA_I2cControllerThrottler_e = ctypes.c_uint32 # enum
I2cControllerThrottler_e = c__EA_I2cControllerThrottler_e
I2cControllerThrottler_e__enumvalues = c__EA_I2cControllerThrottler_e__enumvalues

# values for enumeration 'c__EA_I2cControllerProtocol_e'
c__EA_I2cControllerProtocol_e__enumvalues = {
    0: 'I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5',
    1: 'I2C_CONTROLLER_PROTOCOL_VR_IR35217',
    2: 'I2C_CONTROLLER_PROTOCOL_TMP_MAX31875',
    3: 'I2C_CONTROLLER_PROTOCOL_INA3221',
    4: 'I2C_CONTROLLER_PROTOCOL_TMP_MAX6604',
    5: 'I2C_CONTROLLER_PROTOCOL_COUNT',
}
I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5 = 0
I2C_CONTROLLER_PROTOCOL_VR_IR35217 = 1
I2C_CONTROLLER_PROTOCOL_TMP_MAX31875 = 2
I2C_CONTROLLER_PROTOCOL_INA3221 = 3
I2C_CONTROLLER_PROTOCOL_TMP_MAX6604 = 4
I2C_CONTROLLER_PROTOCOL_COUNT = 5
c__EA_I2cControllerProtocol_e = ctypes.c_uint32 # enum
I2cControllerProtocol_e = c__EA_I2cControllerProtocol_e
I2cControllerProtocol_e__enumvalues = c__EA_I2cControllerProtocol_e__enumvalues
class struct_c__SA_I2cControllerConfig_t(Structure):
    pass

struct_c__SA_I2cControllerConfig_t._pack_ = 1 # source:False
struct_c__SA_I2cControllerConfig_t._fields_ = [
    ('Enabled', ctypes.c_ubyte),
    ('Speed', ctypes.c_ubyte),
    ('SlaveAddress', ctypes.c_ubyte),
    ('ControllerPort', ctypes.c_ubyte),
    ('ControllerName', ctypes.c_ubyte),
    ('ThermalThrotter', ctypes.c_ubyte),
    ('I2cProtocol', ctypes.c_ubyte),
    ('PaddingConfig', ctypes.c_ubyte),
]

I2cControllerConfig_t = struct_c__SA_I2cControllerConfig_t

# values for enumeration 'c__EA_I2cPort_e'
c__EA_I2cPort_e__enumvalues = {
    0: 'I2C_PORT_SVD_SCL',
    1: 'I2C_PORT_GPIO',
}
I2C_PORT_SVD_SCL = 0
I2C_PORT_GPIO = 1
c__EA_I2cPort_e = ctypes.c_uint32 # enum
I2cPort_e = c__EA_I2cPort_e
I2cPort_e__enumvalues = c__EA_I2cPort_e__enumvalues

# values for enumeration 'c__EA_I2cSpeed_e'
c__EA_I2cSpeed_e__enumvalues = {
    0: 'I2C_SPEED_FAST_50K',
    1: 'I2C_SPEED_FAST_100K',
    2: 'I2C_SPEED_FAST_400K',
    3: 'I2C_SPEED_FAST_PLUS_1M',
    4: 'I2C_SPEED_HIGH_1M',
    5: 'I2C_SPEED_HIGH_2M',
    6: 'I2C_SPEED_COUNT',
}
I2C_SPEED_FAST_50K = 0
I2C_SPEED_FAST_100K = 1
I2C_SPEED_FAST_400K = 2
I2C_SPEED_FAST_PLUS_1M = 3
I2C_SPEED_HIGH_1M = 4
I2C_SPEED_HIGH_2M = 5
I2C_SPEED_COUNT = 6
c__EA_I2cSpeed_e = ctypes.c_uint32 # enum
I2cSpeed_e = c__EA_I2cSpeed_e
I2cSpeed_e__enumvalues = c__EA_I2cSpeed_e__enumvalues

# values for enumeration 'c__EA_I2cCmdType_e'
c__EA_I2cCmdType_e__enumvalues = {
    0: 'I2C_CMD_READ',
    1: 'I2C_CMD_WRITE',
    2: 'I2C_CMD_COUNT',
}
I2C_CMD_READ = 0
I2C_CMD_WRITE = 1
I2C_CMD_COUNT = 2
c__EA_I2cCmdType_e = ctypes.c_uint32 # enum
I2cCmdType_e = c__EA_I2cCmdType_e
I2cCmdType_e__enumvalues = c__EA_I2cCmdType_e__enumvalues
class struct_c__SA_SwI2cCmd_t(Structure):
    pass

struct_c__SA_SwI2cCmd_t._pack_ = 1 # source:False
struct_c__SA_SwI2cCmd_t._fields_ = [
    ('ReadWriteData', ctypes.c_ubyte),
    ('CmdConfig', ctypes.c_ubyte),
]

SwI2cCmd_t = struct_c__SA_SwI2cCmd_t
class struct_c__SA_SwI2cRequest_t(Structure):
    pass

struct_c__SA_SwI2cRequest_t._pack_ = 1 # source:False
struct_c__SA_SwI2cRequest_t._fields_ = [
    ('I2CcontrollerPort', ctypes.c_ubyte),
    ('I2CSpeed', ctypes.c_ubyte),
    ('SlaveAddress', ctypes.c_ubyte),
    ('NumCmds', ctypes.c_ubyte),
    ('SwI2cCmds', struct_c__SA_SwI2cCmd_t * 24),
]

SwI2cRequest_t = struct_c__SA_SwI2cRequest_t
class struct_c__SA_SwI2cRequestExternal_t(Structure):
    pass

struct_c__SA_SwI2cRequestExternal_t._pack_ = 1 # source:False
struct_c__SA_SwI2cRequestExternal_t._fields_ = [
    ('SwI2cRequest', SwI2cRequest_t),
    ('Spare', ctypes.c_uint32 * 8),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

SwI2cRequestExternal_t = struct_c__SA_SwI2cRequestExternal_t
class struct_c__SA_EccInfo_t(Structure):
    pass

struct_c__SA_EccInfo_t._pack_ = 1 # source:False
struct_c__SA_EccInfo_t._fields_ = [
    ('mca_umc_status', ctypes.c_uint64),
    ('mca_umc_addr', ctypes.c_uint64),
    ('ce_count_lo_chip', ctypes.c_uint16),
    ('ce_count_hi_chip', ctypes.c_uint16),
    ('eccPadding', ctypes.c_uint32),
]

EccInfo_t = struct_c__SA_EccInfo_t
class struct_c__SA_EccInfoTable_t(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('EccInfo', struct_c__SA_EccInfo_t * 24),
     ]

EccInfoTable_t = struct_c__SA_EccInfoTable_t

# values for enumeration 'c__EA_EPCS_STATUS_e'
c__EA_EPCS_STATUS_e__enumvalues = {
    0: 'EPCS_SHORTED_LIMIT',
    1: 'EPCS_LOW_POWER_LIMIT',
    2: 'EPCS_NORMAL_POWER_LIMIT',
    3: 'EPCS_HIGH_POWER_LIMIT',
    4: 'EPCS_NOT_CONFIGURED',
    5: 'EPCS_STATUS_COUNT',
}
EPCS_SHORTED_LIMIT = 0
EPCS_LOW_POWER_LIMIT = 1
EPCS_NORMAL_POWER_LIMIT = 2
EPCS_HIGH_POWER_LIMIT = 3
EPCS_NOT_CONFIGURED = 4
EPCS_STATUS_COUNT = 5
c__EA_EPCS_STATUS_e = ctypes.c_uint32 # enum
EPCS_STATUS_e = c__EA_EPCS_STATUS_e
EPCS_STATUS_e__enumvalues = c__EA_EPCS_STATUS_e__enumvalues

# values for enumeration 'c__EA_D3HOTSequence_e'
c__EA_D3HOTSequence_e__enumvalues = {
    0: 'BACO_SEQUENCE',
    1: 'MSR_SEQUENCE',
    2: 'BAMACO_SEQUENCE',
    3: 'ULPS_SEQUENCE',
    4: 'D3HOT_SEQUENCE_COUNT',
}
BACO_SEQUENCE = 0
MSR_SEQUENCE = 1
BAMACO_SEQUENCE = 2
ULPS_SEQUENCE = 3
D3HOT_SEQUENCE_COUNT = 4
c__EA_D3HOTSequence_e = ctypes.c_uint32 # enum
D3HOTSequence_e = c__EA_D3HOTSequence_e
D3HOTSequence_e__enumvalues = c__EA_D3HOTSequence_e__enumvalues

# values for enumeration 'c__EA_PowerGatingMode_e'
c__EA_PowerGatingMode_e__enumvalues = {
    0: 'PG_DYNAMIC_MODE',
    1: 'PG_STATIC_MODE',
}
PG_DYNAMIC_MODE = 0
PG_STATIC_MODE = 1
c__EA_PowerGatingMode_e = ctypes.c_uint32 # enum
PowerGatingMode_e = c__EA_PowerGatingMode_e
PowerGatingMode_e__enumvalues = c__EA_PowerGatingMode_e__enumvalues

# values for enumeration 'c__EA_PowerGatingSettings_e'
c__EA_PowerGatingSettings_e__enumvalues = {
    0: 'PG_POWER_DOWN',
    1: 'PG_POWER_UP',
}
PG_POWER_DOWN = 0
PG_POWER_UP = 1
c__EA_PowerGatingSettings_e = ctypes.c_uint32 # enum
PowerGatingSettings_e = c__EA_PowerGatingSettings_e
PowerGatingSettings_e__enumvalues = c__EA_PowerGatingSettings_e__enumvalues
class struct_c__SA_QuadraticInt_t(Structure):
    pass

struct_c__SA_QuadraticInt_t._pack_ = 1 # source:False
struct_c__SA_QuadraticInt_t._fields_ = [
    ('a', ctypes.c_uint32),
    ('b', ctypes.c_uint32),
    ('c', ctypes.c_uint32),
]

QuadraticInt_t = struct_c__SA_QuadraticInt_t
class struct_c__SA_LinearInt_t(Structure):
    pass

struct_c__SA_LinearInt_t._pack_ = 1 # source:False
struct_c__SA_LinearInt_t._fields_ = [
    ('m', ctypes.c_uint32),
    ('b', ctypes.c_uint32),
]

LinearInt_t = struct_c__SA_LinearInt_t
class struct_c__SA_DroopInt_t(Structure):
    pass

struct_c__SA_DroopInt_t._pack_ = 1 # source:False
struct_c__SA_DroopInt_t._fields_ = [
    ('a', ctypes.c_uint32),
    ('b', ctypes.c_uint32),
    ('c', ctypes.c_uint32),
]

DroopInt_t = struct_c__SA_DroopInt_t

# values for enumeration 'c__EA_DCS_ARCH_e'
c__EA_DCS_ARCH_e__enumvalues = {
    0: 'DCS_ARCH_DISABLED',
    1: 'DCS_ARCH_FADCS',
    2: 'DCS_ARCH_ASYNC',
}
DCS_ARCH_DISABLED = 0
DCS_ARCH_FADCS = 1
DCS_ARCH_ASYNC = 2
c__EA_DCS_ARCH_e = ctypes.c_uint32 # enum
DCS_ARCH_e = c__EA_DCS_ARCH_e
DCS_ARCH_e__enumvalues = c__EA_DCS_ARCH_e__enumvalues

# values for enumeration 'c__EA_PPCLK_e'
c__EA_PPCLK_e__enumvalues = {
    0: 'PPCLK_GFXCLK',
    1: 'PPCLK_SOCCLK',
    2: 'PPCLK_UCLK',
    3: 'PPCLK_FCLK',
    4: 'PPCLK_DCLK_0',
    5: 'PPCLK_VCLK_0',
    6: 'PPCLK_DISPCLK',
    7: 'PPCLK_DPPCLK',
    8: 'PPCLK_DPREFCLK',
    9: 'PPCLK_DCFCLK',
    10: 'PPCLK_DTBCLK',
    11: 'PPCLK_COUNT',
}
PPCLK_GFXCLK = 0
PPCLK_SOCCLK = 1
PPCLK_UCLK = 2
PPCLK_FCLK = 3
PPCLK_DCLK_0 = 4
PPCLK_VCLK_0 = 5
PPCLK_DISPCLK = 6
PPCLK_DPPCLK = 7
PPCLK_DPREFCLK = 8
PPCLK_DCFCLK = 9
PPCLK_DTBCLK = 10
PPCLK_COUNT = 11
c__EA_PPCLK_e = ctypes.c_uint32 # enum
PPCLK_e = c__EA_PPCLK_e
PPCLK_e__enumvalues = c__EA_PPCLK_e__enumvalues

# values for enumeration 'c__EA_VOLTAGE_MODE_e'
c__EA_VOLTAGE_MODE_e__enumvalues = {
    0: 'VOLTAGE_MODE_PPTABLE',
    1: 'VOLTAGE_MODE_FUSES',
    2: 'VOLTAGE_MODE_COUNT',
}
VOLTAGE_MODE_PPTABLE = 0
VOLTAGE_MODE_FUSES = 1
VOLTAGE_MODE_COUNT = 2
c__EA_VOLTAGE_MODE_e = ctypes.c_uint32 # enum
VOLTAGE_MODE_e = c__EA_VOLTAGE_MODE_e
VOLTAGE_MODE_e__enumvalues = c__EA_VOLTAGE_MODE_e__enumvalues

# values for enumeration 'c__EA_AVFS_VOLTAGE_TYPE_e'
c__EA_AVFS_VOLTAGE_TYPE_e__enumvalues = {
    0: 'AVFS_VOLTAGE_GFX',
    1: 'AVFS_VOLTAGE_SOC',
    2: 'AVFS_VOLTAGE_COUNT',
}
AVFS_VOLTAGE_GFX = 0
AVFS_VOLTAGE_SOC = 1
AVFS_VOLTAGE_COUNT = 2
c__EA_AVFS_VOLTAGE_TYPE_e = ctypes.c_uint32 # enum
AVFS_VOLTAGE_TYPE_e = c__EA_AVFS_VOLTAGE_TYPE_e
AVFS_VOLTAGE_TYPE_e__enumvalues = c__EA_AVFS_VOLTAGE_TYPE_e__enumvalues

# values for enumeration 'c__EA_AVFS_TEMP_e'
c__EA_AVFS_TEMP_e__enumvalues = {
    0: 'AVFS_TEMP_COLD',
    1: 'AVFS_TEMP_HOT',
    2: 'AVFS_TEMP_COUNT',
}
AVFS_TEMP_COLD = 0
AVFS_TEMP_HOT = 1
AVFS_TEMP_COUNT = 2
c__EA_AVFS_TEMP_e = ctypes.c_uint32 # enum
AVFS_TEMP_e = c__EA_AVFS_TEMP_e
AVFS_TEMP_e__enumvalues = c__EA_AVFS_TEMP_e__enumvalues

# values for enumeration 'c__EA_AVFS_D_e'
c__EA_AVFS_D_e__enumvalues = {
    0: 'AVFS_D_G',
    1: 'AVFS_D_COUNT',
}
AVFS_D_G = 0
AVFS_D_COUNT = 1
c__EA_AVFS_D_e = ctypes.c_uint32 # enum
AVFS_D_e = c__EA_AVFS_D_e
AVFS_D_e__enumvalues = c__EA_AVFS_D_e__enumvalues

# values for enumeration 'c__EA_UCLK_DIV_e'
c__EA_UCLK_DIV_e__enumvalues = {
    0: 'UCLK_DIV_BY_1',
    1: 'UCLK_DIV_BY_2',
    2: 'UCLK_DIV_BY_4',
    3: 'UCLK_DIV_BY_8',
}
UCLK_DIV_BY_1 = 0
UCLK_DIV_BY_2 = 1
UCLK_DIV_BY_4 = 2
UCLK_DIV_BY_8 = 3
c__EA_UCLK_DIV_e = ctypes.c_uint32 # enum
UCLK_DIV_e = c__EA_UCLK_DIV_e
UCLK_DIV_e__enumvalues = c__EA_UCLK_DIV_e__enumvalues

# values for enumeration 'c__EA_GpioIntPolarity_e'
c__EA_GpioIntPolarity_e__enumvalues = {
    0: 'GPIO_INT_POLARITY_ACTIVE_LOW',
    1: 'GPIO_INT_POLARITY_ACTIVE_HIGH',
}
GPIO_INT_POLARITY_ACTIVE_LOW = 0
GPIO_INT_POLARITY_ACTIVE_HIGH = 1
c__EA_GpioIntPolarity_e = ctypes.c_uint32 # enum
GpioIntPolarity_e = c__EA_GpioIntPolarity_e
GpioIntPolarity_e__enumvalues = c__EA_GpioIntPolarity_e__enumvalues

# values for enumeration 'c__EA_PwrConfig_e'
c__EA_PwrConfig_e__enumvalues = {
    0: 'PWR_CONFIG_TDP',
    1: 'PWR_CONFIG_TGP',
    2: 'PWR_CONFIG_TCP_ESTIMATED',
    3: 'PWR_CONFIG_TCP_MEASURED',
    4: 'PWR_CONFIG_TBP_DESKTOP',
    5: 'PWR_CONFIG_TBP_MOBILE',
}
PWR_CONFIG_TDP = 0
PWR_CONFIG_TGP = 1
PWR_CONFIG_TCP_ESTIMATED = 2
PWR_CONFIG_TCP_MEASURED = 3
PWR_CONFIG_TBP_DESKTOP = 4
PWR_CONFIG_TBP_MOBILE = 5
c__EA_PwrConfig_e = ctypes.c_uint32 # enum
PwrConfig_e = c__EA_PwrConfig_e
PwrConfig_e__enumvalues = c__EA_PwrConfig_e__enumvalues
class struct_c__SA_DpmDescriptor_t(Structure):
    pass

struct_c__SA_DpmDescriptor_t._pack_ = 1 # source:False
struct_c__SA_DpmDescriptor_t._fields_ = [
    ('Padding', ctypes.c_ubyte),
    ('SnapToDiscrete', ctypes.c_ubyte),
    ('NumDiscreteLevels', ctypes.c_ubyte),
    ('CalculateFopt', ctypes.c_ubyte),
    ('ConversionToAvfsClk', LinearInt_t),
    ('Padding3', ctypes.c_uint32 * 3),
    ('Padding4', ctypes.c_uint16),
    ('FoptimalDc', ctypes.c_uint16),
    ('FoptimalAc', ctypes.c_uint16),
    ('Padding2', ctypes.c_uint16),
]

DpmDescriptor_t = struct_c__SA_DpmDescriptor_t

# values for enumeration 'c__EA_PPT_THROTTLER_e'
c__EA_PPT_THROTTLER_e__enumvalues = {
    0: 'PPT_THROTTLER_PPT0',
    1: 'PPT_THROTTLER_PPT1',
    2: 'PPT_THROTTLER_PPT2',
    3: 'PPT_THROTTLER_PPT3',
    4: 'PPT_THROTTLER_COUNT',
}
PPT_THROTTLER_PPT0 = 0
PPT_THROTTLER_PPT1 = 1
PPT_THROTTLER_PPT2 = 2
PPT_THROTTLER_PPT3 = 3
PPT_THROTTLER_COUNT = 4
c__EA_PPT_THROTTLER_e = ctypes.c_uint32 # enum
PPT_THROTTLER_e = c__EA_PPT_THROTTLER_e
PPT_THROTTLER_e__enumvalues = c__EA_PPT_THROTTLER_e__enumvalues

# values for enumeration 'c__EA_TEMP_e'
c__EA_TEMP_e__enumvalues = {
    0: 'TEMP_EDGE',
    1: 'TEMP_HOTSPOT',
    2: 'TEMP_HOTSPOT_GFX',
    3: 'TEMP_HOTSPOT_SOC',
    4: 'TEMP_MEM',
    5: 'TEMP_VR_GFX',
    6: 'TEMP_VR_SOC',
    7: 'TEMP_VR_MEM0',
    8: 'TEMP_VR_MEM1',
    9: 'TEMP_LIQUID0',
    10: 'TEMP_LIQUID1',
    11: 'TEMP_PLX',
    12: 'TEMP_COUNT',
}
TEMP_EDGE = 0
TEMP_HOTSPOT = 1
TEMP_HOTSPOT_GFX = 2
TEMP_HOTSPOT_SOC = 3
TEMP_MEM = 4
TEMP_VR_GFX = 5
TEMP_VR_SOC = 6
TEMP_VR_MEM0 = 7
TEMP_VR_MEM1 = 8
TEMP_LIQUID0 = 9
TEMP_LIQUID1 = 10
TEMP_PLX = 11
TEMP_COUNT = 12
c__EA_TEMP_e = ctypes.c_uint32 # enum
TEMP_e = c__EA_TEMP_e
TEMP_e__enumvalues = c__EA_TEMP_e__enumvalues

# values for enumeration 'c__EA_TDC_THROTTLER_e'
c__EA_TDC_THROTTLER_e__enumvalues = {
    0: 'TDC_THROTTLER_GFX',
    1: 'TDC_THROTTLER_SOC',
    2: 'TDC_THROTTLER_COUNT',
}
TDC_THROTTLER_GFX = 0
TDC_THROTTLER_SOC = 1
TDC_THROTTLER_COUNT = 2
c__EA_TDC_THROTTLER_e = ctypes.c_uint32 # enum
TDC_THROTTLER_e = c__EA_TDC_THROTTLER_e
TDC_THROTTLER_e__enumvalues = c__EA_TDC_THROTTLER_e__enumvalues

# values for enumeration 'c__EA_SVI_PLANE_e'
c__EA_SVI_PLANE_e__enumvalues = {
    0: 'SVI_PLANE_VDD_GFX',
    1: 'SVI_PLANE_VDD_SOC',
    2: 'SVI_PLANE_VDDCI_MEM',
    3: 'SVI_PLANE_VDDIO_MEM',
    4: 'SVI_PLANE_COUNT',
}
SVI_PLANE_VDD_GFX = 0
SVI_PLANE_VDD_SOC = 1
SVI_PLANE_VDDCI_MEM = 2
SVI_PLANE_VDDIO_MEM = 3
SVI_PLANE_COUNT = 4
c__EA_SVI_PLANE_e = ctypes.c_uint32 # enum
SVI_PLANE_e = c__EA_SVI_PLANE_e
SVI_PLANE_e__enumvalues = c__EA_SVI_PLANE_e__enumvalues

# values for enumeration 'c__EA_PMFW_VOLT_PLANE_e'
c__EA_PMFW_VOLT_PLANE_e__enumvalues = {
    0: 'PMFW_VOLT_PLANE_GFX',
    1: 'PMFW_VOLT_PLANE_SOC',
    2: 'PMFW_VOLT_PLANE_COUNT',
}
PMFW_VOLT_PLANE_GFX = 0
PMFW_VOLT_PLANE_SOC = 1
PMFW_VOLT_PLANE_COUNT = 2
c__EA_PMFW_VOLT_PLANE_e = ctypes.c_uint32 # enum
PMFW_VOLT_PLANE_e = c__EA_PMFW_VOLT_PLANE_e
PMFW_VOLT_PLANE_e__enumvalues = c__EA_PMFW_VOLT_PLANE_e__enumvalues

# values for enumeration 'c__EA_CUSTOMER_VARIANT_e'
c__EA_CUSTOMER_VARIANT_e__enumvalues = {
    0: 'CUSTOMER_VARIANT_ROW',
    1: 'CUSTOMER_VARIANT_FALCON',
    2: 'CUSTOMER_VARIANT_COUNT',
}
CUSTOMER_VARIANT_ROW = 0
CUSTOMER_VARIANT_FALCON = 1
CUSTOMER_VARIANT_COUNT = 2
c__EA_CUSTOMER_VARIANT_e = ctypes.c_uint32 # enum
CUSTOMER_VARIANT_e = c__EA_CUSTOMER_VARIANT_e
CUSTOMER_VARIANT_e__enumvalues = c__EA_CUSTOMER_VARIANT_e__enumvalues

# values for enumeration 'c__EA_POWER_SOURCE_e'
c__EA_POWER_SOURCE_e__enumvalues = {
    0: 'POWER_SOURCE_AC',
    1: 'POWER_SOURCE_DC',
    2: 'POWER_SOURCE_COUNT',
}
POWER_SOURCE_AC = 0
POWER_SOURCE_DC = 1
POWER_SOURCE_COUNT = 2
c__EA_POWER_SOURCE_e = ctypes.c_uint32 # enum
POWER_SOURCE_e = c__EA_POWER_SOURCE_e
POWER_SOURCE_e__enumvalues = c__EA_POWER_SOURCE_e__enumvalues

# values for enumeration 'c__EA_MEM_VENDOR_e'
c__EA_MEM_VENDOR_e__enumvalues = {
    0: 'MEM_VENDOR_PLACEHOLDER0',
    1: 'MEM_VENDOR_SAMSUNG',
    2: 'MEM_VENDOR_INFINEON',
    3: 'MEM_VENDOR_ELPIDA',
    4: 'MEM_VENDOR_ETRON',
    5: 'MEM_VENDOR_NANYA',
    6: 'MEM_VENDOR_HYNIX',
    7: 'MEM_VENDOR_MOSEL',
    8: 'MEM_VENDOR_WINBOND',
    9: 'MEM_VENDOR_ESMT',
    10: 'MEM_VENDOR_PLACEHOLDER1',
    11: 'MEM_VENDOR_PLACEHOLDER2',
    12: 'MEM_VENDOR_PLACEHOLDER3',
    13: 'MEM_VENDOR_PLACEHOLDER4',
    14: 'MEM_VENDOR_PLACEHOLDER5',
    15: 'MEM_VENDOR_MICRON',
    16: 'MEM_VENDOR_COUNT',
}
MEM_VENDOR_PLACEHOLDER0 = 0
MEM_VENDOR_SAMSUNG = 1
MEM_VENDOR_INFINEON = 2
MEM_VENDOR_ELPIDA = 3
MEM_VENDOR_ETRON = 4
MEM_VENDOR_NANYA = 5
MEM_VENDOR_HYNIX = 6
MEM_VENDOR_MOSEL = 7
MEM_VENDOR_WINBOND = 8
MEM_VENDOR_ESMT = 9
MEM_VENDOR_PLACEHOLDER1 = 10
MEM_VENDOR_PLACEHOLDER2 = 11
MEM_VENDOR_PLACEHOLDER3 = 12
MEM_VENDOR_PLACEHOLDER4 = 13
MEM_VENDOR_PLACEHOLDER5 = 14
MEM_VENDOR_MICRON = 15
MEM_VENDOR_COUNT = 16
c__EA_MEM_VENDOR_e = ctypes.c_uint32 # enum
MEM_VENDOR_e = c__EA_MEM_VENDOR_e
MEM_VENDOR_e__enumvalues = c__EA_MEM_VENDOR_e__enumvalues

# values for enumeration 'c__EA_PP_GRTAVFS_HW_FUSE_e'
c__EA_PP_GRTAVFS_HW_FUSE_e__enumvalues = {
    0: 'PP_GRTAVFS_HW_CPO_CTL_ZONE0',
    1: 'PP_GRTAVFS_HW_CPO_CTL_ZONE1',
    2: 'PP_GRTAVFS_HW_CPO_CTL_ZONE2',
    3: 'PP_GRTAVFS_HW_CPO_CTL_ZONE3',
    4: 'PP_GRTAVFS_HW_CPO_CTL_ZONE4',
    5: 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE0',
    6: 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE0',
    7: 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE1',
    8: 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE1',
    9: 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE2',
    10: 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE2',
    11: 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE3',
    12: 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE3',
    13: 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE4',
    14: 'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE4',
    15: 'PP_GRTAVFS_HW_ZONE0_VF',
    16: 'PP_GRTAVFS_HW_ZONE1_VF1',
    17: 'PP_GRTAVFS_HW_ZONE2_VF2',
    18: 'PP_GRTAVFS_HW_ZONE3_VF3',
    19: 'PP_GRTAVFS_HW_VOLTAGE_GB',
    20: 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE0',
    21: 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE1',
    22: 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE2',
    23: 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE3',
    24: 'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE4',
    25: 'PP_GRTAVFS_HW_RESERVED_0',
    26: 'PP_GRTAVFS_HW_RESERVED_1',
    27: 'PP_GRTAVFS_HW_RESERVED_2',
    28: 'PP_GRTAVFS_HW_RESERVED_3',
    29: 'PP_GRTAVFS_HW_RESERVED_4',
    30: 'PP_GRTAVFS_HW_RESERVED_5',
    31: 'PP_GRTAVFS_HW_RESERVED_6',
    32: 'PP_GRTAVFS_HW_FUSE_COUNT',
}
PP_GRTAVFS_HW_CPO_CTL_ZONE0 = 0
PP_GRTAVFS_HW_CPO_CTL_ZONE1 = 1
PP_GRTAVFS_HW_CPO_CTL_ZONE2 = 2
PP_GRTAVFS_HW_CPO_CTL_ZONE3 = 3
PP_GRTAVFS_HW_CPO_CTL_ZONE4 = 4
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE0 = 5
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE0 = 6
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE1 = 7
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE1 = 8
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE2 = 9
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE2 = 10
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE3 = 11
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE3 = 12
PP_GRTAVFS_HW_CPO_EN_0_31_ZONE4 = 13
PP_GRTAVFS_HW_CPO_EN_32_63_ZONE4 = 14
PP_GRTAVFS_HW_ZONE0_VF = 15
PP_GRTAVFS_HW_ZONE1_VF1 = 16
PP_GRTAVFS_HW_ZONE2_VF2 = 17
PP_GRTAVFS_HW_ZONE3_VF3 = 18
PP_GRTAVFS_HW_VOLTAGE_GB = 19
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE0 = 20
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE1 = 21
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE2 = 22
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE3 = 23
PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE4 = 24
PP_GRTAVFS_HW_RESERVED_0 = 25
PP_GRTAVFS_HW_RESERVED_1 = 26
PP_GRTAVFS_HW_RESERVED_2 = 27
PP_GRTAVFS_HW_RESERVED_3 = 28
PP_GRTAVFS_HW_RESERVED_4 = 29
PP_GRTAVFS_HW_RESERVED_5 = 30
PP_GRTAVFS_HW_RESERVED_6 = 31
PP_GRTAVFS_HW_FUSE_COUNT = 32
c__EA_PP_GRTAVFS_HW_FUSE_e = ctypes.c_uint32 # enum
PP_GRTAVFS_HW_FUSE_e = c__EA_PP_GRTAVFS_HW_FUSE_e
PP_GRTAVFS_HW_FUSE_e__enumvalues = c__EA_PP_GRTAVFS_HW_FUSE_e__enumvalues

# values for enumeration 'c__EA_PP_GRTAVFS_FW_COMMON_FUSE_e'
c__EA_PP_GRTAVFS_FW_COMMON_FUSE_e__enumvalues = {
    0: 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_HOT_T0',
    1: 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_COLD_T0',
    2: 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_HOT_T0',
    3: 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_COLD_T0',
    4: 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_HOT_T0',
    5: 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_COLD_T0',
    6: 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_HOT_T0',
    7: 'PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_COLD_T0',
    8: 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z0',
    9: 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z1',
    10: 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z2',
    11: 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z3',
    12: 'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z4',
    13: 'PP_GRTAVFS_FW_COMMON_FUSE_COUNT',
}
PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_HOT_T0 = 0
PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_COLD_T0 = 1
PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_HOT_T0 = 2
PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_COLD_T0 = 3
PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_HOT_T0 = 4
PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_COLD_T0 = 5
PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_HOT_T0 = 6
PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_COLD_T0 = 7
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z0 = 8
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z1 = 9
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z2 = 10
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z3 = 11
PP_GRTAVFS_FW_COMMON_SRAM_RM_Z4 = 12
PP_GRTAVFS_FW_COMMON_FUSE_COUNT = 13
c__EA_PP_GRTAVFS_FW_COMMON_FUSE_e = ctypes.c_uint32 # enum
PP_GRTAVFS_FW_COMMON_FUSE_e = c__EA_PP_GRTAVFS_FW_COMMON_FUSE_e
PP_GRTAVFS_FW_COMMON_FUSE_e__enumvalues = c__EA_PP_GRTAVFS_FW_COMMON_FUSE_e__enumvalues

# values for enumeration 'c__EA_PP_GRTAVFS_FW_SEP_FUSE_e'
c__EA_PP_GRTAVFS_FW_SEP_FUSE_e__enumvalues = {
    0: 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_NEG_1',
    1: 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_0',
    2: 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_1',
    3: 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_2',
    4: 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_3',
    5: 'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_4',
    6: 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_NEG_1',
    7: 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_0',
    8: 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_1',
    9: 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_2',
    10: 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_3',
    11: 'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_4',
    12: 'PP_GRTAVFS_FW_SEP_FUSE_VF_NEG_1_FREQUENCY',
    13: 'PP_GRTAVFS_FW_SEP_FUSE_VF4_FREQUENCY',
    14: 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_0',
    15: 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_1',
    16: 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_2',
    17: 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_3',
    18: 'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_4',
    19: 'PP_GRTAVFS_FW_SEP_FUSE_COUNT',
}
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_NEG_1 = 0
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_0 = 1
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_1 = 2
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_2 = 3
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_3 = 4
PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_4 = 5
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_NEG_1 = 6
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_0 = 7
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_1 = 8
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_2 = 9
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_3 = 10
PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_4 = 11
PP_GRTAVFS_FW_SEP_FUSE_VF_NEG_1_FREQUENCY = 12
PP_GRTAVFS_FW_SEP_FUSE_VF4_FREQUENCY = 13
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_0 = 14
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_1 = 15
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_2 = 16
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_3 = 17
PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_4 = 18
PP_GRTAVFS_FW_SEP_FUSE_COUNT = 19
c__EA_PP_GRTAVFS_FW_SEP_FUSE_e = ctypes.c_uint32 # enum
PP_GRTAVFS_FW_SEP_FUSE_e = c__EA_PP_GRTAVFS_FW_SEP_FUSE_e
PP_GRTAVFS_FW_SEP_FUSE_e__enumvalues = c__EA_PP_GRTAVFS_FW_SEP_FUSE_e__enumvalues
class struct_c__SA_SviTelemetryScale_t(Structure):
    pass

struct_c__SA_SviTelemetryScale_t._pack_ = 1 # source:False
struct_c__SA_SviTelemetryScale_t._fields_ = [
    ('Offset', ctypes.c_byte),
    ('Padding', ctypes.c_ubyte),
    ('MaxCurrent', ctypes.c_uint16),
]

SviTelemetryScale_t = struct_c__SA_SviTelemetryScale_t

# values for enumeration 'c__EA_PP_OD_POWER_FEATURE_e'
c__EA_PP_OD_POWER_FEATURE_e__enumvalues = {
    0: 'PP_OD_POWER_FEATURE_ALWAYS_ENABLED',
    1: 'PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING',
    2: 'PP_OD_POWER_FEATURE_ALWAYS_DISABLED',
}
PP_OD_POWER_FEATURE_ALWAYS_ENABLED = 0
PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING = 1
PP_OD_POWER_FEATURE_ALWAYS_DISABLED = 2
c__EA_PP_OD_POWER_FEATURE_e = ctypes.c_uint32 # enum
PP_OD_POWER_FEATURE_e = c__EA_PP_OD_POWER_FEATURE_e
PP_OD_POWER_FEATURE_e__enumvalues = c__EA_PP_OD_POWER_FEATURE_e__enumvalues

# values for enumeration 'c__EA_FanMode_e'
c__EA_FanMode_e__enumvalues = {
    0: 'FAN_MODE_AUTO',
    1: 'FAN_MODE_MANUAL_LINEAR',
}
FAN_MODE_AUTO = 0
FAN_MODE_MANUAL_LINEAR = 1
c__EA_FanMode_e = ctypes.c_uint32 # enum
FanMode_e = c__EA_FanMode_e
FanMode_e__enumvalues = c__EA_FanMode_e__enumvalues

# values for enumeration 'c__EA_OD_FAIL_e'
c__EA_OD_FAIL_e__enumvalues = {
    0: 'OD_NO_ERROR',
    1: 'OD_REQUEST_ADVANCED_NOT_SUPPORTED',
    2: 'OD_UNSUPPORTED_FEATURE',
    3: 'OD_INVALID_FEATURE_COMBO_ERROR',
    4: 'OD_GFXCLK_VF_CURVE_OFFSET_ERROR',
    5: 'OD_VDD_GFX_VMAX_ERROR',
    6: 'OD_VDD_SOC_VMAX_ERROR',
    7: 'OD_PPT_ERROR',
    8: 'OD_FAN_MIN_PWM_ERROR',
    9: 'OD_FAN_ACOUSTIC_TARGET_ERROR',
    10: 'OD_FAN_ACOUSTIC_LIMIT_ERROR',
    11: 'OD_FAN_TARGET_TEMP_ERROR',
    12: 'OD_FAN_ZERO_RPM_STOP_TEMP_ERROR',
    13: 'OD_FAN_CURVE_PWM_ERROR',
    14: 'OD_FAN_CURVE_TEMP_ERROR',
    15: 'OD_FULL_CTRL_GFXCLK_ERROR',
    16: 'OD_FULL_CTRL_UCLK_ERROR',
    17: 'OD_FULL_CTRL_FCLK_ERROR',
    18: 'OD_FULL_CTRL_VDD_GFX_ERROR',
    19: 'OD_FULL_CTRL_VDD_SOC_ERROR',
    20: 'OD_TDC_ERROR',
    21: 'OD_GFXCLK_ERROR',
    22: 'OD_UCLK_ERROR',
    23: 'OD_FCLK_ERROR',
    24: 'OD_OP_TEMP_ERROR',
    25: 'OD_OP_GFX_EDC_ERROR',
    26: 'OD_OP_GFX_PCC_ERROR',
    27: 'OD_POWER_FEATURE_CTRL_ERROR',
}
OD_NO_ERROR = 0
OD_REQUEST_ADVANCED_NOT_SUPPORTED = 1
OD_UNSUPPORTED_FEATURE = 2
OD_INVALID_FEATURE_COMBO_ERROR = 3
OD_GFXCLK_VF_CURVE_OFFSET_ERROR = 4
OD_VDD_GFX_VMAX_ERROR = 5
OD_VDD_SOC_VMAX_ERROR = 6
OD_PPT_ERROR = 7
OD_FAN_MIN_PWM_ERROR = 8
OD_FAN_ACOUSTIC_TARGET_ERROR = 9
OD_FAN_ACOUSTIC_LIMIT_ERROR = 10
OD_FAN_TARGET_TEMP_ERROR = 11
OD_FAN_ZERO_RPM_STOP_TEMP_ERROR = 12
OD_FAN_CURVE_PWM_ERROR = 13
OD_FAN_CURVE_TEMP_ERROR = 14
OD_FULL_CTRL_GFXCLK_ERROR = 15
OD_FULL_CTRL_UCLK_ERROR = 16
OD_FULL_CTRL_FCLK_ERROR = 17
OD_FULL_CTRL_VDD_GFX_ERROR = 18
OD_FULL_CTRL_VDD_SOC_ERROR = 19
OD_TDC_ERROR = 20
OD_GFXCLK_ERROR = 21
OD_UCLK_ERROR = 22
OD_FCLK_ERROR = 23
OD_OP_TEMP_ERROR = 24
OD_OP_GFX_EDC_ERROR = 25
OD_OP_GFX_PCC_ERROR = 26
OD_POWER_FEATURE_CTRL_ERROR = 27
c__EA_OD_FAIL_e = ctypes.c_uint32 # enum
OD_FAIL_e = c__EA_OD_FAIL_e
OD_FAIL_e__enumvalues = c__EA_OD_FAIL_e__enumvalues
class struct_c__SA_OverDriveTable_t(Structure):
    pass

struct_c__SA_OverDriveTable_t._pack_ = 1 # source:False
struct_c__SA_OverDriveTable_t._fields_ = [
    ('FeatureCtrlMask', ctypes.c_uint32),
    ('VoltageOffsetPerZoneBoundary', ctypes.c_int16 * 6),
    ('VddGfxVmax', ctypes.c_uint16),
    ('VddSocVmax', ctypes.c_uint16),
    ('IdlePwrSavingFeaturesCtrl', ctypes.c_ubyte),
    ('RuntimePwrSavingFeaturesCtrl', ctypes.c_ubyte),
    ('Padding', ctypes.c_uint16),
    ('GfxclkFoffset', ctypes.c_int16),
    ('Padding1', ctypes.c_uint16),
    ('UclkFmin', ctypes.c_uint16),
    ('UclkFmax', ctypes.c_uint16),
    ('FclkFmin', ctypes.c_uint16),
    ('FclkFmax', ctypes.c_uint16),
    ('Ppt', ctypes.c_int16),
    ('Tdc', ctypes.c_int16),
    ('FanLinearPwmPoints', ctypes.c_ubyte * 6),
    ('FanLinearTempPoints', ctypes.c_ubyte * 6),
    ('FanMinimumPwm', ctypes.c_uint16),
    ('AcousticTargetRpmThreshold', ctypes.c_uint16),
    ('AcousticLimitRpmThreshold', ctypes.c_uint16),
    ('FanTargetTemperature', ctypes.c_uint16),
    ('FanZeroRpmEnable', ctypes.c_ubyte),
    ('FanZeroRpmStopTemp', ctypes.c_ubyte),
    ('FanMode', ctypes.c_ubyte),
    ('MaxOpTemp', ctypes.c_ubyte),
    ('AdvancedOdModeEnabled', ctypes.c_ubyte),
    ('Padding2', ctypes.c_ubyte * 3),
    ('GfxVoltageFullCtrlMode', ctypes.c_uint16),
    ('SocVoltageFullCtrlMode', ctypes.c_uint16),
    ('GfxclkFullCtrlMode', ctypes.c_uint16),
    ('UclkFullCtrlMode', ctypes.c_uint16),
    ('FclkFullCtrlMode', ctypes.c_uint16),
    ('Padding3', ctypes.c_uint16),
    ('GfxEdc', ctypes.c_int16),
    ('GfxPccLimitControl', ctypes.c_int16),
    ('GfxclkFmaxVmax', ctypes.c_uint16),
    ('GfxclkFmaxVmaxTemperature', ctypes.c_ubyte),
    ('Padding4', ctypes.c_ubyte * 1),
    ('Spare', ctypes.c_uint32 * 9),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

OverDriveTable_t = struct_c__SA_OverDriveTable_t
class struct_c__SA_OverDriveTableExternal_t(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('OverDriveTable', OverDriveTable_t),
     ]

OverDriveTableExternal_t = struct_c__SA_OverDriveTableExternal_t
class struct_c__SA_OverDriveLimits_t(Structure):
    pass

struct_c__SA_OverDriveLimits_t._pack_ = 1 # source:False
struct_c__SA_OverDriveLimits_t._fields_ = [
    ('FeatureCtrlMask', ctypes.c_uint32),
    ('VoltageOffsetPerZoneBoundary', ctypes.c_int16 * 6),
    ('VddGfxVmax', ctypes.c_uint16),
    ('VddSocVmax', ctypes.c_uint16),
    ('GfxclkFoffset', ctypes.c_int16),
    ('Padding', ctypes.c_uint16),
    ('UclkFmin', ctypes.c_uint16),
    ('UclkFmax', ctypes.c_uint16),
    ('FclkFmin', ctypes.c_uint16),
    ('FclkFmax', ctypes.c_uint16),
    ('Ppt', ctypes.c_int16),
    ('Tdc', ctypes.c_int16),
    ('FanLinearPwmPoints', ctypes.c_ubyte * 6),
    ('FanLinearTempPoints', ctypes.c_ubyte * 6),
    ('FanMinimumPwm', ctypes.c_uint16),
    ('AcousticTargetRpmThreshold', ctypes.c_uint16),
    ('AcousticLimitRpmThreshold', ctypes.c_uint16),
    ('FanTargetTemperature', ctypes.c_uint16),
    ('FanZeroRpmEnable', ctypes.c_ubyte),
    ('MaxOpTemp', ctypes.c_ubyte),
    ('Padding1', ctypes.c_ubyte * 2),
    ('GfxVoltageFullCtrlMode', ctypes.c_uint16),
    ('SocVoltageFullCtrlMode', ctypes.c_uint16),
    ('GfxclkFullCtrlMode', ctypes.c_uint16),
    ('UclkFullCtrlMode', ctypes.c_uint16),
    ('FclkFullCtrlMode', ctypes.c_uint16),
    ('GfxEdc', ctypes.c_int16),
    ('GfxPccLimitControl', ctypes.c_int16),
    ('Padding2', ctypes.c_int16),
    ('Spare', ctypes.c_uint32 * 5),
]

OverDriveLimits_t = struct_c__SA_OverDriveLimits_t

# values for enumeration 'c__EA_BOARD_GPIO_TYPE_e'
c__EA_BOARD_GPIO_TYPE_e__enumvalues = {
    0: 'BOARD_GPIO_SMUIO_0',
    1: 'BOARD_GPIO_SMUIO_1',
    2: 'BOARD_GPIO_SMUIO_2',
    3: 'BOARD_GPIO_SMUIO_3',
    4: 'BOARD_GPIO_SMUIO_4',
    5: 'BOARD_GPIO_SMUIO_5',
    6: 'BOARD_GPIO_SMUIO_6',
    7: 'BOARD_GPIO_SMUIO_7',
    8: 'BOARD_GPIO_SMUIO_8',
    9: 'BOARD_GPIO_SMUIO_9',
    10: 'BOARD_GPIO_SMUIO_10',
    11: 'BOARD_GPIO_SMUIO_11',
    12: 'BOARD_GPIO_SMUIO_12',
    13: 'BOARD_GPIO_SMUIO_13',
    14: 'BOARD_GPIO_SMUIO_14',
    15: 'BOARD_GPIO_SMUIO_15',
    16: 'BOARD_GPIO_SMUIO_16',
    17: 'BOARD_GPIO_SMUIO_17',
    18: 'BOARD_GPIO_SMUIO_18',
    19: 'BOARD_GPIO_SMUIO_19',
    20: 'BOARD_GPIO_SMUIO_20',
    21: 'BOARD_GPIO_SMUIO_21',
    22: 'BOARD_GPIO_SMUIO_22',
    23: 'BOARD_GPIO_SMUIO_23',
    24: 'BOARD_GPIO_SMUIO_24',
    25: 'BOARD_GPIO_SMUIO_25',
    26: 'BOARD_GPIO_SMUIO_26',
    27: 'BOARD_GPIO_SMUIO_27',
    28: 'BOARD_GPIO_SMUIO_28',
    29: 'BOARD_GPIO_SMUIO_29',
    30: 'BOARD_GPIO_SMUIO_30',
    31: 'BOARD_GPIO_SMUIO_31',
    32: 'MAX_BOARD_GPIO_SMUIO_NUM',
    33: 'BOARD_GPIO_DC_GEN_A',
    34: 'BOARD_GPIO_DC_GEN_B',
    35: 'BOARD_GPIO_DC_GEN_C',
    36: 'BOARD_GPIO_DC_GEN_D',
    37: 'BOARD_GPIO_DC_GEN_E',
    38: 'BOARD_GPIO_DC_GEN_F',
    39: 'BOARD_GPIO_DC_GEN_G',
    40: 'BOARD_GPIO_DC_GENLK_CLK',
    41: 'BOARD_GPIO_DC_GENLK_VSYNC',
    42: 'BOARD_GPIO_DC_SWAPLOCK_A',
    43: 'BOARD_GPIO_DC_SWAPLOCK_B',
    44: 'MAX_BOARD_DC_GPIO_NUM',
    45: 'BOARD_GPIO_LV_EN',
}
BOARD_GPIO_SMUIO_0 = 0
BOARD_GPIO_SMUIO_1 = 1
BOARD_GPIO_SMUIO_2 = 2
BOARD_GPIO_SMUIO_3 = 3
BOARD_GPIO_SMUIO_4 = 4
BOARD_GPIO_SMUIO_5 = 5
BOARD_GPIO_SMUIO_6 = 6
BOARD_GPIO_SMUIO_7 = 7
BOARD_GPIO_SMUIO_8 = 8
BOARD_GPIO_SMUIO_9 = 9
BOARD_GPIO_SMUIO_10 = 10
BOARD_GPIO_SMUIO_11 = 11
BOARD_GPIO_SMUIO_12 = 12
BOARD_GPIO_SMUIO_13 = 13
BOARD_GPIO_SMUIO_14 = 14
BOARD_GPIO_SMUIO_15 = 15
BOARD_GPIO_SMUIO_16 = 16
BOARD_GPIO_SMUIO_17 = 17
BOARD_GPIO_SMUIO_18 = 18
BOARD_GPIO_SMUIO_19 = 19
BOARD_GPIO_SMUIO_20 = 20
BOARD_GPIO_SMUIO_21 = 21
BOARD_GPIO_SMUIO_22 = 22
BOARD_GPIO_SMUIO_23 = 23
BOARD_GPIO_SMUIO_24 = 24
BOARD_GPIO_SMUIO_25 = 25
BOARD_GPIO_SMUIO_26 = 26
BOARD_GPIO_SMUIO_27 = 27
BOARD_GPIO_SMUIO_28 = 28
BOARD_GPIO_SMUIO_29 = 29
BOARD_GPIO_SMUIO_30 = 30
BOARD_GPIO_SMUIO_31 = 31
MAX_BOARD_GPIO_SMUIO_NUM = 32
BOARD_GPIO_DC_GEN_A = 33
BOARD_GPIO_DC_GEN_B = 34
BOARD_GPIO_DC_GEN_C = 35
BOARD_GPIO_DC_GEN_D = 36
BOARD_GPIO_DC_GEN_E = 37
BOARD_GPIO_DC_GEN_F = 38
BOARD_GPIO_DC_GEN_G = 39
BOARD_GPIO_DC_GENLK_CLK = 40
BOARD_GPIO_DC_GENLK_VSYNC = 41
BOARD_GPIO_DC_SWAPLOCK_A = 42
BOARD_GPIO_DC_SWAPLOCK_B = 43
MAX_BOARD_DC_GPIO_NUM = 44
BOARD_GPIO_LV_EN = 45
c__EA_BOARD_GPIO_TYPE_e = ctypes.c_uint32 # enum
BOARD_GPIO_TYPE_e = c__EA_BOARD_GPIO_TYPE_e
BOARD_GPIO_TYPE_e__enumvalues = c__EA_BOARD_GPIO_TYPE_e__enumvalues
class struct_c__SA_BootValues_t(Structure):
    pass

struct_c__SA_BootValues_t._pack_ = 1 # source:False
struct_c__SA_BootValues_t._fields_ = [
    ('InitImuClk', ctypes.c_uint16),
    ('InitSocclk', ctypes.c_uint16),
    ('InitMpioclk', ctypes.c_uint16),
    ('InitSmnclk', ctypes.c_uint16),
    ('InitDispClk', ctypes.c_uint16),
    ('InitDppClk', ctypes.c_uint16),
    ('InitDprefclk', ctypes.c_uint16),
    ('InitDcfclk', ctypes.c_uint16),
    ('InitDtbclk', ctypes.c_uint16),
    ('InitDbguSocClk', ctypes.c_uint16),
    ('InitGfxclk_bypass', ctypes.c_uint16),
    ('InitMp1clk', ctypes.c_uint16),
    ('InitLclk', ctypes.c_uint16),
    ('InitDbguBacoClk', ctypes.c_uint16),
    ('InitBaco400clk', ctypes.c_uint16),
    ('InitBaco1200clk_bypass', ctypes.c_uint16),
    ('InitBaco700clk_bypass', ctypes.c_uint16),
    ('InitBaco500clk', ctypes.c_uint16),
    ('InitDclk0', ctypes.c_uint16),
    ('InitVclk0', ctypes.c_uint16),
    ('InitFclk', ctypes.c_uint16),
    ('Padding1', ctypes.c_uint16),
    ('InitUclkLevel', ctypes.c_ubyte),
    ('Padding', ctypes.c_ubyte * 3),
    ('InitVcoFreqPll0', ctypes.c_uint32),
    ('InitVcoFreqPll1', ctypes.c_uint32),
    ('InitVcoFreqPll2', ctypes.c_uint32),
    ('InitVcoFreqPll3', ctypes.c_uint32),
    ('InitVcoFreqPll4', ctypes.c_uint32),
    ('InitVcoFreqPll5', ctypes.c_uint32),
    ('InitVcoFreqPll6', ctypes.c_uint32),
    ('InitVcoFreqPll7', ctypes.c_uint32),
    ('InitVcoFreqPll8', ctypes.c_uint32),
    ('InitGfx', ctypes.c_uint16),
    ('InitSoc', ctypes.c_uint16),
    ('InitVddIoMem', ctypes.c_uint16),
    ('InitVddCiMem', ctypes.c_uint16),
    ('Spare', ctypes.c_uint32 * 8),
]

BootValues_t = struct_c__SA_BootValues_t
class struct_c__SA_MsgLimits_t(Structure):
    pass

struct_c__SA_MsgLimits_t._pack_ = 1 # source:False
struct_c__SA_MsgLimits_t._fields_ = [
    ('Power', ctypes.c_uint16 * 2 * 4),
    ('Tdc', ctypes.c_uint16 * 2),
    ('Temperature', ctypes.c_uint16 * 12),
    ('PwmLimitMin', ctypes.c_ubyte),
    ('PwmLimitMax', ctypes.c_ubyte),
    ('FanTargetTemperature', ctypes.c_ubyte),
    ('Spare1', ctypes.c_ubyte * 1),
    ('AcousticTargetRpmThresholdMin', ctypes.c_uint16),
    ('AcousticTargetRpmThresholdMax', ctypes.c_uint16),
    ('AcousticLimitRpmThresholdMin', ctypes.c_uint16),
    ('AcousticLimitRpmThresholdMax', ctypes.c_uint16),
    ('PccLimitMin', ctypes.c_uint16),
    ('PccLimitMax', ctypes.c_uint16),
    ('FanStopTempMin', ctypes.c_uint16),
    ('FanStopTempMax', ctypes.c_uint16),
    ('FanStartTempMin', ctypes.c_uint16),
    ('FanStartTempMax', ctypes.c_uint16),
    ('PowerMinPpt0', ctypes.c_uint16 * 2),
    ('Spare', ctypes.c_uint32 * 11),
]

MsgLimits_t = struct_c__SA_MsgLimits_t
class struct_c__SA_DriverReportedClocks_t(Structure):
    pass

struct_c__SA_DriverReportedClocks_t._pack_ = 1 # source:False
struct_c__SA_DriverReportedClocks_t._fields_ = [
    ('BaseClockAc', ctypes.c_uint16),
    ('GameClockAc', ctypes.c_uint16),
    ('BoostClockAc', ctypes.c_uint16),
    ('BaseClockDc', ctypes.c_uint16),
    ('GameClockDc', ctypes.c_uint16),
    ('BoostClockDc', ctypes.c_uint16),
    ('MaxReportedClock', ctypes.c_uint16),
    ('Padding', ctypes.c_uint16),
    ('Reserved', ctypes.c_uint32 * 3),
]

DriverReportedClocks_t = struct_c__SA_DriverReportedClocks_t
class struct_c__SA_AvfsDcBtcParams_t(Structure):
    pass

struct_c__SA_AvfsDcBtcParams_t._pack_ = 1 # source:False
struct_c__SA_AvfsDcBtcParams_t._fields_ = [
    ('DcBtcEnabled', ctypes.c_ubyte),
    ('Padding', ctypes.c_ubyte * 3),
    ('DcTol', ctypes.c_uint16),
    ('DcBtcGb', ctypes.c_uint16),
    ('DcBtcMin', ctypes.c_uint16),
    ('DcBtcMax', ctypes.c_uint16),
    ('DcBtcGbScalar', LinearInt_t),
]

AvfsDcBtcParams_t = struct_c__SA_AvfsDcBtcParams_t
class struct_c__SA_AvfsFuseOverride_t(Structure):
    pass

struct_c__SA_AvfsFuseOverride_t._pack_ = 1 # source:False
struct_c__SA_AvfsFuseOverride_t._fields_ = [
    ('AvfsTemp', ctypes.c_uint16 * 2),
    ('VftFMin', ctypes.c_uint16),
    ('VInversion', ctypes.c_uint16),
    ('qVft', struct_c__SA_QuadraticInt_t * 2),
    ('qAvfsGb', QuadraticInt_t),
    ('qAvfsGb2', QuadraticInt_t),
]

AvfsFuseOverride_t = struct_c__SA_AvfsFuseOverride_t
class struct_c__SA_PFE_Settings_t(Structure):
    pass

struct_c__SA_PFE_Settings_t._pack_ = 1 # source:False
struct_c__SA_PFE_Settings_t._fields_ = [
    ('Version', ctypes.c_ubyte),
    ('Spare8', ctypes.c_ubyte * 3),
    ('FeaturesToRun', ctypes.c_uint32 * 2),
    ('FwDStateMask', ctypes.c_uint32),
    ('DebugOverrides', ctypes.c_uint32),
    ('Spare', ctypes.c_uint32 * 2),
]

PFE_Settings_t = struct_c__SA_PFE_Settings_t
class struct_c__SA_SkuTable_t(Structure):
    pass

struct_c__SA_SkuTable_t._pack_ = 1 # source:False
struct_c__SA_SkuTable_t._fields_ = [
    ('Version', ctypes.c_uint32),
    ('TotalPowerConfig', ctypes.c_ubyte),
    ('CustomerVariant', ctypes.c_ubyte),
    ('MemoryTemperatureTypeMask', ctypes.c_ubyte),
    ('SmartShiftVersion', ctypes.c_ubyte),
    ('SocketPowerLimitSpare', ctypes.c_ubyte * 10),
    ('EnableLegacyPptLimit', ctypes.c_ubyte),
    ('UseInputTelemetry', ctypes.c_ubyte),
    ('SmartShiftMinReportedPptinDcs', ctypes.c_ubyte),
    ('PaddingPpt', ctypes.c_ubyte * 7),
    ('HwCtfTempLimit', ctypes.c_uint16),
    ('PaddingInfra', ctypes.c_uint16),
    ('FitControllerFailureRateLimit', ctypes.c_uint32),
    ('FitControllerGfxDutyCycle', ctypes.c_uint32),
    ('FitControllerSocDutyCycle', ctypes.c_uint32),
    ('FitControllerSocOffset', ctypes.c_uint32),
    ('GfxApccPlusResidencyLimit', ctypes.c_uint32),
    ('ThrottlerControlMask', ctypes.c_uint32),
    ('UlvVoltageOffset', ctypes.c_uint16 * 2),
    ('Padding', ctypes.c_ubyte * 2),
    ('DeepUlvVoltageOffsetSoc', ctypes.c_uint16),
    ('DefaultMaxVoltage', ctypes.c_uint16 * 2),
    ('BoostMaxVoltage', ctypes.c_uint16 * 2),
    ('VminTempHystersis', ctypes.c_int16 * 2),
    ('VminTempThreshold', ctypes.c_int16 * 2),
    ('Vmin_Hot_T0', ctypes.c_uint16 * 2),
    ('Vmin_Cold_T0', ctypes.c_uint16 * 2),
    ('Vmin_Hot_Eol', ctypes.c_uint16 * 2),
    ('Vmin_Cold_Eol', ctypes.c_uint16 * 2),
    ('Vmin_Aging_Offset', ctypes.c_uint16 * 2),
    ('Spare_Vmin_Plat_Offset_Hot', ctypes.c_uint16 * 2),
    ('Spare_Vmin_Plat_Offset_Cold', ctypes.c_uint16 * 2),
    ('VcBtcFixedVminAgingOffset', ctypes.c_uint16 * 2),
    ('VcBtcVmin2PsmDegrationGb', ctypes.c_uint16 * 2),
    ('VcBtcPsmA', ctypes.c_uint32 * 2),
    ('VcBtcPsmB', ctypes.c_uint32 * 2),
    ('VcBtcVminA', ctypes.c_uint32 * 2),
    ('VcBtcVminB', ctypes.c_uint32 * 2),
    ('PerPartVminEnabled', ctypes.c_ubyte * 2),
    ('VcBtcEnabled', ctypes.c_ubyte * 2),
    ('SocketPowerLimitAcTau', ctypes.c_uint16 * 4),
    ('SocketPowerLimitDcTau', ctypes.c_uint16 * 4),
    ('Gfx_Vmin_droop', QuadraticInt_t),
    ('Soc_Vmin_droop', QuadraticInt_t),
    ('SpareVmin', ctypes.c_uint32 * 6),
    ('DpmDescriptor', struct_c__SA_DpmDescriptor_t * 11),
    ('FreqTableGfx', ctypes.c_uint16 * 16),
    ('FreqTableVclk', ctypes.c_uint16 * 8),
    ('FreqTableDclk', ctypes.c_uint16 * 8),
    ('FreqTableSocclk', ctypes.c_uint16 * 8),
    ('FreqTableUclk', ctypes.c_uint16 * 6),
    ('FreqTableShadowUclk', ctypes.c_uint16 * 6),
    ('FreqTableDispclk', ctypes.c_uint16 * 8),
    ('FreqTableDppClk', ctypes.c_uint16 * 8),
    ('FreqTableDprefclk', ctypes.c_uint16 * 8),
    ('FreqTableDcfclk', ctypes.c_uint16 * 8),
    ('FreqTableDtbclk', ctypes.c_uint16 * 8),
    ('FreqTableFclk', ctypes.c_uint16 * 8),
    ('DcModeMaxFreq', ctypes.c_uint32 * 11),
    ('GfxclkAibFmax', ctypes.c_uint16),
    ('GfxDpmPadding', ctypes.c_uint16),
    ('GfxclkFgfxoffEntry', ctypes.c_uint16),
    ('GfxclkFgfxoffExitImu', ctypes.c_uint16),
    ('GfxclkFgfxoffExitRlc', ctypes.c_uint16),
    ('GfxclkThrottleClock', ctypes.c_uint16),
    ('EnableGfxPowerStagesGpio', ctypes.c_ubyte),
    ('GfxIdlePadding', ctypes.c_ubyte),
    ('SmsRepairWRCKClkDivEn', ctypes.c_ubyte),
    ('SmsRepairWRCKClkDivVal', ctypes.c_ubyte),
    ('GfxOffEntryEarlyMGCGEn', ctypes.c_ubyte),
    ('GfxOffEntryForceCGCGEn', ctypes.c_ubyte),
    ('GfxOffEntryForceCGCGDelayEn', ctypes.c_ubyte),
    ('GfxOffEntryForceCGCGDelayVal', ctypes.c_ubyte),
    ('GfxclkFreqGfxUlv', ctypes.c_uint16),
    ('GfxIdlePadding2', ctypes.c_ubyte * 2),
    ('GfxOffEntryHysteresis', ctypes.c_uint32),
    ('GfxoffSpare', ctypes.c_uint32 * 15),
    ('DfllMstrOscConfigA', ctypes.c_uint16),
    ('DfllSlvOscConfigA', ctypes.c_uint16),
    ('DfllBtcMasterScalerM', ctypes.c_uint32),
    ('DfllBtcMasterScalerB', ctypes.c_int32),
    ('DfllBtcSlaveScalerM', ctypes.c_uint32),
    ('DfllBtcSlaveScalerB', ctypes.c_int32),
    ('DfllPccAsWaitCtrl', ctypes.c_uint32),
    ('DfllPccAsStepCtrl', ctypes.c_uint32),
    ('GfxDfllSpare', ctypes.c_uint32 * 9),
    ('DvoPsmDownThresholdVoltage', ctypes.c_uint32),
    ('DvoPsmUpThresholdVoltage', ctypes.c_uint32),
    ('DvoFmaxLowScaler', ctypes.c_uint32),
    ('PaddingDcs', ctypes.c_uint32),
    ('DcsMinGfxOffTime', ctypes.c_uint16),
    ('DcsMaxGfxOffTime', ctypes.c_uint16),
    ('DcsMinCreditAccum', ctypes.c_uint32),
    ('DcsExitHysteresis', ctypes.c_uint16),
    ('DcsTimeout', ctypes.c_uint16),
    ('DcsPfGfxFopt', ctypes.c_uint32),
    ('DcsPfUclkFopt', ctypes.c_uint32),
    ('FoptEnabled', ctypes.c_ubyte),
    ('DcsSpare2', ctypes.c_ubyte * 3),
    ('DcsFoptM', ctypes.c_uint32),
    ('DcsFoptB', ctypes.c_uint32),
    ('DcsSpare', ctypes.c_uint32 * 9),
    ('UseStrobeModeOptimizations', ctypes.c_ubyte),
    ('PaddingMem', ctypes.c_ubyte * 3),
    ('UclkDpmPstates', ctypes.c_ubyte * 6),
    ('UclkDpmShadowPstates', ctypes.c_ubyte * 6),
    ('FreqTableUclkDiv', ctypes.c_ubyte * 6),
    ('FreqTableShadowUclkDiv', ctypes.c_ubyte * 6),
    ('MemVmempVoltage', ctypes.c_uint16 * 6),
    ('MemVddioVoltage', ctypes.c_uint16 * 6),
    ('DalDcModeMaxUclkFreq', ctypes.c_uint16),
    ('PaddingsMem', ctypes.c_ubyte * 2),
    ('PaddingFclk', ctypes.c_uint32),
    ('PcieGenSpeed', ctypes.c_ubyte * 3),
    ('PcieLaneCount', ctypes.c_ubyte * 3),
    ('LclkFreq', ctypes.c_uint16 * 3),
    ('OverrideGfxAvfsFuses', ctypes.c_ubyte),
    ('GfxAvfsPadding', ctypes.c_ubyte * 1),
    ('DroopGBStDev', ctypes.c_uint16),
    ('SocHwRtAvfsFuses', ctypes.c_uint32 * 32),
    ('GfxL2HwRtAvfsFuses', ctypes.c_uint32 * 32),
    ('PsmDidt_Vcross', ctypes.c_uint16 * 2),
    ('PsmDidt_StaticDroop_A', ctypes.c_uint32 * 3),
    ('PsmDidt_StaticDroop_B', ctypes.c_uint32 * 3),
    ('PsmDidt_DynDroop_A', ctypes.c_uint32 * 3),
    ('PsmDidt_DynDroop_B', ctypes.c_uint32 * 3),
    ('spare_HwRtAvfsFuses', ctypes.c_uint32 * 19),
    ('SocCommonRtAvfs', ctypes.c_uint32 * 13),
    ('GfxCommonRtAvfs', ctypes.c_uint32 * 13),
    ('SocFwRtAvfsFuses', ctypes.c_uint32 * 19),
    ('GfxL2FwRtAvfsFuses', ctypes.c_uint32 * 19),
    ('spare_FwRtAvfsFuses', ctypes.c_uint32 * 19),
    ('Soc_Droop_PWL_F', ctypes.c_uint32 * 5),
    ('Soc_Droop_PWL_a', ctypes.c_uint32 * 5),
    ('Soc_Droop_PWL_b', ctypes.c_uint32 * 5),
    ('Soc_Droop_PWL_c', ctypes.c_uint32 * 5),
    ('Gfx_Droop_PWL_F', ctypes.c_uint32 * 5),
    ('Gfx_Droop_PWL_a', ctypes.c_uint32 * 5),
    ('Gfx_Droop_PWL_b', ctypes.c_uint32 * 5),
    ('Gfx_Droop_PWL_c', ctypes.c_uint32 * 5),
    ('Gfx_Static_PWL_Offset', ctypes.c_uint32 * 5),
    ('Soc_Static_PWL_Offset', ctypes.c_uint32 * 5),
    ('dGbV_dT_vmin', ctypes.c_uint32),
    ('dGbV_dT_vmax', ctypes.c_uint32),
    ('PaddingV2F', ctypes.c_uint32 * 4),
    ('DcBtcGfxParams', AvfsDcBtcParams_t),
    ('SSCurve_GFX', QuadraticInt_t),
    ('GfxAvfsSpare', ctypes.c_uint32 * 29),
    ('OverrideSocAvfsFuses', ctypes.c_ubyte),
    ('MinSocAvfsRevision', ctypes.c_ubyte),
    ('SocAvfsPadding', ctypes.c_ubyte * 2),
    ('SocAvfsFuseOverride', struct_c__SA_AvfsFuseOverride_t * 1),
    ('dBtcGbSoc', struct_c__SA_DroopInt_t * 1),
    ('qAgingGb', struct_c__SA_LinearInt_t * 1),
    ('qStaticVoltageOffset', struct_c__SA_QuadraticInt_t * 1),
    ('DcBtcSocParams', struct_c__SA_AvfsDcBtcParams_t * 1),
    ('SSCurve_SOC', QuadraticInt_t),
    ('SocAvfsSpare', ctypes.c_uint32 * 29),
    ('BootValues', BootValues_t),
    ('DriverReportedClocks', DriverReportedClocks_t),
    ('MsgLimits', MsgLimits_t),
    ('OverDriveLimitsBasicMin', OverDriveLimits_t),
    ('OverDriveLimitsBasicMax', OverDriveLimits_t),
    ('OverDriveLimitsAdvancedMin', OverDriveLimits_t),
    ('OverDriveLimitsAdvancedMax', OverDriveLimits_t),
    ('TotalBoardPowerSupport', ctypes.c_ubyte),
    ('TotalBoardPowerPadding', ctypes.c_ubyte * 1),
    ('TotalBoardPowerRoc', ctypes.c_uint16),
    ('qFeffCoeffGameClock', struct_c__SA_QuadraticInt_t * 2),
    ('qFeffCoeffBaseClock', struct_c__SA_QuadraticInt_t * 2),
    ('qFeffCoeffBoostClock', struct_c__SA_QuadraticInt_t * 2),
    ('AptUclkGfxclkLookup', ctypes.c_int32 * 6 * 2),
    ('AptUclkGfxclkLookupHyst', ctypes.c_uint32 * 6 * 2),
    ('AptPadding', ctypes.c_uint32),
    ('GfxXvminDidtDroopThresh', QuadraticInt_t),
    ('GfxXvminDidtResetDDWait', ctypes.c_uint32),
    ('GfxXvminDidtClkStopWait', ctypes.c_uint32),
    ('GfxXvminDidtFcsStepCtrl', ctypes.c_uint32),
    ('GfxXvminDidtFcsWaitCtrl', ctypes.c_uint32),
    ('PsmModeEnabled', ctypes.c_uint32),
    ('P2v_a', ctypes.c_uint32),
    ('P2v_b', ctypes.c_uint32),
    ('P2v_c', ctypes.c_uint32),
    ('T2p_a', ctypes.c_uint32),
    ('T2p_b', ctypes.c_uint32),
    ('T2p_c', ctypes.c_uint32),
    ('P2vTemp', ctypes.c_uint32),
    ('PsmDidtStaticSettings', QuadraticInt_t),
    ('PsmDidtDynamicSettings', QuadraticInt_t),
    ('PsmDidtAvgDiv', ctypes.c_ubyte),
    ('PsmDidtForceStall', ctypes.c_ubyte),
    ('PsmDidtReleaseTimer', ctypes.c_uint16),
    ('PsmDidtStallPattern', ctypes.c_uint32),
    ('CacEdcCacLeakageC0', ctypes.c_uint32),
    ('CacEdcCacLeakageC1', ctypes.c_uint32),
    ('CacEdcCacLeakageC2', ctypes.c_uint32),
    ('CacEdcCacLeakageC3', ctypes.c_uint32),
    ('CacEdcCacLeakageC4', ctypes.c_uint32),
    ('CacEdcCacLeakageC5', ctypes.c_uint32),
    ('CacEdcGfxClkScalar', ctypes.c_uint32),
    ('CacEdcGfxClkIntercept', ctypes.c_uint32),
    ('CacEdcCac_m', ctypes.c_uint32),
    ('CacEdcCac_b', ctypes.c_uint32),
    ('CacEdcCurrLimitGuardband', ctypes.c_uint32),
    ('CacEdcDynToTotalCacRatio', ctypes.c_uint32),
    ('XVmin_Gfx_EdcThreshScalar', ctypes.c_uint32),
    ('XVmin_Gfx_EdcEnableFreq', ctypes.c_uint32),
    ('XVmin_Gfx_EdcPccAsStepCtrl', ctypes.c_uint32),
    ('XVmin_Gfx_EdcPccAsWaitCtrl', ctypes.c_uint32),
    ('XVmin_Gfx_EdcThreshold', ctypes.c_uint16),
    ('XVmin_Gfx_EdcFiltHysWaitCtrl', ctypes.c_uint16),
    ('XVmin_Soc_EdcThreshScalar', ctypes.c_uint32),
    ('XVmin_Soc_EdcEnableFreq', ctypes.c_uint32),
    ('XVmin_Soc_EdcThreshold', ctypes.c_uint32),
    ('XVmin_Soc_EdcStepUpTime', ctypes.c_uint16),
    ('XVmin_Soc_EdcStepDownTime', ctypes.c_uint16),
    ('XVmin_Soc_EdcInitPccStep', ctypes.c_ubyte),
    ('PaddingSocEdc', ctypes.c_ubyte * 3),
    ('GfxXvminFuseOverride', ctypes.c_ubyte),
    ('SocXvminFuseOverride', ctypes.c_ubyte),
    ('PaddingXvminFuseOverride', ctypes.c_ubyte * 2),
    ('GfxXvminFddTempLow', ctypes.c_ubyte),
    ('GfxXvminFddTempHigh', ctypes.c_ubyte),
    ('SocXvminFddTempLow', ctypes.c_ubyte),
    ('SocXvminFddTempHigh', ctypes.c_ubyte),
    ('GfxXvminFddVolt0', ctypes.c_uint16),
    ('GfxXvminFddVolt1', ctypes.c_uint16),
    ('GfxXvminFddVolt2', ctypes.c_uint16),
    ('SocXvminFddVolt0', ctypes.c_uint16),
    ('SocXvminFddVolt1', ctypes.c_uint16),
    ('SocXvminFddVolt2', ctypes.c_uint16),
    ('GfxXvminDsFddDsm', ctypes.c_uint16 * 6),
    ('GfxXvminEdcFddDsm', ctypes.c_uint16 * 6),
    ('SocXvminEdcFddDsm', ctypes.c_uint16 * 6),
    ('Spare', ctypes.c_uint32),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

SkuTable_t = struct_c__SA_SkuTable_t
class struct_c__SA_Svi3RegulatorSettings_t(Structure):
    pass

struct_c__SA_Svi3RegulatorSettings_t._pack_ = 1 # source:False
struct_c__SA_Svi3RegulatorSettings_t._fields_ = [
    ('SlewRateConditions', ctypes.c_ubyte),
    ('LoadLineAdjust', ctypes.c_ubyte),
    ('VoutOffset', ctypes.c_ubyte),
    ('VidMax', ctypes.c_ubyte),
    ('VidMin', ctypes.c_ubyte),
    ('TenBitTelEn', ctypes.c_ubyte),
    ('SixteenBitTelEn', ctypes.c_ubyte),
    ('OcpThresh', ctypes.c_ubyte),
    ('OcpWarnThresh', ctypes.c_ubyte),
    ('OcpSettings', ctypes.c_ubyte),
    ('VrhotThresh', ctypes.c_ubyte),
    ('OtpThresh', ctypes.c_ubyte),
    ('UvpOvpDeltaRef', ctypes.c_ubyte),
    ('PhaseShed', ctypes.c_ubyte),
    ('Padding', ctypes.c_ubyte * 10),
    ('SettingOverrideMask', ctypes.c_uint32),
]

Svi3RegulatorSettings_t = struct_c__SA_Svi3RegulatorSettings_t
class struct_c__SA_BoardTable_t(Structure):
    pass

struct_c__SA_BoardTable_t._pack_ = 1 # source:False
struct_c__SA_BoardTable_t._fields_ = [
    ('Version', ctypes.c_uint32),
    ('I2cControllers', struct_c__SA_I2cControllerConfig_t * 8),
    ('SlaveAddrMapping', ctypes.c_ubyte * 4),
    ('VrPsiSupport', ctypes.c_ubyte * 4),
    ('Svi3SvcSpeed', ctypes.c_uint32),
    ('EnablePsi6', ctypes.c_ubyte * 4),
    ('Svi3RegSettings', struct_c__SA_Svi3RegulatorSettings_t * 4),
    ('LedOffGpio', ctypes.c_ubyte),
    ('FanOffGpio', ctypes.c_ubyte),
    ('GfxVrPowerStageOffGpio', ctypes.c_ubyte),
    ('AcDcGpio', ctypes.c_ubyte),
    ('AcDcPolarity', ctypes.c_ubyte),
    ('VR0HotGpio', ctypes.c_ubyte),
    ('VR0HotPolarity', ctypes.c_ubyte),
    ('GthrGpio', ctypes.c_ubyte),
    ('GthrPolarity', ctypes.c_ubyte),
    ('LedPin0', ctypes.c_ubyte),
    ('LedPin1', ctypes.c_ubyte),
    ('LedPin2', ctypes.c_ubyte),
    ('LedEnableMask', ctypes.c_ubyte),
    ('LedPcie', ctypes.c_ubyte),
    ('LedError', ctypes.c_ubyte),
    ('PaddingLed', ctypes.c_ubyte),
    ('UclkTrainingModeSpreadPercent', ctypes.c_ubyte),
    ('UclkSpreadPadding', ctypes.c_ubyte),
    ('UclkSpreadFreq', ctypes.c_uint16),
    ('UclkSpreadPercent', ctypes.c_ubyte * 16),
    ('GfxclkSpreadEnable', ctypes.c_ubyte),
    ('FclkSpreadPercent', ctypes.c_ubyte),
    ('FclkSpreadFreq', ctypes.c_uint16),
    ('DramWidth', ctypes.c_ubyte),
    ('PaddingMem1', ctypes.c_ubyte * 7),
    ('HsrEnabled', ctypes.c_ubyte),
    ('VddqOffEnabled', ctypes.c_ubyte),
    ('PaddingUmcFlags', ctypes.c_ubyte * 2),
    ('Paddign1', ctypes.c_uint32),
    ('BacoEntryDelay', ctypes.c_uint32),
    ('FuseWritePowerMuxPresent', ctypes.c_ubyte),
    ('FuseWritePadding', ctypes.c_ubyte * 3),
    ('LoadlineGfx', ctypes.c_uint32),
    ('LoadlineSoc', ctypes.c_uint32),
    ('GfxEdcLimit', ctypes.c_uint32),
    ('SocEdcLimit', ctypes.c_uint32),
    ('RestBoardPower', ctypes.c_uint32),
    ('ConnectorsImpedance', ctypes.c_uint32),
    ('EpcsSens0', ctypes.c_ubyte),
    ('EpcsSens1', ctypes.c_ubyte),
    ('PaddingEpcs', ctypes.c_ubyte * 2),
    ('BoardSpare', ctypes.c_uint32 * 52),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

BoardTable_t = struct_c__SA_BoardTable_t
class struct_c__SA_CustomSkuTable_t(Structure):
    pass

struct_c__SA_CustomSkuTable_t._pack_ = 1 # source:False
struct_c__SA_CustomSkuTable_t._fields_ = [
    ('SocketPowerLimitAc', ctypes.c_uint16 * 4),
    ('VrTdcLimit', ctypes.c_uint16 * 2),
    ('TotalIdleBoardPowerM', ctypes.c_int16),
    ('TotalIdleBoardPowerB', ctypes.c_int16),
    ('TotalBoardPowerM', ctypes.c_int16),
    ('TotalBoardPowerB', ctypes.c_int16),
    ('TemperatureLimit', ctypes.c_uint16 * 12),
    ('FanStopTemp', ctypes.c_uint16 * 12),
    ('FanStartTemp', ctypes.c_uint16 * 12),
    ('FanGain', ctypes.c_uint16 * 12),
    ('FanPwmMin', ctypes.c_uint16),
    ('AcousticTargetRpmThreshold', ctypes.c_uint16),
    ('AcousticLimitRpmThreshold', ctypes.c_uint16),
    ('FanMaximumRpm', ctypes.c_uint16),
    ('MGpuAcousticLimitRpmThreshold', ctypes.c_uint16),
    ('FanTargetGfxclk', ctypes.c_uint16),
    ('TempInputSelectMask', ctypes.c_uint32),
    ('FanZeroRpmEnable', ctypes.c_ubyte),
    ('FanTachEdgePerRev', ctypes.c_ubyte),
    ('FanPadding', ctypes.c_uint16),
    ('FanTargetTemperature', ctypes.c_uint16 * 12),
    ('FuzzyFan_ErrorSetDelta', ctypes.c_int16),
    ('FuzzyFan_ErrorRateSetDelta', ctypes.c_int16),
    ('FuzzyFan_PwmSetDelta', ctypes.c_int16),
    ('FanPadding2', ctypes.c_uint16),
    ('FwCtfLimit', ctypes.c_uint16 * 12),
    ('IntakeTempEnableRPM', ctypes.c_uint16),
    ('IntakeTempOffsetTemp', ctypes.c_int16),
    ('IntakeTempReleaseTemp', ctypes.c_uint16),
    ('IntakeTempHighIntakeAcousticLimit', ctypes.c_uint16),
    ('IntakeTempAcouticLimitReleaseRate', ctypes.c_uint16),
    ('FanAbnormalTempLimitOffset', ctypes.c_int16),
    ('FanStalledTriggerRpm', ctypes.c_uint16),
    ('FanAbnormalTriggerRpmCoeff', ctypes.c_uint16),
    ('FanSpare', ctypes.c_uint16 * 1),
    ('FanIntakeSensorSupport', ctypes.c_ubyte),
    ('FanIntakePadding', ctypes.c_ubyte),
    ('FanSpare2', ctypes.c_uint32 * 12),
    ('ODFeatureCtrlMask', ctypes.c_uint32),
    ('TemperatureLimit_Hynix', ctypes.c_uint16),
    ('TemperatureLimit_Micron', ctypes.c_uint16),
    ('TemperatureFwCtfLimit_Hynix', ctypes.c_uint16),
    ('TemperatureFwCtfLimit_Micron', ctypes.c_uint16),
    ('PlatformTdcLimit', ctypes.c_uint16 * 2),
    ('SocketPowerLimitDc', ctypes.c_uint16 * 4),
    ('SocketPowerLimitSmartShift2', ctypes.c_uint16),
    ('CustomSkuSpare16b', ctypes.c_uint16),
    ('CustomSkuSpare32b', ctypes.c_uint32 * 10),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

CustomSkuTable_t = struct_c__SA_CustomSkuTable_t
class struct_c__SA_PPTable_t(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('PFE_Settings', PFE_Settings_t),
    ('SkuTable', SkuTable_t),
    ('CustomSkuTable', CustomSkuTable_t),
    ('BoardTable', BoardTable_t),
     ]

PPTable_t = struct_c__SA_PPTable_t
class struct_c__SA_DriverSmuConfig_t(Structure):
    pass

struct_c__SA_DriverSmuConfig_t._pack_ = 1 # source:False
struct_c__SA_DriverSmuConfig_t._fields_ = [
    ('GfxclkAverageLpfTau', ctypes.c_uint16),
    ('FclkAverageLpfTau', ctypes.c_uint16),
    ('UclkAverageLpfTau', ctypes.c_uint16),
    ('GfxActivityLpfTau', ctypes.c_uint16),
    ('UclkActivityLpfTau', ctypes.c_uint16),
    ('UclkMaxActivityLpfTau', ctypes.c_uint16),
    ('SocketPowerLpfTau', ctypes.c_uint16),
    ('VcnClkAverageLpfTau', ctypes.c_uint16),
    ('VcnUsageAverageLpfTau', ctypes.c_uint16),
    ('PcieActivityLpTau', ctypes.c_uint16),
]

DriverSmuConfig_t = struct_c__SA_DriverSmuConfig_t
class struct_c__SA_DriverSmuConfigExternal_t(Structure):
    pass

struct_c__SA_DriverSmuConfigExternal_t._pack_ = 1 # source:False
struct_c__SA_DriverSmuConfigExternal_t._fields_ = [
    ('DriverSmuConfig', DriverSmuConfig_t),
    ('Spare', ctypes.c_uint32 * 8),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

DriverSmuConfigExternal_t = struct_c__SA_DriverSmuConfigExternal_t
class struct_c__SA_DriverInfoTable_t(Structure):
    pass

struct_c__SA_DriverInfoTable_t._pack_ = 1 # source:False
struct_c__SA_DriverInfoTable_t._fields_ = [
    ('FreqTableGfx', ctypes.c_uint16 * 16),
    ('FreqTableVclk', ctypes.c_uint16 * 8),
    ('FreqTableDclk', ctypes.c_uint16 * 8),
    ('FreqTableSocclk', ctypes.c_uint16 * 8),
    ('FreqTableUclk', ctypes.c_uint16 * 6),
    ('FreqTableDispclk', ctypes.c_uint16 * 8),
    ('FreqTableDppClk', ctypes.c_uint16 * 8),
    ('FreqTableDprefclk', ctypes.c_uint16 * 8),
    ('FreqTableDcfclk', ctypes.c_uint16 * 8),
    ('FreqTableDtbclk', ctypes.c_uint16 * 8),
    ('FreqTableFclk', ctypes.c_uint16 * 8),
    ('DcModeMaxFreq', ctypes.c_uint16 * 11),
    ('Padding', ctypes.c_uint16),
    ('Spare', ctypes.c_uint32 * 32),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

DriverInfoTable_t = struct_c__SA_DriverInfoTable_t
class struct_c__SA_SmuMetrics_t(Structure):
    pass

struct_c__SA_SmuMetrics_t._pack_ = 1 # source:False
struct_c__SA_SmuMetrics_t._fields_ = [
    ('CurrClock', ctypes.c_uint32 * 11),
    ('AverageGfxclkFrequencyTarget', ctypes.c_uint16),
    ('AverageGfxclkFrequencyPreDs', ctypes.c_uint16),
    ('AverageGfxclkFrequencyPostDs', ctypes.c_uint16),
    ('AverageFclkFrequencyPreDs', ctypes.c_uint16),
    ('AverageFclkFrequencyPostDs', ctypes.c_uint16),
    ('AverageMemclkFrequencyPreDs', ctypes.c_uint16),
    ('AverageMemclkFrequencyPostDs', ctypes.c_uint16),
    ('AverageVclk0Frequency', ctypes.c_uint16),
    ('AverageDclk0Frequency', ctypes.c_uint16),
    ('AverageVclk1Frequency', ctypes.c_uint16),
    ('AverageDclk1Frequency', ctypes.c_uint16),
    ('AveragePCIeBusy', ctypes.c_uint16),
    ('dGPU_W_MAX', ctypes.c_uint16),
    ('padding', ctypes.c_uint16),
    ('MovingAverageGfxclkFrequencyTarget', ctypes.c_uint16),
    ('MovingAverageGfxclkFrequencyPreDs', ctypes.c_uint16),
    ('MovingAverageGfxclkFrequencyPostDs', ctypes.c_uint16),
    ('MovingAverageFclkFrequencyPreDs', ctypes.c_uint16),
    ('MovingAverageFclkFrequencyPostDs', ctypes.c_uint16),
    ('MovingAverageMemclkFrequencyPreDs', ctypes.c_uint16),
    ('MovingAverageMemclkFrequencyPostDs', ctypes.c_uint16),
    ('MovingAverageVclk0Frequency', ctypes.c_uint16),
    ('MovingAverageDclk0Frequency', ctypes.c_uint16),
    ('MovingAverageGfxActivity', ctypes.c_uint16),
    ('MovingAverageUclkActivity', ctypes.c_uint16),
    ('MovingAverageVcn0ActivityPercentage', ctypes.c_uint16),
    ('MovingAveragePCIeBusy', ctypes.c_uint16),
    ('MovingAverageUclkActivity_MAX', ctypes.c_uint16),
    ('MovingAverageSocketPower', ctypes.c_uint16),
    ('MovingAveragePadding', ctypes.c_uint16),
    ('MetricsCounter', ctypes.c_uint32),
    ('AvgVoltage', ctypes.c_uint16 * 4),
    ('AvgCurrent', ctypes.c_uint16 * 4),
    ('AverageGfxActivity', ctypes.c_uint16),
    ('AverageUclkActivity', ctypes.c_uint16),
    ('AverageVcn0ActivityPercentage', ctypes.c_uint16),
    ('Vcn1ActivityPercentage', ctypes.c_uint16),
    ('EnergyAccumulator', ctypes.c_uint32),
    ('AverageSocketPower', ctypes.c_uint16),
    ('AverageTotalBoardPower', ctypes.c_uint16),
    ('AvgTemperature', ctypes.c_uint16 * 12),
    ('AvgTemperatureFanIntake', ctypes.c_uint16),
    ('PcieRate', ctypes.c_ubyte),
    ('PcieWidth', ctypes.c_ubyte),
    ('AvgFanPwm', ctypes.c_ubyte),
    ('Padding', ctypes.c_ubyte * 1),
    ('AvgFanRpm', ctypes.c_uint16),
    ('ThrottlingPercentage', ctypes.c_ubyte * 21),
    ('VmaxThrottlingPercentage', ctypes.c_ubyte),
    ('padding1', ctypes.c_ubyte * 2),
    ('D3HotEntryCountPerMode', ctypes.c_uint32 * 4),
    ('D3HotExitCountPerMode', ctypes.c_uint32 * 4),
    ('ArmMsgReceivedCountPerMode', ctypes.c_uint32 * 4),
    ('ApuSTAPMSmartShiftLimit', ctypes.c_uint16),
    ('ApuSTAPMLimit', ctypes.c_uint16),
    ('AvgApuSocketPower', ctypes.c_uint16),
    ('AverageUclkActivity_MAX', ctypes.c_uint16),
    ('PublicSerialNumberLower', ctypes.c_uint32),
    ('PublicSerialNumberUpper', ctypes.c_uint32),
]

SmuMetrics_t = struct_c__SA_SmuMetrics_t
class struct_c__SA_SmuMetricsExternal_t(Structure):
    pass

struct_c__SA_SmuMetricsExternal_t._pack_ = 1 # source:False
struct_c__SA_SmuMetricsExternal_t._fields_ = [
    ('SmuMetrics', SmuMetrics_t),
    ('Spare', ctypes.c_uint32 * 30),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

SmuMetricsExternal_t = struct_c__SA_SmuMetricsExternal_t
class struct_c__SA_WatermarkRowGeneric_t(Structure):
    pass

struct_c__SA_WatermarkRowGeneric_t._pack_ = 1 # source:False
struct_c__SA_WatermarkRowGeneric_t._fields_ = [
    ('WmSetting', ctypes.c_ubyte),
    ('Flags', ctypes.c_ubyte),
    ('Padding', ctypes.c_ubyte * 2),
]

WatermarkRowGeneric_t = struct_c__SA_WatermarkRowGeneric_t

# values for enumeration 'c__EA_WATERMARKS_FLAGS_e'
c__EA_WATERMARKS_FLAGS_e__enumvalues = {
    0: 'WATERMARKS_CLOCK_RANGE',
    1: 'WATERMARKS_DUMMY_PSTATE',
    2: 'WATERMARKS_MALL',
    3: 'WATERMARKS_COUNT',
}
WATERMARKS_CLOCK_RANGE = 0
WATERMARKS_DUMMY_PSTATE = 1
WATERMARKS_MALL = 2
WATERMARKS_COUNT = 3
c__EA_WATERMARKS_FLAGS_e = ctypes.c_uint32 # enum
WATERMARKS_FLAGS_e = c__EA_WATERMARKS_FLAGS_e
WATERMARKS_FLAGS_e__enumvalues = c__EA_WATERMARKS_FLAGS_e__enumvalues
class struct_c__SA_Watermarks_t(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('WatermarkRow', struct_c__SA_WatermarkRowGeneric_t * 4),
     ]

Watermarks_t = struct_c__SA_Watermarks_t
class struct_c__SA_WatermarksExternal_t(Structure):
    pass

struct_c__SA_WatermarksExternal_t._pack_ = 1 # source:False
struct_c__SA_WatermarksExternal_t._fields_ = [
    ('Watermarks', Watermarks_t),
    ('Spare', ctypes.c_uint32 * 16),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

WatermarksExternal_t = struct_c__SA_WatermarksExternal_t
class struct_c__SA_AvfsDebugTable_t(Structure):
    pass

struct_c__SA_AvfsDebugTable_t._pack_ = 1 # source:False
struct_c__SA_AvfsDebugTable_t._fields_ = [
    ('avgPsmCount', ctypes.c_uint16 * 76),
    ('minPsmCount', ctypes.c_uint16 * 76),
    ('maxPsmCount', ctypes.c_uint16 * 76),
    ('avgPsmVoltage', ctypes.c_float * 76),
    ('minPsmVoltage', ctypes.c_float * 76),
    ('maxPsmVoltage', ctypes.c_float * 76),
]

AvfsDebugTable_t = struct_c__SA_AvfsDebugTable_t
class struct_c__SA_AvfsDebugTableExternal_t(Structure):
    pass

struct_c__SA_AvfsDebugTableExternal_t._pack_ = 1 # source:False
struct_c__SA_AvfsDebugTableExternal_t._fields_ = [
    ('AvfsDebugTable', AvfsDebugTable_t),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

AvfsDebugTableExternal_t = struct_c__SA_AvfsDebugTableExternal_t
class struct_c__SA_DpmActivityMonitorCoeffInt_t(Structure):
    pass

struct_c__SA_DpmActivityMonitorCoeffInt_t._pack_ = 1 # source:False
struct_c__SA_DpmActivityMonitorCoeffInt_t._fields_ = [
    ('Gfx_ActiveHystLimit', ctypes.c_ubyte),
    ('Gfx_IdleHystLimit', ctypes.c_ubyte),
    ('Gfx_FPS', ctypes.c_ubyte),
    ('Gfx_MinActiveFreqType', ctypes.c_ubyte),
    ('Gfx_BoosterFreqType', ctypes.c_ubyte),
    ('PaddingGfx', ctypes.c_ubyte),
    ('Gfx_MinActiveFreq', ctypes.c_uint16),
    ('Gfx_BoosterFreq', ctypes.c_uint16),
    ('Gfx_PD_Data_time_constant', ctypes.c_uint16),
    ('Gfx_PD_Data_limit_a', ctypes.c_uint32),
    ('Gfx_PD_Data_limit_b', ctypes.c_uint32),
    ('Gfx_PD_Data_limit_c', ctypes.c_uint32),
    ('Gfx_PD_Data_error_coeff', ctypes.c_uint32),
    ('Gfx_PD_Data_error_rate_coeff', ctypes.c_uint32),
    ('Fclk_ActiveHystLimit', ctypes.c_ubyte),
    ('Fclk_IdleHystLimit', ctypes.c_ubyte),
    ('Fclk_FPS', ctypes.c_ubyte),
    ('Fclk_MinActiveFreqType', ctypes.c_ubyte),
    ('Fclk_BoosterFreqType', ctypes.c_ubyte),
    ('PaddingFclk', ctypes.c_ubyte),
    ('Fclk_MinActiveFreq', ctypes.c_uint16),
    ('Fclk_BoosterFreq', ctypes.c_uint16),
    ('Fclk_PD_Data_time_constant', ctypes.c_uint16),
    ('Fclk_PD_Data_limit_a', ctypes.c_uint32),
    ('Fclk_PD_Data_limit_b', ctypes.c_uint32),
    ('Fclk_PD_Data_limit_c', ctypes.c_uint32),
    ('Fclk_PD_Data_error_coeff', ctypes.c_uint32),
    ('Fclk_PD_Data_error_rate_coeff', ctypes.c_uint32),
    ('Mem_UpThreshold_Limit', ctypes.c_uint32 * 6),
    ('Mem_UpHystLimit', ctypes.c_ubyte * 6),
    ('Mem_DownHystLimit', ctypes.c_uint16 * 6),
    ('Mem_Fps', ctypes.c_uint16),
]

DpmActivityMonitorCoeffInt_t = struct_c__SA_DpmActivityMonitorCoeffInt_t
class struct_c__SA_DpmActivityMonitorCoeffIntExternal_t(Structure):
    pass

struct_c__SA_DpmActivityMonitorCoeffIntExternal_t._pack_ = 1 # source:False
struct_c__SA_DpmActivityMonitorCoeffIntExternal_t._fields_ = [
    ('DpmActivityMonitorCoeffInt', DpmActivityMonitorCoeffInt_t),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

DpmActivityMonitorCoeffIntExternal_t = struct_c__SA_DpmActivityMonitorCoeffIntExternal_t
__AMDGPU_SMU_H__ = True # macro
int32_t = True # macro
uint32_t = True # macro
int8_t = True # macro
uint8_t = True # macro
uint16_t = True # macro
int16_t = True # macro
uint64_t = True # macro
bool = True # macro
u32 = True # macro
SMU_THERMAL_MINIMUM_ALERT_TEMP = 0 # macro
SMU_THERMAL_MAXIMUM_ALERT_TEMP = 255 # macro
SMU_TEMPERATURE_UNITS_PER_CENTIGRADES = 1000 # macro
SMU_FW_NAME_LEN = 0x24 # macro
SMU_DPM_USER_PROFILE_RESTORE = (1<<0) # macro
SMU_CUSTOM_FAN_SPEED_RPM = (1<<1) # macro
SMU_CUSTOM_FAN_SPEED_PWM = (1<<2) # macro
SMU_THROTTLER_PPT0_BIT = 0 # macro
SMU_THROTTLER_PPT1_BIT = 1 # macro
SMU_THROTTLER_PPT2_BIT = 2 # macro
SMU_THROTTLER_PPT3_BIT = 3 # macro
SMU_THROTTLER_SPL_BIT = 4 # macro
SMU_THROTTLER_FPPT_BIT = 5 # macro
SMU_THROTTLER_SPPT_BIT = 6 # macro
SMU_THROTTLER_SPPT_APU_BIT = 7 # macro
SMU_THROTTLER_TDC_GFX_BIT = 16 # macro
SMU_THROTTLER_TDC_SOC_BIT = 17 # macro
SMU_THROTTLER_TDC_MEM_BIT = 18 # macro
SMU_THROTTLER_TDC_VDD_BIT = 19 # macro
SMU_THROTTLER_TDC_CVIP_BIT = 20 # macro
SMU_THROTTLER_EDC_CPU_BIT = 21 # macro
SMU_THROTTLER_EDC_GFX_BIT = 22 # macro
SMU_THROTTLER_APCC_BIT = 23 # macro
SMU_THROTTLER_TEMP_GPU_BIT = 32 # macro
SMU_THROTTLER_TEMP_CORE_BIT = 33 # macro
SMU_THROTTLER_TEMP_MEM_BIT = 34 # macro
SMU_THROTTLER_TEMP_EDGE_BIT = 35 # macro
SMU_THROTTLER_TEMP_HOTSPOT_BIT = 36 # macro
SMU_THROTTLER_TEMP_SOC_BIT = 37 # macro
SMU_THROTTLER_TEMP_VR_GFX_BIT = 38 # macro
SMU_THROTTLER_TEMP_VR_SOC_BIT = 39 # macro
SMU_THROTTLER_TEMP_VR_MEM0_BIT = 40 # macro
SMU_THROTTLER_TEMP_VR_MEM1_BIT = 41 # macro
SMU_THROTTLER_TEMP_LIQUID0_BIT = 42 # macro
SMU_THROTTLER_TEMP_LIQUID1_BIT = 43 # macro
SMU_THROTTLER_VRHOT0_BIT = 44 # macro
SMU_THROTTLER_VRHOT1_BIT = 45 # macro
SMU_THROTTLER_PROCHOT_CPU_BIT = 46 # macro
SMU_THROTTLER_PROCHOT_GFX_BIT = 47 # macro
SMU_THROTTLER_PPM_BIT = 56 # macro
SMU_THROTTLER_FIT_BIT = 57 # macro
# def SMU_TABLE_INIT(tables, table_id, s, a, d):  # macro
#    return {tables[table_id].size=s;tables[table_id].align=a;tables[table_id].domain=d;}(0)
class struct_smu_hw_power_state(Structure):
    pass

struct_smu_hw_power_state._pack_ = 1 # source:False
struct_smu_hw_power_state._fields_ = [
    ('magic', ctypes.c_uint32),
]

class struct_smu_power_state(Structure):
    pass


# values for enumeration 'smu_state_ui_label'
smu_state_ui_label__enumvalues = {
    0: 'SMU_STATE_UI_LABEL_NONE',
    1: 'SMU_STATE_UI_LABEL_BATTERY',
    2: 'SMU_STATE_UI_TABEL_MIDDLE_LOW',
    3: 'SMU_STATE_UI_LABEL_BALLANCED',
    4: 'SMU_STATE_UI_LABEL_MIDDLE_HIGHT',
    5: 'SMU_STATE_UI_LABEL_PERFORMANCE',
    6: 'SMU_STATE_UI_LABEL_BACO',
}
SMU_STATE_UI_LABEL_NONE = 0
SMU_STATE_UI_LABEL_BATTERY = 1
SMU_STATE_UI_TABEL_MIDDLE_LOW = 2
SMU_STATE_UI_LABEL_BALLANCED = 3
SMU_STATE_UI_LABEL_MIDDLE_HIGHT = 4
SMU_STATE_UI_LABEL_PERFORMANCE = 5
SMU_STATE_UI_LABEL_BACO = 6
smu_state_ui_label = ctypes.c_uint32 # enum

# values for enumeration 'smu_state_classification_flag'
smu_state_classification_flag__enumvalues = {
    1: 'SMU_STATE_CLASSIFICATION_FLAG_BOOT',
    2: 'SMU_STATE_CLASSIFICATION_FLAG_THERMAL',
    4: 'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE',
    8: 'SMU_STATE_CLASSIFICATION_FLAG_RESET',
    16: 'SMU_STATE_CLASSIFICATION_FLAG_FORCED',
    32: 'SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE',
    64: 'SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE',
    128: 'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE',
    256: 'SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE',
    512: 'SMU_STATE_CLASSIFICATION_FLAG_UVD',
    1024: 'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW',
    2048: 'SMU_STATE_CLASSIFICATION_FLAG_ACPI',
    4096: 'SMU_STATE_CLASSIFICATION_FLAG_HD2',
    8192: 'SMU_STATE_CLASSIFICATION_FLAG_UVD_HD',
    16384: 'SMU_STATE_CLASSIFICATION_FLAG_UVD_SD',
    32768: 'SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE',
    65536: 'SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE',
    131072: 'SMU_STATE_CLASSIFICATION_FLAG_BACO',
    262144: 'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2',
    524288: 'SMU_STATE_CLASSIFICATION_FLAG_ULV',
    1048576: 'SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC',
}
SMU_STATE_CLASSIFICATION_FLAG_BOOT = 1
SMU_STATE_CLASSIFICATION_FLAG_THERMAL = 2
SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE = 4
SMU_STATE_CLASSIFICATION_FLAG_RESET = 8
SMU_STATE_CLASSIFICATION_FLAG_FORCED = 16
SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE = 32
SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE = 64
SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE = 128
SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE = 256
SMU_STATE_CLASSIFICATION_FLAG_UVD = 512
SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW = 1024
SMU_STATE_CLASSIFICATION_FLAG_ACPI = 2048
SMU_STATE_CLASSIFICATION_FLAG_HD2 = 4096
SMU_STATE_CLASSIFICATION_FLAG_UVD_HD = 8192
SMU_STATE_CLASSIFICATION_FLAG_UVD_SD = 16384
SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE = 32768
SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE = 65536
SMU_STATE_CLASSIFICATION_FLAG_BACO = 131072
SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2 = 262144
SMU_STATE_CLASSIFICATION_FLAG_ULV = 524288
SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC = 1048576
smu_state_classification_flag = ctypes.c_uint32 # enum
class struct_smu_state_classification_block(Structure):
    pass

struct_smu_state_classification_block._pack_ = 1 # source:False
struct_smu_state_classification_block._fields_ = [
    ('ui_label', smu_state_ui_label),
    ('flags', smu_state_classification_flag),
    ('bios_index', ctypes.c_int32),
    ('temporary_state', ctypes.c_bool),
    ('to_be_deleted', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

class struct_smu_state_pcie_block(Structure):
    pass

struct_smu_state_pcie_block._pack_ = 1 # source:False
struct_smu_state_pcie_block._fields_ = [
    ('lanes', ctypes.c_uint32),
]


# values for enumeration 'smu_refreshrate_source'
smu_refreshrate_source__enumvalues = {
    0: 'SMU_REFRESHRATE_SOURCE_EDID',
    1: 'SMU_REFRESHRATE_SOURCE_EXPLICIT',
}
SMU_REFRESHRATE_SOURCE_EDID = 0
SMU_REFRESHRATE_SOURCE_EXPLICIT = 1
smu_refreshrate_source = ctypes.c_uint32 # enum
class struct_smu_state_display_block(Structure):
    pass

struct_smu_state_display_block._pack_ = 1 # source:False
struct_smu_state_display_block._fields_ = [
    ('disable_frame_modulation', ctypes.c_bool),
    ('limit_refreshrate', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('refreshrate_source', smu_refreshrate_source),
    ('explicit_refreshrate', ctypes.c_int32),
    ('edid_refreshrate_index', ctypes.c_int32),
    ('enable_vari_bright', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
]

class struct_smu_state_memory_block(Structure):
    pass

struct_smu_state_memory_block._pack_ = 1 # source:False
struct_smu_state_memory_block._fields_ = [
    ('dll_off', ctypes.c_bool),
    ('m3arb', ctypes.c_ubyte),
    ('unused', ctypes.c_ubyte * 3),
]

class struct_smu_state_software_algorithm_block(Structure):
    pass

struct_smu_state_software_algorithm_block._pack_ = 1 # source:False
struct_smu_state_software_algorithm_block._fields_ = [
    ('disable_load_balancing', ctypes.c_bool),
    ('enable_sleep_for_timestamps', ctypes.c_bool),
]

class struct_smu_temperature_range(Structure):
    pass

struct_smu_temperature_range._pack_ = 1 # source:False
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

class struct_smu_state_validation_block(Structure):
    pass

struct_smu_state_validation_block._pack_ = 1 # source:False
struct_smu_state_validation_block._fields_ = [
    ('single_display_only', ctypes.c_bool),
    ('disallow_on_dc', ctypes.c_bool),
    ('supported_power_levels', ctypes.c_ubyte),
]

class struct_smu_uvd_clocks(Structure):
    pass

struct_smu_uvd_clocks._pack_ = 1 # source:False
struct_smu_uvd_clocks._fields_ = [
    ('vclk', ctypes.c_uint32),
    ('dclk', ctypes.c_uint32),
]


# values for enumeration 'smu_power_src_type'
smu_power_src_type__enumvalues = {
    0: 'SMU_POWER_SOURCE_AC',
    1: 'SMU_POWER_SOURCE_DC',
    2: 'SMU_POWER_SOURCE_COUNT',
}
SMU_POWER_SOURCE_AC = 0
SMU_POWER_SOURCE_DC = 1
SMU_POWER_SOURCE_COUNT = 2
smu_power_src_type = ctypes.c_uint32 # enum

# values for enumeration 'smu_ppt_limit_type'
smu_ppt_limit_type__enumvalues = {
    0: 'SMU_DEFAULT_PPT_LIMIT',
    1: 'SMU_FAST_PPT_LIMIT',
}
SMU_DEFAULT_PPT_LIMIT = 0
SMU_FAST_PPT_LIMIT = 1
smu_ppt_limit_type = ctypes.c_uint32 # enum

# values for enumeration 'smu_ppt_limit_level'
smu_ppt_limit_level__enumvalues = {
    -1: 'SMU_PPT_LIMIT_MIN',
    0: 'SMU_PPT_LIMIT_CURRENT',
    1: 'SMU_PPT_LIMIT_DEFAULT',
    2: 'SMU_PPT_LIMIT_MAX',
}
SMU_PPT_LIMIT_MIN = -1
SMU_PPT_LIMIT_CURRENT = 0
SMU_PPT_LIMIT_DEFAULT = 1
SMU_PPT_LIMIT_MAX = 2
smu_ppt_limit_level = ctypes.c_int32 # enum

# values for enumeration 'smu_memory_pool_size'
smu_memory_pool_size__enumvalues = {
    0: 'SMU_MEMORY_POOL_SIZE_ZERO',
    268435456: 'SMU_MEMORY_POOL_SIZE_256_MB',
    536870912: 'SMU_MEMORY_POOL_SIZE_512_MB',
    1073741824: 'SMU_MEMORY_POOL_SIZE_1_GB',
    2147483648: 'SMU_MEMORY_POOL_SIZE_2_GB',
}
SMU_MEMORY_POOL_SIZE_ZERO = 0
SMU_MEMORY_POOL_SIZE_256_MB = 268435456
SMU_MEMORY_POOL_SIZE_512_MB = 536870912
SMU_MEMORY_POOL_SIZE_1_GB = 1073741824
SMU_MEMORY_POOL_SIZE_2_GB = 2147483648
smu_memory_pool_size = ctypes.c_uint32 # enum

# values for enumeration 'smu_clk_type'
smu_clk_type__enumvalues = {
    0: 'SMU_GFXCLK',
    1: 'SMU_VCLK',
    2: 'SMU_DCLK',
    3: 'SMU_VCLK1',
    4: 'SMU_DCLK1',
    5: 'SMU_ECLK',
    6: 'SMU_SOCCLK',
    7: 'SMU_UCLK',
    8: 'SMU_DCEFCLK',
    9: 'SMU_DISPCLK',
    10: 'SMU_PIXCLK',
    11: 'SMU_PHYCLK',
    12: 'SMU_FCLK',
    13: 'SMU_SCLK',
    14: 'SMU_MCLK',
    15: 'SMU_PCIE',
    16: 'SMU_LCLK',
    17: 'SMU_OD_CCLK',
    18: 'SMU_OD_SCLK',
    19: 'SMU_OD_MCLK',
    20: 'SMU_OD_VDDC_CURVE',
    21: 'SMU_OD_RANGE',
    22: 'SMU_OD_VDDGFX_OFFSET',
    23: 'SMU_OD_FAN_CURVE',
    24: 'SMU_OD_ACOUSTIC_LIMIT',
    25: 'SMU_OD_ACOUSTIC_TARGET',
    26: 'SMU_OD_FAN_TARGET_TEMPERATURE',
    27: 'SMU_OD_FAN_MINIMUM_PWM',
    28: 'SMU_CLK_COUNT',
}
SMU_GFXCLK = 0
SMU_VCLK = 1
SMU_DCLK = 2
SMU_VCLK1 = 3
SMU_DCLK1 = 4
SMU_ECLK = 5
SMU_SOCCLK = 6
SMU_UCLK = 7
SMU_DCEFCLK = 8
SMU_DISPCLK = 9
SMU_PIXCLK = 10
SMU_PHYCLK = 11
SMU_FCLK = 12
SMU_SCLK = 13
SMU_MCLK = 14
SMU_PCIE = 15
SMU_LCLK = 16
SMU_OD_CCLK = 17
SMU_OD_SCLK = 18
SMU_OD_MCLK = 19
SMU_OD_VDDC_CURVE = 20
SMU_OD_RANGE = 21
SMU_OD_VDDGFX_OFFSET = 22
SMU_OD_FAN_CURVE = 23
SMU_OD_ACOUSTIC_LIMIT = 24
SMU_OD_ACOUSTIC_TARGET = 25
SMU_OD_FAN_TARGET_TEMPERATURE = 26
SMU_OD_FAN_MINIMUM_PWM = 27
SMU_CLK_COUNT = 28
smu_clk_type = ctypes.c_uint32 # enum
class struct_smu_user_dpm_profile(Structure):
    pass

struct_smu_user_dpm_profile._pack_ = 1 # source:False
struct_smu_user_dpm_profile._fields_ = [
    ('fan_mode', ctypes.c_uint32),
    ('power_limit', ctypes.c_uint32),
    ('fan_speed_pwm', ctypes.c_uint32),
    ('fan_speed_rpm', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('user_od', ctypes.c_uint32),
    ('clk_mask', ctypes.c_uint32 * 28),
    ('clk_dependency', ctypes.c_uint32),
]

class struct_smu_table(Structure):
    pass

class struct_amdgpu_bo(Structure):
    pass

struct_smu_table._pack_ = 1 # source:False
struct_smu_table._fields_ = [
    ('size', ctypes.c_uint64),
    ('align', ctypes.c_uint32),
    ('domain', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('mc_address', ctypes.c_uint64),
    ('cpu_addr', ctypes.POINTER(None)),
    ('bo', ctypes.POINTER(struct_amdgpu_bo)),
    ('version', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]


# values for enumeration 'smu_perf_level_designation'
smu_perf_level_designation__enumvalues = {
    0: 'PERF_LEVEL_ACTIVITY',
    1: 'PERF_LEVEL_POWER_CONTAINMENT',
}
PERF_LEVEL_ACTIVITY = 0
PERF_LEVEL_POWER_CONTAINMENT = 1
smu_perf_level_designation = ctypes.c_uint32 # enum
class struct_smu_performance_level(Structure):
    pass

struct_smu_performance_level._pack_ = 1 # source:False
struct_smu_performance_level._fields_ = [
    ('core_clock', ctypes.c_uint32),
    ('memory_clock', ctypes.c_uint32),
    ('vddc', ctypes.c_uint32),
    ('vddci', ctypes.c_uint32),
    ('non_local_mem_freq', ctypes.c_uint32),
    ('non_local_mem_width', ctypes.c_uint32),
]

class struct_smu_clock_info(Structure):
    pass

struct_smu_clock_info._pack_ = 1 # source:False
struct_smu_clock_info._fields_ = [
    ('min_mem_clk', ctypes.c_uint32),
    ('max_mem_clk', ctypes.c_uint32),
    ('min_eng_clk', ctypes.c_uint32),
    ('max_eng_clk', ctypes.c_uint32),
    ('min_bus_bandwidth', ctypes.c_uint32),
    ('max_bus_bandwidth', ctypes.c_uint32),
]

class struct_smu_bios_boot_up_values(Structure):
    pass

struct_smu_bios_boot_up_values._pack_ = 1 # source:False
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
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('pp_table_id', ctypes.c_uint32),
    ('format_revision', ctypes.c_uint32),
    ('content_revision', ctypes.c_uint32),
    ('fclk', ctypes.c_uint32),
    ('lclk', ctypes.c_uint32),
    ('firmware_caps', ctypes.c_uint32),
]


# values for enumeration 'smu_table_id'
smu_table_id__enumvalues = {
    0: 'SMU_TABLE_PPTABLE',
    1: 'SMU_TABLE_WATERMARKS',
    2: 'SMU_TABLE_CUSTOM_DPM',
    3: 'SMU_TABLE_DPMCLOCKS',
    4: 'SMU_TABLE_AVFS',
    5: 'SMU_TABLE_AVFS_PSM_DEBUG',
    6: 'SMU_TABLE_AVFS_FUSE_OVERRIDE',
    7: 'SMU_TABLE_PMSTATUSLOG',
    8: 'SMU_TABLE_SMU_METRICS',
    9: 'SMU_TABLE_DRIVER_SMU_CONFIG',
    10: 'SMU_TABLE_ACTIVITY_MONITOR_COEFF',
    11: 'SMU_TABLE_OVERDRIVE',
    12: 'SMU_TABLE_I2C_COMMANDS',
    13: 'SMU_TABLE_PACE',
    14: 'SMU_TABLE_ECCINFO',
    15: 'SMU_TABLE_COMBO_PPTABLE',
    16: 'SMU_TABLE_WIFIBAND',
    17: 'SMU_TABLE_COUNT',
}
SMU_TABLE_PPTABLE = 0
SMU_TABLE_WATERMARKS = 1
SMU_TABLE_CUSTOM_DPM = 2
SMU_TABLE_DPMCLOCKS = 3
SMU_TABLE_AVFS = 4
SMU_TABLE_AVFS_PSM_DEBUG = 5
SMU_TABLE_AVFS_FUSE_OVERRIDE = 6
SMU_TABLE_PMSTATUSLOG = 7
SMU_TABLE_SMU_METRICS = 8
SMU_TABLE_DRIVER_SMU_CONFIG = 9
SMU_TABLE_ACTIVITY_MONITOR_COEFF = 10
SMU_TABLE_OVERDRIVE = 11
SMU_TABLE_I2C_COMMANDS = 12
SMU_TABLE_PACE = 13
SMU_TABLE_ECCINFO = 14
SMU_TABLE_COMBO_PPTABLE = 15
SMU_TABLE_WIFIBAND = 16
SMU_TABLE_COUNT = 17
smu_table_id = ctypes.c_uint32 # enum
__all__ = \
    ['ALLOWED_FEATURE_CTRL_DEFAULT', 'ALLOWED_FEATURE_CTRL_SCPM',
    'AVFS_D_COUNT', 'AVFS_D_G', 'AVFS_D_e', 'AVFS_D_e__enumvalues',
    'AVFS_TEMP_COLD', 'AVFS_TEMP_COUNT', 'AVFS_TEMP_HOT',
    'AVFS_TEMP_e', 'AVFS_TEMP_e__enumvalues', 'AVFS_VOLTAGE_COUNT',
    'AVFS_VOLTAGE_GFX', 'AVFS_VOLTAGE_SOC', 'AVFS_VOLTAGE_TYPE_e',
    'AVFS_VOLTAGE_TYPE_e__enumvalues', 'AvfsDcBtcParams_t',
    'AvfsDebugTableExternal_t', 'AvfsDebugTable_t',
    'AvfsFuseOverride_t', 'BACO_SEQUENCE', 'BAMACO_SEQUENCE',
    'BOARD_GPIO_DC_GENLK_CLK', 'BOARD_GPIO_DC_GENLK_VSYNC',
    'BOARD_GPIO_DC_GEN_A', 'BOARD_GPIO_DC_GEN_B',
    'BOARD_GPIO_DC_GEN_C', 'BOARD_GPIO_DC_GEN_D',
    'BOARD_GPIO_DC_GEN_E', 'BOARD_GPIO_DC_GEN_F',
    'BOARD_GPIO_DC_GEN_G', 'BOARD_GPIO_DC_SWAPLOCK_A',
    'BOARD_GPIO_DC_SWAPLOCK_B', 'BOARD_GPIO_LV_EN',
    'BOARD_GPIO_SMUIO_0', 'BOARD_GPIO_SMUIO_1', 'BOARD_GPIO_SMUIO_10',
    'BOARD_GPIO_SMUIO_11', 'BOARD_GPIO_SMUIO_12',
    'BOARD_GPIO_SMUIO_13', 'BOARD_GPIO_SMUIO_14',
    'BOARD_GPIO_SMUIO_15', 'BOARD_GPIO_SMUIO_16',
    'BOARD_GPIO_SMUIO_17', 'BOARD_GPIO_SMUIO_18',
    'BOARD_GPIO_SMUIO_19', 'BOARD_GPIO_SMUIO_2',
    'BOARD_GPIO_SMUIO_20', 'BOARD_GPIO_SMUIO_21',
    'BOARD_GPIO_SMUIO_22', 'BOARD_GPIO_SMUIO_23',
    'BOARD_GPIO_SMUIO_24', 'BOARD_GPIO_SMUIO_25',
    'BOARD_GPIO_SMUIO_26', 'BOARD_GPIO_SMUIO_27',
    'BOARD_GPIO_SMUIO_28', 'BOARD_GPIO_SMUIO_29',
    'BOARD_GPIO_SMUIO_3', 'BOARD_GPIO_SMUIO_30',
    'BOARD_GPIO_SMUIO_31', 'BOARD_GPIO_SMUIO_4', 'BOARD_GPIO_SMUIO_5',
    'BOARD_GPIO_SMUIO_6', 'BOARD_GPIO_SMUIO_7', 'BOARD_GPIO_SMUIO_8',
    'BOARD_GPIO_SMUIO_9', 'BOARD_GPIO_TYPE_e',
    'BOARD_GPIO_TYPE_e__enumvalues', 'BoardTable_t', 'BootValues_t',
    'CMDCONFIG_READWRITE_BIT', 'CMDCONFIG_READWRITE_MASK',
    'CMDCONFIG_RESTART_BIT', 'CMDCONFIG_RESTART_MASK',
    'CMDCONFIG_STOP_BIT', 'CMDCONFIG_STOP_MASK',
    'CUSTOMER_VARIANT_COUNT', 'CUSTOMER_VARIANT_FALCON',
    'CUSTOMER_VARIANT_ROW', 'CUSTOMER_VARIANT_e',
    'CUSTOMER_VARIANT_e__enumvalues', 'CustomSkuTable_t',
    'D3HOTSequence_e', 'D3HOTSequence_e__enumvalues',
    'D3HOT_SEQUENCE_COUNT', 'DCS_ARCH_ASYNC', 'DCS_ARCH_DISABLED',
    'DCS_ARCH_FADCS', 'DCS_ARCH_e', 'DCS_ARCH_e__enumvalues',
    'DEBUG_OVERRIDE_DFLL_BTC_FCW_LOG',
    'DEBUG_OVERRIDE_DFLL_MASTER_MODE',
    'DEBUG_OVERRIDE_DISABLE_D0i2_REENTRY_HSR_TIMER_CHECK',
    'DEBUG_OVERRIDE_DISABLE_DFLL',
    'DEBUG_OVERRIDE_DISABLE_FAST_FCLK_TIMER',
    'DEBUG_OVERRIDE_DISABLE_FMAX_VMAX',
    'DEBUG_OVERRIDE_DISABLE_IMU_FW_CHECKS',
    'DEBUG_OVERRIDE_DISABLE_MEMORY_VOLTAGE_SCALING',
    'DEBUG_OVERRIDE_DISABLE_VCN_PG',
    'DEBUG_OVERRIDE_DISABLE_VOLT_LINK_DCN_FCLK',
    'DEBUG_OVERRIDE_DISABLE_VOLT_LINK_MP0_FCLK',
    'DEBUG_OVERRIDE_DISABLE_VOLT_LINK_VCN_DCFCLK',
    'DEBUG_OVERRIDE_ENABLE_PER_WGP_RESIENCY',
    'DEBUG_OVERRIDE_ENABLE_PROFILING_MODE',
    'DEBUG_OVERRIDE_ENABLE_RLC_VF_BRINGUP_MODE',
    'DEBUG_OVERRIDE_ENABLE_SOC_VF_BRINGUP_MODE',
    'DEBUG_OVERRIDE_NOT_USE', 'DRAM_BIT_WIDTH_COUNT',
    'DRAM_BIT_WIDTH_DISABLED', 'DRAM_BIT_WIDTH_TYPE_e',
    'DRAM_BIT_WIDTH_TYPE_e__enumvalues', 'DRAM_BIT_WIDTH_X_128',
    'DRAM_BIT_WIDTH_X_16', 'DRAM_BIT_WIDTH_X_32',
    'DRAM_BIT_WIDTH_X_64', 'DRAM_BIT_WIDTH_X_8',
    'DpmActivityMonitorCoeffIntExternal_t',
    'DpmActivityMonitorCoeffInt_t', 'DpmDescriptor_t',
    'DriverInfoTable_t', 'DriverReportedClocks_t',
    'DriverSmuConfigExternal_t', 'DriverSmuConfig_t', 'DroopInt_t',
    'ENABLE_DEBUG_FEATURES', 'EPCS_HIGH_POWER',
    'EPCS_HIGH_POWER_LIMIT', 'EPCS_LOW_POWER', 'EPCS_LOW_POWER_LIMIT',
    'EPCS_NORMAL_POWER', 'EPCS_NORMAL_POWER_LIMIT',
    'EPCS_NOT_CONFIGURED', 'EPCS_NO_BOOTUP', 'EPCS_SHORTED_LIMIT',
    'EPCS_SHORTED_POWER', 'EPCS_STATUS_COUNT', 'EPCS_STATUS_e',
    'EPCS_STATUS_e__enumvalues', 'EccInfoTable_t', 'EccInfo_t',
    'FAN_MODE_AUTO', 'FAN_MODE_MANUAL_LINEAR', 'FEATURE_ACDC_BIT',
    'FEATURE_APT_ALL_ENABLE_BIT', 'FEATURE_APT_PF_DCS_BIT',
    'FEATURE_APT_SQ_THROTTLE_BIT', 'FEATURE_ATHUB_MMHUB_PG_BIT',
    'FEATURE_ATHUB_PG_BIT', 'FEATURE_BACO_BIT', 'FEATURE_BACO_CG_BIT',
    'FEATURE_BACO_MPCLK_DS_BIT', 'FEATURE_BOOT_POWER_OPT_BIT',
    'FEATURE_BOOT_TIME_CAL_BIT', 'FEATURE_BTC_COUNT',
    'FEATURE_BTC_NOP', 'FEATURE_BTC_RESTORE', 'FEATURE_BTC_SAVE',
    'FEATURE_BTC_e', 'FEATURE_BTC_e__enumvalues', 'FEATURE_CC6_BIT',
    'FEATURE_CCLK_DPM_BIT', 'FEATURE_CLOCK_POWER_DOWN_BYPASS_BIT',
    'FEATURE_CLOCK_STRETCH_COMPENSATOR', 'FEATURE_CORE_DLDO_BIT',
    'FEATURE_CPPC_BIT', 'FEATURE_CPPC_PREFERRED_CORES',
    'FEATURE_CPUOFF_BIT', 'FEATURE_DATA_CALCULATION_BIT',
    'FEATURE_DCFCLK_DPM_BIT', 'FEATURE_DF_CSTATES_BIT',
    'FEATURE_DF_CSTATE_BIT', 'FEATURE_DF_LIGHT_CSTATE',
    'FEATURE_DPM_DCN_BIT', 'FEATURE_DPM_FCLK_BIT',
    'FEATURE_DPM_GFXCLK_BIT', 'FEATURE_DPM_GFX_POWER_OPTIMIZER_BIT',
    'FEATURE_DPM_LINK_BIT', 'FEATURE_DPM_SOCCLK_BIT',
    'FEATURE_DPM_UCLK_BIT', 'FEATURE_DS_DCFCLK_BIT',
    'FEATURE_DS_FCLK_BIT', 'FEATURE_DS_GFXCLK_BIT',
    'FEATURE_DS_HSPCLK_BIT', 'FEATURE_DS_IPUCLK_BIT',
    'FEATURE_DS_ISPCLK_BIT', 'FEATURE_DS_LCLK_BIT',
    'FEATURE_DS_MP1CLK_BIT', 'FEATURE_DS_MPIO_BIT',
    'FEATURE_DS_MPM_BIT', 'FEATURE_DS_SHUBCLK_BIT',
    'FEATURE_DS_SMNCLK_BIT', 'FEATURE_DS_SOCCLK_BIT',
    'FEATURE_DS_UCLK_BIT', 'FEATURE_DS_UMCCLK_BIT',
    'FEATURE_DS_VCN_BIT', 'FEATURE_DS_VPECLK_BIT', 'FEATURE_DVO_BIT',
    'FEATURE_EDC_BIT', 'FEATURE_EDC_PWRBRK_BIT',
    'FEATURE_FAN_ABNORMAL_BIT', 'FEATURE_FAN_CONTROLLER_BIT',
    'FEATURE_FAN_CONTROL_BIT', 'FEATURE_FAST_PSTATE_CLDO_BIT',
    'FEATURE_FCLK_DPM_BIT', 'FEATURE_FIT_BIT', 'FEATURE_FW_CTF_BIT',
    'FEATURE_FW_DATA_READ_BIT', 'FEATURE_FW_DSTATE_BIT',
    'FEATURE_GFXCLK_SPREAD_SPECTRUM_BIT', 'FEATURE_GFXOFF_BIT',
    'FEATURE_GFX_DCS_BIT', 'FEATURE_GFX_DEM_BIT',
    'FEATURE_GFX_DIDT_XVMIN_BIT', 'FEATURE_GFX_DPM_BIT',
    'FEATURE_GFX_EDC_BIT', 'FEATURE_GFX_EDC_XVMIN_BIT',
    'FEATURE_GFX_IMU_BIT', 'FEATURE_GFX_PCC_DFLL_BIT',
    'FEATURE_GFX_PSM_DIDT_BIT', 'FEATURE_GFX_READ_MARGIN_BIT',
    'FEATURE_GFX_ULV_BIT', 'FEATURE_GTHR_BIT',
    'FEATURE_IOMMUL2_PG_BIT', 'FEATURE_IPU_DPM_BIT',
    'FEATURE_ISP_DPM_BIT', 'FEATURE_LCLK_DPM_BIT',
    'FEATURE_LED_DISPLAY_BIT', 'FEATURE_LOW_POWER_DCNCLKS_BIT',
    'FEATURE_MEM_TEMP_READ_BIT', 'FEATURE_MM_DPM_BIT',
    'FEATURE_OPTIMIZED_VMIN_BIT', 'FEATURE_OUT_OF_BAND_MONITOR_BIT',
    'FEATURE_P3T_BIT', 'FEATURE_PCC_BIT', 'FEATURE_PERF_LIMIT_BIT',
    'FEATURE_PLL_POWER_DOWN_BIT', 'FEATURE_PPT_BIT',
    'FEATURE_PROCHOT_BIT', 'FEATURE_PSI_BIT', 'FEATURE_PWR_ALL',
    'FEATURE_PWR_BACO', 'FEATURE_PWR_DOMAIN_COUNT',
    'FEATURE_PWR_DOMAIN_e', 'FEATURE_PWR_DOMAIN_e__enumvalues',
    'FEATURE_PWR_GFX', 'FEATURE_PWR_S5', 'FEATURE_PWR_SOC',
    'FEATURE_RESERVED0_BIT', 'FEATURE_RESERVED1_BIT',
    'FEATURE_S0I3_BIT', 'FEATURE_SHUBCLK_DPM_BIT',
    'FEATURE_SMARTSHIFT_BIT', 'FEATURE_SMART_L3_RINSER_BIT',
    'FEATURE_SMU_LOW_POWER_BIT', 'FEATURE_SOCCLK_DPM_BIT',
    'FEATURE_SOC_CG_BIT', 'FEATURE_SOC_EDC_XVMIN_BIT',
    'FEATURE_SOC_MPCLK_DS_BIT', 'FEATURE_SOC_PCC_BIT',
    'FEATURE_SPARE_59_BIT', 'FEATURE_SPARE_60_BIT',
    'FEATURE_SPARE_61_BIT', 'FEATURE_SPARE_62_BIT',
    'FEATURE_SPARE_63_BIT', 'FEATURE_STAPM_BIT', 'FEATURE_TDC_BIT',
    'FEATURE_THERMAL_BIT', 'FEATURE_THROTTLERS_BIT',
    'FEATURE_VCN_DPM_BIT', 'FEATURE_VDDIO_MEM_SCALING_BIT',
    'FEATURE_VDDOFF_BIT', 'FEATURE_VDDOFF_ECO_BIT',
    'FEATURE_VMEMP_SCALING_BIT', 'FEATURE_VPE_DPM_BIT',
    'FEATURE_VR0HOT_BIT', 'FEATURE_WHISPER_MODE_BIT',
    'FEATURE_ZSTATES_BIT', 'FEATURE_ZSTATES_ECO_BIT',
    'FOPT_CALC_AC_CALC_DC', 'FOPT_CALC_AC_PPTABLE_DC', 'FOPT_CALC_e',
    'FOPT_CALC_e__enumvalues', 'FOPT_PPTABLE_AC_CALC_DC',
    'FOPT_PPTABLE_AC_PPTABLE_DC', 'FW_DSTATE_CLDO_PRG_BIT',
    'FW_DSTATE_D0i3_2_QUIET_FW_BIT', 'FW_DSTATE_DF_PLL_PWRDN_BIT',
    'FW_DSTATE_G6_HSR_BIT', 'FW_DSTATE_G6_PHY_VMEMP_OFF_BIT',
    'FW_DSTATE_HSR_NON_STROBE_BIT', 'FW_DSTATE_MALL_ALLOC_BIT',
    'FW_DSTATE_MALL_FLUSH_BIT', 'FW_DSTATE_MEM_PLL_PWRDN_BIT',
    'FW_DSTATE_MEM_PSI_BIT', 'FW_DSTATE_MMHUB_INTERLOCK_BIT',
    'FW_DSTATE_MP0_ENTER_WFI_BIT', 'FW_DSTATE_MP1_WHISPER_MODE_BIT',
    'FW_DSTATE_SMN_DS_BIT', 'FW_DSTATE_SOC_LIV_MIN_BIT',
    'FW_DSTATE_SOC_PLL_PWRDN_BIT', 'FW_DSTATE_SOC_PSI_BIT',
    'FW_DSTATE_SOC_ULV_BIT', 'FanMode_e', 'FanMode_e__enumvalues',
    'FwStatus_t', 'FwStatus_t_v14_0_1',
    'GPIO_INT_POLARITY_ACTIVE_HIGH', 'GPIO_INT_POLARITY_ACTIVE_LOW',
    'GpioIntPolarity_e', 'GpioIntPolarity_e__enumvalues',
    'I2C_CMD_COUNT', 'I2C_CMD_READ', 'I2C_CMD_WRITE',
    'I2C_CONTROLLER_DISABLED', 'I2C_CONTROLLER_ENABLED',
    'I2C_CONTROLLER_NAME_COUNT', 'I2C_CONTROLLER_NAME_FAN_INTAKE',
    'I2C_CONTROLLER_NAME_LIQUID0', 'I2C_CONTROLLER_NAME_LIQUID1',
    'I2C_CONTROLLER_NAME_PLX', 'I2C_CONTROLLER_NAME_VR_GFX',
    'I2C_CONTROLLER_NAME_VR_SOC', 'I2C_CONTROLLER_NAME_VR_VDDIO',
    'I2C_CONTROLLER_NAME_VR_VMEMP', 'I2C_CONTROLLER_PORT_0',
    'I2C_CONTROLLER_PORT_1', 'I2C_CONTROLLER_PORT_COUNT',
    'I2C_CONTROLLER_PROTOCOL_COUNT',
    'I2C_CONTROLLER_PROTOCOL_INA3221',
    'I2C_CONTROLLER_PROTOCOL_TMP_MAX31875',
    'I2C_CONTROLLER_PROTOCOL_TMP_MAX6604',
    'I2C_CONTROLLER_PROTOCOL_VR_IR35217',
    'I2C_CONTROLLER_PROTOCOL_VR_XPDE132G5',
    'I2C_CONTROLLER_THROTTLER_COUNT',
    'I2C_CONTROLLER_THROTTLER_FAN_INTAKE',
    'I2C_CONTROLLER_THROTTLER_INA3221',
    'I2C_CONTROLLER_THROTTLER_LIQUID0',
    'I2C_CONTROLLER_THROTTLER_LIQUID1',
    'I2C_CONTROLLER_THROTTLER_PLX',
    'I2C_CONTROLLER_THROTTLER_TYPE_NONE',
    'I2C_CONTROLLER_THROTTLER_VR_GFX',
    'I2C_CONTROLLER_THROTTLER_VR_SOC',
    'I2C_CONTROLLER_THROTTLER_VR_VDDIO',
    'I2C_CONTROLLER_THROTTLER_VR_VMEMP', 'I2C_PORT_GPIO',
    'I2C_PORT_SVD_SCL', 'I2C_SPEED_COUNT', 'I2C_SPEED_FAST_100K',
    'I2C_SPEED_FAST_400K', 'I2C_SPEED_FAST_50K',
    'I2C_SPEED_FAST_PLUS_1M', 'I2C_SPEED_HIGH_1M',
    'I2C_SPEED_HIGH_2M', 'I2cCmdType_e', 'I2cCmdType_e__enumvalues',
    'I2cControllerConfig_t', 'I2cControllerName_e',
    'I2cControllerName_e__enumvalues', 'I2cControllerPort_e',
    'I2cControllerPort_e__enumvalues', 'I2cControllerProtocol_e',
    'I2cControllerProtocol_e__enumvalues', 'I2cControllerThrottler_e',
    'I2cControllerThrottler_e__enumvalues', 'I2cPort_e',
    'I2cPort_e__enumvalues', 'I2cSpeed_e', 'I2cSpeed_e__enumvalues',
    'IH_INTERRUPT_CONTEXT_ID_AC', 'IH_INTERRUPT_CONTEXT_ID_AUDIO_D0',
    'IH_INTERRUPT_CONTEXT_ID_AUDIO_D3',
    'IH_INTERRUPT_CONTEXT_ID_BACO', 'IH_INTERRUPT_CONTEXT_ID_DC',
    'IH_INTERRUPT_CONTEXT_ID_DYNAMIC_TABLE',
    'IH_INTERRUPT_CONTEXT_ID_FAN_ABNORMAL',
    'IH_INTERRUPT_CONTEXT_ID_FAN_RECOVERY',
    'IH_INTERRUPT_CONTEXT_ID_THERMAL_THROTTLING',
    'IH_INTERRUPT_ID_TO_DRIVER', 'INVALID_BOARD_GPIO',
    'LED_DISPLAY_ERROR_BIT', 'LED_DISPLAY_GFX_DPM_BIT',
    'LED_DISPLAY_PCIE_BIT', 'LinearInt_t', 'MAX_BOARD_DC_GPIO_NUM',
    'MAX_BOARD_GPIO_SMUIO_NUM', 'MAX_SW_I2C_COMMANDS',
    'MEM_TEMP_READ_IN_BAND_DUMMY_PSTATE_BIT',
    'MEM_TEMP_READ_IN_BAND_REFRESH_BIT',
    'MEM_TEMP_READ_OUT_OF_BAND_BIT', 'MEM_VENDOR_COUNT',
    'MEM_VENDOR_ELPIDA', 'MEM_VENDOR_ESMT', 'MEM_VENDOR_ETRON',
    'MEM_VENDOR_HYNIX', 'MEM_VENDOR_INFINEON', 'MEM_VENDOR_MICRON',
    'MEM_VENDOR_MOSEL', 'MEM_VENDOR_NANYA', 'MEM_VENDOR_PLACEHOLDER0',
    'MEM_VENDOR_PLACEHOLDER1', 'MEM_VENDOR_PLACEHOLDER2',
    'MEM_VENDOR_PLACEHOLDER3', 'MEM_VENDOR_PLACEHOLDER4',
    'MEM_VENDOR_PLACEHOLDER5', 'MEM_VENDOR_SAMSUNG',
    'MEM_VENDOR_WINBOND', 'MEM_VENDOR_e', 'MEM_VENDOR_e__enumvalues',
    'MSR_SEQUENCE', 'MsgLimits_t', 'NUM_DCFCLK_DPM_LEVELS',
    'NUM_DCLK_DPM_LEVELS', 'NUM_DISPCLK_DPM_LEVELS',
    'NUM_DPPCLK_DPM_LEVELS', 'NUM_DPREFCLK_DPM_LEVELS',
    'NUM_DTBCLK_DPM_LEVELS', 'NUM_FCLK_DPM_LEVELS', 'NUM_FEATURES',
    'NUM_GFXCLK_DPM_LEVELS', 'NUM_I2C_CONTROLLERS', 'NUM_LINK_LEVELS',
    'NUM_MP0CLK_DPM_LEVELS', 'NUM_OD_FAN_MAX_POINTS',
    'NUM_SOCCLK_DPM_LEVELS', 'NUM_UCLK_DPM_LEVELS',
    'NUM_VCLK_DPM_LEVELS', 'NUM_WM_RANGES', 'OD_FAIL_e',
    'OD_FAIL_e__enumvalues', 'OD_FAN_ACOUSTIC_LIMIT_ERROR',
    'OD_FAN_ACOUSTIC_TARGET_ERROR', 'OD_FAN_CURVE_PWM_ERROR',
    'OD_FAN_CURVE_TEMP_ERROR', 'OD_FAN_MIN_PWM_ERROR',
    'OD_FAN_TARGET_TEMP_ERROR', 'OD_FAN_ZERO_RPM_STOP_TEMP_ERROR',
    'OD_FCLK_ERROR', 'OD_FULL_CTRL_FCLK_ERROR',
    'OD_FULL_CTRL_GFXCLK_ERROR', 'OD_FULL_CTRL_UCLK_ERROR',
    'OD_FULL_CTRL_VDD_GFX_ERROR', 'OD_FULL_CTRL_VDD_SOC_ERROR',
    'OD_GFXCLK_ERROR', 'OD_GFXCLK_VF_CURVE_OFFSET_ERROR',
    'OD_INVALID_FEATURE_COMBO_ERROR', 'OD_NO_ERROR',
    'OD_OP_GFX_EDC_ERROR', 'OD_OP_GFX_PCC_ERROR', 'OD_OP_TEMP_ERROR',
    'OD_POWER_FEATURE_CTRL_ERROR', 'OD_PPT_ERROR',
    'OD_REQUEST_ADVANCED_NOT_SUPPORTED', 'OD_TDC_ERROR',
    'OD_UCLK_ERROR', 'OD_UNSUPPORTED_FEATURE',
    'OD_VDD_GFX_VMAX_ERROR', 'OD_VDD_SOC_VMAX_ERROR',
    'OverDriveLimits_t', 'OverDriveTableExternal_t',
    'OverDriveTable_t', 'PERF_LEVEL_ACTIVITY',
    'PERF_LEVEL_POWER_CONTAINMENT', 'PFE_Settings_t',
    'PG_DYNAMIC_MODE', 'PG_POWER_DOWN', 'PG_POWER_UP',
    'PG_STATIC_MODE', 'PMFW_VOLT_PLANE_COUNT', 'PMFW_VOLT_PLANE_GFX',
    'PMFW_VOLT_PLANE_SOC', 'PMFW_VOLT_PLANE_e',
    'PMFW_VOLT_PLANE_e__enumvalues', 'POWER_SOURCE_AC',
    'POWER_SOURCE_COUNT', 'POWER_SOURCE_DC', 'POWER_SOURCE_e',
    'POWER_SOURCE_e__enumvalues', 'PPCLK_COUNT', 'PPCLK_DCFCLK',
    'PPCLK_DCLK_0', 'PPCLK_DISPCLK', 'PPCLK_DPPCLK', 'PPCLK_DPREFCLK',
    'PPCLK_DTBCLK', 'PPCLK_FCLK', 'PPCLK_GFXCLK', 'PPCLK_SOCCLK',
    'PPCLK_UCLK', 'PPCLK_VCLK_0', 'PPCLK_e', 'PPCLK_e__enumvalues',
    'PPSMC_MSG_AllowGfxDcs', 'PPSMC_MSG_AllowGfxOff',
    'PPSMC_MSG_AllowIHHostInterrupt', 'PPSMC_MSG_ArmD3',
    'PPSMC_MSG_BacoAudioD3PME', 'PPSMC_MSG_DisableAllSmuFeatures',
    'PPSMC_MSG_DisableSmuFeaturesHigh',
    'PPSMC_MSG_DisableSmuFeaturesLow', 'PPSMC_MSG_DisallowGfxDcs',
    'PPSMC_MSG_DisallowGfxOff', 'PPSMC_MSG_DramLogSetDramAddrHigh',
    'PPSMC_MSG_DramLogSetDramAddrLow', 'PPSMC_MSG_DramLogSetDramSize',
    'PPSMC_MSG_DummyUndefined', 'PPSMC_MSG_DumpSTBtoDram',
    'PPSMC_MSG_EnableAllSmuFeatures',
    'PPSMC_MSG_EnableAudioStutterWA', 'PPSMC_MSG_EnableShadowDpm',
    'PPSMC_MSG_EnableSmuFeaturesHigh',
    'PPSMC_MSG_EnableSmuFeaturesLow', 'PPSMC_MSG_EnterBaco',
    'PPSMC_MSG_ExitBaco', 'PPSMC_MSG_ExtPwrConnSupport',
    'PPSMC_MSG_GetAllRunningSmuFeatures',
    'PPSMC_MSG_GetDcModeMaxDpmFreq', 'PPSMC_MSG_GetDpmFreqByIndex',
    'PPSMC_MSG_GetDriverIfVersion', 'PPSMC_MSG_GetMaxDpmFreq',
    'PPSMC_MSG_GetMinDpmFreq', 'PPSMC_MSG_GetPptLimit',
    'PPSMC_MSG_GetRunningSmuFeaturesHigh',
    'PPSMC_MSG_GetRunningSmuFeaturesLow', 'PPSMC_MSG_GetSmuVersion',
    'PPSMC_MSG_GetSvi3Voltage', 'PPSMC_MSG_GetVoltageByDpm',
    'PPSMC_MSG_Mode3Reset', 'PPSMC_MSG_NotifyPowerSource',
    'PPSMC_MSG_OverridePcieParameters', 'PPSMC_MSG_PowerDownJpeg',
    'PPSMC_MSG_PowerDownUmsch', 'PPSMC_MSG_PowerDownVcn',
    'PPSMC_MSG_PowerUpJpeg', 'PPSMC_MSG_PowerUpUmsch',
    'PPSMC_MSG_PowerUpVcn',
    'PPSMC_MSG_PreloadSwPstateForUclkOverDrive',
    'PPSMC_MSG_PrepareMp1ForUnload',
    'PPSMC_MSG_ReenableAcDcInterrupt', 'PPSMC_MSG_RunDcBtc',
    'PPSMC_MSG_STBtoDramLogSetDramAddress',
    'PPSMC_MSG_STBtoDramLogSetDramSize',
    'PPSMC_MSG_SetAllowedFeaturesMaskHigh',
    'PPSMC_MSG_SetAllowedFeaturesMaskLow',
    'PPSMC_MSG_SetBadMemoryPagesRetiredFlagsPerChannel',
    'PPSMC_MSG_SetDcsArch', 'PPSMC_MSG_SetDriverDramAddr',
    'PPSMC_MSG_SetDriverDramAddrHigh',
    'PPSMC_MSG_SetDriverDramAddrLow',
    'PPSMC_MSG_SetExternalClientDfCstateAllow',
    'PPSMC_MSG_SetFwDstatesMask', 'PPSMC_MSG_SetHardMaxByFreq',
    'PPSMC_MSG_SetHardMinByFreq', 'PPSMC_MSG_SetMGpuFanBoostLimitRpm',
    'PPSMC_MSG_SetNumBadMemoryPagesRetired',
    'PPSMC_MSG_SetOBMTraceBufferLogging', 'PPSMC_MSG_SetPptLimit',
    'PPSMC_MSG_SetPriorityDeltaGain', 'PPSMC_MSG_SetSoftMaxByFreq',
    'PPSMC_MSG_SetSoftMinByFreq',
    'PPSMC_MSG_SetSystemVirtualDramAddrHigh',
    'PPSMC_MSG_SetSystemVirtualDramAddrLow',
    'PPSMC_MSG_SetTemperatureInputSelect',
    'PPSMC_MSG_SetThrottlerMask', 'PPSMC_MSG_SetToolsDramAddr',
    'PPSMC_MSG_SetToolsDramAddrHigh', 'PPSMC_MSG_SetToolsDramAddrLow',
    'PPSMC_MSG_SetVideoFps', 'PPSMC_MSG_SetWorkloadMask',
    'PPSMC_MSG_TestMessage', 'PPSMC_MSG_TransferTableDram2Smu',
    'PPSMC_MSG_TransferTableDram2SmuWithAddr',
    'PPSMC_MSG_TransferTableSmu2Dram',
    'PPSMC_MSG_TransferTableSmu2DramWithAddr',
    'PPSMC_MSG_TriggerVFFLR', 'PPSMC_MSG_UpdatePolicy',
    'PPSMC_MSG_UseDefaultPPTable', 'PPSMC_MSG_UseProfilingMode',
    'PPSMC_Message_Count', 'PPSMC_Result_CmdRejectedBusy',
    'PPSMC_Result_CmdRejectedPrereq', 'PPSMC_Result_Failed',
    'PPSMC_Result_OK', 'PPSMC_Result_UnknownCmd', 'PPSMC_VERSION',
    'PPTABLE_VERSION', 'PPT_THROTTLER_COUNT', 'PPT_THROTTLER_PPT0',
    'PPT_THROTTLER_PPT1', 'PPT_THROTTLER_PPT2', 'PPT_THROTTLER_PPT3',
    'PPT_THROTTLER_e', 'PPT_THROTTLER_e__enumvalues', 'PPTable_t',
    'PP_GRTAVFS_FW_COMMON_FUSE_COUNT', 'PP_GRTAVFS_FW_COMMON_FUSE_e',
    'PP_GRTAVFS_FW_COMMON_FUSE_e__enumvalues',
    'PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_COLD_T0',
    'PP_GRTAVFS_FW_COMMON_PPVMIN_Z1_HOT_T0',
    'PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_COLD_T0',
    'PP_GRTAVFS_FW_COMMON_PPVMIN_Z2_HOT_T0',
    'PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_COLD_T0',
    'PP_GRTAVFS_FW_COMMON_PPVMIN_Z3_HOT_T0',
    'PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_COLD_T0',
    'PP_GRTAVFS_FW_COMMON_PPVMIN_Z4_HOT_T0',
    'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z0',
    'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z1',
    'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z2',
    'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z3',
    'PP_GRTAVFS_FW_COMMON_SRAM_RM_Z4', 'PP_GRTAVFS_FW_SEP_FUSE_COUNT',
    'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_0',
    'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_1',
    'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_2',
    'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_3',
    'PP_GRTAVFS_FW_SEP_FUSE_FREQUENCY_TO_COUNT_SCALER_4',
    'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_0',
    'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_1',
    'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_2',
    'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_3',
    'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_4',
    'PP_GRTAVFS_FW_SEP_FUSE_GB1_PWL_VOLTAGE_NEG_1',
    'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_0',
    'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_1',
    'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_2',
    'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_3',
    'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_4',
    'PP_GRTAVFS_FW_SEP_FUSE_GB2_PWL_VOLTAGE_NEG_1',
    'PP_GRTAVFS_FW_SEP_FUSE_VF4_FREQUENCY',
    'PP_GRTAVFS_FW_SEP_FUSE_VF_NEG_1_FREQUENCY',
    'PP_GRTAVFS_FW_SEP_FUSE_e',
    'PP_GRTAVFS_FW_SEP_FUSE_e__enumvalues',
    'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE0',
    'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE1',
    'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE2',
    'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE3',
    'PP_GRTAVFS_HW_CPOSCALINGCTRL_ZONE4',
    'PP_GRTAVFS_HW_CPO_CTL_ZONE0', 'PP_GRTAVFS_HW_CPO_CTL_ZONE1',
    'PP_GRTAVFS_HW_CPO_CTL_ZONE2', 'PP_GRTAVFS_HW_CPO_CTL_ZONE3',
    'PP_GRTAVFS_HW_CPO_CTL_ZONE4', 'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE0',
    'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE1',
    'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE2',
    'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE3',
    'PP_GRTAVFS_HW_CPO_EN_0_31_ZONE4',
    'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE0',
    'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE1',
    'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE2',
    'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE3',
    'PP_GRTAVFS_HW_CPO_EN_32_63_ZONE4', 'PP_GRTAVFS_HW_FUSE_COUNT',
    'PP_GRTAVFS_HW_FUSE_e', 'PP_GRTAVFS_HW_FUSE_e__enumvalues',
    'PP_GRTAVFS_HW_RESERVED_0', 'PP_GRTAVFS_HW_RESERVED_1',
    'PP_GRTAVFS_HW_RESERVED_2', 'PP_GRTAVFS_HW_RESERVED_3',
    'PP_GRTAVFS_HW_RESERVED_4', 'PP_GRTAVFS_HW_RESERVED_5',
    'PP_GRTAVFS_HW_RESERVED_6', 'PP_GRTAVFS_HW_VOLTAGE_GB',
    'PP_GRTAVFS_HW_ZONE0_VF', 'PP_GRTAVFS_HW_ZONE1_VF1',
    'PP_GRTAVFS_HW_ZONE2_VF2', 'PP_GRTAVFS_HW_ZONE3_VF3',
    'PP_NUM_OD_VF_CURVE_POINTS', 'PP_NUM_PSM_DIDT_PWL_ZONES',
    'PP_NUM_RTAVFS_PWL_ZONES', 'PP_OD_FEATURE_COUNT',
    'PP_OD_FEATURE_EDC_BIT', 'PP_OD_FEATURE_FAN_CURVE_BIT',
    'PP_OD_FEATURE_FAN_LEGACY_BIT', 'PP_OD_FEATURE_FCLK_BIT',
    'PP_OD_FEATURE_FULL_CTRL_BIT', 'PP_OD_FEATURE_GFXCLK_BIT',
    'PP_OD_FEATURE_GFX_VF_CURVE_BIT', 'PP_OD_FEATURE_GFX_VMAX_BIT',
    'PP_OD_FEATURE_PPT_BIT', 'PP_OD_FEATURE_SOC_VMAX_BIT',
    'PP_OD_FEATURE_TDC_BIT', 'PP_OD_FEATURE_TEMPERATURE_BIT',
    'PP_OD_FEATURE_UCLK_BIT', 'PP_OD_FEATURE_ZERO_FAN_BIT',
    'PP_OD_POWER_FEATURE_ALWAYS_DISABLED',
    'PP_OD_POWER_FEATURE_ALWAYS_ENABLED',
    'PP_OD_POWER_FEATURE_DISABLED_WHILE_GAMING',
    'PP_OD_POWER_FEATURE_e', 'PP_OD_POWER_FEATURE_e__enumvalues',
    'PSI_SEL_VR0_PLANE0_PSI0', 'PSI_SEL_VR0_PLANE0_PSI1',
    'PSI_SEL_VR0_PLANE1_PSI0', 'PSI_SEL_VR0_PLANE1_PSI1',
    'PSI_SEL_VR1_PLANE0_PSI0', 'PSI_SEL_VR1_PLANE0_PSI1',
    'PSI_SEL_VR1_PLANE1_PSI0', 'PSI_SEL_VR1_PLANE1_PSI1',
    'PWR_CONFIG_TBP_DESKTOP', 'PWR_CONFIG_TBP_MOBILE',
    'PWR_CONFIG_TCP_ESTIMATED', 'PWR_CONFIG_TCP_MEASURED',
    'PWR_CONFIG_TDP', 'PWR_CONFIG_TGP', 'PowerGatingMode_e',
    'PowerGatingMode_e__enumvalues', 'PowerGatingSettings_e',
    'PowerGatingSettings_e__enumvalues', 'PwrConfig_e',
    'PwrConfig_e__enumvalues', 'QuadraticInt_t',
    'SMARTSHIFT_VERSION_1', 'SMARTSHIFT_VERSION_2',
    'SMARTSHIFT_VERSION_3', 'SMARTSHIFT_VERSION_e',
    'SMARTSHIFT_VERSION_e__enumvalues', 'SMU14_DRIVER_IF_V14_0_H',
    'SMU14_Firmware_Footer', 'SMU_CLK_COUNT',
    'SMU_CUSTOM_FAN_SPEED_PWM', 'SMU_CUSTOM_FAN_SPEED_RPM',
    'SMU_DCEFCLK', 'SMU_DCLK', 'SMU_DCLK1', 'SMU_DEFAULT_PPT_LIMIT',
    'SMU_DISPCLK', 'SMU_DPM_USER_PROFILE_RESTORE', 'SMU_ECLK',
    'SMU_FAST_PPT_LIMIT', 'SMU_FCLK', 'SMU_FW_NAME_LEN',
    'SMU_Firmware_Header', 'SMU_GFXCLK', 'SMU_LCLK', 'SMU_MCLK',
    'SMU_MEMORY_POOL_SIZE_1_GB', 'SMU_MEMORY_POOL_SIZE_256_MB',
    'SMU_MEMORY_POOL_SIZE_2_GB', 'SMU_MEMORY_POOL_SIZE_512_MB',
    'SMU_MEMORY_POOL_SIZE_ZERO', 'SMU_OD_ACOUSTIC_LIMIT',
    'SMU_OD_ACOUSTIC_TARGET', 'SMU_OD_CCLK', 'SMU_OD_FAN_CURVE',
    'SMU_OD_FAN_MINIMUM_PWM', 'SMU_OD_FAN_TARGET_TEMPERATURE',
    'SMU_OD_MCLK', 'SMU_OD_RANGE', 'SMU_OD_SCLK', 'SMU_OD_VDDC_CURVE',
    'SMU_OD_VDDGFX_OFFSET', 'SMU_PCIE', 'SMU_PHYCLK', 'SMU_PIXCLK',
    'SMU_POWER_SOURCE_AC', 'SMU_POWER_SOURCE_COUNT',
    'SMU_POWER_SOURCE_DC', 'SMU_PPT_LIMIT_CURRENT',
    'SMU_PPT_LIMIT_DEFAULT', 'SMU_PPT_LIMIT_MAX', 'SMU_PPT_LIMIT_MIN',
    'SMU_REFRESHRATE_SOURCE_EDID', 'SMU_REFRESHRATE_SOURCE_EXPLICIT',
    'SMU_SCLK', 'SMU_SOCCLK',
    'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE',
    'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2',
    'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE',
    'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW',
    'SMU_STATE_CLASSIFICATION_FLAG_ACPI',
    'SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE',
    'SMU_STATE_CLASSIFICATION_FLAG_BACO',
    'SMU_STATE_CLASSIFICATION_FLAG_BOOT',
    'SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE',
    'SMU_STATE_CLASSIFICATION_FLAG_FORCED',
    'SMU_STATE_CLASSIFICATION_FLAG_HD2',
    'SMU_STATE_CLASSIFICATION_FLAG_RESET',
    'SMU_STATE_CLASSIFICATION_FLAG_THERMAL',
    'SMU_STATE_CLASSIFICATION_FLAG_ULV',
    'SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE',
    'SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE',
    'SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE',
    'SMU_STATE_CLASSIFICATION_FLAG_UVD',
    'SMU_STATE_CLASSIFICATION_FLAG_UVD_HD',
    'SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC',
    'SMU_STATE_CLASSIFICATION_FLAG_UVD_SD', 'SMU_STATE_UI_LABEL_BACO',
    'SMU_STATE_UI_LABEL_BALLANCED', 'SMU_STATE_UI_LABEL_BATTERY',
    'SMU_STATE_UI_LABEL_MIDDLE_HIGHT', 'SMU_STATE_UI_LABEL_NONE',
    'SMU_STATE_UI_LABEL_PERFORMANCE', 'SMU_STATE_UI_TABEL_MIDDLE_LOW',
    'SMU_TABLE_ACTIVITY_MONITOR_COEFF', 'SMU_TABLE_AVFS',
    'SMU_TABLE_AVFS_FUSE_OVERRIDE', 'SMU_TABLE_AVFS_PSM_DEBUG',
    'SMU_TABLE_COMBO_PPTABLE', 'SMU_TABLE_COUNT',
    'SMU_TABLE_CUSTOM_DPM', 'SMU_TABLE_DPMCLOCKS',
    'SMU_TABLE_DRIVER_SMU_CONFIG', 'SMU_TABLE_ECCINFO',
    'SMU_TABLE_I2C_COMMANDS', 'SMU_TABLE_OVERDRIVE', 'SMU_TABLE_PACE',
    'SMU_TABLE_PMSTATUSLOG', 'SMU_TABLE_PPTABLE',
    'SMU_TABLE_SMU_METRICS', 'SMU_TABLE_WATERMARKS',
    'SMU_TABLE_WIFIBAND', 'SMU_TEMPERATURE_UNITS_PER_CENTIGRADES',
    'SMU_THERMAL_MAXIMUM_ALERT_TEMP',
    'SMU_THERMAL_MINIMUM_ALERT_TEMP', 'SMU_THROTTLER_APCC_BIT',
    'SMU_THROTTLER_EDC_CPU_BIT', 'SMU_THROTTLER_EDC_GFX_BIT',
    'SMU_THROTTLER_FIT_BIT', 'SMU_THROTTLER_FPPT_BIT',
    'SMU_THROTTLER_PPM_BIT', 'SMU_THROTTLER_PPT0_BIT',
    'SMU_THROTTLER_PPT1_BIT', 'SMU_THROTTLER_PPT2_BIT',
    'SMU_THROTTLER_PPT3_BIT', 'SMU_THROTTLER_PROCHOT_CPU_BIT',
    'SMU_THROTTLER_PROCHOT_GFX_BIT', 'SMU_THROTTLER_SPL_BIT',
    'SMU_THROTTLER_SPPT_APU_BIT', 'SMU_THROTTLER_SPPT_BIT',
    'SMU_THROTTLER_TDC_CVIP_BIT', 'SMU_THROTTLER_TDC_GFX_BIT',
    'SMU_THROTTLER_TDC_MEM_BIT', 'SMU_THROTTLER_TDC_SOC_BIT',
    'SMU_THROTTLER_TDC_VDD_BIT', 'SMU_THROTTLER_TEMP_CORE_BIT',
    'SMU_THROTTLER_TEMP_EDGE_BIT', 'SMU_THROTTLER_TEMP_GPU_BIT',
    'SMU_THROTTLER_TEMP_HOTSPOT_BIT',
    'SMU_THROTTLER_TEMP_LIQUID0_BIT',
    'SMU_THROTTLER_TEMP_LIQUID1_BIT', 'SMU_THROTTLER_TEMP_MEM_BIT',
    'SMU_THROTTLER_TEMP_SOC_BIT', 'SMU_THROTTLER_TEMP_VR_GFX_BIT',
    'SMU_THROTTLER_TEMP_VR_MEM0_BIT',
    'SMU_THROTTLER_TEMP_VR_MEM1_BIT', 'SMU_THROTTLER_TEMP_VR_SOC_BIT',
    'SMU_THROTTLER_VRHOT0_BIT', 'SMU_THROTTLER_VRHOT1_BIT',
    'SMU_UCLK', 'SMU_V14_0_2_PPSMC_H', 'SMU_VCLK', 'SMU_VCLK1',
    'SVI_PLANE_COUNT', 'SVI_PLANE_VDDCI_MEM', 'SVI_PLANE_VDDIO_MEM',
    'SVI_PLANE_VDD_GFX', 'SVI_PLANE_VDD_SOC', 'SVI_PLANE_e',
    'SVI_PLANE_e__enumvalues', 'SVI_PSI_0', 'SVI_PSI_1', 'SVI_PSI_2',
    'SVI_PSI_3', 'SVI_PSI_4', 'SVI_PSI_5', 'SVI_PSI_6', 'SVI_PSI_7',
    'SVI_PSI_e', 'SVI_PSI_e__enumvalues', 'SkuTable_t',
    'SmuMetricsExternal_t', 'SmuMetrics_t', 'Svi3RegulatorSettings_t',
    'SviTelemetryScale_t', 'SwI2cCmd_t', 'SwI2cRequestExternal_t',
    'SwI2cRequest_t', 'TABLE_ACOUSTIC_LIMIT_RPM_FAILED',
    'TABLE_ACOUSTIC_TARGET_RPM_FAILED',
    'TABLE_ACTIVITY_MONITOR_COEFF', 'TABLE_AVFS_PSM_DEBUG',
    'TABLE_COMBO_PPTABLE', 'TABLE_COUNT', 'TABLE_CUSTOM_SKUTABLE',
    'TABLE_DRIVER_INFO', 'TABLE_DRIVER_SMU_CONFIG', 'TABLE_ECCINFO',
    'TABLE_FAN_PWM_MIN_FAILED', 'TABLE_FAN_START_TEMP_FAILED',
    'TABLE_FAN_STOP_TEMP_FAILED', 'TABLE_FAN_TARGET_TEMP_FAILED',
    'TABLE_I2C_COMMANDS', 'TABLE_MGPU_ACOUSTIC_TARGET_RPM_FAILED',
    'TABLE_OVERDRIVE', 'TABLE_PMSTATUSLOG', 'TABLE_PPTABLE',
    'TABLE_PPT_FAILED', 'TABLE_SMU_METRICS', 'TABLE_TDC_FAILED',
    'TABLE_TEMP_FAILED', 'TABLE_TRANSFER_FAILED', 'TABLE_TRANSFER_OK',
    'TABLE_TRANSFER_PENDING', 'TABLE_WATERMARKS',
    'TDC_THROTTLER_COUNT', 'TDC_THROTTLER_GFX', 'TDC_THROTTLER_SOC',
    'TDC_THROTTLER_e', 'TDC_THROTTLER_e__enumvalues', 'TEMP_COUNT',
    'TEMP_EDGE', 'TEMP_HOTSPOT', 'TEMP_HOTSPOT_GFX',
    'TEMP_HOTSPOT_SOC', 'TEMP_LIQUID0', 'TEMP_LIQUID1', 'TEMP_MEM',
    'TEMP_PLX', 'TEMP_VR_GFX', 'TEMP_VR_MEM0', 'TEMP_VR_MEM1',
    'TEMP_VR_SOC', 'TEMP_e', 'TEMP_e__enumvalues', 'THROTTLER_COUNT',
    'THROTTLER_FIT_BIT', 'THROTTLER_GFX_APCC_PLUS_BIT',
    'THROTTLER_GFX_DVO_BIT', 'THROTTLER_PPT0_BIT',
    'THROTTLER_PPT1_BIT', 'THROTTLER_PPT2_BIT', 'THROTTLER_PPT3_BIT',
    'THROTTLER_TDC_GFX_BIT', 'THROTTLER_TDC_SOC_BIT',
    'THROTTLER_TEMP_EDGE_BIT', 'THROTTLER_TEMP_HOTSPOT_BIT',
    'THROTTLER_TEMP_HOTSPOT_GFX_BIT',
    'THROTTLER_TEMP_HOTSPOT_SOC_BIT', 'THROTTLER_TEMP_LIQUID0_BIT',
    'THROTTLER_TEMP_LIQUID1_BIT', 'THROTTLER_TEMP_MEM_BIT',
    'THROTTLER_TEMP_PLX_BIT', 'THROTTLER_TEMP_VR_GFX_BIT',
    'THROTTLER_TEMP_VR_MEM0_BIT', 'THROTTLER_TEMP_VR_MEM1_BIT',
    'THROTTLER_TEMP_VR_SOC_BIT', 'UCLK_DIV_BY_1', 'UCLK_DIV_BY_2',
    'UCLK_DIV_BY_4', 'UCLK_DIV_BY_8', 'UCLK_DIV_e',
    'UCLK_DIV_e__enumvalues', 'ULPS_SEQUENCE', 'VOLTAGE_MODE_COUNT',
    'VOLTAGE_MODE_FUSES', 'VOLTAGE_MODE_PPTABLE', 'VOLTAGE_MODE_e',
    'VOLTAGE_MODE_e__enumvalues', 'VR_MAPPING_PLANE_SELECT_MASK',
    'VR_MAPPING_PLANE_SELECT_SHIFT', 'VR_MAPPING_VR_SELECT_MASK',
    'VR_MAPPING_VR_SELECT_SHIFT', 'WATERMARKS_CLOCK_RANGE',
    'WATERMARKS_COUNT', 'WATERMARKS_DUMMY_PSTATE',
    'WATERMARKS_FLAGS_e', 'WATERMARKS_FLAGS_e__enumvalues',
    'WATERMARKS_MALL', 'WORKLOAD_PPLIB_CGVDI_BIT',
    'WORKLOAD_PPLIB_COMPUTE_BIT', 'WORKLOAD_PPLIB_COUNT',
    'WORKLOAD_PPLIB_CUSTOM_BIT', 'WORKLOAD_PPLIB_DEFAULT_BIT',
    'WORKLOAD_PPLIB_DIRECT_ML_BIT',
    'WORKLOAD_PPLIB_FULL_SCREEN_3D_BIT',
    'WORKLOAD_PPLIB_POWER_SAVING_BIT', 'WORKLOAD_PPLIB_VIDEO_BIT',
    'WORKLOAD_PPLIB_VR_BIT', 'WORKLOAD_PPLIB_WINDOW_3D_BIT',
    'WatermarkRowGeneric_t', 'WatermarksExternal_t', 'Watermarks_t',
    '__AMDGPU_SMU_H__', '__SMU_V14_0_0_PMFW_H__', 'bool',
    'c__EA_AVFS_D_e', 'c__EA_AVFS_TEMP_e',
    'c__EA_AVFS_VOLTAGE_TYPE_e', 'c__EA_BOARD_GPIO_TYPE_e',
    'c__EA_CUSTOMER_VARIANT_e', 'c__EA_D3HOTSequence_e',
    'c__EA_DCS_ARCH_e', 'c__EA_DRAM_BIT_WIDTH_TYPE_e',
    'c__EA_EPCS_STATUS_e', 'c__EA_FEATURE_BTC_e',
    'c__EA_FEATURE_PWR_DOMAIN_e', 'c__EA_FOPT_CALC_e',
    'c__EA_FanMode_e', 'c__EA_GpioIntPolarity_e',
    'c__EA_I2cCmdType_e', 'c__EA_I2cControllerName_e',
    'c__EA_I2cControllerPort_e', 'c__EA_I2cControllerProtocol_e',
    'c__EA_I2cControllerThrottler_e', 'c__EA_I2cPort_e',
    'c__EA_I2cSpeed_e', 'c__EA_MEM_VENDOR_e', 'c__EA_OD_FAIL_e',
    'c__EA_PMFW_VOLT_PLANE_e', 'c__EA_POWER_SOURCE_e',
    'c__EA_PPCLK_e', 'c__EA_PPT_THROTTLER_e',
    'c__EA_PP_GRTAVFS_FW_COMMON_FUSE_e',
    'c__EA_PP_GRTAVFS_FW_SEP_FUSE_e', 'c__EA_PP_GRTAVFS_HW_FUSE_e',
    'c__EA_PP_OD_POWER_FEATURE_e', 'c__EA_PowerGatingMode_e',
    'c__EA_PowerGatingSettings_e', 'c__EA_PwrConfig_e',
    'c__EA_SMARTSHIFT_VERSION_e', 'c__EA_SVI_PLANE_e',
    'c__EA_SVI_PSI_e', 'c__EA_TDC_THROTTLER_e', 'c__EA_TEMP_e',
    'c__EA_UCLK_DIV_e', 'c__EA_VOLTAGE_MODE_e',
    'c__EA_WATERMARKS_FLAGS_e', 'int16_t', 'int32_t', 'int8_t',
    'smu_clk_type', 'smu_memory_pool_size',
    'smu_perf_level_designation', 'smu_power_src_type',
    'smu_ppt_limit_level', 'smu_ppt_limit_type',
    'smu_refreshrate_source', 'smu_state_classification_flag',
    'smu_state_ui_label', 'smu_table_id',
    'struct_SMU14_Firmware_Footer', 'struct_amdgpu_bo',
    'struct_c__SA_AvfsDcBtcParams_t',
    'struct_c__SA_AvfsDebugTableExternal_t',
    'struct_c__SA_AvfsDebugTable_t',
    'struct_c__SA_AvfsFuseOverride_t', 'struct_c__SA_BoardTable_t',
    'struct_c__SA_BootValues_t', 'struct_c__SA_CustomSkuTable_t',
    'struct_c__SA_DpmActivityMonitorCoeffIntExternal_t',
    'struct_c__SA_DpmActivityMonitorCoeffInt_t',
    'struct_c__SA_DpmDescriptor_t', 'struct_c__SA_DriverInfoTable_t',
    'struct_c__SA_DriverReportedClocks_t',
    'struct_c__SA_DriverSmuConfigExternal_t',
    'struct_c__SA_DriverSmuConfig_t', 'struct_c__SA_DroopInt_t',
    'struct_c__SA_EccInfoTable_t', 'struct_c__SA_EccInfo_t',
    'struct_c__SA_FwStatus_t', 'struct_c__SA_FwStatus_t_v14_0_1',
    'struct_c__SA_I2cControllerConfig_t', 'struct_c__SA_LinearInt_t',
    'struct_c__SA_MsgLimits_t', 'struct_c__SA_OverDriveLimits_t',
    'struct_c__SA_OverDriveTableExternal_t',
    'struct_c__SA_OverDriveTable_t', 'struct_c__SA_PFE_Settings_t',
    'struct_c__SA_PPTable_t', 'struct_c__SA_QuadraticInt_t',
    'struct_c__SA_SMU_Firmware_Header', 'struct_c__SA_SkuTable_t',
    'struct_c__SA_SmuMetricsExternal_t', 'struct_c__SA_SmuMetrics_t',
    'struct_c__SA_Svi3RegulatorSettings_t',
    'struct_c__SA_SviTelemetryScale_t', 'struct_c__SA_SwI2cCmd_t',
    'struct_c__SA_SwI2cRequestExternal_t',
    'struct_c__SA_SwI2cRequest_t',
    'struct_c__SA_WatermarkRowGeneric_t',
    'struct_c__SA_WatermarksExternal_t', 'struct_c__SA_Watermarks_t',
    'struct_smu_bios_boot_up_values', 'struct_smu_clock_info',
    'struct_smu_hw_power_state', 'struct_smu_performance_level',
    'struct_smu_power_state', 'struct_smu_state_classification_block',
    'struct_smu_state_display_block', 'struct_smu_state_memory_block',
    'struct_smu_state_pcie_block',
    'struct_smu_state_software_algorithm_block',
    'struct_smu_state_validation_block', 'struct_smu_table',
    'struct_smu_temperature_range', 'struct_smu_user_dpm_profile',
    'struct_smu_uvd_clocks', 'u32', 'uint16_t', 'uint32_t',
    'uint64_t', 'uint8_t']
