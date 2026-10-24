/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GSP_STATIC_CONFIG_H
#define GSP_STATIC_CONFIG_H

//
// This header describes the set of static GPU configuration information
// that is collected during GSP RM init and made available to the
// CPU RM (aka GSP client) via NV_RM_RPC_GET_GSP_STATIC_INFO() call.

#include "ctrl/ctrl0080/ctrl0080gpu.h"
#include "ctrl/ctrl2080/ctrl2080bios.h"
#include "ctrl/ctrl2080/ctrl2080fb.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"

#include "vgpu/rpc_headers.h"
#include "nvacpitypes.h"

#include "ctrl/ctrl0073/ctrl0073system.h"

#define MAX_DSM_SUPPORTED_FUNCS_RTN_LEN 8 // # bytes to store supported functions
#define NV_ACPI_GENERIC_FUNC_COUNT                  8

#define REGISTRY_TABLE_ENTRY_TYPE_UNKNOWN  0
#define REGISTRY_TABLE_ENTRY_TYPE_DWORD    1
#define REGISTRY_TABLE_ENTRY_TYPE_BINARY   2
#define REGISTRY_TABLE_ENTRY_TYPE_STRING   3
typedef struct PACKED_REGISTRY_ENTRY
{
    NvU32                   nameOffset;
    NvU8                    type;
    NvU32                   data;
    NvU32                   length;
} PACKED_REGISTRY_ENTRY;

typedef struct PACKED_REGISTRY_TABLE
{
    NvU32                   size;
    NvU32                   numEntries;
} PACKED_REGISTRY_TABLE;

/* Indicates the current state of mux */
typedef enum
{
    dispMuxState_None = 0,
    dispMuxState_IntegratedGPU,
    dispMuxState_DiscreteGPU,
} DISPMUXSTATE;

typedef struct {
    // supported function status and cache
    NvU32  suppFuncStatus;
    NvU8   suppFuncs[MAX_DSM_SUPPORTED_FUNCS_RTN_LEN];
    NvU32  suppFuncsLen;
    NvBool bArg3isInteger;
    // callback status and cache
    NvU32  callbackStatus;
    NvU32  callback;
} ACPI_DSM_CACHE;

typedef struct {

    ACPI_DSM_CACHE                   dsm[ACPI_DSM_FUNCTION_COUNT];
    ACPI_DSM_FUNCTION                dispStatusHotplugFunc;
    ACPI_DSM_FUNCTION                dispStatusConfigFunc;
    ACPI_DSM_FUNCTION                perfPostPowerStateFunc;
    ACPI_DSM_FUNCTION                stereo3dStateActiveFunc;
    NvU32                            dsmPlatCapsCache[ACPI_DSM_FUNCTION_COUNT];
    NvU32                            MDTLFeatureSupport;

    // cache of generic func/subfunction remappings.
    ACPI_DSM_FUNCTION                dsmCurrentFunc[NV_ACPI_GENERIC_FUNC_COUNT];
    NvU32                            dsmCurrentSubFunc[NV_ACPI_GENERIC_FUNC_COUNT];
    NvU32                            dsmCurrentFuncSupport;

} ACPI_DATA;

typedef struct DOD_METHOD_DATA
{
    NV_STATUS status;
    NvU32     acpiIdListLen;
    NvU32     acpiIdList[NV0073_CTRL_SYSTEM_ACPI_ID_MAP_MAX_DISPLAYS];
} DOD_METHOD_DATA;

typedef struct JT_METHOD_DATA
{
    NV_STATUS status;
    NvU32     jtCaps;
    NvU16     jtRevId;
    NvBool    bSBIOSCaps;
} JT_METHOD_DATA;

typedef struct MUX_METHOD_DATA_ELEMENT
{
    NvU32       acpiId;
    NvU32       mode;
    NV_STATUS   status;
} MUX_METHOD_DATA_ELEMENT;

typedef struct MUX_METHOD_DATA
{
    NvU32                       tableLen;
    MUX_METHOD_DATA_ELEMENT     acpiIdMuxModeTable[NV0073_CTRL_SYSTEM_ACPI_ID_MAP_MAX_DISPLAYS];
    MUX_METHOD_DATA_ELEMENT     acpiIdMuxPartTable[NV0073_CTRL_SYSTEM_ACPI_ID_MAP_MAX_DISPLAYS];
    MUX_METHOD_DATA_ELEMENT     acpiIdMuxStateTable[NV0073_CTRL_SYSTEM_ACPI_ID_MAP_MAX_DISPLAYS];    
} MUX_METHOD_DATA;

typedef struct CAPS_METHOD_DATA
{
    NV_STATUS status;
    NvU32     optimusCaps;
} CAPS_METHOD_DATA;

typedef struct ACPI_METHOD_DATA
{
    NvBool                                               bValid;
    DOD_METHOD_DATA                                      dodMethodData;
    JT_METHOD_DATA                                       jtMethodData;
    MUX_METHOD_DATA                                      muxMethodData;
    CAPS_METHOD_DATA                                     capsMethodData;
} ACPI_METHOD_DATA;

#define MAX_GROUP_COUNT 2

// #include "gpu/nvbitmask.h"
typedef enum
{
    RM_ENGINE_TYPE_NULL                            =       (0x00000000),
    RM_ENGINE_TYPE_GR0                             =       (0x00000001),
    RM_ENGINE_TYPE_GR1                             =       (0x00000002),
    RM_ENGINE_TYPE_GR2                             =       (0x00000003),
    RM_ENGINE_TYPE_GR3                             =       (0x00000004),
    RM_ENGINE_TYPE_GR4                             =       (0x00000005),
    RM_ENGINE_TYPE_GR5                             =       (0x00000006),
    RM_ENGINE_TYPE_GR6                             =       (0x00000007),
    RM_ENGINE_TYPE_GR7                             =       (0x00000008),
    RM_ENGINE_TYPE_COPY0                           =       (0x00000009),
    RM_ENGINE_TYPE_COPY1                           =       (0x0000000a),
    RM_ENGINE_TYPE_COPY2                           =       (0x0000000b),
    RM_ENGINE_TYPE_COPY3                           =       (0x0000000c),
    RM_ENGINE_TYPE_COPY4                           =       (0x0000000d),
    RM_ENGINE_TYPE_COPY5                           =       (0x0000000e),
    RM_ENGINE_TYPE_COPY6                           =       (0x0000000f),
    RM_ENGINE_TYPE_COPY7                           =       (0x00000010),
    RM_ENGINE_TYPE_COPY8                           =       (0x00000011),
    RM_ENGINE_TYPE_COPY9                           =       (0x00000012),
    RM_ENGINE_TYPE_COPY10                          =       (0x00000013),
    RM_ENGINE_TYPE_COPY11                          =       (0x00000014),
    RM_ENGINE_TYPE_COPY12                          =       (0x00000015),
    RM_ENGINE_TYPE_COPY13                          =       (0x00000016),
    RM_ENGINE_TYPE_COPY14                          =       (0x00000017),
    RM_ENGINE_TYPE_COPY15                          =       (0x00000018),
    RM_ENGINE_TYPE_COPY16                          =       (0x00000019),
    RM_ENGINE_TYPE_COPY17                          =       (0x0000001a),
    RM_ENGINE_TYPE_COPY18                          =       (0x0000001b),
    RM_ENGINE_TYPE_COPY19                          =       (0x0000001c),
    RM_ENGINE_TYPE_NVDEC0                          =       (0x0000001d),
    RM_ENGINE_TYPE_NVDEC1                          =       (0x0000001e),
    RM_ENGINE_TYPE_NVDEC2                          =       (0x0000001f),
    RM_ENGINE_TYPE_NVDEC3                          =       (0x00000020),
    RM_ENGINE_TYPE_NVDEC4                          =       (0x00000021),
    RM_ENGINE_TYPE_NVDEC5                          =       (0x00000022),
    RM_ENGINE_TYPE_NVDEC6                          =       (0x00000023),
    RM_ENGINE_TYPE_NVDEC7                          =       (0x00000024),
    RM_ENGINE_TYPE_NVENC0                          =       (0x00000025),
    RM_ENGINE_TYPE_NVENC1                          =       (0x00000026),
    RM_ENGINE_TYPE_NVENC2                          =       (0x00000027),
    // Bug 4175886 - Use this new value for all chips once GB20X is released
    RM_ENGINE_TYPE_NVENC3                          =       (0x00000028),
    RM_ENGINE_TYPE_VP                              =       (0x00000029),
    RM_ENGINE_TYPE_ME                              =       (0x0000002a),
    RM_ENGINE_TYPE_PPP                             =       (0x0000002b),
    RM_ENGINE_TYPE_MPEG                            =       (0x0000002c),
    RM_ENGINE_TYPE_SW                              =       (0x0000002d),
    RM_ENGINE_TYPE_TSEC                            =       (0x0000002e),
    RM_ENGINE_TYPE_VIC                             =       (0x0000002f),
    RM_ENGINE_TYPE_MP                              =       (0x00000030),
    RM_ENGINE_TYPE_SEC2                            =       (0x00000031),
    RM_ENGINE_TYPE_HOST                            =       (0x00000032),
    RM_ENGINE_TYPE_DPU                             =       (0x00000033),
    RM_ENGINE_TYPE_PMU                             =       (0x00000034),
    RM_ENGINE_TYPE_FBFLCN                          =       (0x00000035),
    RM_ENGINE_TYPE_NVJPEG0                         =       (0x00000036),
    RM_ENGINE_TYPE_NVJPEG1                         =       (0x00000037),
    RM_ENGINE_TYPE_NVJPEG2                         =       (0x00000038),
    RM_ENGINE_TYPE_NVJPEG3                         =       (0x00000039),
    RM_ENGINE_TYPE_NVJPEG4                         =       (0x0000003a),
    RM_ENGINE_TYPE_NVJPEG5                         =       (0x0000003b),
    RM_ENGINE_TYPE_NVJPEG6                         =       (0x0000003c),
    RM_ENGINE_TYPE_NVJPEG7                         =       (0x0000003d),
    RM_ENGINE_TYPE_OFA0                            =       (0x0000003e),
    RM_ENGINE_TYPE_OFA1                            =       (0x0000003f),
    RM_ENGINE_TYPE_RESERVED40                      =       (0x00000040),
    RM_ENGINE_TYPE_RESERVED41                      =       (0x00000041),
    RM_ENGINE_TYPE_RESERVED42                      =       (0x00000042),
    RM_ENGINE_TYPE_RESERVED43                      =       (0x00000043),
    RM_ENGINE_TYPE_RESERVED44                      =       (0x00000044),
    RM_ENGINE_TYPE_RESERVED45                      =       (0x00000045),
    RM_ENGINE_TYPE_RESERVED46                      =       (0x00000046),
    RM_ENGINE_TYPE_RESERVED47                      =       (0x00000047),
    RM_ENGINE_TYPE_RESERVED48                      =       (0x00000048),
    RM_ENGINE_TYPE_RESERVED49                      =       (0x00000049),
    RM_ENGINE_TYPE_RESERVED4a                      =       (0x0000004a),
    RM_ENGINE_TYPE_RESERVED4b                      =       (0x0000004b),
    RM_ENGINE_TYPE_RESERVED4c                      =       (0x0000004c),
    RM_ENGINE_TYPE_RESERVED4d                      =       (0x0000004d),
    RM_ENGINE_TYPE_RESERVED4e                      =       (0x0000004e),
    RM_ENGINE_TYPE_RESERVED4f                      =       (0x0000004f),
    RM_ENGINE_TYPE_RESERVED50                      =       (0x00000050),
    RM_ENGINE_TYPE_RESERVED51                      =       (0x00000051),
    RM_ENGINE_TYPE_RESERVED52                      =       (0x00000052),
    RM_ENGINE_TYPE_RESERVED53                      =       (0x00000053),
    RM_ENGINE_TYPE_LAST                            =       (0x00000054),
} RM_ENGINE_TYPE;

//
// The duplicates in the RM_ENGINE_TYPE. Using define instead of putting them
// in the enum to make sure that each item in the enum has a unique number.
//
#define RM_ENGINE_TYPE_GRAPHICS                 RM_ENGINE_TYPE_GR0
#define RM_ENGINE_TYPE_BSP                      RM_ENGINE_TYPE_NVDEC0
#define RM_ENGINE_TYPE_MSENC                    RM_ENGINE_TYPE_NVENC0
#define RM_ENGINE_TYPE_CIPHER                   RM_ENGINE_TYPE_TSEC
#define RM_ENGINE_TYPE_NVJPG                    RM_ENGINE_TYPE_NVJPEG0

#define RM_ENGINE_TYPE_COPY_SIZE 20
// Bug 4175886 - Use this new value for all chips once GB20X is released
#define RM_ENGINE_TYPE_NVENC_SIZE 4
#define RM_ENGINE_TYPE_NVJPEG_SIZE 8
#define RM_ENGINE_TYPE_NVDEC_SIZE 8
#define RM_ENGINE_TYPE_OFA_SIZE 2
#define RM_ENGINE_TYPE_GR_SIZE 8

#define NVGPU_ENGINE_CAPS_MASK_BITS                32
#define NVGPU_ENGINE_CAPS_MASK_ARRAY_MAX           ((RM_ENGINE_TYPE_LAST-1)/NVGPU_ENGINE_CAPS_MASK_BITS + 1)
#define NVGPU_GET_ENGINE_CAPS_MASK(caps, id)       (caps[(id)/NVGPU_ENGINE_CAPS_MASK_BITS] & NVBIT((id) % NVGPU_ENGINE_CAPS_MASK_BITS))
#define NVGPU_SET_ENGINE_CAPS_MASK(caps, id)       (caps[(id)/NVGPU_ENGINE_CAPS_MASK_BITS] |= NVBIT((id) % NVGPU_ENGINE_CAPS_MASK_BITS))


// #include "gpu/gpu.h" // COMPUTE_BRANDING_TYPE
// #include "gpu/gpu_acpi_data.h" // ACPI_METHOD_DATA
// #include "vgpu/rpc_headers.h" // MAX_GPC_COUNT
// #include "platform/chipset/chipset.h" // BUSINFO
// #include "gpu/nvbitmask.h" // NVGPU_ENGINE_CAPS_MASK_ARRAY_MAX

typedef struct
{
    NvU16               deviceID;           // deviceID
    NvU16               vendorID;           // vendorID
    NvU16               subdeviceID;        // subsystem deviceID
    NvU16               subvendorID;        // subsystem vendorID
    NvU8                revisionID;         // revision ID
} BUSINFO;

// VF related info for GSP-RM
typedef struct GSP_VF_INFO
{
    NvU32  totalVFs;
    NvU32  firstVFOffset;
    NvU64  FirstVFBar0Address;
    NvU64  FirstVFBar1Address;
    NvU64  FirstVFBar2Address;
    NvBool b64bitBar0;
    NvBool b64bitBar1;
    NvBool b64bitBar2;
} GSP_VF_INFO;

// Cache config registers from pcie space
typedef struct
{
    // Link capabilities
    NvU32 linkCap;
} GSP_PCIE_CONFIG_REG;

typedef struct
{
    NvU32 ecidLow;
    NvU32 ecidHigh;
    NvU32 ecidExtended;
} EcidManufacturingInfo;

typedef struct
{
    NvU64 nonWprHeapOffset;
    NvU64 frtsOffset;
} FW_WPR_LAYOUT_OFFSET;

// Fetched from GSP-RM into CPU-RM
typedef struct GspStaticConfigInfo_t
{
    NvU8 grCapsBits[NV0080_CTRL_GR_CAPS_TBL_SIZE];
    NV2080_CTRL_GPU_GET_GID_INFO_PARAMS gidInfo;
    NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS SKUInfo;
    NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS fbRegionInfoParams;

    NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS sriovCaps;
    NvU32 sriovMaxGfid;

    NvU32 engineCaps[NVGPU_ENGINE_CAPS_MASK_ARRAY_MAX];

    NvBool poisonFuseEnabled;

    NvU64 fb_length;
    NvU64 fbio_mask;
    NvU32 fb_bus_width;
    NvU32 fb_ram_type;
    NvU64 fbp_mask;
    NvU32 l2_cache_size;

    NvU8 gpuNameString[NV2080_GPU_MAX_NAME_STRING_LENGTH];
    NvU8 gpuShortNameString[NV2080_GPU_MAX_NAME_STRING_LENGTH];
    NvU16 gpuNameString_Unicode[NV2080_GPU_MAX_NAME_STRING_LENGTH];
    NvBool bGpuInternalSku;
    NvBool bIsQuadroGeneric;
    NvBool bIsQuadroAd;
    NvBool bIsNvidiaNvs;
    NvBool bIsVgx;
    NvBool bGeforceSmb;
    NvBool bIsTitan;
    NvBool bIsTesla;
    NvBool bIsMobile;
    NvBool bIsGc6Rtd3Allowed;
    NvBool bIsGc8Rtd3Allowed;
    NvBool bIsGcOffRtd3Allowed;
    NvBool bIsGcoffLegacyAllowed;
    NvBool bIsMigSupported;

    /* "Total Board Power" refers to power requirement of GPU,
     * while in GC6 state. Majority of this power will be used
     * to keep V-RAM active to preserve its content.
     * Some energy maybe consumed by Always-on components on GPU chip.
     * This power will be provided by 3.3v voltage rail.
     */
    NvU16  RTD3GC6TotalBoardPower;

    /* PERST# (i.e. PCI Express Reset) is a sideband signal
     * generated by the PCIe Host to indicate the PCIe devices,
     * that the power-rails and the reference-clock are stable.
     * The endpoint device typically uses this signal as a global reset.
     */
    NvU16  RTD3GC6PerstDelay;

    NvU64 bar1PdeBase;
    NvU64 bar2PdeBase;

    NvBool bVbiosValid;
    NvU32 vbiosSubVendor;
    NvU32 vbiosSubDevice;

    NvBool bPageRetirementSupported;

    NvBool bSplitVasBetweenServerClientRm;

    NvBool bClRootportNeedsNosnoopWAR;

    VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS displaylessMaxHeads;
    VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS displaylessMaxResolution;
    NvU64 displaylessMaxPixels;

    // Client handle for internal RMAPI control.
    NvHandle hInternalClient;

    // Device handle for internal RMAPI control.
    NvHandle hInternalDevice;

    // Subdevice handle for internal RMAPI control.
    NvHandle hInternalSubdevice;

    NvBool bSelfHostedMode;
    NvBool bAtsSupported;

    NvBool bIsGpuUefi;
    NvBool bIsEfiInit;

    EcidManufacturingInfo ecidInfo[MAX_GROUP_COUNT];

    FW_WPR_LAYOUT_OFFSET fwWprLayoutOffset;
} GspStaticConfigInfo;

// Pushed from CPU-RM to GSP-RM
typedef struct GspSystemInfo
{
    NvU64 gpuPhysAddr;
    NvU64 gpuPhysFbAddr;
    NvU64 gpuPhysInstAddr;
    NvU64 gpuPhysIoAddr;
    NvU64 nvDomainBusDeviceFunc;
    NvU64 simAccessBufPhysAddr;
    NvU64 notifyOpSharedSurfacePhysAddr;
    NvU64 pcieAtomicsOpMask;
    NvU64 consoleMemSize;
    NvU64 maxUserVa;
    NvU32 pciConfigMirrorBase;
    NvU32 pciConfigMirrorSize;
    NvU32 PCIDeviceID;
    NvU32 PCISubDeviceID;
    NvU32 PCIRevisionID;
    NvU32 pcieAtomicsCplDeviceCapMask;
    NvU8 oorArch;
    NvU64 clPdbProperties;
    NvU32 Chipset;
    NvBool bGpuBehindBridge;
    NvBool bFlrSupported;
    NvBool b64bBar0Supported;
    NvBool bMnocAvailable;
    NvU32  chipsetL1ssEnable;
    NvBool bUpstreamL0sUnsupported;
    NvBool bUpstreamL1Unsupported;
    NvBool bUpstreamL1PorSupported;
    NvBool bUpstreamL1PorMobileOnly;
    NvBool bSystemHasMux;
    NvU8   upstreamAddressValid;
    BUSINFO FHBBusInfo;
    BUSINFO chipsetIDInfo;
    ACPI_METHOD_DATA acpiMethodData;
    NvU32 hypervisorType;
    NvBool bIsPassthru;
    NvU64 sysTimerOffsetNs;
    GSP_VF_INFO gspVFInfo;
    NvBool bIsPrimary;
    NvBool isGridBuild;
    GSP_PCIE_CONFIG_REG pcieConfigReg;
    NvU32 gridBuildCsp;
    NvBool bPreserveVideoMemoryAllocations;
    NvBool bTdrEventSupported;
    NvBool bFeatureStretchVblankCapable;
    NvBool bEnableDynamicGranularityPageArrays;
    NvBool bClockBoostSupported;
    NvBool bRouteDispIntrsToCPU;
    NvU64  hostPageSize;
} GspSystemInfo;


#endif /* GSP_STATIC_CONFIG_H */
