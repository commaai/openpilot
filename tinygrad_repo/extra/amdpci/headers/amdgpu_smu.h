/*
 * Copyright 2019 Advanced Micro Devices, Inc.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef __AMDGPU_SMU_H__
#define __AMDGPU_SMU_H__

#define int32_t int
#define uint32_t unsigned int
#define int8_t signed char
#define uint8_t unsigned char
#define uint16_t unsigned short
#define int16_t short
#define uint64_t unsigned long long
#define bool _Bool
#define u32 unsigned int

#define SMU_THERMAL_MINIMUM_ALERT_TEMP		0
#define SMU_THERMAL_MAXIMUM_ALERT_TEMP		255
#define SMU_TEMPERATURE_UNITS_PER_CENTIGRADES	1000
#define SMU_FW_NAME_LEN			0x24

#define SMU_DPM_USER_PROFILE_RESTORE (1 << 0)
#define SMU_CUSTOM_FAN_SPEED_RPM     (1 << 1)
#define SMU_CUSTOM_FAN_SPEED_PWM     (1 << 2)

// Power Throttlers
#define SMU_THROTTLER_PPT0_BIT			0
#define SMU_THROTTLER_PPT1_BIT			1
#define SMU_THROTTLER_PPT2_BIT			2
#define SMU_THROTTLER_PPT3_BIT			3
#define SMU_THROTTLER_SPL_BIT			4
#define SMU_THROTTLER_FPPT_BIT			5
#define SMU_THROTTLER_SPPT_BIT			6
#define SMU_THROTTLER_SPPT_APU_BIT		7

// Current Throttlers
#define SMU_THROTTLER_TDC_GFX_BIT		16
#define SMU_THROTTLER_TDC_SOC_BIT		17
#define SMU_THROTTLER_TDC_MEM_BIT		18
#define SMU_THROTTLER_TDC_VDD_BIT		19
#define SMU_THROTTLER_TDC_CVIP_BIT		20
#define SMU_THROTTLER_EDC_CPU_BIT		21
#define SMU_THROTTLER_EDC_GFX_BIT		22
#define SMU_THROTTLER_APCC_BIT			23

// Temperature
#define SMU_THROTTLER_TEMP_GPU_BIT		32
#define SMU_THROTTLER_TEMP_CORE_BIT		33
#define SMU_THROTTLER_TEMP_MEM_BIT		34
#define SMU_THROTTLER_TEMP_EDGE_BIT		35
#define SMU_THROTTLER_TEMP_HOTSPOT_BIT		36
#define SMU_THROTTLER_TEMP_SOC_BIT		37
#define SMU_THROTTLER_TEMP_VR_GFX_BIT		38
#define SMU_THROTTLER_TEMP_VR_SOC_BIT		39
#define SMU_THROTTLER_TEMP_VR_MEM0_BIT		40
#define SMU_THROTTLER_TEMP_VR_MEM1_BIT		41
#define SMU_THROTTLER_TEMP_LIQUID0_BIT		42
#define SMU_THROTTLER_TEMP_LIQUID1_BIT		43
#define SMU_THROTTLER_VRHOT0_BIT		44
#define SMU_THROTTLER_VRHOT1_BIT		45
#define SMU_THROTTLER_PROCHOT_CPU_BIT		46
#define SMU_THROTTLER_PROCHOT_GFX_BIT		47

// Other
#define SMU_THROTTLER_PPM_BIT			56
#define SMU_THROTTLER_FIT_BIT			57

struct smu_hw_power_state {
	unsigned int magic;
};

struct smu_power_state;

enum smu_state_ui_label {
	SMU_STATE_UI_LABEL_NONE,
	SMU_STATE_UI_LABEL_BATTERY,
	SMU_STATE_UI_TABEL_MIDDLE_LOW,
	SMU_STATE_UI_LABEL_BALLANCED,
	SMU_STATE_UI_LABEL_MIDDLE_HIGHT,
	SMU_STATE_UI_LABEL_PERFORMANCE,
	SMU_STATE_UI_LABEL_BACO,
};

enum smu_state_classification_flag {
	SMU_STATE_CLASSIFICATION_FLAG_BOOT                     = 0x0001,
	SMU_STATE_CLASSIFICATION_FLAG_THERMAL                  = 0x0002,
	SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE      = 0x0004,
	SMU_STATE_CLASSIFICATION_FLAG_RESET                    = 0x0008,
	SMU_STATE_CLASSIFICATION_FLAG_FORCED                   = 0x0010,
	SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE      = 0x0020,
	SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE      = 0x0040,
	SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE           = 0x0080,
	SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE   = 0x0100,
	SMU_STATE_CLASSIFICATION_FLAG_UVD                      = 0x0200,
	SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW       = 0x0400,
	SMU_STATE_CLASSIFICATION_FLAG_ACPI                     = 0x0800,
	SMU_STATE_CLASSIFICATION_FLAG_HD2                      = 0x1000,
	SMU_STATE_CLASSIFICATION_FLAG_UVD_HD                   = 0x2000,
	SMU_STATE_CLASSIFICATION_FLAG_UVD_SD                   = 0x4000,
	SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE      = 0x8000,
	SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE   = 0x10000,
	SMU_STATE_CLASSIFICATION_FLAG_BACO                     = 0x20000,
	SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2      = 0x40000,
	SMU_STATE_CLASSIFICATION_FLAG_ULV                      = 0x80000,
	SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC                  = 0x100000,
};

struct smu_state_classification_block {
	enum smu_state_ui_label         ui_label;
	enum smu_state_classification_flag  flags;
	int                          bios_index;
	bool                      temporary_state;
	bool                      to_be_deleted;
};

struct smu_state_pcie_block {
	unsigned int lanes;
};

enum smu_refreshrate_source {
	SMU_REFRESHRATE_SOURCE_EDID,
	SMU_REFRESHRATE_SOURCE_EXPLICIT
};

struct smu_state_display_block {
	bool              disable_frame_modulation;
	bool              limit_refreshrate;
	enum smu_refreshrate_source refreshrate_source;
	int                  explicit_refreshrate;
	int                  edid_refreshrate_index;
	bool              enable_vari_bright;
};

struct smu_state_memory_block {
	bool              dll_off;
	uint8_t                 m3arb;
	uint8_t                 unused[3];
};

struct smu_state_software_algorithm_block {
	bool disable_load_balancing;
	bool enable_sleep_for_timestamps;
};

struct smu_temperature_range {
	int min;
	int max;
	int edge_emergency_max;
	int hotspot_min;
	int hotspot_crit_max;
	int hotspot_emergency_max;
	int mem_min;
	int mem_crit_max;
	int mem_emergency_max;
	int software_shutdown_temp;
	int software_shutdown_temp_offset;
};

struct smu_state_validation_block {
	bool single_display_only;
	bool disallow_on_dc;
	uint8_t supported_power_levels;
};

struct smu_uvd_clocks {
	uint32_t vclk;
	uint32_t dclk;
};

/**
* Structure to hold a SMU Power State.
*/

enum smu_power_src_type {
	SMU_POWER_SOURCE_AC,
	SMU_POWER_SOURCE_DC,
	SMU_POWER_SOURCE_COUNT,
};

enum smu_ppt_limit_type {
	SMU_DEFAULT_PPT_LIMIT = 0,
	SMU_FAST_PPT_LIMIT,
};

enum smu_ppt_limit_level {
	SMU_PPT_LIMIT_MIN = -1,
	SMU_PPT_LIMIT_CURRENT,
	SMU_PPT_LIMIT_DEFAULT,
	SMU_PPT_LIMIT_MAX,
};

enum smu_memory_pool_size {
    SMU_MEMORY_POOL_SIZE_ZERO   = 0,
    SMU_MEMORY_POOL_SIZE_256_MB = 0x10000000,
    SMU_MEMORY_POOL_SIZE_512_MB = 0x20000000,
    SMU_MEMORY_POOL_SIZE_1_GB   = 0x40000000,
    SMU_MEMORY_POOL_SIZE_2_GB   = 0x80000000,
};

enum smu_clk_type {
	SMU_GFXCLK,
	SMU_VCLK,
	SMU_DCLK,
	SMU_VCLK1,
	SMU_DCLK1,
	SMU_ECLK,
	SMU_SOCCLK,
	SMU_UCLK,
	SMU_DCEFCLK,
	SMU_DISPCLK,
	SMU_PIXCLK,
	SMU_PHYCLK,
	SMU_FCLK,
	SMU_SCLK,
	SMU_MCLK,
	SMU_PCIE,
	SMU_LCLK,
	SMU_OD_CCLK,
	SMU_OD_SCLK,
	SMU_OD_MCLK,
	SMU_OD_VDDC_CURVE,
	SMU_OD_RANGE,
	SMU_OD_VDDGFX_OFFSET,
	SMU_OD_FAN_CURVE,
	SMU_OD_ACOUSTIC_LIMIT,
	SMU_OD_ACOUSTIC_TARGET,
	SMU_OD_FAN_TARGET_TEMPERATURE,
	SMU_OD_FAN_MINIMUM_PWM,
	SMU_CLK_COUNT,
};

struct smu_user_dpm_profile {
	uint32_t fan_mode;
	uint32_t power_limit;
	uint32_t fan_speed_pwm;
	uint32_t fan_speed_rpm;
	uint32_t flags;
	uint32_t user_od;

	/* user clock state information */
	uint32_t clk_mask[SMU_CLK_COUNT];
	uint32_t clk_dependency;
};

#define SMU_TABLE_INIT(tables, table_id, s, a, d)	\
	do {						\
		tables[table_id].size = s;		\
		tables[table_id].align = a;		\
		tables[table_id].domain = d;		\
	} while (0)

struct smu_table {
	uint64_t size;
	uint32_t align;
	uint8_t domain;
	uint64_t mc_address;
	void *cpu_addr;
	struct amdgpu_bo *bo;
	uint32_t version;
};

enum smu_perf_level_designation {
	PERF_LEVEL_ACTIVITY,
	PERF_LEVEL_POWER_CONTAINMENT,
};

struct smu_performance_level {
	uint32_t core_clock;
	uint32_t memory_clock;
	uint32_t vddc;
	uint32_t vddci;
	uint32_t non_local_mem_freq;
	uint32_t non_local_mem_width;
};

struct smu_clock_info {
	uint32_t min_mem_clk;
	uint32_t max_mem_clk;
	uint32_t min_eng_clk;
	uint32_t max_eng_clk;
	uint32_t min_bus_bandwidth;
	uint32_t max_bus_bandwidth;
};

struct smu_bios_boot_up_values {
	uint32_t			revision;
	uint32_t			gfxclk;
	uint32_t			uclk;
	uint32_t			socclk;
	uint32_t			dcefclk;
	uint32_t			eclk;
	uint32_t			vclk;
	uint32_t			dclk;
	uint16_t			vddc;
	uint16_t			vddci;
	uint16_t			mvddc;
	uint16_t			vdd_gfx;
	uint8_t				cooling_id;
	uint32_t			pp_table_id;
	uint32_t			format_revision;
	uint32_t			content_revision;
	uint32_t			fclk;
	uint32_t			lclk;
	uint32_t			firmware_caps;
};

enum smu_table_id {
	SMU_TABLE_PPTABLE = 0,
	SMU_TABLE_WATERMARKS,
	SMU_TABLE_CUSTOM_DPM,
	SMU_TABLE_DPMCLOCKS,
	SMU_TABLE_AVFS,
	SMU_TABLE_AVFS_PSM_DEBUG,
	SMU_TABLE_AVFS_FUSE_OVERRIDE,
	SMU_TABLE_PMSTATUSLOG,
	SMU_TABLE_SMU_METRICS,
	SMU_TABLE_DRIVER_SMU_CONFIG,
	SMU_TABLE_ACTIVITY_MONITOR_COEFF,
	SMU_TABLE_OVERDRIVE,
	SMU_TABLE_I2C_COMMANDS,
	SMU_TABLE_PACE,
	SMU_TABLE_ECCINFO,
	SMU_TABLE_COMBO_PPTABLE,
	SMU_TABLE_WIFIBAND,
	SMU_TABLE_COUNT,
};


#endif
