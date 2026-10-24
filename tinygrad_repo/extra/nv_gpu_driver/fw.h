/* adapted from linux/drivers/gpu/drm/nouveau/include/nvfw/fw.h */
/* SPDX-License-Identifier: MIT */
#ifndef __NVFW_FW_H__
#define __NVFW_FW_H__
typedef unsigned int u32;

struct nvfw_bin_hdr {
	u32 bin_magic;
	u32 bin_ver;
	u32 bin_size;
	u32 header_offset;
	u32 data_offset;
	u32 data_size;
};

struct nvfw_bl_desc {
	u32 start_tag;
	u32 dmem_load_off;
	u32 code_off;
	u32 code_size;
	u32 data_off;
	u32 data_size;
};

#endif
