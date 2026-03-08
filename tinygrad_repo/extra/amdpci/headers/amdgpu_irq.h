/*
 * Copyright 2014 Advanced Micro Devices, Inc.
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
 *
 */

#ifndef __AMDGPU_IRQ_H__
#define __AMDGPU_IRQ_H__

// #include <linux/irqdomain.h>
// #include "soc15_ih_clientid.h"
// #include "amdgpu_ih.h"

#define int32_t int
#define uint32_t unsigned int
#define int8_t signed char
#define uint8_t unsigned char
#define uint16_t unsigned short
#define int16_t short
#define uint64_t unsigned long long
#define bool _Bool
#define u32 unsigned int

#define AMDGPU_MAX_IRQ_SRC_ID		0x100
#define AMDGPU_MAX_IRQ_CLIENT_ID	0x100

#define AMDGPU_IRQ_CLIENTID_LEGACY	0
#define AMDGPU_IRQ_CLIENTID_MAX		SOC15_IH_CLIENTID_MAX

#define AMDGPU_IRQ_SRC_DATA_MAX_SIZE_DW	4

struct amdgpu_device;

enum amdgpu_interrupt_state {
	AMDGPU_IRQ_STATE_DISABLE,
	AMDGPU_IRQ_STATE_ENABLE,
};

struct amdgpu_iv_entry {
	// struct amdgpu_ih_ring *ih;
	unsigned client_id;
	unsigned src_id;
	unsigned ring_id;
	unsigned vmid;
	unsigned vmid_src;
	uint64_t timestamp;
	unsigned timestamp_src;
	unsigned pasid;
	unsigned node_id;
	unsigned src_data[AMDGPU_IRQ_SRC_DATA_MAX_SIZE_DW];
	const uint32_t *iv_entry;
};

enum interrupt_node_id_per_aid {
	AID0_NODEID = 0,
	XCD0_NODEID = 1,
	XCD1_NODEID = 2,
	AID1_NODEID = 4,
	XCD2_NODEID = 5,
	XCD3_NODEID = 6,
	AID2_NODEID = 8,
	XCD4_NODEID = 9,
	XCD5_NODEID = 10,
	AID3_NODEID = 12,
	XCD6_NODEID = 13,
	XCD7_NODEID = 14,
	NODEID_MAX,
};

#endif
