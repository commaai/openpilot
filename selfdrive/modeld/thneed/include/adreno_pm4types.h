/* Copyright (c) 2002,2007-2015, The Linux Foundation. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 and
 * only version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */
#ifndef __ADRENO_PM4TYPES_H
#define __ADRENO_PM4TYPES_H

//#include "adreno.h"

#define CP_PKT_MASK	0xc0000000

#define CP_TYPE0_PKT	((unsigned int)0 << 30)
#define CP_TYPE3_PKT	((unsigned int)3 << 30)
#define CP_TYPE4_PKT    ((unsigned int)4 << 28)
#define CP_TYPE7_PKT    ((unsigned int)7 << 28)

#define PM4_TYPE4_PKT_SIZE_MAX  128

/* type3 packets */

/* Enable preemption flag */
#define CP_PREEMPT_ENABLE 0x1C
/* Preemption token command on which preemption occurs */
#define CP_PREEMPT_TOKEN 0x1E
/* Bit to set in CP_PREEMPT_TOKEN ordinal for interrupt on preemption */
#define CP_PREEMPT_ORDINAL_INTERRUPT 24
/* copy from ME scratch RAM to a register */
#define CP_SCRATCH_TO_REG      0x4d

/* Copy from REG to ME scratch RAM */
#define CP_REG_TO_SCRATCH      0x4a

/* Wait for memory writes to complete */
#define CP_WAIT_MEM_WRITES     0x12

/* Conditional execution based on register comparison */
#define CP_COND_REG_EXEC       0x47

/* Memory to REG copy */
#define CP_MEM_TO_REG          0x42

/* initialize CP's micro-engine */
#define CP_ME_INIT		0x48

/* skip N 32-bit words to get to the next packet */
#define CP_NOP			0x10

/* indirect buffer dispatch.  same as IB, but init is pipelined */
#define CP_INDIRECT_BUFFER_PFD	0x37

/* wait for the IDLE state of the engine */
#define CP_WAIT_FOR_IDLE	0x26

/* wait until a register or memory location is a specific value */
#define CP_WAIT_REG_MEM	0x3c

/* wait until a register location is equal to a specific value */
#define CP_WAIT_REG_EQ		0x52

/* switches SMMU pagetable, used on a5xx only */
#define CP_SMMU_TABLE_UPDATE 0x53

/* wait until a read completes */
#define CP_WAIT_UNTIL_READ	0x5c

/* wait until all base/size writes from an IB_PFD packet have completed */
#define CP_WAIT_IB_PFD_COMPLETE 0x5d

/* register read/modify/write */
#define CP_REG_RMW		0x21

/* Set binning configuration registers */
#define CP_SET_BIN_DATA             0x2f

/* reads register in chip and writes to memory */
#define CP_REG_TO_MEM		0x3e

/* write N 32-bit words to memory */
#define CP_MEM_WRITE		0x3d

/* write CP_PROG_COUNTER value to memory */
#define CP_MEM_WRITE_CNTR	0x4f

/* conditional execution of a sequence of packets */
#define CP_COND_EXEC		0x44

/* conditional write to memory or register */
#define CP_COND_WRITE		0x45

/* generate an event that creates a write to memory when completed */
#define CP_EVENT_WRITE		0x46

/* generate a VS|PS_done event */
#define CP_EVENT_WRITE_SHD	0x58

/* generate a cache flush done event */
#define CP_EVENT_WRITE_CFL	0x59

/* generate a z_pass done event */
#define CP_EVENT_WRITE_ZPD	0x5b


/* initiate fetch of index buffer and draw */
#define CP_DRAW_INDX		0x22

/* draw using supplied indices in packet */
#define CP_DRAW_INDX_2		0x36

/* initiate fetch of index buffer and binIDs and draw */
#define CP_DRAW_INDX_BIN	0x34

/* initiate fetch of bin IDs and draw using supplied indices */
#define CP_DRAW_INDX_2_BIN	0x35

/* New draw packets defined for A4XX */
#define CP_DRAW_INDX_OFFSET	0x38
#define CP_DRAW_INDIRECT	0x28
#define CP_DRAW_INDX_INDIRECT	0x29
#define CP_DRAW_AUTO		0x24

/* begin/end initiator for viz query extent processing */
#define CP_VIZ_QUERY		0x23

/* fetch state sub-blocks and initiate shader code DMAs */
#define CP_SET_STATE		0x25

/* load constant into chip and to memory */
#define CP_SET_CONSTANT	0x2d

/* load sequencer instruction memory (pointer-based) */
#define CP_IM_LOAD		0x27

/* load sequencer instruction memory (code embedded in packet) */
#define CP_IM_LOAD_IMMEDIATE	0x2b

/* load constants from a location in memory */
#define CP_LOAD_CONSTANT_CONTEXT 0x2e

/* selective invalidation of state pointers */
#define CP_INVALIDATE_STATE	0x3b


/* dynamically changes shader instruction memory partition */
#define CP_SET_SHADER_BASES	0x4A

/* sets the 64-bit BIN_MASK register in the PFP */
#define CP_SET_BIN_MASK	0x50

/* sets the 64-bit BIN_SELECT register in the PFP */
#define CP_SET_BIN_SELECT	0x51


/* updates the current context, if needed */
#define CP_CONTEXT_UPDATE	0x5e

/* generate interrupt from the command stream */
#define CP_INTERRUPT		0x40

/* A5XX Enable yield in RB only */
#define CP_YIELD_ENABLE 0x1C

/* Enable/Disable/Defer A5x global preemption model */
#define CP_PREEMPT_ENABLE_GLOBAL    0x69

/* Enable/Disable A5x local preemption model */
#define CP_PREEMPT_ENABLE_LOCAL     0x6A

/* Yeild token on a5xx similar to CP_PREEMPT on a4xx */
#define CP_CONTEXT_SWITCH_YIELD     0x6B

/* Inform CP about current render mode (needed for a5xx preemption) */
#define CP_SET_RENDER_MODE          0x6C

/* copy sequencer instruction memory to system memory */
#define CP_IM_STORE            0x2c

/* test 2 memory locations to dword values specified */
#define CP_TEST_TWO_MEMS	0x71

/* Write register, ignoring context state for context sensitive registers */
#define CP_REG_WR_NO_CTXT  0x78

/*
 * for A4xx
 * Write to register with address that does not fit into type-0 pkt
 */
#define CP_WIDE_REG_WRITE           0x74


/* PFP waits until the FIFO between the PFP and the ME is empty */
#define CP_WAIT_FOR_ME		0x13

/* Record the real-time when this packet is processed by PFP */
#define CP_RECORD_PFP_TIMESTAMP	0x11

#define CP_SET_PROTECTED_MODE  0x5f /* sets the register protection mode */

/* Used to switch GPU between secure and non-secure modes */
#define CP_SET_SECURE_MODE 0x66

#define CP_BOOTSTRAP_UCODE  0x6f /* bootstraps microcode */

/*
 * for a3xx
 */

#define CP_LOAD_STATE 0x30 /* load high level sequencer command */

/* Conditionally load a IB based on a flag */
#define CP_COND_INDIRECT_BUFFER_PFE 0x3A /* prefetch enabled */
#define CP_COND_INDIRECT_BUFFER_PFD 0x32 /* prefetch disabled */

/* Load a buffer with pre-fetch enabled */
#define CP_INDIRECT_BUFFER_PFE 0x3F

#define CP_EXEC_CL 0x31

/* (A4x) save PM4 stream pointers to execute upon a visible draw */
#define CP_SET_DRAW_STATE 0x43

#define CP_LOADSTATE_DSTOFFSET_SHIFT 0x00000000
#define CP_LOADSTATE_STATESRC_SHIFT 0x00000010
#define CP_LOADSTATE_STATEBLOCKID_SHIFT 0x00000013
#define CP_LOADSTATE_NUMOFUNITS_SHIFT 0x00000016
#define CP_LOADSTATE_STATETYPE_SHIFT 0x00000000
#define CP_LOADSTATE_EXTSRCADDR_SHIFT 0x00000002

static inline uint pm4_calc_odd_parity_bit(uint val)
{
	return (0x9669 >> (0xf & ((val) ^
	((val) >> 4) ^ ((val) >> 8) ^ ((val) >> 12) ^
	((val) >> 16) ^ ((val) >> 20) ^ ((val) >> 24) ^
	((val) >> 28)))) & 1;
}

/*
 * PM4 packet header functions
 * For all the packet functions the passed in count should be the size of the
 * payload excluding the header
 */
static inline uint cp_type0_packet(uint regindx, uint cnt)
{
	return CP_TYPE0_PKT | ((cnt-1) << 16) | ((regindx) & 0x7FFF);
}

static inline uint cp_type3_packet(uint opcode, uint cnt)
{
	return CP_TYPE3_PKT | ((cnt-1) << 16) | (((opcode) & 0xFF) << 8);
}

static inline uint cp_type4_packet(uint opcode, uint cnt)
{
	return CP_TYPE4_PKT | ((cnt) << 0) |
	(pm4_calc_odd_parity_bit(cnt) << 7) |
	(((opcode) & 0x3FFFF) << 8) |
	((pm4_calc_odd_parity_bit(opcode) << 27));
}

static inline uint cp_type7_packet(uint opcode, uint cnt)
{
	return CP_TYPE7_PKT | ((cnt) << 0) |
	(pm4_calc_odd_parity_bit(cnt) << 15) |
	(((opcode) & 0x7F) << 16) |
	((pm4_calc_odd_parity_bit(opcode) << 23));

}

#define pkt_is_type0(pkt) (((pkt) & 0XC0000000) == CP_TYPE0_PKT)

#define type0_pkt_size(pkt) ((((pkt) >> 16) & 0x3FFF) + 1)
#define type0_pkt_offset(pkt) ((pkt) & 0x7FFF)

/*
 * Check both for the type3 opcode and make sure that the reserved bits [1:7]
 * and 15 are 0
 */

#define pkt_is_type3(pkt) \
	((((pkt) & 0xC0000000) == CP_TYPE3_PKT) && \
	 (((pkt) & 0x80FE) == 0))

#define cp_type3_opcode(pkt) (((pkt) >> 8) & 0xFF)
#define type3_pkt_size(pkt) ((((pkt) >> 16) & 0x3FFF) + 1)

#define pkt_is_type4(pkt) \
	((((pkt) & 0xF0000000) == CP_TYPE4_PKT) && \
	 ((((pkt) >> 27) & 0x1) == \
	 pm4_calc_odd_parity_bit(cp_type4_base_index_one_reg_wr(pkt))) \
	 && ((((pkt) >> 7) & 0x1) == \
	 pm4_calc_odd_parity_bit(type4_pkt_size(pkt))))

#define cp_type4_base_index_one_reg_wr(pkt) (((pkt) >> 8) & 0x7FFFF)
#define type4_pkt_size(pkt) ((pkt) & 0x7F)

#define pkt_is_type7(pkt) \
	((((pkt) & 0xF0000000) == CP_TYPE7_PKT) && \
	 (((pkt) & 0x0F000000) == 0) && \
	 ((((pkt) >> 23) & 0x1) == \
	 pm4_calc_odd_parity_bit(cp_type7_opcode(pkt))) \
	 && ((((pkt) >> 15) & 0x1) == \
	 pm4_calc_odd_parity_bit(type7_pkt_size(pkt))))

#define cp_type7_opcode(pkt) (((pkt) >> 16) & 0x7F)
#define type7_pkt_size(pkt) ((pkt) & 0x3FFF)

/* dword base address of the GFX decode space */
#define SUBBLOCK_OFFSET(reg) ((unsigned int)((reg) - (0x2000)))

/* gmem command buffer length */
#define CP_REG(reg) ((0x4 << 16) | (SUBBLOCK_OFFSET(reg)))

// add these
#define ADRENO_GPUREV(x) 530
#define lower_32_bits(n) ((uint32_t)(n))
#define upper_32_bits(n) ((uint32_t)(((n) >> 16) >> 16))

/* Return true if the hardware uses the legacy (A4XX and older) PM4 format */
#define ADRENO_LEGACY_PM4(_d) (ADRENO_GPUREV(_d) < 500)

/**
 * cp_packet - Generic CP packet to support different opcodes on
 * different GPU cores.
 * @adreno_dev: The adreno device
 * @opcode: Operation for cp packet
 * @size: size for cp packet
 */
static inline uint cp_packet(struct adreno_device *adreno_dev,
				int opcode, uint size)
{
	if (ADRENO_LEGACY_PM4(adreno_dev))
		return cp_type3_packet(opcode, size);

	return cp_type7_packet(opcode, size);
}

/**
 * cp_mem_packet - Generic CP memory packet to support different
 * opcodes on different GPU cores.
 * @adreno_dev: The adreno device
 * @opcode: mem operation for cp packet
 * @size: size for cp packet
 * @num_mem: num of mem access
 */
static inline uint cp_mem_packet(struct adreno_device *adreno_dev,
				int opcode, uint size, uint num_mem)
{
	if (ADRENO_LEGACY_PM4(adreno_dev))
		return cp_type3_packet(opcode, size);

	return cp_type7_packet(opcode, size + num_mem);
}

/* Return 1 if the command is an indirect buffer of any kind */
static inline int adreno_cmd_is_ib(struct adreno_device *adreno_dev,
					unsigned int cmd)
{
	return cmd == cp_mem_packet(adreno_dev,
			CP_INDIRECT_BUFFER_PFE, 2, 1) ||
		cmd == cp_mem_packet(adreno_dev,
			CP_INDIRECT_BUFFER_PFD, 2, 1) ||
		cmd == cp_mem_packet(adreno_dev,
			CP_COND_INDIRECT_BUFFER_PFE, 2, 1) ||
		cmd == cp_mem_packet(adreno_dev,
			CP_COND_INDIRECT_BUFFER_PFD, 2, 1);
}

/**
 * cp_gpuaddr - Generic function to add 64bit and 32bit gpuaddr
 * to pm4 commands
 * @adreno_dev: The adreno device
 * @cmds: command pointer to add gpuaddr
 * @gpuaddr: gpuaddr to add
 */
static inline uint cp_gpuaddr(struct adreno_device *adreno_dev,
		   uint *cmds, uint64_t gpuaddr)
{
	uint *start = cmds;

	if (ADRENO_LEGACY_PM4(adreno_dev))
		*cmds++ = (uint)gpuaddr;
	else {
		*cmds++ = lower_32_bits(gpuaddr);
		*cmds++ = upper_32_bits(gpuaddr);
	}
	return cmds - start;
}

/**
 * cp_register - Generic function for gpu register operation
 * @adreno_dev: The adreno device
 * @reg: GPU register
 * @size: count for PM4 operation
 */
static inline uint cp_register(struct adreno_device *adreno_dev,
			unsigned int reg, unsigned int size)
{
	if (ADRENO_LEGACY_PM4(adreno_dev))
		return cp_type0_packet(reg, size);

	return cp_type4_packet(reg, size);
}

/**
 * cp_wait_for_me - common function for WAIT_FOR_ME
 * @adreno_dev: The adreno device
 * @cmds: command pointer to add gpuaddr
 */
static inline uint cp_wait_for_me(struct adreno_device *adreno_dev,
				uint *cmds)
{
	uint *start = cmds;

	if (ADRENO_LEGACY_PM4(adreno_dev)) {
		*cmds++ = cp_type3_packet(CP_WAIT_FOR_ME, 1);
		*cmds++ = 0;
	} else
		*cmds++ = cp_type7_packet(CP_WAIT_FOR_ME, 0);

	return cmds - start;
}

/**
 * cp_wait_for_idle - common function for WAIT_FOR_IDLE
 * @adreno_dev: The adreno device
 * @cmds: command pointer to add gpuaddr
 */
static inline uint cp_wait_for_idle(struct adreno_device *adreno_dev,
				uint *cmds)
{
	uint *start = cmds;

	if (ADRENO_LEGACY_PM4(adreno_dev)) {
		*cmds++ = cp_type3_packet(CP_WAIT_FOR_IDLE, 1);
		*cmds++ = 0;
	} else
		*cmds++ = cp_type7_packet(CP_WAIT_FOR_IDLE, 0);

	return cmds - start;
}

/**
 * cp_invalidate_state - common function for invalidating cp
 * state
 * @adreno_dev: The adreno device
 * @cmds: command pointer to add gpuaddr
 */
static inline uint cp_invalidate_state(struct adreno_device *adreno_dev,
				uint *cmds)
{
	uint *start = cmds;

	if (ADRENO_GPUREV(adreno_dev) < 500) {
		*cmds++ = cp_type3_packet(CP_INVALIDATE_STATE, 1);
		*cmds++ = 0x7fff;
	} else {
		*cmds++ = cp_type7_packet(CP_SET_DRAW_STATE, 3);
		*cmds++ = 0x40000;
		*cmds++ = 0;
		*cmds++ = 0;
	}

	return cmds - start;
}

#endif	/* __ADRENO_PM4TYPES_H */
