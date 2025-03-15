/*
 * Copyright (c) 2013 Rob Clark <robdclark@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef IR3_H_
#define IR3_H_

#include <stdint.h>
#include <stdbool.h>

#include "shader_enums.h"
#include "util/list.h"

#include "util/bitscan.h"
/*#include "util/list.h"
#include "util/set.h"
#include "util/u_debug.h"*/

#define debug_assert(x) assert(x)

#include "instr-a3xx.h"

/* low level intermediate representation of an adreno shader program */

struct ir3_compiler;
struct ir3;
struct ir3_instruction;
struct ir3_block;

struct ir3_info {
	uint32_t gpu_id;
	uint16_t sizedwords;
	uint16_t instrs_count;   /* expanded to account for rpt's */
	uint16_t nops_count;     /* # of nop instructions, including nopN */
	uint16_t mov_count;
	uint16_t cov_count;
	/* NOTE: max_reg, etc, does not include registers not touched
	 * by the shader (ie. vertex fetched via VFD_DECODE but not
	 * touched by shader)
	 */
	int8_t   max_reg;   /* highest GPR # used by shader */
	int8_t   max_half_reg;
	int16_t  max_const;

	/* number of sync bits: */
	uint16_t ss, sy;

	/* estimate of number of cycles stalled on (ss) */
	uint16_t sstall;

	uint16_t last_baryf;     /* instruction # of last varying fetch */
};

struct ir3_register {
	enum {
		IR3_REG_CONST  = 0x001,
		IR3_REG_IMMED  = 0x002,
		IR3_REG_HALF   = 0x004,
		/* high registers are used for some things in compute shaders,
		 * for example.  Seems to be for things that are global to all
		 * threads in a wave, so possibly these are global/shared by
		 * all the threads in the wave?
		 */
		IR3_REG_HIGH   = 0x008,
		IR3_REG_RELATIV= 0x010,
		IR3_REG_R      = 0x020,
		/* Most instructions, it seems, can do float abs/neg but not
		 * integer.  The CP pass needs to know what is intended (int or
		 * float) in order to do the right thing.  For this reason the
		 * abs/neg flags are split out into float and int variants.  In
		 * addition, .b (bitwise) operations, the negate is actually a
		 * bitwise not, so split that out into a new flag to make it
		 * more clear.
		 */
		IR3_REG_FNEG   = 0x040,
		IR3_REG_FABS   = 0x080,
		IR3_REG_SNEG   = 0x100,
		IR3_REG_SABS   = 0x200,
		IR3_REG_BNOT   = 0x400,
		IR3_REG_EVEN   = 0x800,
		IR3_REG_POS_INF= 0x1000,
		/* (ei) flag, end-input?  Set on last bary, presumably to signal
		 * that the shader needs no more input:
		 */
		IR3_REG_EI     = 0x2000,
		/* meta-flags, for intermediate stages of IR, ie.
		 * before register assignment is done:
		 */
		IR3_REG_SSA    = 0x4000,   /* 'instr' is ptr to assigning instr */
		IR3_REG_ARRAY  = 0x8000,

	} flags;

	/* used for cat5 instructions, but also for internal/IR level
	 * tracking of what registers are read/written by an instruction.
	 * wrmask may be a bad name since it is used to represent both
	 * src and dst that touch multiple adjacent registers.
	 */
	unsigned wrmask : 16;  /* up to vec16 */

	/* for relative addressing, 32bits for array size is too small,
	 * but otoh we don't need to deal with disjoint sets, so instead
	 * use a simple size field (number of scalar components).
	 *
	 * Note the size field isn't important for relative const (since
	 * we don't have to do register allocation for constants).
	 */
	unsigned size : 15;

	bool merged : 1;    /* half-regs conflict with full regs (ie >= a6xx) */

	/* normal registers:
	 * the component is in the low two bits of the reg #, so
	 * rN.x becomes: (N << 2) | x
	 */
	uint16_t num;
	union {
		/* immediate: */
		int32_t  iim_val;
		uint32_t uim_val;
		float    fim_val;
		/* relative: */
		struct {
			uint16_t id;
			int16_t offset;
		} array;
	};

	/* For IR3_REG_SSA, src registers contain ptr back to assigning
	 * instruction.
	 *
	 * For IR3_REG_ARRAY, the pointer is back to the last dependent
	 * array access (although the net effect is the same, it points
	 * back to a previous instruction that we depend on).
	 */
	struct ir3_instruction *instr;
};

/*
 * Stupid/simple growable array implementation:
 */
#define DECLARE_ARRAY(type, name) \
	unsigned name ## _count, name ## _sz; \
	type * name;

#define array_insert(ctx, arr, val) do { \
		if (arr ## _count == arr ## _sz) { \
			arr ## _sz = MAX2(2 * arr ## _sz, 16); \
			arr = reralloc_size(ctx, arr, arr ## _sz * sizeof(arr[0])); \
		} \
		arr[arr ##_count++] = val; \
	} while (0)

struct ir3_instruction {
	struct ir3_block *block;
	opc_t opc;
	enum {
		/* (sy) flag is set on first instruction, and after sample
		 * instructions (probably just on RAW hazard).
		 */
		IR3_INSTR_SY    = 0x001,
		/* (ss) flag is set on first instruction, and first instruction
		 * to depend on the result of "long" instructions (RAW hazard):
		 *
		 *   rcp, rsq, log2, exp2, sin, cos, sqrt
		 *
		 * It seems to synchronize until all in-flight instructions are
		 * completed, for example:
		 *
		 *   rsq hr1.w, hr1.w
		 *   add.f hr2.z, (neg)hr2.z, hc0.y
		 *   mul.f hr2.w, (neg)hr2.y, (neg)hr2.y
		 *   rsq hr2.x, hr2.x
		 *   (rpt1)nop
		 *   mad.f16 hr2.w, hr2.z, hr2.z, hr2.w
		 *   nop
		 *   mad.f16 hr2.w, (neg)hr0.w, (neg)hr0.w, hr2.w
		 *   (ss)(rpt2)mul.f hr1.x, (r)hr1.x, hr1.w
		 *   (rpt2)mul.f hr0.x, (neg)(r)hr0.x, hr2.x
		 *
		 * The last mul.f does not have (ss) set, presumably because the
		 * (ss) on the previous instruction does the job.
		 *
		 * The blob driver also seems to set it on WAR hazards, although
		 * not really clear if this is needed or just blob compiler being
		 * sloppy.  So far I haven't found a case where removing the (ss)
		 * causes problems for WAR hazard, but I could just be getting
		 * lucky:
		 *
		 *   rcp r1.y, r3.y
		 *   (ss)(rpt2)mad.f32 r3.y, (r)c9.x, r1.x, (r)r3.z
		 *
		 */
		IR3_INSTR_SS    = 0x002,
		/* (jp) flag is set on jump targets:
		 */
		IR3_INSTR_JP    = 0x004,
		IR3_INSTR_UL    = 0x008,
		IR3_INSTR_3D    = 0x010,
		IR3_INSTR_A     = 0x020,
		IR3_INSTR_O     = 0x040,
		IR3_INSTR_P     = 0x080,
		IR3_INSTR_S     = 0x100,
		IR3_INSTR_S2EN  = 0x200,
		IR3_INSTR_G     = 0x400,
		IR3_INSTR_SAT   = 0x800,
		/* (cat5/cat6) Bindless */
		IR3_INSTR_B     = 0x1000,
		/* (cat5-only) Get some parts of the encoding from a1.x */
		IR3_INSTR_A1EN  = 0x2000,
		/* meta-flags, for intermediate stages of IR, ie.
		 * before register assignment is done:
		 */
		IR3_INSTR_MARK  = 0x4000,
		IR3_INSTR_UNUSED= 0x8000,
	} flags;
	uint8_t repeat;
	uint8_t nop;
#ifdef DEBUG
	unsigned regs_max;
#endif
	unsigned regs_count;
	struct ir3_register **regs;
	union {
		struct {
			char inv;
			char comp;
			int  immed;
			struct ir3_block *target;
		} cat0;
		struct {
			type_t src_type, dst_type;
		} cat1;
		struct {
			enum {
				IR3_COND_LT = 0,
				IR3_COND_LE = 1,
				IR3_COND_GT = 2,
				IR3_COND_GE = 3,
				IR3_COND_EQ = 4,
				IR3_COND_NE = 5,
			} condition;
		} cat2;
		struct {
			unsigned samp, tex;
			unsigned tex_base : 3;
			type_t type;
		} cat5;
		struct {
			type_t type;
			int src_offset;
			int dst_offset;
			int iim_val : 3;      /* for ldgb/stgb, # of components */
			unsigned d : 3;       /* for ldc, component offset */
			bool typed : 1;
			unsigned base : 3;
		} cat6;
		struct {
			unsigned w : 1;       /* write */
			unsigned r : 1;       /* read */
			unsigned l : 1;       /* local */
			unsigned g : 1;       /* global */
		} cat7;
		/* for meta-instructions, just used to hold extra data
		 * before instruction scheduling, etc
		 */
		struct {
			int off;              /* component/offset */
		} split;
		struct {
			/* for output collects, this maps back to the entry in the
			 * ir3_shader_variant::outputs table.
			 */
			int outidx;
		} collect;
		struct {
			unsigned samp, tex;
			unsigned input_offset;
			unsigned samp_base : 3;
			unsigned tex_base : 3;
		} prefetch;
		struct {
			/* maps back to entry in ir3_shader_variant::inputs table: */
			int inidx;
			/* for sysvals, identifies the sysval type.  Mostly so we can
			 * identify the special cases where a sysval should not be DCE'd
			 * (currently, just pre-fs texture fetch)
			 */
			gl_system_value sysval;
		} input;
	};

	/* When we get to the RA stage, we need instruction's position/name: */
	uint16_t ip;
	uint16_t name;

	/* used for per-pass extra instruction data.
	 *
	 * TODO we should remove the per-pass data like this and 'use_count'
	 * and do something similar to what RA does w/ ir3_ra_instr_data..
	 * ie. use the ir3_count_instructions pass, and then use instr->ip
	 * to index into a table of pass-private data.
	 */
	void *data;

	/**
	 * Valid if pass calls ir3_find_ssa_uses().. see foreach_ssa_use()
	 */
	struct set *uses;

	int sun;            /* Sethi–Ullman number, used by sched */
	int use_count;      /* currently just updated/used by cp */

	/* Used during CP and RA stages.  For collect and shader inputs/
	 * outputs where we need a sequence of consecutive registers,
	 * keep track of each src instructions left (ie 'n-1') and right
	 * (ie 'n+1') neighbor.  The front-end must insert enough mov's
	 * to ensure that each instruction has at most one left and at
	 * most one right neighbor.  During the copy-propagation pass,
	 * we only remove mov's when we can preserve this constraint.
	 * And during the RA stage, we use the neighbor information to
	 * allocate a block of registers in one shot.
	 *
	 * TODO: maybe just add something like:
	 *   struct ir3_instruction_ref {
	 *       struct ir3_instruction *instr;
	 *       unsigned cnt;
	 *   }
	 *
	 * Or can we get away without the refcnt stuff?  It seems like
	 * it should be overkill..  the problem is if, potentially after
	 * already eliminating some mov's, if you have a single mov that
	 * needs to be grouped with it's neighbors in two different
	 * places (ex. shader output and a collect).
	 */
	struct {
		struct ir3_instruction *left, *right;
		uint16_t left_cnt, right_cnt;
	} cp;

	/* an instruction can reference at most one address register amongst
	 * it's src/dst registers.  Beyond that, you need to insert mov's.
	 *
	 * NOTE: do not write this directly, use ir3_instr_set_address()
	 */
	struct ir3_instruction *address;

	/* Tracking for additional dependent instructions.  Used to handle
	 * barriers, WAR hazards for arrays/SSBOs/etc.
	 */
	DECLARE_ARRAY(struct ir3_instruction *, deps);

	/*
	 * From PoV of instruction scheduling, not execution (ie. ignores global/
	 * local distinction):
	 *                            shared  image  atomic  SSBO  everything
	 *   barrier()/            -   R/W     R/W    R/W     R/W       X
	 *     groupMemoryBarrier()
	 *   memoryBarrier()       -           R/W    R/W
	 *     (but only images declared coherent?)
	 *   memoryBarrierAtomic() -                  R/W
	 *   memoryBarrierBuffer() -                          R/W
	 *   memoryBarrierImage()  -           R/W
	 *   memoryBarrierShared() -   R/W
	 *
	 * TODO I think for SSBO/image/shared, in cases where we can determine
	 * which variable is accessed, we don't need to care about accesses to
	 * different variables (unless declared coherent??)
	 */
	enum {
		IR3_BARRIER_EVERYTHING = 1 << 0,
		IR3_BARRIER_SHARED_R   = 1 << 1,
		IR3_BARRIER_SHARED_W   = 1 << 2,
		IR3_BARRIER_IMAGE_R    = 1 << 3,
		IR3_BARRIER_IMAGE_W    = 1 << 4,
		IR3_BARRIER_BUFFER_R   = 1 << 5,
		IR3_BARRIER_BUFFER_W   = 1 << 6,
		IR3_BARRIER_ARRAY_R    = 1 << 7,
		IR3_BARRIER_ARRAY_W    = 1 << 8,
	} barrier_class, barrier_conflict;

	/* Entry in ir3_block's instruction list: */
	struct list_head node;

#ifdef DEBUG
	uint32_t serialno;
#endif

	// TODO only computerator/assembler:
	int line;
};

static inline struct ir3_instruction *
ir3_neighbor_first(struct ir3_instruction *instr)
{
	int cnt = 0;
	while (instr->cp.left) {
		instr = instr->cp.left;
		if (++cnt > 0xffff) {
			debug_assert(0);
			break;
		}
	}
	return instr;
}

static inline int ir3_neighbor_count(struct ir3_instruction *instr)
{
	int num = 1;

	debug_assert(!instr->cp.left);

	while (instr->cp.right) {
		num++;
		instr = instr->cp.right;
		if (num > 0xffff) {
			debug_assert(0);
			break;
		}
	}

	return num;
}

struct ir3 {
	struct ir3_compiler *compiler;
	gl_shader_stage type;

	DECLARE_ARRAY(struct ir3_instruction *, inputs);
	DECLARE_ARRAY(struct ir3_instruction *, outputs);

	/* Track bary.f (and ldlv) instructions.. this is needed in
	 * scheduling to ensure that all varying fetches happen before
	 * any potential kill instructions.  The hw gets grumpy if all
	 * threads in a group are killed before the last bary.f gets
	 * a chance to signal end of input (ei).
	 */
	DECLARE_ARRAY(struct ir3_instruction *, baryfs);

	/* Track all indirect instructions (read and write).  To avoid
	 * deadlock scenario where an address register gets scheduled,
	 * but other dependent src instructions cannot be scheduled due
	 * to dependency on a *different* address register value, the
	 * scheduler needs to ensure that all dependencies other than
	 * the instruction other than the address register are scheduled
	 * before the one that writes the address register.  Having a
	 * convenient list of instructions that reference some address
	 * register simplifies this.
	 */
	DECLARE_ARRAY(struct ir3_instruction *, a0_users);

	/* same for a1.x: */
	DECLARE_ARRAY(struct ir3_instruction *, a1_users);

	/* and same for instructions that consume predicate register: */
	DECLARE_ARRAY(struct ir3_instruction *, predicates);

	/* Track texture sample instructions which need texture state
	 * patched in (for astc-srgb workaround):
	 */
	DECLARE_ARRAY(struct ir3_instruction *, astc_srgb);

	/* List of blocks: */
	struct list_head block_list;

	/* List of ir3_array's: */
	struct list_head array_list;

	unsigned max_sun;   /* max Sethi–Ullman number */

#ifdef DEBUG
	unsigned block_count, instr_count;
#endif
};

struct ir3_array {
	struct list_head node;
	unsigned length;
	unsigned id;

	struct nir_register *r;

	/* To avoid array write's from getting DCE'd, keep track of the
	 * most recent write.  Any array access depends on the most
	 * recent write.  This way, nothing depends on writes after the
	 * last read.  But all the writes that happen before that have
	 * something depending on them
	 */
	struct ir3_instruction *last_write;

	/* extra stuff used in RA pass: */
	unsigned base;      /* base vreg name */
	unsigned reg;       /* base physical reg */
	uint16_t start_ip, end_ip;

	/* Indicates if half-precision */
	bool half;
};

struct ir3_array * ir3_lookup_array(struct ir3 *ir, unsigned id);

struct ir3_block {
	struct list_head node;
	struct ir3 *shader;

	const struct nir_block *nblock;

	struct list_head instr_list;  /* list of ir3_instruction */

	/* each block has either one or two successors.. in case of
	 * two successors, 'condition' decides which one to follow.
	 * A block preceding an if/else has two successors.
	 */
	struct ir3_instruction *condition;
	struct ir3_block *successors[2];

	struct set *predecessors;     /* set of ir3_block */

	uint16_t start_ip, end_ip;

	/* Track instructions which do not write a register but other-
	 * wise must not be discarded (such as kill, stg, etc)
	 */
	DECLARE_ARRAY(struct ir3_instruction *, keeps);

	/* used for per-pass extra block data.  Mainly used right
	 * now in RA step to track livein/liveout.
	 */
	void *data;

#ifdef DEBUG
	uint32_t serialno;
#endif
};

static inline uint32_t
block_id(struct ir3_block *block)
{
#ifdef DEBUG
	return block->serialno;
#else
	return (uint32_t)(unsigned long)block;
#endif
}

struct ir3 * ir3_create(struct ir3_compiler *compiler, gl_shader_stage type);
void ir3_destroy(struct ir3 *shader);
void * ir3_assemble(struct ir3 *shader,
		struct ir3_info *info, uint32_t gpu_id);
void * ir3_alloc(struct ir3 *shader, int sz);

struct ir3_block * ir3_block_create(struct ir3 *shader);

struct ir3_instruction * ir3_instr_create(struct ir3_block *block, opc_t opc);
struct ir3_instruction * ir3_instr_create2(struct ir3_block *block,
		opc_t opc, int nreg);
struct ir3_instruction * ir3_instr_clone(struct ir3_instruction *instr);
void ir3_instr_add_dep(struct ir3_instruction *instr, struct ir3_instruction *dep);
const char *ir3_instr_name(struct ir3_instruction *instr);

struct ir3_register * ir3_reg_create(struct ir3_instruction *instr,
		int num, int flags);
struct ir3_register * ir3_reg_clone(struct ir3 *shader,
		struct ir3_register *reg);

void ir3_instr_set_address(struct ir3_instruction *instr,
		struct ir3_instruction *addr);

static inline bool ir3_instr_check_mark(struct ir3_instruction *instr)
{
	if (instr->flags & IR3_INSTR_MARK)
		return true;  /* already visited */
	instr->flags |= IR3_INSTR_MARK;
	return false;
}

void ir3_block_clear_mark(struct ir3_block *block);
void ir3_clear_mark(struct ir3 *shader);

unsigned ir3_count_instructions(struct ir3 *ir);
unsigned ir3_count_instructions_ra(struct ir3 *ir);

void ir3_find_ssa_uses(struct ir3 *ir, void *mem_ctx, bool falsedeps);

//#include "util/set.h"
#define foreach_ssa_use(__use, __instr) \
	for (struct ir3_instruction *__use = (void *)~0; \
	     __use && (__instr)->uses; __use = NULL) \
		set_foreach ((__instr)->uses, __entry) \
			if ((__use = (void *)__entry->key))

#define MAX_ARRAYS 16

/* comp:
 *   0 - x
 *   1 - y
 *   2 - z
 *   3 - w
 */
static inline uint32_t regid(int num, int comp)
{
	return (num << 2) | (comp & 0x3);
}

static inline uint32_t reg_num(struct ir3_register *reg)
{
	return reg->num >> 2;
}

static inline uint32_t reg_comp(struct ir3_register *reg)
{
	return reg->num & 0x3;
}

#define INVALID_REG      regid(63, 0)
#define VALIDREG(r)      ((r) != INVALID_REG)
#define CONDREG(r, val)  COND(VALIDREG(r), (val))

static inline bool is_flow(struct ir3_instruction *instr)
{
	return (opc_cat(instr->opc) == 0);
}

static inline bool is_kill(struct ir3_instruction *instr)
{
	return instr->opc == OPC_KILL;
}

static inline bool is_nop(struct ir3_instruction *instr)
{
	return instr->opc == OPC_NOP;
}

static inline bool is_same_type_reg(struct ir3_register *reg1,
		struct ir3_register *reg2)
{
	unsigned type_reg1 = (reg1->flags & (IR3_REG_HIGH | IR3_REG_HALF));
	unsigned type_reg2 = (reg2->flags & (IR3_REG_HIGH | IR3_REG_HALF));

	if (type_reg1 ^ type_reg2)
		return false;
	else
		return true;
}

/* Is it a non-transformative (ie. not type changing) mov?  This can
 * also include absneg.s/absneg.f, which for the most part can be
 * treated as a mov (single src argument).
 */
static inline bool is_same_type_mov(struct ir3_instruction *instr)
{
	struct ir3_register *dst;

	switch (instr->opc) {
	case OPC_MOV:
		if (instr->cat1.src_type != instr->cat1.dst_type)
			return false;
		/* If the type of dest reg and src reg are different,
		 * it shouldn't be considered as same type mov
		 */
		if (!is_same_type_reg(instr->regs[0], instr->regs[1]))
			return false;
		break;
	case OPC_ABSNEG_F:
	case OPC_ABSNEG_S:
		if (instr->flags & IR3_INSTR_SAT)
			return false;
		/* If the type of dest reg and src reg are different,
		 * it shouldn't be considered as same type mov
		 */
		if (!is_same_type_reg(instr->regs[0], instr->regs[1]))
			return false;
		break;
	default:
		return false;
	}

	dst = instr->regs[0];

	/* mov's that write to a0 or p0.x are special: */
	if (dst->num == regid(REG_P0, 0))
		return false;
	if (reg_num(dst) == REG_A0)
		return false;

	if (dst->flags & (IR3_REG_RELATIV | IR3_REG_ARRAY))
		return false;

	return true;
}

/* A move from const, which changes size but not type, can also be
 * folded into dest instruction in some cases.
 */
static inline bool is_const_mov(struct ir3_instruction *instr)
{
	if (instr->opc != OPC_MOV)
		return false;

	if (!(instr->regs[1]->flags & IR3_REG_CONST))
		return false;

	type_t src_type = instr->cat1.src_type;
	type_t dst_type = instr->cat1.dst_type;

	return (type_float(src_type) && type_float(dst_type)) ||
		(type_uint(src_type) && type_uint(dst_type)) ||
		(type_sint(src_type) && type_sint(dst_type));
}

static inline bool is_alu(struct ir3_instruction *instr)
{
	return (1 <= opc_cat(instr->opc)) && (opc_cat(instr->opc) <= 3);
}

static inline bool is_sfu(struct ir3_instruction *instr)
{
	return (opc_cat(instr->opc) == 4);
}

static inline bool is_tex(struct ir3_instruction *instr)
{
	return (opc_cat(instr->opc) == 5);
}

static inline bool is_tex_or_prefetch(struct ir3_instruction *instr)
{
	return is_tex(instr) || (instr->opc == OPC_META_TEX_PREFETCH);
}

static inline bool is_mem(struct ir3_instruction *instr)
{
	return (opc_cat(instr->opc) == 6);
}

static inline bool is_barrier(struct ir3_instruction *instr)
{
	return (opc_cat(instr->opc) == 7);
}

static inline bool
is_half(struct ir3_instruction *instr)
{
	return !!(instr->regs[0]->flags & IR3_REG_HALF);
}

static inline bool
is_high(struct ir3_instruction *instr)
{
	return !!(instr->regs[0]->flags & IR3_REG_HIGH);
}

static inline bool
is_store(struct ir3_instruction *instr)
{
	/* these instructions, the "destination" register is
	 * actually a source, the address to store to.
	 */
	switch (instr->opc) {
	case OPC_STG:
	case OPC_STGB:
	case OPC_STIB:
	case OPC_STP:
	case OPC_STL:
	case OPC_STLW:
	case OPC_L2G:
	case OPC_G2L:
		return true;
	default:
		return false;
	}
}

static inline bool is_load(struct ir3_instruction *instr)
{
	switch (instr->opc) {
	case OPC_LDG:
	case OPC_LDGB:
	case OPC_LDIB:
	case OPC_LDL:
	case OPC_LDP:
	case OPC_L2G:
	case OPC_LDLW:
	case OPC_LDC:
	case OPC_LDLV:
		/* probably some others too.. */
		return true;
	default:
		return false;
	}
}

static inline bool is_input(struct ir3_instruction *instr)
{
	/* in some cases, ldlv is used to fetch varying without
	 * interpolation.. fortunately inloc is the first src
	 * register in either case
	 */
	switch (instr->opc) {
	case OPC_LDLV:
	case OPC_BARY_F:
		return true;
	default:
		return false;
	}
}

static inline bool is_bool(struct ir3_instruction *instr)
{
	switch (instr->opc) {
	case OPC_CMPS_F:
	case OPC_CMPS_S:
	case OPC_CMPS_U:
		return true;
	default:
		return false;
	}
}

static inline bool is_meta(struct ir3_instruction *instr)
{
	return (opc_cat(instr->opc) == -1);
}

static inline unsigned dest_regs(struct ir3_instruction *instr)
{
	if ((instr->regs_count == 0) || is_store(instr) || is_flow(instr))
		return 0;

	return util_last_bit(instr->regs[0]->wrmask);
}

static inline bool
writes_gpr(struct ir3_instruction *instr)
{
	if (dest_regs(instr) == 0)
		return false;
	/* is dest a normal temp register: */
	struct ir3_register *reg = instr->regs[0];
	debug_assert(!(reg->flags & (IR3_REG_CONST | IR3_REG_IMMED)));
	if ((reg_num(reg) == REG_A0) ||
			(reg->num == regid(REG_P0, 0)))
		return false;
	return true;
}

static inline bool writes_addr0(struct ir3_instruction *instr)
{
	if (instr->regs_count > 0) {
		struct ir3_register *dst = instr->regs[0];
		return dst->num == regid(REG_A0, 0);
	}
	return false;
}

static inline bool writes_addr1(struct ir3_instruction *instr)
{
	if (instr->regs_count > 0) {
		struct ir3_register *dst = instr->regs[0];
		return dst->num == regid(REG_A0, 1);
	}
	return false;
}

static inline bool writes_pred(struct ir3_instruction *instr)
{
	if (instr->regs_count > 0) {
		struct ir3_register *dst = instr->regs[0];
		return reg_num(dst) == REG_P0;
	}
	return false;
}

/* returns defining instruction for reg */
/* TODO better name */
static inline struct ir3_instruction *ssa(struct ir3_register *reg)
{
	if (reg->flags & (IR3_REG_SSA | IR3_REG_ARRAY)) {
		return reg->instr;
	}
	return NULL;
}

static inline bool conflicts(struct ir3_instruction *a,
		struct ir3_instruction *b)
{
	return (a && b) && (a != b);
}

static inline bool reg_gpr(struct ir3_register *r)
{
	if (r->flags & (IR3_REG_CONST | IR3_REG_IMMED))
		return false;
	if ((reg_num(r) == REG_A0) || (reg_num(r) == REG_P0))
		return false;
	return true;
}

static inline type_t half_type(type_t type)
{
	switch (type) {
	case TYPE_F32: return TYPE_F16;
	case TYPE_U32: return TYPE_U16;
	case TYPE_S32: return TYPE_S16;
	case TYPE_F16:
	case TYPE_U16:
	case TYPE_S16:
		return type;
	default:
		assert(0);
		return ~0;
	}
}

/* some cat2 instructions (ie. those which are not float) can embed an
 * immediate:
 */
static inline bool ir3_cat2_int(opc_t opc)
{
	switch (opc) {
	case OPC_ADD_U:
	case OPC_ADD_S:
	case OPC_SUB_U:
	case OPC_SUB_S:
	case OPC_CMPS_U:
	case OPC_CMPS_S:
	case OPC_MIN_U:
	case OPC_MIN_S:
	case OPC_MAX_U:
	case OPC_MAX_S:
	case OPC_CMPV_U:
	case OPC_CMPV_S:
	case OPC_MUL_U24:
	case OPC_MUL_S24:
	case OPC_MULL_U:
	case OPC_CLZ_S:
	case OPC_ABSNEG_S:
	case OPC_AND_B:
	case OPC_OR_B:
	case OPC_NOT_B:
	case OPC_XOR_B:
	case OPC_BFREV_B:
	case OPC_CLZ_B:
	case OPC_SHL_B:
	case OPC_SHR_B:
	case OPC_ASHR_B:
	case OPC_MGEN_B:
	case OPC_GETBIT_B:
	case OPC_CBITS_B:
	case OPC_BARY_F:
		return true;

	default:
		return false;
	}
}

/* map cat2 instruction to valid abs/neg flags: */
static inline unsigned ir3_cat2_absneg(opc_t opc)
{
	switch (opc) {
	case OPC_ADD_F:
	case OPC_MIN_F:
	case OPC_MAX_F:
	case OPC_MUL_F:
	case OPC_SIGN_F:
	case OPC_CMPS_F:
	case OPC_ABSNEG_F:
	case OPC_CMPV_F:
	case OPC_FLOOR_F:
	case OPC_CEIL_F:
	case OPC_RNDNE_F:
	case OPC_RNDAZ_F:
	case OPC_TRUNC_F:
	case OPC_BARY_F:
		return IR3_REG_FABS | IR3_REG_FNEG;

	case OPC_ADD_U:
	case OPC_ADD_S:
	case OPC_SUB_U:
	case OPC_SUB_S:
	case OPC_CMPS_U:
	case OPC_CMPS_S:
	case OPC_MIN_U:
	case OPC_MIN_S:
	case OPC_MAX_U:
	case OPC_MAX_S:
	case OPC_CMPV_U:
	case OPC_CMPV_S:
	case OPC_MUL_U24:
	case OPC_MUL_S24:
	case OPC_MULL_U:
	case OPC_CLZ_S:
		return 0;

	case OPC_ABSNEG_S:
		return IR3_REG_SABS | IR3_REG_SNEG;

	case OPC_AND_B:
	case OPC_OR_B:
	case OPC_NOT_B:
	case OPC_XOR_B:
	case OPC_BFREV_B:
	case OPC_CLZ_B:
	case OPC_SHL_B:
	case OPC_SHR_B:
	case OPC_ASHR_B:
	case OPC_MGEN_B:
	case OPC_GETBIT_B:
	case OPC_CBITS_B:
		return IR3_REG_BNOT;

	default:
		return 0;
	}
}

/* map cat3 instructions to valid abs/neg flags: */
static inline unsigned ir3_cat3_absneg(opc_t opc)
{
	switch (opc) {
	case OPC_MAD_F16:
	case OPC_MAD_F32:
	case OPC_SEL_F16:
	case OPC_SEL_F32:
		return IR3_REG_FNEG;

	case OPC_MAD_U16:
	case OPC_MADSH_U16:
	case OPC_MAD_S16:
	case OPC_MADSH_M16:
	case OPC_MAD_U24:
	case OPC_MAD_S24:
	case OPC_SEL_S16:
	case OPC_SEL_S32:
	case OPC_SAD_S16:
	case OPC_SAD_S32:
		/* neg *may* work on 3rd src.. */

	case OPC_SEL_B16:
	case OPC_SEL_B32:

	default:
		return 0;
	}
}

#define MASK(n) ((1 << (n)) - 1)

/* iterator for an instructions's sources (reg), also returns src #: */
#define foreach_src_n(__srcreg, __n, __instr) \
	if ((__instr)->regs_count) \
		for (unsigned __cnt = (__instr)->regs_count - 1, __n = 0; __n < __cnt; __n++) \
			if ((__srcreg = (__instr)->regs[__n + 1]))

/* iterator for an instructions's sources (reg): */
#define foreach_src(__srcreg, __instr) \
	foreach_src_n(__srcreg, __i, __instr)

static inline unsigned __ssa_src_cnt(struct ir3_instruction *instr)
{
	unsigned cnt = instr->regs_count + instr->deps_count;
	if (instr->address)
		cnt++;
	return cnt;
}

static inline struct ir3_instruction **
__ssa_srcp_n(struct ir3_instruction *instr, unsigned n)
{
	if (n == (instr->regs_count + instr->deps_count))
		return &instr->address;
	if (n >= instr->regs_count)
		return &instr->deps[n - instr->regs_count];
	if (ssa(instr->regs[n]))
		return &instr->regs[n]->instr;
	return NULL;
}

static inline bool __is_false_dep(struct ir3_instruction *instr, unsigned n)
{
	if (n == (instr->regs_count + instr->deps_count))
		return false;
	if (n >= instr->regs_count)
		return true;
	return false;
}

#define foreach_ssa_srcp_n(__srcp, __n, __instr) \
	for (struct ir3_instruction **__srcp = (void *)~0; __srcp; __srcp = NULL) \
		for (unsigned __cnt = __ssa_src_cnt(__instr), __n = 0; __n < __cnt; __n++) \
			if ((__srcp = __ssa_srcp_n(__instr, __n)))

#define foreach_ssa_srcp(__srcp, __instr) \
	foreach_ssa_srcp_n(__srcp, __i, __instr)

/* iterator for an instruction's SSA sources (instr), also returns src #: */
#define foreach_ssa_src_n(__srcinst, __n, __instr) \
	foreach_ssa_srcp_n(__srcp, __n, __instr) \
		if ((__srcinst = *__srcp))

/* iterator for an instruction's SSA sources (instr): */
#define foreach_ssa_src(__srcinst, __instr) \
	foreach_ssa_src_n(__srcinst, __i, __instr)

/* iterators for shader inputs: */
#define foreach_input_n(__ininstr, __cnt, __ir) \
	for (unsigned __cnt = 0; __cnt < (__ir)->inputs_count; __cnt++) \
		if ((__ininstr = (__ir)->inputs[__cnt]))
#define foreach_input(__ininstr, __ir) \
	foreach_input_n(__ininstr, __i, __ir)

/* iterators for shader outputs: */
#define foreach_output_n(__outinstr, __cnt, __ir) \
	for (unsigned __cnt = 0; __cnt < (__ir)->outputs_count; __cnt++) \
		if ((__outinstr = (__ir)->outputs[__cnt]))
#define foreach_output(__outinstr, __ir) \
	foreach_output_n(__outinstr, __i, __ir)

/* iterators for instructions: */
#define foreach_instr(__instr, __list) \
	list_for_each_entry(struct ir3_instruction, __instr, __list, node)
#define foreach_instr_rev(__instr, __list) \
	list_for_each_entry_rev(struct ir3_instruction, __instr, __list, node)
#define foreach_instr_safe(__instr, __list) \
	list_for_each_entry_safe(struct ir3_instruction, __instr, __list, node)

/* iterators for blocks: */
#define foreach_block(__block, __list) \
	list_for_each_entry(struct ir3_block, __block, __list, node)
#define foreach_block_safe(__block, __list) \
	list_for_each_entry_safe(struct ir3_block, __block, __list, node)
#define foreach_block_rev(__block, __list) \
	list_for_each_entry_rev(struct ir3_block, __block, __list, node)

/* iterators for arrays: */
#define foreach_array(__array, __list) \
	list_for_each_entry(struct ir3_array, __array, __list, node)

/* Check if condition is true for any src instruction.
 */
static inline bool
check_src_cond(struct ir3_instruction *instr, bool (*cond)(struct ir3_instruction *))
{
	struct ir3_register *reg;

	/* Note that this is also used post-RA so skip the ssa iterator: */
	foreach_src (reg, instr) {
		struct ir3_instruction *src = reg->instr;

		if (!src)
			continue;

		/* meta:split/collect aren't real instructions, the thing that
		 * we actually care about is *their* srcs
		 */
		if ((src->opc == OPC_META_SPLIT) || (src->opc == OPC_META_COLLECT)) {
			if (check_src_cond(src, cond))
				return true;
		} else {
			if (cond(src))
				return true;
		}
	}

	return false;
}

/* dump: */
void ir3_print(struct ir3 *ir);
void ir3_print_instr(struct ir3_instruction *instr);

/* delay calculation: */
int ir3_delayslots(struct ir3_instruction *assigner,
		struct ir3_instruction *consumer, unsigned n, bool soft);
unsigned ir3_delay_calc(struct ir3_block *block, struct ir3_instruction *instr,
		bool soft, bool pred);
void ir3_remove_nops(struct ir3 *ir);

/* dead code elimination: */
struct ir3_shader_variant;
void ir3_dce(struct ir3 *ir, struct ir3_shader_variant *so);

/* fp16 conversion folding */
void ir3_cf(struct ir3 *ir);

/* copy-propagate: */
void ir3_cp(struct ir3 *ir, struct ir3_shader_variant *so);

/* group neighbors and insert mov's to resolve conflicts: */
void ir3_group(struct ir3 *ir);

/* Sethi–Ullman numbering: */
void ir3_sun(struct ir3 *ir);

/* scheduling: */
void ir3_sched_add_deps(struct ir3 *ir);
int ir3_sched(struct ir3 *ir);

struct ir3_context;
int ir3_postsched(struct ir3_context *ctx);

bool ir3_a6xx_fixup_atomic_dests(struct ir3 *ir, struct ir3_shader_variant *so);

/* register assignment: */
struct ir3_ra_reg_set * ir3_ra_alloc_reg_set(struct ir3_compiler *compiler);
int ir3_ra(struct ir3_shader_variant *v, struct ir3_instruction **precolor, unsigned nprecolor);

/* legalize: */
void ir3_legalize(struct ir3 *ir, struct ir3_shader_variant *so, int *max_bary);

static inline bool
ir3_has_latency_to_hide(struct ir3 *ir)
{
	/* VS/GS/TCS/TESS  co-exist with frag shader invocations, but we don't
	 * know the nature of the fragment shader.  Just assume it will have
	 * latency to hide:
	 */
	if (ir->type != MESA_SHADER_FRAGMENT)
		return true;

	foreach_block (block, &ir->block_list) {
		foreach_instr (instr, &block->instr_list) {
			if (is_tex_or_prefetch(instr))
				return true;

			if (is_load(instr)) {
				switch (instr->opc) {
				case OPC_LDLV:
				case OPC_LDL:
				case OPC_LDLW:
					break;
				default:
					return true;
				}
			}
		}
	}

	return false;
}

/* ************************************************************************* */
/* instruction helpers */

/* creates SSA src of correct type (ie. half vs full precision) */
static inline struct ir3_register * __ssa_src(struct ir3_instruction *instr,
		struct ir3_instruction *src, unsigned flags)
{
	struct ir3_register *reg;
	if (src->regs[0]->flags & IR3_REG_HALF)
		flags |= IR3_REG_HALF;
	reg = ir3_reg_create(instr, 0, IR3_REG_SSA | flags);
	reg->instr = src;
	reg->wrmask = src->regs[0]->wrmask;
	return reg;
}

static inline struct ir3_register * __ssa_dst(struct ir3_instruction *instr)
{
	struct ir3_register *reg = ir3_reg_create(instr, 0, 0);
	reg->flags |= IR3_REG_SSA;
	return reg;
}

static inline struct ir3_instruction *
create_immed_typed(struct ir3_block *block, uint32_t val, type_t type)
{
	struct ir3_instruction *mov;
	unsigned flags = (type_size(type) < 32) ? IR3_REG_HALF : 0;

	mov = ir3_instr_create(block, OPC_MOV);
	mov->cat1.src_type = type;
	mov->cat1.dst_type = type;
	__ssa_dst(mov)->flags |= flags;
	ir3_reg_create(mov, 0, IR3_REG_IMMED | flags)->uim_val = val;

	return mov;
}

static inline struct ir3_instruction *
create_immed(struct ir3_block *block, uint32_t val)
{
	return create_immed_typed(block, val, TYPE_U32);
}

static inline struct ir3_instruction *
create_uniform_typed(struct ir3_block *block, unsigned n, type_t type)
{
	struct ir3_instruction *mov;
	unsigned flags = (type_size(type) < 32) ? IR3_REG_HALF : 0;

	mov = ir3_instr_create(block, OPC_MOV);
	mov->cat1.src_type = type;
	mov->cat1.dst_type = type;
	__ssa_dst(mov)->flags |= flags;
	ir3_reg_create(mov, n, IR3_REG_CONST | flags);

	return mov;
}

static inline struct ir3_instruction *
create_uniform(struct ir3_block *block, unsigned n)
{
	return create_uniform_typed(block, n, TYPE_F32);
}

static inline struct ir3_instruction *
create_uniform_indirect(struct ir3_block *block, int n,
		struct ir3_instruction *address)
{
	struct ir3_instruction *mov;

	mov = ir3_instr_create(block, OPC_MOV);
	mov->cat1.src_type = TYPE_U32;
	mov->cat1.dst_type = TYPE_U32;
	__ssa_dst(mov);
	ir3_reg_create(mov, 0, IR3_REG_CONST | IR3_REG_RELATIV)->array.offset = n;

	ir3_instr_set_address(mov, address);

	return mov;
}

static inline struct ir3_instruction *
ir3_MOV(struct ir3_block *block, struct ir3_instruction *src, type_t type)
{
	struct ir3_instruction *instr = ir3_instr_create(block, OPC_MOV);
	__ssa_dst(instr);
	if (src->regs[0]->flags & IR3_REG_ARRAY) {
		struct ir3_register *src_reg = __ssa_src(instr, src, IR3_REG_ARRAY);
		src_reg->array = src->regs[0]->array;
	} else {
		__ssa_src(instr, src, src->regs[0]->flags & IR3_REG_HIGH);
	}
	debug_assert(!(src->regs[0]->flags & IR3_REG_RELATIV));
	instr->cat1.src_type = type;
	instr->cat1.dst_type = type;
	return instr;
}

static inline struct ir3_instruction *
ir3_COV(struct ir3_block *block, struct ir3_instruction *src,
		type_t src_type, type_t dst_type)
{
	struct ir3_instruction *instr = ir3_instr_create(block, OPC_MOV);
	unsigned dst_flags = (type_size(dst_type) < 32) ? IR3_REG_HALF : 0;
	unsigned src_flags = (type_size(src_type) < 32) ? IR3_REG_HALF : 0;

	debug_assert((src->regs[0]->flags & IR3_REG_HALF) == src_flags);

	__ssa_dst(instr)->flags |= dst_flags;
	__ssa_src(instr, src, 0);
	instr->cat1.src_type = src_type;
	instr->cat1.dst_type = dst_type;
	debug_assert(!(src->regs[0]->flags & IR3_REG_ARRAY));
	return instr;
}

static inline struct ir3_instruction *
ir3_NOP(struct ir3_block *block)
{
	return ir3_instr_create(block, OPC_NOP);
}

#define IR3_INSTR_0 0

#define __INSTR0(flag, name, opc)                                        \
static inline struct ir3_instruction *                                   \
ir3_##name(struct ir3_block *block)                                      \
{                                                                        \
	struct ir3_instruction *instr =                                      \
		ir3_instr_create(block, opc);                                    \
	instr->flags |= flag;                                                \
	return instr;                                                        \
}
#define INSTR0F(f, name)    __INSTR0(IR3_INSTR_##f, name##_##f, OPC_##name)
#define INSTR0(name)        __INSTR0(0, name, OPC_##name)

#define __INSTR1(flag, name, opc)                                        \
static inline struct ir3_instruction *                                   \
ir3_##name(struct ir3_block *block,                                      \
		struct ir3_instruction *a, unsigned aflags)                      \
{                                                                        \
	struct ir3_instruction *instr =                                      \
		ir3_instr_create(block, opc);                                    \
	__ssa_dst(instr);                                                    \
	__ssa_src(instr, a, aflags);                                         \
	instr->flags |= flag;                                                \
	return instr;                                                        \
}
#define INSTR1F(f, name)    __INSTR1(IR3_INSTR_##f, name##_##f, OPC_##name)
#define INSTR1(name)        __INSTR1(0, name, OPC_##name)

#define __INSTR2(flag, name, opc)                                        \
static inline struct ir3_instruction *                                   \
ir3_##name(struct ir3_block *block,                                      \
		struct ir3_instruction *a, unsigned aflags,                      \
		struct ir3_instruction *b, unsigned bflags)                      \
{                                                                        \
	struct ir3_instruction *instr =                                      \
		ir3_instr_create(block, opc);                                    \
	__ssa_dst(instr);                                                    \
	__ssa_src(instr, a, aflags);                                         \
	__ssa_src(instr, b, bflags);                                         \
	instr->flags |= flag;                                                \
	return instr;                                                        \
}
#define INSTR2F(f, name)    __INSTR2(IR3_INSTR_##f, name##_##f, OPC_##name)
#define INSTR2(name)        __INSTR2(0, name, OPC_##name)

#define __INSTR3(flag, name, opc)                                        \
static inline struct ir3_instruction *                                   \
ir3_##name(struct ir3_block *block,                                      \
		struct ir3_instruction *a, unsigned aflags,                      \
		struct ir3_instruction *b, unsigned bflags,                      \
		struct ir3_instruction *c, unsigned cflags)                      \
{                                                                        \
	struct ir3_instruction *instr =                                      \
		ir3_instr_create2(block, opc, 4);                                \
	__ssa_dst(instr);                                                    \
	__ssa_src(instr, a, aflags);                                         \
	__ssa_src(instr, b, bflags);                                         \
	__ssa_src(instr, c, cflags);                                         \
	instr->flags |= flag;                                                \
	return instr;                                                        \
}
#define INSTR3F(f, name)    __INSTR3(IR3_INSTR_##f, name##_##f, OPC_##name)
#define INSTR3(name)        __INSTR3(0, name, OPC_##name)

#define __INSTR4(flag, name, opc)                                        \
static inline struct ir3_instruction *                                   \
ir3_##name(struct ir3_block *block,                                      \
		struct ir3_instruction *a, unsigned aflags,                      \
		struct ir3_instruction *b, unsigned bflags,                      \
		struct ir3_instruction *c, unsigned cflags,                      \
		struct ir3_instruction *d, unsigned dflags)                      \
{                                                                        \
	struct ir3_instruction *instr =                                      \
		ir3_instr_create2(block, opc, 5);                                \
	__ssa_dst(instr);                                                    \
	__ssa_src(instr, a, aflags);                                         \
	__ssa_src(instr, b, bflags);                                         \
	__ssa_src(instr, c, cflags);                                         \
	__ssa_src(instr, d, dflags);                                         \
	instr->flags |= flag;                                                \
	return instr;                                                        \
}
#define INSTR4F(f, name)    __INSTR4(IR3_INSTR_##f, name##_##f, OPC_##name)
#define INSTR4(name)        __INSTR4(0, name, OPC_##name)

/* cat0 instructions: */
INSTR1(B)
INSTR0(JUMP)
INSTR1(KILL)
INSTR0(END)
INSTR0(CHSH)
INSTR0(CHMASK)
INSTR1(PREDT)
INSTR0(PREDF)
INSTR0(PREDE)

/* cat2 instructions, most 2 src but some 1 src: */
INSTR2(ADD_F)
INSTR2(MIN_F)
INSTR2(MAX_F)
INSTR2(MUL_F)
INSTR1(SIGN_F)
INSTR2(CMPS_F)
INSTR1(ABSNEG_F)
INSTR2(CMPV_F)
INSTR1(FLOOR_F)
INSTR1(CEIL_F)
INSTR1(RNDNE_F)
INSTR1(RNDAZ_F)
INSTR1(TRUNC_F)
INSTR2(ADD_U)
INSTR2(ADD_S)
INSTR2(SUB_U)
INSTR2(SUB_S)
INSTR2(CMPS_U)
INSTR2(CMPS_S)
INSTR2(MIN_U)
INSTR2(MIN_S)
INSTR2(MAX_U)
INSTR2(MAX_S)
INSTR1(ABSNEG_S)
INSTR2(AND_B)
INSTR2(OR_B)
INSTR1(NOT_B)
INSTR2(XOR_B)
INSTR2(CMPV_U)
INSTR2(CMPV_S)
INSTR2(MUL_U24)
INSTR2(MUL_S24)
INSTR2(MULL_U)
INSTR1(BFREV_B)
INSTR1(CLZ_S)
INSTR1(CLZ_B)
INSTR2(SHL_B)
INSTR2(SHR_B)
INSTR2(ASHR_B)
INSTR2(BARY_F)
INSTR2(MGEN_B)
INSTR2(GETBIT_B)
INSTR1(SETRM)
INSTR1(CBITS_B)
INSTR2(SHB)
INSTR2(MSAD)

/* cat3 instructions: */
INSTR3(MAD_U16)
INSTR3(MADSH_U16)
INSTR3(MAD_S16)
INSTR3(MADSH_M16)
INSTR3(MAD_U24)
INSTR3(MAD_S24)
INSTR3(MAD_F16)
INSTR3(MAD_F32)
/* NOTE: SEL_B32 checks for zero vs nonzero */
INSTR3(SEL_B16)
INSTR3(SEL_B32)
INSTR3(SEL_S16)
INSTR3(SEL_S32)
INSTR3(SEL_F16)
INSTR3(SEL_F32)
INSTR3(SAD_S16)
INSTR3(SAD_S32)

/* cat4 instructions: */
INSTR1(RCP)
INSTR1(RSQ)
INSTR1(HRSQ)
INSTR1(LOG2)
INSTR1(HLOG2)
INSTR1(EXP2)
INSTR1(HEXP2)
INSTR1(SIN)
INSTR1(COS)
INSTR1(SQRT)

/* cat5 instructions: */
INSTR1(DSX)
INSTR1(DSXPP_1)
INSTR1(DSY)
INSTR1(DSYPP_1)
INSTR1F(3D, DSX)
INSTR1F(3D, DSY)
INSTR1(RGETPOS)

static inline struct ir3_instruction *
ir3_SAM(struct ir3_block *block, opc_t opc, type_t type,
		unsigned wrmask, unsigned flags, struct ir3_instruction *samp_tex,
		struct ir3_instruction *src0, struct ir3_instruction *src1)
{
	struct ir3_instruction *sam;

	sam = ir3_instr_create(block, opc);
	sam->flags |= flags;
	__ssa_dst(sam)->wrmask = wrmask;
	if (flags & IR3_INSTR_S2EN) {
		__ssa_src(sam, samp_tex, IR3_REG_HALF);
	}
	if (src0) {
		__ssa_src(sam, src0, 0);
	}
	if (src1) {
		__ssa_src(sam, src1, 0);
	}
	sam->cat5.type  = type;

	return sam;
}

/* cat6 instructions: */
INSTR2(LDLV)
INSTR3(LDG)
INSTR3(LDL)
INSTR3(LDLW)
INSTR3(STG)
INSTR3(STL)
INSTR3(STLW)
INSTR1(RESINFO)
INSTR1(RESFMT)
INSTR2(ATOMIC_ADD)
INSTR2(ATOMIC_SUB)
INSTR2(ATOMIC_XCHG)
INSTR2(ATOMIC_INC)
INSTR2(ATOMIC_DEC)
INSTR2(ATOMIC_CMPXCHG)
INSTR2(ATOMIC_MIN)
INSTR2(ATOMIC_MAX)
INSTR2(ATOMIC_AND)
INSTR2(ATOMIC_OR)
INSTR2(ATOMIC_XOR)
INSTR2(LDC)
#if GPU >= 600
INSTR3(STIB);
INSTR2(LDIB);
INSTR3F(G, ATOMIC_ADD)
INSTR3F(G, ATOMIC_SUB)
INSTR3F(G, ATOMIC_XCHG)
INSTR3F(G, ATOMIC_INC)
INSTR3F(G, ATOMIC_DEC)
INSTR3F(G, ATOMIC_CMPXCHG)
INSTR3F(G, ATOMIC_MIN)
INSTR3F(G, ATOMIC_MAX)
INSTR3F(G, ATOMIC_AND)
INSTR3F(G, ATOMIC_OR)
INSTR3F(G, ATOMIC_XOR)
#elif GPU >= 400
INSTR3(LDGB)
INSTR4(STGB)
INSTR4(STIB)
INSTR4F(G, ATOMIC_ADD)
INSTR4F(G, ATOMIC_SUB)
INSTR4F(G, ATOMIC_XCHG)
INSTR4F(G, ATOMIC_INC)
INSTR4F(G, ATOMIC_DEC)
INSTR4F(G, ATOMIC_CMPXCHG)
INSTR4F(G, ATOMIC_MIN)
INSTR4F(G, ATOMIC_MAX)
INSTR4F(G, ATOMIC_AND)
INSTR4F(G, ATOMIC_OR)
INSTR4F(G, ATOMIC_XOR)
#endif

INSTR4F(G, STG)

/* cat7 instructions: */
INSTR0(BAR)
INSTR0(FENCE)

/* meta instructions: */
INSTR0(META_TEX_PREFETCH);

/* ************************************************************************* */
/* split this out or find some helper to use.. like main/bitset.h.. */

#include <string.h>
#include "util/bitset.h"

#define MAX_REG 256

typedef BITSET_DECLARE(regmask_t, 2 * MAX_REG);

static inline bool
__regmask_get(regmask_t *regmask, struct ir3_register *reg, unsigned n)
{
	if (reg->merged) {
		/* a6xx+ case, with merged register file, we track things in terms
		 * of half-precision registers, with a full precisions register
		 * using two half-precision slots:
		 */
		if (reg->flags & IR3_REG_HALF) {
			return BITSET_TEST(*regmask, n);
		} else {
			n *= 2;
			return BITSET_TEST(*regmask, n) || BITSET_TEST(*regmask, n+1);
		}
	} else {
		/* pre a6xx case, with separate register file for half and full
		 * precision:
		 */
		if (reg->flags & IR3_REG_HALF)
			n += MAX_REG;
		return BITSET_TEST(*regmask, n);
	}
}

static inline void
__regmask_set(regmask_t *regmask, struct ir3_register *reg, unsigned n)
{
	if (reg->merged) {
		/* a6xx+ case, with merged register file, we track things in terms
		 * of half-precision registers, with a full precisions register
		 * using two half-precision slots:
		 */
		if (reg->flags & IR3_REG_HALF) {
			BITSET_SET(*regmask, n);
		} else {
			n *= 2;
			BITSET_SET(*regmask, n);
			BITSET_SET(*regmask, n+1);
		}
	} else {
		/* pre a6xx case, with separate register file for half and full
		 * precision:
		 */
		if (reg->flags & IR3_REG_HALF)
			n += MAX_REG;
		BITSET_SET(*regmask, n);
	}
}

static inline void regmask_init(regmask_t *regmask)
{
	memset(regmask, 0, sizeof(*regmask));
}

static inline void regmask_set(regmask_t *regmask, struct ir3_register *reg)
{
	if (reg->flags & IR3_REG_RELATIV) {
		for (unsigned i = 0; i < reg->size; i++)
			__regmask_set(regmask, reg, reg->array.offset + i);
	} else {
		for (unsigned mask = reg->wrmask, n = reg->num; mask; mask >>= 1, n++)
			if (mask & 1)
				__regmask_set(regmask, reg, n);
	}
}

static inline void regmask_or(regmask_t *dst, regmask_t *a, regmask_t *b)
{
	unsigned i;
	for (i = 0; i < ARRAY_SIZE(*dst); i++)
		(*dst)[i] = (*a)[i] | (*b)[i];
}

static inline bool regmask_get(regmask_t *regmask,
		struct ir3_register *reg)
{
	if (reg->flags & IR3_REG_RELATIV) {
		for (unsigned i = 0; i < reg->size; i++)
			if (__regmask_get(regmask, reg, reg->array.offset + i))
				return true;
	} else {
		for (unsigned mask = reg->wrmask, n = reg->num; mask; mask >>= 1, n++)
			if (mask & 1)
				if (__regmask_get(regmask, reg, n))
					return true;
	}
	return false;
}

/* ************************************************************************* */

#endif /* IR3_H_ */
