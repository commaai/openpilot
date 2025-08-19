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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

//#include <util/u_debug.h>

#include "util/macros.h"
#include "instr-a3xx.h"

/* bitmask of debug flags */
enum debug_t {
  PRINT_RAW      = 0x1,    /* dump raw hexdump */
  PRINT_VERBOSE  = 0x2,
  EXPAND_REPEAT  = 0x4,
};

static enum debug_t debug = PRINT_RAW | PRINT_VERBOSE | EXPAND_REPEAT;

static const char *levels[] = {
    "",
    "\t",
    "\t\t",
    "\t\t\t",
    "\t\t\t\t",
    "\t\t\t\t\t",
    "\t\t\t\t\t\t",
    "\t\t\t\t\t\t\t",
    "\t\t\t\t\t\t\t\t",
    "\t\t\t\t\t\t\t\t\t",
    "x",
    "x",
    "x",
    "x",
    "x",
    "x",
};

static const char *component = "xyzw";

static const char *type[] = {
    [TYPE_F16] = "f16",
    [TYPE_F32] = "f32",
    [TYPE_U16] = "u16",
    [TYPE_U32] = "u32",
    [TYPE_S16] = "s16",
    [TYPE_S32] = "s32",
    [TYPE_U8]  = "u8",
    [TYPE_S8]  = "s8",
};

struct disasm_ctx {
  FILE *out;
  int level;
  unsigned gpu_id;

  /* current instruction repeat flag: */
  unsigned repeat;
  /* current instruction repeat indx/offset (for --expand): */
  unsigned repeatidx;

  unsigned instructions;
};

static const char *float_imms[] = {
  "0.0",
  "0.5",
  "1.0",
  "2.0",
  "e",
  "pi",
  "1/pi",
  "1/log2(e)",
  "log2(e)",
  "1/log2(10)",
  "log2(10)",
  "4.0",
};

static void print_reg(struct disasm_ctx *ctx, reg_t reg, bool full,
    bool is_float, bool r,
    bool c, bool im, bool neg, bool abs, bool addr_rel)
{
  const char type = c ? 'c' : 'r';

  // XXX I prefer - and || for neg/abs, but preserving format used
  // by libllvm-a3xx for easy diffing..

  if (abs && neg)
    fprintf(ctx->out, "(absneg)");
  else if (neg)
    fprintf(ctx->out, "(neg)");
  else if (abs)
    fprintf(ctx->out, "(abs)");

  if (r)
    fprintf(ctx->out, "(r)");

  if (im) {
    if (is_float && full && reg.iim_val < ARRAY_SIZE(float_imms)) {
      fprintf(ctx->out, "(%s)", float_imms[reg.iim_val]);
    } else {
      fprintf(ctx->out, "%d", reg.iim_val);
    }
  } else if (addr_rel) {
    /* I would just use %+d but trying to make it diff'able with
     * libllvm-a3xx...
     */
    if (reg.iim_val < 0)
      fprintf(ctx->out, "%s%c<a0.x - %d>", full ? "" : "h", type, -reg.iim_val);
    else if (reg.iim_val > 0)
      fprintf(ctx->out, "%s%c<a0.x + %d>", full ? "" : "h", type, reg.iim_val);
    else
      fprintf(ctx->out, "%s%c<a0.x>", full ? "" : "h", type);
  } else if ((reg.num == REG_A0) && !c) {
    /* This matches libllvm output, the second (scalar) address register
     * seems to be called a1.x instead of a0.y.
     */
    fprintf(ctx->out, "a%d.x", reg.comp);
  } else if ((reg.num == REG_P0) && !c) {
    fprintf(ctx->out, "p0.%c", component[reg.comp]);
  } else {
    fprintf(ctx->out, "%s%c%d.%c", full ? "" : "h", type, reg.num, component[reg.comp]);
  }
}

static unsigned regidx(reg_t reg)
{
  return (4 * reg.num) + reg.comp;
}

static reg_t idxreg(unsigned idx)
{
  return (reg_t){
    .comp = idx & 0x3,
    .num  = idx >> 2,
  };
}

static void print_reg_dst(struct disasm_ctx *ctx, reg_t reg, bool full, bool addr_rel)
{
  reg = idxreg(regidx(reg) + ctx->repeatidx);
  print_reg(ctx, reg, full, false, false, false, false, false, false, addr_rel);
}

/* TODO switch to using reginfo struct everywhere, since more readable
 * than passing a bunch of bools to print_reg_src
 */

struct reginfo {
  reg_t reg;
  bool full;
  bool r;
  bool c;
  bool f; /* src reg is interpreted as float, used for printing immediates */
  bool im;
  bool neg;
  bool abs;
  bool addr_rel;
};

static void print_src(struct disasm_ctx *ctx, struct reginfo *info)
{
  reg_t reg = info->reg;

  if (info->r)
    reg = idxreg(regidx(info->reg) + ctx->repeatidx);

  print_reg(ctx, reg, info->full, info->f, info->r, info->c, info->im,
      info->neg, info->abs, info->addr_rel);
}

//static void print_dst(struct disasm_ctx *ctx, struct reginfo *info)
//{
//  print_reg_dst(ctx, info->reg, info->full, info->addr_rel);
//}

static void print_instr_cat0(struct disasm_ctx *ctx, instr_t *instr)
{
  static const struct {
    const char *suffix;
    int nsrc;
    bool idx;
  } brinfo[7] = {
    [BRANCH_PLAIN] = { "r",   1, false },
    [BRANCH_OR]    = { "rao", 2, false },
    [BRANCH_AND]   = { "raa", 2, false },
    [BRANCH_CONST] = { "rac", 0, true  },
    [BRANCH_ANY]   = { "any", 1, false },
    [BRANCH_ALL]   = { "all", 1, false },
    [BRANCH_X]     = { "rax", 0, false },
  };
  instr_cat0_t *cat0 = &instr->cat0;

  switch (instr_opc(instr, ctx->gpu_id)) {
  case OPC_KILL:
  case OPC_PREDT:
  case OPC_PREDF:
    fprintf(ctx->out, " %sp0.%c", cat0->inv0 ? "!" : "",
        component[cat0->comp0]);
    break;
  case OPC_B:
    fprintf(ctx->out, "%s", brinfo[cat0->brtype].suffix);
    if (brinfo[cat0->brtype].idx) {
      fprintf(ctx->out, ".%u", cat0->idx);
    }
    if (brinfo[cat0->brtype].nsrc >= 1) {
      fprintf(ctx->out, " %sp0.%c,", cat0->inv0 ? "!" : "",
          component[cat0->comp0]);
    }
    if (brinfo[cat0->brtype].nsrc >= 2) {
      fprintf(ctx->out, " %sp0.%c,", cat0->inv1 ? "!" : "",
          component[cat0->comp1]);
    }
    fprintf(ctx->out, " #%d", cat0->a3xx.immed);
    break;
  case OPC_JUMP:
  case OPC_CALL:
  case OPC_BKT:
  case OPC_GETONE:
  case OPC_SHPS:
    fprintf(ctx->out, " #%d", cat0->a3xx.immed);
    break;
  }

  if ((debug & PRINT_VERBOSE) && (cat0->dummy3|cat0->dummy4))
    fprintf(ctx->out, "\t{0: %x,%x}", cat0->dummy3, cat0->dummy4);
}

static void print_instr_cat1(struct disasm_ctx *ctx, instr_t *instr)
{
  instr_cat1_t *cat1 = &instr->cat1;

  if (cat1->ul)
    fprintf(ctx->out, "(ul)");

  if (cat1->src_type == cat1->dst_type) {
    if ((cat1->src_type == TYPE_S16) && (((reg_t)cat1->dst).num == REG_A0)) {
      /* special case (nmemonic?): */
      fprintf(ctx->out, "mova");
    } else {
      fprintf(ctx->out, "mov.%s%s", type[cat1->src_type], type[cat1->dst_type]);
    }
  } else {
    fprintf(ctx->out, "cov.%s%s", type[cat1->src_type], type[cat1->dst_type]);
  }

  fprintf(ctx->out, " ");

  if (cat1->even)
    fprintf(ctx->out, "(even)");

  if (cat1->pos_inf)
    fprintf(ctx->out, "(pos_infinity)");

  print_reg_dst(ctx, (reg_t)(cat1->dst), type_size(cat1->dst_type) == 32,
      cat1->dst_rel);

  fprintf(ctx->out, ", ");

  /* ugg, have to special case this.. vs print_reg().. */
  if (cat1->src_im) {
    if (type_float(cat1->src_type))
      fprintf(ctx->out, "(%f)", cat1->fim_val);
    else if (type_uint(cat1->src_type))
      fprintf(ctx->out, "0x%08x", cat1->uim_val);
    else
      fprintf(ctx->out, "%d", cat1->iim_val);
  } else if (cat1->src_rel && !cat1->src_c) {
    /* I would just use %+d but trying to make it diff'able with
     * libllvm-a3xx...
     */
    char type = cat1->src_rel_c ? 'c' : 'r';
    const char *full = (type_size(cat1->src_type) == 32) ? "" : "h";
    if (cat1->off < 0)
      fprintf(ctx->out, "%s%c<a0.x - %d>", full, type, -cat1->off);
    else if (cat1->off > 0)
      fprintf(ctx->out, "%s%c<a0.x + %d>", full, type, cat1->off);
    else
      fprintf(ctx->out, "%s%c<a0.x>", full, type);
  } else {
    struct reginfo src = {
      .reg = (reg_t)cat1->src,
      .full = type_size(cat1->src_type) == 32,
      .r = cat1->src_r,
      .c = cat1->src_c,
      .im = cat1->src_im,
    };
    print_src(ctx, &src);
  }

  if ((debug & PRINT_VERBOSE) && (cat1->must_be_0))
    fprintf(ctx->out, "\t{1: %x}", cat1->must_be_0);
}

static void print_instr_cat2(struct disasm_ctx *ctx, instr_t *instr)
{
  instr_cat2_t *cat2 = &instr->cat2;
  int opc = _OPC(2, cat2->opc);
  static const char *cond[] = {
      "lt",
      "le",
      "gt",
      "ge",
      "eq",
      "ne",
      "?6?",
  };

  switch (opc) {
  case OPC_CMPS_F:
  case OPC_CMPS_U:
  case OPC_CMPS_S:
  case OPC_CMPV_F:
  case OPC_CMPV_U:
  case OPC_CMPV_S:
    fprintf(ctx->out, ".%s", cond[cat2->cond]);
    break;
  }

  fprintf(ctx->out, " ");
  if (cat2->ei)
    fprintf(ctx->out, "(ei)");
  print_reg_dst(ctx, (reg_t)(cat2->dst), cat2->full ^ cat2->dst_half, false);
  fprintf(ctx->out, ", ");

  struct reginfo src1 = {
    .full = cat2->full,
    .r = cat2->repeat ? cat2->src1_r : 0,
    .f = is_cat2_float(opc),
    .im = cat2->src1_im,
    .abs = cat2->src1_abs,
    .neg = cat2->src1_neg,
  };

  if (cat2->c1.src1_c) {
    src1.reg = (reg_t)(cat2->c1.src1);
    src1.c = true;
  } else if (cat2->rel1.src1_rel) {
    src1.reg = (reg_t)(cat2->rel1.src1);
    src1.c = cat2->rel1.src1_c;
    src1.addr_rel = true;
  } else {
    src1.reg = (reg_t)(cat2->src1);
  }
  print_src(ctx, &src1);

  struct reginfo src2 = {
    .r = cat2->repeat ? cat2->src2_r : 0,
    .full = cat2->full,
    .f = is_cat2_float(opc),
    .abs = cat2->src2_abs,
    .neg = cat2->src2_neg,
    .im = cat2->src2_im,
  };
  switch (opc) {
  case OPC_ABSNEG_F:
  case OPC_ABSNEG_S:
  case OPC_CLZ_B:
  case OPC_CLZ_S:
  case OPC_SIGN_F:
  case OPC_FLOOR_F:
  case OPC_CEIL_F:
  case OPC_RNDNE_F:
  case OPC_RNDAZ_F:
  case OPC_TRUNC_F:
  case OPC_NOT_B:
  case OPC_BFREV_B:
  case OPC_SETRM:
  case OPC_CBITS_B:
    /* these only have one src reg */
    break;
  default:
    fprintf(ctx->out, ", ");
    if (cat2->c2.src2_c) {
      src2.reg = (reg_t)(cat2->c2.src2);
      src2.c = true;
    } else if (cat2->rel2.src2_rel) {
      src2.reg = (reg_t)(cat2->rel2.src2);
      src2.c = cat2->rel2.src2_c;
      src2.addr_rel = true;
    } else {
      src2.reg = (reg_t)(cat2->src2);
    }
    print_src(ctx, &src2);
    break;
  }
}

static void print_instr_cat3(struct disasm_ctx *ctx, instr_t *instr)
{
  instr_cat3_t *cat3 = &instr->cat3;
  bool full = instr_cat3_full(cat3);

  fprintf(ctx->out, " ");
  print_reg_dst(ctx, (reg_t)(cat3->dst), full ^ cat3->dst_half, false);
  fprintf(ctx->out, ", ");

  struct reginfo src1 = {
    .r = cat3->repeat ? cat3->src1_r : 0,
    .full = full,
    .neg = cat3->src1_neg,
  };
  if (cat3->c1.src1_c) {
    src1.reg = (reg_t)(cat3->c1.src1);
    src1.c = true;
  } else if (cat3->rel1.src1_rel) {
    src1.reg = (reg_t)(cat3->rel1.src1);
    src1.c = cat3->rel1.src1_c;
    src1.addr_rel = true;
  } else {
    src1.reg = (reg_t)(cat3->src1);
  }
  print_src(ctx, &src1);

  fprintf(ctx->out, ", ");
  struct reginfo src2 = {
    .reg = (reg_t)cat3->src2,
    .full = full,
    .r = cat3->repeat ? cat3->src2_r : 0,
    .c = cat3->src2_c,
    .neg = cat3->src2_neg,
  };
  print_src(ctx, &src2);

  fprintf(ctx->out, ", ");
  struct reginfo src3 = {
    .r = cat3->src3_r,
    .full = full,
    .neg = cat3->src3_neg,
  };
  if (cat3->c2.src3_c) {
    src3.reg = (reg_t)(cat3->c2.src3);
    src3.c = true;
  } else if (cat3->rel2.src3_rel) {
    src3.reg = (reg_t)(cat3->rel2.src3);
    src3.c = cat3->rel2.src3_c;
    src3.addr_rel = true;
  } else {
    src3.reg = (reg_t)(cat3->src3);
  }
  print_src(ctx, &src3);
}

static void print_instr_cat4(struct disasm_ctx *ctx, instr_t *instr)
{
  instr_cat4_t *cat4 = &instr->cat4;

  fprintf(ctx->out, " ");
  print_reg_dst(ctx, (reg_t)(cat4->dst), cat4->full ^ cat4->dst_half, false);
  fprintf(ctx->out, ", ");

  struct reginfo src = {
    .r = cat4->src_r,
    .im = cat4->src_im,
    .full = cat4->full,
    .neg = cat4->src_neg,
    .abs = cat4->src_abs,
  };
  if (cat4->c.src_c) {
    src.reg = (reg_t)(cat4->c.src);
    src.c = true;
  } else if (cat4->rel.src_rel) {
    src.reg = (reg_t)(cat4->rel.src);
    src.c = cat4->rel.src_c;
    src.addr_rel = true;
  } else {
    src.reg = (reg_t)(cat4->src);
  }
  print_src(ctx, &src);

  if ((debug & PRINT_VERBOSE) && (cat4->dummy1|cat4->dummy2))
    fprintf(ctx->out, "\t{4: %x,%x}", cat4->dummy1, cat4->dummy2);
}

static void print_instr_cat5(struct disasm_ctx *ctx, instr_t *instr)
{
  static const struct {
    bool src1, src2, samp, tex;
  } info[0x1f] = {
      [opc_op(OPC_ISAM)]     = { true,  false, true,  true,  },
      [opc_op(OPC_ISAML)]    = { true,  true,  true,  true,  },
      [opc_op(OPC_ISAMM)]    = { true,  false, true,  true,  },
      [opc_op(OPC_SAM)]      = { true,  false, true,  true,  },
      [opc_op(OPC_SAMB)]     = { true,  true,  true,  true,  },
      [opc_op(OPC_SAML)]     = { true,  true,  true,  true,  },
      [opc_op(OPC_SAMGQ)]    = { true,  false, true,  true,  },
      [opc_op(OPC_GETLOD)]   = { true,  false, true,  true,  },
      [opc_op(OPC_CONV)]     = { true,  true,  true,  true,  },
      [opc_op(OPC_CONVM)]    = { true,  true,  true,  true,  },
      [opc_op(OPC_GETSIZE)]  = { true,  false, false, true,  },
      [opc_op(OPC_GETBUF)]   = { false, false, false, true,  },
      [opc_op(OPC_GETPOS)]   = { true,  false, false, true,  },
      [opc_op(OPC_GETINFO)]  = { false, false, false, true,  },
      [opc_op(OPC_DSX)]      = { true,  false, false, false, },
      [opc_op(OPC_DSY)]      = { true,  false, false, false, },
      [opc_op(OPC_GATHER4R)] = { true,  false, true,  true,  },
      [opc_op(OPC_GATHER4G)] = { true,  false, true,  true,  },
      [opc_op(OPC_GATHER4B)] = { true,  false, true,  true,  },
      [opc_op(OPC_GATHER4A)] = { true,  false, true,  true,  },
      [opc_op(OPC_SAMGP0)]   = { true,  false, true,  true,  },
      [opc_op(OPC_SAMGP1)]   = { true,  false, true,  true,  },
      [opc_op(OPC_SAMGP2)]   = { true,  false, true,  true,  },
      [opc_op(OPC_SAMGP3)]   = { true,  false, true,  true,  },
      [opc_op(OPC_DSXPP_1)]  = { true,  false, false, false, },
      [opc_op(OPC_DSYPP_1)]  = { true,  false, false, false, },
      [opc_op(OPC_RGETPOS)]  = { true,  false, false, false, },
      [opc_op(OPC_RGETINFO)] = { false, false, false, false, },
  };

  static const struct {
    bool indirect;
    bool bindless;
    bool use_a1;
    bool uniform;
  } desc_features[8] = {
    [CAT5_NONUNIFORM] = { .indirect = true, },
    [CAT5_UNIFORM] = { .indirect = true, .uniform = true, },
    [CAT5_BINDLESS_IMM] = { .bindless = true, },
    [CAT5_BINDLESS_UNIFORM] = {
      .bindless = true,
      .indirect = true,
      .uniform = true,
    },
    [CAT5_BINDLESS_NONUNIFORM] = {
      .bindless = true,
      .indirect = true,
    },
    [CAT5_BINDLESS_A1_IMM] = {
      .bindless = true,
      .use_a1 = true,
    },
    [CAT5_BINDLESS_A1_UNIFORM] = {
      .bindless = true,
      .indirect = true,
      .uniform = true,
      .use_a1 = true,
    },
    [CAT5_BINDLESS_A1_NONUNIFORM] = {
      .bindless = true,
      .indirect = true,
      .use_a1 = true,
    },
  };

  instr_cat5_t *cat5 = &instr->cat5;
  int i;

  bool desc_indirect =
    cat5->is_s2en_bindless &&
    desc_features[cat5->s2en_bindless.desc_mode].indirect;
  bool bindless =
    cat5->is_s2en_bindless &&
    desc_features[cat5->s2en_bindless.desc_mode].bindless;
  bool use_a1 =
    cat5->is_s2en_bindless &&
    desc_features[cat5->s2en_bindless.desc_mode].use_a1;
  bool uniform =
    cat5->is_s2en_bindless &&
    desc_features[cat5->s2en_bindless.desc_mode].uniform;

  if (cat5->is_3d)   fprintf(ctx->out, ".3d");
  if (cat5->is_a)    fprintf(ctx->out, ".a");
  if (cat5->is_o)    fprintf(ctx->out, ".o");
  if (cat5->is_p)    fprintf(ctx->out, ".p");
  if (cat5->is_s)    fprintf(ctx->out, ".s");
  if (desc_indirect) fprintf(ctx->out, ".s2en");
  if (uniform)       fprintf(ctx->out, ".uniform");

  if (bindless) {
    unsigned base = (cat5->s2en_bindless.base_hi << 1) | cat5->base_lo;
    fprintf(ctx->out, ".base%d", base);
  }

  fprintf(ctx->out, " ");

  switch (_OPC(5, cat5->opc)) {
  case OPC_DSXPP_1:
  case OPC_DSYPP_1:
    break;
  default:
    fprintf(ctx->out, "(%s)", type[cat5->type]);
    break;
  }

  fprintf(ctx->out, "(");
  for (i = 0; i < 4; i++)
    if (cat5->wrmask & (1 << i))
      fprintf(ctx->out, "%c", "xyzw"[i]);
  fprintf(ctx->out, ")");

  print_reg_dst(ctx, (reg_t)(cat5->dst), type_size(cat5->type) == 32, false);

  if (info[cat5->opc].src1) {
    fprintf(ctx->out, ", ");
    struct reginfo src = { .reg = (reg_t)(cat5->src1), .full = cat5->full };
    print_src(ctx, &src);
  }

  if (cat5->is_o || info[cat5->opc].src2) {
    fprintf(ctx->out, ", ");
    struct reginfo src = { .reg = (reg_t)(cat5->src2), .full = cat5->full };
    print_src(ctx, &src);
  }
  if (cat5->is_s2en_bindless) {
    if (!desc_indirect) {
      if (info[cat5->opc].samp) {
        if (use_a1)
          fprintf(ctx->out, ", s#%d", cat5->s2en_bindless.src3);
        else
          fprintf(ctx->out, ", s#%d", cat5->s2en_bindless.src3 & 0xf);
      }

      if (info[cat5->opc].tex && !use_a1) {
        fprintf(ctx->out, ", t#%d", cat5->s2en_bindless.src3 >> 4);
      }
    }
  } else {
    if (info[cat5->opc].samp)
      fprintf(ctx->out, ", s#%d", cat5->norm.samp);
    if (info[cat5->opc].tex)
      fprintf(ctx->out, ", t#%d", cat5->norm.tex);
  }

  if (desc_indirect) {
    fprintf(ctx->out, ", ");
    struct reginfo src = { .reg = (reg_t)(cat5->s2en_bindless.src3), .full = bindless };
    print_src(ctx, &src);
  }

  if (use_a1)
    fprintf(ctx->out, ", a1.x");

  if (debug & PRINT_VERBOSE) {
    if (cat5->is_s2en_bindless) {
      if ((debug & PRINT_VERBOSE) && cat5->s2en_bindless.dummy1)
        fprintf(ctx->out, "\t{5: %x}", cat5->s2en_bindless.dummy1);
    } else {
      if ((debug & PRINT_VERBOSE) && cat5->norm.dummy1)
        fprintf(ctx->out, "\t{5: %x}", cat5->norm.dummy1);
    }
  }
}

static void print_instr_cat6_a3xx(struct disasm_ctx *ctx, instr_t *instr)
{
  instr_cat6_t *cat6 = &instr->cat6;
  char sd = 0, ss = 0;  /* dst/src address space */
  bool nodst = false;
  struct reginfo dst, src1, src2;
  int src1off = 0, dstoff = 0;

  memset(&dst, 0, sizeof(dst));
  memset(&src1, 0, sizeof(src1));
  memset(&src2, 0, sizeof(src2));

  switch (_OPC(6, cat6->opc)) {
  case OPC_RESINFO:
  case OPC_RESFMT:
    dst.full  = type_size(cat6->type) == 32;
    src1.full = type_size(cat6->type) == 32;
    src2.full = type_size(cat6->type) == 32;
    break;
  case OPC_L2G:
  case OPC_G2L:
    dst.full = true;
    src1.full = true;
    src2.full = true;
    break;
  case OPC_STG:
  case OPC_STL:
  case OPC_STP:
  case OPC_STLW:
  case OPC_STIB:
    dst.full  = type_size(cat6->type) == 32;
    src1.full = type_size(cat6->type) == 32;
    src2.full = type_size(cat6->type) == 32;
    break;
  default:
    dst.full  = type_size(cat6->type) == 32;
    src1.full = true;
    src2.full = true;
    break;
  }

  switch (_OPC(6, cat6->opc)) {
  case OPC_PREFETCH:
    break;
  case OPC_RESINFO:
    fprintf(ctx->out, ".%dd", cat6->ldgb.d + 1);
    break;
  case OPC_LDGB:
    fprintf(ctx->out, ".%s", cat6->ldgb.typed ? "typed" : "untyped");
    fprintf(ctx->out, ".%dd", cat6->ldgb.d + 1);
    fprintf(ctx->out, ".%s", type[cat6->type]);
    fprintf(ctx->out, ".%d", cat6->ldgb.type_size + 1);
    break;
  case OPC_STGB:
  case OPC_STIB:
    fprintf(ctx->out, ".%s", cat6->stgb.typed ? "typed" : "untyped");
    fprintf(ctx->out, ".%dd", cat6->stgb.d + 1);
    fprintf(ctx->out, ".%s", type[cat6->type]);
    fprintf(ctx->out, ".%d", cat6->stgb.type_size + 1);
    break;
  case OPC_ATOMIC_ADD:
  case OPC_ATOMIC_SUB:
  case OPC_ATOMIC_XCHG:
  case OPC_ATOMIC_INC:
  case OPC_ATOMIC_DEC:
  case OPC_ATOMIC_CMPXCHG:
  case OPC_ATOMIC_MIN:
  case OPC_ATOMIC_MAX:
  case OPC_ATOMIC_AND:
  case OPC_ATOMIC_OR:
  case OPC_ATOMIC_XOR:
    ss = cat6->g ? 'g' : 'l';
    fprintf(ctx->out, ".%s", cat6->ldgb.typed ? "typed" : "untyped");
    fprintf(ctx->out, ".%dd", cat6->ldgb.d + 1);
    fprintf(ctx->out, ".%s", type[cat6->type]);
    fprintf(ctx->out, ".%d", cat6->ldgb.type_size + 1);
    fprintf(ctx->out, ".%c", ss);
    break;
  default:
    dst.im = cat6->g && !cat6->dst_off;
    fprintf(ctx->out, ".%s", type[cat6->type]);
    break;
  }
  fprintf(ctx->out, " ");

  switch (_OPC(6, cat6->opc)) {
  case OPC_STG:
    sd = 'g';
    break;
  case OPC_STP:
    sd = 'p';
    break;
  case OPC_STL:
  case OPC_STLW:
    sd = 'l';
    break;

  case OPC_LDG:
  case OPC_LDC:
    ss = 'g';
    break;
  case OPC_LDP:
    ss = 'p';
    break;
  case OPC_LDL:
  case OPC_LDLW:
  case OPC_LDLV:
    ss = 'l';
    break;

  case OPC_L2G:
    ss = 'l';
    sd = 'g';
    break;

  case OPC_G2L:
    ss = 'g';
    sd = 'l';
    break;

  case OPC_PREFETCH:
    ss = 'g';
    nodst = true;
    break;
  }

  if ((_OPC(6, cat6->opc) == OPC_STGB) || (_OPC(6, cat6->opc) == OPC_STIB)) {
    struct reginfo src3;

    memset(&src3, 0, sizeof(src3));

    src1.reg = (reg_t)(cat6->stgb.src1);
    src2.reg = (reg_t)(cat6->stgb.src2);
    src2.im  = cat6->stgb.src2_im;
    src3.reg = (reg_t)(cat6->stgb.src3);
    src3.im  = cat6->stgb.src3_im;
    src3.full = true;

    fprintf(ctx->out, "g[%u], ", cat6->stgb.dst_ssbo);
    print_src(ctx, &src1);
    fprintf(ctx->out, ", ");
    print_src(ctx, &src2);
    fprintf(ctx->out, ", ");
    print_src(ctx, &src3);

    if (debug & PRINT_VERBOSE)
      fprintf(ctx->out, " (pad0=%x, pad3=%x)", cat6->stgb.pad0, cat6->stgb.pad3);

    return;
  }

  if (is_atomic(_OPC(6, cat6->opc))) {

    src1.reg = (reg_t)(cat6->ldgb.src1);
    src1.im  = cat6->ldgb.src1_im;
    src2.reg = (reg_t)(cat6->ldgb.src2);
    src2.im  = cat6->ldgb.src2_im;
    dst.reg  = (reg_t)(cat6->ldgb.dst);

    print_src(ctx, &dst);
    fprintf(ctx->out, ", ");
    if (ss == 'g') {
      struct reginfo src3;
      memset(&src3, 0, sizeof(src3));

      src3.reg = (reg_t)(cat6->ldgb.src3);
      src3.full = true;

      /* For images, the ".typed" variant is used and src2 is
       * the ivecN coordinates, ie ivec2 for 2d.
       *
       * For SSBOs, the ".untyped" variant is used and src2 is
       * a simple dword offset..  src3 appears to be
       * uvec2(offset * 4, 0).  Not sure the point of that.
       */

      fprintf(ctx->out, "g[%u], ", cat6->ldgb.src_ssbo);
      print_src(ctx, &src1);  /* value */
      fprintf(ctx->out, ", ");
      print_src(ctx, &src2);  /* offset/coords */
      fprintf(ctx->out, ", ");
      print_src(ctx, &src3);  /* 64b byte offset.. */

      if (debug & PRINT_VERBOSE) {
        fprintf(ctx->out, " (pad0=%x, pad3=%x, mustbe0=%x)", cat6->ldgb.pad0,
            cat6->ldgb.pad3, cat6->ldgb.mustbe0);
      }
    } else { /* ss == 'l' */
      fprintf(ctx->out, "l[");
      print_src(ctx, &src1);  /* simple byte offset */
      fprintf(ctx->out, "], ");
      print_src(ctx, &src2);  /* value */

      if (debug & PRINT_VERBOSE) {
        fprintf(ctx->out, " (src3=%x, pad0=%x, pad3=%x, mustbe0=%x)",
            cat6->ldgb.src3, cat6->ldgb.pad0,
            cat6->ldgb.pad3, cat6->ldgb.mustbe0);
      }
    }

    return;
  } else if (_OPC(6, cat6->opc) == OPC_RESINFO) {
    dst.reg  = (reg_t)(cat6->ldgb.dst);

    print_src(ctx, &dst);
    fprintf(ctx->out, ", ");
    fprintf(ctx->out, "g[%u]", cat6->ldgb.src_ssbo);

    return;
  } else if (_OPC(6, cat6->opc) == OPC_LDGB) {

    src1.reg = (reg_t)(cat6->ldgb.src1);
    src1.im  = cat6->ldgb.src1_im;
    src2.reg = (reg_t)(cat6->ldgb.src2);
    src2.im  = cat6->ldgb.src2_im;
    dst.reg  = (reg_t)(cat6->ldgb.dst);

    print_src(ctx, &dst);
    fprintf(ctx->out, ", ");
    fprintf(ctx->out, "g[%u], ", cat6->ldgb.src_ssbo);
    print_src(ctx, &src1);
    fprintf(ctx->out, ", ");
    print_src(ctx, &src2);

    if (debug & PRINT_VERBOSE)
      fprintf(ctx->out, " (pad0=%x, pad3=%x, mustbe0=%x)", cat6->ldgb.pad0, cat6->ldgb.pad3, cat6->ldgb.mustbe0);

    return;
  } else if (_OPC(6, cat6->opc) == OPC_LDG && cat6->a.src1_im && cat6->a.src2_im) {
    struct reginfo src3;

    memset(&src3, 0, sizeof(src3));
    src1.reg = (reg_t)(cat6->a.src1);
    src2.reg = (reg_t)(cat6->a.src2);
    src2.im  = cat6->a.src2_im;
    src3.reg = (reg_t)(cat6->a.off);
    src3.full = true;
    dst.reg  = (reg_t)(cat6->d.dst);

    print_src(ctx, &dst);
    fprintf(ctx->out, ", g[");
    print_src(ctx, &src1);
    fprintf(ctx->out, "+");
    print_src(ctx, &src3);
    fprintf(ctx->out, "], ");
    print_src(ctx, &src2);

    return;
  }
  if (cat6->dst_off) {
    dst.reg = (reg_t)(cat6->c.dst);
    dstoff  = cat6->c.off;
  } else {
    dst.reg = (reg_t)(cat6->d.dst);
  }

  if (cat6->src_off) {
    src1.reg = (reg_t)(cat6->a.src1);
    src1.im  = cat6->a.src1_im;
    src2.reg = (reg_t)(cat6->a.src2);
    src2.im  = cat6->a.src2_im;
    src1off  = cat6->a.off;
  } else {
    src1.reg = (reg_t)(cat6->b.src1);
    src1.im  = cat6->b.src1_im;
    src2.reg = (reg_t)(cat6->b.src2);
    src2.im  = cat6->b.src2_im;
  }

  if (!nodst) {
    if (sd)
      fprintf(ctx->out, "%c[", sd);
    /* note: dst might actually be a src (ie. address to store to) */
    print_src(ctx, &dst);
    if (cat6->dst_off && cat6->g) {
      struct reginfo dstoff_reg = {0};
      dstoff_reg.reg = (reg_t) cat6->c.off;
      dstoff_reg.full  = true;
      fprintf(ctx->out, "+");
      print_src(ctx, &dstoff_reg);
    } else if (dstoff)
      fprintf(ctx->out, "%+d", dstoff);
    if (sd)
      fprintf(ctx->out, "]");
    fprintf(ctx->out, ", ");
  }

  if (ss)
    fprintf(ctx->out, "%c[", ss);

  /* can have a larger than normal immed, so hack: */
  if (src1.im) {
    fprintf(ctx->out, "%u", src1.reg.dummy13);
  } else {
    print_src(ctx, &src1);
  }

  if (cat6->src_off && cat6->g)
    print_src(ctx, &src2);
  else if (src1off)
    fprintf(ctx->out, "%+d", src1off);
  if (ss)
    fprintf(ctx->out, "]");

  switch (_OPC(6, cat6->opc)) {
  case OPC_RESINFO:
  case OPC_RESFMT:
    break;
  default:
    fprintf(ctx->out, ", ");
    print_src(ctx, &src2);
    break;
  }
}

static void print_instr_cat6_a6xx(struct disasm_ctx *ctx, instr_t *instr)
{
  instr_cat6_a6xx_t *cat6 = &instr->cat6_a6xx;
  struct reginfo src1, src2, ssbo;
  bool uses_type = _OPC(6, cat6->opc) != OPC_LDC;

  static const struct {
    bool indirect;
    bool bindless;
    const char *name;
  } desc_features[8] = {
    [CAT6_IMM] = {
      .name = "imm"
    },
    [CAT6_UNIFORM] = {
      .indirect = true,
      .name = "uniform"
    },
    [CAT6_NONUNIFORM] = {
      .indirect = true,
      .name = "nonuniform"
    },
    [CAT6_BINDLESS_IMM] = {
      .bindless = true,
      .name = "imm"
    },
    [CAT6_BINDLESS_UNIFORM] = {
      .bindless = true,
      .indirect = true,
      .name = "uniform"
    },
    [CAT6_BINDLESS_NONUNIFORM] = {
      .bindless = true,
      .indirect = true,
      .name = "nonuniform"
    },
  };

  bool indirect_ssbo = desc_features[cat6->desc_mode].indirect;
  bool bindless = desc_features[cat6->desc_mode].bindless;
  bool type_full = cat6->type != TYPE_U16;


  memset(&src1, 0, sizeof(src1));
  memset(&src2, 0, sizeof(src2));
  memset(&ssbo, 0, sizeof(ssbo));

  if (uses_type) {
    fprintf(ctx->out, ".%s", cat6->typed ? "typed" : "untyped");
    fprintf(ctx->out, ".%dd", cat6->d + 1);
    fprintf(ctx->out, ".%s", type[cat6->type]);
  } else {
    fprintf(ctx->out, ".offset%d", cat6->d);
  }
  fprintf(ctx->out, ".%u", cat6->type_size + 1);

  fprintf(ctx->out, ".%s", desc_features[cat6->desc_mode].name);
  if (bindless)
    fprintf(ctx->out, ".base%d", cat6->base);
  fprintf(ctx->out, " ");

  src2.reg = (reg_t)(cat6->src2);
  src2.full = type_full;
  print_src(ctx, &src2);
  fprintf(ctx->out, ", ");

  src1.reg = (reg_t)(cat6->src1);
  src1.full = true; // XXX
  print_src(ctx, &src1);
  fprintf(ctx->out, ", ");
  ssbo.reg = (reg_t)(cat6->ssbo);
  ssbo.im = !indirect_ssbo;
  ssbo.full = true;
  print_src(ctx, &ssbo);

  if (debug & PRINT_VERBOSE) {
    fprintf(ctx->out, " (pad1=%x, pad2=%x, pad3=%x, pad4=%x, pad5=%x)",
        cat6->pad1, cat6->pad2, cat6->pad3, cat6->pad4, cat6->pad5);
  }
}

static void print_instr_cat6(struct disasm_ctx *ctx, instr_t *instr)
{
  if (!is_cat6_legacy(instr, ctx->gpu_id)) {
    print_instr_cat6_a6xx(ctx, instr);
    if (debug & PRINT_VERBOSE)
      fprintf(ctx->out, " NEW");
  } else {
    print_instr_cat6_a3xx(ctx, instr);
    if (debug & PRINT_VERBOSE)
      fprintf(ctx->out, " LEGACY");
  }
}
static void print_instr_cat7(struct disasm_ctx *ctx, instr_t *instr)
{
  instr_cat7_t *cat7 = &instr->cat7;

  if (cat7->g)
    fprintf(ctx->out, ".g");
  if (cat7->l)
    fprintf(ctx->out, ".l");

  if (_OPC(7, cat7->opc) == OPC_FENCE) {
    if (cat7->r)
      fprintf(ctx->out, ".r");
    if (cat7->w)
      fprintf(ctx->out, ".w");
  }
}

/* size of largest OPC field of all the instruction categories: */
#define NOPC_BITS 6

static const struct opc_info {
  uint16_t cat;
  uint16_t opc;
  const char *name;
  void (*print)(struct disasm_ctx *ctx, instr_t *instr);
} opcs[1 << (3+NOPC_BITS)] = {
#define OPC(cat, opc, name) [(opc)] = { (cat), (opc), #name, print_instr_cat##cat }
  /* category 0: */
  OPC(0, OPC_NOP,          nop),
  OPC(0, OPC_B,            b),
  OPC(0, OPC_JUMP,         jump),
  OPC(0, OPC_CALL,         call),
  OPC(0, OPC_RET,          ret),
  OPC(0, OPC_KILL,         kill),
  OPC(0, OPC_END,          end),
  OPC(0, OPC_EMIT,         emit),
  OPC(0, OPC_CUT,          cut),
  OPC(0, OPC_CHMASK,       chmask),
  OPC(0, OPC_CHSH,         chsh),
  OPC(0, OPC_FLOW_REV,     flow_rev),
  OPC(0, OPC_PREDT,        predt),
  OPC(0, OPC_PREDF,        predf),
  OPC(0, OPC_PREDE,        prede),
  OPC(0, OPC_BKT,          bkt),
  OPC(0, OPC_STKS,         stks),
  OPC(0, OPC_STKR,         stkr),
  OPC(0, OPC_XSET,         xset),
  OPC(0, OPC_XCLR,         xclr),
  OPC(0, OPC_GETONE,       getone),
  OPC(0, OPC_DBG,          dbg),
  OPC(0, OPC_SHPS,         shps),
  OPC(0, OPC_SHPE,         shpe),

  /* category 1: */
  OPC(1, OPC_MOV, ),

  /* category 2: */
  OPC(2, OPC_ADD_F,        add.f),
  OPC(2, OPC_MIN_F,        min.f),
  OPC(2, OPC_MAX_F,        max.f),
  OPC(2, OPC_MUL_F,        mul.f),
  OPC(2, OPC_SIGN_F,       sign.f),
  OPC(2, OPC_CMPS_F,       cmps.f),
  OPC(2, OPC_ABSNEG_F,     absneg.f),
  OPC(2, OPC_CMPV_F,       cmpv.f),
  OPC(2, OPC_FLOOR_F,      floor.f),
  OPC(2, OPC_CEIL_F,       ceil.f),
  OPC(2, OPC_RNDNE_F,      rndne.f),
  OPC(2, OPC_RNDAZ_F,      rndaz.f),
  OPC(2, OPC_TRUNC_F,      trunc.f),
  OPC(2, OPC_ADD_U,        add.u),
  OPC(2, OPC_ADD_S,        add.s),
  OPC(2, OPC_SUB_U,        sub.u),
  OPC(2, OPC_SUB_S,        sub.s),
  OPC(2, OPC_CMPS_U,       cmps.u),
  OPC(2, OPC_CMPS_S,       cmps.s),
  OPC(2, OPC_MIN_U,        min.u),
  OPC(2, OPC_MIN_S,        min.s),
  OPC(2, OPC_MAX_U,        max.u),
  OPC(2, OPC_MAX_S,        max.s),
  OPC(2, OPC_ABSNEG_S,     absneg.s),
  OPC(2, OPC_AND_B,        and.b),
  OPC(2, OPC_OR_B,         or.b),
  OPC(2, OPC_NOT_B,        not.b),
  OPC(2, OPC_XOR_B,        xor.b),
  OPC(2, OPC_CMPV_U,       cmpv.u),
  OPC(2, OPC_CMPV_S,       cmpv.s),
  OPC(2, OPC_MUL_U24,      mul.u24),
  OPC(2, OPC_MUL_S24,      mul.s24),
  OPC(2, OPC_MULL_U,       mull.u),
  OPC(2, OPC_BFREV_B,      bfrev.b),
  OPC(2, OPC_CLZ_S,        clz.s),
  OPC(2, OPC_CLZ_B,        clz.b),
  OPC(2, OPC_SHL_B,        shl.b),
  OPC(2, OPC_SHR_B,        shr.b),
  OPC(2, OPC_ASHR_B,       ashr.b),
  OPC(2, OPC_BARY_F,       bary.f),
  OPC(2, OPC_MGEN_B,       mgen.b),
  OPC(2, OPC_GETBIT_B,     getbit.b),
  OPC(2, OPC_SETRM,        setrm),
  OPC(2, OPC_CBITS_B,      cbits.b),
  OPC(2, OPC_SHB,          shb),
  OPC(2, OPC_MSAD,         msad),

  /* category 3: */
  OPC(3, OPC_MAD_U16,      mad.u16),
  OPC(3, OPC_MADSH_U16,    madsh.u16),
  OPC(3, OPC_MAD_S16,      mad.s16),
  OPC(3, OPC_MADSH_M16,    madsh.m16),
  OPC(3, OPC_MAD_U24,      mad.u24),
  OPC(3, OPC_MAD_S24,      mad.s24),
  OPC(3, OPC_MAD_F16,      mad.f16),
  OPC(3, OPC_MAD_F32,      mad.f32),
  OPC(3, OPC_SEL_B16,      sel.b16),
  OPC(3, OPC_SEL_B32,      sel.b32),
  OPC(3, OPC_SEL_S16,      sel.s16),
  OPC(3, OPC_SEL_S32,      sel.s32),
  OPC(3, OPC_SEL_F16,      sel.f16),
  OPC(3, OPC_SEL_F32,      sel.f32),
  OPC(3, OPC_SAD_S16,      sad.s16),
  OPC(3, OPC_SAD_S32,      sad.s32),

  /* category 4: */
  OPC(4, OPC_RCP,          rcp),
  OPC(4, OPC_RSQ,          rsq),
  OPC(4, OPC_LOG2,         log2),
  OPC(4, OPC_EXP2,         exp2),
  OPC(4, OPC_SIN,          sin),
  OPC(4, OPC_COS,          cos),
  OPC(4, OPC_SQRT,         sqrt),
  OPC(4, OPC_HRSQ,         hrsq),
  OPC(4, OPC_HLOG2,        hlog2),
  OPC(4, OPC_HEXP2,        hexp2),

  /* category 5: */
  OPC(5, OPC_ISAM,         isam),
  OPC(5, OPC_ISAML,        isaml),
  OPC(5, OPC_ISAMM,        isamm),
  OPC(5, OPC_SAM,          sam),
  OPC(5, OPC_SAMB,         samb),
  OPC(5, OPC_SAML,         saml),
  OPC(5, OPC_SAMGQ,        samgq),
  OPC(5, OPC_GETLOD,       getlod),
  OPC(5, OPC_CONV,         conv),
  OPC(5, OPC_CONVM,        convm),
  OPC(5, OPC_GETSIZE,      getsize),
  OPC(5, OPC_GETBUF,       getbuf),
  OPC(5, OPC_GETPOS,       getpos),
  OPC(5, OPC_GETINFO,      getinfo),
  OPC(5, OPC_DSX,          dsx),
  OPC(5, OPC_DSY,          dsy),
  OPC(5, OPC_GATHER4R,     gather4r),
  OPC(5, OPC_GATHER4G,     gather4g),
  OPC(5, OPC_GATHER4B,     gather4b),
  OPC(5, OPC_GATHER4A,     gather4a),
  OPC(5, OPC_SAMGP0,       samgp0),
  OPC(5, OPC_SAMGP1,       samgp1),
  OPC(5, OPC_SAMGP2,       samgp2),
  OPC(5, OPC_SAMGP3,       samgp3),
  OPC(5, OPC_DSXPP_1,      dsxpp.1),
  OPC(5, OPC_DSYPP_1,      dsypp.1),
  OPC(5, OPC_RGETPOS,      rgetpos),
  OPC(5, OPC_RGETINFO,     rgetinfo),


  /* category 6: */
  OPC(6, OPC_LDG,          ldg),
  OPC(6, OPC_LDL,          ldl),
  OPC(6, OPC_LDP,          ldp),
  OPC(6, OPC_STG,          stg),
  OPC(6, OPC_STL,          stl),
  OPC(6, OPC_STP,          stp),
  OPC(6, OPC_LDIB,         ldib),
  OPC(6, OPC_G2L,          g2l),
  OPC(6, OPC_L2G,          l2g),
  OPC(6, OPC_PREFETCH,     prefetch),
  OPC(6, OPC_LDLW,         ldlw),
  OPC(6, OPC_STLW,         stlw),
  OPC(6, OPC_RESFMT,       resfmt),
  OPC(6, OPC_RESINFO,      resinfo),
  OPC(6, OPC_ATOMIC_ADD,     atomic.add),
  OPC(6, OPC_ATOMIC_SUB,     atomic.sub),
  OPC(6, OPC_ATOMIC_XCHG,    atomic.xchg),
  OPC(6, OPC_ATOMIC_INC,     atomic.inc),
  OPC(6, OPC_ATOMIC_DEC,     atomic.dec),
  OPC(6, OPC_ATOMIC_CMPXCHG, atomic.cmpxchg),
  OPC(6, OPC_ATOMIC_MIN,     atomic.min),
  OPC(6, OPC_ATOMIC_MAX,     atomic.max),
  OPC(6, OPC_ATOMIC_AND,     atomic.and),
  OPC(6, OPC_ATOMIC_OR,      atomic.or),
  OPC(6, OPC_ATOMIC_XOR,     atomic.xor),
  OPC(6, OPC_LDGB,         ldgb),
  OPC(6, OPC_STGB,         stgb),
  OPC(6, OPC_STIB,         stib),
  OPC(6, OPC_LDC,          ldc),
  OPC(6, OPC_LDLV,         ldlv),

  OPC(7, OPC_BAR,          bar),
  OPC(7, OPC_FENCE,        fence),

#undef OPC
};

#define GETINFO(instr) (&(opcs[((instr)->opc_cat << NOPC_BITS) | instr_opc(instr, ctx->gpu_id)]))

// XXX hack.. probably should move this table somewhere common:
#include "ir3.h"
const char *ir3_instr_name(struct ir3_instruction *instr)
{
  if (opc_cat(instr->opc) == -1) return "??meta??";
  return opcs[instr->opc].name;
}

static void print_single_instr(struct disasm_ctx *ctx, instr_t *instr)
{
  const char *name = GETINFO(instr)->name;
  uint32_t opc = instr_opc(instr, ctx->gpu_id);

  if (name) {
    fprintf(ctx->out, "%s", name);
    GETINFO(instr)->print(ctx, instr);
  } else {
    fprintf(ctx->out, "unknown(%d,%d)", instr->opc_cat, opc);

    switch (instr->opc_cat) {
    case 0: print_instr_cat0(ctx, instr); break;
    case 1: print_instr_cat1(ctx, instr); break;
    case 2: print_instr_cat2(ctx, instr); break;
    case 3: print_instr_cat3(ctx, instr); break;
    case 4: print_instr_cat4(ctx, instr); break;
    case 5: print_instr_cat5(ctx, instr); break;
    case 6: print_instr_cat6(ctx, instr); break;
    case 7: print_instr_cat7(ctx, instr); break;
    }
  }
}

static bool print_instr(struct disasm_ctx *ctx, uint32_t *dwords, int n)
{
  instr_t *instr = (instr_t *)dwords;
  uint32_t opc = instr_opc(instr, ctx->gpu_id);
  unsigned nop = 0;
  unsigned cycles = ctx->instructions;

  if (debug & PRINT_VERBOSE) {
    fprintf(ctx->out, "%s%04d:%04d[%08xx_%08xx] ", levels[ctx->level],
        n, cycles++, dwords[1], dwords[0]);
  }

  /* NOTE: order flags are printed is a bit fugly.. but for now I
   * try to match the order in llvm-a3xx disassembler for easy
   * diff'ing..
   */

  ctx->repeat = instr_repeat(instr);
  ctx->instructions += 1 + ctx->repeat;

  if (instr->sync) {
    fprintf(ctx->out, "(sy)");
  }
  if (instr->ss && ((instr->opc_cat <= 4) || (instr->opc_cat == 7))) {
    fprintf(ctx->out, "(ss)");
  }
  if (instr->jmp_tgt)
    fprintf(ctx->out, "(jp)");
  if ((instr->opc_cat == 0) && instr->cat0.eq)
    fprintf(ctx->out, "(eq)");
  if (instr_sat(instr))
    fprintf(ctx->out, "(sat)");
  if (ctx->repeat)
    fprintf(ctx->out, "(rpt%d)", ctx->repeat);
  else if ((instr->opc_cat == 2) && (instr->cat2.src1_r || instr->cat2.src2_r))
    nop = (instr->cat2.src2_r * 2) + instr->cat2.src1_r;
  else if ((instr->opc_cat == 3) && (instr->cat3.src1_r || instr->cat3.src2_r))
    nop = (instr->cat3.src2_r * 2) + instr->cat3.src1_r;
  ctx->instructions += nop;
  if (nop)
    fprintf(ctx->out, "(nop%d) ", nop);

  if (instr->ul && ((2 <= instr->opc_cat) && (instr->opc_cat <= 4)))
    fprintf(ctx->out, "(ul)");

  print_single_instr(ctx, instr);
  fprintf(ctx->out, "\n");

  if ((instr->opc_cat <= 4) && (debug & EXPAND_REPEAT)) {
    int i;
    for (i = 0; i < nop; i++) {
      if (debug & PRINT_VERBOSE) {
        fprintf(ctx->out, "%s%04d:%04d[                   ] ",
            levels[ctx->level], n, cycles++);
      }
      fprintf(ctx->out, "nop\n");
    }
    for (i = 0; i < ctx->repeat; i++) {
      ctx->repeatidx = i + 1;
      if (debug & PRINT_VERBOSE) {
        fprintf(ctx->out, "%s%04d:%04d[                   ] ",
            levels[ctx->level], n, cycles++);
      }
      print_single_instr(ctx, instr);
      fprintf(ctx->out, "\n");
    }
    ctx->repeatidx = 0;
  }

  return (instr->opc_cat == 0) && (opc == OPC_END);
}

int disasm_a3xx(uint32_t *dwords, int sizedwords, int level, FILE *out, unsigned gpu_id)
{
  struct disasm_ctx ctx;
  int i;
  int nop_count = 0;

  //assert((sizedwords % 2) == 0);

  memset(&ctx, 0, sizeof(ctx));
  ctx.out = out;
  ctx.level = level;
  ctx.gpu_id = gpu_id;

  for (i = 0; i < sizedwords; i += 2) {
    print_instr(&ctx, &dwords[i], i/2);
    if (dwords[i] == 0 && dwords[i + 1] == 0)
      nop_count++;
    else
      nop_count = 0;
    if (nop_count > 3)
      break;
  }

  return 0;
}

// gcc -shared disasm-a3xx.c -o disasm.so
void disasm(uint8_t* buf, int len) {
  disasm_a3xx((uint32_t*)buf, len/4, 0, stdout, 630);
}

/*int main(int argc, char *argv[]) {
  uint32_t buf[0x10000];
  FILE *f = fopen(argv[1], "rb");
  if (argc > 2) {
    int seek = atoi(argv[2]);
    printf("skip %d\n", seek);
    fread(buf, 1, seek , f);
  }
  int len = fread(buf, 1, sizeof(buf), f);
  fclose(f);

  disasm_a3xx(buf, len/4, 0, stdout, 630);
}*/

