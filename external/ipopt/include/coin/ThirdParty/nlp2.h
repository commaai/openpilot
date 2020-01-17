/****************************************************************
Copyright (C) 1997-1998, 2000-2001 Lucent Technologies
All Rights Reserved

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the name of Lucent or any of its entities
not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior
permission.

LUCENT DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL LUCENT OR ANY OF ITS ENTITIES BE LIABLE FOR ANY
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
****************************************************************/

/* Variant of nlp.h for Hessian times vector computations. */

#ifndef NLP_H2_included
#define NLP_H2_included

#ifndef ASL_included
#include "asl.h"
#endif

#ifdef __cplusplus
 extern "C" {
#endif

typedef struct argpair2 argpair2;
typedef struct cde2 cde2;
typedef struct cexp2 cexp2;
typedef struct cexp21 cexp21;
typedef struct de2 de2;
typedef union  ei2 ei2;
typedef struct expr2 expr2;
typedef struct expr2_f expr2_f;
typedef struct expr2_h expr2_h;
typedef struct expr2_if expr2_if;
typedef struct expr2_v expr2_v;
typedef struct expr2_va expr2_va;
typedef struct funnel2 funnel2;
typedef struct hes_fun hes_fun;
typedef struct list2 list2;
typedef union  uir uir;

typedef real  efunc2 ANSI((expr2* A_ASL));
typedef char *sfunc  ANSI((expr2* A_ASL));

 union
uir {
	int	i;
	real	r;
	};

 union
ei2 {
	expr2	*e;
	expr2	**ep;
	expr2_if*eif;
	expr_n	*en;
	expr2_v	*ev;
	int	i;
	plterm	*p;
	de2	*d;
	real	*rp;
	derp	*D;
	cexp2	*ce;
	};

 struct
expr2 {
	efunc2 *op;
	int a;		/* adjoint index (for gradient computation) */
	expr2 *fwd, *bak;
	uir  dO;	/* deriv of op w.r.t. t in x + t*p */
	real aO;	/* adjoint (in Hv computation) of op */
	real adO;	/* adjoint (in Hv computation) of dO */
	real dL;	/* deriv of op w.r.t. left operand */
	ei2 L, R;	/* left and right operands */
	real dR;	/* deriv of op w.r.t. right operand */
	real dL2;	/* second partial w.r.t. L, L */
	real dLR;	/* second partial w.r.t. L, R */
	real dR2;	/* second partial w.r.t. R, R */
	};

 struct
expr2_v {
	efunc2 *op;
	int a;
	expr2 *fwd, *bak;
	uir dO;
	real aO, adO;
	real v;
	};

 struct
expr2_if {
	efunc2 *op;
	int a;
	expr2 *fwd, *bak;
	uir  dO;
	real aO, adO;
	expr2 *val, *vale, *valf, *e, *T, *Te, *Tf, *F, *Fe, *Ff;
	derp *D, *dT, *dF, *d0;
	ei2 Tv, Fv;
	expr2_if *next, *next2;
	derp *dTlast;
	};

 struct
expr2_va {
	efunc2 *op;
	int a;
	expr2 *fwd, *bak;
	uir  dO;
	real aO, adO;
	expr2 *val, *vale, *valf;
	ei2 L, R;
	expr2_va *next, *next2;
	derp *d0;
	};

 struct
cde2 {
	expr2	*e, *ee, *ef;
	derp	*d;
	int	zaplen;
	int	com11, n_com1;
	};

 struct
de2 {			/* for varargs */
	expr2 *e, *ee, *ef;
	derp *d;
	ei2 dv;
	derp *dlast;	/* for sputhes setup */
	};

 struct
list2 {
	list2	*next;
	ei2	item;
	};

 struct
cexp21 {
	expr2	*e, *ee, *ef;
	linpart	*L;
	int	nlin;
	};

 struct
cexp2 {
	expr2	*e, *ee, *ef;
	linpart	*L;
	int	nlin;
	funnel2	*funneled;
	list2	*cref;
	ei2	z;
	int	zlen;
	derp	*d;
	int	*vref;
	hes_fun	*hfun;
	};

 struct
funnel2 {
	funnel2	*next;
	cexp2	*ce;
	cde2	fcde;
	derp	*fulld;
	cplist	*cl;
	};

 struct
argpair2 {
	expr2	*e;
	union {
		char	**s;
		real	*v;
		} u;
	};

 struct
expr2_f {
	efunc2 *op;
	int	a;
	expr2 *fwd, *bak;
	uir  dO;
	real aO, adO;
	func_info *fi;
	arglist *al;
	argpair2 *ap, *ape, *sap, *sape;
	argpair2 *da;	/* differentiable args -- nonconstant */
	argpair2 *dae;
	real	**fh;		/* Hessian info */
	expr2	*args[1];
	};

 struct
expr2_h {
	efunc2 *op;
	int	a;
	char	sym[1];
	};

 typedef struct
Edag2info {
	cde2	*con2_de_;	/* constraint deriv. and expr. info */
	cde2	*lcon2_de_;	/* logical constraints */
	cde2	*obj2_de_;	/* objective  deriv. and expr. info */
	expr2_v	*var2_e_;	/* variable values (and related items) */

			/* stuff for "defined" variables */
	funnel2	*f2_b_;
	funnel2	*f2_c_;
	funnel2	*f2_o_;
	expr2_v	*var2_ex_,
		*var2_ex1_;
	cexp2	*cexps2_, *cexpsc_, *cexpso_, *cexpse_;
	cexp21	*cexps21_;
	hes_fun	*hesthread;
	char	*c_class;	/* class of each constraint: */
				/* 0 = constant */
				/* 1 = linear */
				/* 2 = quadratic */
				/* 3 = general nonlinear */
	char	*o_class;	/* class of each objective */
	char	*v_class;	/* class of each defined variable */
	int	c_class_max;	/* max of c_class values */
	int	o_class_max;	/* max of o_class values */
				/* The above are only computed if requested */
				/* by the ASL_find_c_class and */
				/* ASL_find_o_class bits of the flags arg */
				/* to pfgh_read() and pfg_read() */
	int	x0kind_init;
	} Edag2info;

 typedef struct
ASL_fgh {
	Edagpars  p;
	Edaginfo  i;
	Edag2info I;
	} ASL_fgh;

 extern efunc2 *r2_ops_ASL[];
 extern void com21eval_ASL ANSI((ASL_fgh*, int, int));
 extern void com2eval_ASL ANSI((ASL_fgh*, int, int));
 extern void fun2set_ASL ANSI((ASL_fgh*, funnel2 *));
#ifdef __cplusplus
	}
#endif

#ifndef SKIP_NL2_DEFINES
extern efunc2 f2_OPVARVAL_ASL;

#define cexpsc		asl->I.cexpsc_
#define cexpse		asl->I.cexpse_
#define cexpso		asl->I.cexpso_
#define cexps1		asl->I.cexps21_
#define cexps		asl->I.cexps2_
#define con_de		asl->I.con2_de_
#define f_b		asl->I.f2_b_
#define f_c		asl->I.f2_c_
#define f_o		asl->I.f2_o_
#define lcon_de		asl->I.lcon2_de_
#define obj_de		asl->I.obj2_de_
#define var_e		asl->I.var2_e_
#define var_ex1		asl->I.var2_ex1_
#define var_ex		asl->I.var2_ex_

#define argpair	argpair2
#define cde	cde2
#define cexp	cexp2
#define cexp1	cexp21
#define de	de2
#define ei	ei2
#define expr	expr2
#define expr_f	expr2_f
#define expr_h	expr2_h
#define expr_if	expr2_if
#define expr_v	expr2_v
#define expr_va	expr2_va
#define funnel	funnel2
#define list	list2

#define com1eval	com21eval_ASL
#define comeval		com2eval_ASL
#define funnelset	fun2set_ASL
#undef  r_ops
#define r_ops		r2_ops_ASL

#ifndef PSHVREAD
#define f_OPIFSYM	f2_IFSYM_ASL
#define f_OPPLTERM	f2_PLTERM_ASL
#define f_OPFUNCALL	f2_FUNCALL_ASL
#define f_OP1POW	f2_1POW_ASL
#define f_OP2POW	f2_2POW_ASL
#define f_OPCPOW	f2_CPOW_ASL
#define f_OPPLUS	f2_PLUS_ASL
#define f_OPSUMLIST	f2_SUMLIST_ASL
#define f_OPHOL		f2_HOL_ASL
#define f_OPPOW		f2_POW_ASL
#define f_OPVARVAL	f2_VARVAL_ASL
#endif

/* operation classes (for H*v computation) */

#define Hv_binaryR	0
#define Hv_binaryLR	1
#define Hv_unary	2
#define Hv_vararg	3
#define Hv_if		4
#define Hv_plterm	5
#define Hv_sumlist	6
#define Hv_func		7
#define Hv_negate	8
#define Hv_plusR	9
#define Hv_plusL	10
#define Hv_plusLR	11
#define Hv_minusR	12
#define Hv_minusLR	13
#define Hv_timesR	14
#define Hv_timesL	15
#define Hv_timesLR	16

/* treat if as vararg, minusL as plusL, binaryL as unary */

#endif /* SKIP_NL2_DEFINES */

#undef f_OPNUM
#define f_OPNUM (efunc2*)f_OPNUM_ASL
#endif /* NLP_H2_included */
