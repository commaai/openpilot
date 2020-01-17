/****************************************************************
Copyright (C) 1997-1998, 2001 Lucent Technologies
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

#ifndef NLP_H_included
#define NLP_H_included

#ifndef ASL_included
#include "asl.h"
#endif

typedef struct argpair argpair;
typedef struct cde cde;
typedef struct cexp cexp;
typedef struct cexp1 cexp1;
typedef struct de de;
typedef union  ei ei;
typedef struct expr expr;
typedef struct expr_f expr_f;
typedef struct expr_h expr_h;
typedef struct expr_if expr_if;
typedef struct expr_v expr_v;
typedef struct expr_va expr_va;
typedef struct funnel funnel;
typedef struct list list;

typedef real efunc ANSI((expr * A_ASL));

#define r_ops     r_ops_ASL
#define obj1val   obj1val_ASL
#define obj1grd   obj1grd_ASL
#define con1val   con1val_ASL
#define jac1val   jac1val_ASL
#define con1ival  con1ival_ASL
#define con1grd   con1grd_ASL
#define lcon1val  lcon1val_ASL
#define x1known   x1known_ASL

 union
ei {
	expr	*e;
	expr	**ep;
	expr_if	*eif;
	expr_n	*en;
	int	i;
	plterm	*p;
	de	*d;
	real	*rp;
	derp	*D;
	cexp	*ce;
	};

 struct
expr {
	efunc *op;
	int a;
	real dL;
	ei L, R;
	real dR;
	};

 struct
expr_v {
	efunc *op;
	int a;
	real v;
	};

 struct
expr_if {
	efunc *op;
	int a;
	expr *e, *T, *F;
	derp *D, *dT, *dF, *d0;
	ei Tv, Fv;
	expr_if *next, *next2;
	};

 struct
expr_va {
	efunc *op;
	int a;
	ei L, R;
	expr_va *next, *next2;
	derp *d0;
	};

 struct
cde {
	expr	*e;
	derp	*d;
	int	zaplen;
	};

 struct
de {
	expr *e;
	derp *d;
	ei dv;
	};

 struct
list {
	list	*next;
	ei	item;
	};

 struct
cexp1 {
	expr	*e;
	int	nlin;
	linpart	*L;
	};

 struct
cexp {
	expr	*e;
	int	nlin;
	linpart	*L;
	funnel	*funneled;
	list	*cref;
	ei	z;
	int	zlen;
	derp	*d;
	int	*vref;
	};

 struct
funnel {
	funnel	*next;
	cexp	*ce;
	derp	*fulld;
	cplist	*cl;
	cde	fcde;
	};

 struct
argpair {
	expr	*e;
	union {
		char	**s;
		real	*v;
		} u;
	};

 struct
expr_f {
	efunc *op;
	int	a;
	func_info *fi;
	arglist	*al;
	argpair	*ap, *ape, *sap, *sape;
	expr	*args[1];
	};

 struct
expr_h {
	efunc *op;
	int	a;
	char	sym[1];
	};

 typedef struct
Edag1info {
	cde	*con_de_;	/* constraint deriv. and expr. info */
	cde	*lcon_de_;	/* logical constraints */
	cde	*obj_de_;	/* objective  deriv. and expr. info */
	expr_v	*var_e_;	/* variable values (and related items) */

			/* stuff for "defined" variables */
	funnel	*f_b_;
	funnel	*f_c_;
	funnel	*f_o_;
	expr_v	*var_ex_,
		*var_ex1_;
	cexp	*cexps_;
	cexp1	*cexps1_;
	efunc	**r_ops_;
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
	} Edag1info;

 typedef struct
ASL_fg {
	Edagpars  p;
	Edaginfo  i;
	Edag1info I;
	} ASL_fg;

#ifdef __cplusplus
 extern "C" {
#endif
 extern efunc *r_ops_ASL[];
 extern void com1eval_ASL ANSI((ASL_fg*, int, int));
 extern void comeval_ASL ANSI((ASL_fg*, int, int));
 extern void funnelset_ASL ANSI((ASL_fg*, funnel *));
 extern real obj1val ANSI((ASL*, int nobj, real *X, fint *nerror));
 extern void obj1grd ANSI((ASL*, int nobj, real *X, real *G, fint *nerror));
 extern void con1val ANSI((ASL*, real *X, real *F, fint *nerror));
 extern void jac1val ANSI((ASL*, real *X, real *JAC, fint *nerror));
 extern real con1ival ANSI((ASL*,int nc, real *X, fint *ne));
 extern void con1grd  ANSI((ASL*, int nc, real *X, real *G, fint *nerror));
 extern int  lcon1val ANSI((ASL*,int nc, real *X, fint *ne));
 extern int x0_check_ASL ANSI((ASL_fg*, real *));
 extern void x1known ANSI((ASL*, real*, fint*));
#ifdef __cplusplus
	}
#endif

#define comeval(a,b) comeval_ASL((ASL_fg*)asl,a,b)
#define com1eval(a,b) com1eval_ASL((ASL_fg*)asl,a,b)
#define funnelset(a) funnelset_ASL((ASL_fg*)asl,a)

#define cexps	asl->I.cexps_
#define cexps1	asl->I.cexps1_
#define con_de	asl->I.con_de_
#define f_b	asl->I.f_b_
#define f_c	asl->I.f_c_
#define f_o	asl->I.f_o_
#define lcon_de	asl->I.lcon_de_
#define obj_de	asl->I.obj_de_
#define var_e	asl->I.var_e_
#define var_ex	asl->I.var_ex_
#define var_ex1	asl->I.var_ex1_

#undef f_OPNUM
#define f_OPNUM (efunc*)f_OPNUM_ASL

#endif /* NLP_H_included */
