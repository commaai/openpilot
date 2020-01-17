/****************************************************************
Copyright (C) 1997, 1998, 2001 Lucent Technologies
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

#ifdef PSHVREAD
#ifndef PSINFO_H2_included
#define PSINFO_H2_included
#undef PSINFO_H_included
#ifndef NLP_H2_included
#include "nlp2.h"
#endif
#define cde cde2
#define la_ref la_ref2
#define linarg linarg2
#define range range2
#define rhead rhead2
#define psb_elem psb_elem2
#define psg_elem psg_elem2
#define ps_func ps_func2
#define dv_info dv_info2
#define split_ce split_ce2
#define ps_info ps_info2
#define psinfo psinfo2
#endif /* PSINFO_H2_included */
#else /* PSHVREAD */
#ifndef PSINFO_H1_included
#define PSINFO_H1_included
#undef PSINFO_H_included
#ifndef NLP_H_included
#include "nlp.h"
#endif
#endif
#endif /* PSHVREAD */
#ifndef PSINFO_H_included
#define PSINFO_H_included

 typedef struct la_ref la_ref;
 typedef struct linarg linarg;
 typedef struct range range;

 struct
la_ref {
	la_ref *next;
	expr **ep;
	real c;
	real scale;
	};

 struct
linarg {
	linarg *hnext;	/* for hashing */
	linarg *tnext;	/* next linear argument to this term */
	linarg *lnext;	/* for adjusting v->op */
	la_ref	*refs;	/* references */
	expr_v	*v;	/* variable that evaluates this linear term */
	ograd	*nz;	/* the nonzeros */
	int	nnz;	/* number of nonzeros (to help hashing) */
	int	termno;	/* helps tell whether new to this term */
	};

 typedef struct
rhead {
	range *next, *prev;
	} rhead;

#ifndef PSINFO_H0_included
#define MBLK_KMAX 30
#endif /* PSINFO_H0_included */

 typedef struct psb_elem psb_elem;

 struct
range {
	rhead	rlist;		/* list of all ranges */
	range	*hnext;		/* for hashing U */
	range	*hunext;	/* for hashing unit vectors */
	int	n;		/* rows in U */
	int	nv;		/* variables involved in U */
	int	nintv;		/* number of internal variables (non-unit */
				/* rows in U) */
	int	lasttermno;	/* termno of prev. use in this term */
				/* -1 ==> not yet used in this constr or obj. */
				/* Set to least variable (1st = 0) in this */
				/* range at the end of psedread. */
	int	lastgroupno;	/* groupno at last use of this term */
	unsigned int chksum;	/* for hashing */
	psb_elem *refs;		/* constraints and objectives with this range */
	int	*ui;		/* unit vectors defining this range */
				/* (for n >= nv) */
	linarg	**lap;		/* nonzeros in U */
	int	*cei;		/* common expressions: union over refs */
	real	*hest;		/* nonzero ==> internal Hessian triangle */
				/* computed by hvpinit */
	};

 struct
psb_elem {		/* basic element of partially-separable func */
	psb_elem *next;	/* for range.refs */
	range *U;
	int *ce;	/* common exprs if nonzero: ce[i], 1 <= i <= ce[0] */
	cde D;		/* derivative and expr info */
	int conno;	/* constraint no. (if >= 0) or -2 - obj no. */
	int termno;
	int groupno;
	};

 typedef struct
psg_elem {		/* group element details of partially-separable func */
	real	g0;	/* constant term */
	real	g1;	/* first deriv of g */
	real	g2;	/* 2nd deriv of g */
	real	scale;	/* temporary(?!!) until we introduce unary OPSCALE */
	expr_n	esum;	/* esum.v = result of summing g0, E and L */
	expr	*g;	/* unary operator */
	expr	*ge;	/* "last" unary operator */
	ograd	*og;	/* first deriv = g1 times og */
	int	nlin;	/* number of linear terms */
	int	ns;	/* number of nonlinear terms */
	linpart	*L;	/* the linear terms */
	psb_elem *E;	/* the nonlinear terms */
	} psg_elem;

 typedef struct
ps_func {
	int	nb;		/* number of basic terms */
	int	ng;		/* number of group terms */
	int	nxval;		/* for psgcomp */
	psb_elem *b;		/* the basic terms */
	psg_elem *g;		/* the group terms */
	} ps_func;

 typedef struct
dv_info {			/* defined variable info */
	ograd	*ll;		/* list of linear defined vars referenced */
	linarg	**nl;		/* nonlinear part, followed by 0 */
	real	scale;		/* scale factor for linear term */
	linarg	*lt;		/* linear term of nonlinear defined var */
	} dv_info;

 typedef struct
split_ce {
	range *r;
	int *ce;	/* common expressions */
	} split_ce;

#ifdef PSHVREAD

 struct
hes_fun {
	hes_fun *hfthread;
	cexp2	*c;
	real	*grdhes;
	ograd	*og;
	expr_v	**vp;
	int	n;
	};

 typedef struct Hesoprod Hesoprod;
 struct
Hesoprod {
	Hesoprod *next;
	ograd *left, *right;
	real coef;
	};

 typedef struct uHeswork uHeswork;
 struct
uHeswork {
	uHeswork *next;
	int k;
	range *r;
	int *ui, *uie;
	ograd *ogp[1];		/* scratch of length r->n */
	};

 typedef struct Umultinfo Umultinfo;
 struct
Umultinfo {
	Umultinfo *next;
	ograd *og, *og0;
	expr_v *v;
	int i;
	};

 typedef struct Ihinfo Ihinfo;
 struct
Ihinfo {
	Ihinfo *next;	/* for chaining ihinfo's with positive count */
	range *r;	/* list, on prev, of ranges with this ihd */
	real *hest;	/* hest memory to free */
	int ihd;	/* internal Hessian dimension, min(n,nv) */
	int k;		/* htcl(nr*(ihd*(ihd+1)/2)*sizeof(real)) */
	int nr;		/* number of ranges with this ihd */
	};

#endif /* PSHVREAD */

 typedef struct
ps_info {
	Long merge;	/* for noadjust = 1 */
	ps_func	*cps;
	ps_func	*ops;
	dv_info	*dv;
	expr_v **vp;	/* for values of common variables */
	rhead rlist;
	linarg *lalist;	/* all linargs */
	int *dvsp0;	/* dvsp0[i] = subscript of first var into which */
			/* cexp i was split, 0 <= i <= ncom */
	int nc1;	/* common expressions for just this function */
	int ns0;	/* initial number of elements */
	int ncom;	/* number of common expressions before splitting */
	int ndupdt;	/* duplicate linear terms in different terms */
	int ndupst;	/* duplicate linear terms in the same term */
	int nlttot;	/* total number of distinct linear terms */
	int ndvspcand;	/* # of defined variable candidates for splitting */
	int ndvsplit;	/* number of defined variables actually split */
	int ndvspin;	/* number of incoming terms from split defined vars */
	int ndvspout;	/* number of terms from split defined variables */
	int max_var1_;	/* used in psedread and pshvread */
	int nv0_;	/* used in psedread and pshvread */

#ifdef PSHVREAD
	/* Stuff for partially separable Hessian computations... */
	/* These arrays are allocated and zero-initialized by hes_setup, */
	/* which also supplies the cei field to ranges. */

	range **rtodo;	/* rtodo[i] = ranges first incident on col i */
	uHeswork **utodo;	/* unit ranges affecting this col */
	Hesoprod **otodo;/* otodo[i] = contributions to col i dispatched */
			/* by previous rtodo entries */
	Hesoprod *hop_free;
	real *dOscratch;/* length = nmax (below) */
	int *iOscratch;	/* length = nmax */
	Ihinfo *ihi;
	Ihinfo *ihi1;	/* first with positive count */
	int hes_setup_called;
	int nmax;	/* max{r in ranges} r->n */
	int ihdcur;	/* Current max internal Hessian dimension, */
			/* set by hvpinit. */
	int ihdmax;	/* max possible ihd */
	int ihdmin;	/* min possible ihd > 0 and <= ihdmax, or 0 */
	int khesoprod;	/* used in new_Hesoprod in sputhes.c */
	int pshv_g1;	/* whether pshv_prod should multiply by g1 */
	int linmultr;	/* linear common terms used in more than one range */
	int linhesfun;	/* linear common terms in Hessian funnels */
	int nlmultr;	/* nonlin common terms used in more than one range */
	int nlhesfun;	/* nonlin common terms in Hessian funnels */
	int ncongroups;	/* # of groups in constraints */
	int nobjgroups;	/* # of groups in objectives */
	int nhvprod;	/* # of Hessian-vector products at this Hessian */
	int npsgcomp;	/* Has psgcomp been called?  For sphes_setup. */
	expr_va *valist;	/* for sphes_setup */
	expr_if *iflist;	/* for sphes_setup */
	int *zlsave;	/* for S->_zl */
	real *oyow;	/* for xpsg_check */
	int onobj;	/* for xpsg_check */
	int onxval;	/* for xpsg_check */
	int nynz;	/* for xpsg_check */
	int ndhmax;	/* set by hvpinit_ASL */
#endif /* PSHVREAD */
	split_ce *Split_ce;	/* for sphes_setup */
	} ps_info;

#ifdef PSHVREAD

 typedef struct
ASL_pfgh {
	Edagpars p;
	Edaginfo i;
	Char *mblk_free[MBLK_KMAX];
	Edag2info I;
	ps_info2 P;
	} ASL_pfgh;

#else

 typedef struct
ASL_pfg {
	Edagpars p;
	Edaginfo i;
	Char *mblk_free[MBLK_KMAX];
	Edag1info I;
	ps_info P;
	} ASL_pfg;

#endif /* PSHVREAD */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PSINFO_H0_included
#define PSINFO_H0_included
typedef unsigned Long Ulong;

#endif /* PSINFO_H0_included */
#ifdef PSHVREAD
 extern void duthes_ASL(ASL*, real *H, int nobj, real *ow, real *y);
 extern void fullhes_ASL(ASL*, real*H, fint LH, int nobj, real*ow, real*y);
 extern void hvpinit_ASL(ASL*, int ndhmax, int nobj, real *ow, real *y);
 extern void ihd_clear_ASL(ASL_pfgh*);
 extern ASL_pfgh *pscheck_ASL(ASL*, const char*);
 extern void pshv_prod_ASL(ASL_pfgh*, range*r, int nobj, real*ow, real*y);
 extern fint sphes_setup_ASL(ASL*, SputInfo**, int nobj, int ow, int y, int ul);
 extern void sphes_ASL(ASL*, SputInfo**, real *H, int nobj, real*ow, real *y);
 extern void xpsg_check_ASL(ASL_pfgh*, int nobj, real *ow, real *y);
#else /* PSHVREAD */
 extern void xp1known_ASL(ASL*, real*, fint*);
#endif /* PSHVREAD */

#ifdef __cplusplus
	}
#endif

#define pshv_prod(r,no,ow,y) pshv_prod_ASL(asl,r,no,ow,y)

#endif /* PSINFO_H_included */
