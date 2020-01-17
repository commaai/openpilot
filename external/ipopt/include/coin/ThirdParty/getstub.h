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

#ifndef GETSTUB_H_included
#define GETSTUB_H_included
#ifndef ASL_included
#include "asl.h"
#endif

 typedef struct keyword keyword;

 typedef char *
Kwfunc(Option_Info *oi, keyword *kw, char *value);

 struct
keyword {
	char *name;
	Kwfunc *kf;
	void *info;
	char *desc;
	};

#define KW(a,b,c,d) {a,b,(void*)(c),d}
#define nkeywds (int)(sizeof(keywds)/sizeof(keyword))

 typedef fint Solver_KW_func(char*, fint);
 typedef fint Fileeq_func(fint*, char*, fint);

 struct
Option_Info {
	char *sname;		/* invocation name of solver */
	char *bsname;		/* solver name in startup "banner" */
	char *opname;		/* name of solver_options environment var */
	keyword *keywds;	/* key words */
	int n_keywds;		/* number of key words */
	int flags;		/* whether funcadd will be called, etc.: */
				/* see the first enum below  */
	char *version;		/* for -v and Ver_key_ASL() */
	char **usage;		/* solver-specific usage message */
	Solver_KW_func *kwf;	/* solver-specific keyword function */
	Fileeq_func *feq;	/* for nnn=filename */
	keyword *options;	/* command-line options (with -) before stub */
	int n_options;		/* number of options */
	long driver_date;	/* YYYYMMDD for driver */

	/* For write_sol: */

	int wantsol;		/* write .sol file without -AMPL */
	int nS;			/* transmit S[i], 0 <= i < nS */
	SufDesc *S;

	/* For possible use by "nonstandard" Kwfunc's: */

	char *uinfo;

	/* Stuff provided/used by getopts (and getstops): */

	ASL *asl;
	char *eqsign;
	int n_badopts;	/* number of bad options: bail out if != 0*/
	int option_echo;/* whether to echo: see the second enum below.  */
	/* Kwfunc's may set option_echo &= ~ASL_OI_echo to turn off all */
	/* keyword echoing or option_echo &= ~ASL_OI_echothis to turn 	*/
	/* off echoing of the present keyword.  If they detect but do	*/
	/* not themselves report a bad value, they should set		*/
	/* option_echo |= ASL_OI_badvalue.   During command-line option	*/
	/* processing (for -... args), (option_echo & ASL_OI_clopt) is	*/
	/* nonzero. */

	int nnl;	/* internal use: copied to asl->i.need_nl_ */
	};

 enum { /* bits for Option_Info.flags */
	ASL_OI_want_funcadd = 1,
	ASL_OI_keep_underscores = 2,
	ASL_OI_show_version = 4
	} ;

 enum { /* bits for Option_Info.option_echo */
	ASL_OI_echo	= 1,
	ASL_OI_echothis = 2,
	ASL_OI_clopt	= 4,
	ASL_OI_badvalue = 8,
	ASL_OI_never_echo = 16,
	ASL_OI_tabexpand  = 32,	/* have shownames() expand tabs */
	ASL_OI_addnewline = 64, /* have shownames() add a newline */
				/* after each keyword description */
	ASL_OI_showname_bits = 96,
	ASL_OI_defer_bsname = 128 /* print "bsname: " only if there */
				  /* are options to echo */
	} ;

#ifdef __cplusplus
 extern "C" {
#endif

/* Kwfuncs should invoke badopt_ASL() if they complain. */
extern void  badopt_ASL (Option_Info*);
extern char *badval_ASL (Option_Info*, keyword*, char *value, char *badc);
extern char* get_opt_ASL  (Option_Info*, char*);
extern int   getopts_ASL  (ASL*, char **argv, Option_Info*);
extern char* getstops_ASL (ASL*, char **argv, Option_Info*);
extern char* getstub_ASL (ASL*, char ***pargv, Option_Info*);
extern void show_version_ASL(Option_Info*);
extern char  sysdetails_ASL[];
extern void  usage_ASL(Option_Info*, int exit_code);
extern void  usage_noexit_ASL(Option_Info*, int exit_code);

#define getstub(a,b)	getstub_ASL((ASL*)asl,a,b)
#define getstops(a,b)	getstops_ASL((ASL*)asl,a,b)
#define getopts(a,b)	getopts_ASL((ASL*)asl,a,b)

#define CK_val CK_val_ASL	/* known character value in known place */
#define C_val C_val_ASL		/* character value in known place */
#define DA_val DA_val_ASL	/* real (double) value in asl */
#define DK_val DK_val_ASL	/* known real (double) value in known place */
#define DU_val DU_val_ASL	/* real (double) value: offset from uinfo */
#define D_val D_val_ASL		/* real (double) value in known place */
#define FI_val FI_val_ASL	/* fint value in known place */
#define IA_val IA_val_ASL	/* int value in asl */
#define IK0_val IK0_val_ASL	/* int value 0 in known place */
#define IK1_val IK1_val_ASL	/* int value 1 in known place */
#define IK_val IK_val_ASL	/* known int value in known place */
#define IU_val IU_val_ASL	/* int value: offset from uinfo */
#define I_val I_val_ASL		/* int value in known place */
#define LK_val LK_val_ASL	/* known Long value in known place */
#define LU_val LU_val_ASL	/* Long value: offset from uinfo */
#define L_val L_val_ASL		/* Long value in known place */
#define SU_val SU_val_ASL	/* short value: offset from uinfo */
#define Ver_val Ver_val_ASL	/* report version */
#define WS_val WS_val_ASL	/* set wantsol in Option_Info */

extern char *Lic_info_add_ASL;	/* for show_version_ASL() */
extern char WS_desc_ASL[];	/* desc for WS_val, constrained problems */
extern char WSu_desc_ASL[];	/* desc for WS_val, unconstrained problems */

extern Kwfunc C_val, CK_val, DA_val, DK_val, DU_val, D_val, FI_val, IA_val;
extern Kwfunc IK0_val, IK1_val, IK_val, IU_val, I_val, LK_val, LU_val;
extern Kwfunc L_val, Ver_val, WS_val;
extern Kwfunc SU_val;

/* Routines for converting Double (real), Long, and int values: */

extern char *Dval_ASL (Option_Info*, keyword*, char*, real*);
extern char *Ival_ASL (Option_Info*, keyword*, char*, int*);
extern char *Lval_ASL (Option_Info*, keyword*, char*, Long*);

#define voffset_of(t,c) ((void *)&((t*)0)->c)

/* Structs whose address can be the info field for known values... */

#define C_Known C_Known_ASL	/* char* value for CK_val */
#define D_Known D_Known_ASL	/* real (double) value for DK_val */
#define I_Known I_Known_ASL	/* int  value for IK_val */
#define L_Known L_Known_ASL	/* Long value for LK_val */

 typedef struct
C_Known {
	char *val;
	char **valp;
	} C_Known;

 typedef struct
D_Known {
	real val;
	real *valp;
	} D_Known;

 typedef struct
I_Known {
	int val;
	int *valp;
	} I_Known;

 typedef struct
L_Known {
	Long val;
	Long *valp;
	} L_Known;

#ifdef __cplusplus
	}
#endif

#endif /* GETSTUB_H_included */
