/****************************************************************
Copyright (C) 1997-2001 Lucent Technologies
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

#ifndef FUNCADD_H_INCLUDED
#define FUNCADD_H_INCLUDED
#include "stdio1.h"	/* for ANSI and any printing */

#ifndef VA_LIST
#define VA_LIST va_list
#endif

#ifdef _WIN32
#define Stdio_redefs
#endif

 typedef struct cryptblock cryptblock;

#ifdef __cplusplus
#undef KR_headers
extern "C" {
#endif

#ifndef real
typedef double real;
#endif
typedef struct arglist arglist;
typedef struct function function;
typedef struct TVA TVA;
typedef struct AmplExports AmplExports;
typedef struct AuxInfo AuxInfo;
typedef struct TableInfo TableInfo;
typedef struct TMInfo TMInfo;

#ifndef No_arglist_def

#undef Const
#ifdef KR_headers
#define Const /* nothing */
#else
#define Const const
#endif

 struct
arglist {			/* Information sent to user-defined functions */
	int n;			/* number of args */
	int nr;			/* number of real input args */
	int *at;		/* argument types -- see DISCUSSION below */
	real *ra;		/* pure real args (IN, OUT, and INOUT) */
	Const char **sa;	/* symbolic IN args */
	real *derivs;		/* for partial derivatives (if nonzero) */
	real *hes;		/* for second partials (if nonzero) */
	char *dig;		/* if (dig && dig[i]) { partials w.r.t.	*/
				/*	ra[i] will not be used }	*/
	Char *funcinfo;		/* for use by the function (if desired) */
	AmplExports *AE;	/* functions made visible (via #defines below) */
	function *f;		/* for internal use by AMPL */
	TVA *tva;		/* for internal use by AMPL */
	char *Errmsg;		/* To indicate an error, set this to a */
				/* description of the error.  When derivs */
				/* is nonzero and the error is that first */
				/* derivatives cannot or are not computed, */
				/* a single quote character (') should be */
				/* the first character in the text assigned */
				/* to Errmsg, followed by the actual error */
				/* message.  Similarly, if hes is nonzero */
				/* and the error is that second derivatives */
				/* are not or cannot be computed, a double */
				/* quote character (") should be the first */
				/* character in Errmsg, followed by the */
				/* actual error message text. */
	TMInfo *TMI;		/* used in Tempmem calls */
	Char *Private;
				/* The following fields are relevant */
				/* only when imported functions are called */
				/* by AMPL commands (not declarations). */

	int nin;		/* number of input (IN and INOUT) args */
	int nout;		/* number of output (OUT and INOUT) args */
	int nsin;		/* number of symbolic input arguments */
	int nsout;		/* number of symbolic OUT and INOUT args */
	};

typedef real (*rfunc) ANSI((arglist *));
typedef real (ufunc) ANSI((arglist *));

#endif /* No_arglist_def */

 enum AMPLFUNC_AT_BITS {	/* Intrepretation of at[i] when the type */
				/* arg to addfunc has the */
				/* FUNCADD_OUTPUT_ARGS bit on.*/
	AMPLFUNC_INARG  = 1,	/* IN or INOUT */
	AMPLFUNC_OUTARG = 2,	/* OUT or INOUT */
	AMPLFUNC_STRING = 4,	/* Input value is a string (sa[i]) */
	AMPLFUNC_STROUT = 8	/* String output value allowed */
	};

 enum FUNCADD_TYPE {			/* bits in "type" arg to addfunc */

		/* The type arg to addfunc should consist of one of the */
		/* following values ... */

	FUNCADD_REAL_VALUED = 0,	/* real (double) valued function */
	FUNCADD_STRING_VALUED = 2,	/* char* valued function (AMPL only) */
	FUNCADD_RANDOM_VALUED = 4,	/* real random valued */
	FUNCADD_012ARGS = 6,		/* Special case: real random valued */
					/* with 0 <= nargs <= 2 arguments */
					/* passed directly, rather than in */
					/* an arglist structure (AMPL only). */

		/* possibly or-ed with the following... */

	FUNCADD_STRING_ARGS = 1,	/* allow string args */
	FUNCADD_OUTPUT_ARGS = 16,	/* allow output args (AMPL only) */
	FUNCADD_TUPLE_VALUED = 32,	/* not yet allowed */

		/* internal use */
	FUNCADD_NO_ARGLIST = 8,
	FUNCADD_NO_DUPWARN = 64,	/* no complaint if already defined */
	FUNCADD_NONRAND_BUILTIN = 128	/* mean, variance, moment, etc. */
	};

/* If a constraint involves an imported function and presolve fixes all
 * the arguments of the function, AMPL may later need to ask the
 * function for its partial derivatives -- even though the solver had
 * no reason to call the function.  If so, it will pass an arglist *al
 * with al->derivs nonzero, and it will expect the function to set
 * al->derivs[i] to the partial derivative of the function with respect
 * to al->ra[i].  Solvers that need to evaluate an imported function
 * work the same way -- they set al->derivs to a nonzero value if they
 * require both the function value and its first derivatives.  Solvers
 * that expect Hessians to be supplied to them also set al->hes to a
 * nonzero value if they require second derivatives at the current
 * argument.  In this case, the function should set
 * al->hes[i + j*(j+1)/2] to the partial derivative of the function with
 * respect to al->ra[i] and al->ra[j] for all 0 <= i <= j < al->nr.
 */

typedef void AddFunc ANSI((
		const char *name,
		rfunc f,	/* cast f to (rfunc) if it returns char* */
		int type,	/* see FUNCADD_TYPE above */
		int nargs,	/* >=  0 ==> exactly that many args
				 * <= -1 ==> at least -(nargs+1) args
				 */
		void *funcinfo,	/* for use by the function (if desired) */
		AmplExports *ae
		));

typedef void AddRand ANSI((
		const char *name,
		rfunc f,	/* assumed to be a random function */
		rfunc icdf,	/* inverse CDF */
		int type,	/* FUNCADD_STRING_ARGS or 0 */
		int nargs,	/* >=  0 ==> exactly that many args
				 * <= -1 ==> at least -(nargs+1) args
				 */
		void *funcinfo,	/* for use by the function (if desired) */
		AmplExports *ae
		));

typedef void (*RandSeedSetter) ANSI((void*, unsigned long));
typedef void AddRandInit ANSI((AmplExports *ae, RandSeedSetter, void*));
typedef void Exitfunc ANSI((void*));

 struct
AuxInfo {
	AuxInfo *next;
	char *auxname;
	void *v;
	void (*f) ANSI((AmplExports*, void*, ...));
	};

 struct
AmplExports {
	FILE *StdErr;
	AddFunc *Addfunc;
	long ASLdate;
	int (*FprintF)  ANSI((FILE*, const char*, ...));
	int (*PrintF)   ANSI((const char*, ...));
	int (*SprintF)  ANSI((char*, const char*, ...));
	int (*VfprintF) ANSI((FILE*, const char*, VA_LIST));
	int (*VsprintF) ANSI((char*, const char*, VA_LIST));
	double (*Strtod) ANSI((const char*, char**));
	cryptblock *(*Crypto) ANSI((char *key, size_t scrbytes));
	Char *asl;
	void (*AtExit)  ANSI((AmplExports *ae, Exitfunc*, void*));
	void (*AtReset) ANSI((AmplExports *ae, Exitfunc*, void*));
	Char *(*Tempmem) ANSI((TMInfo*, size_t));
	void (*Add_table_handler) ANSI((
		int (*DbRead) (AmplExports *ae, TableInfo *TI),
		int (*DbWrite)(AmplExports *ae, TableInfo *TI),
		char *handler_info,
		int flags,
		void *Vinfo
		));
	Char *Private;
	void (*Qsortv) ANSI((void*, size_t, size_t, int(*)(const void*,const void*,void*), void*));

	/* More stuff for stdio in DLLs... */

	FILE	*StdIn;
	FILE	*StdOut;
	void	(*Clearerr)	ANSI((FILE*));
	int	(*Fclose)	ANSI((FILE*));
	FILE*	(*Fdopen)	ANSI((int, const char*));
	int	(*Feof)		ANSI((FILE*));
	int	(*Ferror)	ANSI((FILE*));
	int	(*Fflush)	ANSI((FILE*));
	int	(*Fgetc)	ANSI((FILE*));
	char*	(*Fgets)	ANSI((char*, int, FILE*));
	int	(*Fileno)	ANSI((FILE*));
	FILE*	(*Fopen)	ANSI((const char*, const char*));
	int	(*Fputc)	ANSI((int, FILE*));
	int	(*Fputs)	ANSI((const char*, FILE*));
	size_t	(*Fread)	ANSI((void*, size_t, size_t, FILE*));
	FILE*	(*Freopen)	ANSI((const char*, const char*, FILE*));
	int	(*Fscanf)	ANSI((FILE*, const char*, ...));
	int	(*Fseek)	ANSI((FILE*, long, int));
	long	(*Ftell)	ANSI((FILE*));
	size_t	(*Fwrite)	ANSI((const void*, size_t, size_t, FILE*));
	int	(*Pclose)	ANSI((FILE*));
	void	(*Perror)	ANSI((const char*));
	FILE*	(*Popen)	ANSI((const char*, const char*));
	int	(*Puts)		ANSI((const char*));
	void	(*Rewind)	ANSI((FILE*));
	int	(*Scanf)	ANSI((const char*, ...));
	void	(*Setbuf)	ANSI((FILE*, char*));
	int	(*Setvbuf)	ANSI((FILE*, char*, int, size_t));
	int	(*Sscanf)	ANSI((const char*, const char*, ...));
	char*	(*Tempnam)	ANSI((const char*, const char*));
	FILE*	(*Tmpfile)	ANSI((void));
	char*	(*Tmpnam)	ANSI((char*));
	int	(*Ungetc)	ANSI((int, FILE*));
	AuxInfo *AI;
	char*	(*Getenv)	ANSI((const char*));
	void	(*Breakfunc)	ANSI((int,void*));
	Char	*Breakarg;
	/* Items available with ASLdate >= 20020501 start here. */
	int (*SnprintF) ANSI((char*, size_t, const char*, ...));
	int (*VsnprintF) ANSI((char*, size_t, const char*, VA_LIST));

	AddRand *Addrand;	/* for random function/inverse CDF pairs */
	AddRandInit *Addrandinit; /* for adding a function to receive a new random seed */
	};

extern const char *i_option_ASL, *ix_details_ASL[];

#define funcadd funcadd_ASL

#if defined(_WIN32) && !defined(__MINGW32__)
__declspec(dllexport)
#endif
extern void funcadd ANSI((AmplExports*));	/* dynamically linked */
extern void af_libnamesave_ASL ANSI((AmplExports*, const char *fullname, const char *name, int nlen));
extern void note_libuse_ASL ANSI((void));	/* If funcadd() does not provide any imported */
						/* functions, it can call note_libuse_ASL() to */
						/* keep the library loaded; note_libuse_ASL() is */
						/* called, e.g., by the tableproxy table handler. */

#ifdef __cplusplus
	}
#endif

 typedef struct
DbCol {
	real	*dval;
	char	**sval;
	} DbCol;

 struct
TableInfo {
	int (*AddRows) ANSI((TableInfo *TI, DbCol *cols, long nrows));
	char *tname;	/* name of this table */
	char **strings;
	char **colnames;
	DbCol *cols;
	char *Missing;
	char *Errmsg;
	void *Vinfo;
	TMInfo *TMI;
	int nstrings;
	int arity;
	int ncols;
	int flags;
	long nrows;
	void *Private;
	int (*Lookup) ANSI((real*, char**, TableInfo*));
	long (*AdjustMaxrows) ANSI((TableInfo*, long new_maxrows));
	void *(*ColAlloc) ANSI((TableInfo*, int ncol, int sval));
	long maxrows;
	};

enum {	/* return values from (*DbRead)(...) and (*DbWrite)(...) */
	DB_Done = 0,	/* Table read or written. */
	DB_Refuse = 1,	/* Refuse to handle this table. */
	DB_Error = 2	/* Error reading or writing table. */
	};

enum {	/* bits in flags field of TableInfo */
	DBTI_flags_IN = 1,	/* table has IN  or INOUT entities */
	DBTI_flags_OUT = 2,	/* table has OUT or INOUT entities */
	DBTI_flags_INSET = 4	/* table has "in set" phrase: */
				/* DbRead could omit rows for */
				/* which Lookup(...) == -1; AMPL */
				/* will ignore such rows if DbRead */
				/* offers them. */
	};

#endif /* FUNCADD_H_INCLUDED */

#ifndef No_AE_redefs
/* Assume "{extern|static} AmplExports *ae;" is given elsewhere. */
#undef Stderr
#undef addfunc
#undef fprintf
#undef getenv
#undef printf
#undef sprintf
#undef snprintf
#undef strtod
#undef vfprintf
#undef vsprintf
#undef vsnprintf
#define Stderr (ae->StdErr)
#define addfunc(a,b,c,d,e) (*ae->Addfunc)(a,b,c,d,e,ae)
#define addrand(a,b,c,d,e,f) (*ae->Addrand)(a,b,c,d,e,f,ae)
#define addrandinit(a,b) (*ae->Addrandinit)(ae,a,b)
#define printf	(*ae->PrintF)
#define fprintf (*ae->FprintF)
#define snprintf (*ae->SnprintF)
#define sprintf (*ae->SprintF)
#define strtod  (*ae->Strtod)
#define vfprintf (*ae->VfprintF)
#define vsprintf (*ae->VsprintF)
#define vsnprintf (*ae->VsnprintF)
#define TempMem(x,y) (*ae->Tempmem)(x,y)
#define at_exit(x,y) (*ae->AtExit)(ae,x,y)
#define at_reset(x,y) (*ae->AtReset)(ae,x,y)
#define add_table_handler(a,b,c,d,e) (*ae->Add_table_handler)(a,b,c,d,e)
#define qsortv(a,b,c,d,e) (*ae->Qsortv)(a,b,c,d,e)
#define getenv(x) (*ae->Getenv)(x)
#ifdef Stdio_redefs
#undef clearerr
#undef fclose
#undef fdopen
#undef feof
#undef ferror
#undef fflush
#undef fgetc
#undef fgets
#undef fileno
#undef fopen
#undef fputc
#undef fputs
#undef fread
#undef freopen
#undef fscanf
#undef fseek
#undef ftell
#undef fwrite
#undef getc
#undef getchar
#undef gets
#undef pclose
#undef perror
#undef popen
#undef putc
#undef putchar
#undef puts
#undef rewind
#undef scanf
#undef setbuf
#undef setvbuf
#undef sscanf
#undef tempnam
#undef tmpfile
#undef tmpnam
#undef ungetc
#undef vprintf
#define clearerr	(*ae->Clearerr)
#define fclose		(*ae->Fclose)
#define fdopen		(*ae->Fdopen)
#define feof		(*ae->Feof)
#define ferror		(*ae->Ferror)
#define fflush		(*ae->Fflush)
#define fgetc		(*ae->Fgetc)
#define fgets		(*ae->Fgets)
#define fileno		(*ae->Fileno)
#define fopen		(*ae->Fopen)
#define fputc		(*ae->Fputc)
#define fputs		(*ae->Fputs)
#define fread		(*ae->Fread)
#define freopen		(*ae->Freopen)
#define fscanf		(*ae->Fscanf)
#define fseek		(*ae->Fseek)
#define ftell		(*ae->Ftell)
#define fwrite		(*ae->Fwrite)
#define getc		(*ae->Fgetc)
#define getchar()	(*ae->Getc)(ae->StdIn)
#define gets		Error - use "fgets" rather than "gets"
#define pclose		(*ae->Pclose)
#define perror		(*ae->Perror)
#define popen		(*ae->Popen)
#define putc		(*ae->Fputc)
#define putchar(x)	(*ae->Fputc)(ae->StdOut,(x))
#define puts		(*ae->Puts)
#define rewind		(*ae->Rewind)
#define scanf		(*ae->Scanf)
#define setbuf		(*ae->Setbuf)
#define setvbuf		(*ae->Setvbuf)
#define sscanf		(*ae->Sscanf)
#define tempnam		(*ae->Tempnam)
#define tmpfile		(*ae->Tmpfile)
#define tmpnam		(*ae->Tmpnam)
#define ungetc		(*ae->Ungetc)
#define vprintf(x,y)	(*ae->VfprintF)(ae->StdOut,(x),(y))
#define Stdin		(ae->StdIn)
#define Stdout		(ae->StdOut)
#ifndef No_std_FILE_redefs	/* may elicit compiler warnings */
#undef stdin
#undef stdout
#undef stderr
#define stdin		(ae->StdIn)
#define stdout		(ae->StdOut)
#define stderr		(ae->StdErr)
#endif /* No_std_FILE_redefs */
#endif /* Stdio_redefs */
#endif /* ifndef No_AE_redefs */

/* DISCUSSION: the "at" field of an arglist...
 *
 * OUT and INOUT arguments are only permitted in AMPL commands,
 * such as "let" and "call" commands (and not in declarations, e.g.,
 * of constraints and variables).
 *
 * When addfunc was called with type <= 6 (so there can be no OUT or
 * INOUT arguments), for 0 <= i < n,
 *		at[i] >= 0 ==> arg i is ra[at[i]]
 *		at[i] <  0 ==> arg i is sa[-(at[i]+1)].
 *
 * When addfunc was called with type & FUNCADD_OUTPUT_ARGS on (permitting
 * OUT and INOUT arguments), arg i is in ra[i] or sa[i] (as explained
 * below), derivs and hes are both null, and at[i] is the union of bits
 * that describe arg i:
 *	AMPLFUNC_INARG  = 1 ==> input arg;
 *	AMPLFUNC_OUTARG = 2 ==> output arg;
 *	AMPLFUNC_STROUT = 4 ==> can be assigned a string value.
 *
 * INOUT args have both the AMPLFUNC_INARG and the AMPLFUNC_OUTARG bits
 * are on, i.e., (at[i] & 3) == 3.
 *
 * Symbolic OUT and INOUT arguments are a bit complicated.  They can only
 * correspond to symbolic parameters in AMPL, which may have either a
 * string or a numeric value.  Thus there is provision for specifying
 * output values to be either numbers or strings.  For simplicity, when
 * the function accepts output arguments, ra[i] and sa[i] together describe
 * argument i.  In general (whentype & FUNCADD_OUTPUT_ARGS is nonzero in
 * the addfunc call), the incoming value of argument i is ra[i]
 * (a numeric value) if sa[i] is null and is otherwise sa[i].
 * To assign a value to argument i, either assign a numeric value to
 * ra[i] and set sa[i] = 0, or assign a non-null value to sa[i]
 * (in which case ra[i] will be ignored).  A value assigned to argument
 * i is ignored unless at[i] & AMPLFUNC_OUTARG is nonzero; if so
 * and if (at[i] & AMPLFUNC_STROUT) == 0, string values cause an error
 * message.
 */
