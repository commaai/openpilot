/* Copyright (C) 2008, 2011 GAMS Development and others
 All Rights Reserved.
 This code is published under the Eclipse Public License.

 $Id: HSLLoader.h 2317 2013-06-01 13:16:07Z stefan $

 Author: Stefan Vigerske
*/

#ifndef HSLLOADER_H_
#define HSLLOADER_H_

#include "IpoptConfig.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ma77_default_control
#define ma77_control ma77_control_d
#define ma77_info ma77_info_d
#define ma77_default_control ma77_default_control_d
#define ma77_open_nelt ma77_open_nelt_d
#define ma77_open ma77_open_d
#define ma77_input_vars ma77_input_vars_d
#define ma77_input_reals ma77_input_reals_d
#define ma77_analyse ma77_analyse_d
#define ma77_factor ma77_factor_d
#define ma77_factor_solve ma77_factor_solve_d
#define ma77_solve ma77_solve_d
#define ma77_resid ma77_resid_d
#define ma77_scale ma77_scale_d
#define ma77_enquire_posdef ma77_enquire_posdef_d
#define ma77_enquire_indef ma77_enquire_indef_d
#define ma77_alter ma77_alter_d
#define ma77_restart ma77_restart_d
#define ma77_finalise ma77_finalise_d
#endif

struct ma77_control;
struct ma77_info;
typedef double ma77pkgtype_d_;


#ifndef ma86_default_control
#define ma86_control ma86_control_d
#define ma86_info ma86_info_d
#define ma86_default_control ma86_default_control_d
#define ma86_analyse ma86_analyse_d
#define ma86_factor ma86_factor_d
#define ma86_factor_solve ma86_factor_solve_d
#define ma86_solve ma86_solve_d
#define ma86_finalise ma86_finalise_d
#endif

struct ma86_control;
struct ma86_info;
typedef double ma86pkgtype_d_;
typedef double ma86realtype_d_;

#ifndef ma97_default_control
#define ma97_control ma97_control_d
#define ma97_info ma97_info_d
#define ma97_default_control ma97_default_control_d
#define ma97_analyse ma97_analyse_d
#define ma97_factor ma97_factor_d
#define ma97_factor_solve ma97_factor_solve_d
#define ma97_solve ma97_solve_d
#define ma97_finalise ma97_finalise_d
#define ma97_free_akeep ma97_free_akeep_d
#endif

struct ma97_control;
struct ma97_info;
typedef double ma97pkgtype_d_;
typedef double ma97realtype_d_;

struct mc68_control_i;
struct mc68_info_i;

#ifndef __IPTYPES_HPP__
/* Type of Fortran integer translated into C */
typedef FORTRAN_INTEGER_TYPE ipfint;
#endif

typedef void (*ma27ad_t)(ipfint *N, ipfint *NZ, const ipfint *IRN, const ipfint* ICN,
          ipfint *IW, ipfint* LIW, ipfint* IKEEP, ipfint *IW1,
          ipfint* NSTEPS, ipfint* IFLAG, ipfint* ICNTL,
          double* CNTL, ipfint *INFO, double* OPS);
typedef void (*ma27bd_t)(ipfint *N, ipfint *NZ, const ipfint *IRN, const ipfint* ICN,
          double* A, ipfint* LA, ipfint* IW, ipfint* LIW,
          ipfint* IKEEP, ipfint* NSTEPS, ipfint* MAXFRT,
          ipfint* IW1, ipfint* ICNTL, double* CNTL,
          ipfint* INFO);
typedef void (*ma27cd_t)(ipfint *N, double* A, ipfint* LA, ipfint* IW,
          ipfint* LIW, double* W, ipfint* MAXFRT,
          double* RHS, ipfint* IW1, ipfint* NSTEPS,
          ipfint* ICNTL, double* CNTL);
typedef void (*ma27id_t)(ipfint* ICNTL, double* CNTL);

typedef void (*ma28ad_t)(void* nsize, void* nz, void* rw, void* licn, void* iw,
          void* lirn, void* iw2, void* pivtol, void* iw3, void* iw4, void* rw2, void* iflag);

typedef void (*ma57ad_t) (
    ipfint    *n,     /* Order of matrix. */
    ipfint    *ne,            /* Number of entries. */
    const ipfint    *irn,       /* Matrix nonzero row structure */
    const ipfint    *jcn,       /* Matrix nonzero column structure */
    ipfint    *lkeep,     /* Workspace for the pivot order of lenght 3*n */
    ipfint    *keep,      /* Workspace for the pivot order of lenght 3*n */
    /* Automatically iflag = 0; ikeep pivot order iflag = 1 */
    ipfint    *iwork,     /* Integer work space. */
    ipfint    *icntl,     /* Integer Control parameter of length 30*/
    ipfint    *info,      /* Statistical Information; Integer array of length 20 */
    double    *rinfo);    /* Double Control parameter of length 5 */

typedef void (*ma57bd_t) (
    ipfint    *n,     /* Order of matrix. */
    ipfint    *ne,            /* Number of entries. */
    double    *a,     /* Numerical values. */
    double    *fact,      /* Entries of factors. */
    ipfint    *lfact,     /* Length of array `fact'. */
    ipfint    *ifact,     /* Indexing info for factors. */
    ipfint    *lifact,    /* Length of array `ifact'. */
    ipfint    *lkeep,     /* Length of array `keep'. */
    ipfint    *keep,      /* Integer array. */
    ipfint    *iwork,     /* Workspace of length `n'. */
    ipfint    *icntl,     /* Integer Control parameter of length 20. */
    double    *cntl,      /* Double Control parameter of length 5. */
    ipfint    *info,      /* Statistical Information; Integer array of length 40. */
    double    *rinfo);    /* Statistical Information; Real array of length 20. */

typedef void (*ma57cd_t) (
    ipfint    *job,       /* Solution job.  Solve for... */
    ipfint    *n,         /* Order of matrix. */
    double    *fact,      /* Entries of factors. */
    ipfint    *lfact,     /* Length of array `fact'. */
    ipfint    *ifact,     /* Indexing info for factors. */
    ipfint    *lifact,    /* Length of array `ifact'. */
    ipfint    *nrhs,      /* Number of right hand sides. */
    double    *rhs,       /* Numerical Values. */
    ipfint    *lrhs,      /* Leading dimensions of `rhs'. */
    double    *work,      /* Real workspace. */
    ipfint    *lwork,     /* Length of `work', >= N*NRHS. */
    ipfint    *iwork,     /* Integer array of length `n'. */
    ipfint    *icntl,     /* Integer Control parameter array of length 20. */
    ipfint    *info);     /* Statistical Information; Integer array of length 40. */

typedef void (*ma57ed_t) (
    ipfint    *n,
    ipfint    *ic,        /* 0: copy real array.  >=1:  copy integer array. */
    ipfint    *keep,
    double    *fact,
    ipfint    *lfact,
    double    *newfac,
    ipfint    *lnew,
    ipfint    *ifact,
    ipfint    *lifact,
    ipfint    *newifc,
    ipfint    *linew,
    ipfint    *info);

typedef void (*ma57id_t) (double *cntl, ipfint *icntl);

typedef void (*ma77_default_control_t)(struct ma77_control_d *control);
typedef void (*ma77_open_nelt_t)(const int n, const char* fname1, const char* fname2,
   const char *fname3, const char *fname4, void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info,
   const int nelt);
typedef void (*ma77_open_t)(const int n, const char* fname1, const char* fname2,
   const char *fname3, const char *fname4, void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info);
typedef void (*ma77_input_vars_t)(const int idx, const int nvar, const int list[],
   void **keep, const struct ma77_control_d *control, struct ma77_info_d *info);
typedef void (*ma77_input_reals_t)(const int idx, const int length,
   const double reals[], void **keep, const struct ma77_control_d *control,
   struct ma77_info_d *info);
typedef void (*ma77_analyse_t)(const int order[], void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info);
typedef void (*ma77_factor_t)(const int posdef, void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info,
   const double *scale);
typedef void (*ma77_factor_solve_t)(const int posdef, void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info,
   const double *scale, const int nrhs, const int lx,
   double rhs[]);
typedef void (*ma77_solve_t)(const int job, const int nrhs, const int lx, double x[],
   void **keep, const struct ma77_control_d *control, struct ma77_info_d *info,
   const double *scale);
typedef void (*ma77_resid_t)(const int nrhs, const int lx, const double x[],
   const int lresid, double resid[], void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info,
   double *anorm_bnd);
typedef void (*ma77_scale_t)(double scale[], void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info,
   double *anorm);
typedef void (*ma77_enquire_posdef_t)(double d[], void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info);
typedef void (*ma77_enquire_indef_t)(int piv_order[], double d[], void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info);
typedef void (*ma77_alter_t)(const double d[], void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info);
typedef void (*ma77_restart_t)(const char *restart_file, const char *fname1,
   const char *fname2, const char *fname3, const char *fname4, void **keep,
   const struct ma77_control_d *control, struct ma77_info_d *info);
typedef void (*ma77_finalise_t)(void **keep, const struct ma77_control_d *control,
   struct ma77_info_d *info);

typedef void (*ma86_default_control_t)(struct ma86_control *control);
typedef void (*ma86_analyse_t)(const int n, const int ptr[], const int row[],
   int order[], void **keep, const struct ma86_control *control,
   struct ma86_info *info);
typedef void (*ma86_factor_t)(const int n, const int ptr[], const int row[],
   const ma86pkgtype_d_ val[], const int order[], void **keep,
   const struct ma86_control *control, struct ma86_info *info,
   const ma86pkgtype_d_ scale[]);
typedef void (*ma86_factor_solve_t)(const int n, const int ptr[],
   const int row[], const ma86pkgtype_d_ val[], const int order[], void **keep,
   const struct ma86_control *control, struct ma86_info *info, const int nrhs,
   const int ldx, ma86pkgtype_d_ x[], const ma86pkgtype_d_ scale[]);
typedef void (*ma86_solve_t)(const int job, const int nrhs, const int ldx,
   ma86pkgtype_d_ *x, const int order[], void **keep,
   const struct ma86_control *control, struct ma86_info *info,
   const ma86pkgtype_d_ scale[]);
typedef void (*ma86_finalise_t)(void **keep,
   const struct ma86_control *control);

typedef void (*ma97_default_control_t)(struct ma97_control *control);
typedef void (*ma97_analyse_t)(const int check, const int n, const int ptr[],
   const int row[], ma97pkgtype_d_ val[], void **akeep,
   const struct ma97_control *control, struct ma97_info *info, int order[]);
typedef void (*ma97_factor_t)(const int matrix_type, const int ptr[],
   const int row[], const ma97pkgtype_d_ val[], void **akeep, void **fkeep,
   const struct ma97_control *control, struct ma97_info *info,
   const ma97pkgtype_d_ scale[]);
typedef void (*ma97_factor_solve_t)(const int matrix_type, const int ptr[],
   const int row[], const ma97pkgtype_d_ val[], const int nrhs,
   ma97pkgtype_d_ x[], const int ldx,  void **akeep, void **fkeep,
   const struct ma97_control *control, struct ma97_info *info,
   const ma97pkgtype_d_ scale[]);
typedef void (*ma97_solve_t)(const int job, const int nrhs, ma97pkgtype_d_ *x,
   const int ldx, void **akeep, void **fkeep,
   const struct ma97_control *control, struct ma97_info *info);
typedef void (*ma97_finalise_t)(void **akeep, void **fkeep);
typedef void (*ma97_free_akeep_t)(void **akeep);

typedef void (*mc19ad_t)(ipfint *N, ipfint *NZ, double* A, ipfint *IRN, ipfint* ICN, float* R, float* C, float* W);

typedef void (*mc68_default_control_t)(struct mc68_control_i *control);
typedef void (*mc68_order_t)(int ord, int n, const int ptr[],
   const int row[], int perm[], const struct mc68_control_i *control,
   struct mc68_info_i *info);

  /** Tries to load a dynamically linked library with HSL routines.
   * Also tries to load symbols for those HSL routines that are not linked into Ipopt, i.e., HAVE_... is not defined. 
   * Return a failure if the library cannot be loaded, but not if a symbol is not found.
   * @see LSL_isMA27available
   * @see LSL_isMA28available
   * @see LSL_isMA57available
   * @see LSL_isMA77available
   * @see LSL_isMA86available
   * @see LSL_isMA97available
   * @see LSL_isMC19available
   * @param libname The name under which the HSL lib can be found, or NULL to use a default name (libhsl.SHAREDLIBEXT).
   * @param msgbuf A buffer where we can store a failure message. Assumed to be NOT NULL!
   * @param msglen Length of the message buffer.
   * @return Zero on success, nonzero on failure.
   */
  int LSL_loadHSL(const char* libname, char* msgbuf, int msglen);

  /** Unloads a loaded HSL library.
   * @return Zero on success, nonzero on failure.
   */
  int LSL_unloadHSL();

  /** Indicates whether a HSL library has been loaded.
   * @return Zero if not loaded, nonzero if handle is loaded
   */
  int LSL_isHSLLoaded();
  
  /** Indicates whether a HSL library is loaded and all symbols necessary to use MA27 have been found.
   * @return Zero if not available, nonzero if MA27 is available in the loaded library.
   */
  int LSL_isMA27available();

  /** Indicates whether a HSL library is loaded and all symbols necessary to use MA28 have been found.
   * @return Zero if not available, nonzero if MA28 is available in the loaded library.
   */
  int LSL_isMA28available();

  /** Indicates whether a HSL library is loaded and all symbols necessary to use MA57 have been found.
   * @return Zero if not available, nonzero if MA57 is available in the loaded library.
   */
  int LSL_isMA57available();

  /** Indicates whether a HSL library is loaded and all symbols necessary to use MA77 have been found.
   * @return Zero if not available, nonzero if HSL_MA77 is available in the loaded library.
   */
  int LSL_isMA77available();

  /** Indicates whether a HSL library is loaded and all symbols necessary to use HSL_MA86 have been found.
   * @return Zero if not available, nonzero if HSL_MA86 is available in the loaded library.
   */
  int LSL_isMA86available();

  /** Indicates whether a HSL library is loaded and all symbols necessary to use HSL_MA97 have been found.
   * @return Zero if not available, nonzero if HSL_MA97 is available in the loaded library.
   */
  int LSL_isMA97available();
  
  /** Indicates whether a HSL library is loaded and all symbols necessary to use MA57 have been found.
   * @return Zero if not available, nonzero if MC19 is available in the loaded library.
   */
  int LSL_isMC19available();
  
  /** Indicates whether a HSL library is loaded and all symbols necessary to use HSL_MC68 have been found.
   * @return Zero if not available, nonzero if MC68 is available in the loaded library.
   */
  int LSL_isMC68available();

  /** Returns name of the shared library that should contain HSL */
  char* LSL_HSLLibraryName();

  /** sets pointers to MA27 functions */
  void LSL_setMA27(ma27ad_t ma27ad, ma27bd_t ma27bd, ma27cd_t ma27cd, ma27id_t ma27id);

  /** sets pointers to MA28 functions */
  void LSL_setMA28(ma28ad_t ma28ad);

  /** sets pointers to MA57 functions */
  void LSL_setMA57(ma57ad_t ma57ad, ma57bd_t ma57bd, ma57cd_t ma57cd, ma57ed_t ma57ed, ma57id_t ma57id);

  /** sets pointers to MA77 functions */
  void LSL_setMA77(ma77_default_control_t ma77_default_control,
     ma77_open_nelt_t ma77_open_nelt,
     ma77_open_t ma77_open,
     ma77_input_vars_t ma77_input_vars,
     ma77_input_reals_t ma77_input_reals,
     ma77_analyse_t ma77_analyse,
     ma77_factor_t ma77_factor,
     ma77_factor_solve_t ma77_factor_solve,
     ma77_solve_t ma77_solve,
     ma77_resid_t ma77_resid,
     ma77_scale_t ma77_scale,
     ma77_enquire_posdef_t ma77_enquire_posdef,
     ma77_enquire_indef_t ma77_enquire_indef,
     ma77_alter_t ma77_alter,
     ma77_restart_t ma77_restart,
     ma77_finalise_t ma77_finalise);

  /** sets pointers to MA86 functions */
  void LSL_setMA86(ma86_default_control_t ma86_default_control,
     ma86_analyse_t ma86_analyse,
     ma86_factor_t ma86_factor,
     ma86_factor_solve_t ma86_factor_solve,
     ma86_solve_t ma86_solve,
     ma86_finalise_t ma86_finalise);

  /** sets pointers to MA97 functions */
  void LSL_setMA97(ma97_default_control_t ma97_default_control,
     ma97_analyse_t ma97_analyse,
     ma97_factor_t ma97_factor,
     ma97_factor_solve_t ma97_factor_solve,
     ma97_solve_t ma97_solve,
     ma97_finalise_t ma97_finalise,
     ma97_free_akeep_t ma97_free_akeep);

  /** sets pointer to MC19 function */
  void LSL_setMC19(mc19ad_t mc19ad);

  /** sets pointers to MC68 functions */
  void LSL_setMC68(mc68_default_control_t mc68_default_control, mc68_order_t mc68_order);

#ifdef __cplusplus
}
#endif

#endif /*HSLLOADER_H_*/
