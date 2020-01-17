/*
 *
 *  This file is part of MUMPS 4.10.0, built on Tue May 10 12:56:32 UTC 2011
 *
 *
 *  This version of MUMPS is provided to you free of charge. It is public
 *  domain, based on public domain software developed during the Esprit IV
 *  European project PARASOL (1996-1999). Since this first public domain
 *  version in 1999, research and developments have been supported by the
 *  following institutions: CERFACS, CNRS, ENS Lyon, INPT(ENSEEIHT)-IRIT,
 *  INRIA, and University of Bordeaux.
 *
 *  The MUMPS team at the moment of releasing this version includes
 *  Patrick Amestoy, Maurice Bremond, Alfredo Buttari, Abdou Guermouche,
 *  Guillaume Joslin, Jean-Yves L'Excellent, Francois-Henry Rouet, Bora
 *  Ucar and Clement Weisbecker.
 *
 *  We are also grateful to Emmanuel Agullo, Caroline Bousquet, Indranil
 *  Chowdhury, Philippe Combes, Christophe Daniel, Iain Duff, Vincent Espirat,
 *  Aurelia Fevre, Jacko Koster, Stephane Pralet, Chiara Puglisi, Gregoire
 *  Richard, Tzvetomila Slavova, Miroslav Tuma and Christophe Voemel who
 *  have been contributing to this project.
 *
 *  Up-to-date copies of the MUMPS package can be obtained
 *  from the Web pages:
 *  http://mumps.enseeiht.fr/  or  http://graal.ens-lyon.fr/MUMPS
 *
 *
 *   THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
 *   EXPRESSED OR IMPLIED. ANY USE IS AT YOUR OWN RISK.
 *
 *
 *  User documentation of any code that uses this software can
 *  include this complete notice. You can acknowledge (using
 *  references [1] and [2]) the contribution of this package
 *  in any scientific publication dependent upon the use of the
 *  package. You shall use reasonable endeavours to notify
 *  the authors of the package of this publication.
 *
 *   [1] P. R. Amestoy, I. S. Duff, J. Koster and  J.-Y. L'Excellent,
 *   A fully asynchronous multifrontal solver using distributed dynamic
 *   scheduling, SIAM Journal of Matrix Analysis and Applications,
 *   Vol 23, No 1, pp 15-41 (2001).
 *
 *   [2] P. R. Amestoy and A. Guermouche and J.-Y. L'Excellent and
 *   S. Pralet, Hybrid scheduling for the parallel solution of linear
 *   systems. Parallel Computing Vol 32 (2), pp 136-156 (2006).
 *
 */

/* Mostly written in march 2002 (JYL) */

#ifndef DMUMPS_C_H
#define DMUMPS_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mumps_compat.h"
/* Next line defines MUMPS_INT, DMUMPS_COMPLEX and DMUMPS_REAL */
#include "mumps_c_types.h"

#ifndef MUMPS_VERSION
/* Protected in case headers of other arithmetics are included */
#define MUMPS_VERSION "4.10.0"
#endif
#ifndef MUMPS_VERSION_MAX_LEN
#define MUMPS_VERSION_MAX_LEN 14
#endif

/*
 * Definition of the (simplified) MUMPS C structure.
 * NB: DMUMPS_COMPLEX are REAL types in s and d arithmetics.
 */
typedef struct {

    MUMPS_INT      sym, par, job;
    MUMPS_INT      comm_fortran;    /* Fortran communicator */
    MUMPS_INT      icntl[40];
    DMUMPS_REAL    cntl[15];
    MUMPS_INT      n;

    MUMPS_INT      nz_alloc; /* used in matlab interface to decide if we
                                free + malloc when we have large variation */

    /* Assembled entry */
    MUMPS_INT      nz;
    MUMPS_INT      *irn;
    MUMPS_INT      *jcn;
    DMUMPS_COMPLEX *a;

    /* Distributed entry */
    MUMPS_INT      nz_loc;
    MUMPS_INT      *irn_loc;
    MUMPS_INT      *jcn_loc;
    DMUMPS_COMPLEX *a_loc;

    /* Element entry */
    MUMPS_INT      nelt;
    MUMPS_INT      *eltptr;
    MUMPS_INT      *eltvar;
    DMUMPS_COMPLEX *a_elt;

    /* Ordering, if given by user */
    MUMPS_INT      *perm_in;

    /* Orderings returned to user */
    MUMPS_INT      *sym_perm;    /* symmetric permutation */
    MUMPS_INT      *uns_perm;    /* column permutation */

    /* Scaling (input only in this version) */
    DMUMPS_REAL    *colsca;
    DMUMPS_REAL    *rowsca;

    /* RHS, solution, ouptput data and statistics */
    DMUMPS_COMPLEX *rhs, *redrhs, *rhs_sparse, *sol_loc;
    MUMPS_INT      *irhs_sparse, *irhs_ptr, *isol_loc;
    MUMPS_INT      nrhs, lrhs, lredrhs, nz_rhs, lsol_loc;
    MUMPS_INT      schur_mloc, schur_nloc, schur_lld;
    MUMPS_INT      mblock, nblock, nprow, npcol;
    MUMPS_INT      info[40],infog[40];
    DMUMPS_REAL    rinfo[40], rinfog[40];

    /* Null space */
    MUMPS_INT      deficiency;
    MUMPS_INT      *pivnul_list;
    MUMPS_INT      *mapping;

    /* Schur */
    MUMPS_INT      size_schur;
    MUMPS_INT      *listvar_schur;
    DMUMPS_COMPLEX *schur;

    /* Internal parameters */
    MUMPS_INT      instance_number;
    DMUMPS_COMPLEX *wk_user;

    /* Version number: length=14 in FORTRAN + 1 for final \0 + 1 for alignment */
    char version_number[MUMPS_VERSION_MAX_LEN + 1 + 1];
    /* For out-of-core */
    char ooc_tmpdir[256];
    char ooc_prefix[64];
    /* To save the matrix in matrix market format */
    char write_problem[256];
    MUMPS_INT      lwk_user;

} DMUMPS_STRUC_C;


void MUMPS_CALL
dmumps_c( DMUMPS_STRUC_C * dmumps_par );

#ifdef __cplusplus
}
#endif

#endif /* DMUMPS_C_H */

