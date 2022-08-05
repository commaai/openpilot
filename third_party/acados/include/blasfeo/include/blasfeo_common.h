/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#ifndef BLASFEO_COMMON_H_
#define BLASFEO_COMMON_H_



#include "blasfeo_target.h"



#ifdef __cplusplus
extern "C" {
#endif



#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__INTEL_LLVM_COMPILER)
#define ALIGNED(VEC, BYTES) VEC __attribute__ ((aligned ( BYTES )))
#elif defined (_MSC_VER)
#define ALIGNED(VEC, BYTES) __declspec(align( BYTES )) VEC
#else
#define ALIGNED(VEC, BYTES) VEC
#endif




#if ( defined(LA_HIGH_PERFORMANCE) & defined(MF_PANELMAJ) ) | ( defined(LA_REFERENCE) & defined(MF_PANELMAJ) )

#include "blasfeo_block_size.h"

// matrix structure
struct blasfeo_dmat
	{
	double *mem; // pointer to passed chunk of memory
	double *pA; // pointer to a pm*pn array of doubles, the first is aligned to cache line size
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

struct blasfeo_smat
	{
	float *mem; // pointer to passed chunk of memory
	float *pA; // pointer to a pm*pn array of floats, the first is aligned to cache line size
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

// vector structure
struct blasfeo_dvec
	{
	double *mem; // pointer to passed chunk of memory
	double *pa; // pointer to a pm array of doubles, the first is aligned to cache line size
	int m; // size
	int pm; // packed size
	int memsize; // size of needed memory
	};

struct blasfeo_svec
	{
	float *mem; // pointer to passed chunk of memory
	float *pa; // pointer to a pm array of floats, the first is aligned to cache line size
	int m; // size
	int pm; // packed size
	int memsize; // size of needed memory
	};

#define BLASFEO_DMATEL(sA,ai,aj) ((sA)->pA[((ai)-((ai)&(D_PS-1)))*(sA)->cn+(aj)*D_PS+((ai)&(D_PS-1))])
#define BLASFEO_SMATEL(sA,ai,aj) ((sA)->pA[((ai)-((ai)&(S_PS-1)))*(sA)->cn+(aj)*S_PS+((ai)&(S_PS-1))])
#define BLASFEO_DVECEL(sa,ai) ((sa)->pa[ai])
#define BLASFEO_SVECEL(sa,ai) ((sa)->pa[ai])

#elif ( defined(LA_HIGH_PERFORMANCE) & defined(MF_COLMAJ) ) | ( defined(LA_REFERENCE) & defined(MF_COLMAJ) ) | defined(LA_EXTERNAL_BLAS_WRAPPER)

// matrix structure
struct blasfeo_dmat
	{
	double *mem; // pointer to passed chunk of memory
	double *pA; // pointer to a m*n array of doubles
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int m; // rows
	int n; // cols
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

struct blasfeo_smat
	{
	float *mem; // pointer to passed chunk of memory
	float *pA; // pointer to a m*n array of floats
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int m; // rows
	int n; // cols
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

// vector structure
struct blasfeo_dvec
	{
	double *mem; // pointer to passed chunk of memory
	double *pa; // pointer to a m array of doubles, the first is aligned to cache line size
	int m; // size
	int memsize; // size of needed memory
	};

struct blasfeo_svec
	{
	float *mem; // pointer to passed chunk of memory
	float *pa; // pointer to a m array of floats, the first is aligned to cache line size
	int m; // size
	int memsize; // size of needed memory
	};

#define BLASFEO_DMATEL(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#define BLASFEO_SMATEL(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#define BLASFEO_DVECEL(sa,ai) ((sa)->pa[ai])
#define BLASFEO_SVECEL(sa,ai) ((sa)->pa[ai])

#else

#error : wrong LA or MF choice

#endif



// Explicitly panel-major matrix structure
struct blasfeo_pm_dmat
	{
	double *mem; // pointer to passed chunk of memory
	double *pA; // pointer to a pm*pn array of doubles, the first is aligned to cache line size
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	int use_dA; // flag to tell if dA can be used
	int ps; // panel size
	int memsize; // size of needed memory
	};

struct blasfeo_pm_smat
	{
	float *mem; // pointer to passed chunk of memory
	float *pA; // pointer to a pm*pn array of floats, the first is aligned to cache line size
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	int use_dA; // flag to tell if dA can be used
	int ps; // panel size
	int memsize; // size of needed memory
	};

struct blasfeo_pm_dvec
	{
	double *mem; // pointer to passed chunk of memory
	double *pa; // pointer to a pm array of doubles, the first is aligned to cache line size
	int m; // size
	int pm; // packed size
	int memsize; // size of needed memory
	};

struct blasfeo_pm_svec
	{
	float *mem; // pointer to passed chunk of memory
	float *pa; // pointer to a pm array of floats, the first is aligned to cache line size
	int m; // size
	int pm; // packed size
	int memsize; // size of needed memory
	};

// Explicitly column-major matrix structure
struct blasfeo_cm_dmat
	{
	double *mem; // pointer to passed chunk of memory
	double *pA; // pointer to a m*n array of doubles
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int m; // rows
	int n; // cols
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

struct blasfeo_cm_smat
	{
	float *mem; // pointer to passed chunk of memory
	float *pA; // pointer to a m*n array of floats
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int m; // rows
	int n; // cols
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

struct blasfeo_cm_dvec
	{
	double *mem; // pointer to passed chunk of memory
	double *pa; // pointer to a m array of doubles, the first is aligned to cache line size
	int m; // size
	int memsize; // size of needed memory
	};

struct blasfeo_cm_svec
	{
	float *mem; // pointer to passed chunk of memory
	float *pa; // pointer to a m array of floats, the first is aligned to cache line size
	int m; // size
	int memsize; // size of needed memory
	};


#define BLASFEO_PM_DMATEL(sA,ai,aj) ((sA)->pA[((ai)-((ai)&((sA)->ps-1)))*(sA)->cn+(aj)*((sA)->ps)+((ai)&((sA)->ps-1))])
#define BLASFEO_PM_SMATEL(sA,ai,aj) ((sA)->pA[((ai)-((ai)&((sA)->ps-1)))*(sA)->cn+(aj)*((sA)->ps)+((ai)&((sA)->ps-1))])
#define BLASFEO_PM_DVECEL(sa,ai) ((sa)->pa[ai])
#define BLASFEO_PM_SVECEL(sa,ai) ((sa)->pa[ai])
#define BLASFEO_CM_DMATEL(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#define BLASFEO_CM_SMATEL(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#define BLASFEO_CM_DVECEL(sa,ai) ((sa)->pa[ai])
#define BLASFEO_CM_SVECEL(sa,ai) ((sa)->pa[ai])



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_COMMON_H_
