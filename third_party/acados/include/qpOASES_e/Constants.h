/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2015 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file include/qpOASES_e/Constants.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Definition of all global constants.
 */


#ifndef QPOASES_CONSTANTS_H
#define QPOASES_CONSTANTS_H


#include <qpOASES_e/Types.h>

#ifdef __CODE_GENERATION__

  #define CONVERTTOSTRINGAUX(x) #x
  #define CONVERTTOSTRING(x) CONVERTTOSTRINGAUX(x)

  #ifndef QPOASES_CUSTOM_INTERFACE
  #include "acado_qpoases3_interface.h"
  #else
  #include CONVERTTOSTRING(QPOASES_CUSTOM_INTERFACE)
  #endif

#endif


BEGIN_NAMESPACE_QPOASES


#ifndef __EXTERNAL_DIMENSIONS__

  /*#define QPOASES_NVMAX  50
  #define QPOASES_NCMAX  100*/
  #define QPOASES_NVMAX  287
  #define QPOASES_NCMAX  709

#endif /* __EXTERNAL_DIMENSIONS__ */


/** Maximum number of variables within a QP formulation.
 *	Note: this value has to be positive! */
#define NVMAX  QPOASES_NVMAX

/** Maximum number of constraints within a QP formulation.
 *	Note: this value has to be positive! */
#define NCMAX  QPOASES_NCMAX

#if ( QPOASES_NVMAX > QPOASES_NCMAX )
#define NVCMAX QPOASES_NVMAX
#else
#define NVCMAX QPOASES_NCMAX
#endif

#if ( QPOASES_NVMAX > QPOASES_NCMAX )
#define NVCMIN QPOASES_NCMAX
#else
#define NVCMIN QPOASES_NVMAX
#endif


/** Maximum number of QPs in a sequence solved by means of the OQP interface.
 *	Note: this value has to be positive! */
#define NQPMAX 1000


/** Numerical value of machine precision (min eps, s.t. 1+eps > 1).
 *	Note: this value has to be positive! */
#ifndef __CODE_GENERATION__

  #ifdef __USE_SINGLE_PRECISION__
  static const real_t QPOASES_EPS = 1.193e-07;
  #else
  static const real_t QPOASES_EPS = 2.221e-16;
  #endif /* __USE_SINGLE_PRECISION__ */

#endif /* __CODE_GENERATION__ */


/** Numerical value of zero (for situations in which it would be
 *	unreasonable to compare with 0.0).
 *  Note: this value has to be positive! */
static const real_t QPOASES_ZERO = 1.0e-25;

/** Numerical value of infinity (e.g. for non-existing bounds).
 *	Note: this value has to be positive! */
static const real_t QPOASES_INFTY = 1.0e20;

/** Tolerance to used for isEqual, isZero etc.
 *	Note: this value has to be positive! */
static const real_t QPOASES_TOL = 1.0e-25;


/** Maximum number of characters within a string.
 *	Note: this value should be at least 41! */
#define QPOASES_MAX_STRING_LENGTH 160


END_NAMESPACE_QPOASES


#endif	/* QPOASES_CONSTANTS_H */


/*
 *	end of file
 */
