/*
 *    This file is part of ACADO Toolkit.
 *
 *    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
 *    Copyright (C) 2008-2014 by Boris Houska, Hans Joachim Ferreau,
 *    Milan Vukov, Rien Quirynen, KU Leuven.
 *    Developed within the Optimization in Engineering Center (OPTEC)
 *    under supervision of Moritz Diehl. All rights reserved.
 *
 *    ACADO Toolkit is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with ACADO Toolkit; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *    \file include/acado/utils/acado_constants.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 *    \date 2008 - 2013
 *
 *    This file collects all global constants used within the toolkit.
 */

#ifndef ACADO_TOOLKIT_ACADO_CONSTANTS_HPP
#define ACADO_TOOLKIT_ACADO_CONSTANTS_HPP

#include <acado/utils/acado_types.hpp>

BEGIN_NAMESPACE_ACADO

/** Numerical value of machine precision (min eps, s.t. 1+eps > 1). */
const double EPS = 2.221e-16;

/** Numerical value of the zero bound that checked for every division. */
const double ZERO_EPS = 2.221e-16;

/** Numerical value which is used for numerical equality checks. */
const double EQUALITY_EPS = 1.0e-12;

/** Numerical value of sqrt of EPS. */
const double SQRT_EPS = 1.490301982821e-08;

/** Numerical value of the third root of EPS. */
const double THIRD_ROOT_EPS = 6.055957976542e-06;

/** Numerical value of the fourth root of EPS. */
const double FOURTH_ROOT_EPS = 1.22077925228967e-04;

/** Numerical value of zero (for situations in which it would be
 *	unreasonable to compare with 0.0). */
const double ZERO = 1.0e-50;

/** Numerical value of infinity (e.g. for non-existing bounds). */
const double INFTY = 1.0e12;

/** Algebraic value of infinity. \n
 ** (for situations in which the use of "ACADO::INFTY" is critical). */
#ifndef INFINITY
const double INFINITY = 1.0/1e-53;
#endif

/** Numerical value of NAN (not a nummber). */
const double ACADO_NAN = INFTY;

/** Lower/upper (constraints') bound tolerance (an inequality constraint
 *	whose lower and upper bound differ by less than BOUNDTOL is regarded
 *	to be an equality constraint). */
const double BOUNDTOL = 1.0e-10;

/** Offset for relaxing (constraints') bounds at beginning of
 *  an initial homotopy.
 *  Note: this value has to be positive! */
const double BOUNDRELAXATION = 1.0e3;

/** Default sampling time for blocks of the simulation environment. */
const double DEFAULT_SAMPLING_TIME = 1.0;

/** Maximum length of a string. */
const unsigned int MAX_LENGTH_STRING = 1024;

/** Maximum length of a name or unit. */
const unsigned int MAX_LENGTH_NAME = 80;

CLOSE_NAMESPACE_ACADO

#endif	// ACADO_TOOLKIT_ACADO_CONSTANTS_HPP

/*
 *	end of file
 */
