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
 *    \file include/acado/symbolic_expression/acado_syntax.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_SYNTAX_HPP
#define ACADO_TOOLKIT_SYNTAX_HPP

#include <acado/symbolic_expression/expression.hpp>
#include <acado/symbolic_expression/variable_types.hpp>

/** Syntax of the ACADO toolkit symbolic core.
 *
 *  \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 *
 *  This list is a forward declaration of functions that operate on Expressions.
 *  (note that this function are not in a namespace.) It defines the syntax of
 *  the ACADO Toolkit.
 *
 */

/** \name Standard math operators
 *  @{
 */

REFER_NAMESPACE_ACADO IntermediateState sin ( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState cos ( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState tan ( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState asin( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState acos( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState atan( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState exp ( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState sqrt( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState ln  ( const REFER_NAMESPACE_ACADO Expression &arg   );
REFER_NAMESPACE_ACADO IntermediateState log ( const REFER_NAMESPACE_ACADO Expression &arg   );


REFER_NAMESPACE_ACADO IntermediateState pow (	const REFER_NAMESPACE_ACADO Expression &arg1,
                 	 	 	 	 	 	const REFER_NAMESPACE_ACADO Expression &arg2  );
REFER_NAMESPACE_ACADO IntermediateState pow (	const double     &arg1,
                 	 	 	 	 	 	const REFER_NAMESPACE_ACADO Expression &arg2  );
REFER_NAMESPACE_ACADO IntermediateState pow (	const REFER_NAMESPACE_ACADO Expression &arg1,
                 	 	 	 	 	 	const double     &arg2  );

/** @} */

/** \name Special convex disciplined programming functions.
 *  @{
 */

REFER_NAMESPACE_ACADO IntermediateState square         ( const REFER_NAMESPACE_ACADO Expression &arg );
REFER_NAMESPACE_ACADO IntermediateState sum_square     ( const REFER_NAMESPACE_ACADO Expression &arg );
REFER_NAMESPACE_ACADO IntermediateState log_sum_exp    ( const REFER_NAMESPACE_ACADO Expression &arg );
REFER_NAMESPACE_ACADO IntermediateState euclidean_norm ( const REFER_NAMESPACE_ACADO Expression &arg );
REFER_NAMESPACE_ACADO IntermediateState entropy        ( const REFER_NAMESPACE_ACADO Expression &arg );

/** @} */


/** \name Special routines for the set up of dynamic systems.
 *  @{
 */

REFER_NAMESPACE_ACADO Expression dot ( const REFER_NAMESPACE_ACADO Expression& arg );
REFER_NAMESPACE_ACADO Expression next( const REFER_NAMESPACE_ACADO Expression& arg );

/** @} */

/** \name Symbolic derivative operators.
 *  @{
 */

REFER_NAMESPACE_ACADO Expression forwardDerivative ( const REFER_NAMESPACE_ACADO Expression &arg1,
                                                     const REFER_NAMESPACE_ACADO Expression &arg2 );

REFER_NAMESPACE_ACADO Expression backwardDerivative( const REFER_NAMESPACE_ACADO Expression &arg1,
                                                     const REFER_NAMESPACE_ACADO Expression &arg2 );

REFER_NAMESPACE_ACADO Expression forwardDerivative ( const REFER_NAMESPACE_ACADO Expression &arg1,
                                                     const REFER_NAMESPACE_ACADO Expression &arg2,
                                                     const REFER_NAMESPACE_ACADO Expression &seed );

REFER_NAMESPACE_ACADO Expression backwardDerivative( const REFER_NAMESPACE_ACADO Expression &arg1,
                                                     const REFER_NAMESPACE_ACADO Expression &arg2,
                                                     const REFER_NAMESPACE_ACADO Expression &seed );

REFER_NAMESPACE_ACADO Expression multipleForwardDerivative ( const REFER_NAMESPACE_ACADO Expression &arg1,
                                                     	 	 	 const REFER_NAMESPACE_ACADO Expression &arg2,
                                                     	 	 	 const REFER_NAMESPACE_ACADO Expression &seed );

REFER_NAMESPACE_ACADO Expression multipleBackwardDerivative ( const REFER_NAMESPACE_ACADO Expression &arg1,
                                                     	 	 	 const REFER_NAMESPACE_ACADO Expression &arg2,
                                                     	 	 	 const REFER_NAMESPACE_ACADO Expression &seed );

REFER_NAMESPACE_ACADO Expression symmetricDerivative( 	const REFER_NAMESPACE_ACADO Expression &arg1,
 	 	 	 	 	 	 	 	 	 	 	 	 	 		const REFER_NAMESPACE_ACADO Expression &arg2,
 	 	 	 	 	 	 	 	 	 	 	 	 	 		const REFER_NAMESPACE_ACADO Expression &forward_seed,
 	 	 	 	 	 	 	 	 	 	 	 	 	 		const REFER_NAMESPACE_ACADO Expression &backward_seed,
 	 	 	 	 	 	 	 	 	 	 	 	 	 		REFER_NAMESPACE_ACADO Expression *forward_result = 0,
 	 	 	 	 	 	 	 	 	 	 	 	 	 		REFER_NAMESPACE_ACADO Expression *backward_result = 0 );

REFER_NAMESPACE_ACADO Expression jacobian           ( const REFER_NAMESPACE_ACADO Expression &arg1,
                                                      const REFER_NAMESPACE_ACADO Expression &arg2 );

REFER_NAMESPACE_ACADO Expression laplace           ( const REFER_NAMESPACE_ACADO Expression &arg1,
                                                     const REFER_NAMESPACE_ACADO Expression &arg2 );


REFER_NAMESPACE_ACADO Expression getRiccatiODE( const REFER_NAMESPACE_ACADO Expression        &rhs,
                                                const REFER_NAMESPACE_ACADO DifferentialState &x  ,
                                                const REFER_NAMESPACE_ACADO Control           &u  ,
                                                const REFER_NAMESPACE_ACADO DifferentialState &P  ,
                                                const REFER_NAMESPACE_ACADO DMatrix            &Q  ,
                                                const REFER_NAMESPACE_ACADO DMatrix            &R   );


REFER_NAMESPACE_ACADO Expression chol( const REFER_NAMESPACE_ACADO Expression &arg );


/** Function which clears all the static counters, used throughout ACADO symbolics. */
REFER_NAMESPACE_ACADO returnValue clearAllStaticCounters();


/** @} */

#endif  // ACADO_TOOLKIT_SYNTAX_HPP
