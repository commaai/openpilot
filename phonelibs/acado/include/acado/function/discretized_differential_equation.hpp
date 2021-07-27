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
 *    \file include/acado/function/discretized_differential_equation.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_DISCRETIZED_DIFFERENTIAL_EQUATION_HPP
#define ACADO_TOOLKIT_DISCRETIZED_DIFFERENTIAL_EQUATION_HPP


#include <acado/function/function_fwd.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to setup and evaluate discretized differential equations based on SymbolicExpressions.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class DiscretizedDifferentialEquation allows to setup and evaluate 
 *	discretized differential equations (ODEs and DAEs) based on SymbolicExpressions.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class DiscretizedDifferentialEquation : public DifferentialEquation
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. If this constructor is used a nonlinear         \n
		*  autonomous discrete time system will be set up, i.e. the class       \n
		*  represtents a system of the form                                     \n
		*                                                                       \n
		*  x_{k+1} = f( x_k, u_k, p, ... )                                      \n
		*                                                                       \n
		*  Here, the function f can be defined as usual with the                \n
		*  << operator which is inherited form the class DifferentialEquation.  \n
		*  Note that this function will in general assume that the function     \n
		*  f  is not explictly time dependent.                                  \n
		*  If this constructor is used, the step length will be 1 by default.   \n
		*                                                                       \n
		*/
		DiscretizedDifferentialEquation( );



		/** Constructor, which is equivalent to the defaul constructor, but the  \n
		*  step length can be defined.                                          \n
		*/
		DiscretizedDifferentialEquation( const double &stepLength_ );


		/** Copy constructor (deep copy). */
		DiscretizedDifferentialEquation( const DiscretizedDifferentialEquation& arg );

		/** Destructor. */
		virtual ~DiscretizedDifferentialEquation( );

		/** Assignment operator (deep copy). */
		DiscretizedDifferentialEquation& operator=( const DiscretizedDifferentialEquation& arg );

		/** Clone constructor (deep copy). */
		virtual DifferentialEquation* clone() const;



	protected:

};


CLOSE_NAMESPACE_ACADO



//#include <acado/function/discretized_differential_equation.ipp>


#endif  // ACADO_TOOLKIT_DISCRETIZED_DIFFERENTIAL_EQUATION_HPP

// end of file.
