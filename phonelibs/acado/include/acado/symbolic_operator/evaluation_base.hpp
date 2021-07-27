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
 *    \file include/acado/symbolic_operator/evaluation_base.hpp
 *    \author Boris Houska
 */


#ifndef ACADO_TOOLKIT_EVALUATION_BASE_HPP
#define ACADO_TOOLKIT_EVALUATION_BASE_HPP


#include <acado/symbolic_operator/symbolic_operator_fwd.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Abstract base class for templated evaluation of operators.
 *
 *	\ingroup BasicDataStructures
 *
 *	\author Boris Houska
 */

class EvaluationBase{

public:

	/** Default constructor. */
	EvaluationBase(){};

	virtual ~EvaluationBase(){};

	virtual void addition   ( Operator &arg1, Operator &arg2 ) = 0;
	virtual void subtraction( Operator &arg1, Operator &arg2 ) = 0;
	virtual void product    ( Operator &arg1, Operator &arg2 ) = 0;
	virtual void quotient   ( Operator &arg1, Operator &arg2 ) = 0;
	virtual void power      ( Operator &arg1, Operator &arg2 ) = 0;
	virtual void powerInt   ( Operator &arg1, int      &arg2 ) = 0;

	virtual void project    ( int      &idx ) = 0;
	virtual void set        ( double   &arg ) = 0;
	virtual void Acos       ( Operator &arg ) = 0;
	virtual void Asin       ( Operator &arg ) = 0;
	virtual void Atan       ( Operator &arg ) = 0;
	virtual void Cos        ( Operator &arg ) = 0;
	virtual void Exp        ( Operator &arg ) = 0;
	virtual void Log        ( Operator &arg ) = 0;
	virtual void Sin        ( Operator &arg ) = 0;
	virtual void Tan        ( Operator &arg ) = 0;

};

CLOSE_NAMESPACE_ACADO

#endif
