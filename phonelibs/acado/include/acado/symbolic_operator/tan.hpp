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
*    \file include/acado/symbolic_operator/tan.hpp
*    \author Boris Houska, Hans Joachim Ferreau
*    \date 2008
*/


#ifndef ACADO_TOOLKIT_TAN_HPP
#define ACADO_TOOLKIT_TAN_HPP


#include <acado/symbolic_operator/symbolic_operator_fwd.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Implements the scalar tangens operator within the symbolic operators family.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class Tan implements the scalar tangens operator within the 
 *	symbolic operators family.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class Tan : public UnaryOperator{

public:

    /** Default constructor. */
    Tan();

    /** Default constructor. */
    Tan( Operator *_argument );

    /** Copy constructor (deep copy). */
    Tan( const Tan &arg );

    /** Default destructor. */
    ~Tan();

    /** Assignment Operator (deep copy). */
    Tan& operator=( const Tan &arg );


	/** Evaluates the expression (templated version) */
	virtual returnValue evaluate( EvaluationBase *x );



    /** Substitutes var(index) with the expression sub.           \n
     *  \return The substituted expression.                       \n
     *
     */
     virtual Operator* substitute( int   index           /**< subst. index    */,
                                     const Operator *sub /**< the substitution*/);


     /** Provides a deep copy of the expression. \n
      *  \return a clone of the expression.      \n
      */
     virtual Operator* clone() const;

     virtual returnValue initDerivative();



//
//  PROTECTED FUNCTIONS:
//

protected:
};


CLOSE_NAMESPACE_ACADO



#endif
