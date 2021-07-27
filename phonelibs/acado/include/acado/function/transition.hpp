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
 *    \file include/acado/function/transition.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_TRANSITION_HPP
#define ACADO_TOOLKIT_TRANSITION_HPP

#include <acado/function/function_fwd.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to setup and evaluate transition functions based on SymbolicExpressions.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class OutputFcn allows to setup and evaluate transition functions
 *	based on SymbolicExpressions.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class Transition : public Function{


//
// PUBLIC MEMBER FUNCTIONS:
//
public:


    /** Default constructor. */
    Transition( );

    /** Copy constructor (deep copy). */
    Transition( const Transition& arg );

    /** Destructor. */
    virtual ~Transition( );

    /** Assignment operator (deep copy). */
    Transition& operator=( const Transition& arg );


    /** Loading Expressions (deep copy). */
    Transition& operator<<( const DifferentialState& arg       );


    /** Loading Expressions (deep copy). */
    Transition& operator==( const Expression& arg );

    /** Loading Expressions (deep copy). */
    Transition& operator==( const double &arg );

    /** Loading Symbolic DVector (deep copy). */
    Transition& operator==( const DVector& arg );

    /** Loading Symbolic DMatrix (deep copy). */
    Transition& operator==( const DMatrix& arg );



    inline DVector getDifferentialStateComponents() const;




// PROTECTED MEMBERS:
// ------------------

   protected:

       int  counter  ;
       int *component;
};


CLOSE_NAMESPACE_ACADO



#include <acado/function/transition.ipp>


#endif  // ACADO_TOOLKIT_TRANSITION_HPP

// end of file.
