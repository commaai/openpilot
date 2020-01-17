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
 *    \file include/acado/symbolic_expression/expression.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO



inline unsigned int Expression::getDim( ) const{

    return dim;
}

inline unsigned int Expression::getNumRows( ) const{

    return nRows;
}

inline unsigned int Expression::getNumCols( ) const{

    return nCols;
}

inline unsigned int Expression::getComponent( const unsigned int idx ) const{

    ASSERT( idx < getDim() );
    return component + idx;
}

inline VariableType Expression::getVariableType( ) const{

    return variableType;
}

inline BooleanType Expression::isVariable( ) const{

   if( getVariableType() != VT_DIFFERENTIAL_STATE &&
       getVariableType() != VT_ALGEBRAIC_STATE &&
       getVariableType() != VT_CONTROL &&
       getVariableType() != VT_INTEGER_CONTROL &&
       getVariableType() != VT_PARAMETER &&
       getVariableType() != VT_INTEGER_PARAMETER &&
       getVariableType() != VT_DISTURBANCE &&
       getVariableType() != VT_TIME &&
       getVariableType() != VT_DDIFFERENTIAL_STATE &&
       getVariableType() != VT_VARIABLE ) return BT_FALSE;

    return BT_TRUE;
}

CLOSE_NAMESPACE_ACADO

// end of file.
