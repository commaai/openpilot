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
 *    \file include/acado/curve/curve.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 31.05.2008
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO


inline int Curve::getDim( ) const{

    if( isEmpty() == BT_TRUE ) return 0;
    return dim;
}


inline BooleanType Curve::isEmpty( ) const{

    if( getNumIntervals() == 0 ) return BT_TRUE;
    return BT_FALSE;
}


inline BooleanType Curve::isContinuous( ) const{

    ASSERT( isEmpty() == BT_FALSE );

    ACADOERROR(RET_NOT_IMPLEMENTED_YET);
    ASSERT( 1 == 0 );

    return BT_TRUE;
}



inline int Curve::getNumIntervals() const{

    return nIntervals;
}



CLOSE_NAMESPACE_ACADO

/*
 *    end of file
 */
