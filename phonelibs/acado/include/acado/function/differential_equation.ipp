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
 *    \file include/acado/function/differential_equation.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO


inline int DifferentialEquation::getNumDynamicEquations() const{

    return getDim() - getNumAlgebraicEquations();
}


inline int DifferentialEquation::getNumAlgebraicEquations() const{

    return getNXA();
}



inline BooleanType DifferentialEquation::isImplicit( ) const{

    return is_implicit;
}

inline int DifferentialEquation::getStateEnumerationIndex( int index_ ){

    if( counter != 0 ){
       ASSERT( counter == getNumDynamicEquations() );
       ASSERT( index_ < counter );
       return index(VT_DIFFERENTIAL_STATE,component[index_]);
    }
    return index(VT_DIFFERENTIAL_STATE,index_);
}


inline DVector DifferentialEquation::getDifferentialStateComponents() const{

    int run1;
    DVector tmp(getNumDynamicEquations());

    if( counter != 0 ){
        ASSERT( counter == getNumDynamicEquations() );
        for( run1 = 0; run1 < counter; run1++ )
            tmp(run1) = component[run1];
    }
    else{
        for( run1 = 0; run1 < getNumDynamicEquations(); run1++ )
            tmp(run1) = run1;
    }
    return tmp;
}


inline double DifferentialEquation::getStartTime() const{

    if( T1 == 0 ) return t1;
    return -INFTY;
}


inline double DifferentialEquation::getEndTime() const{

    if( T2 == 0 ) return t2;
    return INFTY;
}


inline int DifferentialEquation::getStartTimeIdx() const{

    if( T1 != 0 ) return T1->getComponent(0);
    return -1;
}


inline int DifferentialEquation::getEndTimeIdx() const{

    if( T2 != 0 ) return T2->getComponent(0);
    return -1;
}


inline int* DifferentialEquation::getComponents() const{ return component; }



CLOSE_NAMESPACE_ACADO

// end of file.
