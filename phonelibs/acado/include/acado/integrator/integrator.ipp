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
 *    \file include/acado/integrator/integrator.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date   2009/2010
 */


//
// PUBLIC MEMBER FUNCTIONS:
//

BEGIN_NAMESPACE_ACADO



// ======================================================================================

inline returnValue Integrator::getX( DVector &xEnd_ ) const{

    int run1;
    const int N = rhs->getDim()-ma;

    xEnd_.init(N);
    for( run1 = 0; run1 < N; run1++ )
       xEnd_(run1) = xE(run1);

    return SUCCESSFUL_RETURN;
}


inline returnValue Integrator::getXA( DVector &xaEnd_ ) const{

    int run1;
    const int N = rhs->getDim()-ma;

    xaEnd_.init(ma);
    for( run1 = N; run1 < N+ma; run1++ )
       xaEnd_(run1-N) = xE(run1);

    return SUCCESSFUL_RETURN;
}



inline returnValue Integrator::getX( VariablesGrid &X ) const{

    ASSERT( rhs != 0 );
    uint run1,run2;

    DVector components = rhs->getDifferentialStateComponents();
    X.init( m-ma, xStore.getTimePoints() );

    for( run1 = 0; run1 < X.getNumPoints(); run1++ )
        for( run2 = 0; run2 < X.getNumValues(); run2++ )
            X(run1,(int) components(run2)) = xStore(run1,run2);

    return SUCCESSFUL_RETURN;
}



inline returnValue Integrator::getXA( VariablesGrid &XA ) const{

    uint run1,run2;
    XA.init( ma, xStore.getTimePoints() );

    for( run1 = 0; run1 < XA.getNumPoints(); run1++ )
        for( run2 = 0; run2 < XA.getNumValues(); run2++ )
            XA(run1,run2) = xStore(run1,md+run2);

    return SUCCESSFUL_RETURN;
}


inline returnValue Integrator::getI( VariablesGrid &I ) const{

    I = iStore;
    return SUCCESSFUL_RETURN;
}



CLOSE_NAMESPACE_ACADO

// end of file.
