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
 *    \file include/acado/objective/lagrange_term.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO




inline const Grid& LagrangeTerm::getGrid() const{

    return grid;
}


inline returnValue LagrangeTerm::addLagrangeTerm( const Expression& arg ){

    if( lagrangeFcn != 0 ){
        *lagrangeFcn[0] += arg;
    }
    else{
        nLagrangeTerms = 1;
        lagrangeFcn = (Expression**)realloc(lagrangeFcn,nLagrangeTerms*sizeof(Expression*));
        lagrangeFcn[0] = new Expression(arg);
    }

    return SUCCESSFUL_RETURN;
}


inline returnValue LagrangeTerm::addLagrangeTerm( const Expression& arg,
                                                  const int& stageNumber ){

    int run1;

    if( nLagrangeTerms > stageNumber ){

        *lagrangeFcn[stageNumber] = *lagrangeFcn[stageNumber] + arg;
    }
    else{

        lagrangeFcn = (Expression**)realloc(lagrangeFcn,(stageNumber+1)*sizeof(Expression*));

        for( run1 = nLagrangeTerms; run1 < stageNumber+1; run1++ )
            lagrangeFcn[run1] = 0;

        nLagrangeTerms = stageNumber+1;
        lagrangeFcn[stageNumber] = new Expression( arg );
    }

    return SUCCESSFUL_RETURN;
}


CLOSE_NAMESPACE_ACADO

// end of file.
