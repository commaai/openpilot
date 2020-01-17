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
 *    \file include/acado/objective/lsq_term.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */

BEGIN_NAMESPACE_ACADO



//
// PUBLIC MEMBER FUNCTIONS:
//


returnValue LSQTerm::setGrid( const Grid &grid_ ){

    uint run1, run2, run3;

    grid = grid_;

    if( S_temp != 0 ){
         if( S_temp->getNumPoints() == 1 ){
             DMatrix tmp = S_temp->getMatrix(0);
             S_temp->init( tmp,grid );
         }
         else{
              //printf("%d  %d  \n", grid.getNumPoints(), S_temp->getNumPoints() );
              ASSERT( grid.getNumPoints() == S_temp->getNumPoints() );
         }
    }

    if( r_temp != 0 ){
         if( r_temp->getNumPoints() == 1 ){
             DVector tmp = r_temp->getVector(0);
             r_temp->init( tmp,grid );
         }
         else {ASSERT( grid.getNumPoints() == r_temp->getNumPoints() );}
    }

    if( S_temp != 0 ){

        S = new DMatrix[grid.getNumPoints()];
        for( run1 = 0; run1 < grid.getNumPoints(); run1++ ){
            S[run1] = S_temp->getMatrix(run1);
        }
    }
    else S = 0;

    if( r_temp != 0 ){

        r = new DVector[grid.getNumPoints()];
        for( run1 = 0; run1 < grid.getNumPoints(); run1++ ){
            r[run1] = r_temp->getVector(run1);
            for( run2 = 0; run2 < r[run1].getDim(); run2++ ){
                if( r[run1](run2) >= ACADO_NAN - 1.0 ){
                    if( S == 0 ){
                        S = new DMatrix[grid.getNumPoints()];
                        for( run3 = 0; run3 < grid.getNumPoints(); run3++ ){
                            S[run3].init( r[run1].getDim(), r[run1].getDim() );
                            S[run3].setIdentity();
                        }
                    }
                    for( run3 = 0; run3 < r[run1].getDim(); run3++ ){
                        S[run1]( run2, run3 ) = 0.0;
                        S[run1]( run3, run2 ) = 0.0;
                    }
                }
            }
        }
    }
    else r = 0;

    S_h_res = new double*[grid.getNumPoints()];

    for( run1 = 0; run1 < grid.getNumPoints(); run1++ )
        S_h_res[run1] = new double[fcn.getDim()];

    return SUCCESSFUL_RETURN;
}



// inline BooleanType isAffine(){
//
// }


inline BooleanType LSQTerm::isQuadratic(){

    if( fcn.isSymbolic() == BT_FALSE ) return BT_FALSE;
    if( fcn.isAffine()   == BT_FALSE ) return BT_FALSE;
    return BT_FALSE;
}




CLOSE_NAMESPACE_ACADO

// end of file.
