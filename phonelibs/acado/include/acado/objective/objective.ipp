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
 *    \file include/acado/objective/objective.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO




inline returnValue Objective::addMayerTerm( const Expression& arg ){

    nMayer++;
    mayerTerm = (MayerTerm**)realloc(mayerTerm,nMayer*sizeof(MayerTerm*));
    mayerTerm[nMayer-1] = new MayerTerm(grid,arg);
    return SUCCESSFUL_RETURN;
}

inline returnValue Objective::addMayerTerm( const Function& arg ){

    nMayer++;
    mayerTerm = (MayerTerm**)realloc(mayerTerm,nMayer*sizeof(MayerTerm*));
    mayerTerm[nMayer-1] = new MayerTerm(grid,arg);
    return SUCCESSFUL_RETURN;
}


inline int Objective::getNX() const{

    uint run1;
    int n = 0;

    for( run1 = 0; run1 < nLSQ   ; run1++ ) n = acadoMax( lsqTerm   [run1]->getNX() , n );
    for( run1 = 0; run1 < nEndLSQ; run1++ ) n = acadoMax( lsqEndTerm[run1]->getNX() , n );
    for( run1 = 0; run1 < nMayer ; run1++ ) n = acadoMax( mayerTerm [run1]->getNX() , n );

    return n;
}


inline int Objective::getNXA() const{

    uint run1;
    int n = 0;

    for( run1 = 0; run1 < nLSQ   ; run1++ ) n = acadoMax( lsqTerm   [run1]->getNXA() , n );
    for( run1 = 0; run1 < nEndLSQ; run1++ ) n = acadoMax( lsqEndTerm[run1]->getNXA() , n );
    for( run1 = 0; run1 < nMayer ; run1++ ) n = acadoMax( mayerTerm [run1]->getNXA() , n );

    return n;
}


inline int Objective::getNP() const{

    uint run1;
    int n = 0;

    for( run1 = 0; run1 < nLSQ   ; run1++ ) n = acadoMax( lsqTerm   [run1]->getNP() , n );
    for( run1 = 0; run1 < nEndLSQ; run1++ ) n = acadoMax( lsqEndTerm[run1]->getNP() , n );
    for( run1 = 0; run1 < nMayer ; run1++ ) n = acadoMax( mayerTerm [run1]->getNP() , n );

    return n;
}


inline int Objective::getNU() const{

    uint run1;
    int n = 0;

    for( run1 = 0; run1 < nLSQ   ; run1++ ) n = acadoMax( lsqTerm   [run1]->getNU() , n );
    for( run1 = 0; run1 < nEndLSQ; run1++ ) n = acadoMax( lsqEndTerm[run1]->getNU() , n );
    for( run1 = 0; run1 < nMayer ; run1++ ) n = acadoMax( mayerTerm [run1]->getNU() , n );

    return n;
}


inline int Objective::getNW() const{

    uint run1;
    int n = 0;

    for( run1 = 0; run1 < nLSQ   ; run1++ ) n = acadoMax( lsqTerm   [run1]->getNW() , n );
    for( run1 = 0; run1 < nEndLSQ; run1++ ) n = acadoMax( lsqEndTerm[run1]->getNW() , n );
    for( run1 = 0; run1 < nMayer ; run1++ ) n = acadoMax( mayerTerm [run1]->getNW() , n );

    return n;
}


inline BooleanType Objective::hasLSQform(){

    if( lagrangeFcn == 0 && nMayer == 0 ) return BT_TRUE;
    return BT_FALSE;
}


inline BooleanType Objective::isQuadratic(){

    uint run1;

    for( run1 = 0; run1 < nLSQ   ; run1++ ) if( lsqTerm   [run1]->isQuadratic() ) return BT_FALSE;
    for( run1 = 0; run1 < nEndLSQ; run1++ ) if( lsqEndTerm[run1]->isQuadratic() ) return BT_FALSE;
    for( run1 = 0; run1 < nMayer ; run1++ ) return BT_FALSE;

    return BT_TRUE;
}


inline returnValue Objective::setReference( const VariablesGrid &ref ){

    uint run1;
// 	printf( "Objective::setReference: %d, %d\n", nLSQ,nEndLSQ );
    for( run1 = 0; run1 < nLSQ   ; run1++ ) lsqTerm   [run1]->setReference( ref );
    for( run1 = 0; run1 < nEndLSQ; run1++ ) lsqEndTerm[run1]->setReference( ref.getLastVector() );
    for( run1 = 0; run1 < nMayer ; run1++ ) return ACADOERROR( RET_REFERENCE_SHIFTING_WORKS_FOR_LSQ_TERMS_ONLY );

    return SUCCESSFUL_RETURN;
}

CLOSE_NAMESPACE_ACADO

// end of file.
