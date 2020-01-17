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
 *    \file include/acado/constraint/constraint.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//



BEGIN_NAMESPACE_ACADO


inline Grid& Constraint::getGrid(){

    return grid;
}


inline int Constraint::getNC(){

    uint run1;
    int nc = 0;

    if( boundary_constraint              != 0 )  nc += boundary_constraint              ->getNC();
    if( coupled_path_constraint          != 0 )  nc += coupled_path_constraint          ->getNC();
    if( path_constraint                  != 0 )  nc += path_constraint                  ->getNC();
    if( algebraic_consistency_constraint != 0 )  nc += algebraic_consistency_constraint ->getNC();
    if( point_constraints                != 0 )
        for( run1 = 0; run1 < grid.getNumPoints(); run1++ )
            if( point_constraints[run1] != 0 )  nc += point_constraints[run1]->getNC();

    return nc;
}


inline int Constraint::getNX() const{

    uint run1;
    int n = 0;

    if( boundary_constraint              != 0 )  n = acadoMax( boundary_constraint              ->getNX() , n );
    if( coupled_path_constraint          != 0 )  n = acadoMax( coupled_path_constraint          ->getNX() , n );
    if( path_constraint                  != 0 )  n = acadoMax( path_constraint                  ->getNX() , n );
    if( algebraic_consistency_constraint != 0 )  n = acadoMax( algebraic_consistency_constraint ->getNX() , n );
    if( point_constraints                != 0 )
        for( run1 = 0; run1 < grid.getNumPoints(); run1++ )
            if( point_constraints[run1] != 0 )  n = acadoMax( point_constraints[run1]->getNX() , n );

    return n;
}


inline int Constraint::getNXA() const{

    uint run1;
    int n = 0;

    if( boundary_constraint              != 0 )  n = acadoMax( boundary_constraint              ->getNXA() , n );
    if( coupled_path_constraint          != 0 )  n = acadoMax( coupled_path_constraint          ->getNXA() , n );
    if( path_constraint                  != 0 )  n = acadoMax( path_constraint                  ->getNXA() , n );
    if( algebraic_consistency_constraint != 0 )  n = acadoMax( algebraic_consistency_constraint ->getNXA() , n );
    if( point_constraints                != 0 )
        for( run1 = 0; run1 < grid.getNumPoints(); run1++ )
            if( point_constraints[run1] != 0 )  n = acadoMax( point_constraints[run1]->getNXA() , n );

    return n;
}


inline int Constraint::getNP() const{

    uint run1;
    int n = 0;

    if( boundary_constraint              != 0 )  n = acadoMax( boundary_constraint              ->getNP() , n );
    if( coupled_path_constraint          != 0 )  n = acadoMax( coupled_path_constraint          ->getNP() , n );
    if( path_constraint                  != 0 )  n = acadoMax( path_constraint                  ->getNP() , n );
    if( algebraic_consistency_constraint != 0 )  n = acadoMax( algebraic_consistency_constraint ->getNP() , n );
    if( point_constraints                != 0 )
        for( run1 = 0; run1 < grid.getNumPoints(); run1++ )
            if( point_constraints[run1] != 0 )  n = acadoMax( point_constraints[run1]->getNP() , n );

    return n;
}


inline int Constraint::getNU() const{

    uint run1;
    int n = 0;

    if( boundary_constraint              != 0 )  n = acadoMax( boundary_constraint              ->getNU() , n );
    if( coupled_path_constraint          != 0 )  n = acadoMax( coupled_path_constraint          ->getNU() , n );
    if( path_constraint                  != 0 )  n = acadoMax( path_constraint                  ->getNU() , n );
    if( algebraic_consistency_constraint != 0 )  n = acadoMax( algebraic_consistency_constraint ->getNU() , n );
    if( point_constraints                != 0 )
        for( run1 = 0; run1 < grid.getNumPoints(); run1++ )
            if( point_constraints[run1] != 0 )  n = acadoMax( point_constraints[run1]->getNU() , n );

    return n;
}


inline int Constraint::getNW() const{

    uint run1;
    int n = 0;

    if( boundary_constraint              != 0 )  n = acadoMax( boundary_constraint              ->getNW() , n );
    if( coupled_path_constraint          != 0 )  n = acadoMax( coupled_path_constraint          ->getNW() , n );
    if( path_constraint                  != 0 )  n = acadoMax( path_constraint                  ->getNW() , n );
    if( algebraic_consistency_constraint != 0 )  n = acadoMax( algebraic_consistency_constraint ->getNW() , n );
    if( point_constraints                != 0 )
        for( run1 = 0; run1 < grid.getNumPoints(); run1++ )
            if( point_constraints[run1] != 0 )  n = acadoMax( point_constraints[run1]->getNW() , n );

    return n;
}


inline BooleanType Constraint::isAffine() const{

    if( boundary_constraint              ->isAffine() == BT_FALSE )  return BT_FALSE;
    if( coupled_path_constraint          ->isAffine() == BT_FALSE )  return BT_FALSE;
    if( path_constraint                  ->isAffine() == BT_FALSE )  return BT_FALSE;
    if( algebraic_consistency_constraint ->isAffine() == BT_FALSE )  return BT_FALSE;
    if( point_constraints != 0 )
        for( uint run1 = 0; run1 < grid.getNumPoints(); run1++ )
            if( point_constraints[run1] != 0 )
                if( point_constraints[run1]->isAffine() == BT_FALSE ) return BT_FALSE;

    return BT_TRUE;
}


inline BooleanType Constraint::isBoxConstraint() const
{
	if( boundary_constraint             ->getNC() > 0 ) return BT_FALSE;
	if( coupled_path_constraint         ->getNC() > 0 ) return BT_FALSE;
	if( algebraic_consistency_constraint->getNC() > 0 ) return BT_FALSE;

	if( path_constraint->isBoxConstraint() == BT_FALSE ) return BT_FALSE;
	if( point_constraints != 0 )
		for( uint run1 = 0; run1 < grid.getNumPoints(); run1++ )
			if( point_constraints[run1] != 0 )
				if( point_constraints[run1]->isBoxConstraint() == BT_FALSE ) return BT_FALSE;

	return BT_TRUE;
}



inline int Constraint::getNumberOfBlocks() const
{
    uint run1;
    int nc = 0;

    const uint N = grid.getNumPoints();

    if( boundary_constraint              ->getNC() != 0 )  nc ++  ;
    if( coupled_path_constraint          ->getNC() != 0 )  nc ++  ;
    if( path_constraint                  ->getNC() != 0 )  nc += N;
    if( algebraic_consistency_constraint ->getNC() != 0 )  nc += N;
    if( point_constraints                          != 0 )
        for( run1 = 0; run1 < N; run1++ )
            if( point_constraints[run1] != 0 )  nc ++;

    return nc;
}


inline int Constraint::getBlockDim( int idx ) const
{
    uint run1;
    int nc = 0;

    const uint N = grid.getNumPoints();

    if( boundary_constraint->getNC() != 0 ){
         if( idx == nc ) return boundary_constraint->getNC();
         nc ++  ;
    }
    if( coupled_path_constraint->getNC() != 0 ){
        if( idx == nc ) return coupled_path_constraint->getNC();
        nc ++  ;
    }
    if( path_constraint->getNC() != 0 ){
        for( run1 = 0; run1 < N; run1++ ){
            if( idx == nc ) return path_constraint->getDim( run1 );
            nc ++;
        }
    }
    if( algebraic_consistency_constraint->getNC() != 0 ){
        for( run1 = 0; run1 < N; run1++ ){
            if( idx == nc ) return algebraic_consistency_constraint->getDim( run1 );
            nc ++;
        }
    }
    if( point_constraints  != 0 ){
        for( run1 = 0; run1 < N; run1++ ){
            if( point_constraints[run1] != 0 ){
                if( idx == nc ) return point_constraints[run1]->getNC();
                nc ++;
            }
        }
    }

    return -1;
}


inline DVector Constraint::getBlockDims( ) const
{
	uint dim = getNumberOfBlocks();
	
	DVector result( dim );
	for( uint i=0; i<dim; ++i )
		result( i ) = (double)getBlockDim( i );
		
	return result;
}


CLOSE_NAMESPACE_ACADO

// end of file.
