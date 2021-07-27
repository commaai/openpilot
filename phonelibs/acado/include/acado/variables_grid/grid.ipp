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
 *    \file include/acado/variables_grid/grid.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 31.05.2008
 */


//
// PUBLIC MEMBER FUNCTIONS:
//



BEGIN_NAMESPACE_ACADO



inline BooleanType Grid::operator==(	const Grid& arg
										) const
{

	if ( getNumPoints( ) != arg.getNumPoints( ) )
		return BT_FALSE;

	for( uint i=0; i<getNumPoints( ); ++i )
		if ( acadoIsEqual( getTime( i ) , arg.getTime( i ) ) == BT_FALSE )
			return BT_FALSE;

	return BT_TRUE;
}


inline BooleanType Grid::operator!=(	const Grid& arg
										) const
{
	if ( operator==( arg ) == BT_TRUE )
		return BT_FALSE;
	else
		return BT_TRUE;
}


inline BooleanType Grid::operator<(	const Grid& arg
									) const
{
	if ( getNumPoints( ) >= arg.getNumPoints( ) )
		return BT_FALSE;

	int idx = 0;

	for( uint i=0; i<getNumPoints( ); ++i )
	{
		idx = arg.findTime( times[i],idx );
		if ( idx < 0 )
			return BT_FALSE;
	}

	return BT_TRUE;
}


inline BooleanType Grid::operator<=(	const Grid& arg
										) const
{
	if ( ( operator<( arg ) == BT_TRUE ) || ( operator==( arg ) == BT_TRUE ) )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline BooleanType Grid::operator>(	const Grid& arg
									) const
{
	if ( getNumPoints( ) <= arg.getNumPoints( ) )
		return BT_FALSE;

	int idx = 0;

	for( uint i=0; i<arg.getNumPoints( ); ++i )
	{
		idx = findTime( arg.times[i],idx );
		if ( idx < 0 )
			return BT_FALSE;
	}

	return BT_TRUE;
}


inline BooleanType Grid::operator>=(	const Grid& arg
										) const
{
	if ( ( operator>( arg ) == BT_TRUE ) || ( operator==( arg ) == BT_TRUE ) )
		return BT_TRUE;
	else
		return BT_FALSE;
}



inline BooleanType Grid::isEmpty( ) const
{
	if ( nPoints == 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline uint Grid::getNumPoints( ) const
{
	return nPoints;
}


inline uint Grid::getNumIntervals( ) const
{
	if ( nPoints > 0 )
		return getLastIndex( );
	else
		return 0;
}


inline double Grid::getFirstTime( ) const
{
	ASSERT( times != 0 );
	return times[0];
}


inline double Grid::getLastTime( ) const
{
	ASSERT( times != 0 );
	return times[nPoints-1];
}


inline double Grid::getTime(	uint pointIdx
								) const
{
	ASSERT( times != 0 );

        if( pointIdx >= getNumPoints( ) ){
            return getLastTime();
	}

	return times[pointIdx];
}



inline BooleanType Grid::isEquidistant( ) const
{
	if ( getNumIntervals( ) <= 1 )
		return BT_TRUE;

	double length = getIntervalLength( 0 );
	for( uint i=1; i<getNumIntervals( ); ++i )
		if ( acadoIsEqual( getIntervalLength(i),length ) == BT_FALSE )
			return BT_FALSE;
		
	return BT_TRUE;
}


inline double Grid::getIntervalLength( ) const{

    if ( times == 0 )
		return -1.0;

    return times[nPoints-1] - times[0];
}


inline double Grid::getIntervalLength(	uint pointIdx
										) const
{
	if ( times == 0 )
		return -1.0;

	if ( pointIdx >= getNumPoints( ) )
	{
		ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );
		return -INFTY;
	}

	if ( pointIdx == getLastIndex( ) )
		return 0.0;
	else
		return times[pointIdx+1] - times[pointIdx];
}


inline uint Grid::getLastIndex( ) const
{
	if ( getNumPoints( ) > 0 )
		return getNumPoints( ) - 1;
	else
		return 0;
}


inline BooleanType Grid::isLast(	uint pointIdx
									) const
{
	if ( pointIdx == getLastIndex( ) )
		return BT_TRUE;
	else
		return BT_FALSE;
}



inline BooleanType Grid::isInInterval(	double _time
										) const
{
    if ( times == 0 )
		return BT_FALSE;

    if ( acadoIsSmaller( getTime( 0             ) , _time ) == BT_TRUE &&
         acadoIsGreater( getTime( getLastIndex()) , _time ) == BT_TRUE    )  return BT_TRUE ;
    else                                                                     return BT_FALSE;
}


inline BooleanType Grid::isInInterval( uint pointIdx, double _time ) const{

    if ( times == 0 )
		return BT_FALSE;

	if ( pointIdx >= getNumPoints( ) )
	{
		ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );
		return BT_FALSE;
	}

	uint idxPlusOne = pointIdx;

	if ( pointIdx < getLastIndex( ) )
		++idxPlusOne;

    if ( acadoIsSmaller( getTime( pointIdx   ) , _time ) == BT_TRUE &&
         acadoIsGreater( getTime( idxPlusOne ) , _time ) == BT_TRUE    )  return BT_TRUE ;
    else                                                                  return BT_FALSE;
}


inline BooleanType Grid::isInUpperHalfOpenInterval( uint pointIdx, double _time ) const
{
	if ( times == 0 )
		return BT_FALSE;

	if ( pointIdx >= getNumPoints( ) )
	{
		ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );
		return BT_FALSE;
	}

	uint idxPlusOne = pointIdx;

	if ( pointIdx < getLastIndex( ) )
		++idxPlusOne;

    if ( acadoIsSmaller        ( getTime( pointIdx )  , _time ) == BT_TRUE &&
         acadoIsStrictlyGreater( getTime( idxPlusOne ), _time ) == BT_TRUE    ) return BT_TRUE ;
    else                                                                        return BT_FALSE;
}


inline BooleanType Grid::isInLowerHalfOpenInterval( uint pointIdx, double _time ) const{

	if ( times == 0 )
		return BT_FALSE;

	if ( pointIdx >= getNumPoints( ) )
	{
		ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );
		return BT_FALSE;
	}

	uint idxMinusOne = pointIdx;

	if ( pointIdx > 0 )
		idxMinusOne--;

    if ( acadoIsStrictlySmaller( getTime( idxMinusOne ) , _time ) == BT_TRUE &&
         acadoIsGreater        ( getTime( pointIdx )    , _time ) == BT_TRUE     )  return BT_TRUE;
    else                                                                            return BT_FALSE;
}




CLOSE_NAMESPACE_ACADO

/*
 *	end of file
 */
