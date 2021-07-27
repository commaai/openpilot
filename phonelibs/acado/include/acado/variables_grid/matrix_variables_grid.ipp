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
 *    \file include/acado/variables_grid/matrix_variables_grid.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 10.06.2008
 */


//
// mvukov: 
// Disable stupid warning on line 417
//
#ifdef WIN32
#pragma warning( disable : 4390 )
#endif

//
// PUBLIC MEMBER FUNCTIONS:
//

BEGIN_NAMESPACE_ACADO


inline double& MatrixVariablesGrid::operator()( uint pointIdx, uint rowIdx, uint colIdx )
{
	ASSERT( values != 0 );
	ASSERT( pointIdx < getNumPoints( ) );

    return values[pointIdx]->operator()( rowIdx,colIdx );
}


inline double MatrixVariablesGrid::operator()( uint pointIdx, uint rowIdx, uint colIdx ) const
{
	ASSERT( values != 0 );
	ASSERT( pointIdx < getNumPoints( ) );

    return values[pointIdx]->operator()( rowIdx,colIdx );
}



inline MatrixVariablesGrid MatrixVariablesGrid::operator()(	const uint rowIdx
															) const
{
    ASSERT( values != 0 );
	if ( rowIdx >= getNumRows( ) )
	{
		ACADOERROR( RET_INVALID_ARGUMENTS );
		return MatrixVariablesGrid();
	}

	Grid tmpGrid;
	getGrid( tmpGrid );

	MatrixVariablesGrid rowGrid( 1,1,tmpGrid,getType( ) );

    for( uint run1 = 0; run1 < getNumPoints(); run1++ )
         rowGrid( run1,0,0 ) = values[run1]->operator()( rowIdx,0 );

    return rowGrid;
}


inline MatrixVariablesGrid MatrixVariablesGrid::operator[](	const uint pointIdx
															) const
{
    ASSERT( values != 0 );
	if ( pointIdx >= getNumPoints( ) )
	{
		ACADOERROR( RET_INVALID_ARGUMENTS );
		return MatrixVariablesGrid();
	}

	MatrixVariablesGrid pointGrid;
	pointGrid.addMatrix( *(values[pointIdx]),getTime( pointIdx ) );

    return pointGrid;
}



inline MatrixVariablesGrid MatrixVariablesGrid::operator+(	const MatrixVariablesGrid& arg
															) const
{
	ASSERT( getNumPoints( ) == arg.getNumPoints( ) );

	MatrixVariablesGrid tmp( *this );

	for( uint i=0; i<getNumPoints( ); ++i )
		*(tmp.values[i]) += *(arg.values[i]);

	return tmp;
}





inline MatrixVariablesGrid& MatrixVariablesGrid::operator+=(	const MatrixVariablesGrid& arg
																)
{
	ASSERT( getNumPoints( ) == arg.getNumPoints( ) );

	for( uint i=0; i<getNumPoints( ); ++i )
		*(values[i]) += *(arg.values[i]);

	return *this;
}



inline MatrixVariablesGrid MatrixVariablesGrid::operator-(	const MatrixVariablesGrid& arg
															) const
{
	ASSERT( getNumPoints( ) == arg.getNumPoints( ) );

	MatrixVariablesGrid tmp( *this );

	for( uint i=0; i<getNumPoints( ); ++i )
		*(tmp.values[i]) -= *(arg.values[i]);

	return tmp;
}


inline MatrixVariablesGrid& MatrixVariablesGrid::operator-=(	const MatrixVariablesGrid& arg
																)
{
	ASSERT( getNumPoints( ) == arg.getNumPoints( ) );

	for( uint i=0; i<getNumPoints( ); ++i )
		*(values[i]) -= *(arg.values[i]);

	return *this;
}



inline uint MatrixVariablesGrid::getDim( ) const
{
	uint totalDim = 0;

	for( uint i=0; i<getNumPoints( ); ++i )
		totalDim += values[i]->getDim( );

	return totalDim;
}



inline uint MatrixVariablesGrid::getNumRows( ) const
{
	if ( values == 0 )
		return 0;

	return getNumRows( 0 );
}


inline uint MatrixVariablesGrid::getNumCols( ) const
{
	if ( values == 0 )
		return 0;

	return getNumCols( 0 );
}


inline uint MatrixVariablesGrid::getNumValues( ) const
{
	if ( values == 0 )
		return 0;

	return getNumValues( 0 );
}


inline uint MatrixVariablesGrid::getNumRows(	uint pointIdx
												) const
{
	if( values == 0 )
		return 0;

	ASSERT( pointIdx < getNumPoints( ) );

    return values[pointIdx]->getNumRows( );
}


inline uint MatrixVariablesGrid::getNumCols(	uint pointIdx
												) const
{
	if( values == 0 )
		return 0;

	ASSERT( pointIdx < getNumPoints( ) );

    return values[pointIdx]->getNumCols( );
}


inline uint MatrixVariablesGrid::getNumValues(	uint pointIdx
												) const
{
	if( values == 0 )
		return 0;

	ASSERT( pointIdx < getNumPoints( ) );

    return values[pointIdx]->getDim( );
}



inline VariableType MatrixVariablesGrid::getType( ) const
{
	if ( getNumPoints() == 0 )
		return VT_UNKNOWN;

	return getType( 0 );
}


inline returnValue MatrixVariablesGrid::setType(	VariableType _type
													)
{
	for( uint i=0; i<getNumPoints( ); ++i )
		setType( i,_type );

	return SUCCESSFUL_RETURN;
}


inline VariableType MatrixVariablesGrid::getType(	uint pointIdx
													) const
{
	if ( pointIdx >= getNumPoints( ) )
		return VT_UNKNOWN;

	return values[pointIdx]->getType( );
}


inline returnValue MatrixVariablesGrid::setType(	uint pointIdx,
													VariableType _type
													)
{
	if ( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );
	
	return values[pointIdx]->setType( _type );
}



inline returnValue MatrixVariablesGrid::getName(	uint pointIdx,
													uint idx,
													char* const _name
													) const
{
	if( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	return values[pointIdx]->getName( idx,_name );
}


inline returnValue MatrixVariablesGrid::setName(	uint pointIdx,
													uint idx,
													const char* const _name
													)
{
	if( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	return values[pointIdx]->setName( idx,_name );
}



inline returnValue MatrixVariablesGrid::getUnit(	uint pointIdx,
													uint idx,
													char* const _unit
													) const
{
	if( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	return values[pointIdx]->getUnit( idx,_unit );
}


inline returnValue MatrixVariablesGrid::setUnit(	uint pointIdx,
													uint idx,
													const char* const _unit
													)
{
	if( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	return values[pointIdx]->setUnit( idx,_unit );
}



inline DVector MatrixVariablesGrid::getScaling(	uint pointIdx
															) const
{
	if( pointIdx >= getNumPoints( ) )
		return emptyVector;

	return values[pointIdx]->getScaling( );
}


inline returnValue MatrixVariablesGrid::setScaling(	uint pointIdx,
													const DVector& _scaling
													)
{
    if ( pointIdx >= getNumPoints( ) )
        return ACADOERROR(RET_INDEX_OUT_OF_BOUNDS);

    return values[pointIdx]->setScaling( _scaling );
}


inline double MatrixVariablesGrid::getScaling(	uint pointIdx,
												uint valueIdx
												) const
{
    if( pointIdx >= getNumPoints( ) )
        return -1.0;

	return values[pointIdx]->getScaling( valueIdx );
}


inline returnValue MatrixVariablesGrid::setScaling(	uint pointIdx,
													uint valueIdx,
													double _scaling
													)
{
	if( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	if( valueIdx >= values[pointIdx]->getDim( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

    values[pointIdx]->setScaling( valueIdx,_scaling );
    return SUCCESSFUL_RETURN;
}



inline DVector MatrixVariablesGrid::getLowerBounds(	uint pointIdx
																) const
{
	if( pointIdx >= getNumPoints( ) )
		return emptyVector;

	return values[pointIdx]->getLowerBounds( );
}


inline returnValue MatrixVariablesGrid::setLowerBounds(	uint pointIdx,
														const DVector& _lb
														)
{
    if( pointIdx >= nPoints )
        return ACADOERROR(RET_INDEX_OUT_OF_BOUNDS);

    return values[pointIdx]->setLowerBounds( _lb );
}


inline double MatrixVariablesGrid::getLowerBound(	uint pointIdx,
													uint valueIdx
													) const
{
    if( pointIdx >= getNumPoints( ) )
        return -INFTY;

	return values[pointIdx]->getLowerBound( valueIdx );
}


inline returnValue MatrixVariablesGrid::setLowerBound(	uint pointIdx, 
														uint valueIdx, 
														double _lb
														)
{
	if( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	if( valueIdx >= values[pointIdx]->getDim( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	values[pointIdx]->setLowerBound( valueIdx,_lb );
    return SUCCESSFUL_RETURN;
}



inline DVector MatrixVariablesGrid::getUpperBounds(	uint pointIdx
																) const
{
	if( pointIdx >= getNumPoints( ) )
		return emptyVector;

	return values[pointIdx]->getUpperBounds( );
}


inline returnValue MatrixVariablesGrid::setUpperBounds(	uint pointIdx,
														const DVector& _ub
														)
{
    if( pointIdx >= getNumPoints( ) )
        return ACADOERROR(RET_INDEX_OUT_OF_BOUNDS);

    return values[pointIdx]->setUpperBounds( _ub );
}


inline double MatrixVariablesGrid::getUpperBound(	uint pointIdx,
													uint valueIdx
													) const
{
    if( pointIdx >= getNumPoints( ) )
        return INFTY;

	return values[pointIdx]->getUpperBound( valueIdx );
}


inline returnValue MatrixVariablesGrid::setUpperBound(	uint pointIdx,
														uint valueIdx,
														double _ub
														)
{
	if( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	if( valueIdx >= values[pointIdx]->getDim( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

    values[pointIdx]->setUpperBound( valueIdx,_ub );
    return SUCCESSFUL_RETURN;
}



inline BooleanType MatrixVariablesGrid::getAutoInit(	uint pointIdx 
														) const
{
	if ( pointIdx >= getNumPoints( ) )
	{
		ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );
		return defaultAutoInit;
	}

	return values[pointIdx]->getAutoInit( );
}


inline returnValue MatrixVariablesGrid::setAutoInit(	uint pointIdx,
														BooleanType _autoInit
														)
{
	if ( pointIdx >= getNumPoints( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	return values[pointIdx]->setAutoInit( _autoInit );
}


inline returnValue MatrixVariablesGrid::disableAutoInit( )
{ 
	for( uint i=0; i<getNumPoints( ); ++i )
		values[i]->setAutoInit( BT_FALSE );

	return SUCCESSFUL_RETURN;
}


inline returnValue MatrixVariablesGrid::enableAutoInit( )
{
	for( uint i=0; i<getNumPoints( ); ++i )
		values[i]->setAutoInit( BT_TRUE );

	return SUCCESSFUL_RETURN;
}



inline BooleanType MatrixVariablesGrid::hasNames( ) const
{
	for( uint i=0; i<getNumPoints( ); ++i )
	{
		if ( values[i]->hasNames( ) == BT_TRUE )
			return BT_TRUE;
	}

	return BT_FALSE;
}


inline BooleanType MatrixVariablesGrid::hasUnits( ) const
{
	for( uint i=0; i<getNumPoints( ); ++i )
	{
		if ( values[i]->hasUnits( ) == BT_TRUE )
			return BT_TRUE;
	}

	return BT_FALSE;
}


inline BooleanType MatrixVariablesGrid::hasScaling( ) const
{
	for( uint i=0; i<getNumPoints( ); ++i )
	{
		if ( values[i]->hasScaling( ) == BT_TRUE )
			return BT_TRUE;
	}

	return BT_FALSE;
}


inline BooleanType MatrixVariablesGrid::hasLowerBounds( ) const
{
	for( uint i=0; i<getNumPoints( ); ++i )
	{
		if ( values[i]->hasLowerBounds( ) == BT_TRUE )
			return BT_TRUE;
	}

	return BT_FALSE;
}


inline BooleanType MatrixVariablesGrid::hasUpperBounds( ) const
{
	for( uint i=0; i<getNumPoints( ); ++i )
	{
		if ( values[i]->hasUpperBounds( ) == BT_TRUE )
			return BT_TRUE;
	}

	return BT_FALSE;
}



inline double MatrixVariablesGrid::getMax( ) const
{
	double maxValue = -INFTY;

	for( uint i=0; i<getNumPoints( ); ++i )
	{
		if ( values[i]->getMax( ) > maxValue )
			maxValue = values[i]->getMax( );
	}

	return maxValue;
}


inline double MatrixVariablesGrid::getMin( ) const
{
	double minValue = INFTY;

	for( uint i=0; i<getNumPoints( ); ++i )
	{
		if ( values[i]->getMin( ) < minValue )
			minValue = values[i]->getMin( );
	}

	return minValue;
}


inline double MatrixVariablesGrid::getMean( ) const
{
	double meanValue = 0.0;

	if ( getNumPoints( ) == 0 )
		return meanValue;

	for( uint i=0; i<getNumPoints( ); ++i )
		meanValue += values[i]->getMean( );

	return ( meanValue / (double)getNumPoints( ) );
}



inline returnValue MatrixVariablesGrid::setZero( )
{
	for( uint i=0; i<getNumPoints( ); ++i )
		values[i]->setZero( );

	return SUCCESSFUL_RETURN;
}


inline returnValue MatrixVariablesGrid::setAll(	double _value
												)
{
    for( uint i = 0; i<getNumPoints( ); ++i )
		values[i]->setAll( _value );

    return SUCCESSFUL_RETURN;
}



inline returnValue MatrixVariablesGrid::getGrid(	Grid& _grid
													) const
{
	return _grid.init( getNumPoints(),times );
}


inline Grid MatrixVariablesGrid::getTimePoints( ) const
{
    Grid tmp;
    getGrid( tmp );
    return tmp;
}


CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
