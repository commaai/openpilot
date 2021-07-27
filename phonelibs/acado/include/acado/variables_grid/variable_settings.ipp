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
 *    \file include/acado/variables_grid/variable_settings.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 11.06.2008
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO



inline VariableType VariableSettings::getType( ) const
{
	return type;
}


inline returnValue VariableSettings::setType(	VariableType _type
												)
{
	type = _type;
	return SUCCESSFUL_RETURN;
}



inline DVector VariableSettings::getScaling( ) const
{
	if ( scaling.isEmpty( ) == BT_TRUE )
	{
		DVector tmp( dim );
		tmp.setAll( defaultScaling );
		return tmp;
	}
	else
		return scaling;
}


inline returnValue VariableSettings::setScaling(	const DVector& _scaling
													)
{
	if ( dim != _scaling.getDim( ) )
		return ACADOERROR( RET_VECTOR_DIMENSION_MISMATCH );

	if ( acadoIsSmaller( _scaling.getMin( ),0.0 ) == BT_TRUE )
		return ACADOERROR( RET_INVALID_ARGUMENTS );
	
	scaling = _scaling;
	return SUCCESSFUL_RETURN;
}


inline double VariableSettings::getScaling(	uint idx
											) const
{
	if ( idx >= dim )
		return -1.0;

	if ( scaling.isEmpty( ) == BT_TRUE )
		return defaultScaling;
	else
		return scaling( idx );
}


inline returnValue VariableSettings::setScaling(	uint idx,
													double _scaling
													)
{
	if ( idx >= dim )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	if ( scaling.isEmpty( ) == BT_TRUE )
	{
		scaling.init( dim );
		scaling.setAll( defaultScaling );
	}

	if ( acadoIsSmaller( _scaling,0.0 ) == BT_TRUE )
		return ACADOERROR( RET_INVALID_ARGUMENTS );
	
	scaling( idx ) = _scaling;
	return SUCCESSFUL_RETURN;
}


inline DVector VariableSettings::getLowerBounds( ) const
{
	if ( lb.isEmpty( ) == BT_TRUE )
	{
		DVector tmp( dim );
		tmp.setAll( defaultLowerBound );
		return tmp;
	}
	else
		return lb;
}


inline returnValue VariableSettings::setLowerBounds(	const DVector& _lb
														)
{
	if( _lb.getDim() != dim )
		return ACADOERROR( RET_VECTOR_DIMENSION_MISMATCH );

	lb = _lb;
	return SUCCESSFUL_RETURN;
}


inline double VariableSettings::getLowerBound(	uint idx
												) const
{
	if ( idx >= dim )
		return INFTY;

	if ( lb.isEmpty( ) == BT_TRUE )
		return defaultLowerBound;
	else
		return lb( idx );
}


inline returnValue VariableSettings::setLowerBound(	uint idx,
													double _lb
													)
{
	if( idx >= dim )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	if ( lb.isEmpty( ) == BT_TRUE )
	{
		lb.init( dim );
		lb.setAll( defaultLowerBound );
	}

	lb( idx ) = _lb;
	return SUCCESSFUL_RETURN;
}



inline DVector VariableSettings::getUpperBounds( ) const
{
	if ( ub.isEmpty( ) == BT_TRUE )
	{
		DVector tmp( dim );
		tmp.setAll( defaultUpperBound );
		return tmp;
	}
	else
		return ub;
}


inline returnValue VariableSettings::setUpperBounds(	const DVector& _ub 
														)
{
	if( _ub.getDim() != dim )
		return ACADOERROR(RET_VECTOR_DIMENSION_MISMATCH);

	ub = _ub;
	return SUCCESSFUL_RETURN;
}


inline double VariableSettings::getUpperBound(	uint idx
												) const
{
	if ( idx >= dim )
		return -INFTY;

	if ( ub.isEmpty( ) == BT_TRUE )
		return defaultUpperBound;
	else
		return ub( idx );
}


inline returnValue VariableSettings::setUpperBound(	uint idx,
													double _ub 
													)
{
	if( idx >= dim )
		return ACADOERROR(RET_INDEX_OUT_OF_BOUNDS);

	if ( ub.isEmpty( ) == BT_TRUE )
	{
		ub.init( dim );
		ub.setAll( defaultUpperBound );
	}

	ub( idx ) = _ub;
	return SUCCESSFUL_RETURN;
}


inline BooleanType VariableSettings::getAutoInit( ) const
{
	return autoInit;
}


inline returnValue VariableSettings::setAutoInit(	BooleanType _autoInit
													)
{  
	autoInit = _autoInit;
	return SUCCESSFUL_RETURN;
}


inline BooleanType VariableSettings::hasNames( ) const
{
	if ( names != 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline BooleanType VariableSettings::hasUnits( ) const
{
	if ( units != 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline BooleanType VariableSettings::hasScaling( ) const
{
	if ( scaling.isEmpty( ) == BT_TRUE )
		return BT_FALSE;
	else
		return BT_TRUE;
}



inline BooleanType VariableSettings::hasLowerBounds( ) const
{
	if ( lb.isEmpty( ) == BT_TRUE )
		return BT_FALSE;
	else
		return BT_TRUE;
}


inline BooleanType VariableSettings::hasUpperBounds( ) const
{
	if ( ub.isEmpty( ) == BT_TRUE )
		return BT_FALSE;
	else
		return BT_TRUE;
}



CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
