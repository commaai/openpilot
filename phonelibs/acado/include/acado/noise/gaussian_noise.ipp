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
 *	\file include/acado/noise/gaussian_noise.ipp
 *	\author Hans Joachim Ferreau, Boris Houska
 *	\date 24.08.2008
 */



BEGIN_NAMESPACE_ACADO


//
// PUBLIC MEMBER FUNCTIONS:
//

inline returnValue GaussianNoise::setMeans( const DVector& _mean )
{
	if ( mean.getDim( ) != _mean.getDim( ) )
		return ACADOERROR( RET_VECTOR_DIMENSION_MISMATCH );

	mean = _mean;
	return SUCCESSFUL_RETURN;
}


inline returnValue GaussianNoise::setMeans( double _mean )
{
	for( uint i=0; i<getDim( ); ++i )
		mean(i) = _mean;

	return SUCCESSFUL_RETURN;
}


inline returnValue GaussianNoise::setMean(	uint idx,
											double _mean
											)
{
	if ( idx >= getDim( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	mean(idx) = _mean;
	return SUCCESSFUL_RETURN;
}



inline returnValue GaussianNoise::setVariances( const DVector& _variance )
{
	if ( variance.getDim( ) != _variance.getDim( ) )
		return ACADOERROR( RET_VECTOR_DIMENSION_MISMATCH );

	

	ASSERT( _variance > DVector( _variance.size() ) );

	variance = _variance;
	return SUCCESSFUL_RETURN;
}


inline returnValue GaussianNoise::setVariances( double _variance )
{
	for( uint i=0; i<getDim( ); ++i )
		variance(i) = _variance;

	return SUCCESSFUL_RETURN;
}


inline returnValue GaussianNoise::setVariance(	uint idx,
												double _variance
												)
{
	if ( idx >= getDim( ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	variance(idx) = _variance;
	return SUCCESSFUL_RETURN;
}



inline const DVector& GaussianNoise::getMean( ) const
{
	return mean;
}


inline const DVector& GaussianNoise::getVariance( ) const
{
	return variance;
}



CLOSE_NAMESPACE_ACADO

// end of file.
