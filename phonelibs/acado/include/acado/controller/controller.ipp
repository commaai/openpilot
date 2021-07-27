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
 *    \file include/acado/controller/controller.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 20.08.2008
 */



BEGIN_NAMESPACE_ACADO


//
// PUBLIC MEMBER FUNCTIONS:
//


inline returnValue Controller::getU(	DVector& _u
										) const
{
	if ( controlLaw != 0 )
		return controlLaw->getU( _u );
	else
		return ACADOERROR( RET_NO_CONTROLLAW_SPECIFIED );
}


inline returnValue Controller::getP(	DVector& _p
										) const
{
	if ( controlLaw != 0 )
		return controlLaw->getP( _p );
	else
		return ACADOERROR( RET_NO_CONTROLLAW_SPECIFIED );
}



inline uint Controller::getNY( ) const
{
	if ( estimator != 0 )
		return estimator->getNY( );

	if ( controlLaw != 0 )
		return controlLaw->getNX( );

	return 0;
}


inline uint Controller::getNU( ) const
{
	if ( controlLaw != 0 )
		return controlLaw->getNU( );
	else
		return 0;
}


inline uint Controller::getNP( ) const
{
	if ( controlLaw != 0 )
		return controlLaw->getNP( );
	else
		return 0;
}



inline BooleanType Controller::hasDynamicControlLaw( ) const
{
	if ( controlLaw == 0 )
		return BT_FALSE;

	return controlLaw->isDynamic( );
}


inline BooleanType Controller::hasStaticControlLaw( ) const
{
	if ( controlLaw == 0 )
		return BT_FALSE;

	return controlLaw->isStatic( );
}


inline BooleanType Controller::hasEstimator( ) const
{
	if ( estimator != 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline BooleanType Controller::hasReferenceTrajectory( ) const
{
	if ( referenceTrajectory != 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}



inline double Controller::getSamplingTimeControlLaw( )
{
	if ( controlLaw != 0 )
		return controlLaw->getSamplingTime( );
	else
		return -1.0;
}


inline double Controller::getSamplingTimeEstimator( )
{
	if ( estimator != 0 )
		return estimator->getSamplingTime( );
	else
		return -1.0;
}



inline double Controller::getPreviousRealRuntime( )
{
	return realClock.getTime( );
}


inline returnValue Controller::enable( )
{
	isEnabled = BT_TRUE;
	return SUCCESSFUL_RETURN;
}


inline returnValue Controller::disable( )
{
	isEnabled = BT_FALSE;
	return SUCCESSFUL_RETURN;
}



CLOSE_NAMESPACE_ACADO

// end of file.
