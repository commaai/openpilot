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
 *    \file include/acado/code_generation/integrators/irk_export.ipp
 *    \author Rien Quirynen
 *    \date 2013
 */


BEGIN_NAMESPACE_ACADO


//
// Create the correct integrator
//
inline ImplicitRungeKuttaExport* createImplicitRungeKuttaExport(	UserInteraction* _userInteraction,
																	const std::string &_commonHeaderName	)
{
	int sensGen;
	_userInteraction->get( DYNAMIC_SENSITIVITY, sensGen );
	int liftedGen;
	_userInteraction->get( IMPLICIT_INTEGRATOR_MODE, liftedGen );
	
	if ( (ImplicitIntegratorMode)liftedGen == LIFTED && ((ExportSensitivityType)sensGen == FORWARD || (ExportSensitivityType)sensGen == INEXACT) ) {
		return new ForwardLiftedIRKExport(_userInteraction, _commonHeaderName);
	}
	else if ( (ImplicitIntegratorMode)liftedGen == LIFTED && (ExportSensitivityType)sensGen == BACKWARD ) {
		return new AdjointLiftedIRKExport(_userInteraction, _commonHeaderName);
	}
	else if ( (ImplicitIntegratorMode)liftedGen == LIFTED && (ExportSensitivityType)sensGen == SYMMETRIC ) {
		return new SymmetricLiftedIRKExport(_userInteraction, _commonHeaderName);
	}
	else if ( (ImplicitIntegratorMode)liftedGen == LIFTED && (ExportSensitivityType)sensGen == FORWARD_OVER_BACKWARD ) {
		return new ForwardBackwardLiftedIRKExport(_userInteraction, _commonHeaderName);
	}
	else if ( (ImplicitIntegratorMode)liftedGen == LIFTED_FEEDBACK && ((ExportSensitivityType)sensGen == FORWARD || (ExportSensitivityType)sensGen == INEXACT) ) {
		return new FeedbackLiftedIRKExport(_userInteraction, _commonHeaderName);
	}
	else if ( (ExportSensitivityType)sensGen == SYMMETRIC || (ExportSensitivityType)sensGen == FORWARD_OVER_BACKWARD ) {
		return new SymmetricIRKExport(_userInteraction, _commonHeaderName);
	}
	else if ( (ExportSensitivityType)sensGen == FORWARD ) {
		return new ForwardIRKExport(_userInteraction, _commonHeaderName);
	}
	else if( (ExportSensitivityType)sensGen == NO_SENSITIVITY ) {
		return new ImplicitRungeKuttaExport(_userInteraction, _commonHeaderName);
	}
	else {
		ACADOERROR( RET_INVALID_OPTION );
		return new ImplicitRungeKuttaExport(_userInteraction, _commonHeaderName);
	}
}


CLOSE_NAMESPACE_ACADO


// end of file.
