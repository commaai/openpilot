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
 *    \file include/acado/utils/matlab_acado_utils.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 17.08.2008
 *
 *    This file declares several global utility functions that
 *    are needed in the MATLAB interfaces. Please never use
 *    this file in stand-alone C/C++ code.
 *
 */


#ifndef ACADO_TOOLKIT_MATLAB_UTILS_HPP
#define ACADO_TOOLKIT_MATLAB_UTILS_HPP


#include <math.h>
#include <mex.h>

#include <acado/utils/acado_namespace_macros.hpp>


BEGIN_NAMESPACE_ACADO


int isScalar( const mxArray* const M )
{
	if ( ( mxGetM( M ) != 1 ) || ( mxGetN( M ) != 1 ) )
		return 0;
	else
		return 1;
}


int isFunctionHandle( const mxArray* const M )
{
	if ( M == NULL )
		return 0;

	if ( mxIsCell(M) )
		return 0;

	if ( mxIsChar(M) )
		return 0;

	if ( mxIsComplex(M) )
		return 0;

	if ( mxIsDouble(M) )
		return 0;

	if ( mxIsEmpty(M) )
		return 0;

	if ( mxIsInt8(M) )
		return 0;

	if ( mxIsInt16(M) )
		return 0;

	if ( mxIsInt32(M) )
		return 0;

	if ( mxIsLogical(M) )
		return 0;

	if ( mxIsLogicalScalar(M) )
		return 0;

	if ( mxIsLogicalScalarTrue(M) )
		return 0;

	if ( mxIsNumeric(M) )
		return 0;

	if ( mxIsSingle(M) )
		return 0;

	if ( mxIsSparse(M) )
		return 0;

	if ( mxIsStruct(M) )
		return 0;

	if ( mxIsUint8(M) )
		return 0;

	if ( mxIsUint16(M) )
		return 0;

	if ( mxIsUint32(M) )
		return 0;

	// assume to be a function handle iff it is nothing else
	return 1;
}


void acadoPlot(VariablesGrid grid){

	mxArray *XTrajectory = NULL;
	double  *xTrajectory = NULL;

    XTrajectory = mxCreateDoubleMatrix( grid.getNumPoints(),1+grid.getNumValues(),mxREAL );
    xTrajectory = mxGetPr( XTrajectory );

    for( int i=0; i<grid.getNumPoints(); ++i ){
        xTrajectory[0*grid.getNumPoints() + i] = grid.getTime(i);
        for( int j=0; j<grid.getNumValues(); ++j ){
            xTrajectory[(1+j)*grid.getNumPoints() + i] = grid(i, j);
        }
    }


    mxArray* plotArguments[] = { XTrajectory };
    mexCallMATLAB( 0,0,1,plotArguments,"acadoPlot" );

}



CLOSE_NAMESPACE_ACADO



#endif   // ACADO_TOOLKIT_MATLAB_UTILS_HPP


/*
 *    end of file
 */
