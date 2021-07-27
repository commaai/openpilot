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
 *    \file include/acado/estimator/estimator.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 20.08.2008
 */



BEGIN_NAMESPACE_ACADO


//
// PUBLIC MEMBER FUNCTIONS:
//

inline returnValue Estimator::getOutputs(	DVector& _x,
											DVector& _xa,
											DVector& _u,
											DVector& _p,
											DVector& _w 
											) const
{
	_x   = x;
	_xa  = xa;
	_u   = u;
	_p   = p;
	_w   = w;
	
	return SUCCESSFUL_RETURN;
}

inline returnValue Estimator::getX(	DVector& _x
									) const
{
	_x = x;
	return SUCCESSFUL_RETURN;
}


inline returnValue Estimator::getXA(	DVector& _xa
									) const
{
	_xa = xa;
	return SUCCESSFUL_RETURN;
}


inline returnValue Estimator::getU(	DVector& _u
									) const
{
	_u = u;
	return SUCCESSFUL_RETURN;
}


inline returnValue Estimator::getP(	DVector& _p
									) const
{
	_p = p;
	return SUCCESSFUL_RETURN;
}


inline returnValue Estimator::getW(	DVector& _w
									) const
{
	_w = w;
	return SUCCESSFUL_RETURN;
}


inline uint Estimator::getNX( ) const
{
	return x.getDim( );
}


inline uint Estimator::getNXA( ) const
{
	return xa.getDim( );
}


inline uint Estimator::getNU( ) const
{
	return u.getDim( );
}


inline uint Estimator::getNP( ) const
{
	return p.getDim( );
}


inline uint Estimator::getNW( ) const
{
	return w.getDim( );
}


inline uint Estimator::getNY( ) const
{
	return 0;
}



CLOSE_NAMESPACE_ACADO

// end of file.
