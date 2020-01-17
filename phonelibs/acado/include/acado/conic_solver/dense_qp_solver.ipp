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
 *    \file include/acado/conic_solver/dense_qp_solver.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date   2008-2010
 */


//
// PUBLIC MEMBER FUNCTIONS:
//



BEGIN_NAMESPACE_ACADO


inline QPStatus DenseQPsolver::getStatus( ) const
{
	return qpStatus;
}


inline BooleanType DenseQPsolver::isSolved( ) const
{
	if ( ( getStatus( ) == QPS_SOLVED ) || ( getStatus( ) == QPS_SOLVED_RELAXATION ) )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline BooleanType DenseQPsolver::isInfeasible( ) const
{
	if ( qpStatus == QPS_INFEASIBLE )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline BooleanType DenseQPsolver::isUnbounded( ) const
{
	if ( qpStatus == QPS_UNBOUNDED )
		return BT_TRUE;
	else
		return BT_FALSE;
}



CLOSE_NAMESPACE_ACADO

// end of file.
