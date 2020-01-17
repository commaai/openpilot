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
 *    \file include/acado/constraint/coupled_path_constraint.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//



BEGIN_NAMESPACE_ACADO



inline int CoupledPathConstraint::getNC() const{

    if( fcn == 0 )
        return 0;

    return fcn[0].getDim();
}


inline returnValue CoupledPathConstraint::add( const double lb_, const Expression* arg, const double ub_ ){

    uint run1;

    if( fcn == 0 )
        return ACADOERROR(RET_MEMBER_NOT_INITIALISED);

    for( run1 = 0; run1 < grid.getNumPoints(); run1++ )
        fcn[run1] << arg[run1];

    lb[0] = (double*)realloc(lb[0],fcn[0].getDim()*sizeof(double));
    ub[0] = (double*)realloc(ub[0],fcn[0].getDim()*sizeof(double));

    ub[0][fcn[0].getDim()-1] = ub_;
    lb[0][fcn[0].getDim()-1] = lb_;

    return SUCCESSFUL_RETURN;
}



CLOSE_NAMESPACE_ACADO

// end of file.
