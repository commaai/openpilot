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
 *    \file include/acado/constraint/path_constraint.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//



BEGIN_NAMESPACE_ACADO



inline int PathConstraint::getNC() const{

    if( fcn == 0 )
        return 0;

    return (fcn[0].getDim())*((int) grid.getNumPoints());
}


inline int PathConstraint::getDim( const int& idx_)
{
	if (idx_ >= (int) grid.getNumPoints())
	{
		LOG( LVL_WARNING ) << "Invalid index, will be ignored" << std::endl;
	}

    if(fcn == 0)
        return 0;

    return fcn[0].getDim();
}


inline BooleanType PathConstraint::isBoxConstraint( ) const
{
	return isAffine( ); //TODO: implement this correctly!
}



inline returnValue PathConstraint::add( const DVector lb_, const Expression& arg, const DVector ub_  ){

    uint run1;

    if( fcn == 0 )
        return ACADOERROR(RET_MEMBER_NOT_INITIALISED);


    fcn[0] << arg;

    for( run1 = 0; run1 < grid.getNumPoints(); run1++ ){

       lb[run1] = (double*)realloc(lb[run1],fcn[0].getDim()*sizeof(double));
       ub[run1] = (double*)realloc(ub[run1],fcn[0].getDim()*sizeof(double));

       ub[run1][fcn[0].getDim()-1] = ub_(run1);
       lb[run1][fcn[0].getDim()-1] = lb_(run1);
    }

    return SUCCESSFUL_RETURN;
}


CLOSE_NAMESPACE_ACADO

// end of file.
