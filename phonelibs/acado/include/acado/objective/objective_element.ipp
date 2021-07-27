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
 *    \file include/acado/objective/objective_element.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO


inline returnValue ObjectiveElement::setGrid( const Grid &grid_ ){

    grid = grid_;
    return SUCCESSFUL_RETURN;
}


inline Grid ObjectiveElement::getGrid() const{

    return grid;
}


inline int ObjectiveElement::getNX() const{

    return fcn.getNX();
}


inline int ObjectiveElement::getNXA() const{

    return fcn.getNXA();
}


inline int ObjectiveElement::getNU() const{

    return fcn.getNU();
}


inline int ObjectiveElement::getNP() const{

    return fcn.getNP();
}


inline int ObjectiveElement::getNW() const{

    return fcn.getNW();
}

inline returnValue ObjectiveElement::getFunction( Function& _function )
{
	_function = fcn;
	
	return SUCCESSFUL_RETURN;
}


CLOSE_NAMESPACE_ACADO

// end of file.
