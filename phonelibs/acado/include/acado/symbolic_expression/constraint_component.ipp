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
 *    \file include/acado/symbolic_expression/constraint_component.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO




inline ConstraintComponent ConstraintComponent::operator<=( const double& ub_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setUB( ub_ );
    return tmp;
}

inline ConstraintComponent ConstraintComponent::operator>=( const double& lb_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setLB( lb_ );
    return tmp;
}

inline ConstraintComponent ConstraintComponent::operator==( const double&  b_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setLB( b_ );
    tmp.setUB( b_ );
    return tmp;
}

inline ConstraintComponent ConstraintComponent::operator<=( const DVector& ub_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setUB( ub_ );
    return tmp;
}

inline ConstraintComponent ConstraintComponent::operator>=( const DVector& lb_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setLB( lb_ );
    return tmp;
}

inline ConstraintComponent ConstraintComponent::operator==( const DVector&  b_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setLB( b_ );
    tmp.setUB( b_ );
    return tmp;
}

inline ConstraintComponent ConstraintComponent::operator<=( const VariablesGrid& ub_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setUB( ub_ );
    return tmp;
}

inline ConstraintComponent ConstraintComponent::operator>=( const VariablesGrid& lb_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setLB( lb_ );
    return tmp;
}

inline ConstraintComponent ConstraintComponent::operator==( const VariablesGrid&  b_ ) const{

    ConstraintComponent tmp(*this);
    tmp.setLB( b_ );
    tmp.setUB( b_ );
    return tmp;
}


inline uint ConstraintComponent::getDim( ) const{

    return expression.getDim();
}


inline const DVector& ConstraintComponent::getLB() const{

    return lb;
}


inline const DVector& ConstraintComponent::getUB() const{

    return ub;
}


inline returnValue ConstraintComponent::setLB( const double &lb_ ){

    lb.init( expression.getDim() );
    lb.setAll( lb_ );

    return SUCCESSFUL_RETURN;
}


inline returnValue ConstraintComponent::setLB( const DVector& lb_ ){

    if( lb_.getDim() != expression.getDim() )
        return ACADOERROR( RET_INCOMPATIBLE_DIMENSIONS );;

    lb = lb_;

    return SUCCESSFUL_RETURN;
}


inline returnValue ConstraintComponent::setLB( const VariablesGrid& lb_ ){

    if( lb_.getNumValues() != expression.getDim() )
        return ACADOERROR( RET_INCOMPATIBLE_DIMENSIONS );;

    lbGrid = lb_;

    return SUCCESSFUL_RETURN;
}


inline returnValue ConstraintComponent::setUB( const double &ub_ ){

    ub.init( expression.getDim() );
    ub.setAll( ub_ );

    return SUCCESSFUL_RETURN;
}


inline returnValue ConstraintComponent::setUB( const DVector& ub_ ){

    if( ub_.getDim() != expression.getDim() )
        return ACADOERROR( RET_INCOMPATIBLE_DIMENSIONS );;

    ub = ub_;

    return SUCCESSFUL_RETURN;
}


inline returnValue ConstraintComponent::setUB( const VariablesGrid& ub_ ){

    if( ub_.getNumValues() != expression.getDim() )
        return ACADOERROR( RET_INCOMPATIBLE_DIMENSIONS );;

    ubGrid = ub_;

    return SUCCESSFUL_RETURN;
}



inline Expression ConstraintComponent::getExpression() const{

    return expression;
}


inline BooleanType ConstraintComponent::hasUBgrid( ) const{

    if( ubGrid.isEmpty() == BT_TRUE ) return BT_FALSE;
    return BT_TRUE;
}


inline BooleanType ConstraintComponent::hasLBgrid( ) const{

    if( lbGrid.isEmpty() == BT_TRUE ) return BT_FALSE;
    return BT_TRUE;
}


inline const VariablesGrid& ConstraintComponent::getLBgrid() const{

    return lbGrid;
}


inline const VariablesGrid& ConstraintComponent::getUBgrid() const{

    return ubGrid;
}


CLOSE_NAMESPACE_ACADO

// end of file.
