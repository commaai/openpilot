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
*    \file include/acado/symbolic_expression/constraint_component.hpp
*    \author Boris Houska, Hans Joachim Ferreau
*/


#ifndef ACADO_TOOLKIT_CONSTRAINT_COMPONENT_HPP
#define ACADO_TOOLKIT_CONSTRAINT_COMPONENT_HPP


#include <acado/symbolic_expression/symbolic_expression.hpp>
#include <acado/variables_grid/variables_grid.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Data class for symbolically formulating constraints within optimal control problems.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class ConstraintComponent is a data class for symbolically formulating
 *  constraints within optimal control problems.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */

class ConstraintComponent{

public:

    /** Default constructor */
    ConstraintComponent();

    // --------------------------------------------------------------------------------------------


    /** Copy constructor (deep copy). */
    ConstraintComponent( const ConstraintComponent &arg );

    /** Default destructor. */
    ~ConstraintComponent();

    /** Assignment Operator (deep copy). */
    ConstraintComponent& operator=( const ConstraintComponent &arg );


    /** Access Operator. */
    ConstraintComponent operator()( const uint &index ) const;


    /** Initializes the constraint component. */
    returnValue initialize( const DVector& lb, Expression arg, const DVector& ub );


    /** Initializes the constraint component. */
    returnValue initialize( const VariablesGrid& lb, Expression arg, const VariablesGrid& ub );


    // --------------------------------------------------------------------------------------------


    // Friend Functions:
    // -----------------
    friend ConstraintComponent operator<=( double lb_, const ConstraintComponent &arg );
    friend ConstraintComponent operator>=( double ub_, const ConstraintComponent &arg );

    friend ConstraintComponent operator<=( DVector lb_, const ConstraintComponent &arg );
    friend ConstraintComponent operator>=( DVector ub_, const ConstraintComponent &arg );

    friend ConstraintComponent operator<=( VariablesGrid lb_, const ConstraintComponent &arg );
    friend ConstraintComponent operator>=( VariablesGrid ub_, const ConstraintComponent &arg );


    // --------------------------------------------------------------------------------------------


    // Operators:
    // ----------
    inline ConstraintComponent operator<=( const double& ub ) const;
    inline ConstraintComponent operator>=( const double& lb ) const;
    inline ConstraintComponent operator==( const double&  b ) const;

    inline ConstraintComponent operator<=( const DVector& ub ) const;
    inline ConstraintComponent operator>=( const DVector& lb ) const;
    inline ConstraintComponent operator==( const DVector&  b ) const;

    inline ConstraintComponent operator<=( const VariablesGrid& ub ) const;
    inline ConstraintComponent operator>=( const VariablesGrid& lb ) const;
    inline ConstraintComponent operator==( const VariablesGrid&  b ) const;


    // --------------------------------------------------------------------------------------------

    inline const DVector& getLB() const;
    inline const DVector& getUB() const;

    inline returnValue setLB( const double&        lb_ );
    inline returnValue setLB( const DVector&        lb_ );
    inline returnValue setLB( const VariablesGrid& lb_ );

    inline returnValue setUB( const double&        lb_ );
    inline returnValue setUB( const DVector&        lb_ );
    inline returnValue setUB( const VariablesGrid& lb_ );

    inline Expression getExpression( ) const;

    inline BooleanType hasUBgrid( ) const;
    inline BooleanType hasLBgrid( ) const;

    inline const VariablesGrid& getLBgrid() const;
    inline const VariablesGrid& getUBgrid() const;

    inline uint getDim( ) const;


//
//  PROTECTED MEMBERS:
//

protected:

    Expression     expression  ;

    DVector         lb          ;
    DVector         ub          ;

    VariablesGrid  lbGrid      ;
    VariablesGrid  ubGrid      ;
};

ConstraintComponent operator<=( const Expression& arg, const double& ub );
ConstraintComponent operator>=( const Expression& arg, const double& lb );
ConstraintComponent operator==( const Expression& arg, const double&  b );

ConstraintComponent operator<=( const Expression& arg, const DVector& ub );
ConstraintComponent operator>=( const Expression& arg, const DVector& lb );
ConstraintComponent operator==( const Expression& arg, const DVector&  b );

ConstraintComponent operator<=( const Expression& arg, const VariablesGrid& ub );
ConstraintComponent operator>=( const Expression& arg, const VariablesGrid& lb );
ConstraintComponent operator==( const Expression& arg, const VariablesGrid&  b );

ConstraintComponent operator<=( double lb, const Expression &arg );
ConstraintComponent operator==( double  b, const Expression &arg );
ConstraintComponent operator>=( double ub, const Expression &arg );

ConstraintComponent operator<=( DVector lb, const Expression &arg );
ConstraintComponent operator==( DVector  b, const Expression &arg );
ConstraintComponent operator>=( DVector ub, const Expression &arg );

ConstraintComponent operator<=( VariablesGrid lb, const Expression &arg );
ConstraintComponent operator==( VariablesGrid  b, const Expression &arg );
ConstraintComponent operator>=( VariablesGrid ub, const Expression &arg );

CLOSE_NAMESPACE_ACADO

#include <acado/symbolic_expression/constraint_component.ipp>

#endif
