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
 *    \file include/acado/integrator/integrator_discretized_ode.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_INTEGRATOR_DISCRETIZED_ODE_HPP
#define ACADO_TOOLKIT_INTEGRATOR_DISCRETIZED_ODE_HPP


#include <acado/integrator/integrator_fwd.hpp>
#include <acado/integrator/integrator_runge_kutta12.hpp>




BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Implements a scheme for evaluating discretized ODEs.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class IntegratorDiscretizedODE implements a scheme
 *	for evaluating discretized ordinary differential equations (ODEs).
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class IntegratorDiscretizedODE : public IntegratorRK12{


friend class ShootingMethod;

//
// PUBLIC MEMBER FUNCTIONS:
//
public:

    /** Default constructor. */
    IntegratorDiscretizedODE( );

    /** Default constructor. */
    IntegratorDiscretizedODE( const DifferentialEquation &rhs_ );

    /** Copy constructor (deep copy). */
    IntegratorDiscretizedODE( const IntegratorDiscretizedODE& arg );

    /** Destructor. */
    virtual ~IntegratorDiscretizedODE( );

    /** Assignment operator (deep copy). */
    virtual IntegratorDiscretizedODE& operator=( const IntegratorDiscretizedODE& arg );

	/** The (virtual) copy constructor */
	virtual Integrator* clone() const;


	virtual returnValue init( const DifferentialEquation &rhs_ );
	
    virtual returnValue step(	int number  /**< the step number */
								);


//
// PROTECTED MEMBER FUNCTIONS:
//
protected:

    returnValue performDiscreteStep ( const int& number_ );

    returnValue performADforwardStep( const int& number_ );

    returnValue performADbackwardStep( const int& number_ );

    returnValue performADforwardStep2( const int& number_ );

    returnValue performADbackwardStep2( const int& number_ );


//
// PROTECTED MEMBERS:
//
protected:

    double stepLength;
};


CLOSE_NAMESPACE_ACADO



#include <acado/integrator/integrator_discretized_ode.ipp>


#endif  // ACADO_TOOLKIT_INTEGRATOR_DISCRETIZED_ODE_HPP

// end of file.
