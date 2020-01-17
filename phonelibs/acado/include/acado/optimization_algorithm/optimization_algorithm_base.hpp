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
 *    \file include/acado/optimization_algorithm/optimization_algorithm_base.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_OPTIMIZATION_ALGORITHM_BASE_HPP
#define ACADO_TOOLKIT_OPTIMIZATION_ALGORITHM_BASE_HPP

#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>
//#include <acado/ocp/ocp.hpp>
#include <acado/nlp_solver/nlp_solver.hpp>
#include <acado/nlp_solver/scp_method.hpp>


BEGIN_NAMESPACE_ACADO

class OCP;


/**
Notes:
  - time is normalised to [0,1] when doing time optimal control.

**/

/** 
 *	\brief Base class for user-interfaces to formulate and solve optimal control problems and static NLPs.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class OptimizationAlgorithmBase serves as a base class for user-interfaces 
 *  to formulate and solve optimal control problems and static nonlinear programming (NLP) 
 *	problems.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */
class OptimizationAlgorithmBase
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        OptimizationAlgorithmBase();

        /** Default constructor. */
        OptimizationAlgorithmBase( const OCP& ocp_ );

        /** Copy constructor (deep copy). */
        OptimizationAlgorithmBase( const OptimizationAlgorithmBase& arg );

        /** Destructor. */
        virtual ~OptimizationAlgorithmBase( );

        /** Assignment operator (deep copy). */
        OptimizationAlgorithmBase& operator=( const OptimizationAlgorithmBase& arg );


        /** Initialization of the optimization variables. */
        returnValue initializeDifferentialStates( const char* fileName , BooleanType autoinit=BT_FALSE);
        returnValue initializeAlgebraicStates   ( const char* fileName , BooleanType autoinit=BT_FALSE);
        returnValue initializeParameters        ( const char* fileName);
        returnValue initializeControls          ( const char* fileName);
        returnValue initializeDisturbances      ( const char* fileName);

        returnValue initializeDifferentialStates( const VariablesGrid &xd_init_ , BooleanType autoinit=BT_FALSE);
        returnValue initializeAlgebraicStates   ( const VariablesGrid &xa_init_ , BooleanType autoinit=BT_FALSE);
        returnValue initializeParameters        ( const VariablesGrid &u_init_);
        returnValue initializeControls          ( const VariablesGrid &p_init_);
        returnValue initializeDisturbances      ( const VariablesGrid &w_init_);

        /** Use this call to overwrite all states by a single shooting initialization.
         *  This function takes the initial state and controls and overwrite all states
         *  apart from the first one by simulation.
         */
        returnValue simulateStatesForInitialization();

        returnValue getDifferentialStates( VariablesGrid &xd_ ) const;
        returnValue getAlgebraicStates   ( VariablesGrid &xa_ ) const;
        returnValue getParameters        ( VariablesGrid &u_  ) const;
        returnValue getParameters        ( DVector &u_  ) const;
        returnValue getControls          ( VariablesGrid &p_  ) const;
        returnValue getDisturbances      ( VariablesGrid &w_  ) const;

        returnValue getDifferentialStates( const char* fileName ) const;
        returnValue getAlgebraicStates   ( const char* fileName ) const;
        returnValue getParameters        ( const char* fileName ) const;
        returnValue getControls          ( const char* fileName ) const;
        returnValue getDisturbances      ( const char* fileName ) const;

        double getObjectiveValue         ( const char* fileName ) const;
        double getObjectiveValue         () const;

		
		returnValue getSensitivitiesX(	BlockMatrix& _sens
										) const;

		returnValue getSensitivitiesXA(	BlockMatrix& _sens
										) const;

		returnValue getSensitivitiesP(	BlockMatrix& _sens
										) const;

		returnValue getSensitivitiesU(	BlockMatrix& _sens
										) const;

		returnValue getSensitivitiesW(	BlockMatrix& _sens
										) const;


		/** Returns number of differential states.
		 *  \return Number of differential states */
		virtual uint getNX( ) const;

		/** Returns number of algebraic states.
		 *  \return Number of algebraic states */
		virtual uint getNXA( ) const;

		/** Returns number of parameters.
		 *  \return Number of parameters */
		virtual uint getNP( ) const;

		/** Returns number of controls.
		 *  \return Number of controls */
		virtual uint getNU( ) const;

		/** Returns number of disturbances.
		 *  \return Number of disturbances */
		virtual uint getNW( ) const;


		double getStartTime ( ) const;

		double getEndTime( ) const;


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		returnValue clear( );

		/** Initializes everything. */
		returnValue init(	UserInteraction* _userIteraction
							);

		BooleanType isLinearQuadratic(	Objective *F,
										DynamicDiscretization *G,
										Constraint *H
										) const;

        virtual returnValue allocateNlpSolver(	Objective *F,
												DynamicDiscretization *G,
												Constraint *H
												) = 0;

        virtual returnValue initializeNlpSolver(	const OCPiterate& _userInit
													) = 0;

        virtual returnValue initializeObjective(	Objective* F
													) = 0;


		virtual returnValue extractOCPdata(	Objective** objective,
											DifferentialEquation*** differentialEquation,
											Constraint** constraint,
											Grid& unionGrid
											);

		virtual returnValue setupObjective(	Objective* objective,
											DifferentialEquation** differentialEquation,
											Constraint* constraint,
											Grid unionGrid
											);

		virtual returnValue setupDifferentialEquation(	Objective* objective,
														DifferentialEquation** differentialEquation,
														Constraint* constraint,
														Grid unionGrid
														);

		virtual returnValue setupDynamicDiscretization(	UserInteraction* _userIteraction,
														Objective* objective,
														DifferentialEquation** differentialEquation,
														Constraint* constraint,
														Grid unionGrid,
														DynamicDiscretization** dynamicDiscretization
														);

		virtual returnValue determineDimensions(	Objective* const _objective,
													DifferentialEquation** const _differentialEquation,
													Constraint* const _constraint,
													uint& _nx,
													uint& _nxa,
													uint& _np,
													uint& _nu,
													uint& _nw
													) const;

		virtual returnValue initializeOCPiterate(	Constraint* const _constraint,
													const Grid& _unionGrid,
													uint nx,
													uint nxa,
													uint np,
													uint nu,
													uint nw
													);

    //
    // DATA MEMBERS:
    //
    protected:

        NLPsolver *nlpSolver ;
        OCP       *ocp       ;

		OCPiterate iter;
		OCPiterate userInit;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_OPTIMIZATION_ALGORITHM_BASE_HPP

/*
 *   end of file
 */
