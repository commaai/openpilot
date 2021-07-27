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
 *    \file include/acado/nlp_solver/scp_evaluation.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_SCP_EVALUATION_HPP
#define ACADO_TOOLKIT_SCP_EVALUATION_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>

#include <acado/objective/objective.hpp>
#include <acado/dynamic_discretization/dynamic_discretization.hpp>
#include <acado/constraint/constraint.hpp>

#include <acado/function/ocp_iterate.hpp>
#include <acado/conic_program/banded_cp.hpp>



BEGIN_NAMESPACE_ACADO


/**
 *	\brief Base class for different ways to evaluate functions and derivatives within an SCPmethod for solving NLPs.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class SCPevaluation serves as a base class for different ways to evaluate  
 *  functions and derivatives within an SCPmethod for solving nonlinear programming problems.
 *
 *	 \author Boris Houska, Hans Joachim Ferreau
 */
class SCPevaluation : public AlgorithmicBase
{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        SCPevaluation( );

		SCPevaluation(	UserInteraction* _userInteraction,
						const Objective* const objective_,
						const DynamicDiscretization* const dynamic_discretization_,
						const Constraint* const constraint_,
						BooleanType _isCP = BT_FALSE
						);

        /** Copy constructor (deep copy). */
        SCPevaluation( const SCPevaluation& rhs );

        /** Destructor. */
        virtual ~SCPevaluation( );

        /** Assignment operator (deep copy). */
        SCPevaluation& operator=( const SCPevaluation& rhs );

        virtual SCPevaluation* clone() const;


		virtual returnValue init(	const OCPiterate& iter
									);


        /** Evaluates the objective as well as the ODE/DAE discretization
         *  and the constraints for the case that they exist.
         */
        virtual returnValue evaluate( OCPiterate& iter, BandedCP& cp );


        /** Evaluates the sensitivities of the objective as well as
         *  the ODE/DAE discretization
         *  and the constraints for the case that they exist.
         */
        virtual returnValue evaluateSensitivities(	const OCPiterate& iter,
        											BandedCP& cp
        											);

        /** Evaluates the gradient "nablaL" of the Lagrangian function.
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual returnValue evaluateLagrangeGradient(	uint N,
														const OCPiterate& iter,
        												const BandedCP& cp,
														BlockMatrix &nablaL
														);



        /** computes the KKT-tolerance (only for internal termination check). \n
         *
         *  \return The requested KKT tolerance.
         */
        virtual double getKKTtolerance(	const OCPiterate& iter,
        								const BandedCP& cp,
        								double KKTmultiplierRegularisation = 0.0
        								);


        virtual double getObjectiveValue( ) const;






        returnValue setReference(	const VariablesGrid& ref
        							);


		returnValue clearDynamicDiscretization( );


		inline BooleanType hasLSQobjective( ) const;

		inline BooleanType isDynamicNLP( ) const;
		inline BooleanType isStaticNLP( ) const;

		inline uint getNumConstraints( ) const;
		inline uint getNumConstraintBlocks( ) const;
		inline DVector getConstraintBlockDims( ) const;


		virtual returnValue freezeSensitivities( );

		virtual returnValue unfreezeSensitivities( );



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		virtual returnValue setupOptions( );
		virtual returnValue setupLogging( );


    //
    // DATA MEMBERS:
    //
    protected:

        Objective*             objective            ;  /**< Objective function     */
        DynamicDiscretization* dynamicDiscretization;  /**< Descretized ODE or DAE */
        Constraint*            constraint           ;  /**< Constraint functions   */

        double objectiveValue; 	   /**< The objective value. */

		BooleanType isCP;
		BooleanType areSensitivitiesFrozen;
};


CLOSE_NAMESPACE_ACADO



#include <acado/nlp_solver/scp_evaluation.ipp>


#endif  // ACADO_TOOLKIT_SCP_EVALUATION_HPP

/*
 *  end of file
 */
