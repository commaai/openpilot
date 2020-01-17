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
 *    \file include/acado/nlp_solver/scp_method.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_SCP_METHOD_HPP
#define ACADO_TOOLKIT_SCP_METHOD_HPP


#include <acado/utils/acado_utils.hpp>

#include <acado/function/ocp_iterate.hpp>

#include <acado/nlp_solver/nlp_solver.hpp>
#include <acado/conic_solver/dense_qp_solver.hpp>
#include <acado/conic_solver/banded_cp_solver.hpp>
#include <acado/conic_solver/condensing_based_cp_solver.hpp>

#include <acado/nlp_solver/scp_evaluation.hpp>
#include <acado/nlp_solver/scp_step_linesearch.hpp>
#include <acado/nlp_solver/scp_step_fullstep.hpp>
#include <acado/nlp_derivative_approximation/nlp_derivative_approximation.hpp>



BEGIN_NAMESPACE_ACADO


/**
 *	\brief Implements different sequential convex programming methods for solving NLPs.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class SCPmethod implements different sequential convex programming methods 
 *  for solving nonlinear programming problems.
 *
 *	 \author Boris Houska, Hans Joachim Ferreau
 */
class SCPmethod : public NLPsolver
{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        SCPmethod( );

        /** Default constructor. */
        SCPmethod(	UserInteraction* _userInteraction,
					const Objective             *objective_             ,
					const DynamicDiscretization *dynamic_discretization_,
					const Constraint            *constraint_,
					BooleanType _isCP = BT_FALSE
					);

        /** Copy constructor (deep copy). */
        SCPmethod( const SCPmethod& rhs );

        /** Destructor. */
        virtual ~SCPmethod( );

        /** Assignment operator (deep copy). */
        SCPmethod& operator=( const SCPmethod& rhs );

        virtual NLPsolver* clone() const;


        /** Initialization. */
		virtual returnValue init(	VariablesGrid* x_init ,
									VariablesGrid* xa_init,
									VariablesGrid* p_init ,
									VariablesGrid* u_init ,
									VariablesGrid* w_init  );


        /** Solves current, possibly parametric, optimization problem. */
        virtual returnValue solve(	const DVector &x0_ = emptyConstVector,
									const DVector &p_ = emptyConstVector
									);

        /** Executes a complete real-time step. */
        virtual returnValue step(	const DVector &x0_ = emptyConstVector,
									const DVector &p_ = emptyConstVector
									);

        /** Executes a real-time feedback step */
        virtual returnValue feedbackStep(	const DVector &x0_,
											const DVector &p_ = emptyConstVector
											);

		virtual returnValue performCurrentStep( );
		

		/** Executes a real-time preparation step */
        virtual returnValue prepareNextStep( );


        /** Sets the reference to be used in the LSQ tracking terms. If the objective     \n
         *  has also non-LSQ terms or no LSQ terms, an error message will be returned     \n
         *  (cf. objective.hpp).                                                          \n
         *  This routine has been designed for real-time applications where the reference \n
         *  is explicitly time-dependent.                                                 \n
         *                                                                                \n
         *  \return SUCCESSFUL_RETURN                                                     \n
         */
        virtual returnValue setReference(	const VariablesGrid &ref
											);

// 		virtual returnValue enableNeedToReevaluate( );
		
											
		/** Shifts the data for preparating the next real-time step.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED
		 */
		virtual returnValue shiftVariables(	double timeShift = -1.0,
									DVector  lastX    =  emptyVector,
									DVector lastXA    =  emptyVector,
									DVector lastP     =  emptyVector,
									DVector lastU     =  emptyVector,
									DVector lastW     =  emptyVector  );

		
        /** Returns a variance-covariance estimate if possible or an error message otherwise.
         *
         *  \return SUCCESSFUL_RETURN
         *          RET_MEMBER_NOT_INITIALISED
         */
        virtual returnValue getVarianceCovariance( DMatrix &var );


        /** Prints the run-time profile. This routine \n
         *  can be used after an integration run in   \n
         *  order to assess the performance.          \n
         */
        virtual returnValue printRuntimeProfile() const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		virtual returnValue setupLogging( );


		returnValue setup( );
		

        /** Prints the actual values of x, xa, p, u, and w.
         */
        returnValue printIterate( ) const;

		returnValue printIteration( );
		
        returnValue checkForConvergence( );
		

		returnValue computeHessianMatrix(	const BlockMatrix& oldLagrangeGradient,
											const BlockMatrix& newLagrangeGradient
											);


        returnValue initializeHessianProjection( );


		returnValue checkForRealTimeMode(	const DVector &x0_,
											const DVector &p_
											);

		returnValue setupRealTimeParameters(	const DVector &x0_ = emptyConstVector,
												const DVector &p_ = emptyConstVector
												);


		returnValue stopClockAndPrintRuntimeProfile( );


        virtual returnValue getDifferentialStates( VariablesGrid &xd_ ) const;
        virtual returnValue getAlgebraicStates   ( VariablesGrid &xa_ ) const;
        virtual returnValue getParameters        ( VariablesGrid &p_  ) const;
		virtual returnValue getParameters        ( DVector        &p_  ) const;
        virtual returnValue getControls          ( VariablesGrid &u_  ) const;
		virtual returnValue getFirstControl      ( DVector        &u0_ ) const;
        virtual returnValue getDisturbances      ( VariablesGrid &w_  ) const;
        virtual double getObjectiveValue         (                    ) const;


		virtual returnValue getSensitivitiesX(	BlockMatrix& _sens
												) const;

		virtual returnValue getSensitivitiesXA(	BlockMatrix& _sens
													) const;

		virtual returnValue getSensitivitiesP(	BlockMatrix& _sens
												) const;

		virtual returnValue getSensitivitiesU(	BlockMatrix& _sens
													) const;

		virtual returnValue getSensitivitiesW(	BlockMatrix& _sens
												) const;

		virtual returnValue getAnySensitivities(	BlockMatrix& _sens,
													uint idx
													) const;
											

		inline uint getNumPoints( ) const;

		inline uint getNX( ) const;
		inline uint getNXA( ) const;
		inline uint getNP( ) const;
		inline uint getNU( ) const;
		inline uint getNW( ) const;

		inline uint getNC( ) const;


    //
    // DATA MEMBERS:
    //
    protected:

		int timeLoggingIdx;
		RealClock clock;
		RealClock clockTotalTime;

		OCPiterate iter;
		OCPiterate oldIter;

		SCPevaluation* eval;
		SCPstep* scpStep;
		NLPderivativeApproximation* derivativeApproximation;

		BandedCP        bandedCP;
		BandedCPsolver* bandedCPsolver;
		
		BlockStatus status;
		BooleanType isCP;
		
		BooleanType hasPerformedStep;
		BooleanType isInRealTimeMode;
		BooleanType needToReevaluate;
};


CLOSE_NAMESPACE_ACADO



#include <acado/nlp_solver/scp_method.ipp>


#endif  // ACADO_TOOLKIT_SCP_METHOD_HPP

/*
 *  end of file
 */
