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
 *    \file include/acado/nlp_solver/nlp_solver.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_NLP_SOLVER_HPP
#define ACADO_TOOLKIT_NLP_SOLVER_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>

#include <acado/variables_grid/variables_grid.hpp>
#include <acado/clock/real_clock.hpp>



BEGIN_NAMESPACE_ACADO



/**
 *	\brief Base class for different algorithms for solving nonlinear programming (NLP) problems.
 *
 *	\ingroup AlgorithmInterfaces
 *
 *  The class NLPsolver serves as a base class for all kind of different
 *  algorithms for solving nonlinear programming (NLP) problems.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */
class NLPsolver : public AlgorithmicBase
{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        NLPsolver(	UserInteraction* _userInteraction = 0
					);

        /** Copy constructor (deep copy). */
        NLPsolver( const NLPsolver& rhs );

        /** Destructor. */
        virtual ~NLPsolver( );

        /** Assignment operator (deep copy). */
        NLPsolver& operator=( const NLPsolver& rhs );

        virtual NLPsolver* clone( ) const = 0;


        /** Initializes the problem. */
        virtual returnValue init(VariablesGrid    *xd,
                                 VariablesGrid    *xa,
                                 VariablesGrid    *p,
                                 VariablesGrid    *u,
                                 VariablesGrid    *w   ) = 0;


        /** Solves current real-time optimization problem. */
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

        /** Executes a real-time preparation step */
        virtual returnValue performCurrentStep( );

        /** Executes a real-time preparation step */
        virtual returnValue prepareNextStep( );

        /** Applies a shift of the SQP data (for moving horizons) */
        virtual returnValue shiftVariables(	double timeShift,
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
        virtual returnValue getVarianceCovariance( DMatrix &var ) = 0;


        /** Sets the reference to be used in the LSQ tracking terms. If the objective     \n
         *  has also non-LSQ terms a error message will be returned (cf. objective.hpp).  \n
         *  This routine has been designed for real-time applications where the reference \n
         *  is explicitly time-dependent.                                                 \n
         *                                                                                \n
         *  \return SUCCESSFUL_RETURN                                                     \n
         */
        virtual returnValue setReference( const VariablesGrid &ref );

// 		virtual returnValue enableNeedToReevaluate( ) = 0;


		inline int getNumberOfSteps( ) const;

		inline returnValue resetNumberOfSteps( );



        virtual returnValue getDifferentialStates( VariablesGrid &xd_ ) const;
        virtual returnValue getAlgebraicStates   ( VariablesGrid &xa_ ) const;
        virtual returnValue getParameters        ( VariablesGrid &p_  ) const;
		virtual returnValue getParameters        ( DVector        &p_  ) const;
        virtual returnValue getControls          ( VariablesGrid &u_  ) const;
		virtual returnValue getFirstControl      ( DVector        &u0_ ) const;
        virtual returnValue getDisturbances      ( VariablesGrid &w_  ) const;
        virtual double      getObjectiveValue    (                    ) const;

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

        int numberOfSteps;
};


CLOSE_NAMESPACE_ACADO



#include <acado/nlp_solver/nlp_solver.ipp>


#endif  // ACADO_TOOLKIT_NLP_SOLVER_HPP

/*
 *   end of file
 */
