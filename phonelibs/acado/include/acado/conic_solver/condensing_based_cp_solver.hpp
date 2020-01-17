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
 *    \file include/acado/conic_solver/condensing_based_cp_solver.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_CONDENSING_BASED_CP_SOLVER_HPP
#define ACADO_TOOLKIT_CONDENSING_BASED_CP_SOLVER_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/conic_solver/banded_cp_solver.hpp>
#include <acado/conic_solver/dense_qp_solver.hpp>



BEGIN_NAMESPACE_ACADO


/**
 *	\brief Solves banded conic programs arising in optimal control using condensing.
 *
 *	\ingroup NumericalAlgorithm
 *
 *  The class condensing based CP solver is a special solver for
 *  band structured conic programs that can be solved via a
 *  condensing technique.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */

class CondensingBasedCPsolver: public BandedCPsolver {


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        CondensingBasedCPsolver( );
		
        CondensingBasedCPsolver(	UserInteraction* _userInteraction,
									uint nConstraints_,
        							const DVector& blockDims_
        							);

        /** Copy constructor (deep copy). */
        CondensingBasedCPsolver( const CondensingBasedCPsolver& rhs );

        /** Destructor. */
        virtual ~CondensingBasedCPsolver( );

        /** Assignment operator (deep copy). */
        CondensingBasedCPsolver& operator=( const CondensingBasedCPsolver& rhs );


        /** Assignment operator (deep copy). */
        virtual BandedCPsolver* clone() const;


        /** initializes the banded conic solver */
        virtual returnValue init( const OCPiterate &iter_ );


        /** Solves a given banded conic program */
        virtual returnValue prepareSolve(	BandedCP& cp
											);

		/** Solves a given banded conic program in feedback mode:                   \n
         *                                                                          \n
         *  \param cp     the banded conic program to be solved                     \n
         *  \param DeltaX difference between state estimate and previous prediction \n
         *  \param DeltaP difference between current and previous parameter value   \n
         *                                                                          \n
         *  \return SUCCESSFUL_RETURN   (if successful)                             \n
         *          or a specific error message from the dense CP solver.           \n
         */
        virtual returnValue solve(	BandedCP& cp
									);

        /** Solves a given banded conic program */
        virtual returnValue finalizeSolve(	BandedCP& cp
											);


		inline uint getNX( ) const;
		inline uint getNXA( ) const;
		inline uint getNP( ) const;
		inline uint getNU( ) const;
		inline uint getNW( ) const;

		inline uint getNC( ) const;
		inline uint getNF( ) const;
		inline uint getNA( ) const;

		inline uint getNumPoints( ) const;


		virtual returnValue getParameters        ( DVector        &p_  ) const;
		virtual returnValue getFirstControl      ( DVector        &u0_ ) const;


        /** Returns a variance-covariance estimate if possible or an error message otherwise.
         *
         *  \return SUCCESSFUL_RETURN
         *          RET_MEMBER_NOT_INITIALISED
         */
        virtual returnValue getVarianceCovariance( DMatrix &var );

		
		virtual returnValue setRealTimeParameters(	const DVector& DeltaX,
													const DVector& DeltaP = emptyConstVector
													);

		inline BooleanType areRealTimeParametersDefined( ) const;


		virtual returnValue freezeCondensing( );

		virtual returnValue unfreezeCondensing( );



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

        /** Initializes QP objects.
		 *  \return SUCCESSFUL_RETURN \n
		 *          RET_QP_INIT_FAILED */
        virtual returnValue initializeCPsolver(	InfeasibleQPhandling infeasibleQPhandling
												);


        /** Solves current QP (or relaxation) using not more than the 
		 *  given number of iterations.
		 *  \return SUCCESSFUL_RETURN \n
		 *          RET_QP_INIT_FAILED */
		virtual returnValue solveQP(	uint maxIter,				/**< Maximum number of iterations. */
										InfeasibleQPhandling infeasibleQPhandling = IQH_UNDEFINED
										);





        virtual returnValue solveCPsubproblem( );



        /** Checks whether the Hessian is positive definite and projects \n
         *  the Hessian based on a heuristic damping factor. If this     \n
         *  damping factor is smaller than 0, the routine does nothing.  \n
         *                                                               \n
         *  \return SUCCESSFUL_RETURN.                                   \n
         */
        returnValue projectHessian( DMatrix &H_, double dampingFactor );


		// --------
		// SQP DATA
		// --------

        /** Performes the condensing of the dynamic system if necessary.
         */
        returnValue condense(	BandedCP& cp
								);


        /** Expands the KKT system if necessary.
         */
        returnValue expand(		BandedCP& cp
								);


        returnValue generateHessianBlockLine   ( uint nn, uint rowOffset, uint& rowOffset1 );
        returnValue generateConstraintBlockLine( uint nn, uint rowOffset, uint& rowOffset1 );
        returnValue generateStateBoundBlockLine( uint nn, uint rowOffset, uint& rowOffset1 );
        returnValue generateConstraintVectors  ( uint nn, uint rowOffset, uint& rowOffset1 );
        returnValue generateStateBoundVectors  ( uint nn, uint rowOffset, uint& rowOffset1 );


        returnValue generateBoundVectors     ( );
        returnValue generateObjectiveGradient( );


        returnValue initializeCondensingOperator( );
		
		returnValue computeCondensingOperator(	BandedCP& cp
												);


        /** Determines relaxed (constraints') bounds of an infeasible QP. */
        virtual returnValue setupRelaxedQPdata(	InfeasibleQPhandling infeasibleQPhandling,
												DenseCP& _denseCPrelaxed					/**< OUTPUT: Relaxed QP data. */
												) const;

        /** Determines relaxed (constraints') bounds of an infeasible QP. */
        virtual returnValue setupRelaxedQPdataL1(	DenseCP& _denseCPrelaxed					/**< OUTPUT: Relaxed QP data. */
													) const;

        /** Determines relaxed (constraints') bounds of an infeasible QP. */
        virtual returnValue setupRelaxedQPdataL2(	DenseCP& _denseCPrelaxed					/**< OUTPUT: Relaxed QP data. */
													) const;



    //
    // DATA MEMBERS:
    //
    protected:

        OCPiterate iter;
        DVector blockDims;
        uint nConstraints;

		CondensingStatus condensingStatus;


        // THE CONDENSING OPERATORS:
        // -----------------------------------------------------
        BlockMatrix   T;    /**< the condensing operator */
        BlockMatrix   d;    /**< the condensing offset   */

		BlockMatrix  hT;
        // ------------------------------------------------


        // DENSE QP IN BLOCK-MATRIX FORM:
        // ----------------------------------------------------------------------

        BlockMatrix        HDense;    /**< Hessian after condensing            */
        BlockMatrix        gDense;    /**< Objective gradient after condensing */
        BlockMatrix        ADense;    /**< Constraint matrix                   */
        BlockMatrix      lbADense;    /**< Constraint lower bounds             */
        BlockMatrix      ubADense;    /**< Constraint upper bounds             */
        BlockMatrix       lbDense;    /**< Simple lower bounds                 */
        BlockMatrix       ubDense;    /**< Simple upper bounds                 */
        // ----------------------------------------------------------------------


        DenseCPsolver* cpSolver;
        DenseQPsolver* cpSolverRelaxed;

        DenseCP        denseCP;

		DVector deltaX;
		DVector deltaP;
};


CLOSE_NAMESPACE_ACADO


#include <acado/conic_solver/condensing_based_cp_solver.ipp>


#endif  // ACADO_TOOLKIT_CONDENSING_BASED_CP_SOLVER_HPP

/*
 *  end of file
 */
