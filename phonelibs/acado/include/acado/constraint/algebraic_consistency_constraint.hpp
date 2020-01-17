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
 *    \file include/acado/constraint/algebraic_consistency_constraint.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_ALGEBRAIC_CONSISTENCY_CONSTRAINT_HPP
#define ACADO_TOOLKIT_ALGEBRAIC_CONSISTENCY_CONSTRAINT_HPP


#include <acado/constraint/constraint_element.hpp>


BEGIN_NAMESPACE_ACADO


/**   
 *	\brief Deals with algebraic consistency constraints within optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class AlgebraicConsistencyConstraint has been introduced in order to deal with
 *  algebraic consistency constraints. Usually, the user does not get in touch with this
 *  class, as algebraic consistency constraints should automatically be introduced by
 *  the optimization routine. It is possible to add several DAE right-hand sides which
 *  will be evaluated on a grid (usually, the multiple shooting ot collocation grid)
 *  depending on the model stage number. In the easiest case, i.e. for one stage, the
 *  evaluation routine will return the algebraic residuum of the DAE at a given point.
 *  The differentiation of the AlgebraicConsistencyConstraint will return the derivatives
 *  in form of a structured block matrix, which is typically sparse.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 *
 */


class AlgebraicConsistencyConstraint : public ConstraintElement{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        AlgebraicConsistencyConstraint( );

        /** Default constructor. */
        AlgebraicConsistencyConstraint( const Grid& grid_         , /**< union grid of the constraint */
                                        const uint& numberOfStages  /**< number of model stages       */ );

        /** Copy constructor (deep copy). */
        AlgebraicConsistencyConstraint( const AlgebraicConsistencyConstraint& rhs );

        /** Destructor. */
        virtual ~AlgebraicConsistencyConstraint( );

        /** Assignment operator (deep copy). */
        AlgebraicConsistencyConstraint& operator=( const AlgebraicConsistencyConstraint& rhs );


// =======================================================================================

        /**  Adds a consistency constraint for a specified stage.
         *   \return SUCCESSFUL_RETURN
         *           RET_INDEX_OUT_OF_BOUNDS (if the start/end of the stage are not well defined)
         */
        inline returnValue add( const uint&                 endOfStage_  ,  /**< end of the stage   */
                                const DifferentialEquation& dae             /**< the DAE itself     */ );


// =======================================================================================
//
//                                   EVALUATION ROUTINES
//
// =======================================================================================


        /** Evaluates all components in this constraint and stores the \n
          * residuum.                                                  \n
          *                                                            \n
          * \return SUCESSFUL_RETURN                                   \n
          */
        returnValue evaluate( const OCPiterate& iter );


        /** Evaluates the sensitivities of all components in this      \n
          * constraint. Note that the seed can be defined via the base \n
          * class ConstraintElement.                                   \n
          *                                                            \n
          * \return SUCESSFUL_RETURN                                   \n
          */
        returnValue evaluateSensitivities();



        /** Evaluates the sensitivities and Hessian.                   \n
          *                                                            \n
          * \return SUCESSFUL_RETURN                                   \n
          */
        returnValue evaluateSensitivities( int &count, const BlockMatrix &seed, BlockMatrix &hessian );


//  =========================================================================

        /** returns the number of constraints */
        inline int getNC() const;

        /** returns the dimensions of the idx-th block */
        inline int getDim( const int& idx_ );


    // PROOTECTED MEMBER FUNCIONS:
    // ---------------------------
    protected:

		virtual returnValue initializeEvaluationPoints(	const OCPiterate& iter
														);

		
        /** only for internal use (routine which computes a part of the block
         *  matrix needed for forward differentiation.) */
        inline returnValue computeForwardSensitivityBlock( int offset1, int offset2, int offset3, int stageIdx, DMatrix *seed );




    //
    // DATA MEMBERS:
    //
    protected:

        int  numberOfStages            ;
        int  counter                   ;
        int *numberOfDifferentialStates;
        int *numberOfAlgebraicStates   ;
        int *breakPoints               ;
};


CLOSE_NAMESPACE_ACADO



#include <acado/constraint/algebraic_consistency_constraint.ipp>


#endif  // ACADO_TOOLKIT_ALGEBRAIC_CONSISTENCY_CONSTRAINT_HPP

/*
 *    end of file
 */
