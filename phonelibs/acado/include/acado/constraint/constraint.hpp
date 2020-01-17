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
 *    \file include/acado/constraint/constraint.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_CONSTRAINT_HPP
#define ACADO_TOOLKIT_CONSTRAINT_HPP

#include <acado/constraint/box_constraint.hpp>
#include <acado/constraint/boundary_constraint.hpp>
#include <acado/constraint/coupled_path_constraint.hpp>
#include <acado/constraint/path_constraint.hpp>
#include <acado/constraint/algebraic_consistency_constraint.hpp>
#include <acado/constraint/point_constraint.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Stores and evaluates the constraints of optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class Constraint allows to manage and evaluate the constraints
 *	of optimal control problems. It consists of a list of all different 
 *	types of constraints that are derived from the base class ConstraintElement.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class Constraint : public BoxConstraint{

	friend class OptimizationAlgorithmBase;
	friend class OptimizationAlgorithm;
	friend class RealTimeAlgorithm;
	friend class TESTExport;
	friend class ExportNLPSolver;

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        Constraint( );

        /** Copy constructor (deep copy). */
        Constraint( const Constraint& rhs );

        /** Destructor. */
        virtual ~Constraint( );

        /** Assignment operator (deep copy). */
        Constraint& operator=( const Constraint& rhs );


        /** Initializes the constraint.
         *
         *  \param grid_            the discretization grid.
         *  \param numberOfStages_  the number of stages (default = 1).
         *
         *  \return SUCCESSFUL_RETURN
         */
        returnValue init( const Grid& grid_, const int& numberOfStages_ = 1 );


        // ===========================================================================
        //
        //                          CONTINOUS CONSTRAINTS
        //                       --------------------------
        //
        //
        //     (general form  lb(i) <= h( t,x(i),u(i),p,... ) <= ub(i)  for all i)
        //
        // ===========================================================================


        /**< adds a constraint of the form  lb_ <= arg <= ub  with constant lower \n
         *   and upper bounds.
         *
         *   \return  SUCCESSFUL_RETURN
         *            RET_INFEASIBLE_CONSTRAINT
         *
         */
        returnValue add( const double lb_, const Expression& arg, const double ub_  );


        /**< adds a constraint of the form  lb_ <= arg <= ub where the   \n
         *   lower bound is varying over the grid points.                 \n
         *
         *   \return  SUCCESSFUL_RETURN
         *            RET_INFEASIBLE_CONSTRAINT
         *
         */
        returnValue add( const DVector lb_, const Expression& arg, const double ub_  );


        /**< adds a constraint of the form  lb_ <= arg <= ub where the   \n
         *   upper bound is varying over the grid points.                 \n
         *
         *   \return  SUCCESSFUL_RETURN
         *            RET_INFEASIBLE_CONSTRAINT
         *
         */
        returnValue add( const double lb_, const Expression& arg, const DVector ub_  );


        /**< adds a constraint of the form  lb_ <= arg <= ub where the   \n
         *   upper and the lower bound are varying over the grid points.  \n
         *
         *   \return  SUCCESSFUL_RETURN
         *            RET_INFEASIBLE_CONSTRAINT
         *
         */
        returnValue add( const DVector lb_, const Expression& arg, const DVector ub_  );


        // ===========================================================================
        //
        //                           DISCRETE CONSTRAINTS
        //                       --------------------------
        //
        //
        //   (general form  lb(i) <= h( t,x(i),u(i),p,... ) <= ub(i)  for a given i)
        //
        // ===========================================================================


        /**< adds a constraint of the form  lb_ <= arg <= ub  with constant lower \n
         *   and upper bounds.                                                    \n
         *
         *   \return  SUCCESSFUL_RETURN
         *            RET_INFEASIBLE_CONSTRAINT
         *
         */
        returnValue add( const int index_, const double lb_, const Expression& arg, const double ub_  );


        // ===========================================================================
        //
        //                       COUPLED BOUNDARY CONSTRAINTS
        //                       ----------------------------
        //
        //
        //   (general form  lb <=   h_1( t_0,x(t_0),u(t_0),p,... )
        //                        + h_2( t_e,x(t_e),u(t_e),p,... ) <= ub(i)  )
        //
        //    where t_0 is the first and t_e the last time point in the grid.
        //
        // ===========================================================================

        /**< adds a constraint of the form  lb_ <= arg1(0) + arg_2(T) <= ub  with   \n
         *   constant lower and upper bounds.                                       \n
         *
         *   \return  SUCCESSFUL_RETURN
         *            RET_INFEASIBLE_CONSTRAINT
         *
         */
        returnValue add( const double lb_, const Expression& arg1,
                                const Expression& arg2, const double ub_ );



        // ===========================================================================
        //
        //                         GENERAL COUPLED CONSTRAINTS
        //                       -------------------------------
        //
        //
        //   (general form  lb <= sum_i  h_i( t_i,x(t_i),u(t_i),p,... ) <= ub(i)  )
        //
        //
        // ===========================================================================

        /**< adds a constraint of the form  lb_ <= sum_i arg_i(t_i) <= ub  with     \n
         *   constant lower and upper bounds.                                       \n
         *
         *   \return  SUCCESSFUL_RETURN
         *            RET_INFEASIBLE_CONSTRAINT
         *
         */
        returnValue add( const double lb_, const Expression *arguments, const double ub_ );




        // ===========================================================================
        //
        //                        ALGEBRAIC CONSISTENCY CONSTRAINTS
        //                       -----------------------------------
        //
        // ===========================================================================

        /**  Adds an algebraic consistency constraint for a specified stage. This    \n
         *   method is rather for internal use, as the optimization routine will     \n
         *   care about the transformation of DAE optimization problems adding       \n
         *   consistency constraints, if necessary. Note that the number of stages   \n
         *   has to be specified in advance within the constructor of the constraint.\n
         *   Actually, the "endOfStage" does at the same specify the start of        \n
         *   the next stage. Thus the DAE's should be added in the correct order.    \n
         *
         *   \return SUCCESSFUL_RETURN
         *           RET_INDEX_OUT_OF_BOUNDS
         */
        returnValue add( const uint&                 endOfStage_  ,  /**< end of the stage   */
                                const DifferentialEquation& dae             /**< the DAE itself     */ );



// =======================================================================================
//
//                                   LOADING ROUTINES
//
// =======================================================================================


        /**< adds a (continuous) contraint.                \n
          *  \return SUCCESSFUL_RETURN
          *          RET_INFEASIBLE_CONSTRAINT
          */
        returnValue add( const ConstraintComponent& component );


        /**< adds a (discrete) contraint.                  \n
          *  \return SUCCESSFUL_RETURN
          *          RET_INFEASIBLE_CONSTRAINT
          */
        returnValue add( const int index_, const ConstraintComponent& component );




// =======================================================================================
//
//                                  DEFINITION OF SEEDS:
//
// =======================================================================================


    /** Define a forward seed in form of a block matrix.   \n
     *                                                     \n
     *  \return SUCCESFUL RETURN                           \n
     *          RET_INPUT_OUT_OF_RANGE                     \n
     */
    virtual returnValue setForwardSeed( BlockMatrix *xSeed_ ,   /**< the seed in x -direction */
                                        BlockMatrix *xaSeed_,   /**< the seed in xa-direction */
                                        BlockMatrix *pSeed_ ,   /**< the seed in p -direction */
                                        BlockMatrix *uSeed_ ,   /**< the seed in u -direction */
                                        BlockMatrix *wSeed_ ,   /**< the seed in w -direction */
                                        int          order      /**< the order of the  seed. */ );


    /**  Defines the first order forward seed to be         \n
     *   the unit-directions matrix.                        \n
     *                                                      \n
     *   \return SUCCESFUL_RETURN                           \n
     *           RET_INPUT_OUT_OF_RANGE                     \n
     */
    virtual returnValue setUnitForwardSeed( );



    /**  Define a backward seed in form of a block matrix.  \n
     *                                                      \n
     *   \return SUCCESFUL_RETURN                           \n
     *           RET_INPUT_OUT_OF_RANGE                     \n
     */
    virtual returnValue setBackwardSeed( BlockMatrix *seed,    /**< the seed matrix       */
                                         int          order    /**< the order of the seed.*/  );



    /**  Defines the first order backward seed to be        \n
     *   a unit matrix.                                     \n
     *                                                      \n
     *   \return SUCCESFUL_RETURN                           \n
     *           RET_INPUT_OUT_OF_RANGE                     \n
     */
    virtual returnValue setUnitBackwardSeed( );



// =======================================================================================
//
//                                   EVALUATION ROUTINES
//
// =======================================================================================



        returnValue evaluate( const OCPiterate& iter );




        returnValue evaluateSensitivities();


        /**  Return the sensitivities and the hessian term contribution of the constraint \n
         *   components. The seed should be a (1 x getNumberOfBlocks())-matrix, which are \n
         *   is in an optimization context the multiplier associated with the constraint. \n
         *                                                                                \n
         *   \return SUCCESSFUL_RETURN                                                    \n
         */
        returnValue evaluateSensitivities( const BlockMatrix &seed, BlockMatrix &hessian );




// =======================================================================================
//
//                               RESULTS OF THE EVALUATION
//
// =======================================================================================


    /** Returns the result for the residuum of the constraints. \n
     *                                                          \n
     *  \return SUCCESSFUL_RETURN                               \n
     */
    virtual returnValue getConstraintResiduum( BlockMatrix &lowerRes, /**< the lower residuum */
                                               BlockMatrix &upperRes  /**< the upper residuum */ );


    /** Returns the result for the residuum of the bounds.      \n
     *                                                          \n
     *  \return SUCCESSFUL_RETURN                               \n
     */
    virtual returnValue getBoundResiduum( BlockMatrix &lowerRes, /**< the lower residuum */
                                          BlockMatrix &upperRes  /**< the upper residuum */ );



    /** Returns the result for the forward sensitivities in BlockMatrix form.        \n
     *                                                                               \n
     *  \return SUCCESSFUL_RETURN                                                    \n
     *          RET_INPUT_OUT_OF_RANGE                                               \n
     */
    virtual returnValue getForwardSensitivities( BlockMatrix &D  /**< the result for the
                                                                  *   forward sensitivi-
                                                                  *   ties               */,
                                                 int order       /**< the order          */  );



    /** Returns the result for the backward sensitivities in BlockMatrix form.       \n
     *                                                                               \n
     *  \return SUCCESSFUL_RETURN                                                    \n
     *          RET_INPUT_OUT_OF_RANGE                                               \n
     */
    virtual returnValue getBackwardSensitivities( BlockMatrix &D  /**< the result for the
                                                                   *   forward sensitivi-
                                                                   *   ties               */,
                                                  int order       /**< the order          */  );





//  =========================================================================
//
//                               MISCELLANEOUS:
//
//  =========================================================================

        /** returns the constraint grid */
        inline Grid& getGrid();

        /** returns the number of constraints */
        inline int getNC();

        /** Returns the number of differential states                 \n
         *  \return The requested number of differential states.      \n
         */
        inline int getNX    () const;

        /** Returns the number of algebraic states                    \n
         *  \return The requested number of algebraic states.         \n
         */
        inline int getNXA   () const;

        /** Returns the number of parameters                          \n
         *  \return The requested number of parameters.               \n
         */
        inline int getNP   () const;

        /** Returns the number of controls                            \n
         *  \return The requested number of controls.                 \n
         */
        inline int getNU   () const;

        /** Returns the number of disturbances                        \n
         *  \return The requested number of disturbances.             \n
         */
        inline int getNW  () const;


        /** returns the number of constraint blocks */
        inline int getNumberOfBlocks() const;


        /** returns the dimension of the requested sub-block */
        inline int getBlockDim( int idx ) const;

        /** returns the dimension of the requested sub-block */
        inline DVector getBlockDims( ) const;



        /** returns whether the constraint is affine. */
        inline BooleanType isAffine() const;

		/** returns whether object only comprises box constraints. */
		inline BooleanType isBoxConstraint() const;

        /** Returns whether or not the constraint is empty.    \n
         *                                                     \n
         *  \return BT_TRUE if no constraint is specified yet. \n
         *          BT_FALSE otherwise.                        \n
         */
        BooleanType isEmpty() const;

        returnValue getPathConstraints(Function& function_, DMatrix& lb_, DMatrix& ub_) const;

        returnValue getPointConstraint(const unsigned index, Function& function_, DMatrix& lb_, DMatrix& ub_) const;

    //
    // DATA MEMBERS:
    //
    protected:


        BoundaryConstraint              *boundary_constraint             ;
        CoupledPathConstraint           *coupled_path_constraint         ;
        PathConstraint                  *path_constraint                 ;
        AlgebraicConsistencyConstraint  *algebraic_consistency_constraint;
        PointConstraint                **point_constraints               ;


    // PROTECTED MEMBER FUNCTIONS:
    // ---------------------------

    protected:

        /** CAUTION: This function is protected and stictly for internal use.
         *  Note that e.g. the expression pointer will be deleted when using this function.
         */
        returnValue add( const int index_, const double lb_, Expression* arg, const double ub_  );
		
        /** CAUTION: This function is protected and strictly for internal use.
         *  Note that e.g. the expression pointer will be deleted when using this function.
         */
        returnValue add( const DVector lb_, Expression* arg, const DVector ub_  );


        /** Writes a special copy of the bounds that is needed within the
         *  OptimizationAlgorithm into the optimization variables.
         */
        virtual returnValue getBounds( const OCPiterate& iter );

        /** Protected version of the destructor. */
        void deleteAll();

};


CLOSE_NAMESPACE_ACADO



#include <acado/constraint/constraint.ipp>


#endif  // ACADO_TOOLKIT_CONSTRAINT_HPP

/*
 *    end of file
 */
