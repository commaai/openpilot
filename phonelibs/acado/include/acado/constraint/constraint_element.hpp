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
 *    \file include/acado/constraint/constraint_element.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_CONSTRAINT_ELEMENT_HPP
#define ACADO_TOOLKIT_CONSTRAINT_ELEMENT_HPP


#include <acado/symbolic_expression/expression.hpp>
#include <acado/function/function.hpp>
#include <acado/variables_grid/variables_grid.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for all kind of constraints (except for bounds) within optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class ConstraintElement serves as base class for all kind of different 
 *  constraints (except for box constraints) within optimal control problems.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class ConstraintElement{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        ConstraintElement( );

        /** Default constructor. */
        ConstraintElement( const Grid& grid_, int nFcn_, int nB_ );

        /** Copy constructor (deep copy). */
        ConstraintElement( const ConstraintElement& rhs );

        /** Destructor. */
        virtual ~ConstraintElement( );

        /** Assignment operator (deep copy). */
        ConstraintElement& operator=( const ConstraintElement& rhs );



// ==========================================================================
//
//                               INITIALIZATION
//
// ==========================================================================


        /** Initializes the Constraint Element: The dimensions and  \n
         *  index lists.                                            \n
         *  \return SUCCESSFUL_RETURN
         */
        returnValue init(  const OCPiterate& iter  );



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






// =======================================================================================
//
//                               RESULTS OF THE EVALUATION
//
// =======================================================================================


    /** Returns the result for the residuum. \n
     *                                       \n
     *  \return SUCCESSFUL_RETURN            \n
     */
    virtual returnValue getResiduum( BlockMatrix &lower_residuum, /**< the lower residuum */
                                     BlockMatrix &upper_residuum  /**< the upper residuum */ );



    /** Returns the result for the forward sensitivities in BlockMatrix form.        \n
     *                                                                               \n
     *  \return SUCCESSFUL_RETURN                                                    \n
     *          RET_INPUT_OUT_OF_RANGE                                               \n
     */
    virtual returnValue getForwardSensitivities( BlockMatrix *D  /**< the result for the
                                                                  *   forward sensitivi-
                                                                  *   ties               */,
                                                 int order       /**< the order          */  );



    /** Returns the result for the backward sensitivities in BlockMatrix form.       \n
     *                                                                               \n
     *  \return SUCCESSFUL_RETURN                                                    \n
     *          RET_INPUT_OUT_OF_RANGE                                               \n
     */
    virtual returnValue getBackwardSensitivities( BlockMatrix *D  /**< the result for the
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

        /** Returns the number of differential states                 \n
         *  \return The requested number of differential states.      \n
         */
        inline int getNX    () const;

        /** Returns the number of algebraic states                    \n
         *  \return The requested number of algebraic states.         \n
         */
        inline int getNXA   () const;

         /** Returns the number of controls                            \n
          *  \return The requested number of controls.                 \n
          */
        inline int getNU   () const;

        /** Returns the number of parameters                          \n
         *  \return The requested number of parameters.               \n
         */
        inline int getNP   () const;

        /** Returns the number of disturbances                        \n
         *  \return The requested number of disturbances.             \n
         */
        inline int getNW  () const;



        /** returns whether the constraint element is affine. */
        inline BooleanType isAffine() const;

        returnValue get(Function& function_, DMatrix& lb_, DMatrix& ub_);

// ==========================================================================
//
//                          PROTECTED MEMBER FUNCTIONS:
//
// ==========================================================================

    protected:

		virtual returnValue initializeEvaluationPoints(	const OCPiterate& iter
														);



    //
    // DATA MEMBERS:
    //
    protected:


        // DEFINITIONS OF THE CONSTRAINT FUNCTION, GRID, AND BOUNDS:
        // ---------------------------------------------------------

        Grid             grid   ;   /**< the constraint grid     */
        Function        *fcn    ;   /**< the functions           */
        double         **lb     ;   /**< lower bounds            */
        double         **ub     ;   /**< upper bounds            */

        EvaluationPoint  *z      ;   /**< the evaluation points    */
        EvaluationPoint  *JJ     ;


        // LOW_LEVEL EVALUATION INDICES:
        // ----------------------------- 

        int            **y_index;   /**< index lists             */
        int             *t_index;   /**< time indices            */


        // DIMENSIONS:
        // ----------------------

        int              nx     ;   /**< number of diff. states  */
        int              na     ;   /**< number of alg. states   */
        int              nu     ;   /**< number of controls      */
        int              np     ;   /**< number of parameters    */
        int              nw     ;   /**< number of disturbances  */
        int              ny     ;   /**< := nx+na+nu+np+nw       */

        int              nFcn   ;   /**< number of functions     */
        int              nB     ;   /**< number of bounds        */


        // INPUT STORAGE:
        // ------------------------
        BlockMatrix      *xSeed   ;   /**< the 1st order forward seed in x-direction */
        BlockMatrix      *xaSeed  ;   /**< the 1st order forward seed in x-direction */
        BlockMatrix      *pSeed   ;   /**< the 1st order forward seed in p-direction */
        BlockMatrix      *uSeed   ;   /**< the 1st order forward seed in u-direction */
        BlockMatrix      *wSeed   ;   /**< the 1st order forward seed in w-direction */

        BlockMatrix      *bSeed   ;   /**< the 1st order backward seed */

        BlockMatrix      *xSeed2  ;   /**< the 2nd order forward seed in x-direction */
        BlockMatrix      *xaSeed2 ;   /**< the 2nd order forward seed in x-direction */
        BlockMatrix      *pSeed2  ;   /**< the 2nd order forward seed in p-direction */
        BlockMatrix      *uSeed2  ;   /**< the 2nd order forward seed in u-direction */
        BlockMatrix      *wSeed2  ;   /**< the 2nd order forward seed in w-direction */

        BlockMatrix      *bSeed2  ;   /**< the 2nd order backward seed */


        // RESULTS:
        // ------------------------
        BlockMatrix      residuumL;   /**< the residuum vectors */
        BlockMatrix      residuumU;   /**< the residuum vectors */

        BlockMatrix      dForward ;   /**< the first order forward  derivatives */
        BlockMatrix      dBackward;   /**< the first order backward derivatives */

        CondensingType   condType ;   /**< the condensing type */

};


CLOSE_NAMESPACE_ACADO



#include <acado/constraint/constraint_element.ipp>


#endif  // ACADO_TOOLKIT_CONSTRAINT_ELEMENT_HPP

/*
 *    end of file
 */
