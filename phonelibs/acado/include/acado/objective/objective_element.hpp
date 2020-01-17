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
 *    \file include/acado/objective/objective_element.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_OBJECTIVE_ELEMENT_HPP
#define ACADO_TOOLKIT_OBJECTIVE_ELEMENT_HPP


#include <acado/variables_grid/variables_grid.hpp>
#include <acado/function/function.hpp>

BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for all kind of objective function terms within optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class ObjectiveElement serves as base class for all kind of different 
 *  objective function terms within optimal control problems.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class ObjectiveElement{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        ObjectiveElement( );

        /** Default constructor. */
        ObjectiveElement( const Grid &grid_ );

        /** Copy constructor (deep copy). */
        ObjectiveElement( const ObjectiveElement& rhs );

        /** Destructor. */
        virtual ~ObjectiveElement( );

        /** Assignment operator (deep copy). */
        ObjectiveElement& operator=( const ObjectiveElement& rhs );





// ==========================================================================
//
//                               INITIALIZATION
//
// ==========================================================================


        /**  Sets the discretization grid.   \n
         *                                   \n
         *   \return SUCCESSFUL_RETURN       \n
         */
        inline returnValue setGrid( const Grid &grid_ );



        /** Initializes the Objective Element: The dimensions and   \n
         *  index lists.                                            \n
         *  \return SUCCESSFUL_RETURN
         */
        returnValue init( const OCPiterate &x );



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
    virtual returnValue getObjectiveValue( double &objectiveValue  /**< the objective value */ );



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




// =======================================================================================
//
//                                     DIMENSIONS
//
// =======================================================================================


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

        /** Returns the element's function                 			  \n
         *  \return SUCCESSFUL_RETURN.                                \n
         */
        returnValue getFunction( Function& _function );



// ==========================================================================
//
//                          PROTECTED MEMBER FUNCTIONS:
//
// ==========================================================================

    protected:


        /** returns the constraint grid */
        inline Grid getGrid() const;


    //
    // DATA MEMBERS:
    //
    protected:

        Grid             grid   ;   /**< the objective grid      */
        Function         fcn    ;   /**< the function            */

        EvaluationPoint  z      ;   /**< the evaluation point    */
        EvaluationPoint  JJ     ;

        int             *y_index;   /**< index lists             */
        int              t_index;   /**< time index              */

        int              nx     ;   /**< number of diff. states  */
        int              na     ;   /**< number of alg. states   */
        int              nu     ;   /**< number of controls      */
        int              np     ;   /**< number of parameters    */
        int              nw     ;   /**< number of disturbances  */
        int              ny     ;   /**< := nx+na+nu+np+nw       */


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
        double                 obj;   /**< the objective value */

        BlockMatrix      dForward ;   /**< the first order forward  derivatives */
        BlockMatrix      dBackward;   /**< the first order backward derivatives */

};


CLOSE_NAMESPACE_ACADO



#include <acado/objective/objective_element.ipp>


#endif  // ACADO_TOOLKIT_OBJECTIVE_ELEMENT_HPP

/*
 *     end of file
 */
