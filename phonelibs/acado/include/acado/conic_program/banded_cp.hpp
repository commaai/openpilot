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
 *    \file include/acado/conic_program/banded_cp.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_BANDED_CP_HPP
#define ACADO_TOOLKIT_BANDED_CP_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/function/ocp_iterate.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Data class for storing conic programs arising from optimal control.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class BandedCP (banded conic programs) is a data class
 *  to store conic programs that arise in the context of optimal control.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */

class BandedCP{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        BandedCP( );

        /** Copy constructor (deep copy). */
        BandedCP( const BandedCP& rhs );

        /** Destructor. */
        virtual ~BandedCP( );

        /** Assignment operator (deep copy). */
        BandedCP& operator=( const BandedCP& rhs );


        /** Returns whether or not the conic program is an LP */
        inline BooleanType isLP () const;

        /** Returns whether or not the conic program is an LP */
        inline BooleanType isQP () const;

        /** Returns whether or not the conic program is an SDP */
        inline BooleanType isSDP() const;



    //
    // PUBLIC DATA MEMBERS:
    //
    public:


    // DIMENSIONS OF THE CP:
    // ---------------------
    int nS;    /**< Number of SDP constraints      */



    // BANDED CP IN BLOCK-MATRIX FORMAT:
    // -----------------------------------------------------------------------------------

    BlockMatrix                   hessian;    /**< the Hessian matrix                  */
    BlockMatrix         objectiveGradient;    /**< the gradient of the objective       */

    BlockMatrix        lowerBoundResiduum;    /**< lower residuum of the bounds        */
    BlockMatrix        upperBoundResiduum;    /**< upper residuum of the bounds        */

    BlockMatrix               dynGradient;    /**< the sensitivities of the ODE/DAE    */
    BlockMatrix               dynResiduum;    /**< residuum of the ODE/DAE             */

    BlockMatrix        constraintGradient;    /**< the gradient of the constraints     */
    BlockMatrix   lowerConstraintResiduum;    /**< lower residuum of the constraints   */
    BlockMatrix   upperConstraintResiduum;    /**< upper residuum of the constraints   */

    BlockMatrix                       **B;    /**< SDP constraint tensor               */
    BlockMatrix                      *lbB;    /**< SDP lower bounds                    */
    BlockMatrix                      *ubB;    /**< SDP upper bounds                    */


    // SOLUTION OF THE BANDED CP:
    // ----------------------------------------------------------------------------------
    BlockMatrix                    deltaX;    /**< Primal solution of the banded QP    */
    BlockMatrix               lambdaBound;    /**< Dual solution w.r.t. bounds         */
    BlockMatrix             lambdaDynamic;    /**< Dual solution w.r.t. constraints    */
    BlockMatrix          lambdaConstraint;    /**< Dual solution w.r.t. constraints    */

    BlockMatrix                    **ylbB;    /**< Dual solution, SDB lower bound      */
    BlockMatrix                    **yubB;    /**< Dual solution, SDP upper bound      */



    // PROTECTED MEMBER FUNCTIONS:
    // ---------------------------
    protected:

    void copy (const BandedCP& rhs);
    void clean();
};


CLOSE_NAMESPACE_ACADO


#include <acado/conic_program/banded_cp.ipp>


#endif  // ACADO_TOOLKIT_BANDED_CP_HPP

/*
 *  end of file
 */
