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
 *    \file include/acado/nlp_derivative_approximation/bfgs_update.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_BFGS_UPDATE_HPP
#define ACADO_TOOLKIT_BFGS_UPDATE_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/nlp_derivative_approximation/constant_hessian.hpp>


BEGIN_NAMESPACE_ACADO


/** Definition of the BFGS modifications
 */

enum BFGSModificationType{

    MOD_POWELLS_MODIFICATION,   /**< Use "Powell's trick".                */
    MOD_NOCEDALS_MODIFICATION,  /**< Skip update in "critical" situations */
    MOD_NO_MODIFICATION         /**< Allow possibly indefinite updates    */
};



/** 
 *	\brief Implements BFGS updates for approximating second-order derivatives within NLPsolvers.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class BFGSupdate implements BFGS updates for approximating second-order 
 *	derivative information within iterative NLPsolvers.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class BFGSupdate : public ConstantHessian
{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        BFGSupdate( );
		
        /** Constructor that takes the number of blocks for matrix block updates. */
        BFGSupdate(	UserInteraction* _userInteraction,
					uint _nBlocks = 0
					);

        /** Copy constructor (deep copy). */
        BFGSupdate( const BFGSupdate& rhs );

        /** Destructor. */
        virtual ~BFGSupdate( );

        /** Assignment operator (deep copy). */
        BFGSupdate& operator=( const BFGSupdate& rhs );

		virtual NLPderivativeApproximation* clone( ) const;



        virtual returnValue initHessian(	BlockMatrix& B, 	    /**< matrix to be initialised */
											uint N,                 /**< number of intervals      */
											const OCPiterate& iter  /**< current iterate          */
											);

        /** Applies an initial scaling of the form:                   \n
         *                                                            \n
         *  B = B*sqrt( (y^T*y)/(x^T*x) )                             \n
         *                                                            \n
         *  This rescaling can be used if the initial Hessian was a   \n
         *  unit matrix which should be auto-scaled in the first step \n
         *  Note that the update will be skipped for the case that    \n
         *  x^T*x or y^T*y is less than 1000.0*EPS (safeguard         \n
         *  against unreasonable scaling).                            \n
         *                                                            \n
         *  \return sqrt( (y^T*y)/(x^T*x) )                           \n
         */
        virtual returnValue initScaling(	BlockMatrix& B, /**< matrix to be updated */
											const BlockMatrix& x, /**< direction x          */
											const BlockMatrix& y  /**< residuum             */
											);


        /** Applies a BFGS update.
         *                                                            \n
         *  \return SUCCESSFUL_RETURN                                 \n
         */
        virtual returnValue apply(       BlockMatrix &B, /**< matrix to be updated */
                                   const BlockMatrix &x, /**< direction x          */
                                   const BlockMatrix &y  /**< residuum             */ );



        inline returnValue setBFGSModification(	const BFGSModificationType &modification_
												);


		inline BooleanType performsBlockUpdates( ) const;





    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



        /** Applies a BFGS update in its "standard" form:             \n
         *                                                            \n
         *  B = B - B*x*x^T*B/(x^T*B*x) + y*y^T/(x^T*y)               \n
         *                                                            \n
         *  \return SUCCESSFUL_RETURN                                 \n
         */
        virtual returnValue applyUpdate(	BlockMatrix &B, /**< matrix to be updated */
											const BlockMatrix &x, /**< direction x          */
											const BlockMatrix &y  /**< residuum             */
											);



        /** Applies a block BFGS update to the diagonal of the matrix B. The integer N \n
         *  specifies the number of blocks.                                            \n
         *                                                                             \n
         *  B_ii = B_ii - B_ii*x_i*x_i^T*B_ii/(x_i^T*B_ii*x_i) + y_i*y_i^T/(x_i^T*y_i) \n
         *                                                            \n
         *  \return SUCCESSFUL_RETURN                                 \n
         */
        virtual returnValue applyBlockDiagonalUpdate(	BlockMatrix &B, /**< matrix to be updated */
														const BlockMatrix &x, /**< direction x          */
														const BlockMatrix &y  /**< residuum             */
														);



        returnValue getSubBlockLine( const int         &N     ,
                                     const int         &line1 ,
                                     const int         &line2 ,
                                     const int         &offset,
                                     const BlockMatrix &M     ,
                                           BlockMatrix &x      );

        returnValue setSubBlockLine( const int         &N     ,
                                     const int         &line1 ,
                                     const int         &line2 ,
                                     const int         &offset,
                                           BlockMatrix &M     ,
                                     const BlockMatrix &x      );


    //
    // PROTECTED DATA MEMBERS:
    //
    protected:

		BFGSModificationType modification;

		uint nBlocks;

};


CLOSE_NAMESPACE_ACADO

#include <acado/nlp_derivative_approximation/bfgs_update.ipp>


#endif  // ACADO_TOOLKIT_BFGS_UPDATE_HPP

/*
 *  end of file
 */
