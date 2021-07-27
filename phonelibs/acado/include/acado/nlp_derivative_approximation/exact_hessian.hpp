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
 *    \file include/acado/nlp_derivative_approximation/exact_hessian.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_EXACT_HESSIAN_HPP
#define ACADO_TOOLKIT_EXACT_HESSIAN_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/nlp_derivative_approximation/nlp_derivative_approximation.hpp>


BEGIN_NAMESPACE_ACADO



/** 
 *	\brief Implements an exact Hessian computation for obtaining second-order derivatives within NLPsolvers.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class ExactHessian implements an exact Hessian computation for obtaining
 *	second-order derivatives within iterative NLPsolvers.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class ExactHessian : public NLPderivativeApproximation
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        ExactHessian( );
		
		/** Default constructor. */
        ExactHessian(	UserInteraction* _userInteraction
						);

        /** Copy constructor (deep copy). */
        ExactHessian( const ExactHessian& rhs );

        /** Destructor. */
        virtual ~ExactHessian( );

        /** Assignment operator (deep copy). */
        ExactHessian& operator=( const ExactHessian& rhs );


		virtual NLPderivativeApproximation* clone( ) const;



        virtual returnValue initHessian(	BlockMatrix& B, 	    /**< matrix to be initialised */
											uint N,                 /**< number of intervals      */
											const OCPiterate& iter  /**< current iterate          */
											);

        virtual returnValue initScaling(	BlockMatrix& B, /**< matrix to be updated */
											const BlockMatrix& x, /**< direction x          */
											const BlockMatrix& y  /**< residuum             */
											);


        /** Applies a BFGS update in its "standard" form:             \n
         *                                                            \n
         *  B = B - B*x*x^T*B/(x^T*B*x) + y*y^T/(x^T*y)               \n
         *                                                            \n
         *  \return SUCCESSFUL_RETURN                                 \n
         */
        virtual returnValue apply(       BlockMatrix &B, /**< matrix to be updated */
                                   const BlockMatrix &x, /**< direction x          */
                                   const BlockMatrix &y  /**< residuum             */ );


		inline double getHessianScaling( ) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



    //
    // PROTECTED DATA MEMBERS:
    //
    protected:

};


CLOSE_NAMESPACE_ACADO


//#include <acado/nlp_derivative_approximation/exact_hessian.ipp>


#endif  // ACADO_TOOLKIT_EXACT_HESSIAN_HPP

/*
 *  end of file
 */
