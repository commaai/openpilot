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
 *    \file include/acado/nlp_derivative_approximation/nlp_derivative_approximation.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_NLP_DERIVATIVE_APPROXIMATION_HPP
#define ACADO_TOOLKIT_NLP_DERIVATIVE_APPROXIMATION_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>

#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/matrix_vector/block_matrix.hpp>
#include <acado/function/ocp_iterate.hpp>



BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for techniques of approximating second-order derivatives within NLPsolvers.
 *
 *	\ingroup AlgorithmInterfaces
 *
 *  The class NLPderivativeApproximation serves as a base class for different 
 *	techniques of approximating second-order derivative information within 
 *	iterative NLPsolvers.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class NLPderivativeApproximation : public AlgorithmicBase
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        NLPderivativeApproximation( );

        NLPderivativeApproximation(	UserInteraction* _userInteraction
									);
		
        /** Copy constructor (deep copy). */
        NLPderivativeApproximation( const NLPderivativeApproximation& rhs );

        /** Destructor. */
        virtual ~NLPderivativeApproximation( );

        /** Assignment operator (deep copy). */
        NLPderivativeApproximation& operator=( const NLPderivativeApproximation& rhs );


		virtual NLPderivativeApproximation* clone( ) const = 0;



        virtual returnValue initHessian(	BlockMatrix& B, 	    /**< matrix to be initialised */
											uint N,                 /**< number of intervals      */
											const OCPiterate& iter  /**< current iterate          */
											) = 0;

        virtual returnValue initScaling(	BlockMatrix& B, /**< matrix to be updated */
											const BlockMatrix& x, /**< direction x          */
											const BlockMatrix& y  /**< residuum             */
											) = 0;


		virtual returnValue apply(	BlockMatrix &B, /**< matrix to be updated */
									const BlockMatrix &x, /**< direction x          */
									const BlockMatrix &y  /**< residuum             */
									) = 0;


		inline double getHessianScaling( ) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		virtual returnValue setupOptions( );
		virtual returnValue setupLogging( );


    //
    // PROTECTED DATA MEMBERS:
    //
    protected:

        double hessianScaling;

};


CLOSE_NAMESPACE_ACADO


#include <acado/nlp_derivative_approximation/nlp_derivative_approximation.ipp>


// collect remaining headers
#include <acado/nlp_derivative_approximation/exact_hessian.hpp>
#include <acado/nlp_derivative_approximation/constant_hessian.hpp>
#include <acado/nlp_derivative_approximation/gauss_newton_approximation.hpp>


#endif  // ACADO_TOOLKIT_NLP_DERIVATIVE_APPROXIMATION_HPP

/*
 *  end of file
 */
