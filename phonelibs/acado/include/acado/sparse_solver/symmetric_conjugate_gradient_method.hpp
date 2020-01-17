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
 *    \file include/acado/sparse_solver/symmetric_conjugate_gradient_method.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date   2009
 */


#ifndef ACADO_TOOLKIT_SYMMETRIC_CONJUGATE_GRADIENT_METHOD_HPP
#define ACADO_TOOLKIT_SYMMETRIC_CONJUGATE_GRADIENT_METHOD_HPP


#include <acado/utils/acado_utils.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Implements a conjugate gradient method as sparse linear algebra solver for symmetric linear systems.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class SymmetricConjugateGradientMethod is a conjugate gradient method \n
 *  which allows to solve symmetric linear equation of the form               \n
 *                                                                            \n
 *    A * x = b                                                               \n
 *                                                                            \n
 *  where A is symmetric and positive definite.                               \n
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 *  \date   2009
 */


class SymmetricConjugateGradientMethod : public ConjugateGradientMethod{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:


        /** Default constructor. */
        SymmetricConjugateGradientMethod( );

        /** Copy constructor (deep copy). */
        SymmetricConjugateGradientMethod( const SymmetricConjugateGradientMethod &arg );

        /** Destructor. */
        virtual ~SymmetricConjugateGradientMethod( );

        /** Clone operator (deep copy). */
        virtual SparseSolver* clone() const;


        /** Sets an index list containing the positions of the \n
         *  non-zero elements in the matrix  A.
         */
        virtual returnValue setIndices( const int *rowIdx_,
                                        const int *colIdx_  );



        /** Sets an index list containing the positions of the \n
         *  non-zero elements in the matrix  A.
         */
        virtual returnValue setIndices( const int *indices_ );


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



    /** Evaluates the matrix-vector product  result = A*xx efficiently. (only internal use)*/
    virtual void multiply( double *xx , double *result );


    /** Computes the preconditioner and Applies it to the input matrix. */
    virtual returnValue computePreconditioner( double* A_ );


    /** Applies the preconditioner to the vector b (only internal use) */
    virtual returnValue applyPreconditioner( double *b );

    /** Applies the inverse of the preconditioner to the vector x (only internal use) */
    virtual returnValue applyInversePreconditioner( double *x_ );



    //
    // DATA MEMBERS:
    //
    protected:


        int     **index;
        int     *nIndex;
        int       *diag;

};


CLOSE_NAMESPACE_ACADO



#include <acado/sparse_solver/symmetric_conjugate_gradient_method.ipp>


#endif  // ACADO_TOOLKIT_SYMMETRIC_CONJUGATE_GRADIENT_METHOD_HPP

/*
 *   end of file
 */

