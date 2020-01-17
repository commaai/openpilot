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
 *    \file include/acado/sparse_solver/conjugate_gradient_method.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_CONJUGATE_GRADIENT_MEHTOD_HPP
#define ACADO_TOOLKIT_CONJUGATE_GRADIENT_MEHTOD_HPP


#include <acado/utils/acado_utils.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Implements a conjugate gradient method as sparse linear algebra solver.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class ConjugateGradientMethod implements a special sparse       \n
 *  linear algebra solver. After the application of an preconditioner   \n
 *  an iterative conjugate basis of the sparse data matrix  A  is       \n
 *  computed. The algotithm stops if the required tolerance is achieved.\n
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */


class ConjugateGradientMethod : public SparseSolver{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        ConjugateGradientMethod( );

        /** Copy constructor (deep copy). */
        ConjugateGradientMethod( const ConjugateGradientMethod &arg );

        /** Destructor. */
        virtual ~ConjugateGradientMethod( );

        /** Clone operator (deep copy). */
        virtual SparseSolver* clone() const = 0;


        /** Defines the dimension n of  A \in R^{n \times n} \n
         *                                                   \n
         *  \return SUCCESSFUL_RETURN                        \n
         */
        virtual returnValue setDimension( const int &n );


        /** Defines the number of non-zero elements in the   \n
         *  matrix  A                                        \n
         *                                                   \n
         *  \return SUCCESSFUL_RETURN                        \n
         */
        virtual returnValue setNumberOfEntries( const int &nDense_ );



        /** Sets an index list containing the positions of the \n
         *  non-zero elements in the matrix  A.
         */
        virtual returnValue setIndices( const int *rowIdx_,
                                        const int *colIdx_  ) = 0;



        /** Sets the non-zero elements of the matrix A. The double* A  \n
         *  is assumed to contain  nDense  entries corresponding to    \n
         *  non-zero elements of A.                                    \n
         */
        virtual returnValue setMatrix( double *A_ );



        /**  Solves the system  A*x = b  for the specified data.       \n
         *                                                             \n
         *   \return SUCCESSFUL_RETURN                                 \n
         *           RET_LINEAR_SYSTEM_NUMERICALLY_SINGULAR            \n
         */
        virtual returnValue solve( double *b );



        /**  Returns the solution of the equation  A*x = b  if solved. \n
         *                                                             \n
         *   \return SUCCESSFUL_RETURN                                 \n
         */
        virtual returnValue getX( double *x_ );



        /**  Sets the required tolerance (accuracy) for the solution of \n
         *   the linear equation. For large tolerances an iterative     \n
         *   algorithm might converge earlier.                          \n
         *                                                              \n
         *   Requires   || A*x - b || <= TOL                            \n
         *                                                              \n
         *   The norm || .  ||  is possibly scaled by a preconditioner. \n
         *                                                              \n
         *   \return SUCCESSFUL_RETURN                                  \n
         */
        virtual returnValue setTolerance( double TOL_ );


        /** Sets the print level.                                       \n
         *                                                              \n
         *  \return SUCCESSFUL_RETURN                                   \n
         */
        virtual returnValue setPrintLevel( PrintLevel printLevel_ );



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


    /** Returns the scalar product of aa and bb.  (only internal use)*/
    double scalarProduct( double *aa, double *bb );

    /** Evaluates the matrix-vector product  result = A*xx efficiently. (only internal use)*/
    virtual void multiply( double *xx , double *result ) = 0;

    /** Applies the preconditioner to the vector b (only internal use) */
    virtual returnValue applyPreconditioner( double *b ) = 0;

    /** Applies the inverse of the preconditioner to the vector x (only internal use) */
    virtual returnValue applyInversePreconditioner( double *x_ ) = 0;

    /** Computes the preconditioner and Applies it to the input matrix. */
    virtual returnValue computePreconditioner( double* A_ ) = 0;


    //
    // DATA MEMBERS:
    //
    protected:


    // DIMENSIONS:
    // --------------------
    int                dim;          // dimension of the matrix A
    int             nDense;          // number of non-zero entries in A

    // DATA:
    // --------------------
    double              *A;          // The (sparse) matrix  A
    double              *x;          // The result vector    x


    // AUXILIARY VARIABLES:
    // --------------------
    double          *norm2;          // Auxiliary variables
    double             **p;          // conjugate basis vectors
    double              *r;          // the actual residuum
    int           pCounter;          // a counter for the iterates
    double      *condScale;          // scaling factors to improve
                                     // the conditioning of the system

    double             TOL;          // The required tolerance. (default 10^(-10))
    PrintLevel  printLevel;          // The PrintLevel.
};


CLOSE_NAMESPACE_ACADO



#include <acado/sparse_solver/conjugate_gradient_method.ipp>


#endif  // ACADO_TOOLKIT_CONJUGATE_GRADIENT_METHOD_HPP

/*
 *   end of file
 */

