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
 *    \file include/acado/sparse_solver/sparse_solver.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_SPARSE_SOLVER_HPP
#define ACADO_TOOLKIT_SPARSE_SOLVER_HPP


#include <acado/utils/acado_utils.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Generic interface for sparse solvers to be coupled with ACADO Toolkit.
 *
 *	\ingroup AlgorithmInterfaces
 *
 *  The class SparseSolver is a generic interface for sparse solver to be
 *  coupled with ACADO Toolkit.
 *
 *  A SparseSolver deals with linear equations of the form
 *
 *     A * x = b
 *
 *  where  A  is (a possibly large but sparse) given matrix and b a given
 *  vector. Here, the non-zero elements of A need to be specified by one
 *  of the routines that are provided in this class. The aim of the sparse
 *  solver is to find the vector x, which is assumed to be uniquely defined
 *  by the above equation. For solving the linear equation the zero entries
 *  of the matrix  A  are used efficiently.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */


class SparseSolver{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        SparseSolver( );

        /** Destructor. */
        virtual ~SparseSolver( );

        /** Clone operator (deep copy). */
        virtual SparseSolver* clone() const = 0;


        /** Defines the dimension n of  A \in R^{n \times n} \n
         *                                                   \n
         *  \return SUCCESSFUL_RETURN                        \n
         */
        virtual returnValue setDimension( const int &n ) = 0;


        /** Defines the number of non-zero elements in the   \n
         *  matrix  A                                        \n
         *                                                   \n
         *  \return SUCCESSFUL_RETURN                        \n
         */
        virtual returnValue setNumberOfEntries( const int &nDense_ ) = 0;



        /** Sets an index list containing the positions of the \n
         *  non-zero elements in the matrix  A.
         */
        virtual returnValue setIndices( const int *rowIdx_,
                                        const int *colIdx_  ) = 0;



        /** Sets the non-zero elements of the matrix A. The double* A  \n
         *  is assumed to contain  nDense  entries corresponding to    \n
         *  non-zero elements of A.                                    \n
         */
        virtual returnValue setMatrix( double *A_ ) = 0;



        /**  Solves the system  A*x = b  for the specified data.       \n
         *                                                             \n
         *   \return SUCCESSFUL_RETURN                                 \n
         *           RET_LINEAR_SYSTEM_NUMERICALLY_SINGULAR            \n
         */
        virtual returnValue solve( double *b ) = 0;


        /**  Solves the system  A^T*x = b  for the specified data.     \n
         *                                                             \n
         *   \return SUCCESSFUL_RETURN                                 \n
         *           RET_LINEAR_SYSTEM_NUMERICALLY_SINGULAR            \n
         */
        virtual returnValue solveTranspose( double *b );


        /**  Returns the solution of the equation  A*x = b  if solved. \n
         *                                                             \n
         *   \return SUCCESSFUL_RETURN                                 \n
         */
        virtual returnValue getX( double *x ) = 0;



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
        virtual returnValue setTolerance( double TOL ) = 0;


        /** Sets the print level.                                       \n
         *                                                              \n
         *  \return SUCCESSFUL_RETURN                                   \n
         */
        virtual returnValue setPrintLevel( PrintLevel PrintLevel_ ) = 0;




    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



    //
    // DATA MEMBERS:
    //
    protected:


};


CLOSE_NAMESPACE_ACADO

#include <acado/sparse_solver/sparse_solver.ipp>

#endif  // ACADO_TOOLKIT_SPARSE_SOLVER_HPP

#include <acado/sparse_solver/conjugate_gradient_method.hpp>
#include <acado/sparse_solver/normal_conjugate_gradient_method.hpp>
#include <acado/sparse_solver/symmetric_conjugate_gradient_method.hpp>

/*
 *   end of file
 */

