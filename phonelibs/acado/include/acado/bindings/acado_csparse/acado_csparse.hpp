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
 *    \file   external_packages/include/acado_csparse.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date   2009
 */


#ifndef ACADO_TOOLKIT_ACADO_CSPARSE_HPP
#define ACADO_TOOLKIT_ACADO_CSPARSE_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/sparse_solver/sparse_solver.hpp>




// FORWARD DECLARATIONS:
// ---------------------
   struct cs_numeric ;
   struct cs_symbolic;



BEGIN_NAMESPACE_ACADO


/**
 *	\brief (not yet documented)
 *
 *	\ingroup ExternalFunctionality
 *
 *	The class ...
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */



class ACADOcsparse : public SparseSolver{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        ACADOcsparse( );

        /** Copy constructor (deep copy). */
        ACADOcsparse( const ACADOcsparse &arg );

        /** Destructor. */
        virtual ~ACADOcsparse( );

        /** Clone operator (deep copy). */
        virtual ACADOcsparse* clone() const;


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
                                        const int *colIdx_  );



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



    //
    // DATA MEMBERS:
    //
    protected:


    // DIMENSIONS:
    // --------------------
    int                dim;          // dimension of the matrix A
    int             nDense;          // number of non-zero entries in A
    int   *index1, *index2;          // and the associated indices



    // DATA:
    // --------------------
    double              *x;          // The result vector    x


    // AUXILIARY VARIABLES:
    // --------------------
    cs_symbolic         *S;          // pointer to a struct, which contains symbolic information about the matrix
    cs_numeric          *N;          // pointer to a struct, which contains numeric information about the matrix


    double             TOL;          // The required tolerance. (default 10^(-10))
    PrintLevel  printLevel;          // The PrintLevel.
};


CLOSE_NAMESPACE_ACADO

#endif

/*
 *   end of file
 */

