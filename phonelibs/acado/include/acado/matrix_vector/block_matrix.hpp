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
 *    \file include/acado/matrix_vector/block_matrix.hpp
 *    \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 */


#ifndef ACADO_TOOLKIT_BLOCK_MATRIX_HPP
#define ACADO_TOOLKIT_BLOCK_MATRIX_HPP

#include <acado/utils/acado_types.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief Implements a very rudimentary block sparse matrix class.
 *
 *	\ingroup BasicDataStructures
 *	
 *  The class BlockMatrix is a very rudimentary block sparse matrix class. It is only
 *  intended to provide a convenient way to deal with linear algebra objects
 *  and to provide a wrapper for more efficient implementations. It should
 *  not be used for efficiency-critical operations.
 *
 *	 \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 */
class BlockMatrix
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        BlockMatrix( );

		/** Constructor which takes dimensions of the block matrix. */
        BlockMatrix( uint _nRows,	/**< Number of block rows.    */
				     uint _nCols    /**< Number of block columns. */ );

		/** Constructor which takes ... */
        BlockMatrix(	const DMatrix& value
						);

        /** Destructor. */
        virtual ~BlockMatrix( );

        /** Initializer */
		returnValue init( uint _nRows, uint _nCols );

		/** Set method that defines the value of a certain component.
		 *  \return SUCCESSFUL_RETURN */
		returnValue setDense( uint           rowIdx, /**< Row index of the component.    */
                              uint           colIdx, /**< Column index of the component. */
                              const DMatrix&  value );

		/** Add method that adds a matrix to a certain component.
		 *  \return SUCCESSFUL_RETURN */
		returnValue addDense( uint           rowIdx, /**< Row index of the component.    */
                              uint           colIdx, /**< Column index of the component. */
                              const DMatrix&  value );

		/** Access method that returns the value of a certain component.
		 *  \return SUCCESSFUL_RETURN
         */
		inline returnValue getSubBlock( uint    rowIdx,  /**< Row index of the component.    */
                                        uint    colIdx,  /**< Column index of the component. */
                                        DMatrix &value   )  const;

		/** Access method that returns the value of a certain component and requiring
         *  a given dimension.
		 *  \return SUCCESSFUL_RETURN
         */
		returnValue getSubBlock( uint    rowIdx,  /**< Row index of the component.    */
                                 uint    colIdx,  /**< Column index of the component. */
                                 DMatrix &value,
                                 uint nR,
                                 uint nC          )  const;

		/** Adds (element-wise) two matrices to a temporary object.
		 *  \return Temporary object containing the sum of the block matrices. */
		BlockMatrix operator+( const BlockMatrix& arg	/**< Second summand. */ ) const;

		/** Adds (element-wise) a matrix to object.
		 *  \return Reference to object after addition. */
		BlockMatrix& operator+=( const BlockMatrix& arg /**< Second summand. */ );

		/** Subtracts (element-wise) a matrix from the object and and stores
		 *  the result to a temporary object.
		 *  \return Temporary object containing the difference of the matrices. */
		BlockMatrix operator-( const BlockMatrix& arg /**< Subtrahend. */ ) const;

		/** Multiplies each component of the object with a given scalar.
		 *  \return Reference to object after multiplication. */
		BlockMatrix operator*=( double scalar /**< Scalar factor. */ );

		/** Multiplies a matrix from the right to the matrix object and
		 *  stores the result to a temporary object.
		 *  \return Temporary object containing result of multiplication. */
		BlockMatrix operator*( const BlockMatrix& arg /**< Block DMatrix Factor. */ ) const;

		/** Multiplies a matrix from the right to the transposed matrix object and
		 *  stores the result to a temporary object.
		 *  \return Temporary object containing result of multiplication. */
		BlockMatrix operator^( const BlockMatrix& arg	/**< Block DMatrix Factor. */ ) const;

		/** Returns number of block rows of the block matrix object.
		 *  \return Number of rows. */
		inline uint getNumRows( ) const;

		/** Returns number of block columns of the block matrix object.
		 *  \return Number of columns. */
		inline uint getNumCols( ) const;

		/** Returns the number of rows of a specified sub-matrix.
		 *  \return Number of rows. */
		inline uint getNumRows( uint rowIdx, uint colIdx ) const;

		/** Returns number of block columns of a specified sub-matrix.
		 *  \return Number of columns. */
		inline uint getNumCols( uint rowIdx, uint colIdx ) const;

		/** Sets a specified sub block to the (dim x dim)-identity matrix.
		 *  \return SUCCESSFUL_RETURN */
		inline returnValue setIdentity( uint rowIdx,  /**< row    index of the sub block */
                                        uint colIdx,  /**< column index of the sub block */
                                        uint dim      /**< dimension    of the sub block */  );

		/** Sets a specified sub block to the be a zero matrix.
		 *  \return SUCCESSFUL_RETURN */
		inline returnValue setZero( uint rowIdx, uint colIdx );

		/** Returns whether the block matrix element is empty. */
		inline bool isEmpty() const;

		/** Sets everyting to zero.
		 *  \return SUCCESSFUL_RETURN */
		inline returnValue setZero();

		/** Sets all values to a given constant.
		 *  \return SUCCESSFUL_RETURN */
		inline returnValue addRegularisation( uint rowIdx, uint colIdx, double eps );

		/** Sets everyting to zero.
		 *  \return SUCCESSFUL_RETURN */
		inline BlockMatrix addRegularisation( double eps );

        /** Returns the transpose of the object */
        BlockMatrix transpose() const;

		/** Tests if object is a block-square matrix.
		 *  \return BT_TRUE iff block matrix object is square. */
		inline bool isSquare( ) const;

		/** Tests if a specified sub-matrix is a square matrix.
		 *  \return BT_TRUE iff submatrix object is square. */
		inline bool isSquare( uint rowIdx, uint colIdx ) const;

        /** Returns the a block matrix whose components are the absolute
         *  values of the components of this object.
         */
        BlockMatrix getAbsolute( ) const;

        /** Returns the a block matrix whose components are equal to
         *  the components of this object, if they are positive or zero,
         *  but zero otherwise.
         */
        BlockMatrix getPositive( ) const;

        /** Returns the a block matrix whose components are equal to
         *  the components of this object, if they are negative or zero,
         *  but zero otherwise.
         */
        BlockMatrix getNegative( ) const;

		/** Prints object to standard ouput stream.
		 *  \return SUCCESSFUL_RETURN */
		returnValue print(	std::ostream& stream = std::cout
							) const;

    //
    // DATA MEMBERS:
    //
    protected:

		uint nRows;			/**< Number of rows. */
		uint nCols;			/**< Number of columns. */

        std::vector< std::vector< DMatrix > > elements;
        std::vector< std::vector< SubBlockMatrixType > > types;
};

static       BlockMatrix emptyBlockMatrix;
static const BlockMatrix emptyConstBlockMatrix;

CLOSE_NAMESPACE_ACADO

#include <acado/matrix_vector/block_matrix.ipp>

#endif  // ACADO_TOOLKIT_BLOCK_MATRIX_HPP

/*
 *	end of file
 */
