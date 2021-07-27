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
 *    \file include/acado/matrix_vector/block_matrix.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date 19.01.2009
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO



inline returnValue BlockMatrix::getSubBlock( uint rowIdx, uint colIdx, DMatrix &value )  const{

	ASSERT( rowIdx < getNumRows( ) );
	ASSERT( colIdx < getNumCols( ) );

    value = elements[rowIdx][colIdx];
    return SUCCESSFUL_RETURN;
}


inline uint BlockMatrix::getNumRows( ) const{

	return nRows;
}

inline uint BlockMatrix::getNumCols( ) const{

	return nCols;
}


inline uint BlockMatrix::getNumRows( uint rowIdx, uint colIdx ) const{

    ASSERT( rowIdx < getNumRows( ) );
    ASSERT( colIdx < getNumCols( ) );

    return elements[rowIdx][colIdx].getNumRows();
}


inline uint BlockMatrix::getNumCols( uint rowIdx, uint colIdx ) const{

    ASSERT( rowIdx < getNumRows( ) );
    ASSERT( colIdx < getNumCols( ) );

    return elements[rowIdx][colIdx].getNumCols();
}


inline returnValue BlockMatrix::setIdentity( uint rowIdx, uint colIdx, uint dim ){

    ASSERT( rowIdx < getNumRows( ) );
    ASSERT( colIdx < getNumCols( ) );

           types   [rowIdx][colIdx] = SBMT_ONE   ;
           elements[rowIdx][colIdx] = DMatrix(dim, dim);     
    elements[rowIdx][colIdx].setIdentity();
    
    return SUCCESSFUL_RETURN;
}


inline returnValue BlockMatrix::setZero( uint rowIdx, uint colIdx ){

    ASSERT( rowIdx < getNumRows( ) );
    ASSERT( colIdx < getNumCols( ) );

           types   [rowIdx][colIdx] = SBMT_ZERO;
           elements[rowIdx][colIdx].setZero()  ;
    return SUCCESSFUL_RETURN; 
}


inline returnValue BlockMatrix::addRegularisation( uint rowIdx, uint colIdx, double eps ){

    ASSERT( rowIdx < getNumRows( ) );
    ASSERT( colIdx < getNumCols( ) );

    if( types[rowIdx][colIdx] != SBMT_ZERO ){
        DMatrix tmp( elements[rowIdx][colIdx].getNumRows(), elements[rowIdx][colIdx].getNumCols() );
        tmp.setConstant( eps );
        elements[rowIdx][colIdx] += tmp;
    }

    return SUCCESSFUL_RETURN;
}


inline returnValue BlockMatrix::setZero(){

    uint run1, run2;

    for( run1 = 0; run1 < getNumRows(); run1++ )
        for( run2 = 0; run2 < getNumCols(); run2++ )
            setZero( run1, run2 );

    return SUCCESSFUL_RETURN;
}


inline BlockMatrix BlockMatrix::addRegularisation( double eps ){

    uint run1, run2;

    for( run1 = 0; run1 < getNumRows(); run1++ )
        for( run2 = 0; run2 < getNumCols(); run2++ )
            addRegularisation( run1, run2, eps );

    return *this;
}


inline bool BlockMatrix::isSquare( ) const{

    if( getNumRows() == getNumCols() ) return BT_TRUE;
    return BT_FALSE;
}



inline bool BlockMatrix::isSquare( uint rowIdx, uint colIdx ) const{

    ASSERT( rowIdx < getNumRows( ) );
    ASSERT( colIdx < getNumCols( ) );

    return elements[rowIdx][colIdx].isSquare();
}


inline bool BlockMatrix::isEmpty() const{

    if( (getNumRows() == 0) && (getNumCols() == 0) ) return BT_TRUE;
    return BT_FALSE;
}



CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
