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
 *    \file include/acado/function/evaluation_point.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date 2010
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO



template <typename T> inline returnValue TevaluationPoint<T>::copy( const int *order, const Tmatrix<T> &rhs ){

    for( uint i = 0; i < rhs.getDim(); i++ )
        z->operator()(order[i]) = rhs(i);

    return SUCCESSFUL_RETURN;
}


template <typename T> inline Tmatrix<T> TevaluationPoint<T>::backCopy( const int *order, const uint &dim ) const{

    Tmatrix<T> tmp(dim);
    uint i;
    for( i = 0; i < dim; i++ )
        tmp(i) = z->operator()(order[i]);
    return tmp;
}

template <typename T> inline returnValue TevaluationPoint<T>::setT ( const Tmatrix<T> &t  ){ return copy( idx[0], t  ); }
template <typename T> inline returnValue TevaluationPoint<T>::setX ( const Tmatrix<T> &x  ){ return copy( idx[1], x  ); }
template <typename T> inline returnValue TevaluationPoint<T>::setXA( const Tmatrix<T> &xa ){ return copy( idx[2], xa ); }
template <typename T> inline returnValue TevaluationPoint<T>::setP ( const Tmatrix<T> &p  ){ return copy( idx[3], p  ); }
template <typename T> inline returnValue TevaluationPoint<T>::setU ( const Tmatrix<T> &u  ){ return copy( idx[4], u  ); }
template <typename T> inline returnValue TevaluationPoint<T>::setW ( const Tmatrix<T> &w  ){ return copy( idx[5], w  ); }
template <typename T> inline returnValue TevaluationPoint<T>::setDX( const Tmatrix<T> &dx ){ return copy( idx[6], dx ); }


template <typename T> inline Tmatrix<T> TevaluationPoint<T>::getT () const{ return backCopy( idx[0], 1  ); }
template <typename T> inline Tmatrix<T> TevaluationPoint<T>::getX () const{ return backCopy( idx[1], nx ); }
template <typename T> inline Tmatrix<T> TevaluationPoint<T>::getXA() const{ return backCopy( idx[2], na ); }
template <typename T> inline Tmatrix<T> TevaluationPoint<T>::getP () const{ return backCopy( idx[3], np ); }
template <typename T> inline Tmatrix<T> TevaluationPoint<T>::getU () const{ return backCopy( idx[4], nu ); }
template <typename T> inline Tmatrix<T> TevaluationPoint<T>::getW () const{ return backCopy( idx[5], nw ); }
template <typename T> inline Tmatrix<T> TevaluationPoint<T>::getDX() const{ return backCopy( idx[6], nd ); }

template <typename T> inline Tmatrix<T>* TevaluationPoint<T>::getEvaluationPointer() const{ return z; }

CLOSE_NAMESPACE_ACADO

// end of file.
