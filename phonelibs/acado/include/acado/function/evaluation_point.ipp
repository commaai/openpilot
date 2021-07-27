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



inline returnValue EvaluationPoint::copy( const int *order, const DVector &rhs ){

    uint i;
    for( i = 0; i < rhs.getDim(); i++ )
        z[order[i]] = rhs(i);
    return SUCCESSFUL_RETURN;
}


inline DVector EvaluationPoint::backCopy( const int *order, const uint &dim ) const{

    DVector tmp(dim);
    uint i;
    for( i = 0; i < dim; i++ )
        tmp(i) = z[order[i]];
    return tmp;
}

inline returnValue EvaluationPoint::setT ( const double &t  ){ return copy( idx[0], DVector(1,&t) ); }
inline returnValue EvaluationPoint::setX ( const DVector &x  ){ return copy( idx[1], x            ); }
inline returnValue EvaluationPoint::setXA( const DVector &xa ){ return copy( idx[2], xa           ); }
inline returnValue EvaluationPoint::setP ( const DVector &p  ){ return copy( idx[3], p            ); }
inline returnValue EvaluationPoint::setU ( const DVector &u  ){ return copy( idx[4], u            ); }
inline returnValue EvaluationPoint::setW ( const DVector &w  ){ return copy( idx[5], w            ); }
inline returnValue EvaluationPoint::setDX( const DVector &dx ){ return copy( idx[6], dx           ); }


inline returnValue EvaluationPoint::setZ ( const uint       &idx_,
                                           const OCPiterate &iter  ){

    setT ( iter.getTime (idx_) );
    setX ( iter.getX    (idx_) );
    setXA( iter.getXA   (idx_) );
    setP ( iter.getP    (idx_) );
    setU ( iter.getU    (idx_) );
    setW ( iter.getW    (idx_) );

    return SUCCESSFUL_RETURN;
}




inline returnValue EvaluationPoint::setZero( )
{
	if ( z != 0 ) 
	{
		for( uint run1 = 0; run1 < N; run1++ )
			z[run1] = 0.0;
	}

	return SUCCESSFUL_RETURN;
}


inline double EvaluationPoint::getT () const{ return z[*idx[0]]            ; }
inline DVector EvaluationPoint::getX () const{ return backCopy( idx[1], nx ); }
inline DVector EvaluationPoint::getXA() const{ return backCopy( idx[2], na ); }
inline DVector EvaluationPoint::getP () const{ return backCopy( idx[3], np ); }
inline DVector EvaluationPoint::getU () const{ return backCopy( idx[4], nu ); }
inline DVector EvaluationPoint::getW () const{ return backCopy( idx[5], nw ); }
inline DVector EvaluationPoint::getDX() const{ return backCopy( idx[6], nd ); }

inline double* EvaluationPoint::getEvaluationPointer() const{ return z; }

CLOSE_NAMESPACE_ACADO

// end of file.
