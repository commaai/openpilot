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
 *    \file include/acado/function/ocp_iterate.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//



BEGIN_NAMESPACE_ACADO


inline uint OCPiterate::getNX ( ) const{ return getDim(x ); }
inline uint OCPiterate::getNXA( ) const{ return getDim(xa); }
inline uint OCPiterate::getNP ( ) const{ return getDim(p ); }
inline uint OCPiterate::getNU ( ) const{ return getDim(u ); }
inline uint OCPiterate::getNW ( ) const{ return getDim(w ); }

inline DVector OCPiterate::getX ( const uint &idx ) const{ return copy( x , idx ); }
inline DVector OCPiterate::getXA( const uint &idx ) const{ return copy( xa, idx ); }
inline DVector OCPiterate::getP ( const uint &idx ) const{ return copy( p , idx ); }
inline DVector OCPiterate::getU ( const uint &idx ) const{ return copy( u , idx ); }
inline DVector OCPiterate::getW ( const uint &idx ) const{ return copy( w , idx ); }


inline uint OCPiterate::getDim( VariablesGrid *z ) const{

    if( z == 0 ) return 0;
    else         return z->getNumValues();
}


inline DVector OCPiterate::copy( const VariablesGrid *z, const uint &idx ) const{

    if( z == 0 ) return DVector();
    else         return z->getVector(idx);
}


inline double OCPiterate::getTime( const uint &idx ) const{

    if( x  != 0 ) return x ->getTime(idx);
    if( xa != 0 ) return xa->getTime(idx);
    if( p  != 0 ) return p ->getTime(idx);
    if( u  != 0 ) return u ->getTime(idx);
    if( w  != 0 ) return w ->getTime(idx);

    return 0.0;
}


inline Grid OCPiterate::getGrid() const{

    if( x  != 0 ) return x ->getTimePoints();
    if( xa != 0 ) return xa->getTimePoints();
    if( p  != 0 ) return p ->getTimePoints();
    if( u  != 0 ) return u ->getTimePoints();
    if( w  != 0 ) return w ->getTimePoints();

    return Grid();
}



inline BooleanType OCPiterate::isInSimulationMode( ) const
{
	return inSimulationMode;
}



CLOSE_NAMESPACE_ACADO

// end of file.
