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
 *    \file   include/acado/set_arithmetics/interval.ipp
 *    \author Boris Houska, Mario Villanueva, Benoit Chachuat
 *    \date   2013
 */

#include <algorithm>

BEGIN_NAMESPACE_ACADO


inline Interval operator+(                    const Interval &I ){                                       return I;  }
inline Interval operator-(                    const Interval &I ){ Interval I2(     -I._u ,     -I._l ); return I2; }
inline Interval operator+( const double    c, const Interval &I ){ Interval I2(c    +I._l ,c    +I._u ); return I2; }
inline Interval operator+( const Interval &I, const double    c ){ Interval I2(c    +I._l ,c    +I._u ); return I2; }
inline Interval operator+( const Interval&I1, const Interval&I2 ){ Interval I3(I1._l+I2._l,I1._u+I2._u); return I3; }
inline Interval operator-( const double    c, const Interval &I ){ Interval I2(c    -I._u ,c    -I._l ); return I2; }
inline Interval operator-( const Interval &I, const double    c ){ Interval I2(I._l -c    ,I._u -c    ); return I2; }
inline Interval operator-( const Interval&I1, const Interval&I2 ){ Interval I3(I1._l-I2._u,I1._u-I2._l); return I3; }
inline Interval operator*( const double    c, const Interval&I  ){
  Interval I2( c>=0? c*I._l: c*I._u, c>=0? c*I._u: c*I._l );
  return I2;
}
inline Interval operator*( const Interval&I, const double c ){
  Interval I2( c>=0? c*I._l: c*I._u, c>=0? c*I._u: c*I._l );
  return I2;
}
inline Interval operator*( const Interval&I1, const Interval&I2 ){
  Interval I3( std::min(std::min(I1._l*I2._l,I1._l*I2._u),
                        std::min(I1._u*I2._l,I1._u*I2._u)),
               std::max(std::max(I1._l*I2._l,I1._l*I2._u),
                        std::max(I1._u*I2._l,I1._u*I2._u)) );
  return I3;
}
inline Interval operator/( const Interval &I, const double c   ){

	if( acadoIsZero(c) == BT_TRUE ){
	    Interval I2(-INFINITY,INFINITY);
	    return I2;
	}
	return (1./c)*I;
}
inline Interval operator/( const double    c, const Interval&I ){ return c*inv(I)  ; }
inline Interval operator/( const Interval&I1, const Interval&I2){ return I1*inv(I2); }

inline double   diam( const Interval &I ){ return I._u-I._l;       }
inline double   mid ( const Interval &I ){ return 0.5*(I._u+I._l); }
inline double   abs ( const Interval &I ){ return std::max(::fabs(I._l),::fabs(I._u)); }
inline Interval inv ( const Interval &I ){
 
  if ( I._l <= EQUALITY_EPS && I._u >= -EQUALITY_EPS ){
		Interval I2(-INFINITY,INFINITY);
		return I2;
  }
  Interval I2( 1./I._u, 1./I._l );
  return I2;
}
inline Interval sqr( const Interval&I ){
  int imid = -1;  
  Interval I2( ::pow(I.mid(I._l,I._u,0.,imid),2.),
               std::max((I._l)*(I._l),(I._u)*(I._u)) );
  return I2;
}
inline Interval exp( const Interval &I ){
  Interval I2( ::exp(I._l), ::exp(I._u) );
  return I2;
}
inline Interval arh( const Interval &I, const double a ){
  Interval I2( ::exp(-a/(a>=0?I._l:I._u)), ::exp(-a/(a>=0?I._u:I._l)) );
  return I2;
}
inline Interval log( const Interval &I ){
  if ( I._l <= EQUALITY_EPS ){
		Interval I2(-INFINITY,INFINITY);
		return I2;
  }
  Interval I2( ::log(I._l), ::log(I._u) );
  return I2;
}
inline Interval xlog( const Interval&I ){
  if ( I._l <= EQUALITY_EPS ){
		Interval I2(-INFINITY,INFINITY);
		return I2;
  }
  int imid = -1;
  Interval I2( I.xlog(I.mid(I._l,I._u,::exp(-1.),imid)),
               std::max(I._l*(::log(I._l)),I._u*(::log(I._u))) );
  return I2;
}


inline Interval sqrt( const Interval&I ){
  if ( I._l <= EQUALITY_EPS ){
		Interval I2(-INFINITY,INFINITY);
		return I2;
  }
  Interval I2( ::sqrt(I._l), ::sqrt(I._u) );
  return I2;
}

inline Interval fabs( const Interval&I ){
  int imid = -1;
  Interval I2( ::fabs(I.mid(I._l,I._u,0.,imid)),
               std::max(::fabs(I._l),::fabs(I._u)) );
  return I2;
}

inline Interval pow( const Interval&I, const int n ){
  if( n == 0 ){
    return 1.;
  }
  if( n == 1 ){
    return I;
  }
  if( n >= 2 && n%2 == 0 ){ 
    int imid = -1;
    Interval I2( ::pow(I.mid(I._l,I._u,0.,imid),n),
                 std::max(::pow(I._l,n),::pow(I._u,n)) );
    return I2;
  }
  if ( n >= 3 ){
    Interval I2( ::pow(I._l,n), ::pow(I._u,n) );
    return I2;
  }
  return inv( pow( I, -n ) );
}

inline Interval pow( const Interval&I, const double a ){
  return exp( a * log( I ) );
}

inline Interval pow( const Interval&I1, const Interval&I2 ){
  return exp( I2 * log( I1 ) );
}

inline Interval hull( const Interval&I1, const Interval&I2 ){
  Interval I3( std::min( I1._l, I2._l ), std::max( I1._u, I2._u ) );
  return I3;
}

inline Interval min( const Interval&I1, const Interval&I2 ){
  Interval I3( std::min( I1._l, I2._l ), std::min( I1._u, I2._u ) );
  return I3;
}

inline Interval max( const Interval&I1, const Interval&I2 ){
  Interval I3( std::max( I1._l, I2._l ), std::max( I1._u, I2._u ) );
  return I3;
}

inline Interval min( const unsigned int n, const Interval*I ){
  Interval I2( n==0 || !I ? 0.: I[0] );
  for( unsigned int i=1; i<n; i++ ) I2 = min( I2, I[i] );
  return I2;
}

inline Interval max( const unsigned int n, const Interval*I ){
  Interval I2( n==0 || !I ? 0.: I[0] );
  for( unsigned int i=1; i<n; i++ ) I2 = max( I2, I[i] );
  return I2;
}

inline Interval cos( const Interval&I ){
  const double k = ::ceil(-(1.+I._l/M_PI)/2.);
  const double l = I._l+2.*M_PI*k, u = I._u+2.*M_PI*k;
  if( l <= 0 ){
    if( u <= 0 ){
      Interval I2( ::cos(l), ::cos(u) );
      return I2;
    }
    if( u >= M_PI ){
      Interval I2( -1., 1. );
      return I2;
    }
    Interval I2( std::min(::cos(l), ::cos(u)), 1. );
    return I2;
  }
  if( u <= M_PI ){
    Interval I2( ::cos(u), ::cos(l) );
    return I2;
  }
  if( u >= 2.*M_PI ){
    Interval I2( -1., 1. );
    return I2;
  }
  Interval I2( -1., std::max(::cos(l), ::cos(u)));
  return I2;
}

inline Interval sin( const Interval &I ){
  return cos( I - M_PI/2. );
}

inline Interval tan( const Interval&I ){
  const double k = ::ceil(-0.5-I._l/M_PI);
  const double l = I._l+M_PI*k, u = I._u+M_PI*k;
  if( u >= 0.5*M_PI - EQUALITY_EPS ){
     Interval I2( -INFINITY , INFINITY );
     return I2;
  }
  Interval I2( ::tan(l), ::tan(u) );
  return I2;
}

inline Interval acos( const Interval &I ){

  if ( I._l <= -1.+EQUALITY_EPS || I._u >= 1.-EQUALITY_EPS ){
       Interval I2( -INFINITY , INFINITY );
	   return I2;   
  }
  Interval I2( ::acos(I._u), ::acos(I._l) );
  return I2;
}

inline Interval asin( const Interval &I ){

  if ( I._l <= -1.+EQUALITY_EPS || I._u >= 1.-EQUALITY_EPS ){
		Interval I2( -INFINITY , INFINITY );
		return I2;   
  }
  Interval I2( ::asin(I._l), ::asin(I._u) );
  return I2;
}

inline Interval atan ( const Interval &I ){
  Interval I2( ::atan(I._l), ::atan(I._u) );
  return I2;
}

inline std::ostream& operator<<( std::ostream&out, const Interval&I){

  out << std::right << std::scientific;
  out << "[ "  << I.l()
      << " : " << I.u() << " ]";
  return out;
}

inline bool inter( Interval &XIY, const Interval &X, const Interval &Y ){

  if( X._l > Y._u || Y._l > X._u ) return false;
  XIY._l = std::max( X._l, Y._l );
  XIY._u = std::min( X._u, Y._u );
  return true;
}


inline bool operator==( const Interval&I1, const Interval&I2 ){

  return( ::fabs( I1._l - I2._l ) <= EPS && ::fabs( I1._u - I2._u ) <= EPS );
}

inline bool operator!=( const Interval&I1, const Interval&I2 ){

  return( ::fabs( I1._l - I2._l ) > EPS || ::fabs( I1._u - I2._u ) > EPS );
}

inline bool operator<=( const Interval&I1, const Interval&I2 ){

  return( I1._l >= I2._l && I1._u <= I2._u );
}

inline bool operator>=( const Interval&I1, const Interval&I2 ){

  return( I1._l <= I2._l && I1._u >= I2._u );
}

inline bool operator<( const Interval&I1, const Interval&I2 ){

  return( I1._l > I2._l && I1._u < I2._u );
}

inline bool operator>( const Interval&I1, const Interval&I2 ){

  return( I1._l < I2._l && I1._u > I2._u );
}

CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
