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
 *    \file   include/acado/set_arithmetics/taylor_model.ipp
 *    \author Boris Houska, Mario Villanueva, Benoit Chachuat
 *    \date   2013
 */


BEGIN_NAMESPACE_ACADO


template <typename T> inline void
TaylorModel<T>::_size
( const unsigned int nvar_, const unsigned int nord_ )
{
  if( !nvar_ ) throw Exceptions( Exceptions::SIZE );

  //_cleanup();
  _nvar = nvar_;
  _nord = nord_; 
  _set_binom();
  _set_posord();
  _nmon = _posord[_nord+1];
  _set_expmon();
  _set_prodmon();
  _bndpow = new T*[_nvar];
  for( unsigned int i=0; i<_nvar; i++ ) _bndpow[i] = 0;
  _bndmon = new T[_nmon];  
  _refpoint = new double[_nvar];
  _scaling = new double[_nvar];
  _modvar = true;

  _TV = new TaylorVariable<T>( this );
}


template <typename T> inline void
TaylorModel<T>::_set_bndpow
( const unsigned int ivar, const T&X, const double scaling )
{
  if( ivar>=_nvar ) throw Exceptions( Exceptions::INIT );

  delete[] _bndpow[ivar];
  _bndpow[ivar] = new T [_nord+1];
  _refpoint[ivar] = mid(X)/scaling;
  _scaling[ivar] = scaling;
  T Xr = X/scaling - _refpoint[ivar];
  _bndpow[ivar][0] = 1.;
  for( unsigned int i=1; i<=_nord; i++ ){
    _bndpow[ivar][i] = pow(Xr,(int)i);
  }
  _modvar = true;
}

template <typename T> inline void
TaylorModel<T>::_set_bndmon()
{
  if( !_modvar ) return;
  
  _bndmon[0] = 1.;
  for( unsigned int i=1; i<_nmon; i++ ){
    _bndmon[i] = 1.;
    for( unsigned int j=0; j<_nvar; j++)
      if( _bndpow[j] ) _bndmon[i] *= _bndpow[j][_expmon[i*_nvar+j]];
  }
  _modvar = false;
}

template <typename T> inline void
TaylorModel<T>::_set_posord()
{
  _posord = new unsigned int[_nord+2];
  _posord[0] = 0;
  _posord[1] = 1;
  for( unsigned int i=1; i<=_nord; i++ )
    _posord[i+1] = _posord[i] + _get_binom( _nvar+i-1, i );
}

template <typename T> inline void
TaylorModel<T>::_set_expmon()
{
  _expmon = new unsigned int[_nmon*_nvar];
  unsigned int *iexp = new unsigned int[_nvar] ;
  for( unsigned int k=0; k<_nvar; k++ ) _expmon[k] = 0;
  for( unsigned int i=1; i<=_nord; i++ ){
    for( unsigned int j=0; j<_nvar; j++ ) iexp[j] = 0;
    for( unsigned int j=_posord[i]; j<_posord[i+1]; j++ ){
      _next_expmon( iexp, i );
      for( unsigned int k=0; k<_nvar; k++ )
        _expmon[j*_nvar+k] = iexp[k];
    }
  }
  delete[] iexp;
}
  
template <typename T> inline void
TaylorModel<T>::_next_expmon
( unsigned int *iexp, const unsigned int iord )
{
  unsigned int curord;
  do{
    iexp[_nvar-1] += iord;
    unsigned int j = _nvar;
    while( j > 0 && iexp[j-1] > iord ){
      iexp[j-1] -= iord + 1;
      j-- ;
      iexp[j-1]++;
    }
    curord = 0;
    for( unsigned int i=0; i<_nvar; i++ ) curord += iexp[i];
  } while( curord != iord );
}

template <typename T> inline void
TaylorModel<T>::_set_prodmon()
{
  _prodmon = new unsigned int*[_nmon];
  _prodmon[0] = new unsigned int[_nmon+1];
  _prodmon[0][0] = _nmon;
  for( unsigned int i=1; i<=_nmon; i++ ) _prodmon[0][i] = i-1;

  unsigned int *iexp = new unsigned int[_nvar];
  for( unsigned int i=1; i<_nord; i++ ){    
    for( unsigned int j=_posord[i]; j<_posord[i+1]; j++ ){
      _prodmon[j] = new unsigned int [_posord[_nord+1-i]+1];
      _prodmon[j][0] = _posord[_nord+1-i];
      for( unsigned int k=0; k<_posord[_nord+1-i]; k++ ){
        for( unsigned int in=0; in<_nvar; in++ ) 
          iexp[in] = _expmon[j*_nvar+in] + _expmon[k*_nvar+in] ;
        _prodmon[j][k+1] = _loc_expmon( iexp );
      }
    }
  }
  delete[] iexp;

  for( unsigned int i=_posord[_nord]; i<_nmon; i++ ){
    _prodmon[i] = new unsigned int[2];
    _prodmon[i][0] = 1;
    _prodmon[i][1] = i;
  }
}
    
template <typename T> inline unsigned int
TaylorModel<T>::_loc_expmon
( const unsigned int *iexp )
{
  unsigned int ord = 0;
  for( unsigned int i=0; i<_nvar; i++ ) ord += iexp[i];
  ASSERT( ord<_nord+2 );
  unsigned int pos = _posord[ord];
  
  unsigned int p = _nvar ; 
  for( unsigned int i=0; i<_nvar-1; i++ ){
    p--;
    for( unsigned int j=0; j<iexp[i]; j++ )
      pos += _get_binom( p-1+ord-j, ord-j );
    ord -= iexp[i];
  }

  return pos;    
}
    
template <typename T> inline void
TaylorModel<T>::_set_binom()
{
  _binom = new unsigned int[(_nvar+_nord-1)*(_nord+1)];
  unsigned int *p, k;
  for( unsigned int i=0; i<_nvar+_nord-1; i++ ){
    p = &_binom[i*(_nord+1)];
    *p = 1;
    p++;
    *p = i+1;
    p++;
    k = ( i+1<_nord? i+1: _nord );
    for( unsigned int j=2; j<=k; j++, p++ ) *p = *(p-1) * (i+2-j)/j;
    for( unsigned int j=k+1; j<=_nord; j++, p++ ) *p = 0.;
  }
}

template <typename T> inline unsigned int
TaylorModel<T>::_get_binom
( const unsigned int n, const unsigned int k ) const
{
  ASSERT( n>0 && n<_nord+_nvar && k<=n );
  return _binom[(n-1)*(_nord+1)+k] ;
}

template <typename T> inline void
TaylorModel<T>::_reset()
{
  for( unsigned int i=0; i<_nvar; i++ ){
    delete[] _bndpow[i];
    _bndpow[i] = 0;
  }
}

template <typename T> inline void
TaylorModel<T>::_cleanup()
{
  for( unsigned int i=0; i<_nmon; i++ ) delete[] _prodmon[i];
  delete[] _prodmon;
  delete[] _expmon;
  delete[] _posord;
  for( unsigned int i=0; i<_nvar; i++ ) delete[] _bndpow[i];
  delete[] _bndpow;
  delete[] _bndmon;
  delete[] _refpoint;
  delete[] _scaling;
  delete[] _binom;
  delete _TV;
}

template <typename T> template< typename U > inline void
TaylorModel<T>::_display
( const unsigned int m, const unsigned int n, U*&a, const unsigned int lda,
  const std::string&stra, std::ostream&os )
{
  os << stra << " =" << std::endl << std::scientific;
  for( unsigned int im=0; a && im<m; im++ ){
    for( unsigned int in=0; in<n; in++ ){
      os << a[in*lda+im] << "  ";
    }
    os << std::endl;
  }
  os << std::endl;

  if( &os == &std::cout || &os == &std::cerr ) pause();
}

template <typename T> void
TaylorModel<T>::pause()
{
  double tmp;
  std::cout << "ENTER <1> TO CONTINUE" << std::endl;
  std::cin  >> tmp;
}


CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
