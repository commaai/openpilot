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
 *    \file   include/acado/set_arithmetics/taylor_variable.ipp
 *    \author Boris Houska, Mario Villanueva, Benoit Chachuat
 *    \date   2013
 */


BEGIN_NAMESPACE_ACADO


template <typename T> inline
TaylorVariable<T>::TaylorVariable
( const double d )
: _TM( 0 ), _bndT( d )
{
  _init();
  _coefmon[0] = d;
  _bndord[0] = 0.;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator =
( const double d )
{
  if( _TM ){ _TM = 0; _reinit(); }
  _coefmon[0] = d;
  _bndord[0] = 0.;
  return *this;
}

template <typename T> inline
TaylorVariable<T>::TaylorVariable
( TaylorModel<T>*TM, const double d )
: _TM( TM )
{
  if( !_TM ){
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::INIT );
  }
  _init();
  _coefmon[0] = d;
  for( unsigned int i=1; i<_nmon(); i++ ) _coefmon[i] = 0.;
  _bndord[0] = d;
  for( unsigned int i=1; i<_nord()+2; i++) _bndord[i] = 0.;
  if( _TM->options.PROPAGATE_BNDT ) _bndT = d;
}

template <typename T> inline
TaylorVariable<T>::TaylorVariable
( const T&B_ )
: _TM( 0 ), _bndT( B_ )
{
  _init();
  _coefmon[0] = 0.;
  _bndord[0] = B_;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator =
( const T&B_ )
{
  if( _TM ){ _TM = 0; _reinit(); }
  _coefmon[0] = 0.;
  _bndord[0] = B_;
  return *this;
}

template <typename T> inline
TaylorVariable<T>::TaylorVariable
( TaylorModel<T>*TM, const T&B_ )
: _TM( TM )
{
  if( !_TM ) throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::INIT );
  _init();
  for( unsigned int i=0; i<_nmon(); i++ ) _coefmon[i] = 0.;
  for( unsigned int i=0; i<_nord()+1; i++) _bndord[i] = 0.;
  *_bndrem = B_;
  if( _TM->options.PROPAGATE_BNDT ) _bndT = B_;
  if( _TM->options.CENTER_REMAINDER ) _center_TM();
}

template <typename T> inline
TaylorVariable<T>::TaylorVariable
( const TaylorVariable<T>&TV )
: _TM(0)
{
  _init();
  *this = TV;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator =
( const TaylorVariable<T>&TV )
{
  // Same TaylorVariable
  if( this == &TV ) return *this;

  // Reinitialization needed?
  if( _TM != TV._TM ){ _TM = TV._TM; _reinit(); }

  // Set to TaylorVariable not linked to TaylorModel (either scalar or range)
  if( !_TM ){
    _coefmon[0] = TV._coefmon[0];
    _bndord[0] = TV._bndord[0];
    return *this; 
  }
  // Set to TaylorVariable linked to TaylorModel
  for( unsigned int i=0; i<_nmon(); i++ ) _coefmon[i] = TV._coefmon[i];
  for( unsigned int i=0; i<_nord()+2; i++) _bndord[i] = TV._bndord[i];
  if( _TM->options.PROPAGATE_BNDT ) _bndT = TV._bndT;
  return *this;
}

template <typename T> template <typename U> inline
TaylorVariable<T>::TaylorVariable
( TaylorModel<T>*&TM, const TaylorVariable<U>&TV )
: _TM(TM), _coefmon(0), _bndord(0), _bndrem(0)
{
  _init();
  TaylorVariable<U> TVtrunc( TV );
  _coefmon[0] = TVtrunc._coefmon[0];
  TVtrunc._coefmon[0] = 0. ;
  for( unsigned int i=1; _TM && i<_nmon(); i++ ){
    if( TVtrunc._TM && i < TVtrunc._nmon() ){
      _coefmon[i] = TVtrunc._coefmon[i];
      TVtrunc._coefmon[i] = 0.;
    }
    else
      _coefmon[i] = 0.;
  }
  TVtrunc._update_bndord();
  *_bndrem = T( TVtrunc.B() );
  if( !_TM ) return;
  _update_bndord();
  if( _TM->options.PROPAGATE_BNDT ) _bndT = T( TV._bndT );
  return;
}


template <typename T> template <typename U> inline
TaylorVariable<T>::TaylorVariable
( TaylorModel<T>*&TM, const TaylorVariable<U>&TV, T (*method)( const U& ) )
: _TM(TM), _coefmon(0), _bndord(0), _bndrem( 0 )
{
  if( !method )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::INIT );
  _init();
  TaylorVariable<U> TVtrunc( TV );
  _coefmon[0] = TVtrunc._coefmon[0];
  TVtrunc._coefmon[0] = 0. ;
  for( unsigned int i=1; _TM && i<_nmon(); i++ ){
    if( TVtrunc._TM && i < TVtrunc._nmon() ){
      _coefmon[i] = TVtrunc._coefmon[i];
      TVtrunc._coefmon[i] = 0.;
    }
    else
      _coefmon[i] = 0.;
  }
  TVtrunc._update_bndord();
  *_bndrem = (*method)( TVtrunc.B() );
  if( !_TM ) return;
  _update_bndord();
  if( _TM->options.PROPAGATE_BNDT ) _bndT = (*method)( TV._bndT );
  return;
}

template <typename T> template <typename U> inline
TaylorVariable<T>::TaylorVariable
( TaylorModel<T>*&TM, const TaylorVariable<U>&TV, const T& (U::*method)() const )
: _TM(TM), _coefmon(0), _bndord(0), _bndrem( 0 )
{
  if( !method )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::INIT );
  _init();
  TaylorVariable<U> TVtrunc( TV );
  _coefmon[0] = TVtrunc._coefmon[0];
  TVtrunc._coefmon[0] = 0. ;
  for( unsigned int i=1; _TM && i<_nmon(); i++ ){
    if( TVtrunc._TM && i < TVtrunc._nmon() ){
      _coefmon[i] = TVtrunc._coefmon[i];
      TVtrunc._coefmon[i] = 0.;
    }
    else
      _coefmon[i] = 0.;
  }
  TVtrunc._update_bndord();
  *_bndrem = (TVtrunc.B().*method)();
  if( !_TM ) return;
  _update_bndord();
  if( _TM->options.PROPAGATE_BNDT ) _bndT = (TV._bndT.*method)();
  return;
}


template <typename T> inline
TaylorVariable<T>::TaylorVariable
( TaylorModel<T>*TM, const unsigned int ivar, const T&X )
: _TM( TM )
{
 
 if( !TM ){
    std::cerr << "No Environment!\n";
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::INIT );
  }

  // Scale variables and keep track of them in TaylorModel
  double scaling = ( _TM->options.SCALE_VARIABLES? diam(X)/2.: 1. );
   
  if( fabs( scaling ) < EQUALITY_EPS ) scaling = 1.;
    
  _TM->_set_bndpow( ivar, X, scaling );
  _TM->_set_bndmon();
  _init();


  // Populate _coefmon w/ TaylorVariable coefficients
  _coefmon[0] = mid(X);
  for( unsigned int i=1; i<_nmon(); i++ ) _coefmon[i] = 0.;
  if( _nord() > 0 ) _coefmon[_nvar()-ivar] = scaling;

  // Populate _bndord w/ bounds on TaylorVariable terms
  _bndord[0] = _coefmon[0];
  _bndord[1] = X-_coefmon[0];
  for( unsigned int i=2; i<_nord()+2; i++) _bndord[i] = 0.;
  if( _TM->options.PROPAGATE_BNDT ) _bndT = X;
}

template <typename T> inline void
TaylorVariable<T>::_init()
{
  if( !_TM ){
    _coefmon = new double[1];
    _bndord  = new T[1];
    _bndrem  = _bndord;
    return;
  }
  _coefmon = new double[_nmon()];
  _bndord  = new T[_nord()+2];
  _bndrem  = _bndord + _nord()+1;
}

template <typename T> inline void
TaylorVariable<T>::_clean()
{
  delete [] _coefmon; delete [] _bndord;
  _coefmon = 0; _bndord = _bndrem = 0;
}

template <typename T> inline void
TaylorVariable<T>::_reinit()
{
  _clean(); _init();
}

template <typename T> inline void
TaylorVariable<T>::_update_bndord()
{
  if( !_TM ) return;
  _TM->_set_bndmon();
  _bndord[0] = _coefmon[0];
  for( unsigned int i=1; i<=_nord(); i++ ){
    _bndord[i] = 0.; 
    for( unsigned int j=_posord(i); j<_posord(i+1); j++ )
      _bndord[i] += _coefmon[j] * _bndmon(j);
  }
}

template <typename T> inline void
TaylorVariable<T>::_center_TM()
{
  const double remmid = mid(*_bndrem);
  _coefmon[0] += remmid;
  if( _TM ) _bndord[0] = _coefmon[0];
  *_bndrem -= remmid;
}

template <typename T> inline double*
TaylorVariable<T>::_eigen
( const unsigned int n, double*a )
{
  //int info;
  double*d = new double[n];
  
  ASSERT(1==0);
  
// #ifdef MC__TVAR_DEBUG_EIGEN
//   TaylorModel<T>::_display( n, n, a, n, "DMatrix Q", std::cout );
// #endif
// 
//   // get optimal size
//   double worktmp;
//   int lwork = -1;
//   dsyev_( "Vectors", "Upper", &n, a, &n, d, &worktmp, &lwork, &info );
// 
//   // perform eigenvalue decomposition
//   lwork = (int)worktmp;
//   double*work = new double[lwork];
//   dsyev_( "Vectors", "Upper", &n, a, &n, d, work, &lwork, &info );
// #ifdef MC__TVAR_DEBUG_EIGEN
//   TaylorModel<T>::_display( n, n, a, n, "DMatrix U", std::cout );
//   TaylorModel<T>::_display( 1, n, d, 1, "DMatrix D", std::cout );
// #endif
//   delete[] work;
// 
// #ifdef MC__TVAR_DEBUG_EIGEN
//   std::cout << "INFO: " << info << std::endl;
//   TaylorModel<T>::pause();
// #endif
//   if( info ){ delete[] d; return 0; }
  return d;
}

template <typename T> inline T&
TaylorVariable<T>::_bound_eigen
( T& bndmod ) const
{
  static const double TOL = 1e-8;

  bndmod = _coefmon[0];
  if( _nord() == 1 ) bndmod += _bndord[1];

  else if( _nord() > 1 ){
    double*U = new double[_nvar()*_nvar()];
    for( unsigned int i=0; i<_nvar(); i++ ){
      for( unsigned int j=0; j<i; j++ ){
        U[_nvar()*(_nvar()-i-1)+_nvar()-j-1] = 0.;
        U[_nvar()*(_nvar()-j-1)+_nvar()-i-1] = _coefmon[_prodmon(i+1,j+2)]/2.;
      }
      U[(_nvar()+1)*(_nvar()-i-1)] = _coefmon[_prodmon(i+1,i+2)];
    }
    double*D = _eigen( _nvar(), U );
    if( !D ){
      delete[] U;
      throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::EIGEN );
    }
    T bndtype2(0.);
    for( unsigned int i=0; i<_nvar(); i++ ){
      double linaux = 0.;
      T bndaux(0.);
      for( unsigned int k=0; k<_nvar(); k++ ){
        linaux += U[i*_nvar()+k] * _coefmon[_nvar()-k];
        bndaux += U[i*_nvar()+k] * _bndmon(_nvar()-k);
      }
      if( ::fabs(D[i]) > TOL )
        bndtype2 += D[i] * sqr( linaux/D[i]/2. + bndaux )
          - linaux*linaux/D[i]/4.;
      else
        bndtype2 += linaux * bndaux + D[i] * sqr( bndaux );
    }
    delete[] U;
    delete[] D;

    bndmod += bndtype2;
  }

  for( unsigned int i=3; i<=_nord(); i++ ) bndmod += _bndord[i];
  bndmod += *_bndrem;

  return bndmod;
}

template <typename T> inline T&
TaylorVariable<T>::_bound_LSB
( T& bndmod ) const
{
  static const double TOL = 1e-8;
  bndmod = _coefmon[0];
  if( _nord() == 1 ) bndmod += _bndord[1];
  else if( _nord() > 1 ){
    for( unsigned int i=1; i<=_nvar(); i++ ){
      // linear and diagonal quadratic terms
      unsigned int ii = _prodmon(i,i+1);
      if( ::fabs(_coefmon[ii]) > TOL )
        bndmod += _coefmon[ii] * sqr( _coefmon[i]/_coefmon[ii]/2.
          + _bndmon(i) ) - _coefmon[i]*_coefmon[i]/_coefmon[ii]/4.;
      else
        bndmod += _coefmon[i] * _bndmon(i) + _coefmon[ii] * _bndmon(ii);
      // off-diagonal quadratic terms
      for( unsigned int k=i+1; k<=_nvar(); k++ ){
	unsigned int ik = _prodmon(i,k+1) ;
	bndmod += _coefmon[ik] * _bndmon(ik);
      }
    }
  }
  // higher-order terms
  for( unsigned int i=3; i<=_nord(); i++ ) bndmod += _bndord[i];
  bndmod += *_bndrem;
  return bndmod;
}

template <typename T> inline T&
TaylorVariable<T>::_bound_naive
( T& bndmod ) const
{
  bndmod = _coefmon[0];
  for( unsigned int i=1; i<=_nord()+1; i++ ) bndmod += _bndord[i];
  return bndmod;
}

template <typename T> inline T&
TaylorVariable<T>::_bound
( T& bndmod ) const
{
  if( !_TM ){ bndmod = _coefmon[0] + _bndord[0]; return bndmod; }

  switch( _TM->options.BOUNDER_TYPE ){
  case TaylorModel<T>::Options::NAIVE: bndmod = _bound_naive(bndmod); break;
  case TaylorModel<T>::Options::LSB:   bndmod = _bound_LSB(bndmod);   break;
  case TaylorModel<T>::Options::EIGEN: bndmod = _bound_eigen(bndmod); break;
  case TaylorModel<T>::Options::HYBRID: default:{
    T bndlsb(0.), bndeig(0.);
    if( !inter( bndmod, _bound_LSB(bndlsb), _bound_eigen(bndeig) ) )
      throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::INCON );
   }
  }

  if( _TM->options.PROPAGATE_BNDT && _TM->options.INTER_WITH_BNDT
    && !inter( bndmod, bndmod, _bndT ) )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::INCON );

  return bndmod;
}

template <typename T> inline double
TaylorVariable<T>::polynomial
( const double*x ) const
{
  if( !_TM ) return _coefmon[0];
  double Pval = _coefmon[0];
  for( unsigned int i=1; i<_nmon(); i++ ){
    double valmon = 1.;
    for( unsigned int k=0; k<_nvar(); k++ )
      valmon *= ::pow( x[k]/_scaling(k)-_refpoint(k),
                          _expmon(i*_nvar()+k) );
    Pval += _coefmon[i] * valmon;
  }
  return Pval;
}

template <typename T> inline double*
TaylorVariable<T>::reference() const
{
  if( !_TM ) return 0;
  if( _nvar() < 1 ) return 0;
  double*pref = new double[_nvar()];
  for( unsigned int i=0; i<_nvar(); i++ ) pref[i] = _refpoint(i)*_scaling(i);
  return pref;
}

template <typename T> inline double
TaylorVariable<T>::constant() const
{
  return _coefmon[0];
}

template <typename T> inline double*
TaylorVariable<T>::linear() const
{
  if( !_TM || !_nvar() || !_nord() ) return 0;

  double*plin = new double[_nvar()];
  for( unsigned int i=0; i<_nvar(); i++ )
    plin[i] = _coefmon[_nvar()-i] / _scaling(i);
  return plin;
}

template <typename T> inline double*
TaylorVariable<T>::quadratic
( const int opt ) const
{
  if( !_TM || !_nvar() || _nord() < 2 ) return 0;

  if( opt == 0 ){
    double*pquad = new double[_nvar()];
    for( unsigned int i=0; i<_nvar(); i++ )
      pquad[_nvar()-i-1] = _coefmon[_prodmon(i+1,i+2)]
        / _scaling(_nvar()-i-1) / _scaling(_nvar()-i-1);
    return pquad;
  }

  double*pquad = new double[_nvar()*_nvar()];
  for( unsigned int i=0; i<_nvar(); i++ ){
    for( unsigned int j=0; j<i; j++ ){
      pquad[_nvar()*(_nvar()-i-1)+_nvar()-j-1] = 0.;
      pquad[_nvar()*(_nvar()-j-1)+_nvar()-i-1] = _coefmon[_prodmon(i+1,j+2)]/2.
        / _scaling(_nvar()-i-1) / _scaling(_nvar()-j-1);
    }       
    pquad[(_nvar()+1)*(_nvar()-i-1)] = _coefmon[_prodmon(i+1,i+2)]
      / _scaling(_nvar()-i-1) / _scaling(_nvar()-i-1);
  }
  return pquad;  
}

template <typename T> inline std::ostream&
operator <<
( std::ostream&out, const TaylorVariable<T>&TV )
{
  out << std::endl
      << std::scientific << std::right;

  if( TV.isCompact() == BT_TRUE ){
  // Display constant model
  if( !TV._TM ){
    out << "    a0   = " << std::right << TV._coefmon[0]
        << std::endl << std::endl;
    out << "  Order:        Bound:" << std::endl;
    out << "R" << "        " << *(TV._bndrem) << std::endl;
  }

  // Display monomial term coefficients and corresponding exponents
  else{
    for( unsigned int i=0; i<TV._nmon(); i++ ){
      out << "    a" << std::left << i << " = "
          << std::right << TV._coefmon[i] << "      ";
      for( unsigned int k=0; k<TV._nvar(); k++ )
        out << TV._expmon(i*TV._nvar()+k);
      out << "    B" << std::left << i << " = "
          << TV._bndmon(i) << std::endl;
    }
    out << std::endl;

    // Display bounds on terms of order 0,...,nord and remainder term
    out << "  Order:        Bound:" << std::endl;
    for( unsigned int i=0; i<=TV._nord(); i++ )
      out << std::right << i << "        " << TV._bndord[i]
          << std::endl;
    out << std::right << "R" << "        " << *(TV._bndrem)
        << std::endl << std::endl
        << "  TM Bounder:" << std::right << TV._TM->options.BOUNDER_TYPE;
  }

  // Display Taylor model bounds
  out << std::endl
      << "  TM Bound:" << std::right << TV.B()
      << std::endl;
  if( TV._TM && TV._TM->options.PROPAGATE_BNDT )
    out << "   T Bound:" << std::right << TV.boundT()
        << std::endl;

  }
  else{
     out << "[-inf,inf]" << std::endl;
  }
  return out;
}

template <typename T> inline TaylorVariable<T>
operator +
( const TaylorVariable<T>&TV )
{
  return TV;
}

template <typename T> template <typename U> inline TaylorVariable<T>&
TaylorVariable<T>::operator +=
( const TaylorVariable<U>&TV )
{
  if( !TV._TM ){
    if( _TM && _TM->options.PROPAGATE_BNDT ) _bndT += TV._bndT;
    _coefmon[0] += TV._coefmon[0];
    *_bndrem += *(TV._bndrem);
  }
  else if( !_TM ){
    TaylorVariable<T> TV2(*this);
    *this = TV; *this += TV2;
  }
  else{
    if( _TM != TV._TM )
      throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::TMODEL );
    if( _TM->options.PROPAGATE_BNDT ) _bndT += TV._bndT;
    for( unsigned int i=0; i<_nmon(); i++ )
      _coefmon[i] += TV._coefmon[i];
    *_bndrem += *(TV._bndrem);
    _update_bndord();
  }
  if( _TM && _TM->options.CENTER_REMAINDER ) _center_TM();
  return *this;
}

template <typename T, typename U> inline TaylorVariable<T>
operator +
( const TaylorVariable<T>&TV1, const TaylorVariable<U>&TV2 )
{
  TaylorVariable<T> TV3( TV1 );
  TV3 += TV2;
  return TV3;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator +=
( const double c )
{
  _coefmon[0] += c;
  if( _TM ) _bndord[0] = _coefmon[0];
  if( _TM && _TM->options.PROPAGATE_BNDT ) _bndT += c;
  return *this;
}

template <typename T> inline TaylorVariable<T>
operator +
( const TaylorVariable<T>&TV1, const double c )
{
  TaylorVariable<T> TV3( TV1 );
  TV3 += c;
  return TV3;
}

template <typename T> inline TaylorVariable<T>
operator +
( const double c, const TaylorVariable<T>&TV2 )
{
  TaylorVariable<T> TV3( TV2 );
  TV3 += c;
  return TV3;
}

template <typename T> template <typename U> inline TaylorVariable<T>&
TaylorVariable<T>::operator +=
( const U&I )
{
  *_bndrem += I;
  if( _TM && _TM->options.PROPAGATE_BNDT ) _bndT += I;
  if( _TM && _TM->options.CENTER_REMAINDER ) _center_TM();
  return *this;
}

template <typename T, typename U> inline TaylorVariable<T>
operator +
( const TaylorVariable<T>&TV1, const U&I )
{
  TaylorVariable<T> TV3( TV1 );
  TV3 += I;
  return TV3;
}

template <typename T, typename U> inline TaylorVariable<T>
operator +
( const U&I, const TaylorVariable<T>&TV2 )
{
  TaylorVariable<T> TV3( TV2 );
  TV2 += I;
  return TV3;
}

template <typename T> inline TaylorVariable<T>
operator -
( const TaylorVariable<T>&TV )
{
  if( !TV._TM ){
    TaylorVariable<T> TV2;
    TV2._coefmon[0] = -TV._coefmon[0];
    TV2._bndord[0] = -TV._bndord[0];
    return TV2;
  }
  TaylorVariable<T>& TV2 = *TV._TV();
  //TaylorVariable<T> TV2( TV._TM );
  for( unsigned int i=0; i<TV._nmon(); i++ ) TV2._coefmon[i] = -TV._coefmon[i];
  for( unsigned int i=0; i<TV._nord()+2; i++ ) TV2._bndord[i] = -TV._bndord[i];
  if( TV._TM->options.PROPAGATE_BNDT ) TV2._bndT = -TV._bndT;
  if( TV._TM->options.CENTER_REMAINDER ) TV2._center_TM();
  return TV2;
}

template <typename T> template <typename U> inline TaylorVariable<T>&
TaylorVariable<T>::operator -=
( const TaylorVariable<U>&TV )
{
  if( !TV._TM ){
    if( _TM && _TM->options.PROPAGATE_BNDT ) _bndT -= TV._bndT;
    _coefmon[0] -= TV._coefmon[0];
    *_bndrem -= *(TV._bndrem);
  }
  else if( !_TM ){
    TaylorVariable<T> TV2(*this);
    *this = -TV; *this += TV2;
  }
  else{
    if( _TM != TV._TM )
      throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::TMODEL );
    if( _TM->options.PROPAGATE_BNDT ) _bndT -= TV._bndT;
    for( unsigned int i=0; i<_nmon(); i++ )
      _coefmon[i] -= TV._coefmon[i];
    *_bndrem -= *(TV._bndrem);
    _update_bndord();
  }
  if( _TM && _TM->options.CENTER_REMAINDER ) _center_TM();
  return *this;
}

template <typename T, typename U> inline TaylorVariable<T>
operator-
( const TaylorVariable<T>&TV1, const TaylorVariable<U>&TV2 )
{
  TaylorVariable<T> TV3( TV1 );
  TV3 -= TV2;
  return TV3;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator -=
( const double c )
{
  _coefmon[0] -= c;
  if( _TM ) _bndord[0] = _coefmon[0];
  if( _TM && _TM->options.PROPAGATE_BNDT ) _bndT -= c;
  return *this;
}

template <typename T> inline TaylorVariable<T>
operator -
( const TaylorVariable<T>&TV1, const double c )
{
  TaylorVariable<T> TV3( TV1 );
  TV3 -= c;
  return TV3;
}

template <typename T> inline TaylorVariable<T>
operator -
( const double c, const TaylorVariable<T>&TV2 )
{
  TaylorVariable<T> TV3( -TV2 );
  TV3 += c;
  return TV3;
}

template <typename T> template <typename U> inline TaylorVariable<T>&
TaylorVariable<T>::operator -=
( const U&I )
{
  *_bndrem -= I;
  if( _TM && _TM->options.PROPAGATE_BNDT ) _bndT -= I;
  if( _TM && _TM->options.CENTER_REMAINDER ) _center_TM();
  return *this;
}

template <typename T, typename U> inline TaylorVariable<T>
operator -
( const TaylorVariable<T>&TV1, const U&I )
{
  TaylorVariable<T> TV3( TV1 );
  TV3 -= I;
  return TV3;
}

template <typename T, typename U> inline TaylorVariable<T>
operator -
( const U&I, const TaylorVariable<T>&TV2 )
{
  TaylorVariable<T> TV3( -TV2 );
  TV3 += I;
  return TV3;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator *=
( const TaylorVariable<T>&TV )
{
   TaylorVariable<T> TV2( *this );
   *this = TV * TV2;
   return *this;
}

template <typename T> inline TaylorVariable<T>
operator *
( const TaylorVariable<T>&TV1, const TaylorVariable<T>&TV2 )
{
  if( !TV2._TM )      return( TV1 * TV2._coefmon[0] + TV1 * *(TV2._bndrem) );
  else if( !TV1._TM ) return( TV2 * TV1._coefmon[0] + TV2 * *(TV1._bndrem) );

  if( TV1._TM != TV2._TM )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::TMODEL );
  TaylorVariable<T>& TV3 = *TV1._TV();
  for( unsigned int i=0; i<TV3._nmon(); i++ ) TV3._coefmon[i] = 0.;
  //TaylorVariable<T> TV3( TV1._TM, 0. );

  // Populate _coefmon for product term
  for( unsigned int i=0; i<TV3._posord(TV3._nord()/2+1); i++){
    TV3._coefmon[TV3._prodmon(i,i+1)] += TV1._coefmon[i] * TV2._coefmon[i];
    for( unsigned int j=i+1; j<TV3._prodmon(i,0); j++ )
      TV3._coefmon[TV3._prodmon(i,j+1)] += TV1._coefmon[i] * TV2._coefmon[j]
                                         + TV1._coefmon[j] * TV2._coefmon[i];
  }
  // Calculate remainder term _bndrem for product term
  T s1 = 0., s2 = 0.;
  for( unsigned int i=0; i<=TV3._nord()+1; i++ ){
    T r1 = 0., r2 = 0.;
    for( unsigned int j=TV3._nord()+1-i; j<=TV3._nord()+1; j++ ){
      r1 += TV1._bndord[j];
      r2 += TV2._bndord[j];
    }
    s1 += TV2._bndord[i] * r1 ;
    s2 += TV1._bndord[i] * r2 ;
  }
  if( !inter( *(TV3._bndrem), s1, s2) )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::INTER );
  // Populate _bndord for product term (except remainder term)
  TV3._update_bndord();
  if( TV3._TM->options.PROPAGATE_BNDT ) TV3._bndT = TV1._bndT * TV2._bndT;
  if( TV3._TM->options.CENTER_REMAINDER ) TV3._center_TM();
  return TV3;
}

template <typename T> inline TaylorVariable<T>
sqr
( const TaylorVariable<T>&TV )
{
  if( !TV._TM ){
    TaylorVariable<T> TV2( TV );
    TV2._coefmon[0] *= TV2._coefmon[0];
    *(TV2._bndrem) *= 2. + *(TV2._bndrem);
    return TV2;
 }

  // Populate _coefmon for product term
  TaylorVariable<T> TV2( TV._TM, 0. );
  for( unsigned int i=0; i<TV2._posord(TV2._nord()/2+1); i++){
    TV2._coefmon[TV2._prodmon(i,i+1)] += TV._coefmon[i] * TV._coefmon[i];
    for( unsigned int j=i+1; j<TV2._prodmon(i,0); j++ )
      TV2._coefmon[TV2._prodmon(i,j+1)] += TV._coefmon[i] * TV._coefmon[j] * 2.;
  }

  T s = 0.;
  for( unsigned int i=0; i<=TV2._nord()+1; i++ ){
    unsigned int k = std::max(TV2._nord()+1-i, i+1);
    T r = 0.;
    for( unsigned int j=k; j<=TV2._nord()+1; j++ )
      r += TV._bndord[j];
    s += TV._bndord[i] * r;
  }

  T r = 0.;
  for( unsigned int i=TV2._nord()/2+1; i<=TV2._nord()+1; i++ )
    r += sqr(TV._bndord[i]) ;
  *(TV2._bndrem) = 2. * s + r;
  
  // Populate _bndord for product term (except remainder term)
  TV2._update_bndord();
  if( TV2._TM->options.PROPAGATE_BNDT ) TV2._bndT = sqr( TV._bndT );
  if( TV2._TM->options.CENTER_REMAINDER ) TV2._center_TM();
  return TV2;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator *=
( const double c )
{
  if( !_TM ){
    _coefmon[0] *= c;
    *(_bndrem) *= c;
  }
  else{
    for( unsigned int i=0; i<_nmon(); i++ ) _coefmon[i] *= c;
    for( unsigned int i=0; i<_nord()+2; i++ ) _bndord[i] *= c;
    if( _TM->options.PROPAGATE_BNDT ) _bndT *= c;
  }
  return *this;
}

template <typename T> inline TaylorVariable<T>
operator *
( const TaylorVariable<T>&TV1, const double c )
{
  TaylorVariable<T> TV3( TV1 );
  TV3 *= c;
  return TV3;
}

template <typename T> inline TaylorVariable<T>
operator *
( const double c, const TaylorVariable<T>&TV2 )
{
  TaylorVariable<T> TV3( TV2 );
  TV3 *= c;
  return TV3;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator *=
( const T&I )
{
  if( !_TM ){
    *(_bndrem) += _coefmon[0];
    _coefmon[0] = 0.;
    *(_bndrem) *= I;
  }
  else{
    const double Imid = mid(I);
    T Icur = bound();
    for( unsigned int i=0; i<_nmon(); i++ ) _coefmon[i] *= Imid;
    for( unsigned int i=0; i<_nord()+2; i++ ) _bndord[i] *= Imid;
    *_bndrem += (I-Imid)*Icur;
  }
  if( _TM && _TM->options.CENTER_REMAINDER ) _center_TM();
  if( _TM && _TM->options.PROPAGATE_BNDT ) _bndT *= I;
  return (*this);
}

template <typename T> inline TaylorVariable<T>
operator *
( const TaylorVariable<T>&TV1, const T&I )
{
  TaylorVariable<T> TV3( TV1 );
  TV3 *= I;
  return TV3;
}

template <typename T> inline TaylorVariable<T>
operator *
( const T&I, const TaylorVariable<T>&TV2 )
{
  TaylorVariable<T> TV3( TV2 );
  TV3 *= I;
  return TV3;
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator /=
( const TaylorVariable<T>&TV )
{
   *this *= inv(TV);
   return *this;
}

template <typename T> inline TaylorVariable<T>
operator /
( const TaylorVariable<T>&TV1, const TaylorVariable<T>&TV2 )
{
  return TV1 * inv(TV2);
}

template <typename T> inline TaylorVariable<T>&
TaylorVariable<T>::operator /=
( const double c )
{
  if ( fabs( c ) <= EQUALITY_EPS )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::DIV );
   *this *= (1./c);
   return *this;
}

template <typename T> inline TaylorVariable<T>
operator /
( const TaylorVariable<T>&TV, const double c )
{
  if ( fabs( c ) <= EQUALITY_EPS )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::DIV );
  return TV * (1./c);
}

template <typename T> inline TaylorVariable<T>
operator /
( const double c, const TaylorVariable<T>&TV )
{
  return inv(TV) * c;
}

template <typename T> inline TaylorVariable<T>
inv
( const TaylorVariable<T>&TV )
{
  if( !TV._TM ){
    TaylorVariable<T> TV2( TV );
    TV2._coefmon[0] = 0.;
    *(TV2._bndrem) = inv(TV._coefmon[0] + *(TV._bndrem));
    return TV2;
  }

  const T I( TV.B() );
  double x0 = ( TV._TM->options.REF_MIDPOINT? mid(I):
                TV._coefmon[0] + mid(*TV._bndrem) );
  const TaylorVariable<T> TVmx0( TV - x0 );
  const T Imx0( I - x0 );

  TaylorVariable<T> TV2( 1. ), MON( 1. );
  for( unsigned int i=1; i<=TV._nord(); i++ ){
    MON *= TVmx0 / (-x0);
    TV2 += MON;
  }
  TV2 /= x0;
  TV2 += pow( -Imx0, (int)TV2._nord()+1 )
       / pow( T(0.,1.)*Imx0+x0, (int)TV2._nord()+2 );
  if( TV2._TM->options.PROPAGATE_BNDT ) TV2._bndT = inv( TV._bndT );
  if( TV2._TM->options.CENTER_REMAINDER ) TV2._center_TM();
  return TV2;
}

template <typename T> inline TaylorVariable<T>
sqrt
( const TaylorVariable<T>&TV )
{
  if( !TV._TM ){
    TaylorVariable<T> TV2( TV );
    TV2._coefmon[0] = 0.;
    *(TV2._bndrem) = sqrt(TV._coefmon[0] + *(TV._bndrem));
    return TV2;
  }

  const T I( TV.B() );
  double x0 = ( TV._TM->options.REF_MIDPOINT? mid(I):
                TV._coefmon[0] + mid(*TV._bndrem) );
  const TaylorVariable<T> TVmx0( TV - x0 );
  const T Imx0( I - x0 );

  double s = 0.5;
  TaylorVariable<T> TV2( 1. ), MON( 1. );
  for( unsigned int i=1; i<=TV._nord(); i++ ){
    MON *= TVmx0 / x0;
    TV2 += MON * s;
    s *= -(2.*i-1.)/(2.*i+2.);
  }
  TV2 *= ::sqrt(x0);
  TV2 += s * pow( Imx0, (int)TV2._nord()+1 )
           / pow( T(0.,1.)*Imx0+x0, (int)TV2._nord()+1/2 );
  if( TV2._TM->options.PROPAGATE_BNDT ) TV2._bndT = sqrt( TV._bndT );
  if( TV2._TM->options.CENTER_REMAINDER ) TV2._center_TM();
  return TV2;
}

template <typename T> inline TaylorVariable<T>
exp
( const TaylorVariable<T>&TV )
{ 
  if( !TV._TM ){
    TaylorVariable<T> TV2( TV );
    TV2._coefmon[0] = 0.;
    *(TV2._bndrem) = exp(TV._coefmon[0] + *(TV._bndrem));
    return TV2;
  }

  const T I( TV.B() );
  double x0 = ( TV._TM->options.REF_MIDPOINT? mid(I):
                TV._coefmon[0] + mid(*TV._bndrem) );
  const TaylorVariable<T> TVmx0( TV - x0 );
  const T Imx0( I - x0 );

  double s = 1.;
  TaylorVariable<T> TV2( 1. ), MON( 1. );
  for( unsigned int i=1; i<=TV._nord(); i++ ){
    MON *= TVmx0;
    TV2 += MON * s;
    s /= i+1.;
  }
  TV2 += s * pow( Imx0, (int)TV2._nord()+1 )
           * exp( T(0.,1.)*Imx0 );
  TV2 *= ::exp(x0);
  if( TV2._TM->options.PROPAGATE_BNDT ) TV2._bndT = exp( TV._bndT );
  if( TV2._TM->options.CENTER_REMAINDER ) TV2._center_TM();
  return TV2;
}

template <typename T> inline TaylorVariable<T>
log
( const TaylorVariable<T>&TV )
{
  if( !TV._TM ){
    TaylorVariable<T> TV2( TV );
    TV2._coefmon[0] = 0.;
    *(TV2._bndrem) = log(TV._coefmon[0] + *(TV._bndrem));
    return TV2;
  }

  const T I( TV.B() );
  double x0 = ( TV._TM->options.REF_MIDPOINT? mid(I):
                TV._coefmon[0] + mid(*TV._bndrem) );
  const TaylorVariable<T> TVmx0( TV - x0 );
  const T Imx0( I - x0 );

  TaylorVariable<T> TV2( 0. ), MON( -1. );
  for( unsigned int i=1; i<=TV._nord(); i++ ){
    MON *= TVmx0 / (-x0);
    TV2 += MON / (double)i;
  }
  TV2 += ::log(x0) - pow( - Imx0 / ( T(0.,1.)*Imx0+x0 ),
       (int)TV2._nord()+1 ) / ( TV2._nord()+1. );
  if( TV2._TM->options.PROPAGATE_BNDT ) TV2._bndT = log( TV._bndT );
  if( TV2._TM->options.CENTER_REMAINDER ) TV2._center_TM();
  return TV2;
}

template <typename T> inline TaylorVariable<T>
xlog
( const TaylorVariable<T>&TV )
{
  return TV * log( TV );
}

template <typename T> inline TaylorVariable<T>
pow
( const TaylorVariable<T>&TV, const int n )
{
  if( !TV._TM ){
    TaylorVariable<T> TV2( TV );
    TV2._coefmon[0] = 0.;
    *(TV2._bndrem) = pow(TV._coefmon[0] + *(TV._bndrem), n);
    return TV2;
  }

  if( n < 0 ) return pow( inv( TV ), -n );
  TaylorVariable<T> TV2( _intpow( TV, n ) );
  if( TV2._TM->options.PROPAGATE_BNDT ) TV2._bndT = pow( TV._bndT, n );
  if( TV2._TM->options.CENTER_REMAINDER ) TV2._center_TM();
  return TV2;
}

template <typename T> inline TaylorVariable<T>
_intpow
( const TaylorVariable<T>&TV, const int n )
{
  if( n == 0 ) return 1.;
  else if( n == 1 ) return TV;
  return n%2 ? sqr( _intpow( TV, n/2 ) ) * TV : sqr( _intpow( TV, n/2 ) );
}

template <typename T> inline TaylorVariable<T>
pow
( const TaylorVariable<T> &TV, const double a )
{
  return exp( a * log( TV ) );
}

template <typename T> inline TaylorVariable<T>
pow
( const TaylorVariable<T> &TV1, const TaylorVariable<T> &TV2 )
{
  return exp( TV2 * log( TV1 ) );
}

template <typename T> inline TaylorVariable<T>
pow
( const double a, const TaylorVariable<T> &TV )
{
  return exp( TV * ::log( a ) );
}

template <typename T> inline TaylorVariable<T>
cos
( const TaylorVariable<T> &TV )
{
  if( !TV._TM ){
    TaylorVariable<T> TV2( TV );
    TV2._coefmon[0] = 0.;
    *(TV2._bndrem) = cos(TV._coefmon[0] + *(TV._bndrem));
    return TV2;
  }

  const T I( TV.B() );
  double x0 = ( TV._TM->options.REF_MIDPOINT? mid(I):
                TV._coefmon[0] + mid(*TV._bndrem) );
  const TaylorVariable<T> TVmx0( TV - x0 );
  const T Imx0( I - x0 );
  double s = 1., c;

  TaylorVariable<T> TV2( 0. ), MON( 1. );
  for( unsigned int i=1; i<=TV._nord(); i++ ){
    switch( i%4 ){
    case 0: c =  ::cos(x0); break;
    case 1: c = -::sin(x0); break;
    case 2: c = -::cos(x0); break;
    case 3:
    default: c = ::sin(x0); break;
    }
    MON *= TVmx0;
    TV2 += c * s * MON;
    s /= i+1;
  }

  switch( (TV2._nord()+1)%4 ){
  case 0: TV2 += s * pow( Imx0, (int)TV2._nord()+1 )
                   * cos( T(0.,1.)*Imx0+x0 ); break;
  case 1: TV2 -= s * pow( Imx0, (int)TV2._nord()+1 )
                   * sin( T(0.,1.)*Imx0+x0 ); break;
  case 2: TV2 -= s * pow( Imx0, (int)TV2._nord()+1 )
                   * cos( T(0.,1.)*Imx0+x0 ); break;
  case 3: TV2 += s * pow( Imx0, (int)TV2._nord()+1 )
                   * sin( T(0.,1.)*Imx0+x0 ); break;
  }
  TV2 += ::cos(x0);

  if( TV2._TM->options.PROPAGATE_BNDT ) TV2._bndT = cos( TV._bndT );
  if( TV2._TM->options.CENTER_REMAINDER ) TV2._center_TM();
  return TV2;
}

template <typename T> inline TaylorVariable<T>
sin
( const TaylorVariable<T> &TV )
{
  return cos( TV - M_PI/2. );
}

template <typename T> inline TaylorVariable<T>
asin
( const TaylorVariable<T> &TV )
{
  throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::UNDEF );
  return 0.;
}

template <typename T> inline TaylorVariable<T>
acos
( const TaylorVariable<T> &TV )
{
  return asin( -TV ) + M_PI/2.;
}

template <typename T> inline TaylorVariable<T>
tan
( const TaylorVariable<T> &TV )
{
  return sin(TV) / cos(TV);
}

template <typename T> inline TaylorVariable<T>
atan
( const TaylorVariable<T> &TV )
{
  return asin(TV) / acos(TV);
}

template <typename T> inline TaylorVariable<T>
hull
( const TaylorVariable<T>&TV1, const TaylorVariable<T>&TV2 )
{
  if( TV1._TM != TV2._TM )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::TMODEL );
  return TV1.P() + ( TV2.P() - TV1.P() ).B() + hull( TV1.R(), TV2.R() );
}

template <typename T> inline bool
inter
( TaylorVariable<T>&TVR, const TaylorVariable<T>&TV1, const TaylorVariable<T>&TV2 )
{
  if( TV1._TM != TV2._TM )
    throw typename TaylorModel<T>::Exceptions( TaylorModel<T>::Exceptions::TMODEL );
  TaylorVariable<T> TV1C( TV1 ), TV2C( TV2 );
  TVR = TV1C.C();
  TVR += TV2C.C();
  TVR *= 0.5;
  T R1C = TV1C.R(), R2C = TV2C.R(); 
  TV1C -= TV2C;
  TV1C *= 0.5;
  *(TV1C._bndrem) = 0;
  T BTVD = TV1C.B();
  return( inter( *(TVR._bndrem), R1C+BTVD, R2C-BTVD ) ? true: false );
}


template <typename T> BooleanType TaylorVariable<T>::isCompact() const{

  BooleanType result = BT_TRUE;
 
  // Display constant model
  if( !_TM ){
    if( acadoIsNaN   ( _coefmon[0] ) == BT_TRUE  ) result = BT_FALSE;
    if( acadoIsFinite( _coefmon[0] ) == BT_FALSE ) result = BT_FALSE;
	if( _bndrem->isCompact()         == BT_FALSE ) result = BT_FALSE;
  }

  // Display monomial term coefficients and corresponding exponents
  else{
    for( unsigned int i=0; i<_nmon(); i++ ){

      if( acadoIsNaN   ( _coefmon[i] ) == BT_TRUE  ) result = BT_FALSE;
      if( acadoIsFinite( _coefmon[i] ) == BT_FALSE ) result = BT_FALSE;
      if( (_bndmon(i)).isCompact()     == BT_FALSE ) result = BT_FALSE;
    }
    for( unsigned int i=0; i<=_nord(); i++ )
      if( (_bndord[i]).isCompact() == BT_FALSE ) result = BT_FALSE;
    if( _bndrem->isCompact()     == BT_FALSE ) result = BT_FALSE;
  }

  return result;
}


CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
