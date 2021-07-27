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
 *    \file   include/acado/set_arithmetics/taylor_variable.hpp
 *    \author Boris Houska, Mario Villanueva, Benoit Chachuat
 *    \date   2013
 */


#ifndef ACADO_TOOLKIT_TAYLOR_VARIABLE_HPP
#define ACADO_TOOLKIT_TAYLOR_VARIABLE_HPP

#include <acado/utils/acado_utils.hpp>


BEGIN_NAMESPACE_ACADO

//! @brief C++ template class for definition of and operation on variables in a Taylor model
////////////////////////////////////////////////////////////////////////
//! mc::TaylorVariable<T> is a C++ template class for definition of and operation
//! on the variables participating in a Taylor model of a factorable
//! function. The template parameter T corresponds to the type used in
//! computing the remainder error bound in the Taylor model.
////////////////////////////////////////////////////////////////////////
template <typename T>
class TaylorVariable
////////////////////////////////////////////////////////////////////////
{
  // friends of class TaylorVariable
  template <typename U> friend void TaylorModel<U>::_size
    ( const unsigned int nvar, const unsigned int nord );

  template <typename U> friend class TaylorVariable;

  template <typename U> friend TaylorVariable<U> operator+
    ( const TaylorVariable<U>& );
  template <typename U, typename V> friend TaylorVariable<U> operator+
    ( const TaylorVariable<U>&, const TaylorVariable<V>& );
  template <typename U, typename V> friend TaylorVariable<U> operator+
    ( const TaylorVariable<U>&, const V& );
  template <typename U, typename V> friend TaylorVariable<U> operator+
    ( const V&, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator+
    ( const double, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator+
    ( const TaylorVariable<U>&, const double );
  template <typename U> friend TaylorVariable<U> operator-
    ( const TaylorVariable<U>& );
  template <typename U, typename V> friend TaylorVariable<U> operator-
    ( const TaylorVariable<U>&, const TaylorVariable<V>& );
  template <typename U, typename V> friend TaylorVariable<U> operator-
    ( const TaylorVariable<U>&, const V& );
  template <typename U, typename V> friend TaylorVariable<U> operator-
    ( const V&, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator-
    ( const double, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator-
    ( const TaylorVariable<U>&, const double );
  template <typename U> friend TaylorVariable<U> operator*
    ( const TaylorVariable<U>&, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator*
    ( const double, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator*
    ( const TaylorVariable<U>&, const double );
  template <typename U> friend TaylorVariable<U> operator*
    ( const U&, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator*
    ( const TaylorVariable<U>&, const U& );
  template <typename U> friend TaylorVariable<U> operator/
    ( const TaylorVariable<U>&, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator/
    ( const double, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> operator/
    ( const TaylorVariable<U>&, const double );
  template <typename U> friend std::ostream& operator<<
    ( std::ostream&, const TaylorVariable<U>& );

  template <typename U> friend TaylorVariable<U> inv
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> sqr
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> sqrt
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> exp
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> log
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> xlog
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> pow
    ( const TaylorVariable<U>&, const int );
  template <typename U> friend TaylorVariable<U> pow
    ( const TaylorVariable<U>&, const double );
  template <typename U> friend TaylorVariable<U> pow
    ( const double, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> pow
    ( const TaylorVariable<U>&, const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> cos
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> sin
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> tan
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> acos
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> asin
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> atan
    ( const TaylorVariable<U>& );
  template <typename U> friend TaylorVariable<U> hull
    ( const TaylorVariable<U>&, const TaylorVariable<U>& );
  template <typename U> friend bool inter
    ( TaylorVariable<U>&, const TaylorVariable<U>&, const TaylorVariable<U>& );

private:
  //! @brief Pointer to underlying TaylorModel object
  TaylorModel<T> *_TM;
  //! @brief Get pointer to internal TaylorVariable in TaylorModel
  TaylorVariable<T>* _TV() const
    { return _TM->_TV; };
  //! @brief Get model order in TaylorModel
  unsigned int _nord() const
    { return _TM->_nord; };
  //! @brief Get number of variables in TaylorModel
  unsigned int _nvar() const
    { return _TM->_nvar; };
  //! @brief Get number of monomial terms in TaylorModel
  unsigned int _nmon() const
    { return _TM->_nmon; };
  //! @brief Get term positions in TaylorModel
  unsigned int _posord
    ( const unsigned int i ) const
    { return _TM->_posord[i]; };
  //! @brief Get monomial term exponents in TaylorModel
  unsigned int _expmon
    ( const unsigned int i ) const
    { return _TM->_expmon[i]; };
  //! @brief Get exponents of monomial term products in TaylorModel
  unsigned int _prodmon
    ( const unsigned int i, const unsigned int j ) const
    { return _TM->_prodmon[i][j]; };
  //! @brief Get bounds on all monomial terms in TaylorModel
  const T& _bndmon
    ( const unsigned int i ) const
    { return _TM->_bndmon[i]; };
  //! @brief Get reference point in TaylorModel
  double _refpoint
    ( const unsigned int i ) const
    { return _TM->_refpoint[i]; };
  //! @brief Get scaling in TaylorModel
  double _scaling
    ( const unsigned int i ) const
    { return _TM->_scaling[i]; };

public:
  /** @defgroup TAYLOR Taylor Model Arithmetic for Factorable Functions
   *  @{
   */
  //! @brief Constructor for a real scalar
  TaylorVariable
    ( const double d=0. );
  //! @brief Constructor for a T variable
  TaylorVariable
    ( const T&B );
  //! @brief Constructor for the variable <a>ix</a>, that belongs to the interval <a>X</a>
  TaylorVariable
    ( TaylorModel<T>*TM, const unsigned int ix, const T&X );
  //! @brief Copy constructor for identical range bounder
  TaylorVariable
    ( const TaylorVariable<T>&TV );
  //! @brief Copy constructor for different range bounder with implicit type conversion
  template <typename U> TaylorVariable
    ( TaylorModel<T>*&TM, const TaylorVariable<U>&TV );
  //! @brief Copy constructor for different range bounder with explicit type conversion class member function
  template <typename U> TaylorVariable
    ( TaylorModel<T>*&TM, const TaylorVariable<U>&TV, const T& (U::*method)() const );
  //! @brief Copy constructor for different range bounder with explicit type conversion non-class function
  template <typename U> TaylorVariable
    ( TaylorModel<T>*&TM, const TaylorVariable<U>&TV, T (*method)( const U& ) );

  //! @brief Class destructor
  ~TaylorVariable()
    { delete [] _coefmon; delete [] _bndord; }

  //! @brief Set the index and range for the variable <a>ivar</a>, that belongs to the interval <a>X</a>.
  TaylorVariable<T>& set
    ( TaylorModel<T>*TM, const unsigned int ix, const T&X )
    { *this = TaylorVariable( TM, ix, X ); return *this; }

  //! @brief Return pointer to TaylorModel environment
  TaylorModel<T>* env() const
    { return _TM; }

  //! @brief Return range bounder
  T bound() const
    { T bndmod; return _bound( bndmod ); }

  //! @brief Return range bounder
  T B() const
    { T bndmod; return _bound( bndmod ); }

  //! @brief Return reference to range bounder in T arithmetic
  const T& boundT() const
    { return _bndT; }

  //! @brief Return const reference to current remainder
  T remainder() const
    { return( *_bndrem ); }

  //! @brief Return const reference to current remainder
  T R() const
    { return remainder(); }

  //! @brief Return reference to current remainder
  //T& R()
  //  { return remainder(); }

  //! @brief Recenter remainder term
  TaylorVariable<T>& center()
    { _center_TM(); return *this; }

  //! @brief Recenter remainder term
  TaylorVariable<T>& C()
    { return center(); }

  //! @brief Cancel remainder term and return polynomial part
  TaylorVariable<T> polynomial() const
  { TaylorVariable<T> TV = *this; *(TV._bndrem) = 0.; return TV; }

  //! @brief Evaluate polynomial part at point <a>x</a>
  double polynomial
    ( const double*x ) const;

  //! @brief Cancel remainder error term and return polynomial part
  TaylorVariable<T> P() const
    { return polynomial(); }

  //! @brief Evaluate polynomial part at point <a>x</a>
  double P
    ( const double*x ) const
    { return polynomial( x ); }

  //! @brief Return pointer to array of size <a>nvar</a> comprising reference point
  double* reference() const;

  //! @brief Return coefficient value in constant term
  double constant() const;

  //! @brief Return pointer to array of size <a>nvar</a> comprising coefficients in linear term
  double* linear() const;

  //! @brief Return pointer to array of size <a>nvar</a> comprising coefficients in quadratic term (diagonal coefs if <a>opt</a>=0; upper triangular part otherwise)
  double* quadratic
    ( const int opt=0 ) const;
  /** @} */

  //! @brief More overloaded operators
  TaylorVariable<T>& operator =
    ( const double );
  TaylorVariable<T>& operator =
    ( const TaylorVariable<T>& );
  TaylorVariable<T>& operator =
    ( const T& );
  template <typename U> TaylorVariable<T>& operator +=
    ( const TaylorVariable<U>& );
  template <typename U> TaylorVariable<T>& operator +=
    ( const U& );
  TaylorVariable<T>& operator +=
    ( const double );
  template <typename U> TaylorVariable<T>& operator -=
    ( const TaylorVariable<U>& );
  template <typename U> TaylorVariable<T>& operator -=
    ( const U& );
  TaylorVariable<T>& operator -=
    ( const double );
  TaylorVariable<T>& operator *=
    ( const TaylorVariable<T>& );
  TaylorVariable<T>& operator *=
    ( const double );
  TaylorVariable<T>& operator *=
    ( const T& );
  TaylorVariable<T>& operator /=
    ( const TaylorVariable<T>& );
  TaylorVariable<T>& operator /=
    ( const double );

	
	
	//! Routine which returns BT_FALSE if there are components equal to "nan" or "INFTY".\n
	//! Otherwise, BT_TRUE is returned.
	BooleanType isCompact() const;
	
	

	//! @brief Private constructor for a real scalar in a specific TaylorModel
	TaylorVariable
	( TaylorModel<T>*TM, const double d=0. );

private:


  //! @brief Private constructor for a range in a specific TaylorModel
  TaylorVariable
    ( TaylorModel<T>*TM, const T&B );

  //! @brief Coefficients for monomial terms 1,...,nmon
  double *_coefmon;
  //! @brief Bounds for individual terms of degrees 0,...,_nord+1
  T * _bndord; 
  //! @brief Pointer to the remainder error bound
  T * _bndrem;
  //! @brief Interval bound evaluated in T arithmetic
  T _bndT;

  //! @brief Initialize private members
  void _init();
  //! @brief Reinitialize private members
  void _reinit();
  //! @brief Clean private members
  void _clean();

  //! @brief Update _bndord w/ (naive) bounds for individual terms of degrees 0,...,_nord()
  void _update_bndord();

  //! @brief Center remainder error term _bndrem
  void _center_TM();

  //! @brief Range bounder
  T& _bound
    ( T& bndmod ) const;
  //! @brief Range bounder - naive approach
  T& _bound_naive
    ( T& bndmod ) const;
  //! @brief Range bounder - Lin & Stadtherr approach
  T& _bound_LSB
    ( T& bndmod ) const;
  //! @brief Range bounder - eignevalue decomposition-based approach
  T& _bound_eigen
    ( T& bndmod ) const;
  //! @brief Wrapper to LAPACK function _dsyev doing eigenvalue decomposition of a symmetric matrix
  static double* _eigen
    ( const unsigned int n, double*a );

  //! @brief Recursive calculation of nonnegative integer powers
  TaylorVariable<T> _intpow
    ( const TaylorVariable<T>&TM, const int n );

};


CLOSE_NAMESPACE_ACADO

//#include <acado/set_arithmetics/interval.ipp>

#endif  // ACADO_TOOLKIT_TAYLOR_VARIABLE_HPP

