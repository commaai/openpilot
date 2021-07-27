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
 *    \file include/acado/matrix_vector/t_matrix.hpp
 *    \author Boris Houska
 */


#ifndef ACADO_TOOLKIT_T_MATRIX_HPP
#define ACADO_TOOLKIT_T_MATRIX_HPP


BEGIN_NAMESPACE_ACADO

/**
 *  \brief Implements a templated dense matrix class.
 *
 *	\ingroup BasicDataStructures
 *
 *  Implements a templated matrix class.                          \n
 *  The class can be used to represent, e.g., interval matrices,  \n
 *  matrices of symbolic functions, etc..                         \n
 *
 *  \author Boris Houska
 */


template <typename T>
class Tmatrix
////////////////////////////////////////////////////////////////////////
{
  template <typename U> friend class Tmatrix;
  template <typename U> friend std::ostream& operator<<
    ( std::ostream&, const Tmatrix<U>& );
  template <typename U> friend Tmatrix<U> operator+
    ( const Tmatrix<U>& );
  template <typename U, typename V> friend Tmatrix<U> operator+
    ( const Tmatrix<U>&, const V& );
  template <typename U, typename V> friend Tmatrix<U> operator+
    ( const V&, const Tmatrix<U>& );
  template <typename U, typename V> friend Tmatrix<U> operator+
    ( const Tmatrix<U>&, const Tmatrix<V>& );
  template <typename U> friend Tmatrix<U> operator-
    ( const Tmatrix<U>& );
  template <typename U, typename V> friend Tmatrix<U> operator-
    ( const Tmatrix<U>&, const V& );
  template <typename U, typename V> friend Tmatrix<U> operator-
    ( const U&, const Tmatrix<V>& );
  template <typename U, typename V> friend Tmatrix<U> operator-
    ( const Tmatrix<U>&, const Tmatrix<V>& );
  template <typename U, typename V> friend Tmatrix<U> operator*
    ( const Tmatrix<U>&, const V& );
  template <typename U, typename V> friend Tmatrix<U> operator*
    ( const V&, const Tmatrix<U>& );
  template <typename U, typename V> friend Tmatrix<U> operator*
    ( const Tmatrix<U>&, const Tmatrix<V>& );
  template <typename U, typename V> friend Tmatrix<U> operator/
    ( const Tmatrix<U>&, const V& );

public:

  /** @defgroup MATRIX Tmatrix Computations with Arbitrary Types
   *  @{
   */
  //! @brief Default constructor
  Tmatrix():
    _nr(0), _nc(0), _sub(false), _pcol(0), _prow(0), _pblock(0)
    {}
  //! @brief Constructor doing size assignment
  Tmatrix
    ( const unsigned int nr, const unsigned int nc=1, const bool alloc=true ):
    _nr(nr), _nc(nc), _sub(!alloc), _pcol(0), _prow(0), _pblock(0)
    {
      unsigned int ne = nr*nc;
      for( unsigned int ie=0; ie<ne; ie++ ){
        T*pvall = ( alloc? new T: 0 );
        _data.push_back( pvall );
      }
    }
  //! @brief Constructor doing size assignment and element initialization
  template <typename U> Tmatrix
    ( const unsigned int nr, const unsigned int nc, const U&v,
      const bool alloc=true ):
    _nr(nr), _nc(nc), _sub(!alloc), _pcol(0), _prow(0), _pblock(0)
    {
      unsigned int ne = nr*nc;
      for( unsigned int ie=0; ie<ne; ie++ ){
        T*pvall = ( alloc? new T(v): 0 );
        _data.push_back( pvall );
      }
    }
  //! @brief Copy Constructor
  Tmatrix
    ( const Tmatrix<T>&M ):
    _nr(M._nr), _nc(M._nc), _sub(false), _pcol(0), _prow(0), _pblock(0)
    {
      typename std::vector<T*>::const_iterator it = M._data.begin();
      for( ; it!=M._data.end(); it++ ){
        T*pvall = new T(**it);
        _data.push_back( pvall );
      }
    }
  //! @brief Copy Constructor doing type conversion
  template <typename U> Tmatrix
    ( const Tmatrix<U>&M ):
    _nr(M._nr), _nc(M._nc), _sub(false), _pcol(0), _prow(0), _pblock(0)
    {
      typename std::vector<U*>::const_iterator it = M._data.begin();
      for( ; it!=M._data.end(); it++ ){
        T*pvall = new T(**it);
        _data.push_back( pvall );
      }
    }
  //! @brief Destructor
  ~Tmatrix()
    {
      delete _pcol;
      delete _prow;
      delete _pblock;
      if( !_sub ) _reset();
    }

  //! @brief Resets the dimension of a Tmatrix objects
  void resize
    ( const unsigned int nr, const unsigned int nc=1, const bool alloc=true )
    {
      delete _pcol;
      delete _prow;
      delete _pblock;
      if( !_sub ) _reset();
      _nr = nr; _nc = nc; _sub = !alloc;
      _pcol = _prow = _pblock = 0;
      unsigned int ne = nr*nc;
      for( unsigned int ie=0; ie<ne; ie++ ){
        T*pvall = ( alloc? new T: 0 );
        _data.push_back( pvall );
      }
    }
  //! @brief Sets/retrieves value of column ic
  Tmatrix<T>& col
    ( const unsigned int ic );
  //! @brief Sets/retrieves value of row ir
  Tmatrix<T>& row
    ( const unsigned int ir );
  //! @brief Sets/retrieves pointer to entry (ir,ic)
  T*& pval
    ( const unsigned int ir, const unsigned int ic );
  //! @brief Retrieves number of columns
  unsigned int col() const
    { return _nc; };
  //! @brief Retrieves number of rows
  unsigned int row() const
    { return _nr; };
  //! @brief Converts a matrix into array form (columnwise storage)
  T* array() const;
  //! @brief Sets the elements of a matrix from an array (columnwise storage)
  void array
    ( const T*pM );

	unsigned int getDim() const{ return _nc*_nr; }
	unsigned int getNumRows() const{ return _nr; }
	unsigned int getNumCols() const{ return _nc; }
	
	
  //! @brief Other operators
  Tmatrix<unsigned int> sort();
  T& operator()
    ( const unsigned int ir, const unsigned int ic );
  const T& operator()
    ( const unsigned int ir, const unsigned int ic ) const;
  T& operator()
    ( const unsigned int ie );
  const T& operator()
    ( const unsigned int ie ) const;
  Tmatrix<T>& operator=
  ( const Tmatrix<T>&M );
  Tmatrix<T>& operator=
  ( const T&m );
  template <typename U> Tmatrix<T>& operator+=
  ( const Tmatrix<U>&M );
  template <typename U> Tmatrix<T>& operator+=
  ( const U&m );
  template <typename U> Tmatrix<T>& operator-=
  ( const Tmatrix<U>&M );
  template <typename U> Tmatrix<T>& operator-=
  ( const U&m );
  template <typename U> Tmatrix<T>& operator*=
  ( const Tmatrix<U>&M );
  template <typename U> Tmatrix<T>& operator*=
  ( const U&m );
  template <typename U> Tmatrix<T>& operator/=
  ( const U&m );

private:

  //! @brief Number of rows
  unsigned int _nr;
  //! @brief Number of columns
  unsigned int _nc;
  //! @brief Vector of pointers to elements (column-wise storage)
  std::vector<T*> _data;
  //! @brief Flag indicating whether the current object is a submatrix
  bool _sub;
  //! @brief Pointer to Tmatrix<T> container storing column
  Tmatrix<T>* _pcol;
  //! @brief Pointer to Tmatrix<T> container storing row
  Tmatrix<T>* _prow;
  //! @brief Pointer to Tmatrix<T> container storing blocks
  Tmatrix<T>* _pblock;

  //! @brief Returns a reference to actual value at position ie
  T& _val
    ( const unsigned int ir, const unsigned int ic );
  const T& _val
    ( const unsigned int ir, const unsigned int ic ) const;
  T& _val
    ( const unsigned int ie );
  const T& _val
    ( const unsigned int ie ) const;
  //! @brief Returns a pointer to actual value at position ie
  T*& _pval
    ( const unsigned int ie );
  //! @brief Resets the entries in vector _data
  void _reset();
  //! @brief Returns the number of digits of an unsigned int value
  static unsigned int _digits
    ( unsigned int n );
};

////////////////////////////////////////////////////////////////////////

template <typename T> inline T&
Tmatrix<T>::operator()
( const unsigned int ir, const unsigned int ic )
{
  return _val(ic*_nr+ir);
}

template <typename T> inline const T&
Tmatrix<T>::operator()
( const unsigned int ir, const unsigned int ic ) const
{
  return _val(ic*_nr+ir);
}

template <typename T> inline T&
Tmatrix<T>::operator()
( const unsigned int ie )
{
  return _val(ie);
}

template <typename T> inline const T&
Tmatrix<T>::operator()
( const unsigned int ie ) const
{
  return _val(ie);
}

template <typename T> inline T&
Tmatrix<T>::_val
( const unsigned int ir, const unsigned int ic )
{
  return _val(ic*_nr+ir);
}

template <typename T> inline const T&
Tmatrix<T>::_val
( const unsigned int ir, const unsigned int ic ) const
{
  return _val(ic*_nr+ir);
}

template <typename T> inline T&
Tmatrix<T>::_val
( const unsigned int ie )
{
  ASSERT( ie<_nr*_nc && ie>=0 );
  return *(_data[ie]);
}

template <typename T> inline const T&
Tmatrix<T>::_val
( const unsigned int ie ) const
{
  ASSERT( ie<_nr*_nc && ie>=0 );
  return *(_data[ie]);
}

template <typename T> inline T*&
Tmatrix<T>::pval
( const unsigned int ir, const unsigned int ic )
{
  return _pval(ic*_nr+ir);
}

template <typename T> inline T*&
Tmatrix<T>::_pval
( const unsigned int ie )
{
  ASSERT( ie<_nr*_nc && ie>=0 );
  return _data[ie];
}

template <typename T> inline void
Tmatrix<T>::_reset()
{
  typename std::vector<T*>::iterator it = _data.begin();
  for( ; it!=_data.end(); it++ ) delete *it;
}

template <typename T> inline Tmatrix<T>&
Tmatrix<T>::col
( const unsigned int ic )
{
  ASSERT( ic<_nc && ic>=0 );
  if( !_pcol ) _pcol = new Tmatrix<T>( _nr, 1, false );
  for( unsigned int ir=0; ir<_nr; ir++ ){
    _pcol->pval(ir,0) = pval(ir,ic);
#ifdef DEBUG__MATRIX_COL
    std::cout << ir << " , 0 : " << pval(ir,ic) << " "
              << _pcol->pval(ir,0) << std::endl;
#endif
  }
#ifdef DEBUG__MATRIX_COL
  std::cout << std::endl;
#endif
  return *_pcol;
}

template <typename T> inline Tmatrix<T>&
Tmatrix<T>::row
( const unsigned int ir )
{
  ASSERT( ir<_nr && ir>=0 );
  if( !_prow ) _prow = new Tmatrix<T>( 1, _nc, false );
  for( unsigned int ic=0; ic<_nc; ic++ ){
    _prow->pval(0,ic) = pval(ir,ic);
#ifdef DEBUG__MATRIX_ROW
    std::cout << "0 , " << ic << " : " << pval(ir,ic) << " "
              << _prow->pval(0,ic) << std::endl;
#endif
  }
#ifdef DEBUG__MATRIX_ROW
  std::cout << std::endl;
#endif
  return *_prow;
}

template <typename T> inline Tmatrix<T>&
Tmatrix<T>::operator=
( const Tmatrix<T>&M )
{
  if( M._nr!=_nr || M._nc!=_nc )
    resize( M._nr, M._nc );
  for( unsigned int ie=0; ie!=_nr*_nc; ie++ ){
    _val(ie) = M._val(ie);
  }
  return *this;
}

template <typename T> inline Tmatrix<T>&
Tmatrix<T>::operator=
( const T&m )
{
  for( unsigned int ie=0; ie!=_nr*_nc; ie++ ){
    _val(ie) = m;
  }
  return *this;
}

template <typename T> template <typename U> inline Tmatrix<T>&
Tmatrix<T>::operator+=
( const Tmatrix<U>&M )
{
  ASSERT( M._nr==_nr && M._nc==_nc );
  for( unsigned int ie=0; ie!=_nr*_nc; ie++ )
    _val(ie) += M._val(ie);
  return *this;
}

template <typename T> template <typename U> inline Tmatrix<T>&
Tmatrix<T>::operator+=
( const U&m )
{
  for( unsigned int ie=0; ie!=_nr*_nc; ie++ )
    _val(ie) += m;
  return *this;
}

template <typename T> template <typename U> inline Tmatrix<T>&
Tmatrix<T>::operator-=
( const Tmatrix<U>&M )
{
  ASSERT( M._nr==_nr && M._nc==_nc );
  for( unsigned int ie=0; ie!=_nr*_nc; ie++ )
    _val(ie) -= M._val(ie);
  return *this;
}

template <typename T> template <typename U> inline Tmatrix<T>&
Tmatrix<T>::operator-=
( const U&m )
{
  for( unsigned int ie=0; ie!=_nr*_nc; ie++ )
    _val(ie) -= m;
  return *this;
}

template <typename T> template <typename U> inline Tmatrix<T>&
Tmatrix<T>::operator*=
( const Tmatrix<U>&M )
{
  ASSERT( M._nr==M._nc && M._nr==_nc );
  for( unsigned int ir=0; ir<_nr; ir++ ){
    Tmatrix<T> irow = row(ir);
    for( unsigned int ic=0; ic<_nc; ic++ ){
      _val(ir,ic) = irow(0)*M(0,ic);
      for( unsigned int k=1; k<_nc; k++ ){
        _val(ir,ic) += irow(k)*M(k,ic);
      }
    }
  }
  return *this;
}

template <typename T> template <typename U> inline Tmatrix<T>&
Tmatrix<T>::operator*=
( const U&m )
{
  for( unsigned int ie=0; ie!=_nr*_nc; ie++ )
    _val(ie) *= m;
  return *this;
}

template <typename T> template <typename U> inline Tmatrix<T>&
Tmatrix<T>::operator/=
( const U&m )
{
  for( unsigned int ie=0; ie!=_nr*_nc; ie++ )
    _val(ie) /= m;
  return *this;
}

template <typename T> inline Tmatrix<T>
operator+
( const Tmatrix<T>&M )
{
  return M;
}

template <typename T, typename U> inline Tmatrix<T>
operator+
( const Tmatrix<T>&M, const Tmatrix<U>&N )
{
  Tmatrix<T> P( M );   
  P += N;
  return P;
}

template <typename T, typename U> inline Tmatrix<T>
operator+
( const Tmatrix<T>&M, const U&n )
{
  Tmatrix<T> P( M );
  P += n;
  return P;
}

template <typename T, typename U> inline Tmatrix<T>
operator+
( const U&m, const Tmatrix<T>&N )
{
  return N+m;
}

template <typename T> inline Tmatrix<T>
operator-
( const Tmatrix<T>&M )
{
  return -M;
}

template <typename T, typename U> inline Tmatrix<T>
operator-
( const Tmatrix<T>&M, const Tmatrix<U>&N )
{
  Tmatrix<T> P( M );
  P -= N;
  return P;
}

template <typename T, typename U> inline Tmatrix<T>
operator-
( const Tmatrix<T>&M, const U&n )
{
  Tmatrix<T> P( M );
  P -= n;
  return P;
}

template <typename T, typename U> inline Tmatrix<T>
operator-
( const T&m, const Tmatrix<U>&N )
{
  Tmatrix<T> P( -N );
  P += m;
  return P;
}

template <typename T, typename U> inline Tmatrix<T>
operator*
( const Tmatrix<T>&M, const Tmatrix<U>&N )
{
  ASSERT( M._nc==N._nr );
  Tmatrix<T> P( M._nr, N._nc );
  for( unsigned int ir=0; ir<M._nr; ir++ ){
    for( unsigned int ic=0; ic<N._nc; ic++ ){
      P(ir,ic) = M(ir,0)*N(0,ic);
      for( unsigned int k=1; k<M._nc; k++ ){
        P(ir,ic) += M(ir,k)*N(k,ic);
      }
    }
  }
  return P;
}

template <typename T, typename U> inline Tmatrix<T>
operator*
( const Tmatrix<T>&M, const U&n )
{
  Tmatrix<T> P( M );
  P *= n;
  return P;
}

template <typename T, typename U> inline Tmatrix<T>
operator*
( const U&m, const Tmatrix<T>&N )
{
  return N*m;
}

template <typename T, typename U> inline Tmatrix<T>
operator/
( const Tmatrix<T>&M, const U&n )
{
  Tmatrix<T> P( M );
  P /= n;
  return P;
}

template <typename T> inline std::ostream&
operator<<
( std::ostream& os, const Tmatrix<T>&M )
{
  for( unsigned int ir=0; ir<M._nr; ir++ )
    for( unsigned int ic=0; ic<M._nc; ic++ )
      os << ir << " , " << ic << " : " << M(ir,ic) << std::endl;
  return os;
}

template <typename T> inline T*
Tmatrix<T>::array() const
{
  T*pM = new T[_nr*_nc];
  for( unsigned int ic=0, ie=0; ic<_nc; ic++ )
    for( unsigned int ir=0; ir<_nr; ir++, ie++ )
      pM[ie] = _val (ir,ic);
  return pM;
}

template <typename T> inline void
Tmatrix<T>::array
( const T*pM )
{
  for( unsigned int ic=0, ie=0; ic<_nc; ic++ )
    for( unsigned int ir=0; ir<_nr; ir++, ie++ )
      _val(ir,ic) = pM[ie];
}


template <typename T> inline unsigned int
Tmatrix<T>::_digits
( unsigned int n )
{
  if( n == 0 ) return 1;
  int l = 0;
  while( n > 0 ){
    ++l;
    n /= 10;
  }
  return l;
}



CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_T_MATRIX_HPP

/*
 *	end of file
 */
