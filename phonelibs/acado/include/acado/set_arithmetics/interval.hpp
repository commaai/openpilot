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
 *    \file   include/acado/set_arithmetics/interval.hpp
 *    \author Boris Houska, Mario Villanueva, Benoit Chachuat
 *    \date   2013
 */


#ifndef ACADO_TOOLKIT_INTERVAL_HPP
#define ACADO_TOOLKIT_INTERVAL_HPP

#include <acado/utils/acado_utils.hpp>


BEGIN_NAMESPACE_ACADO

/**
 *  \brief Implements a rudimentary interval class.
 *
 *	\ingroup BasicDataStructures
 *
 *  Interval is a C++ class for calculating the natural interval              \n
 *  extension of a factorable function on a box.                              \n
 *                                                                            \n
 *  Example Code:                                                             \n
 *                                                                            \n
 *  - A simple function of two intervals:                                     \n
 *                                                                            \n
 *    \verbatim
      Interval X(-1,1);      // An interval X = [-1,1].
      Interval Y( 1,2);      // An interval Y = y[1,2].

      Interval Z = X + X*Y;  // An interval Z that contains
                             // the set { x + x*y | x in X, y in Y };

      Z.print();             // display the result.
      \endverbatim                                                            \n
 *                                                                            \n
 */


class Interval{
 
  // friends of class Interval for operator overloading
  friend Interval operator+( const Interval& );
  friend Interval operator+( const Interval&, const Interval& );
  friend Interval operator+( const double, const Interval& );
  friend Interval operator+( const Interval&, const double );
  friend Interval operator-( const Interval& );
  friend Interval operator-( const Interval&, const Interval& );
  friend Interval operator-( const double, const Interval& );
  friend Interval operator-( const Interval&, const double );
  friend Interval operator*( const Interval&, const Interval& );
  friend Interval operator*( const Interval&, const double );
  friend Interval operator*( const double, const Interval& );
  friend Interval operator/( const Interval&, const Interval& );
  friend Interval operator/( const Interval&, const double );
  friend Interval operator/( const double, const Interval& );
  friend std::ostream& operator<<( std::ostream&, const Interval& );
  friend bool operator==( const Interval&, const Interval& );
  friend bool operator!=( const Interval&, const Interval& );
  friend bool operator<=( const Interval&, const Interval& );
  friend bool operator>=( const Interval&, const Interval& );
  friend bool operator<( const Interval&, const Interval& );
  friend bool operator>( const Interval&, const Interval& );

  // friends of class Interval for function overloading
  friend double diam( const Interval& );
  friend double abs ( const Interval& );
  friend double mid ( const Interval& );
  friend double mid ( const double, const double, const double, int& );

  friend Interval inv ( const Interval& );
  friend Interval sqr ( const Interval& );
  friend Interval exp ( const Interval& );
  friend Interval log ( const Interval& );
  friend Interval cos ( const Interval& );
  friend Interval sin ( const Interval& );
  friend Interval tan ( const Interval& );
  friend Interval acos( const Interval& );
  friend Interval asin( const Interval& );
  friend Interval atan( const Interval& );
  friend Interval fabs( const Interval& );
  friend Interval sqrt( const Interval& );
  friend Interval xlog( const Interval& );
  friend Interval pow ( const Interval&, const int );
  friend Interval arh ( const Interval&, const double );
  friend Interval pow ( const Interval&, const double );
  friend Interval pow ( const Interval&, const Interval& );
  friend Interval hull( const Interval&, const Interval& );
  friend Interval min ( const Interval&, const Interval& );
  friend Interval max ( const Interval&, const Interval& );
  friend Interval min ( const unsigned int, const Interval* );
  friend Interval max ( const unsigned int, const Interval* );

  friend bool inter( Interval&, const Interval&, const Interval& );

public:

  Interval& operator= ( const double   c ){ _l  = c   ; _u  = c   ; return *this; }
  Interval& operator= ( const Interval&I ){ _l  = I._l; _u  = I._u; return *this; }
  Interval& operator+=( const double   c ){ _l += c   ; _u += c   ; return *this; }
  Interval& operator+=( const Interval&I ){ _l += I._l; _u += I._u; return *this; }
  Interval& operator-=( const double   c ){ _l -= c   ; _u -= c   ; return *this; }
  Interval& operator-=( const Interval&I ){ _l -= I._u; _u -= I._l; return *this; }
  Interval& operator*=( const double   c ){ _l *= c   ; _u *= c   ; return *this; }

  Interval& operator*=( const Interval&I ){ *this = operator*(*this,I); return *this; }
  Interval& operator/=( const double   c ){ *this = operator/(*this,c); return *this; }
  Interval& operator/=( const Interval&I ){ *this = operator/(*this,I); return *this; }

  //! @brief Default constructor
  Interval(){}

  //! @brief Constructor for a constant value <a>c</a>
  Interval( const double c ):_l(c), _u(c) {}

  //! @brief Constructor for a variable that belongs to the interval [<a>l</a>,<a>u</a>]
  Interval( const double l_, const double u_ ): _l(l_<u_?l_:u_), _u(l_<u_?u_:l_) {}

  //! @brief Copy constructor
  Interval( const Interval&I ): _l(I._l), _u(I._u) {}

  //! @brief Destructor
  ~Interval(){}

  //! @brief Returns the lower bounding value.
  const double& l() const{ return _l; }
  
  //! @brief Returns the upper bounding value.
  const double& u() const{ return _u; }

  //! @brief Sets the lower bound.
  void l ( const double lb ){ _l = lb; }
  
  //! @brief Sets the upper bound.
  void u ( const double ub ){ _u = ub; }
  /** @} */  

  void print() const{ std::cout << *this << "\n"; }

	//! Routine which returns BT_FALSE if the lower or upper bounds is equal to "nan" or "INFTY".\n
	//! Otherwise, BT_TRUE is returned.
	BooleanType isCompact() const;
	
	
private:

  double mid( const double convRel, const double concRel, const double valCut, int &indexMid ) const;
  double xlog( const double x ) const{ return x*(::log(x)); }
 
  //! @brief Lower bound
  double _l;
  //! @brief Upper bound
  double _u;
};

CLOSE_NAMESPACE_ACADO

#include <acado/set_arithmetics/interval.ipp>

#endif  // ACADO_TOOLKIT_INTERVAL_HPP

