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
 *    \file   include/acado/set_arithmetics/taylor_model.hpp
 *    \author Boris Houska, Mario Villanueva, Benoit Chachuat
 *    \date   2013
 */


#ifndef ACADO_TOOLKIT_TAYLOR_MODEL_HPP
#define ACADO_TOOLKIT_TAYLOR_MODEL_HPP

#include <acado/utils/acado_utils.hpp>


BEGIN_NAMESPACE_ACADO

template <typename T> class TaylorVariable;

//! @brief C++ class supporting the definition and computation of Taylor models for factorable functions
////////////////////////////////////////////////////////////////////////
//! mc::TaylorModel<T> is a C++ base class that supports the definition and
//! computation of Taylor models for factorable functions on a box. The
//! template parameter T corresponds to the type used in computing the
//! remainder error bound.
////////////////////////////////////////////////////////////////////////
template <typename T>
class TaylorModel
////////////////////////////////////////////////////////////////////////
{

friend class TaylorVariable<T>;
template <typename U> friend class TaylorModel;

public:

  //! @brief Constructor
  TaylorModel
    ( const unsigned int nvar_, const unsigned int nord_ )
    { _size( nvar_, nord_ ); }

  //! @brief Destructor
  ~TaylorModel()
    { _cleanup(); }

  //! @brief Get number of variables in the TaylorModel
  unsigned int nvar() const
    { return _nvar; };
  //! @brief Get order of the TaylorModel
  unsigned int nord() const
    { return _nord; };
  //! @brief Reset the TaylorModel
  void reset()
    { _reset(); };

  //! @brief Taylor model exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for TaylorModel exception handling
    enum TYPE{
      DIV=1,	//!< Error during calculation of a TaylorVariable for a division term (division by zero)
      INTER,	//!< Error during intersection of two sets of bounds (empty intersection)
      EIGEN,	//!< Error during eigenvalue decomposition in range bounder
      SCALING,	//!< Error during scaling (degenerate variable range)
      SIZE=-1,	//!< Error due to an inconsistent number of variables (zero variables)
      INIT=-2,	//!< Error due to invalid initialization of a TaylorVariable
      INCON=-3, //!< Error due to inconsistency between the Taylor model and T bounders
      TMODEL=-4,//!< Error due to an operation between TaylorVariable linked to different TaylorModel
      UNDEF=-33	//!< Error due to calling a function/feature not yet implemented
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr_ ) : _ierr( ierr_ ){}
    //! @brief Inline function returning the error flag
    int ierr(){ return _ierr; }

  private:
    TYPE _ierr;
  };

  //! @brief Taylor model options
  struct Options
  {
    //! @brief Constructor
    Options():
      BOUNDER_TYPE(LSB), PROPAGATE_BNDT(false), INTER_WITH_BNDT(false),
      SCALE_VARIABLES(true), CENTER_REMAINDER(true), REF_MIDPOINT(true)
      {}
    //! @brief Copy constructor
    template <typename U> Options
      ( U&_options )
      : BOUNDER_TYPE( _options.BOUNDER_TYPE ),
        PROPAGATE_BNDT( _options.PROPAGATE_BNDT ),
        INTER_WITH_BNDT( _options.INTER_WITH_BNDT ),
        SCALE_VARIABLES( _options.SCALE_VARIABLES ),
        CENTER_REMAINDER( _options.CENTER_REMAINDER ),
        REF_MIDPOINT( _options.REF_MIDPOINT )
      {}
    template <typename U> Options& operator =
      ( U&_options ){
        BOUNDER_TYPE = _options.BOUNDER_TYPE;
        PROPAGATE_BNDT = _options.PROPAGATE_BNDT;
        INTER_WITH_BNDT = _options.INTER_WITH_BNDT;
        SCALE_VARIABLES = _options.SCALE_VARIABLES;
        CENTER_REMAINDER = _options.CENTER_REMAINDER;
        REF_MIDPOINT = _options.REF_MIDPOINT;
        return *this;
      }
    //! @brief Taylor model range bounders
    enum BOUNDER{
      NAIVE=0,	//!< Naive polynomial range bounder
      LSB,	//!< Lin & Stadtherr range bounder
      EIGEN,	//!< Eigenvalue decomposition-based bounder
      HYBRID	//!< Hybrid LSB + EIGEN range bounder
    };
    //! @brief Flag indicating the Taylor model range bounder
    int BOUNDER_TYPE;
    //! @brief Flag indicating whether interval bound are to be propagated in T arithmetic
    bool PROPAGATE_BNDT;
    //! @brief Flag indicating whether interval bound for the Taylor model and T arithmetic are to be intersected
    bool INTER_WITH_BNDT;
    //! @brief Flag indicating whether the variables are to be scaled to [-1,1] internally -- this requires proper intervals!
    bool SCALE_VARIABLES;
    //! @brief Flag indicating whether the remainder term is to be centered after each operation
    bool CENTER_REMAINDER;
    //! @brief Flag indicating whether the reference in Taylor expansion of univariate function is taken as the mid-point of the inner Taylor model (true) or the constant term in the centered inner Taylor model (false)
    bool REF_MIDPOINT;
  } options;
  
  //! @brief Pause the program execution and prompt the user
  static void pause();
    
private:  
  //! @brief Model order of the model
  unsigned int _nord;
  //! @brief Number of independent variables
  unsigned int _nvar;
  //! @brief Number of monomial terms
  unsigned int _nmon;
  //! @brief Positions of terms of degrees 1,...,_nord
  unsigned int *_posord;
  //! @brief Variable exponents for monomial terms 1,...,_nmon
  unsigned int *_expmon;
  //! @brief Variable exponents resulting from the product of two monomial terms 1,...,_nmon
  unsigned int **_prodmon;
  //! @brief Bounds on all the monomial terms 1,...,_nmon for given interval vector \f$X\f$
  T *_bndmon;
  //! @brief Have any of the model variables been modified?
  bool _modvar;
  //! @brief Binomial coefficients
  unsigned int *_binom;
  //! @brief Bounds on the terms \f$[X-{\rm mid}(X)]^i\f$ for given \f$X\f$
  T **_bndpow;
  //! @brief Reference point
  double *_refpoint;
  //! @brief Variable scaling
  double *_scaling; 

  //! @brief Taylor variable to speed-up computations and reduce dynamic allocation
  TaylorVariable<T>* _TV;

  //! @brief Set the order (nord) and number of variables (nvar)
  void _size
    ( const unsigned int nvar, const unsigned int nord );

//   //! @brief Set the order (nord) and number of variables (nvar)
//   template <typename U> void _copy_data
//     ( const TaylorModel<U>&TM );

  //! @brief Populate _bndpow[ix] w/ bounds on the terms \f$[X-{\rm mid}(X)]^{ix}\f$
  void _set_bndpow
    ( const unsigned int ix, const T&X, const double scaling );

  //! @brief Populate _bndmon w/ bounds on all possible monomial terms
  void _set_bndmon();

  //! @brief Populate array _posord w/ positions of terms of degrees 1,...,nord
  void _set_posord();

  //! @brief Populate array _expmon w/ exponents for monomial terms 1,...,nmon
  void _set_expmon();
  
  //! @brief Generate exponent configuration for subsequent monomial terms
  void _next_expmon
    ( unsigned int *iexp, const unsigned int iord );

  //! @brief Populate array _prodmon w/ exponents resulting from the product of two monomial terms 1,...,nmon
  void _set_prodmon();
    
  //! @brief Locates position in _posord of monomial term with variable exponents iexp
  unsigned int _loc_expmon
    ( const unsigned int *iexp );
    
  //! @brief Populate array _binom w/ binomial coefficients
  void _set_binom();

  //! @brief Return binomial coefficient \f$\left(\stackrel{i}{j}\right)\f$
  unsigned int _get_binom
    ( const unsigned int n, const unsigned int k ) const;

  //! @brief Reset the variable bound arrays
  void _reset();

  //! @brief Clean up the arrays
  void _cleanup();

  //! @brief Display the elements of a 2D array
  template< typename U > static void _display
    ( const unsigned int m, const unsigned int n, U*&a, const unsigned int lda,
      const std::string&stra, std::ostream&os );
};


CLOSE_NAMESPACE_ACADO

#include <acado/set_arithmetics/taylor_variable.hpp>
#include <acado/set_arithmetics/taylor_model.ipp>
#include <acado/set_arithmetics/taylor_variable.ipp>

#endif  // ACADO_TOOLKIT_TAYLOR_MODEL_HPP

