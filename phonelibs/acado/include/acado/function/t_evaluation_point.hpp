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
 *    \file include/acado/function/t_evaluation_point.hpp
 *    \author Boris Houska
 */


#ifndef ACADO_TOOLKIT_T_EVALUATION_POINT_HPP
#define ACADO_TOOLKIT_T_EVALUATION_POINT_HPP

#include <acado/function/ocp_iterate.hpp>
#include <acado/function/function_evaluation_tree.hpp>

BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to setup function evaluation points.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class TevaluationPoint is an efficient data class for storing points at which \n
 *  a function can be evaluated. This class can be used in combination with the class \n
 *  function. The difference to the class EvaluationPoint is that it can be used with \n
 *  any templated basis class. 
 *
 *	\author Boris Houska
 */

template <typename T> class TevaluationPoint{

friend class Function;

//
// PUBLIC MEMBER FUNCTIONS:
//

public:


    /** Default constructor. */
    TevaluationPoint();


    /** Constructor which takes the function. */
    TevaluationPoint( const Function &f,
                      uint  nx_ = 0,
                      uint  na_ = 0,
                      uint  nu_ = 0,
                      uint  np_ = 0,
                      uint  nw_ = 0,
                      uint  nd_ = 0,
                      uint  N_  = 0  );


    /** Copy constructor (deep copy). */
    TevaluationPoint( const TevaluationPoint<T>& rhs );

    /** Destructor. */
    virtual ~TevaluationPoint( );

    /** Assignment operator (deep copy). */
    TevaluationPoint<T>& operator=( const TevaluationPoint<T>& rhs );


    /** Initializer which takes the dimensions as arguments.  \n
     *                                                        \n
     *  \param f  the function to be evaluated.               \n
     *  \param nx  number of differential states.             \n
     *  \param na  number of algebraic    states.             \n
     *  \param np  number of parameters.                      \n
     *  \param nu  number of controls.                        \n
     *  \param nw  number of disturbances.                    \n
     *  \param nd  number of diff. state derivatives.         \n
     *                                                        \n
     *  \return SUCCESSFUL_RETURN                             \n
     */
    returnValue init( const Function &f,
                      uint  nx_ = 0,
                      uint  na_ = 0,
                      uint  np_ = 0,
                      uint  nu_ = 0,
                      uint  nw_ = 0,
                      uint  nd_ = 0,
                      uint  N_  = 0      );


    inline returnValue setT ( const Tmatrix<T> &t  );
    inline returnValue setX ( const Tmatrix<T> &x  );
    inline returnValue setXA( const Tmatrix<T> &xa );
    inline returnValue setP ( const Tmatrix<T> &p  );
    inline returnValue setU ( const Tmatrix<T> &u  );
    inline returnValue setW ( const Tmatrix<T> &w  );
    inline returnValue setDX( const Tmatrix<T> &dx );

    inline Tmatrix<T> getT () const;
    inline Tmatrix<T> getX () const;
    inline Tmatrix<T> getXA() const;
    inline Tmatrix<T> getP () const;
    inline Tmatrix<T> getU () const;
    inline Tmatrix<T> getW () const;
    inline Tmatrix<T> getDX() const;


// PROTECTED MEMBER FUNCTIONS:
// ---------------------------

protected:

    inline returnValue copy( const int *order, const Tmatrix<T> &rhs );

    void copy( const TevaluationPoint &rhs );
    void deleteAll();

    void copyIdx( const uint &dim, const int *idx1, int **idx2 );

    inline Tmatrix<T> backCopy( const int *order, const uint &dim ) const;

    inline Tmatrix<T>* getEvaluationPointer() const;


// PROTECTED MEMBERS:
// ------------------

protected:

    Tmatrix<T> *z;   /**< the function evaluation point.         */
    int     **idx;   /**< index lists (for efficient reordering) */

    uint       nx;   /**< number of diff. states     */
    uint       na;   /**< number of alg. states      */
    uint       nu;   /**< number of controls         */
    uint       np;   /**< number of parameters       */
    uint       nw;   /**< number of disturbances     */
    uint       nd;   /**< number of diff. state der. */
    uint       N ;   /**< total number of variables  */
};


CLOSE_NAMESPACE_ACADO

#include <acado/function/function_.hpp>

BEGIN_NAMESPACE_ACADO


template <typename T> TevaluationPoint<T>::TevaluationPoint( ){ idx = 0; z = 0; }
template <typename T> TevaluationPoint<T>::TevaluationPoint( const TevaluationPoint<T>& rhs ){ copy(rhs); }
template <typename T> TevaluationPoint<T>::~TevaluationPoint(){ deleteAll(); }

template <typename T> TevaluationPoint<T>::TevaluationPoint( const Function &f ,
                                                             uint nx_, uint na_,
                                                             uint nu_, uint np_,
                                                             uint nw_, uint nd_, uint N_){

    idx = 0;
    z = 0;
    init(f,nx_,na_,nu_,np_,nw_,nd_,N_);
}



template <typename T> TevaluationPoint<T>& TevaluationPoint<T>::operator=( const TevaluationPoint<T>& rhs ){

    if( this != &rhs ){
        deleteAll();
        copy(rhs);
    }
    return *this;
}


template <typename T> returnValue TevaluationPoint<T>::init( const Function &f ,
                                                             uint nx_, uint na_, uint np_,
                                                             uint nu_, uint nw_, uint nd_,
                                                             uint N_                       ){

    uint run1;
    deleteAll();

    nx = acadoMax( nx_, f.getNX ()                 );
    na = acadoMax( na_, f.getNXA()                 );
    np = acadoMax( np_, f.getNP ()                 );
    nu = acadoMax( nu_, f.getNU ()                 );
    nw = acadoMax( nw_, f.getNW ()                 );
    nd = acadoMax( nd_, f.getNDX()                 );
    N  = acadoMax( N_ , f.getNumberOfVariables()+1 );

    z = new Tmatrix<T>(N);

    idx = new int*[7];

    idx[0] = new int [1 ];
    idx[1] = new int [nx];
    idx[2] = new int [na];
    idx[3] = new int [np];
    idx[4] = new int [nu];
    idx[5] = new int [nw];
    idx[6] = new int [nd];

    idx[0][0] = f.index( VT_TIME, 0 );

    for( run1 = 0; run1 < nx; run1++ )
        idx[1][run1] = f.index( VT_DIFFERENTIAL_STATE, run1 );

    for( run1 = 0; run1 < na; run1++ )
        idx[2][run1] = f.index( VT_ALGEBRAIC_STATE, run1 );

    for( run1 = 0; run1 < np; run1++ )
        idx[3][run1] = f.index( VT_PARAMETER, run1 );

    for( run1 = 0; run1 < nu; run1++ )
        idx[4][run1] = f.index( VT_CONTROL, run1 );

    for( run1 = 0; run1 < nw; run1++ )
        idx[5][run1] = f.index( VT_DISTURBANCE, run1 );

    for( run1 = 0; run1 < nd; run1++ )
        idx[6][run1] = f.index( VT_DDIFFERENTIAL_STATE, run1 );

    return SUCCESSFUL_RETURN;
}



//
// PROTECTED MEMBER FUNCTIONS:
//

template <typename T> void TevaluationPoint<T>::copyIdx( const uint &dim, const int *idx1, int **idx2 ){

    uint i;
    *idx2 = new int[dim];
    for( i = 0; i < N; i++ )
        *idx2[i] = idx1[i];
}


template <typename T> void TevaluationPoint<T>::copy( const TevaluationPoint<T> &rhs ){

    nx = rhs.nx;
    na = rhs.na;
    np = rhs.np;
    nu = rhs.nu;
    nw = rhs.nw;
    nd = rhs.nd;
    N  = rhs.N ;

    if( rhs.z != 0 ){
        z = new Tmatrix<T>(*rhs.z);
    }
    else z = 0;

    if( rhs.idx != 0 ){

        idx = new int*[7];
        copyIdx(  1, rhs.idx[0], &idx[0] );
        copyIdx( nx, rhs.idx[1], &idx[1] );
        copyIdx( na, rhs.idx[2], &idx[2] );
        copyIdx( np, rhs.idx[3], &idx[3] );
        copyIdx( nu, rhs.idx[4], &idx[4] );
        copyIdx( nw, rhs.idx[5], &idx[5] );
        copyIdx( nd, rhs.idx[6], &idx[6] );
    }
    else idx = 0;
}


template <typename T> void TevaluationPoint<T>::deleteAll(){

    if( z != 0 ) delete z;
  
    if( idx != 0 ){
        uint i;
        for( i = 0; i < 7; i++ )
            delete[] idx[i];
        delete[] idx;
    }
}

CLOSE_NAMESPACE_ACADO



#include <acado/function/t_evaluation_point.ipp>


#endif  // ACADO_TOOLKIT_T_EVALUATION_POINT_HPP

/*
 *   end of file
 */
