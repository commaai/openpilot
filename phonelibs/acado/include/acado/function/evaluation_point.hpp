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
 *    \file include/acado/function/evaluation_point.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_EVALUATION_POINT_HPP
#define ACADO_TOOLKIT_EVALUATION_POINT_HPP

#include <acado/function/ocp_iterate.hpp>
#include <acado/function/function_evaluation_tree.hpp>


BEGIN_NAMESPACE_ACADO


class Function;

/** 
 *	\brief Allows to setup function evaluation points.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class EvaluationPoint is an efficient data class for storing points at which  \n
 *  a function can be evaluated. This class can be used in combination with the class \n
 *  function.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */

class EvaluationPoint{

friend class Function;

//
// PUBLIC MEMBER FUNCTIONS:
//

public:


    /** Default constructor. */
    EvaluationPoint();


    /** Constructor which takes the function. */
    EvaluationPoint( const Function &f,
                     uint  nx_ = 0,
                     uint  na_ = 0,
                     uint  nu_ = 0,
                     uint  np_ = 0,
                     uint  nw_ = 0,
                     uint  nd_ = 0,
                     uint  N_  = 0  );


    /** Constructor which takes the function. */
    EvaluationPoint( const Function   &f   ,
                     const OCPiterate &iter  );


    /** Copy constructor (deep copy). */
    EvaluationPoint( const EvaluationPoint& rhs );

    /** Destructor. */
    virtual ~EvaluationPoint( );

    /** Assignment operator (deep copy). */
    EvaluationPoint& operator=( const EvaluationPoint& rhs );


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


    returnValue init( const Function   &f   ,
                      const OCPiterate &iter  );



    inline returnValue setT ( const double &t  );
    inline returnValue setX ( const DVector &x  );
    inline returnValue setXA( const DVector &xa );
    inline returnValue setP ( const DVector &p  );
    inline returnValue setU ( const DVector &u  );
    inline returnValue setW ( const DVector &w  );
    inline returnValue setDX( const DVector &dx );

    inline returnValue setZ ( const uint       &idx ,
                              const OCPiterate &iter  );

	inline returnValue setZero( );


    inline double getT () const;
    inline DVector getX () const;
    inline DVector getXA() const;
    inline DVector getP () const;
    inline DVector getU () const;
    inline DVector getW () const;
    inline DVector getDX() const;


	    /** Prints the data of this object.              \n
     *  Due to the efficient implementation of       \n
     *  this class not everything might be stored.   \n
     *  Please, use this routine for debugging only. \n
     *  This print routine does only work properly   \n
     *  if ALL values are assigned.                  \n
     */
    returnValue print() const;


// PROTECTED MEMBER FUNCTIONS:
// ---------------------------

protected:

    inline returnValue copy( const int *order, const DVector &rhs );

    void copy( const EvaluationPoint &rhs );
    void deleteAll();

    void copyIdx( const uint &dim, const int *idx1, int **idx2 );

    inline DVector backCopy( const int *order, const uint &dim ) const;

    inline double* getEvaluationPointer() const;



// PROTECTED MEMBERS:
// ------------------

protected:

    double     *z;   /**< the function evaluation point.         */
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



#include <acado/function/evaluation_point.ipp>


#endif  // ACADO_TOOLKIT_EVALUATION_POINT_HPP

/*
 *   end of file
 */
