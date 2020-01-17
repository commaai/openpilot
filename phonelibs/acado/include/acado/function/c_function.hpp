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
 *    \file include/acado/function/c_function.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_C_FUNCTION_HPP
#define ACADO_TOOLKIT_C_FUNCTION_HPP


#include <acado/symbolic_expression/symbolic_expression.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief (no description yet)
 *
 *	\ingroup BasicDataStructures
 *
 *	...
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class CFunction{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //

    public:

    /** Default constructor. */
    CFunction( );


    /** Constructor which takes a C-function pointer.
     */
    CFunction( uint dim, cFcnPtr cFcn_ /** function pointer */ );


    /** Constructor which takes a C-function pointer as well as \n
     *  function pointers for the corresponding derivatives.    \n
     */
    CFunction( uint dim, cFcnPtr  cFcn_        ,  /** function pointer                         */
                         cFcnDPtr cFcnDForward_,  /** function pointer to forward derivatives  */
                         cFcnDPtr cFcnDBackward_  /** function pointer to backward derivatives */ );


    /** Copy constructor (deep copy). */
    CFunction( const CFunction& rhs );

    /** Destructor. */
    virtual ~CFunction( );

    /** Assignment operator (deep copy). */
    CFunction& operator=( const CFunction& rhs );


    /** Loading Expressions (deep copy). */
    virtual Expression operator()( const Expression &arg );


    /** Returns the dimension of the symbolic expression  \n
     *  \return The requested dimension.
     */
    virtual uint getDim () const;


    /** Evaluates the expression
     *  \return SUCCESSFUL_RETURN                  \n
     *          RET_NAN                            \n
     * */
    virtual returnValue evaluate( double *x         /**< the input variable x */,
                                  double *result    /**< the result           */  );



     /** Possible user implementation of a C function \n
      *  \return (void)                               \n
      */
     virtual void evaluateCFunction( double *x         /**< the input variable x */,
                                     double *result    /**< the result           */   );



    /** Evaluates the expression and also prints   \n
     *  the intermediate results with a specified  \n
     *  print level.                               \n
     *  \return SUCCESFUL_RETURN                   \n
     *          RET_NAN                            \n
     * */
    virtual returnValue evaluate( double *x         /**< the input variable x */,
                                  double *result    /**< the result           */,
                                  PrintLevel printL /**< the print level      */    );


    /** Evaluates the expression and stores the intermediate      \n
     *  results in a buffer (needed for automatic differentiation \n
     *  in backward mode)                                         \n
     *  \return SUCCESFUL_RETURN                   \n
     *          RET_NAN                            \n
     * */
    virtual returnValue evaluate( int     number    /**< storage position     */,
                                  double *x         /**< the input variable x */,
                                  double *result    /**< the result           */  );


    /** Automatic Differentiation in forward mode.                \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     virtual returnValue AD_forward( double *x     /**< The evaluation
                                                        point x          */,
                                     double *seed  /**< the seed         */,
                                     double *f     /**< the value of the
                                                        expression at x  */,
                                     double *df    /**< the derivative of
                                                        the expression   */  );


    /** Automatic Differentiation in forward mode.                \n
     *  This function uses the intermediate                       \n
     *  results from a buffer                                     \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     virtual returnValue AD_forward( int     number  /**< storage position */,
                                     double *seed    /**< the seed         */,
                                     double *df      /**< the derivative of
                                                          the expression   */  );


    /** Automatic Differentiation in forward mode.                \n
     *  This function stores the intermediate                     \n
     *  results in a buffer (needed for 2nd order automatic       \n
     *  differentiation in backward mode)                         \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     virtual returnValue AD_forward( int     number  /**< storage position */,
                                     double *x       /**< The evaluation
                                                          point x          */,
                                     double *seed    /**< the seed         */,
                                     double *f       /**< the value of the
                                                          expression at x  */,
                                     double *df      /**< the derivative of
                                                          the expression   */  );


    // IMPORTANT REMARK FOR AD_BACKWARD: run evaluate first to define
    //                                   the point x and to compute f.

    /** Automatic Differentiation in backward mode.               \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     virtual returnValue AD_backward( double *seed /**< the seed         */,
                                      double  *df  /**< the derivative of
                                                        the expression   */   );



    // IMPORTANT REMARK FOR AD_BACKWARD: run evaluate first to define
    //                                   the point x and to compute f.

    /** Automatic Differentiation in backward mode based on       \n
     *  buffered values                                           \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     virtual returnValue AD_backward( int    number /**< the buffer
                                                         position         */,
                                      double *seed  /**< the seed         */,
                                      double  *df   /**< the derivative of
                                                         the expression   */);


    /** Automatic Differentiation in forward mode for             \n
     *  2nd derivatives.                                          \n
     *  This function uses intermediate                           \n
     *  results from a buffer.                                    \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     virtual returnValue AD_forward2( int    number  /**< the buffer
                                                          position         */,
                                      double *seed1  /**< the seed         */,
                                      double *seed2  /**< the seed for the
                                                          first derivative */,
                                      double *df     /**< the derivative of
                                                          the expression   */,
                                      double *ddf    /**< the 2nd derivative
                                                          of the expression*/);


    // IMPORTANT REMARK FOR AD_BACKWARD2: run AD_forward first to define
    //                                    the point x and to compute f and df.

    /** Automatic Differentiation in backward mode for 2nd order  \n
     *  derivatives based on buffered values.                     \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     virtual returnValue AD_backward2( int    number /**< the buffer
                                                          position           */,
                                       double *seed1 /**< the seed1          */,
                                       double *seed2 /**< the seed2          */,
                                       double   *df  /**< the 1st derivative
                                                          of the expression  */,
                                       double  *ddf  /**< the 2nd derivative
                                                          of the expression  */   );



     /** Clears the buffer and resets the buffer size \n
      *  to 1.                                        \n
      *  \return SUCCESFUL_RETURN                     \n
      */
     virtual returnValue clearBuffer();


    /** Specify a pointer to be passed to the c-function          \n
     *  during every call.                                        \n
     *  This pointer can e.g. point to to measurements,           \n
     *  to an object of any class or to a struct.                 \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     virtual returnValue setUserData( void * user_data_ /**< the user-defined pointer */ );



//
//  PROTECTED FUNCTIONS:
//

protected:

     /** Protected copy routine.
      */
     void copy( const CFunction &arg );


     /** Protected delete routine.
      */
     void deleteAll();



//
//  PROTECTED MEMBERS:
//

     protected:

        /** Initializes the CFunction */
        returnValue initialize();

        cFcnPtr   cFcn         ;
        cFcnDPtr  cFcnDForward ;
        cFcnDPtr  cFcnDBackward;

  
	void* user_data        ;    /**< pointer specified by the setUserData function, passed to the c function when being called */


        uint     nn            ;    /**< size of the argument (input)     */ 
        uint     dim           ;    /**< size of the function (output)    */

        uint     maxAlloc      ;    /**< actual memory allocation         */
        double **xStore        ;    /**< storage of evaluation variables  */
        double **seedStore     ;    /**< storage of evaluation seeds      */
};


CLOSE_NAMESPACE_ACADO




#endif  // ACADO_TOOLKIT_C_FUNCTION_HPP

/*
 *   end of file
 */

