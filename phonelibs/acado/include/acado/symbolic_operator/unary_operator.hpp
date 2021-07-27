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
 *    \file   include/symbolic_operator/unary_operator.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date   2010
 */


#ifndef ACADO_TOOLKIT_UNARY_OPERATOR_HPP
#define ACADO_TOOLKIT_UNARY_OPERATOR_HPP


#include <acado/symbolic_operator/symbolic_operator_fwd.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Abstract base class for all scalar-valued unary operators within the symbolic operators family.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class UnaryOperator serves as a base class all scalar-valued 
 *	unary operators within the symbolic operators family.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */

class UnaryOperator : public SmoothOperator{

public:

    /** Default constructor. */
    UnaryOperator();

    /** Default constructor. */
    UnaryOperator( Operator *_argument );

    /** Copy constructor (deep copy). */
    UnaryOperator( const UnaryOperator &arg );

    /** Default destructor. */
    virtual ~UnaryOperator();

    /** Assignment Operator (deep copy). */
    UnaryOperator& operator=( const UnaryOperator &arg );


    /** Evaluates the expression and stores the intermediate      \n
     *  results in a buffer (needed for automatic differentiation \n
     *  in backward mode)                                         \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
    virtual returnValue evaluate( int     number    /**< storage position     */,
                                  double *x         /**< the input variable x */,
                                  double *result    /**< the result           */  );


    /** Evaluates the expression (templated version) */
    virtual returnValue evaluate( EvaluationBase *x ) = 0;
	
	
    /** Returns the derivative of the expression with respect     \n
     *  to the variable var(index).                               \n
     *  \return The expression for the derivative.                \n
     *
     */
    Operator* differentiate( int index  /**< diff. index    */ );


    /** Automatic Differentiation in forward mode on the symbolic \n
     *  level. This function generates an expression for a        \n
     *  forward derivative                                        \n
     *  \return SUCCESSFUL_RETURN                                 \n
     */
    virtual Operator* AD_forward( int                dim      , /**< dimension of the seed */
                                  VariableType      *varType  , /**< the variable types    */
                                  int               *component, /**< and their components  */
                                  Operator         **seed     , /**< the forward seed      */
                                  int                &nNewIS  , /**< the number of new IS  */
                                  TreeProjection  ***newIS      /**< the new IS-pointer    */ );


    /** Automatic Differentiation in backward mode on the symbolic \n
     *  level. This function generates an expression for a         \n
     *  backward derivative                                        \n
     *  \return SUCCESSFUL_RETURN                                  \n
     */
    virtual returnValue AD_backward( int           dim      , /**< number of directions  */
                                     VariableType *varType  , /**< the variable types    */
                                     int          *component, /**< and their components  */
                                     Operator     *seed     , /**< the backward seed     */
                                     Operator    **df       , /**< the result            */
                                     int           &nNewIS  , /**< the number of new IS  */
                                     TreeProjection ***newIS  /**< the new IS-pointer    */ );

    
    
    /** Automatic Differentiation in symmetric mode on the symbolic \n
     *  level. This function generates an expression for a          \n
     *  second order derivative.                                    \n
     *  \return SUCCESSFUL_RETURN                                   \n
     */
     virtual returnValue AD_symmetric( int            dim       , /**< number of directions  */
                                      VariableType  *varType   , /**< the variable types    */
                                      int           *component , /**< and their components  */
                                      Operator      *l         , /**< the backward seed     */
                                      Operator     **S         , /**< forward seed matrix   */
                                      int            dimS      , /**< dimension of forward seed             */
                                      Operator     **dfS       , /**< first order foward result             */
                                      Operator     **ldf       , /**< first order backward result           */
                                      Operator     **H         , /**< upper trianglular part of the Hessian */
                                      int            &nNewLIS  , /**< the number of newLIS  */
                                      TreeProjection ***newLIS , /**< the new LIS-pointer   */
                                      int            &nNewSIS  , /**< the number of newSIS  */
                                      TreeProjection ***newSIS , /**< the new SIS-pointer   */
                                      int            &nNewHIS  , /**< the number of newHIS  */
                                      TreeProjection ***newHIS   /**< the new HIS-pointer   */ );
       

    /** Substitutes var(index) with the expression sub.           \n
     *  \return The substituted expression.                       \n
     *
     */
    virtual Operator* substitute( int index           /**< subst. index    */,
                                  const Operator *sub /**< the substitution*/  ) = 0;



    /** Checks whether the expression is zero or one              \n
     *  \return NE_ZERO                                           \n
     *          NE_ONE                                            \n
     *          NE_NEITHER_ONE_NOR_ZERO                           \n
     *
     */
    virtual NeutralElement isOneOrZero() const;



    /** Asks the expression whether it is depending on a certian type of \n
     * variable.                                                         \n
     * \return BT_TRUE if a dependency is detected,                      \n
     *         BT_FALSE otherwise.                                       \n
     */
    virtual BooleanType isDependingOn( VariableType var ) const;


    /** Checks whether the expression is depending on a variable  \n
     *  \return BT_FALSE if no dependence is detected             \n
     *          BT_TRUE  otherwise                                \n
     *
     */
    virtual BooleanType isDependingOn( int           dim      ,    /**< number of directions  */
                                       VariableType *varType  ,    /**< the variable types    */
                                       int          *component,    /**< and their components  */
                                       BooleanType   *implicit_dep /**< implicit dependencies */ );


    /** Checks whether the expression is linear in                \n
     *  (or not depending on) a variable                          \n
     *  \return BT_FALSE if no linearity is                       \n
     *                detected                                    \n
     *          BT_TRUE  otherwise                                \n
     *
     */
    virtual BooleanType isLinearIn( int           dim      ,    /**< number of directions  */
                                    VariableType *varType  ,    /**< the variable types    */
                                    int          *component,    /**< and their components  */
                                    BooleanType  *implicit_dep  /**< implicit dependencies */ );


    /** Checks whether the expression is polynomial in            \n
     *  the specified variables                                   \n
     *  \return BT_FALSE if the expression is not  polynomial     \n
     *          BT_TRUE  otherwise                                \n
     */
    virtual BooleanType isPolynomialIn( int           dim      ,    /**< number of directions  */
                                        VariableType *varType  ,    /**< the variable types    */
                                        int          *component,    /**< and their components  */
                                        BooleanType  *implicit_dep  /**< implicit dependencies */ );


    /** Checks whether the expression is rational in              \n
     *  the specified variables                                   \n
     *  \return BT_FALSE if the expression is not rational        \n
     *          BT_TRUE  otherwise                                \n
     */
    virtual BooleanType isRationalIn( int           dim      ,    /**< number of directions  */
                                      VariableType *varType  ,    /**< the variable types    */
                                      int          *component,    /**< and their components  */
                                      BooleanType  *implicit_dep  /**< implicit dependencies */ );


    /** Returns the monotonicity of the expression.               \n
     *  \return MT_NONDECREASING                                  \n
     *          MT_NONINCREASING                                  \n
     *          MT_NONMONOTONIC                                   \n
     */
    virtual MonotonicityType getMonotonicity( );


    /** Returns the curvature of the expression                   \n
     *  \return CT_CONSTANT                                       \n
     *          CT_AFFINE                                         \n
     *          CT_CONVEX                                         \n
     *          CT_CONCAVE                                        \n
     */
    virtual CurvatureType getCurvature( );


    /** Overwrites the monotonicity of the expression.            \n
     *  (For the case that the monotonicity is explicitly known)  \n
     *  \return SUCCESSFUL_RETURN                                 \n
     *
     */
    virtual returnValue setMonotonicity( MonotonicityType monotonicity_ );


    /** Overwrites the curvature of the expression.               \n
     *  (For the case that the curvature is explicitly known)     \n
     *  \return SUCCESSFUL_RETURN                                 \n
     *
     */
    virtual returnValue setCurvature( CurvatureType curvature_  );


    /** Automatic Differentiation in forward mode.                \n
     *  This function uses the intermediate                       \n
     *  results from a buffer                                     \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
    virtual returnValue AD_forward( int     number  /**< storage position */,
                                    double *seed    /**< the seed         */,
                                    double *df      /**< the derivative   */  );


    /** Automatic Differentiation in forward mode.                \n
     *  This function stores the intermediate                     \n
     *  results in a buffer (needed for 2nd order automatic       \n
     *  differentiation in backward mode)                         \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
    virtual returnValue AD_forward( int     number  /**< storage position                  */,
                                    double *x       /**< The evaluation point x            */,
                                    double *seed    /**< the seed                          */,
                                    double *f       /**< the value of the expression at x  */,
                                    double *df      /**< the derivative of the expression  */  );


    /** Automatic Differentiation in backward mode based on       \n
     *  buffered values                                           \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
    virtual returnValue AD_backward( int    number /**< the buffer position */,
                                     double seed   /**< the seed            */,
                                     double  *df   /**< the derivative      */ );


    /** Automatic Differentiation in forward mode for             \n
     *  2nd derivatives.                                          \n
     *  This function uses intermediate                           \n
     *  results from a buffer.                                    \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
    virtual returnValue AD_forward2( int    number  /**< the buffer position               */,
                                     double *seed1  /**< the seed                          */,
                                     double *seed2  /**< the seed for the first derivative */,
                                     double *df     /**< the derivative                    */,
                                     double *ddf    /**< the 2nd derivative                */);


    /** Automatic Differentiation in backward mode for 2nd order  \n
     *  derivatives based on buffered values.                     \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
    virtual returnValue AD_backward2( int    number /**< the buffer position  */,
                                      double seed1  /**< the seed1            */,
                                      double seed2  /**< the seed2            */,
                                      double   *df  /**< the 1st derivative   */,
                                      double  *ddf  /**< the 2nd derivative   */  );



    /** Prints the expression into a stream. \n
     *  \return SUCCESFUL_RETURN             \n
     */
     virtual std::ostream& print( std::ostream &stream ) const;


    /** Provides a deep copy of the expression. \n
     *  \return a clone of the expression.      \n
     */
    virtual Operator* clone() const = 0;


    /** Clears the buffer and resets the buffer size \n
     *  to 1.                                        \n
     *  \return SUCCESFUL_RETURN                     \n
     */
    virtual returnValue clearBuffer();



    /** Enumerates all variables based on a common   \n
     *  IndexList.                                   \n
     *  \return SUCCESFUL_RETURN
     */
    virtual returnValue enumerateVariables( SymbolicIndexList *indexList );



    /** Asks the expression for its name.   \n
     *  \return the name of the expression. \n
     */
    virtual OperatorName getName();


    /** Asks the expression whether it is a variable.   \n
     *  \return The answer. \n
     */
    virtual BooleanType isVariable( VariableType &varType,
                                    int          &component ) const;


    /** The function loadIndices passes an IndexList through    \n
     *  the whole expression tree. Whenever a variable gets the \n
     *  IndexList it tries to make an entry. However if a       \n
     *  variable recognices that it has already been added      \n
     *  before it will not be allowed to make a second entry.   \n
     *  Note that all variables, in paticular the intermediate  \n
     *  states, will keep in mind whether they were allowed     \n
     *  to make an entry or not. This guarantees that           \n
     *  intermediate states are never evaluated twice if they   \n
     *  occur at several knots of the tree.                     \n
     *                                                          \n
     *  THIS FUNCTION IS FOR INTERNAL USE ONLY.                 \n
     *                                                          \n
     *  PLEASE CALL THIS FUNTION AT MOST ONES FOR AN EXPRESSION \n
     *  AS A KIND OF INIT ROUTINE.                              \n
     *                                                          \n
     *  \return the name of the expression.                     \n
     */
    virtual returnValue loadIndices( SymbolicIndexList *indexList );


    /** Asks whether all elements are purely symbolic.                \n
      *                                                               \n
      * \return BT_TRUE  if the complete tree is symbolic.            \n
      *         BT_FALSE otherwise (e.g. if C functions are linked).  \n
      */
    virtual BooleanType isSymbolic() const;


//
//  PROTECTED FUNCTIONS:
//

protected:


    /** Automatic Differentiation in forward mode on the symbolic \n
     *  level. This function generates an expression for a        \n
     *  forward derivative                                        \n
     *  \return SUCCESSFUL_RETURN                                 \n
     */
     Operator* ADforwardProtected( int                dim      , /**< dimension of the seed */
                                             VariableType      *varType  , /**< the variable types    */
                                             int               *component, /**< and their components  */
                                             Operator       **seed     , /**< the forward seed      */
                                             int                &nNewIS  , /**< the number of new IS  */
                                             TreeProjection ***newIS    /**< the new IS-pointer    */ );



    /** Automatic Differentiation in backward mode on the symbolic \n
     *  level. This function generates an expression for a         \n
     *  backward derivative                                        \n
     *  \return SUCCESSFUL_RETURN                                  \n
     */
     returnValue ADbackwardProtected( int            dim      , /**< number of directions  */
                                              VariableType  *varType  , /**< the variable types    */
                                              int           *component, /**< and their components  */
                                              Operator      *seed     , /**< the backward seed     */
                                              Operator     **df       , /**< the result            */
                                              int            &nNewIS  , /**< the number of new IS  */
                                              TreeProjection ***newIS    /**< the new IS-pointer   */ );


    /** Automatic Differentiation in symmetric mode on the symbolic \n
     *  level. This function generates an expression for a          \n
     *  second order derivative.                                    \n
     *  \return SUCCESSFUL_RETURN                                   \n
     */
     returnValue ADsymmetricProtected( int            dim       , /**< number of directions  */
                                               VariableType  *varType   , /**< the variable types    */
                                               int           *component , /**< and their components  */
                                               Operator      *l         , /**< the backward seed     */
                                               Operator     **S         , /**< forward seed matrix   */
                                               int            dimS      , /**< dimension of forward seed             */
                                               Operator     **dfS       , /**< first order foward result             */
                                               Operator     **ldf       , /**< first order backward result           */
                                               Operator     **H         , /**< upper trianglular part of the Hessian */
                                               int            &nNewLIS  , /**< the number of newLIS  */
                                               TreeProjection ***newLIS , /**< the new LIS-pointer   */
                                               int            &nNewSIS  , /**< the number of newSIS  */
                                               TreeProjection ***newSIS , /**< the new SIS-pointer   */
                                               int            &nNewHIS  , /**< the number of newHIS  */
                                               TreeProjection ***newHIS   /**< the new HIS-pointer   */ );


					      
	/** Sets the name of the variable that is used for code export.   \n
	 *  \return SUCCESSFUL_RETURN                                     \n
	 */
     virtual returnValue setVariableExportName(	const VariableType &_type,
         										const std::vector< std::string >& _name
         										);


  //  PROTECTED MEMBERS:
  // -------------------------
  protected:

    Operator  *derivative;		/**< The derivative of this unary operator. */
    Operator  *derivative2;		/**< The second order derivative of this unary operator. */

    Operator *argument        ;     /**< The argument                         */
    Operator *dargument       ;     /**< The derivative                       */
    double   *argument_result ;     /**< The results for the argument.        */
    double   *dargument_result;     /**< The results for the first derivative */
    int       bufferSize      ;     /**< The size of the buffer               */

    CurvatureType     curvature   ;
    MonotonicityType  monotonicity;
    OperatorName      operatorName;
    std::string       cName       ;

    double (*fcn)(double);
    double (*dfcn)(double);
    double (*ddfcn)(double);
};


CLOSE_NAMESPACE_ACADO



#endif
