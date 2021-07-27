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
 *    \file include/acado/function/function_evaluation_tree.hpp
 *    \authors Boris Houska, Hans Joachim Ferreau, Milan Vukov
 *    \date 2008 - 2013
 */

#ifndef ACADO_TOOLKIT_FUNCTION_EVALUATION_TREE_HPP
#define ACADO_TOOLKIT_FUNCTION_EVALUATION_TREE_HPP

#include <acado/symbolic_expression/expression.hpp>
#include <acado/symbolic_operator/evaluation_template.hpp>
#include <acado/symbolic_operator/symbolic_index_list.hpp>

BEGIN_NAMESPACE_ACADO

/** 
 *	\brief Organizes the evaluation of the function tree.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class FunctionEvaluationTree is designed to organize the evaluation of
 *  tree structured expressions.
 *
 *	\author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 */
class FunctionEvaluationTree
{

//
// PUBLIC MEMBER FUNCTIONS:
//
public:

    /** Default constructor. */
    FunctionEvaluationTree( );

    /** Copy constructor (deep copy). */
    FunctionEvaluationTree( const FunctionEvaluationTree& arg );

    /** Destructor. */
    virtual ~FunctionEvaluationTree( );

    /** Assignment operator (deep copy). */
    FunctionEvaluationTree& operator=( const FunctionEvaluationTree& arg );

    /** Loading Expressions (deep copy). */
    virtual returnValue operator<<( const Expression& arg );


    /** Returns the dimension of the symbolic expression  \n
     *  \return The requested dimension.
     */
    virtual int getDim () const;


    /** Returns the number of intermediate expressions that have  \n
     *  been detected in the symbolic expression.                 \n
     *  \return The requested number of intermediate states.      \n
     */
    virtual int getN   () const;


    /** Returns the number of differential states                 \n
     *  \return The requested number of differential states.      \n
     */
    virtual int getNX    () const;

    /** Returns the number of algebraic states                    \n
     *  \return The requested number of algebraic states.         \n
     */
    virtual int getNXA   () const;

    /** Returns the number of differential states derivatives     \n
     *  \return The requested number of differential state        \n
     *          derivatives.                                      \n
     */
    virtual int getNDX   () const;


    /** Returns the number of controls                            \n
     *  \return The requested number of controls.                 \n
     */
    virtual int getNU   () const;

    /** Returns the number of integer controls                    \n
     *  \return The requested number of integer controls.         \n
     */
    virtual int getNUI  () const;

    /** Returns the number of parameters                          \n
     *  \return The requested number of parameters.               \n
     */
    virtual int getNP   () const;

    /** Returns the number of integer parameters                  \n
     *  \return The requested number of integer parameters.       \n
     */
    virtual int getNPI  () const;

    /** Returns the number of disturbances                        \n
     *  \return The requested number of disturbances.             \n
     */
    virtual int getNW  () const;

    /** Returns the number of time variables                        \n
     *  \return The requested number of time variables.             \n
     */
    virtual int getNT  () const;

    /** Return number of "online data" objects. */
    virtual int getNOD  () const;


    /** Returns the index of the variable with specified type and \n
     *  component.                                                \n
     *  \return The index of the requested variable.              \n
     */
    virtual int index( VariableType variableType_, int index_ ) const;


    /** Returns the scale of a given variable.   \n
     *  \return The requested scale      or      \n
     *          1.0  if index is out of range    \n
     */
    virtual double scale( VariableType variableType_, int index_ ) const;


    /** Returns the variable counter.           \n
     *  \return The number of variables         \n
     */
    virtual int getNumberOfVariables() const;


    /** Returns the symbolic expression of the given component of the function.  \n
     *  \return The symbolic expression         \n
     */
    virtual Operator* getExpression(	uint componentIdx
										) const;


    /** Evaluates the expression
     *  \return SUCCESSFUL_RETURN                  \n
     *          RET_NAN                            \n
     * */
    virtual returnValue evaluate( double *x         /**< the input variable x */,
                                  double *result    /**< the result           */  );



    /** Evaluates the expression */
    template <typename T> returnValue evaluate( Tmatrix<T> *x, Tmatrix<T> *result );
	
	
	
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



    /** Returns the derivative of the expression with respect     \n
     *  to the variable var(index).                               \n
     *  \return The symbolic expression for the derivative.       \n
     *
     */
     FunctionEvaluationTree* differentiate( int index /**< diff. index    */ );



    /** Substitutes var(index) with the double sub.               \n
     *  \return The substituted expression.                       \n
     *
     */
     virtual FunctionEvaluationTree substitute( VariableType variableType_,
                                             int index_      /**< subst. index    */,
                                             double sub_     /**< the substitution*/);


    /** Checks whether the expression is zero or one              \n
     *  \return NE_ZERO                                           \n
     *          NE_ONE                                            \n
     *          NE_NEITHER_ONE_NOR_ZERO                           \n
     *
     */
     virtual NeutralElement isOneOrZero();



    /** Checks whether the symbolic expression is depending on    \n
     *  a specified variable.                                     \n
     *  \return BT_FALSE if no dependence is detected             \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     virtual BooleanType isDependingOn( const Expression     &variable );


    /** Checks whether the symbolic expression is linear in       \n
     *  a specified variable.                                     \n
     *  \return BT_FALSE if no linearity is                       \n
     *                detected                                    \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     virtual BooleanType isLinearIn( const Expression     &variable );


    /** Checks whether the expression is polynomial in            \n
     *  a variable.                                               \n
     *  \return BT_FALSE if the expression is not  polynomial     \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     virtual BooleanType isPolynomialIn( const Expression     &variable );


    /** Checks whether the expression is rational in              \n
     *  the variable var(index)                                   \n
     *  \return BT_FALSE if the expression is not rational        \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     virtual BooleanType isRationalIn( const Expression     &variable );


    /** Returns the monotonicity of the expression.               \n
     *  \return MT_NONDECREASING                                  \n
     *          MT_NONINCREASING                                  \n
     *          MT_NONMONOTONIC                                   \n
     *
     */
     virtual MonotonicityType getMonotonicity( );



    /** Returns the curvature of the expression                   \n
     *  \return CT_CONSTANT                                       \n
     *          CT_AFFINE                                         \n
     *          CT_CONVEX                                         \n
     *          CT_CONCAVE                                        \n
     *
     */
     virtual CurvatureType getCurvature( );



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


    /** Prints the expression as C-code into a file.
     *                                                                                    \n
     *  \param file       The file to which the expression should be printed.             \n
     *  \param fcnName    The name of the generated function (default: "ACADOfcn").       \n
	 *  \param realString                                                                 \n
     *                                                                                    \n
     *  \return SUCCESFUL_RETURN                                                          \n
     */
     returnValue C_print(	std::ostream& stream = std::cout,
							const char *fcnName = "ACADOfcn",
							const char *realString = "double"
							) const;

     returnValue exportForwardDeclarations(	std::ostream& stream = std::cout,
											const char *fcnName = "ACADOfcn",
											const char *realString = "double"
											) const;

     returnValue exportCode(	std::ostream& stream = std::cout,
								const char *fcnName = "ACADOfcn",
								const char *realString = "double",
								uint       _numX = 0,
								uint	   _numXA = 0,
								uint       _numU = 0,
								uint       _numP = 0,
								uint       _numDX = 0,
								uint       _numOD = 0,
								bool       allocateMemory = true,
								bool       staticMemory   = false
								) const;

     /** Clears the buffer and resets the buffer size \n
      *  to 1.                                        \n
      *  \return SUCCESFUL_RETURN                     \n
      */
     virtual returnValue clearBuffer();


     /** Make the symbolic expression implicit. This functionality  \n
      *  makes only sense for Differential Equation and should in   \n
      *  general not be used for anything else.  (Although it is    \n
      *  public here.)                                              \n
      */
     virtual returnValue makeImplicit();
     virtual returnValue makeImplicit( int dim_ );


     /** Returns whether the function is symbolic or not. If BT_TRUE \n
      *  is returned, automatic differentiation will be used by      \n
      *  default.
      */
     virtual BooleanType isSymbolic() const;


     /** Defines scalings for the variables. */
     virtual returnValue setScale( double *scale_ );

     virtual returnValue getExpression( Expression& expression ) const;

     returnValue setGlobalExportVariableName(const std::string& _name);

     std::string getGlobalExportVariableName() const;

     unsigned getGlobalExportVariableSize() const;

     //
     // DATA MEMBERS:
     //
protected:

     Operator           **f        ;   /**< The right-hand side expressions */
     Operator           **sub      ;   /**< The intermediate expressions    */
     int                 *lhs_comp ;   /**< The components of the intermediate states */
     SymbolicIndexList   *indexList;   /**< an SymbolicIndexList            */
     int                  dim      ;   /**< The dimension of the function.  */
     int                  n        ;   /**< The number of Intermediate expressions */

     Expression           safeCopy ;

     /** Name of the variable that holds intermediate expressions. */
     std::string		globalExportVariableName;
};


template <typename T> returnValue FunctionEvaluationTree::evaluate( Tmatrix<T> *x,
																	Tmatrix<T> *result ){

    int run1;

	EvaluationTemplate<T> y(x);
	
	for( run1 = 0; run1 < n; run1++ ){
		sub[run1]->evaluate(&y);
		x->operator()(indexList->index(VT_INTERMEDIATE_STATE,lhs_comp[run1])) = y.res;
	}
	
	for( run1 = 0; run1 < dim; run1++ ){
		f[run1]->evaluate(&y);
		result->operator()(run1) = y.res;
	}

    return SUCCESSFUL_RETURN;
}



CLOSE_NAMESPACE_ACADO



#include <acado/function/function_evaluation_tree.ipp>


#endif  // ACADO_TOOLKIT_FUNCTION_HPP

// end of file.
