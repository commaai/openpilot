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
 *    \file include/acado/function/function_.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_FUNCTION__HPP
#define ACADO_TOOLKIT_FUNCTION__HPP


#include <acado/function/function_evaluation_tree.hpp>


BEGIN_NAMESPACE_ACADO


class EvaluationPoint;
template <typename T> class TevaluationPoint;


/** 
 *	\brief Allows to setup and evaluate a general function based on SymbolicExpressions.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class Function allows to setup and evaluate general functions
 *	based on SymbolicExpressions.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */

class Function{

//
// PUBLIC MEMBER FUNCTIONS:
//

public:

    /** Default constructor. */
    Function();

    /** Copy constructor (deep copy). */
    Function( const Function& rhs );

    /** Destructor. */
    virtual ~Function( );

    /** Assignment operator (deep copy). */
    Function& operator=( const Function& rhs );


    /** Loading Expressions (deep copy). */
    Function& operator<<( const Expression& arg );

    /** Loading Expressions (deep copy). */
    Function& operator<<( const double &arg );

    /** Loading Symbolic DVector (deep copy). */
    Function& operator<<( const DVector& arg );

    /** Loading Symbolic DMatrix (deep copy). */
    Function& operator<<( const DMatrix& arg );

	Function operator()(	uint idx
							) const;


    returnValue getExpression( Expression& expression ) const;


	returnValue reset( );


    /** Returns the dimension of the function  \n
     *  \return The requested dimension.
     */
    inline int getDim () const;


    /** Returns the number of intermediate expressions that have  \n
     *  been detected in the function.                            \n
     *  \return The requested number of intermediate states.      \n
     */
    inline int getN   () const;

    /** Returns the number of differential states                 \n
     *  \return The requested number of differential states.      \n
     */
    int getN    (VariableType &variableType_) const;

    /** Returns the number of differential states                 \n
     *  \return The requested number of differential states.      \n
     */
    int getNX    () const;

    /** Returns the number of algebraic states                    \n
     *  \return The requested number of algebraic states.         \n
     */
    int getNXA   () const;

    /** Returns the number of differential states derivatives     \n
     *  \return The requested number of differential state        \n
     *          derivatives.                                      \n
     */
    int getNDX   () const;

    /** Returns the number of controls                            \n
     *  \return The requested number of controls.                 \n
     */
    int getNU   () const;

    /** Returns the number of integer controls                    \n
     *  \return The requested number of integer controls.         \n
     */
    int getNUI  () const;

    /** Returns the number of parameters                          \n
     *  \return The requested number of parameters.               \n
     */
    int getNP   () const;

    /** Returns the number of integer parameters                  \n
     *  \return The requested number of integer parameters.       \n
     */
    int getNPI  () const;

    /** Returns the number of disturbances                        \n
     *  \return The requested number of disturbances.             \n
     */
    int getNW  () const;

    /** Returns the number of time variables                        \n
     *  \return The requested number of time variables.             \n
     */
    int getNT  () const;

    /** Return number of "online data" objects. */
    int getNOD() const;


    /** Returns the index of the variable with specified type and \n
     *  component.                                                \n
     *  \return The index of the requested variable.              \n
     */
    int index( VariableType variableType_, int index_ ) const;


    /** Returns the scale of a given variable.   \n
     *  \return The requested scale      or      \n
     *          1.0  if index is out of range    \n
     */
    double scale( VariableType variableType_, int index_ ) const;


    /** Returns the variable counter.           \n
     *  \return The number of variables         \n
     */
    int getNumberOfVariables() const;

    /** Returns the symbolic expression of the given component of the function.  \n
     *  \return The symbolic expression         \n
     */
    Operator* getExpression(	uint componentIdx
								) const;


    /** Evaluates the function.                \n
     *                                         \n
     *  \param x       the evaluation point    \n
     *  \param number  the storage position    \n
     *                                         \n
     *  \return The result of the evaluation.  \n
     */
    DVector evaluate( const EvaluationPoint &x         ,
                     const int             &number = 0  );


    /** Redundant evaluation routine which is equivalent to \n
     *  to the evaluate routine above.                      \n
     *                                                      \n
     *  \param x       the evaluation point                 \n
     *  \param number  the storage position                 \n
     *                                                      \n
     *  \return The result of the evaluation.               \n
     */
    inline DVector operator()( const EvaluationPoint &x         ,
                              const int             &number = 0  );



	/** Evaluates the function at a templated  \n
	 *  evaluation point.                      \n
     *                                         \n
     *  \param x       the evaluation point    \n
     *                                         \n
     *  \return The result of the evaluation.  \n
     */
    template <typename T> Tmatrix<T> evaluate( const TevaluationPoint<T> &x );
	
	
	
    /** Evaluates the function and stores the intermediate        \n
     *  results in a buffer (needed for automatic differentiation \n
     *  in backward mode)                                         \n
     *  \return SUCCESFUL_RETURN                   \n
     *          RET_NAN                            \n
     * */
    returnValue evaluate( int     number    /**< storage position     */,
                          double *x         /**< the input variable x */,
                          double *_result    /**< the result           */  );



    /** Substitutes var(index) with the double sub.               \n
     *  \return The substituted expression.                       \n
     *
     */
     returnValue substitute( VariableType variableType_,
                             int index_      /**< subst. index    */,
                             double sub_     /**< the substitution*/ );



    /** Checks whether the function is zero or one                \n
     *  \return NE_ZERO                                           \n
     *          NE_ONE                                            \n
     *          NE_NEITHER_ONE_NOR_ZERO                           \n
     *
     */
     NeutralElement isOneOrZero();



    /** Checks whether the function is depending on               \n
     *  var(index)                                                \n
     *  \return BT_FALSE if no dependence is detected             \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isDependingOn( const Expression     &variable );



    /** Checks whether the function is linear in                  \n
     *  (or not depending on)  var(index)                         \n
     *  \return BT_FALSE if no linearity is                       \n
     *                detected                                    \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isLinearIn( const Expression     &variable );



    /** Checks whether the function is polynomial in              \n
     *  the variable var(index)                                   \n
     *  \return BT_FALSE if the expression is not  polynomial     \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isPolynomialIn( const Expression     &variable );




    /** Checks whether the function is rational in                \n
     *  the variable var(index)                                   \n
     *  \return BT_FALSE if the expression is not rational        \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isRationalIn( const Expression     &variable );



    /** Checks whether the function is nondecreasing.             \n
     *  \return BT_FALSE if the expression is not nondecreasing   \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isNondecreasing();


    /** Checks whether the function is nonincreasing.             \n
     *  \return BT_FALSE if the expression is not nonincreasing   \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isNonincreasing();

     /** Checks whether function is a constant. */
     BooleanType isConstant();

    /** Checks whether the function is affine.                    \n
     *  \return BT_FALSE if the expression is not affine          \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isAffine();


    /** Checks whether the function is convex.                    \n
     *  \return BT_FALSE if the expression is not (DCP-) convex   \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isConvex();


    /** Checks whether the function is concave.                   \n
     *  \return BT_FALSE if the expression is not (DCP-) concave  \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     BooleanType isConcave();



    /** Automatic Differentiation in forward mode. \n
     *                                             \n
     *  \param x       the evaluation point        \n
     *  \param number  the storage position        \n
     *                                             \n
     *  \return The result of the evaluation.      \n
     */
    DVector AD_forward( const EvaluationPoint &x         ,
                       const int             &number = 0  );



    /** Automatic Differentiation in forward mode.                \n
     *  This function uses the intermediate                       \n
     *  results from a buffer                                     \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     returnValue AD_forward(  int     number  /**< storage position */,
                              double *seed    /**< the seed         */,
                              double *df      /**< the derivative of
                                                   the expression   */  );



    /** Automatic Differentiation in backward mode.                \n
     *                                                             \n
     *  \param seed    the backward seed                           \n
     *  \param df      the directional derivative (output)         \n
     *  \param number  the storage position                        \n
     *                                                             \n
     *  \return the result for the derivative.                     \n
     */
     returnValue AD_backward( const    DVector &seed      ,
                              EvaluationPoint &df        ,
                              const    int    &number = 0  );


    /** Automatic Differentiation in backward mode based on       \n
     *  buffered values                                           \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     returnValue AD_backward( int    number /**< the buffer
                                                 position         */,
                              double *seed  /**< the seed         */,
                              double  *df   /**< the derivative of
                                                 the expression   */  );



    /** Automatic Differentiation in forward mode for             \n
     *  2nd derivatives.                                          \n
     *  This function uses intermediate                           \n
     *  results from a buffer.                                    \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     returnValue AD_forward2( int    number  /**< the buffer
                                                  position         */,
                              double *seed1  /**< the seed         */,
                              double *seed2  /**< the seed for the
                                                  first derivative */,
                              double *df     /**< the derivative of
                                                  the expression   */,
                              double *ddf    /**< the 2nd derivative
                                                  of the expression*/);



    /** Automatic Differentiation in backward mode for 2nd order  \n
     *  derivatives based on buffered values.                     \n
     *  IMPORTANT REMARK: run AD_forward first to define          \n
     *                    the point x and to compute f and df.    \n
     *  \return SUCCESFUL_RETURN                                  \n
     *          RET_NAN                                           \n
     */
     returnValue AD_backward2( int    number  /**< the buffer
                                                   position           */,
                               double *seed1  /**< the seed1          */,
                               double *seed2  /**< the seed2          */,
                               double    *df  /**< the 1st derivative
                                                   of the expression  */,
                               double   *ddf  /**< the 2nd derivative
                                                   of the expression  */   );


    /** \brief calculate the jacobian of an evaluated function 
    *
    * Calculates the matrix diff(fun(x,u,v,p,q,w),x)
    *
    * \param x will be assigned jacobian(fun,differential states)
    */
    returnValue jacobian(DMatrix &x);
    //returnValue jacobian(DMatrix &x,DMatrix &p=emptyMatrix,DMatrix &u=emptyMatrix,DMatrix &w=emptyMatrix);

    /** Prints the function into a stream. */
    friend std::ostream& operator<<( std::ostream& stream, const Function &arg);

    /** Prints the function in form of plain C-code into a file. The integer             \n
     *  "precision" must be in [1,16] and defines the number of internal decimal places  \n
     *  which occur in "double" - valued parts of the expression tree.                   \n
     *                                                                                   \n
     *  \param file      The file to which the expression should be printed.             \n
     *  \param fcnName   The name of the generated function (default: "ACADOfcn").       \n
     *  \param precision The number of internal dec. places to be printed (default: 16). \n
     *                                                                                   \n
     *  \return SUCCESFUL_RETURN                                                         \n
     */
     returnValue print(	std::ostream& stream,
						const char *fcnName = "ACADOfcn",
						const char *realString = "double"
						) const;

     returnValue exportForwardDeclarations(	std::ostream& stream,
											const char *fcnName = "ACADOfcn",
											const char *realString = "double"
											) const;

     returnValue exportCode(	std::ostream& stream,
								const char *fcnName = "ACADOfcn",
								const char *realString = "double",
								uint        _numX = 0,     
								uint        _numXA = 0,
								uint		_numU = 0,
								uint		_numP = 0,
								uint		_numDX = 0,
								uint		_numOD = 0,
								bool       allocateMemory = true,
								bool       staticMemory   = false
								) const;

     /** Clears the buffer and resets the buffer size \n
      *  to 1.                                        \n
      *  \return SUCCESFUL_RETURN                     \n
      */
     returnValue clearBuffer();



     /** Defines a scale for the case that a C-function is used \n
      *  \return SUCCESSFUL_RETURN
      */
     returnValue setScale( double *scale_ );


     /** Returns whether the function is symbolic or not. If BT_TRUE \n
      *  is returned, automatic differentiation will be used by      \n
      *  default.
      */
     inline BooleanType isSymbolic() const;


     inline BooleanType ADisSupported() const;


     inline returnValue setMemoryOffset( int memoryOffset_ );

     /** Set name of the variable that holds intermediate values. */
     returnValue setGlobalExportVariableName(const std::string& var);

     /** Get name of the variable that holds intermediate values. */
     std::string getGlobalExportVariableName( ) const;

     /** Get size of the variable that holds intermediate values. */
     unsigned getGlobalExportVariableSize( ) const;

// PROTECTED MEMBERS:
// ------------------

protected:

    FunctionEvaluationTree evaluationTree;
    int                    memoryOffset  ;
	
	double* result;
};


template <typename T> Tmatrix<T> Function::evaluate( const TevaluationPoint<T> &x ){

	Tmatrix<T> Tresult(getDim());
	evaluationTree.evaluate( x.getEvaluationPointer(), &Tresult );
	return Tresult;
}


CLOSE_NAMESPACE_ACADO



#include <acado/function/function.ipp>


#endif  // ACADO_TOOLKIT_FUNCTION__HPP

/*
 *   end of file
 */
