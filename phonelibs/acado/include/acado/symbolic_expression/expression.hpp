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
 *    \file include/acado/symbolic_expression/expression.hpp
 *    \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 *
 */


#ifndef ACADO_TOOLKIT_EXPRESSION_HPP
#define ACADO_TOOLKIT_EXPRESSION_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>

BEGIN_NAMESPACE_ACADO

class Operator;

/**
 *  \brief Base class for all variables within the symbolic expressions family.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class Expression serves as a base class for all
 *  symbolic variables within the symbolic expressions family.
 *  Moreover, the Expression class defines all kind of matrix
 *  and vector operations on a symbolic level.
 *
 *  \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 */
class Expression
{
	friend class COperator;
	friend class CFunction;
	friend class FunctionEvaluationTree;

public:
	/** Default constructor. */
	Expression( );

	/** Casting constructor. */
	Expression( const Operator &tree_ );

	/** Casting constructor. */
	explicit Expression(	const std::string& name_
							);

	/** Constructor which takes the arguments. */
	explicit Expression(	const std::string&  name_,							/**< the name                */
							uint          		nRows_,							/**< number of rows          */
							uint          		nCols_,							/**< number of columns       */
							VariableType  		variableType_  = VT_UNKNOWN,	/**< the variable type       */
							uint          		globalTypeID   = 0				/**< the global type ID      */
							);

	/** Constructor which takes the arguments. */
	explicit Expression(	int          nRows_                     ,  /**< number of rows          */
							int          nCols_         = 1         ,  /**< number of columns       */
							VariableType variableType_  = VT_UNKNOWN,  /**< the variable type       */
							int          globalTypeID   = 0            /**< the global type ID      */
							);

	/** Constructor which takes the arguments. */
	explicit Expression(	uint         nRows_                     ,  /**< number of rows          */
							uint         nCols_         = 1         ,  /**< number of columns       */
							VariableType variableType_  = VT_UNKNOWN,  /**< the variable type       */
							uint         globalTypeID   = 0            /**< the global type ID      */
							);

	/** Copy constructor (deep copy). */
	Expression( const double& rhs );
	Expression( const DVector      & rhs );
	Expression( const DMatrix      & rhs );
	Expression( const Expression  & rhs );

	/** Destructor. */
	virtual ~Expression( );

	/** Function for cloning. */
	virtual Expression* clone() const
	{ return new Expression( *this ); }

	/** Assignment Operator. */
	Expression& operator=( const Expression& arg );

	/** An operator for adding elements. */
	Expression& operator<<( const Expression  & arg );

	/** Appends an Expression matrix (n x m)
	with an argument matrix (s x m) in the row direction,
	such that the result is ( (n + s) x m).
	
	As a special case, when applied on an empty Expression,
	the Expression will be assigned the argument. */
	Expression&  appendRows(const Expression& arg);
	
	/** Appends an Expression matrix (n x m)
	with an argument matrix (n x s) in the column direction,
	such that the result is ( n x (m + s) ).
	
	As a special case, when applied on an empty Expression,
	the Expression will be assigned the argument. */
	Expression&  appendCols(const Expression& arg);

	Expression operator()( uint idx                 ) const;
	Expression operator()( uint rowIdx, uint colIdx ) const;

	Operator& operator()( uint idx                 );
	Operator& operator()( uint rowIdx, uint colIdx );

	friend Expression operator+( const Expression  & arg1, const Expression  & arg2 );
	friend Expression operator-( const Expression  & arg1, const Expression  & arg2 );
	Expression operator-( ) const;
	friend Expression operator*( const Expression  & arg1, const Expression  & arg2 );
	friend Expression operator/( const Expression  & arg1, const Expression  & arg2 );

	Expression& operator+=(const Expression &arg);
	Expression& operator-=(const Expression &arg);
	Expression& operator*=(const Expression &arg);
	Expression& operator/=(const Expression &arg);

	std::ostream& print( std::ostream &stream ) const;
	friend std::ostream& operator<<( std::ostream& stream, const Expression &arg );

	/** Returns the symbolic inverse of a matrix (only for square matrices) */
	Expression getInverse( ) const;
	
	Expression getRow( const uint& rowIdx ) const;
	Expression getRows( const uint& rowIdx1, const uint& rowIdx2 ) const;
	Expression getCol( const uint& colIdx ) const;
	Expression getCols( const uint& colIdx1, const uint& colIdx2 ) const;
	Expression getSubMatrix( const uint& rowIdx1, const uint& rowIdx2, const uint& colIdx1, const uint& colIdx2 ) const;
	
	/** When operated on an n x 1 Expression, returns an m x n DMatrix.
	* The element (i,j) of this matrix is zero when this(i) does not depend on arg(j)
	* \param arg m x 1 Expression
	*/
	DMatrix getDependencyPattern( const Expression& arg ) const;

	DMatrix getSparsityPattern() const;

	Expression getSin    ( ) const;
	Expression getCos    ( ) const;
	Expression getTan    ( ) const;
	Expression getAsin   ( ) const;
	Expression getAcos   ( ) const;
	Expression getAtan   ( ) const;
	Expression getExp    ( ) const;
	Expression getSqrt   ( ) const;
	Expression getLn     ( ) const;

	Expression getPow   ( const Expression &arg ) const;
	Expression getPowInt( const int        &arg ) const;

	Expression getSumSquare    ( ) const;
	Expression getLogSumExp    ( ) const;
	Expression getEuclideanNorm( ) const;
	Expression getEntropy      ( ) const;

	Expression getDot          ( ) const;
	Expression getNext         ( ) const;

	Expression ADforward  ( const Expression &arg ) const;
	Expression ADforward  ( const VariableType &varType_, const int *arg, int nV ) const;
	Expression ADbackward ( const Expression &arg ) const;
	
	/** Second order symmetric AD routine returning \n
	 *  S^T*(l^T*f'')*S  with f'' being the second  \n
	 * order derivative of the current expression.  \n
	 * The matrix S and the vector l can be         \n
	 * interpreted as forward/backward seeds,        \n
	 * respectively. Optionally, this routine also  \n
	 * returns expressions for the first order      \n
	 * order terms f'*S  and  l^T*f' computed by    \n
	 * first order forward and first order backward \n
	 * automatic differentiation, respectively.     \n
	 * Caution: this routine is tailored for        \n
	 * full Hessian computation exploiting symmetry.\n
	 * If only single elements of the Hessian are   \n
	 * needed, forward-over-adjoint or ajoint-over- \n
	 * forward differentiation may be more          \n
	 * efficient.                                   \n
	 */
	Expression ADsymmetric(	const Expression &arg, /** argument      */
				 	 	 	const Expression &S  , /** forward seed  */
				 	 	 	const Expression &l  , /** backward seed */
				 	 	 	Expression *dfS = 0,    /** first order forward  result */
				 	 	 	Expression *ldf = 0    /** first order backward result */
							) const;
	
	/** Second order symmetric AD routine returning \n
	 *  l^T*f''  with f'' being the second          \n
	 * order derivative of the current expression.  \n
	 * The he vector l can be                       \n
	 * interpreted as backward seed,                \n
	 * respectively. Optionally, this routine also  \n
	 * returns expressions for the first order      \n
	 * order terms f'*S  and  l^T*f' computed by    \n
	 * first order forward and first order backward \n
	 * automatic differentiation, respectively.     \n
	 * Caution: this routine is tailored for        \n
	 * full Hessian computation exploiting symmetry.\n
	 * If only single elements of the Hessian are   \n
	 * needed, forward-over-adjoint or ajoint-over- \n
	 * forward differentiation may be more          \n
	 * efficient.                                   \n
	 */
	Expression ADsymmetric( const Expression &arg, /** argument      */
							const Expression &l  , /** backward seed */
							Expression *dfS = 0,    /** first order forward  result */
							Expression *ldf = 0    /** first order backward result */
							) const;
	
	Expression getODEexpansion( const int &order, const int *arg ) const;

	Expression ADforward ( const Expression &arg, const Expression &seed ) const;
	Expression ADforward ( const VariableType &varType_, const int *arg, const Expression &seed ) const;
	Expression ADforward ( const VariableType *varType_, const int *arg, const Expression &seed ) const;
	Expression ADbackward( const Expression &arg, const Expression &seed ) const;

	/** Returns the transpose of this expression.
	 *  \return The transposed expression. */
	Expression transpose( ) const;

	/** Returns dimension of vector space.
	 *  \return Dimension of vector space. */
	inline uint getDim( ) const;

	/** Returns the number of rows.
	 *  \return The number of rows. */
	inline uint getNumRows( ) const;

	/** Returns the number of columns.
	 *  \return The number of columns. */
	inline uint getNumCols( ) const;

	/** Returns the global type idea of the idx-component.
	 *  \return The global type ID. */
	inline uint getComponent( const unsigned int idx ) const;

	/** Returns the number of columns.
	 *  \return The number of columns. */
	inline BooleanType isVariable( ) const;

	/** Returns a clone of the operator with index idx.
	 *  \return A clone of the requested operator. */
	Operator* getOperatorClone( uint idx ) const;

	/** Returns the variable type
	 *  \return The the variable type. */
	inline VariableType getVariableType( ) const;

	BooleanType isDependingOn( VariableType type ) const;
	BooleanType isDependingOn(const  Expression &e ) const;

	/** Substitutes a given variable with an expression. */
	returnValue substitute( int idx, const Expression &arg ) const;

protected:

	Expression add(const Expression& arg) const;
	Expression sub(const Expression& arg) const;
	Expression mul(const Expression& arg) const;
	Expression div(const Expression& arg) const;

	/** Generic constructor (protected, only for internal use). */
	void construct( VariableType  variableType_,  /**< The variable type.     */
					uint          globalTypeID_,  /**< the global type ID     */
					uint                 nRows_,  /**< The number of rows.    */
					uint                 nCols_,  /**< The number of columns. */
					const std::string&   name_    /**< The name               */ );

	/** Generic copy routine (protected, only for internal use). */
	void copy( const Expression &rhs );

	/** Generic destructor (protected, only for internal use). */
	void deleteAll( );

	/** Generic copy routine (protected, only for internal use).
	 */
	Expression& assignmentSetup( const Expression &arg );

	/** Internal product routine (protected, only for internal use). */
	Operator* product( const Operator *a, const Operator *b ) const;

	Operator**         element     ;   /**< Element of vector space.   */
	uint               dim         ;   /**< DVector space dimension.    */
	uint               nRows, nCols;   /**< DMatrix dimension.          */
	VariableType       variableType;   /**< Variable type.             */
	uint               component   ;   /**< The expression component   */
	std::string             name   ;   /**< The name of the expression */
};

CLOSE_NAMESPACE_ACADO

#include <acado/symbolic_expression/expression.ipp>

BEGIN_NAMESPACE_ACADO

/** A helper class implementing the CRTP design pattern.
 *
 *  This class gives object counting and clone capability to a derived
 *  class via static polymorphism.
 *
 *  \tparam Derived      The derived class.
 *  \tparam Type         The expression type. \sa VariableType
 *  \tparam AllowCounter Allow object instance counting.
 *
 *  \note Unfortunately the derived classes have to implement all necessary
 *        ctors. In C++11, this can be done in a much simpler way. One only
 *        needs to say: using Base::Base.
 *
 */
template<class Derived, VariableType Type, bool AllowCounter = true>
class ExpressionType : public Expression
{
public:

	/** Default constructor. */
	ExpressionType()
		: Expression("", 1, 1, Type, AllowCounter ? count : 0)
	{
		if (AllowCounter == true)
			count++;
	}

	/** The constructor with arguments. */
	ExpressionType(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Expression(_name, _nRows, _nCols, Type, AllowCounter ? count : 0)
	{
		if (AllowCounter == true)
			count += _nRows * _nCols;
	}

	/** The constructor from an expression. */
	ExpressionType(const Expression& _expression, unsigned _componentIdx = 0)
		: Expression( _expression )
	{
		variableType = Type;
		component += _componentIdx;
		if (AllowCounter == true)
			count++;
	}

	/** The constructor from a scalar number. */
	ExpressionType( const double& _arg )
		: Expression( _arg )
	{}

	/** The constructor from a vector. */
	ExpressionType( const DVector& _arg )
		: Expression( _arg )
	{}

	/** The constructor from a matrix. */
	ExpressionType( const DMatrix& _arg )
		: Expression( _arg )
	{}

	/** The constructor from an operator. */
	ExpressionType( const Operator& _arg )
		: Expression( _arg )
	{}

	/** Destructor. */
	virtual ~ExpressionType() {}

	/** Function for cloning. */
	virtual Expression* clone() const
	{ return new Derived( static_cast< Derived const& >( *this ) ); }

	/** A function for resetting of the istance counter. */
	returnValue clearStaticCounters()
	{ count = 0; return SUCCESSFUL_RETURN; }

private:
	static unsigned count;
};

template<class Derived, VariableType Type, bool AllowCounter>
unsigned ExpressionType<Derived, Type, AllowCounter>::count( 0 );

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPRESSION_HPP
