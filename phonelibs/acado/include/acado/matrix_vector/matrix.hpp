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
 *    \file include/acado/matrix_vector/matrix.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 *    \date 2008 - 2013
 *    \note The code in the class GenericMatrix is not 100% compatible with the
 *          original code developed by B. Houska and H.J. Ferreau.
 */

#ifndef ACADO_TOOLKIT_MATRIX_HPP
#define ACADO_TOOLKIT_MATRIX_HPP

#include <memory>

#include <acado/matrix_vector/vector.hpp>

BEGIN_NAMESPACE_ACADO

/** A generic matrix class based on Eigen's matrix class. */
template<typename T>
class GenericMatrix : public Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>
{
public:

	/** \internal
	 *  \name Eigen compatibility layer
	 *  @{
	 */

	/** Handy typedef for the base matrix class. */
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> Base;


	/** Constructor from any other Eigen::MatrixBase derived class. */
	template<typename OtherDerived>
	inline GenericMatrix(const Eigen::MatrixBase<OtherDerived>& other) : Base( other ) {}

	/** Constructor from any other Eigen::ReturnByValue derived class. */
	template<typename OtherDerived>
	inline GenericMatrix(const Eigen::ReturnByValue<OtherDerived>& other) : Base( other ) {}

	/** Constructor from any other Eigen::EigenBase derived class. */
	template<typename OtherDerived>
	inline GenericMatrix(const Eigen::EigenBase<OtherDerived>& other) : Base( other ) {}

	/** @}
	 *  \endinternal
	 */

	/** \name Constructors. */
	/** @{ */

	/** Default ctor */
	GenericMatrix() : Base() {}

	/** Ctor with scalar initializtion. */
	GenericMatrix(	const T& _value
					)
		: Base(1, 1)
	{ Base::data()[ 0 ] = _value; }

	/** Ctor that accepts matrix dimensions. */
	GenericMatrix(	unsigned _nRows,
					unsigned _nCols
					)
		: Base(_nRows, _nCols)
	{ Base::setZero(); }

	/** Ctor that accepts matrix dimensions and initialization data in C-like array. */
	GenericMatrix(	unsigned _nRows,
					unsigned _nCols,
					const T* const _values
					)
		: Base(_nRows, _nCols)
	{ std::copy(_values, _values + _nRows * _nCols, Base::data()); }

	/** Ctor that accepts matrix dimensions and initialization data in STL 1D vector. */
	GenericMatrix(	unsigned _nRows,
					unsigned _nCols,
					std::vector< T >& _values)
		: Base(_nRows, _nCols)
	{ std::copy(_values.begin(), _values.end(), Base::data()); }

	/** Ctor that accepts matrix dimensions and initialization data in STL 2D array. */
	GenericMatrix(	unsigned _nRows,
					unsigned _nCols,
					std::vector< std::vector< T > >& _values
					);

	/** @} */

	/** Destructor. */
	virtual ~GenericMatrix()
	{}

	/** Equality operator. */
	bool operator==(const GenericMatrix& arg) const
	{
		if (Base::rows() != arg.rows() || Base::cols() != arg.cols())
			return false;
		return Base::isApprox(arg, EQUALITY_EPS);
	}

	/** Inequality operator. */
	bool operator!=(const GenericMatrix& arg) const
	{
		return (operator==( arg ) == false);
	}

	/** Initialization routine. */
	void init(	unsigned _nRows = 0,
				unsigned _nCols = 0
				)
	{ Base::_set(GenericMatrix<T>(_nRows, _nCols)); }

	/** Set all elements constant. */
	void setAll( const T& _value)
	{ Base::setConstant( _value ); }

	/** Appends rows at the end of the matrix. */
	GenericMatrix& appendRows(	const GenericMatrix& _arg
								);

	/** Appends columns at the end of the matrix. */
	GenericMatrix& appendCols(	const GenericMatrix& _arg
								);

	/** Computes the column-wise sum the DMatrix
	 *
	 *  Example:
	 *  \code
	 *  a   |  b
	 *  c   |  d
	 *  \endcode
	 *
	 *  returns [a+b;c+d]
	 */
	GenericVector< T > sumCol() const;

	/** Computes the row-wise sum the DMatrix
	 *
	 *  Example:
	 *
	 *  \code
	 *  a   |  b
	 *  c   |  d
	 *  \endcode
	 *
	 *  returns [a+c|b+d]
	 */
	GenericVector< T > sumRow() const;

	/** Reshapes a matrix into a column vector. */
	GenericMatrix& makeVector( );

	/** Returns total number of elements of the matrix object. */
	unsigned getDim( ) const
	{ return (Base::rows() * Base::cols()); }

	/** Returns number of rows of the matrix object. */
	unsigned getNumRows( ) const
	{ return Base::rows(); }

	/** Returns number of columns of the matrix object. */
	unsigned getNumCols( ) const
	{ return Base::cols(); }

	/** Returns whether the vector is empty. */
	bool isEmpty( ) const
	{ return Base::rows() == 0 || Base::cols() == 0; }

	/** Returns a given row of the matrix object. */
	GenericVector< T > getRow(	unsigned _idx
								) const
	{
		ASSERT( _idx < Base::rows() );
		return Base::row( _idx ).transpose();
	}

	/** Returns a given column of the matrix object. */
	GenericVector< T > getCol(	unsigned _idx
								) const
	{
		ASSERT( _idx < Base::cols() );
		return Base::col( _idx );
	}

	/** Assigns new values to a given row of the matrix object. */
	GenericMatrix& setRow(	unsigned _idx,
							const GenericVector< T >& _values
							)
	{
		ASSERT(Base::cols() == _values.rows() && _idx < Base::rows());
		Base::row( _idx ) = _values.transpose();

		return *this;
	}

	/** Assigns new values to a given column of the matrix object. */
	GenericMatrix& setCol(	unsigned _idx,
							const GenericVector< T >& _arg
							)
	{
		ASSERT(Base::rows() == _arg.rows() && _idx < Base::cols());
		Base::col( _idx ) = _arg;

		return *this;
	}

	/** Returns given rows of the matrix object. */
	GenericMatrix getRows(	unsigned _start,
							unsigned _end
							) const
	{
		if (_start >= Base::rows() || _end >= Base::rows() || _start > _end)
			return GenericMatrix();

		return Base::block(_start, 0, _end - _start + 1, Base::cols());
	}

	/** Returns given columns of the matrix object. */
	GenericMatrix getCols(	unsigned _start,
							unsigned _end
							) const
	{
		if (_start >= Base::cols() || _end >= Base::cols() || _start > _end)
			return GenericMatrix();

		return Base::block(0, _start, Base::rows(), _end - _start + 1);
	}

	/** Returns a vector containing the diagonal elements of a square matrix. */
	GenericVector< T > getDiag( ) const;

	/** Is the matrix square? */
	bool isSquare() const;

	/** Tests if object is a symmetric matrix. */
	bool isSymmetric( ) const;

	/** Make the matrix symmetric. */
	returnValue symmetrize( );

	/** Tests if object is a positive semi-definite matrix.
	 *	\note This test involves a Cholesky decomposition.
	 */
	bool isPositiveSemiDefinite( ) const;

	/** Tests if object is a (strictly) positive definite matrix.
	 *	\note This test involves a Cholesky decomposition.
	 */
	bool isPositiveDefinite( ) const;

	/** Returns the a matrix whose components are the absolute
	 *  values of the components of this object. */
	GenericMatrix absolute() const;

	/** Returns the a matrix whose components are equal to the components of
	 *  this object, if they are positive or zero, but zero otherwise.
	 */
	GenericMatrix positive() const;

	/** Returns the a matrix whose components are equal to the components of
	 *  this object, if they are negative or zero, but zero otherwise.
	 */
	GenericMatrix negative() const;

	/** Returns maximum element. */
	T getMax( ) const
	{ return Base::maxCoeff(); }

	/** Returns minimum element. */
	T getMin( ) const
	{ return Base::minCoeff(); }

	/** Returns mean value of all elements. */
	T getMean( ) const
	{ return Base::mean(); }

	/** Return a new vector with absolute elements. */
	GenericMatrix< T > getAbsolute() const
	{ return Base::cwiseAbs(); }

	/** Returns Frobenius norm of the matrix. */
	T getNorm( ) const;

	/** Returns trace of the matrix. */
	T getTrace( ) const;

	/** Returns condition number of the square matrix based on SVD.
	 *  \note Works for square matrices, only.
	 */
	T getConditionNumber( ) const;

	/** Prints object to given file. Various settings can
	 *  be specified defining its output format.
	 *
	 *  @param[in] _stream		  Output stream for printing.
	 *  @param[in] _name          Name label to be printed before the numerical values.
	 *  @param[in] _startString   Prefix before printing the numerical values.
	 *  @param[in] _endString     Suffix after printing the numerical values.
	 *  @param[in] _width         Total number of digits per single numerical value.
	 *  @param[in] _precision     Number of decimals per single numerical value.
	 *  @param[in] _colSeparator  Separator between the columns of the numerical values.
	 *  @param[in] _rowSeparator  Separator between the rows of the numerical values.
	 *
	 *  \return SUCCESSFUL_RETURN, \n
	 *          RET_FILE_CAN_NOT_BE_OPENED, \n
	 *          RET_UNKNOWN_BUG
	 */
	virtual returnValue print(	std::ostream& _stream            = std::cout,
								const std::string& _name         = DEFAULT_LABEL,
								const std::string& _startString  = DEFAULT_START_STRING,
								const std::string& _endString    = DEFAULT_END_STRING,
								uint _width                      = DEFAULT_WIDTH,
								uint _precision                  = DEFAULT_PRECISION,
								const std::string& _colSeparator = DEFAULT_COL_SEPARATOR,
								const std::string& _rowSeparator = DEFAULT_ROW_SEPARATOR
								) const;

	/** Prints object to given file. Various settings can
	 *  be specified defining its output format.
	 *
	 *  @param[in] _stream       Output stream for printing.
	 *  @param[in] _name         Name label to be printed before the numerical values.
	 *  @param[in] _printScheme  Print scheme defining the output format of the information.
	 *
	 *  \return SUCCESSFUL_RETURN, \n
	 *          RET_FILE_CAN_NOT_BE_OPENED, \n
	 *          RET_UNKNOWN_BUG
	 */
	virtual returnValue print(	std::ostream& stream,
								const std::string& name,
								PrintScheme printScheme
								) const;

	/** Prints object to file with given name. Various settings can
	 *  be specified defining its output format.
	 *
	 *  @param[in] _filename     Filename for printing.
	 *  @param[in] _name         Name label to be printed before the numerical values.
	 *  @param[in] _startString  Prefix before printing the numerical values.
	 *  @param[in] _endString    Suffix after printing the numerical values.
	 *  @param[in] _width        Total number of digits per single numerical value.
	 *  @param[in] _precision    Number of decimals per single numerical value.
	 *  @param[in] _colSeparator Separator between the columns of the numerical values.
	 *  @param[in] _rowSeparator Separator between the rows of the numerical values.
	 *
	 *  \return SUCCESSFUL_RETURN, \n
	 *          RET_FILE_CAN_NOT_BE_OPENED, \n
	 *          RET_UNKNOWN_BUG
	 */
	virtual returnValue print(	const std::string& _filename,
								const std::string& _name         = DEFAULT_LABEL,
								const std::string& _startString  = DEFAULT_START_STRING,
								const std::string& _endString    = DEFAULT_END_STRING,
								uint _width                      = DEFAULT_WIDTH,
								uint _precision                  = DEFAULT_PRECISION,
								const std::string& _colSeparator = DEFAULT_COL_SEPARATOR,
								const std::string& _rowSeparator = DEFAULT_ROW_SEPARATOR
								) const;

	/** Prints object to given file. Various settings can
	 *  be specified defining its output format.
	 *
	 *  @param[in] _filename    Filename for printing.
	 *  @param[in] _name        Name label to be printed before the numerical values.
	 *  @param[in] _printScheme Print scheme defining the output format of the information.
	 *
	 *  \return SUCCESSFUL_RETURN, \n
	 *          RET_FILE_CAN_NOT_BE_OPENED, \n
	 *          RET_UNKNOWN_BUG
	 */
	virtual returnValue print(	const std::string& _filename,
								const std::string& _name,
								PrintScheme _printScheme
								) const;

	/** Read matrix data from an input stream. */
	virtual returnValue read(	std::istream& _stream
								);

	/** Read data from an input file. */
	virtual returnValue read(	const std::string& _filename
								);
};

/** Prints the matrix into a stream. */
template<typename T>
std::ostream& operator<<(	std::ostream& _stream,
							const GenericMatrix< T >& _arg
							)
{
	if (_arg.print( _stream ) != SUCCESSFUL_RETURN)
		ACADOERRORTEXT(RET_INVALID_ARGUMENTS, "Cannot write to output stream.");

	return _stream;
}

/** Read a matrix from an input stream. */
template<typename T>
std::istream& operator>>(	std::istream& _stream,
							GenericMatrix< T >& _arg
)
{
	if (_arg.read( _stream ) != SUCCESSFUL_RETURN )
		ACADOERRORTEXT(RET_INVALID_ARGUMENTS, "Cannot read from input stream.");

	return _stream;
}

/** Create a square matrix with all T( 1 ) elements. */
template<typename T>
GenericMatrix< T > ones(	unsigned _nRows,
							unsigned _nCols = 1
							)
{ return GenericMatrix< T >(_nRows, _nCols).setOnes(); }

/** Create a square matrix with all T( 0 ) elements. */
template<typename T>
GenericMatrix< T > zeros(	unsigned _nRows,
							unsigned _nCols = 1
							)
{ return GenericMatrix< T >(_nRows, _nCols).setZero(); }

/** Create an identity matrix. */
template<typename T>
GenericMatrix< T > eye(	unsigned _dim
						)
{ return GenericMatrix< T >(_dim, _dim).setIdentity(); }

/** Type definition of the matrix of doubles. */
typedef GenericMatrix< double > DMatrix;
/** Type definition of the matrix of integers. */
typedef GenericMatrix< int > IMatrix;
/** Type definition of the matrix of booleans. */
typedef GenericMatrix< bool > BMatrix;
/** Shared pointer to a matrix of doubles. */
typedef std::shared_ptr< GenericMatrix< double > > DMatrixPtr;

static       DMatrix emptyMatrix;
static const DMatrix emptyConstMatrix;

CLOSE_NAMESPACE_ACADO

/** \internal */
namespace Eigen
{
namespace internal
{
template<typename T>
struct traits< ACADO::GenericMatrix< T > >
	: traits<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor | Eigen::AutoAlign> >
{};

} // namespace internal
} // namespace Eigen
/** \endinternal */

#endif  // ACADO_TOOLKIT_MATRIX_HPP
