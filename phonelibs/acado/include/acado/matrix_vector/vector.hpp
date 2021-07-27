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
 *    \file include/acado/matrix_vector/vector.hpp
 *    \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 *    \date 2008 - 2013
 *    \note The code in the class GenericVector is compatible with the original
 *          code developed by B. Houska and H.J. Ferreau.
 */

#ifndef ACADO_TOOLKIT_VECTOR_HPP
#define ACADO_TOOLKIT_VECTOR_HPP

#include <complex>
#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector_tools.hpp>

BEGIN_NAMESPACE_ACADO

/** Defines flags for different vector norms. */
enum VectorNorm
{
    VN_L1,
    VN_L2,
    VN_LINF
};

/** A generic vector class based on Eigen's matrix class. */
template<typename T>
class GenericVector : public Eigen::Matrix<T, Eigen::Dynamic, 1>
{
public:

	/** \internal
	 *  \name Eigen compatibility layer
	 *  @{
	 */

	/** Handy typedef for the base vector class. */
	typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Base;

	/** Constructor from any other Eigen::MatrixBase derived class. */
	template<typename OtherDerived>
	inline GenericVector(const Eigen::MatrixBase<OtherDerived>& other) : Base( other ) {}

	/** Constructor from any other Eigen::ReturnByValue derived class. */
	template<typename OtherDerived>
	inline GenericVector(const Eigen::ReturnByValue<OtherDerived>& other) : Base( other ) {}

	/** Constructor from any other Eigen::EigenBase derived class. */
	template<typename OtherDerived>
	inline GenericVector(const Eigen::EigenBase<OtherDerived>& other) : Base( other ) {}

	/** @}
	 *  \endinternal
	 */

	/** \name Constructors. */
	/** @{ */

	/** Default ctor */
	GenericVector() : Base() {}

	/** Ctor which accepts size of the vector. */
	GenericVector(	unsigned _dim ) : Base( _dim ) { Base::setZero(); }

	/** Ctor with an initializing C-like array. */
	GenericVector(	unsigned _dim,
					const T* const _values
					)
		: Base( _dim )
	{ std::copy(_values, _values + _dim, Base::data()); }

	/** Ctor with an STL vector. */
	GenericVector(	std::vector< T > _values
					)
		: Base( _values.size() )
	{ std::copy(_values.begin(), _values.end(), Base::data()); }

	/** @} */

	/** Destructor. */
	virtual ~GenericVector()
	{}

	/** Equality operator. */
	bool operator==(const GenericVector& _arg) const
	{
		if (Base::rows() != _arg.rows())
			return false;
		return Base::isApprox(_arg, EQUALITY_EPS);
	}

	/** Inequality operator. */
	bool operator!=(const GenericVector& _arg) const
	{
		return (operator==( _arg ) == false);
	}

	bool operator<=(const GenericVector& _arg) const
	{
		if (Base::rows() != _arg.rows())
			return false;
		for (unsigned el = 0; el < Base::rows(); ++el)
			if (Base::data()[ el ] > _arg.data()[ el ])
				return false;
		return true;
	}

	bool operator>=(const GenericVector& _arg) const
	{
		if (Base::rows() != _arg.rows())
			return false;
		for (unsigned el = 0; el < Base::rows(); ++el)
			if (Base::data()[ el ] < _arg.data()[ el ])
				return false;
		return true;
	}

	bool operator>(const GenericVector& _arg) const
	{
		return operator<=( _arg ) == false;
	}

	bool operator<(const GenericVector& _arg) const
	{
		return operator>=( _arg ) == false;
	}

	/** Initialization routine. */
	void init(	unsigned _dim = 0
				)
	{ Base::_set(GenericVector< T >( _dim )); }

	/** Set all elements constant. */
	void setAll( const T& _value)
	{ Base::setConstant( _value ); }

	/** Append elements to the vector. */
	GenericVector& append(	const GenericVector& _arg
							);

	/** Sets vector to the _idx-th unit vector. */
	GenericVector& setUnitVector(	unsigned _idx
									);

	/** Returns dimension of vector space. */
	unsigned getDim( ) const
	{ return Base::rows(); }

	/** Returns whether the vector is empty. */
	bool isEmpty( ) const
	{ return Base::rows() == 0; }

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
	GenericVector getAbsolute() const
	{ return Base::cwiseAbs(); }

	/** Returns specified norm interpreted as a vector.
	 *
	 *  \param norm   the type of norm to be computed.
	 */
	T getNorm(	VectorNorm _norm
				) const;

	/** Returns specified norm interpreted as a vector (with scaling).
	 *
	 *  \param norm   the type of norm to be computed.
	 *  \param scale  the element-wise scale.
	 */
	T getNorm(	VectorNorm _norm,
				const GenericVector& _scale
				) const;

	/** Prints object to given file. Various settings can
	 *	be specified defining its output format.
	 *
	 *	@param[in] stream			Output stream for printing.
	 *	@param[in] name				Name label to be printed before the numerical values.
	 *	@param[in] startString		Prefix before printing the numerical values.
	 *	@param[in] endString		Suffix after printing the numerical values.
	 *	@param[in] width			Total number of digits per single numerical value.
	 *	@param[in] precision		Number of decimals per single numerical value.
	 *	@param[in] colSeparator		Separator between the columns of the numerical values.
	 *	@param[in] rowSeparator		Separator between the rows of the numerical values.
	 *
	 *  \return SUCCESSFUL_RETURN, \n
	 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
	 *	        RET_UNKNOWN_BUG
	 */
	virtual returnValue print(	std::ostream& stream            = std::cout,
								const std::string& name         = DEFAULT_LABEL,
								const std::string& startString  = DEFAULT_START_STRING,
								const std::string& endString    = DEFAULT_END_STRING,
								uint width                      = DEFAULT_WIDTH,
								uint precision                  = DEFAULT_PRECISION,
								const std::string& colSeparator = DEFAULT_COL_SEPARATOR,
								const std::string& rowSeparator = DEFAULT_ROW_SEPARATOR
								) const;

	/** Prints object to file with given name. Various settings can
	 *	be specified defining its output format.
	 *
	 *	@param[in] filename			Filename for printing.
	 *	@param[in] name				Name label to be printed before the numerical values.
	 *	@param[in] startString		Prefix before printing the numerical values.
	 *	@param[in] endString		Suffix after printing the numerical values.
	 *	@param[in] width			Total number of digits per single numerical value.
	 *	@param[in] precision		Number of decimals per single numerical value.
	 *	@param[in] colSeparator		Separator between the columns of the numerical values.
	 *	@param[in] rowSeparator		Separator between the rows of the numerical values.
	 *
	 *  \return SUCCESSFUL_RETURN, \n
	 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
	 *	        RET_UNKNOWN_BUG
	 */
	virtual returnValue print(	const std::string& filename,
								const std::string& name         = DEFAULT_LABEL,
								const std::string& startString  = DEFAULT_START_STRING,
								const std::string& endString    = DEFAULT_END_STRING,
								uint width                      = DEFAULT_WIDTH,
								uint precision                  = DEFAULT_PRECISION,
								const std::string& colSeparator = DEFAULT_COL_SEPARATOR,
								const std::string& rowSeparator = DEFAULT_ROW_SEPARATOR
								) const;

	/** Prints object to given file. Various settings can
	 *	be specified defining its output format.
	 *
	 *	@param[in] stream			Output stream for printing.
	 *	@param[in] name				Name label to be printed before the numerical values.
	 *	@param[in] printScheme		Print scheme defining the output format of the information.
	 *
	 *  \return SUCCESSFUL_RETURN, \n
	 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
	 *	        RET_UNKNOWN_BUG
	 */
	virtual returnValue print(	std::ostream& stream,
								const std::string& name,
								PrintScheme printScheme
								) const;

	/** Prints object to given file. Various settings can
	 *	be specified defining its output format.
	 *
	 *	@param[in] filename			Filename for printing.
	 *	@param[in] name				Name label to be printed before the numerical values.
	 *	@param[in] printScheme		Print scheme defining the output format of the information.
	 *
	 *  \return SUCCESSFUL_RETURN, \n
	 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
	 *	        RET_UNKNOWN_BUG
	 */
	virtual returnValue print(	const std::string& filename,
								const std::string& name,
								PrintScheme printScheme
								) const;

	/** Read data from an input file. */
	virtual returnValue read(	std::istream& stream
								);

	/** Read data from an input file. */
	virtual returnValue read(	const std::string& filename
								);
};

/** Prints the matrix into a stream. */
template<typename T>
std::ostream& operator<<(	std::ostream& _stream,
							const GenericVector< T >& _arg
							)
{
	if (_arg.print( _stream ) != SUCCESSFUL_RETURN)
		ACADOERRORTEXT(RET_INVALID_ARGUMENTS, "Cannot write to output stream.");

	return _stream;
}

/** Read a matrix from an input stream. */
template<typename T>
std::istream& operator>>(	std::istream& _stream,
							GenericVector< T >& _arg
)
{
	if (_arg.read( _stream ) != SUCCESSFUL_RETURN )
		ACADOERRORTEXT(RET_INVALID_ARGUMENTS, "Cannot read from input stream.");

	return _stream;
}

/** Type definition of the vector of doubles. */
typedef GenericVector< double > DVector;
/** Type definition of the vector of doubles. */
typedef GenericVector< int > IVector;
/** Type definition of the vector of doubles. */
typedef GenericVector< bool > BVector;

static       DVector emptyVector;
static const DVector emptyConstVector;

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_VECTOR_HPP
