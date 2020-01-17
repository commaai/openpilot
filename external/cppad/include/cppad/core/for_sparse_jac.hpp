# ifndef CPPAD_CORE_FOR_SPARSE_JAC_HPP
# define CPPAD_CORE_FOR_SPARSE_JAC_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin ForSparseJac$$
$spell
	std
	var
	Jacobian
	Jac
	const
	Bool
	proportional
	VecAD
	CondExpRel
	optimizer
	cpp
$$

$section Jacobian Sparsity Pattern: Forward Mode$$

$head Syntax$$
$icode%s% = %f%.ForSparseJac(%q%, %r%)
%$$
$icode%s% = %f%.ForSparseJac(%q%, %r%, %transpose%, %dependency%)%$$

$head Purpose$$
We use $latex F : B^n \rightarrow B^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to $icode f$$.
For a fixed $latex n \times q$$ matrix $latex R$$,
the Jacobian of $latex F[ x + R * u ]$$
with respect to $latex u$$ at $latex u = 0$$ is
$latex \[
	S(x) = F^{(1)} ( x ) * R
\] $$
Given a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for $latex R$$,
$code ForSparseJac$$ returns a sparsity pattern for the $latex S(x)$$.

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$.
After a call to $code ForSparseJac$$, the sparsity pattern
for each of the variables in the operation sequence
is held in $icode f$$ (for possible later use by $cref RevSparseHes$$).
These sparsity patterns are stored with elements of type $code bool$$
or elements of type $code std::set<size_t>$$
(see $cref/VectorSet/ForSparseJac/VectorSet/$$ below).

$subhead size_forward_bool$$
After $code ForSparseJac$$, if $icode k$$ is a $code size_t$$ object,
$codei%
	%k% = %f%.size_forward_bool()
%$$
sets $icode k$$ to the amount of memory (in unsigned character units)
used to store the sparsity pattern with elements of type $code bool$$
in the function object $icode f$$.
If the sparsity patterns for the previous $code ForSparseJac$$ used
elements of type $code bool$$,
the return value for $code size_forward_bool$$ will be non-zero.
Otherwise, its return value will be zero.
This sparsity pattern is stored for use by $cref RevSparseHes$$ and
when it is not longer needed, it can be deleted
(and the corresponding memory freed) using
$codei%
	%f%.size_forward_bool(0)
%$$
After this call, $icode%f%.size_forward_bool()%$$ will return zero.

$subhead size_forward_set$$
After $code ForSparseJac$$, if $icode k$$ is a $code size_t$$ object,
$codei%
	%k% = %f%.size_forward_set()
%$$
sets $icode k$$ to the amount of memory (in unsigned character units)
used to store the
$cref/vector of sets/glossary/Sparsity Pattern/Vector of Sets/$$
sparsity patterns.
If the sparsity patterns for this operation use elements of type $code bool$$,
the return value for $code size_forward_set$$ will be zero.
Otherwise, its return value will be non-zero.
This sparsity pattern is stored for use by $cref RevSparseHes$$ and
when it is not longer needed, it can be deleted
(and the corresponding memory freed) using
$codei%
	%f%.size_forward_set(0)
%$$
After this call, $icode%f%.size_forward_set()%$$ will return zero.

$head x$$
If the operation sequence in $icode f$$ is
$cref/independent/glossary/Operation/Independent/$$ of
the independent variables in $latex x \in B^n$$,
the sparsity pattern is valid for all values of
(even if it has $cref CondExp$$ or $cref VecAD$$ operations).

$head q$$
The argument $icode q$$ has prototype
$codei%
	size_t %q%
%$$
It specifies the number of columns in
$latex R \in B^{n \times q}$$ and the Jacobian
$latex S(x) \in B^{m \times q}$$.

$head transpose$$
The argument $icode transpose$$ has prototype
$codei%
	bool %transpose%
%$$
The default value $code false$$ is used when $icode transpose$$ is not present.

$head dependency$$
The argument $icode dependency$$ has prototype
$codei%
	bool %dependency%
%$$
If $icode dependency$$ is true,
the $cref/dependency pattern/dependency.cpp/Dependency Pattern/$$
(instead of sparsity pattern) is computed.

$head r$$
The argument $icode r$$ has prototype
$codei%
	const %VectorSet%& %r%
%$$
see $cref/VectorSet/ForSparseJac/VectorSet/$$ below.

$subhead transpose false$$
If $icode r$$ has elements of type $code bool$$,
its size is $latex n * q$$.
If it has elements of type $code std::set<size_t>$$,
its size is $latex n$$ and all the set elements must be between
zero and $icode%q%-1%$$ inclusive.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the matrix $latex R \in B^{n \times q}$$.

$subhead transpose true$$
If $icode r$$ has elements of type $code bool$$,
its size is $latex q * n$$.
If it has elements of type $code std::set<size_t>$$,
its size is $latex q$$ and all the set elements must be between
zero and $icode%n%-1%$$ inclusive.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the matrix $latex R^\R{T} \in B^{q \times n}$$.

$head s$$
The return value $icode s$$ has prototype
$codei%
	%VectorSet% %s%
%$$
see $cref/VectorSet/ForSparseJac/VectorSet/$$ below.

$subhead transpose false$$
If $icode s$$ has elements of type $code bool$$,
its size is $latex m * q$$.
If it has elements of type $code std::set<size_t>$$,
its size is $latex m$$ and all its set elements are between
zero and $icode%q%-1%$$ inclusive.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the matrix $latex S(x) \in B^{m \times q}$$.

$subhead transpose true$$
If $icode s$$ has elements of type $code bool$$,
its size is $latex q * m$$.
If it has elements of type $code std::set<size_t>$$,
its size is $latex q$$ and all its set elements are between
zero and $icode%m%-1%$$ inclusive.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the matrix $latex S(x)^\R{T} \in B^{q \times m}$$.

$head VectorSet$$
The type $icode VectorSet$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code bool$$ or $code std::set<size_t>$$;
see $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for a discussion
of the difference.

$head Entire Sparsity Pattern$$
Suppose that $latex q = n$$ and
$latex R$$ is the $latex n \times n$$ identity matrix.
In this case,
the corresponding value for $icode s$$ is a
sparsity pattern for the Jacobian $latex S(x) = F^{(1)} ( x )$$.

$head Example$$
$children%
	example/sparse/for_sparse_jac.cpp
%$$
The file
$cref for_sparse_jac.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.
The file
$cref/sparsity_sub.cpp/sparsity_sub.cpp/ForSparseJac/$$
contains an example and test of using $code ForSparseJac$$
to compute the sparsity pattern for a subset of the Jacobian.

$end
-----------------------------------------------------------------------------
*/

# include <cppad/local/std_set.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file for_sparse_jac.hpp
Forward mode Jacobian sparsity patterns.
*/
// ---------------------------------------------------------------------------
/*!
Private helper function for ForSparseJac(q, r) boolean sparsity patterns.

All of the description in the public member function ForSparseJac(q, r)
applies.

\param set_type
is a \c bool value. This argument is used to dispatch to the proper source
code depending on the value of \c VectorSet::value_type.

\param transpose
See \c ForSparseJac(q, r, transpose, dependency).

\param dependency
See \c ForSparseJac(q, r, transpose, dependency).

\param q
See \c ForSparseJac(q, r, transpose, dependency).

\param r
See \c ForSparseJac(q, r, transpose, dependency).

\param s
is the return value for the corresponding call to \c ForSparseJac(q, r).
*/

template <class Base>
template <class VectorSet>
void ADFun<Base>::ForSparseJacCase(
	bool                set_type      ,
	bool                transpose     ,
	bool                dependency    ,
	size_t              q             ,
	const VectorSet&    r             ,
	VectorSet&          s             )
{	size_t m = Range();
	size_t n = Domain();

	// check VectorSet is Simple Vector class with bool elements
	CheckSimpleVector<bool, VectorSet>();

	// dimension size of result vector
	s.resize( m * q );

	// temporary indices
	size_t i, j;
	//
	CPPAD_ASSERT_KNOWN(
		q > 0,
		"ForSparseJac: q is not greater than zero"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(r.size()) == n * q,
		"ForSparseJac: size of r is not equal to\n"
		"q times domain dimension for ADFun object."
	);
	//
	// allocate memory for the requested sparsity calculation result
	for_jac_sparse_pack_.resize(num_var_tape_, q);

	// set values corresponding to independent variables
	for(i = 0; i < n; i++)
	{	CPPAD_ASSERT_UNKNOWN( ind_taddr_[i] < num_var_tape_ );
		// ind_taddr_[i] is operator taddr for i-th independent variable
		CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[i] ) == local::InvOp );

		// set bits that are true
		if( transpose )
		{	for(j = 0; j < q; j++) if( r[ j * n + i ] )
				for_jac_sparse_pack_.add_element( ind_taddr_[i], j);
		}
		else
		{	for(j = 0; j < q; j++) if( r[ i * q + j ] )
				for_jac_sparse_pack_.add_element( ind_taddr_[i], j);
		}
	}

	// evaluate the sparsity patterns
	local::ForJacSweep(
		dependency,
		n,
		num_var_tape_,
		&play_,
		for_jac_sparse_pack_
	);

	// return values corresponding to dependent variables
	CPPAD_ASSERT_UNKNOWN( size_t(s.size()) == m * q );
	for(i = 0; i < m; i++)
	{	CPPAD_ASSERT_UNKNOWN( dep_taddr_[i] < num_var_tape_ );

		// extract the result from for_jac_sparse_pack_
		if( transpose )
		{	for(j = 0; j < q; j++)
				s[ j * m + i ] = false;
		}
		else
		{	for(j = 0; j < q; j++)
				s[ i * q + j ] = false;
		}
		CPPAD_ASSERT_UNKNOWN( for_jac_sparse_pack_.end() == q );
		local::sparse_pack::const_iterator itr(for_jac_sparse_pack_, dep_taddr_[i] );
		j = *itr;
		while( j < q )
		{	if( transpose )
				s[j * m + i] = true;
			else	s[i * q + j] = true;
			j = *(++itr);
		}
	}
}
// ---------------------------------------------------------------------------
/*!
Private helper function for \c ForSparseJac(q, r) set sparsity.

All of the description in the public member function \c ForSparseJac(q, r)
applies.

\param set_type
is a \c std::set<size_t> object.
This argument is used to dispatch to the proper source
code depending on the value of \c VectorSet::value_type.

\param transpose
See \c ForSparseJac(q, r, transpose, dependency).

\param dependency
See \c ForSparseJac(q, r, transpose, dependency).

\param q
See \c ForSparseJac(q, r, transpose, dependency).

\param r
See \c ForSparseJac(q, r, transpose, dependency).

\param s
is the return value for the corresponding call to \c ForSparseJac(q, r).
*/
template <class Base>
template <class VectorSet>
void ADFun<Base>::ForSparseJacCase(
	const std::set<size_t>&    set_type      ,
	bool                       transpose     ,
	bool                       dependency    ,
	size_t                     q             ,
	const VectorSet&           r             ,
	VectorSet&                 s             )
{	size_t m = Range();
	size_t n = Domain();

	// check VectorSet is Simple Vector class with sets for elements
	CheckSimpleVector<std::set<size_t>, VectorSet>(
		local::one_element_std_set<size_t>(), local::two_element_std_set<size_t>()
	);

	// dimension size of result vector
	if( transpose )
		s.resize(q);
	else	s.resize( m );

	// temporary indices
	size_t i, j;
	std::set<size_t>::const_iterator itr_1;
	//
	CPPAD_ASSERT_KNOWN(
		q > 0,
		"ForSparseJac: q is not greater than zero"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(r.size()) == n || transpose,
		"ForSparseJac: size of r is not equal to n and transpose is false."
	);
	CPPAD_ASSERT_KNOWN(
		size_t(r.size()) == q || ! transpose,
		"ForSparseJac: size of r is not equal to q and transpose is true."
	);
	//
	// allocate memory for the requested sparsity calculation
	for_jac_sparse_set_.resize(num_var_tape_, q);

	// set values corresponding to independent variables
	if( transpose )
	{	for(i = 0; i < q; i++)
		{	// add the elements that are present
			itr_1 = r[i].begin();
			while( itr_1 != r[i].end() )
			{	j = *itr_1++;
				CPPAD_ASSERT_KNOWN(
				j < n,
				"ForSparseJac: transpose is true and element of the set\n"
				"r[j] has value greater than or equal n."
				);
				CPPAD_ASSERT_UNKNOWN( ind_taddr_[j] < num_var_tape_ );
				// operator for j-th independent variable
				CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[j] ) == local::InvOp );
				for_jac_sparse_set_.add_element( ind_taddr_[j], i);
			}
		}
	}
	else
	{	for(i = 0; i < n; i++)
		{	CPPAD_ASSERT_UNKNOWN( ind_taddr_[i] < num_var_tape_ );
			// ind_taddr_[i] is operator taddr for i-th independent variable
			CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[i] ) == local::InvOp );

			// add the elements that are present
			itr_1 = r[i].begin();
			while( itr_1 != r[i].end() )
			{	j = *itr_1++;
				CPPAD_ASSERT_KNOWN(
					j < q,
					"ForSparseJac: an element of the set r[i] "
					"has value greater than or equal q."
				);
				for_jac_sparse_set_.add_element( ind_taddr_[i], j);
			}
		}
	}
	// evaluate the sparsity patterns
	local::ForJacSweep(
		dependency,
		n,
		num_var_tape_,
		&play_,
		for_jac_sparse_set_
	);

	// return values corresponding to dependent variables
	CPPAD_ASSERT_UNKNOWN( size_t(s.size()) == m || transpose );
	CPPAD_ASSERT_UNKNOWN( size_t(s.size()) == q || ! transpose );
	for(i = 0; i < m; i++)
	{	CPPAD_ASSERT_UNKNOWN( dep_taddr_[i] < num_var_tape_ );

		// extract results from for_jac_sparse_set_
		// and add corresponding elements to sets in s
		CPPAD_ASSERT_UNKNOWN( for_jac_sparse_set_.end() == q );
		local::sparse_list::const_iterator itr_2(for_jac_sparse_set_, dep_taddr_[i] );
		j = *itr_2;
		while( j < q )
		{	if( transpose )
				s[j].insert(i);
			else	s[i].insert(j);
			j = *(++itr_2);
		}
	}
}
// ---------------------------------------------------------------------------

/*!
User API for Jacobian sparsity patterns using forward mode.

The C++ source code corresponding to this operation is
\verbatim
	s = f.ForSparseJac(q, r, transpose, dependency)
\endverbatim

\tparam Base
is the base type for this recording.

\tparam VectorSet
is a simple vector with elements of type \c bool
or \c std::set<size_t>.

\param q
is the number of columns in the matrix \f$ R \f$.

\param r
is a sparsity pattern for the matrix \f$ R \f$.

\param transpose
are sparsity patterns for \f$ R \f$ and \f$ S(x) \f$ transposed.

\param dependency
Are the derivatives with respect to left and right of the expression below
considered to be non-zero:
\code
	CondExpRel(left, right, if_true, if_false)
\endcode
This is used by the optimizer to obtain the correct dependency relations.

\return
The value of \c transpose is false (true),
the return value is a sparsity pattern for \f$ S(x) \f$ (\f$ S(x)^T \f$) where
\f[
	S(x) = F^{(1)} (x) * R
\f]
where \f$ F \f$ is the function corresponding to the operation sequence
and \a x is any argument value.
If \c VectorSet::value_type is \c bool,
the return value has size \f$ m * q \f$ (\f$ q * m \f$).
where \c m is the number of dependent variables
corresponding to the operation sequence stored in \c f.
If \c VectorSet::value_type is \c std::set<size_t>,
the return value has size \f$ m \f$ ( \f$ q \f$ )
and with all its elements between zero and
\f$ q - 1 \f$ ( \f$ m - 1 \f$).

\par Side Effects
If \c VectorSet::value_type is \c bool,
the forward sparsity pattern for all of the variables on the
tape is stored in \c for_jac_sparse_pack__.
In this case
\verbatim
	for_jac_sparse_pack_.n_set() == num_var_tape_
	for_jac_sparse_pack_.end() == q
	for_jac_sparse_set_.n_set()  == 0
	for_jac_sparse_set_.end()  == 0
\endverbatim
\n
\n
If \c VectorSet::value_type is \c std::set<size_t>,
the forward sparsity pattern for all of the variables on the
tape is stored in \c for_jac_sparse_set__.
In this case
\verbatim
	for_jac_sparse_set_.n_set()   == num_var_tape_
	for_jac_sparse_set_.end()   == q
	for_jac_sparse_pack_.n_set()  == 0
	for_jac_sparse_pack_.end()  == 0
\endverbatim
*/
template <class Base>
template <class VectorSet>
VectorSet ADFun<Base>::ForSparseJac(
	size_t             q             ,
	const VectorSet&   r             ,
	bool               transpose     ,
	bool               dependency    )
{	VectorSet s;
	typedef typename VectorSet::value_type Set_type;

	// free all memory currently in sparsity patterns
	for_jac_sparse_pack_.resize(0, 0);
	for_jac_sparse_set_.resize(0, 0);

	ForSparseJacCase(
		Set_type()  ,
		transpose   ,
		dependency  ,
		q           ,
		r           ,
		s
	);

	return s;
}
// ===========================================================================
// ForSparseJacCheckpoint
/*!
Forward mode Jacobian sparsity calculation used by checkpoint functions.

\tparam Base
is the base type for this recording.

\param transpose
is true (false) s is equal to \f$ S(x) \f$ (\f$ S(x)^T \f$)
where
\f[
	S(x) = F^{(1)} (x) * R
\f]
where \f$ F \f$ is the function corresponding to the operation sequence
and \f$ x \f$ is any argument value.

\param q
is the number of columns in the matrix \f$ R \f$.

\param r
is a sparsity pattern for the matrix \f$ R \f$.

\param transpose
are the sparsity patterns for \f$ R \f$ and \f$ S(x) \f$ transposed.

\param dependency
Are the derivatives with respect to left and right of the expression below
considered to be non-zero:
\code
	CondExpRel(left, right, if_true, if_false)
\endcode
This is used by the optimizer to obtain the correct dependency relations.

\param s
The input size and elements of s do not matter.
On output, s is the sparsity pattern for the matrix \f$ S(x) \f$
or \f$ S(x)^T \f$ depending on transpose.

\par Side Effects
If \c VectorSet::value_type is \c bool,
the forward sparsity pattern for all of the variables on the
tape is stored in \c for_jac_sparse_pack__.
In this case
\verbatim
	for_jac_sparse_pack_.n_set() == num_var_tape_
	for_jac_sparse_pack_.end() == q
	for_jac_sparse_set_.n_set()  == 0
	for_jac_sparse_set_.end()  == 0
\endverbatim
\n
\n
If \c VectorSet::value_type is \c std::set<size_t>,
the forward sparsity pattern for all of the variables on the
tape is stored in \c for_jac_sparse_set__.
In this case
\verbatim
	for_jac_sparse_set_.n_set()   == num_var_tape_
	for_jac_sparse_set_.end()   == q
	for_jac_sparse_pack_.n_set()  == 0
	for_jac_sparse_pack_.end()  == 0
\endverbatim
*/
template <class Base>
void ADFun<Base>::ForSparseJacCheckpoint(
	size_t                        q          ,
	const local::sparse_list&     r          ,
	bool                          transpose  ,
	bool                          dependency ,
	local::sparse_list&                  s          )
{	size_t n = Domain();
	size_t m = Range();

# ifndef NDEBUG
	if( transpose )
	{	CPPAD_ASSERT_UNKNOWN( r.n_set() == q );
		CPPAD_ASSERT_UNKNOWN( r.end()   == n );
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( r.n_set() == n );
		CPPAD_ASSERT_UNKNOWN( r.end()   == q );
	}
	for(size_t j = 0; j < n; j++)
	{	CPPAD_ASSERT_UNKNOWN( ind_taddr_[j] == (j+1) );
		CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[j] ) == local::InvOp );
	}
# endif

	// free all memory currently in sparsity patterns
	for_jac_sparse_pack_.resize(0, 0);
	for_jac_sparse_set_.resize(0, 0);

	// allocate new sparsity pattern
	for_jac_sparse_set_.resize(num_var_tape_, q);

	// set sparsity pattern for dependent variables
	if( transpose )
	{	for(size_t i = 0; i < q; i++)
		{	local::sparse_list::const_iterator itr(r, i);
			size_t j = *itr;
			while( j < n )
			{	for_jac_sparse_set_.add_element( ind_taddr_[j], i );
				j = *(++itr);
			}
		}
	}
	else
	{	for(size_t j = 0; j < n; j++)
		{	local::sparse_list::const_iterator itr(r, j);
			size_t i = *itr;
			while( i < q )
			{	for_jac_sparse_set_.add_element( ind_taddr_[j], i );
				i = *(++itr);
			}
		}
	}

	// evaluate the sparsity pattern for all variables
	local::ForJacSweep(
		dependency,
		n,
		num_var_tape_,
		&play_,
		for_jac_sparse_set_
	);

	// dimension the return value
	if( transpose )
		s.resize(q, m);
	else
		s.resize(m, q);

	// return values corresponding to dependent variables
	for(size_t i = 0; i < m; i++)
	{	CPPAD_ASSERT_UNKNOWN( dep_taddr_[i] < num_var_tape_ );

		// extract the result from for_jac_sparse_set_
		CPPAD_ASSERT_UNKNOWN( for_jac_sparse_set_.end() == q );
		local::sparse_list::const_iterator itr(for_jac_sparse_set_, dep_taddr_[i] );
		size_t j = *itr;
		while( j < q )
		{	if( transpose )
				s.add_element(j, i);
			else
				s.add_element(i, j);
			j  = *(++itr);
		}
	}

}


} // END_CPPAD_NAMESPACE
# endif
