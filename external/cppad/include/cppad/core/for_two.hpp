# ifndef CPPAD_CORE_FOR_TWO_HPP
# define CPPAD_CORE_FOR_TWO_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin ForTwo$$
$spell
	ddy
	typename
	Taylor
	const
$$





$section Forward Mode Second Partial Derivative Driver$$
$mindex order easy$$

$head Syntax$$
$icode%ddy% = %f%.ForTwo(%x%, %j%, %k%)%$$


$head Purpose$$
We use $latex F : B^n \rightarrow B^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to $icode f$$.
The syntax above sets
$latex \[
	ddy [ i * p + \ell ]
	=
	\DD{ F_i }{ x_{j[ \ell ]} }{ x_{k[ \ell ]} } (x)
\] $$
for $latex i = 0 , \ldots , m-1$$
and $latex \ell = 0 , \ldots , p$$,
where $latex p$$ is the size of the vectors $icode j$$ and $icode k$$.

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$
(see $cref/ForTwo Uses Forward/ForTwo/ForTwo Uses Forward/$$ below).

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %VectorBase% &%x%
%$$
(see $cref/VectorBase/ForTwo/VectorBase/$$ below)
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
It specifies
that point at which to evaluate the partial derivatives listed above.

$head j$$
The argument $icode j$$ has prototype
$codei%
	const %VectorSize_t% &%j%
%$$
(see $cref/VectorSize_t/ForTwo/VectorSize_t/$$ below)
We use $icode p$$ to denote the size of the vector $icode j$$.
All of the indices in $icode j$$
must be less than $icode n$$; i.e.,
for $latex \ell = 0 , \ldots , p-1$$, $latex j[ \ell ]  < n$$.

$head k$$
The argument $icode k$$ has prototype
$codei%
	const %VectorSize_t% &%k%
%$$
(see $cref/VectorSize_t/ForTwo/VectorSize_t/$$ below)
and its size must be equal to $icode p$$,
the size of the vector $icode j$$.
All of the indices in $icode k$$
must be less than $icode n$$; i.e.,
for $latex \ell = 0 , \ldots , p-1$$, $latex k[ \ell ]  < n$$.

$head ddy$$
The result $icode ddy$$ has prototype
$codei%
	%VectorBase% %ddy%
%$$
(see $cref/VectorBase/ForTwo/VectorBase/$$ below)
and its size is $latex m * p$$.
It contains the requested partial derivatives; to be specific,
for $latex i = 0 , \ldots , m - 1 $$
and $latex \ell = 0 , \ldots , p - 1$$
$latex \[
	ddy [ i * p + \ell ]
	=
	\DD{ F_i }{ x_{j[ \ell ]} }{ x_{k[ \ell ]} } (x)
\] $$

$head VectorBase$$
The type $icode VectorBase$$ must be a $cref SimpleVector$$ class with
$cref/elements of type Base/SimpleVector/Elements of Specified Type/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head VectorSize_t$$
The type $icode VectorSize_t$$ must be a $cref SimpleVector$$ class with
$cref/elements of type size_t/SimpleVector/Elements of Specified Type/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head ForTwo Uses Forward$$
After each call to $cref Forward$$,
the object $icode f$$ contains the corresponding
$cref/Taylor coefficients/glossary/Taylor Coefficient/$$.
After a call to $code ForTwo$$,
the zero order Taylor coefficients correspond to
$icode%f%.Forward(0, %x%)%$$
and the other coefficients are unspecified.

$head Examples$$
$children%
	example/general/for_two.cpp
%$$
The routine
$cref/ForTwo/for_two.cpp/$$ is both an example and test.
It returns $code true$$, if it succeeds and $code false$$ otherwise.

$end
-----------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

template <typename Base>
template <typename VectorBase, typename VectorSize_t>
VectorBase ADFun<Base>::ForTwo(
	const VectorBase   &x,
	const VectorSize_t &j,
	const VectorSize_t &k)
{	size_t i;
	size_t j1;
	size_t k1;
	size_t l;

	size_t n = Domain();
	size_t m = Range();
	size_t p = j.size();

	// check VectorBase is Simple Vector class with Base type elements
	CheckSimpleVector<Base, VectorBase>();

	// check VectorSize_t is Simple Vector class with size_t elements
	CheckSimpleVector<size_t, VectorSize_t>();

	CPPAD_ASSERT_KNOWN(
		x.size() == n,
		"ForTwo: Length of x not equal domain dimension for f."
	);
	CPPAD_ASSERT_KNOWN(
		j.size() == k.size(),
		"ForTwo: Lenght of the j and k vectors are not equal."
	);
	// point at which we are evaluating the second partials
	Forward(0, x);


	// dimension the return value
	VectorBase ddy(m * p);

	// allocate memory to hold all possible diagonal Taylor coefficients
	// (for large sparse cases, this is not efficient)
	VectorBase D(m * n);

	// boolean flag for which diagonal coefficients are computed
	CppAD::vector<bool> c(n);
	for(j1 = 0; j1 < n; j1++)
		c[j1] = false;

	// direction vector in argument space
	VectorBase dx(n);
	for(j1 = 0; j1 < n; j1++)
		dx[j1] = Base(0.0);

	// result vector in range space
	VectorBase dy(m);

	// compute the diagonal coefficients that are needed
	for(l = 0; l < p; l++)
	{	j1 = j[l];
		k1 = k[l];
		CPPAD_ASSERT_KNOWN(
		j1 < n,
		"ForTwo: an element of j not less than domain dimension for f."
		);
		CPPAD_ASSERT_KNOWN(
		k1 < n,
		"ForTwo: an element of k not less than domain dimension for f."
		);
		size_t count = 2;
		while(count)
		{	count--;
			if( ! c[j1] )
			{	// diagonal term in j1 direction
				c[j1]  = true;
				dx[j1] = Base(1.0);
				Forward(1, dx);

				dx[j1] = Base(0.0);
				dy     = Forward(2, dx);
				for(i = 0; i < m; i++)
					D[i * n + j1 ] = dy[i];
			}
			j1 = k1;
		}
	}
	// compute all the requested cross partials
	for(l = 0; l < p; l++)
	{	j1 = j[l];
		k1 = k[l];
		if( j1 == k1 )
		{	for(i = 0; i < m; i++)
				ddy[i * p + l] = Base(2.0) * D[i * n + j1];
		}
		else
		{
			// cross term in j1 and k1 directions
			dx[j1] = Base(1.0);
			dx[k1] = Base(1.0);
			Forward(1, dx);

			dx[j1] = Base(0.0);
			dx[k1] = Base(0.0);
			dy = Forward(2, dx);

			// place result in return value
			for(i = 0; i < m; i++)
				ddy[i * p + l] = dy[i] - D[i*n+j1] - D[i*n+k1];

		}
	}
	return ddy;
}

} // END CppAD namespace

# endif
