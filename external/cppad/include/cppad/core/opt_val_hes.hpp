# ifndef CPPAD_CORE_OPT_VAL_HES_HPP
# define CPPAD_CORE_OPT_VAL_HES_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin opt_val_hes$$
$spell
	hes
	sy
	Jacobian
	hes
	signdet
	jac
	Bradley
	const
	CppAD
$$



$section Jacobian and Hessian of Optimal Values$$
$mindex opt_val_hes$$

$head Syntax$$
$icode%signdet% = opt_val_hes(%x%, %y%, %fun%, %jac%, %hes%)%$$

$head See Also$$
$cref BenderQuad$$

$head Reference$$
Algorithmic differentiation of implicit functions and optimal values,
Bradley M. Bell and James V. Burke, Advances in Automatic Differentiation,
2008, Springer.

$head Purpose$$
We are given a function
$latex S : \B{R}^n \times \B{R}^m \rightarrow \B{R}^\ell$$
and we define $latex F : \B{R}^n \times \B{R}^m \rightarrow \B{R}$$
and $latex V : \B{R}^n \rightarrow \B{R} $$ by
$latex \[
\begin{array}{rcl}
	F(x, y) & = & \sum_{k=0}^{\ell-1} S_k ( x , y)
	\\
	V(x)    & = & F [ x , Y(x) ]
	\\
	0       & = & \partial_y F [x , Y(x) ]
\end{array}
\] $$
We wish to compute the Jacobian
and possibly also the Hessian, of $latex V (x)$$.

$head BaseVector$$
The type $icode BaseVector$$ must be a
$cref SimpleVector$$ class.
We use $icode Base$$ to refer to the type of the elements of
$icode BaseVector$$; i.e.,
$codei%
	%BaseVector%::value_type
%$$

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %BaseVector%& %x%
%$$
and its size must be equal to $icode n$$.
It specifies the point at which we evaluating
the Jacobian $latex V^{(1)} (x)$$
(and possibly the Hessian $latex V^{(2)} (x)$$).


$head y$$
The argument $icode y$$ has prototype
$codei%
	const %BaseVector%& %y%
%$$
and its size must be equal to $icode m$$.
It must be equal to $latex Y(x)$$; i.e.,
it must solve the implicit equation
$latex \[
	0 = \partial_y F ( x , y)
\] $$

$head Fun$$
The argument $icode fun$$ is an object of type $icode Fun$$
which must support the member functions listed below.
CppAD will may be recording operations of the type $codei%AD<%Base%>%$$
when these member functions are called.
These member functions must not stop such a recording; e.g.,
they must not call $cref/AD<Base>::abort_recording/abort_recording/$$.

$subhead Fun::ad_vector$$
The type $icode%Fun%::ad_vector%$$ must be a
$cref SimpleVector$$ class with elements of type $codei%AD<%Base%>%$$; i.e.
$codei%
	%Fun%::ad_vector::value_type
%$$
is equal to $codei%AD<%Base%>%$$.

$subhead fun.ell$$
The type $icode Fun$$ must support the syntax
$codei%
	%ell% = %fun%.ell()
%$$
where $icode ell$$ has prototype
$codei%
	size_t %ell%
%$$
and is the value of $latex \ell$$; i.e.,
the number of terms in the summation.
$pre

$$
One can choose $icode ell$$ equal to one, and have
$latex S(x,y)$$ the same as $latex F(x, y)$$.
Each of the functions $latex S_k (x , y)$$,
(in the summation defining $latex F(x, y)$$)
is differentiated separately using AD.
For very large problems, breaking $latex F(x, y)$$ into the sum
of separate simpler functions may reduce the amount of memory necessary for
algorithmic differentiation and there by speed up the process.

$subhead fun.s$$
The type $icode Fun$$ must support the syntax
$codei%
	%s_k% = %fun%.s(%k%, %x%, %y%)
%$$
The $icode%fun%.s%$$ argument $icode k$$ has prototype
$codei%
	size_t %k%
%$$
and is between zero and $icode%ell% - 1%$$.
The argument $icode x$$ to $icode%fun%.s%$$ has prototype
$codei%
	const %Fun%::ad_vector& %x%
%$$
and its size must be equal to $icode n$$.
The argument $icode y$$ to $icode%fun%.s%$$ has prototype
$codei%
	const %Fun%::ad_vector& %y%
%$$
and its size must be equal to $icode m$$.
The $icode%fun%.s%$$ result $icode s_k$$ has prototype
$codei%
	AD<%Base%> %s_k%
%$$
and its value must be given by $latex s_k = S_k ( x , y )$$.

$subhead fun.sy$$
The type $icode Fun$$ must support the syntax
$codei%
	%sy_k% = %fun%.sy(%k%, %x%, %y%)
%$$
The  argument $icode k$$ to $icode%fun%.sy%$$ has prototype
$codei%
	size_t %k%
%$$
The  argument $icode x$$ to $icode%fun%.sy%$$ has prototype
$codei%
	const %Fun%::ad_vector& %x%
%$$
and its size must be equal to $icode n$$.
The  argument $icode y$$ to $icode%fun%.sy%$$ has prototype
$codei%
	const %Fun%::ad_vector& %y%
%$$
and its size must be equal to $icode m$$.
The $icode%fun%.sy%$$ result $icode sy_k$$ has prototype
$codei%
	%Fun%::ad_vector %sy_k%
%$$
its size must be equal to $icode m$$,
and its value must be given by $latex sy_k = \partial_y S_k ( x , y )$$.

$head jac$$
The argument $icode jac$$ has prototype
$codei%
	%BaseVector%& %jac%
%$$
and has size $icode n$$ or zero.
The input values of its elements do not matter.
If it has size zero, it is not affected. Otherwise, on output
it contains the Jacobian of $latex V (x)$$; i.e.,
for $latex j = 0 , \ldots , n-1$$,
$latex \[
	jac[ j ] = V^{(1)} (x)_j
\] $$
where $icode x$$ is the first argument to $code opt_val_hes$$.

$head hes$$
The argument $icode hes$$ has prototype
$codei%
	%BaseVector%& %hes%
%$$
and has size $icode%n% * %n%$$ or zero.
The input values of its elements do not matter.
If it has size zero, it is not affected. Otherwise, on output
it contains the Hessian of $latex V (x)$$; i.e.,
for $latex i = 0 , \ldots , n-1$$, and
$latex j = 0 , \ldots , n-1$$,
$latex \[
	hes[ i * n + j ] = V^{(2)} (x)_{i,j}
\] $$


$head signdet$$
If $icode%hes%$$ has size zero, $icode signdet$$ is not defined.
Otherwise
the return value $icode signdet$$ is the sign of the determinant for
$latex \partial_{yy}^2 F(x , y) $$.
If it is zero, then the matrix is singular and
the Hessian is not computed ($icode hes$$ is not changed).

$head Example$$
$children%
	example/general/opt_val_hes.cpp
%$$
The file
$cref opt_val_hes.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.

$end
-----------------------------------------------------------------------------
*/

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file opt_val_hes.hpp
\brief Computing Jabobians and Hessians of Optimal Values
*/

/*!
Computing Jabobians and Hessians of Optimal Values

We are given a function
\f$ S : {\rm R}^n \times {\rm R}^m \rightarrow {\rm R}^\ell \f$
and we define \f$ F : {\rm R}^n \times {\rm R}^m \rightarrow {\rm R} \f$
and \f$ V : {\rm R}^n \rightarrow {\rm R}  \f$ by
\f[
\begin{array}{rcl}
	F(x, y) & = & \sum_{k=0}^{\ell-1} S_k ( x , y)
	\\
	V(x)    & = & F [ x , Y(x) ]
	\\
	0       & = & \partial_y F [x , Y(x) ]
\end{array}
\f]
We wish to compute the Jacobian
and possibly also the Hessian, of \f$ V (x) \f$.

\tparam BaseVector
The type \c BaseVector must be a SimpleVector class.
We use \c Base to refer to the type of the elements of
\c BaseVector; i.e.,
<tt>BaseVector::value_type</tt>.

\param x
is a vector with size \c n.
It specifies the point at which we evaluating
the Jacobian \f$ V^{(1)} (x) \f$
(and possibly the Hessian \f$ V^{(2)} (x) \f$).


\param y
is a vector with size \c m.
It must be equal to \f$ Y(x) \f$; i.e.,
it must solve the implicit equation
\f[
	0 = \partial_y F ( x , y)
\f]

\param fun
The argument \c fun is an object of type \c Fun
wich must support the member functions listed below.
CppAD will may be recording operations of the type  \c AD<Base>
when these member functions are called.
These member functions must not stop such a recording; e.g.,
they must not call \c AD<Base>::abort_recording.

\par Fun::ad_vector</tt>
The type <tt>Fun::ad_vector</tt> must be a
SimpleVector class with elements of type  \c AD<Base>; i.e.
<tt>Fun::ad_vector::value_type</tt>
is equal to  \c AD<Base>.

\par fun.ell
the type \c Fun must support the syntax
\verbatim
	ell = fun.ell()
\endverbatim
where \c ell is a \c size_t value that is set to \f$ \ell \f$; i.e.,
the number of terms in the summation.

\par fun.s
The type \c Fun must support the syntax
\verbatim
	s_k = fun.s(k, x, y)
\endverbatim
The argument \c k has prototype <tt>size_t k</tt>.
The argument \c x has prototype <tt>const Fun::ad_vector& x</tt>
and its size must be equal to \c n.
The argument \c y has prototype <tt>const Fun::ad_vector& y</tt>
and its size must be equal to \c m.
The return value \c s_k has prototype \c AD<Base> s_k
and its value must be given by \f$ s_k = S_k ( x , y ) \f$.

\par fun.sy
The type \c Fun must support the syntax
\verbatim
	sy_k = fun.sy(k, x, y)
\endverbatim
The argument \c k has prototype <tt>size_t k</tt>.
The argument \c x has prototype <tt>const Fun::ad_vector& x</tt>
and its size must be equal to \c n.
The argument \c y has prototype <tt>const Fun::ad_vector& y</tt>
and its size must be equal to \c m.
The return value \c sy_k has prototype <tt>Fun::ad_vector& sy_k</tt>,
its size is \c m
and its value must be given by \f$ sy_k = \partial_y S_k ( x , y ) \f$.

\param jac
is a vector with size \c n or zero.
The input values of its elements do not matter.
If it has size zero, it is not affected. Otherwise, on output
it contains the Jacobian of \f$ V (x) \f$; i.e.,
for \f$ j = 0 , \ldots , n-1 \f$,
\f[
	jac[ j ] = V^{(1)} (x)_j
\f] $$
where \c x is the first argument to \c opt_val_hes.

\param hes
is a vector with size <tt>n * n</tt> or zero.
The input values of its elements do not matter.
If it has size zero, it is not affected. Otherwise, on output
it contains the Hessian of \f$ V (x) \f$; i.e.,
for \f$ i = 0 , \ldots , n-1 \f$, and
\f$ j = 0 , \ldots , n-1 \f$,
\f[
	hes[ i * n + j ] = V^{(2)} (x)_{i,j}
\f]

\return
If <tt>hes.size() == 0</tt>, the return value is not defined.
Otherwise,
the return value is the sign of the determinant for
\f$ \partial_{yy}^2 F(x , y) \f$$.
If it is zero, then the matrix is singular and \c hes is not set
to its specified value.
*/


template <class BaseVector, class Fun>
int opt_val_hes(
	const BaseVector&   x     ,
	const BaseVector&   y     ,
	Fun                 fun   ,
	BaseVector&         jac   ,
	BaseVector&         hes   )
{	// determine the base type
	typedef typename BaseVector::value_type Base;

	// check that BaseVector is a SimpleVector class with Base elements
	CheckSimpleVector<Base, BaseVector>();

	// determine the AD vector type
	typedef typename Fun::ad_vector ad_vector;

	// check that ad_vector is a SimpleVector class with AD<Base> elements
	CheckSimpleVector< AD<Base> , ad_vector >();

	// size of the x and y spaces
	size_t n = size_t(x.size());
	size_t m = size_t(y.size());

	// number of terms in the summation
	size_t ell = fun.ell();

	// check size of return values
	CPPAD_ASSERT_KNOWN(
		size_t(jac.size()) == n || jac.size() == 0,
		"opt_val_hes: size of the vector jac is not equal to n or zero"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(hes.size()) == n * n || hes.size() == 0,
		"opt_val_hes: size of the vector hes is not equal to n * n or zero"
	);

	// some temporary indices
	size_t i, j, k;

	// AD version of S_k(x, y)
	ad_vector s_k(1);

	// ADFun version of S_k(x, y)
	ADFun<Base> S_k;

	// AD version of x
	ad_vector a_x(n);

	// AD version of y
	ad_vector a_y(n);

	if( jac.size() > 0  )
	{	// this is the easy part, computing the V^{(1)} (x) which is equal
		// to \partial_x F (x, y) (see Thoerem 2 of the reference).

		// copy x and y to AD version
		for(j = 0; j < n; j++)
			a_x[j] = x[j];
		for(j = 0; j < m; j++)
			a_y[j] = y[j];

		// initialize summation
		for(j = 0; j < n; j++)
			jac[j] = Base(0.);

		// add in \partial_x S_k (x, y)
		for(k = 0; k < ell; k++)
		{	// start recording
			Independent(a_x);
			// record
			s_k[0] = fun.s(k, a_x, a_y);
			// stop recording and store in S_k
			S_k.Dependent(a_x, s_k);
			// compute partial of S_k with respect to x
			BaseVector jac_k = S_k.Jacobian(x);
			// add \partial_x S_k (x, y) to jac
			for(j = 0; j < n; j++)
				jac[j] += jac_k[j];
		}
	}
	// check if we are done
	if( hes.size() == 0 )
		return 0;

	/*
	In this case, we need to compute the Hessian. Using Theorem 1 of the
	reference:
		Y^{(1)}(x) = - F_yy (x, y)^{-1} F_yx (x, y)
	Using Theorem 2 of the reference:
		V^{(2)}(x) = F_xx (x, y) + F_xy (x, y)  Y^{(1)}(x)
	*/
	// Base and AD version of xy
	BaseVector xy(n + m);
	ad_vector a_xy(n + m);
	for(j = 0; j < n; j++)
		a_xy[j] = xy[j] = x[j];
	for(j = 0; j < m; j++)
		a_xy[n+j] = xy[n+j] = y[j];

	// Initialization summation for Hessian of F
	size_t nm_sq = (n + m) * (n + m);
	BaseVector F_hes(nm_sq);
	for(j = 0; j < nm_sq; j++)
		F_hes[j] = Base(0.);
	BaseVector hes_k(nm_sq);

	// add in Hessian of S_k to hes
	for(k = 0; k < ell; k++)
	{	// start recording
		Independent(a_xy);
		// split out x
		for(j = 0; j < n; j++)
			a_x[j] = a_xy[j];
		// split out y
		for(j = 0; j < m; j++)
			a_y[j] = a_xy[n+j];
		// record
		s_k[0] = fun.s(k, a_x, a_y);
		// stop recording and store in S_k
		S_k.Dependent(a_xy, s_k);
		// when computing the Hessian it pays to optimize the tape
		S_k.optimize();
		// compute Hessian of S_k
		hes_k = S_k.Hessian(xy, 0);
		// add \partial_x S_k (x, y) to jac
		for(j = 0; j < nm_sq; j++)
			F_hes[j] += hes_k[j];
	}
	// Extract F_yx
	BaseVector F_yx(m * n);
	for(i = 0; i < m; i++)
	{	for(j = 0; j < n; j++)
			F_yx[i * n + j] = F_hes[ (i+n)*(n+m) + j ];
	}
	// Extract F_yy
	BaseVector F_yy(n * m);
	for(i = 0; i < m; i++)
	{	for(j = 0; j < m; j++)
			F_yy[i * m + j] = F_hes[ (i+n)*(n+m) + j + n ];
	}

	// compute - Y^{(1)}(x) = F_yy (x, y)^{-1} F_yx (x, y)
	BaseVector neg_Y_x(m * n);
	Base logdet;
	int signdet = CppAD::LuSolve(m, n, F_yy, F_yx, neg_Y_x, logdet);
	if( signdet == 0 )
		return signdet;

	// compute hes = F_xx (x, y) + F_xy (x, y)  Y^{(1)}(x)
	for(i = 0; i < n; i++)
	{	for(j = 0; j < n; j++)
		{	hes[i * n + j] = F_hes[ i*(n+m) + j ];
			for(k = 0; k < m; k++)
				hes[i*n+j] -= F_hes[i*(n+m) + k+n] * neg_Y_x[k*n+j];
		}
	}
	return signdet;
}

} // END_CPPAD_NAMESPACE

# endif
