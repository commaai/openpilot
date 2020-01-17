# ifndef CPPAD_CORE_CAPACITY_ORDER_HPP
# define CPPAD_CORE_CAPACITY_ORDER_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin capacity_order$$
$spell
	var
	taylor_
	xq
	yq
$$


$section Controlling Taylor Coefficients Memory Allocation$$
$mindex Forward capacity_order control$$

$head Syntax$$
$icode%f%.capacity_order(%c%)%$$

$subhead See Also$$
$cref seq_property$$

$head Purpose$$
The Taylor coefficients calculated by $cref Forward$$ mode calculations
are retained in an $cref ADFun$$ object for subsequent use during
$cref Reverse$$ mode and higher order Forward mode calculations.
For example, a call to $cref/Forward/forward_order/$$ with the syntax
$codei%
        %yq% = %f%.Forward(%q%, %xq%)
%$$
where $icode%q% > 0%$$ and  $code%xq%.size() == %f%.Domain()%$$,
uses the lower order Taylor coefficients and
computes the $th q$$ order Taylor coefficients for all
the variables in the operation sequence corresponding to $icode f$$.
The $code capacity_order$$ operation allows you to control that
amount of memory that is retained by an AD function object
(to hold $code Forward$$ results for subsequent calculations).

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$

$head c$$
The argument $icode c$$ has prototype
$codei%
	size_t %c%
%$$
It specifies the number of Taylor coefficient orders that are allocated
in the AD operation sequence corresponding to $icode f$$.

$subhead Pre-Allocating Memory$$
If you plan to make calls to $code Forward$$ with the maximum value of
$icode q$$ equal to $icode Q$$,
it should be faster to pre-allocate memory for these calls using
$codei%
	%f%.capacity_order(%c%)
%$$
with $icode c$$ equal to $latex Q + 1$$.
If you do no do this, $code Forward$$ will automatically allocate memory
and will copy the results to a larger buffer, when necessary.
$pre

$$
Note that each call to $cref Dependent$$ frees the old memory
connected to the function object and sets the corresponding
taylor capacity to zero.

$subhead Freeing Memory$$
If you no longer need the Taylor coefficients of order $icode q$$
and higher (that are stored in $icode f$$),
you can reduce the memory allocated to $icode f$$ using
$codei%
	%f%.capacity_order(%c%)
%$$
with $icode c$$ equal to $icode q$$.
Note that, if $cref ta_hold_memory$$ is true, this memory is not actually
returned to the system, but rather held for future use by the same thread.

$head Original State$$
If $icode f$$ is $cref/constructed/FunConstruct/$$ with the syntax
$codei%
	ADFun<%Base%> %f%(%x%, %y%)
%$$,
there is an implicit call to $cref forward_zero$$ with $icode xq$$ equal to
the value of the
$cref/independent variables/glossary/Tape/Independent Variable/$$
when the AD operation sequence was recorded.
This corresponds to $icode%c% == 1%$$.

$children%
	example/general/capacity_order.cpp
%$$
$head Example$$
The file
$cref capacity_order.cpp$$
contains an example and test of these operations.
It returns true if it succeeds and false otherwise.

$end
-----------------------------------------------------------------------------
*/

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file capacity_order.hpp
Control of number of orders allocated.
\}
*/

/*!
Control of number of orders and directions allocated.

\tparam Base
The type used during the forward mode computations; i.e., the corresponding
recording of operations used the type AD<Base>.

\param c
is the number of orders to allocate memory for.
If <code>c == 0</code> then \c r must also be zero.
In this case num_order_taylor_, cap_order_taylor_, and num_direction_taylor_
are all set to zero.
In addition, taylor_.free() is called.

\param r
is the number of directions to allocate memory for.
If <code>c == 1</code> then \c r must also be one.
In all cases, it must hold that
<code>
	r == num_direction_taylor_ || num_order_taylor <= 1
</code>
Upon return, num_direction_taylor_ is equal to r.

\par num_order_taylor_
The output value of num_order_taylor_ is the mininumum of its input
value and c. This minimum is the number of orders that are copied to the
new taylor coefficient buffer.

\par num_direction_taylor_
The output value of num_direction_taylor_ is equal to \c r.
*/

template <typename Base>
void ADFun<Base>::capacity_order(size_t c, size_t r)
{	// temporary indices
	size_t i, k, ell;

	if( (c == cap_order_taylor_) & (r == num_direction_taylor_) )
		return;

	if( c == 0 )
	{	CPPAD_ASSERT_UNKNOWN( r == 0 );
		taylor_.free();
		num_order_taylor_     = 0;
		cap_order_taylor_     = 0;
		num_direction_taylor_ = r;
		return;
	}
	CPPAD_ASSERT_UNKNOWN(r==num_direction_taylor_ || num_order_taylor_<=1);

	// Allocate new taylor with requested number of orders and directions
	size_t new_len   = ( (c-1)*r + 1 ) * num_var_tape_;
	local::pod_vector<Base> new_taylor;
	new_taylor.extend(new_len);

	// number of orders to copy
	size_t p = std::min(num_order_taylor_, c);
	if( p > 0 )
	{
		// old order capacity
		size_t C = cap_order_taylor_;

		// old number of directions
		size_t R = num_direction_taylor_;

		// copy the old data into the new matrix
		CPPAD_ASSERT_UNKNOWN( p == 1 || r == R );
		for(i = 0; i < num_var_tape_; i++)
		{	// copy zero order
			size_t old_index = ((C-1) * R + 1) * i + 0;
			size_t new_index = ((c-1) * r + 1) * i + 0;
			new_taylor[ new_index ] = taylor_[ old_index ];
			// copy higher orders
			for(k = 1; k < p; k++)
			{	for(ell = 0; ell < R; ell++)
				{	old_index = ((C-1) * R + 1) * i + (k-1) * R + ell + 1;
					new_index = ((c-1) * r + 1) * i + (k-1) * r + ell + 1;
					new_taylor[ new_index ] = taylor_[ old_index ];
				}
			}
		}
	}

	// replace taylor_ by new_taylor
	taylor_.swap(new_taylor);
	cap_order_taylor_     = c;
	num_order_taylor_     = p;
	num_direction_taylor_ = r;

	// note that the destructor for new_taylor will free the old taylor memory
	return;
}

/*!
User API control of number of orders allocated.

\tparam Base
The type used during the forward mode computations; i.e., the corresponding
recording of operations used the type AD<Base>.

\param c
is the number of orders to allocate memory for.
If <code>c == 0</code>,
num_order_taylor_, cap_order_taylor_, and num_direction_taylor_
are all set to zero.
In addition, taylor_.free() is called.

\par num_order_taylor_
The output value of num_order_taylor_ is the mininumum of its input
value and c. This minimum is the number of orders that are copied to the
new taylor coefficient buffer.

\par num_direction_taylor_
If \c is zero (one), \c num_direction_taylor_ is set to zero (one).
Otherwise, if \c num_direction_taylor_ is zero, it is set to one.
Othwerwise, \c num_direction_taylor_ is not modified.
*/

template <typename Base>
void ADFun<Base>::capacity_order(size_t c)
{	size_t r;
	if( (c == 0) | (c == 1) )
	{	r = c;
		capacity_order(c, r);
		return;
	}
	r = num_direction_taylor_;
	if( r == 0 )
		r = 1;
	capacity_order(c, r);
	return;
}

} // END CppAD namespace


# endif
