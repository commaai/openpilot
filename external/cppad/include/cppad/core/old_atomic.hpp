# ifndef CPPAD_CORE_OLD_ATOMIC_HPP
# define CPPAD_CORE_OLD_ATOMIC_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin old_atomic$$
$spell
	hes
	std
	Jacobian
	jac
	Tvector
	afun
	vx
	vy
	bool
	namespace
	CppAD
	const
	Taylor
	tx
	ty
	px
	py
$$

$section User Defined Atomic AD Functions$$
$mindex operation old_atomic$$

$head Deprecated 2013-05-27$$
Using $code CPPAD_USER_ATOMIC$$ has been deprecated.
Use $cref atomic_base$$ instead.

$head Syntax Function$$
$codei%CPPAD_USER_ATOMIC(%afun%, %Tvector%, %Base%,
	%forward%, %reverse%, %for_jac_sparse%, %rev_jac_sparse%, %rev_hes_sparse%
)
%$$

$subhead Use Function$$
$icode%afun%(%id%, %ax%, %ay%)
%$$

$subhead Callback Routines$$
$icode%ok% = %forward%(%id%, %k%, %n%, %m%, %vx%, %vy%, %tx%, %ty%)
%$$
$icode%ok% = %reverse%(%id%, %k%, %n%, %m%, %tx%, %ty%, %px%, %py%)
%$$
$icode%ok% = %for_jac_sparse%(%id%, %n%, %m%, %q%, %r%, %s%)
%$$
$icode%ok% = %rev_jac_sparse%(%id%, %n%, %m%, %q%, %r%, %s%)
%$$
$icode%ok% = %rev_hes_sparse%(%id%, %n%, %m%, %q%, %r%, %s%, %t%, %u%, %v%)
%$$

$subhead Free Static Memory$$
$codei%user_atomic<%Base%>::clear()%$$

$head Purpose$$
In some cases, the user knows how to compute the derivative
of a function
$latex \[
	y = f(x) \; {\rm where} \; f : B^n \rightarrow B^m
\] $$
more efficiently than by coding it using $codei%AD<%Base%>%$$
$cref/atomic/glossary/Operation/Atomic/$$ operations
and letting CppAD do the rest.
In this case, $code CPPAD_USER_ATOMIC$$ can be used
add the user code for $latex f(x)$$, and its derivatives,
to the set of $codei%AD<%Base%>%$$ atomic operations.
$pre

$$
Another possible purpose is to reduce the size of the tape;
see $cref/use AD/old_atomic/Example/Use AD/$$

$head Partial Implementation$$
The routines
$cref/forward/old_atomic/forward/$$,
$cref/reverse/old_atomic/reverse/$$,
$cref/for_jac_sparse/old_atomic/for_jac_sparse/$$,
$cref/rev_jac_sparse/old_atomic/rev_jac_sparse/$$, and
$cref/rev_hes_sparse/old_atomic/rev_hes_sparse/$$,
must be defined by the user.
The $icode forward$$ the routine,
for the case $icode%k% = 0%$$, must be implemented.
Functions with the correct prototype,
that just return $code false$$,
can be used for the other cases
(unless they are required by your calculations).
For example, you need not implement
$icode forward$$ for the case $icode%k% == 2%$$ until you require
forward mode calculation of second derivatives.

$head CPPAD_USER_ATOMIC$$
The macro
$codei%
CPPAD_USER_ATOMIC(%afun%, %Tvector%, %Base%,
	%forward%, %reverse%, %for_jac_sparse%, %rev_jac_sparse%, %rev_hes_sparse%
)
%$$
defines the $codei%AD<%Base%>%$$ routine $icode afun$$.
This macro can be placed within a namespace
(not the $code CppAD$$ namespace)
but must be outside of any routine.

$subhead Tvector$$
The macro argument $icode Tvector$$ must be a
$cref/simple vector template class/SimpleVector/$$.
It determines the type of vectors used as arguments to the routine
$icode afun$$.

$subhead Base$$
The macro argument $icode Base$$ specifies the
$cref/base type/base_require/$$
corresponding to $codei%AD<%Base>%$$ operation sequences.
Calling the routine $icode afun$$ will add the operator defined
by this macro to an $codei%AD<%Base>%$$ operation sequence.

$head ok$$
For all routines documented below,
the return value $icode ok$$ has prototype
$codei%
	bool %ok%
%$$
If it is $code true$$, the corresponding evaluation succeeded,
otherwise it failed.

$head id$$
For all routines documented below,
the argument $icode id$$ has prototype
$codei%
	size_t %id%
%$$
Its value in all other calls is the same as in the corresponding
call to $icode afun$$.
It can be used to store and retrieve extra information about
a specific call to $icode afun$$.

$head k$$
For all routines documented below, the argument $icode k$$ has prototype
$codei%
	size_t %k%
%$$
The value $icode%k%$$ is the order of the Taylor coefficient that
we are evaluating ($cref/forward/old_atomic/forward/$$)
or taking the derivative of ($cref/reverse/old_atomic/reverse/$$).

$head n$$
For all routines documented below,
the argument $icode n$$ has prototype
$codei%
	size_t %n%
%$$
It is the size of the vector $icode ax$$ in the corresponding call to
$icode%afun%(%id%, %ax%, %ay%)%$$; i.e.,
the dimension of the domain space for $latex y = f(x)$$.

$head m$$
For all routines documented below, the argument $icode m$$ has prototype
$codei%
	size_t %m%
%$$
It is the size of the vector $icode ay$$ in the corresponding call to
$icode%afun%(%id%, %ax%, %ay%)%$$; i.e.,
the dimension of the range space for $latex y = f(x)$$.

$head tx$$
For all routines documented below,
the argument $icode tx$$ has prototype
$codei%
	const CppAD::vector<%Base%>& %tx%
%$$
and $icode%tx%.size() >= (%k% + 1) * %n%$$.
For $latex j = 0 , \ldots , n-1$$ and $latex \ell = 0 , \ldots , k$$,
we use the Taylor coefficient notation
$latex \[
\begin{array}{rcl}
	x_j^\ell & = & tx [ j * ( k + 1 ) + \ell ]
	\\
	X_j (t) & = & x_j^0 + x_j^1 t^1 + \cdots + x_j^k t^k
\end{array}
\] $$
If $icode%tx%.size() > (%k% + 1) * %n%$$,
the other components of $icode tx$$ are not specified and should not be used.
Note that superscripts represent an index for $latex x_j^\ell$$
and an exponent for $latex t^\ell$$.
Also note that the Taylor coefficients for $latex X(t)$$ correspond
to the derivatives of $latex X(t)$$ at $latex t = 0$$ in the following way:
$latex \[
	x_j^\ell = \frac{1}{ \ell ! } X_j^{(\ell)} (0)
\] $$

$head ty$$
In calls to $cref/forward/old_atomic/forward/$$,
the argument $icode ty$$ has prototype
$codei%
	CppAD::vector<%Base%>& %ty%
%$$
while in calls to $cref/reverse/old_atomic/reverse/$$ it has prototype
$codei%
	const CppAD::vector<%Base%>& %ty%
%$$
For all calls, $icode%tx%.size() >= (%k% + 1) * %m%$$.
For $latex i = 0 , \ldots , m-1$$ and $latex \ell = 0 , \ldots , k$$,
we use the Taylor coefficient notation
$latex \[
\begin{array}{rcl}
	y_i^\ell & = & ty [ i * ( k + 1 ) + \ell ]
	\\
	Y_i (t)  & = & y_i^0 + y_i^1 t^1 + \cdots + y_i^k t^k + o ( t^k )
\end{array}
\] $$
where $latex o( t^k ) / t^k \rightarrow 0$$ as $latex t \rightarrow 0$$.
If $icode%ty%.size() > (%k% + 1) * %m%$$,
the other components of $icode ty$$ are not specified and should not be used.
Note that superscripts represent an index for $latex y_j^\ell$$
and an exponent for $latex t^\ell$$.
Also note that the Taylor coefficients for $latex Y(t)$$ correspond
to the derivatives of $latex Y(t)$$ at $latex t = 0$$ in the following way:
$latex \[
	y_j^\ell = \frac{1}{ \ell ! } Y_j^{(\ell)} (0)
\] $$

$subhead forward$$
In the case of $icode forward$$,
for $latex i = 0 , \ldots , m-1$$, $latex ty[ i *( k  + 1) + k ]$$ is an output
and all the other components of $icode ty$$ are inputs.

$subhead reverse$$
In the case of $icode reverse$$,
all the components of $icode ty$$ are inputs.

$head afun$$
The macro argument $icode afun$$,
is the name of the AD function corresponding to this atomic
operation (as it is used in the source code).
CppAD uses the other functions,
where the arguments are vectors with elements of type $icode Base$$,
to implement the function
$codei%
	%afun%(%id%, %ax%, %ay%)
%$$
where the argument are vectors with elements of type $codei%AD<%Base%>%$$.

$subhead ax$$
The $icode afun$$ argument $icode ax$$ has prototype
$codei%
	const %Tvector%< AD<%Base%> >& %ax%
%$$
It is the argument vector $latex x \in B^n$$
at which the $codei%AD<%Base%>%$$ version of
$latex y = f(x)$$ is to be evaluated.
The dimension of the domain space for $latex y = f (x)$$
is specified by $cref/n/old_atomic/n/$$ $codei%= %ax%.size()%$$,
which must be greater than zero.

$subhead ay$$
The $icode afun$$ result $icode ay$$ has prototype
$codei%
	%Tvector%< AD<%Base%> >& %ay%
%$$
The input values of its elements
are not specified (must not matter).
Upon return, it is the $codei%AD<%Base%>%$$ version of the
result vector $latex y = f(x)$$.
The dimension of the range space for $latex y = f (x)$$
is specified by $cref/m/old_atomic/m/$$ $codei%= %ay%.size()%$$,
which must be greater than zero.

$subhead Parallel Mode$$
The first call to
$codei%
	%afun%(%id%, %ax%, %ay%)
%$$
must not be in $cref/parallel/ta_in_parallel/$$ mode.
In addition, the
$cref/old_atomic clear/old_atomic/clear/$$
routine cannot be called while in parallel mode.

$head forward$$
The macro argument $icode forward$$ is a
user defined function
$codei%
	%ok% = %forward%(%id%, %k%, %n%, %m%, %vx%, %vy%, %tx%, %ty%)
%$$
that computes results during a $cref/forward/Forward/$$ mode sweep.
For this call, we are given the Taylor coefficients in $icode tx$$
form order zero through $icode k$$,
and the Taylor coefficients in  $icode ty$$ with order less than $icode k$$.
The $icode forward$$ routine computes the
$icode k$$ order Taylor coefficients for $latex y$$ using the definition
$latex Y(t) = f[ X(t) ]$$.
For example, for $latex i = 0 , \ldots , m-1$$,
$latex \[
\begin{array}{rcl}
y_i^0 & = & Y(0)
        = f_i ( x^0 )
\\
y_i^1 & = & Y^{(1)} ( 0 )
        = f_i^{(1)} ( x^0 ) X^{(1)} ( 0 )
        = f_i^{(1)} ( x^0 ) x^1
\\
y_i^2
& = & \frac{1}{2 !} Y^{(2)} (0)
\\
& = & \frac{1}{2} X^{(1)} (0)^\R{T} f_i^{(2)} ( x^0 ) X^{(1)} ( 0 )
  +   \frac{1}{2} f_i^{(1)} ( x^0 ) X^{(2)} ( 0 )
\\
& = & \frac{1}{2} (x^1)^\R{T} f_i^{(2)} ( x^0 ) x^1
  +    f_i^{(1)} ( x^0 ) x^2
\end{array}
\] $$
Then, for $latex i = 0 , \ldots , m-1$$, it sets
$latex \[
	ty [ i * (k + 1) + k ] = y_i^k
\] $$
The other components of $icode ty$$ must be left unchanged.

$subhead Usage$$
This routine is used,
with $icode%vx%.size() > 0%$$ and $icode%k% == 0%$$,
by calls to $icode afun$$.
It is used,
with $icode%vx%.size() = 0%$$ and
$icode k$$ equal to the order of the derivative begin computed,
by calls to $cref/forward/forward_order/$$.

$subhead vx$$
The $icode forward$$ argument $icode vx$$ has prototype
$codei%
	const CppAD::vector<bool>& %vx%
%$$
The case $icode%vx%.size() > 0%$$ occurs
once for each call to $icode afun$$,
during the call,
and before any of the other callbacks corresponding to that call.
Hence such a call can be used to cache information attached to
the corresponding $icode id$$
(such as the elements of $icode vx$$).
If $icode%vx%.size() > 0%$$ then
$icode%k% == 0%$$,
$icode%vx%.size() >= %n%$$, and
for $latex j = 0 , \ldots , n-1$$,
$icode%vx%[%j%]%$$ is true if and only if
$icode%ax%[%j%]%$$ is a $cref/variable/glossary/Variable/$$.
$pre

$$
If $icode%vx%.size() == 0%$$,
then $icode%vy%.size() == 0%$$ and neither of these vectors
should be used.

$subhead vy$$
The $icode forward$$ argument $icode vy$$ has prototype
$codei%
	CppAD::vector<bool>& %vy%
%$$
If $icode%vy%.size() == 0%$$, it should not be used.
Otherwise,
$icode%k% == 0%$$ and $icode%vy%.size() >= %m%$$.
The input values of the elements of $icode vy$$
are not specified (must not matter).
Upon return, for $latex j = 0 , \ldots , m-1$$,
$icode%vy%[%i%]%$$ is true if and only if
$icode%ay%[%j%]%$$ is a variable.
(CppAD uses $icode vy$$ to reduce the necessary computations.)

$head reverse$$
The macro argument $icode reverse$$
is a user defined function
$codei%
	%ok% = %reverse%(%id%, %k%, %n%, %m%, %tx%, %ty%, %px%, %py%)
%$$
that computes results during a $cref/reverse/Reverse/$$ mode sweep.
The input value of the vectors $icode tx$$ and $icode ty$$
contain Taylor coefficient, up to order $icode k$$,
for $latex X(t)$$ and $latex Y(t)$$ respectively.
We use the $latex \{ x_j^\ell \}$$ and $latex \{ y_i^\ell \}$$
to denote these Taylor coefficients where the implicit range indices are
$latex i = 0 , \ldots , m-1$$,
$latex j = 0 , \ldots , n-1$$,
$latex \ell = 0 , \ldots , k$$.
Using the calculations done by $cref/forward/old_atomic/forward/$$,
the Taylor coefficients $latex \{ y_i^\ell \}$$ are a function of the Taylor
coefficients for $latex \{ x_j^\ell \}$$; i.e., given $latex y = f(x)$$
we define the function
$latex F : B^{n \times (k+1)} \rightarrow B^{m \times (k+1)}$$ by
$latex \[
y_i^\ell =  F_i^\ell ( \{ x_j^\ell \} )
\] $$
We use $latex G : B^{m \times (k+1)} \rightarrow B$$
to denote an arbitrary scalar valued function of the Taylor coefficients for
$latex Y(t)$$ and write  $latex z = G( \{ y_i^\ell \} )$$.
The $code reverse$$ routine
is given the derivative of $latex z$$ with respect to
$latex \{ y_i^\ell \}$$ and computes its derivative with respect
to $latex \{ x_j^\ell \}$$.

$subhead Usage$$
This routine is used,
with $icode%k% + 1%$$ equal to the order of the derivative being calculated,
by calls to $cref/reverse/reverse_any/$$.

$subhead py$$
The $icode reverse$$ argument $icode py$$ has prototype
$codei%
	const CppAD::vector<%Base%>& %py%
%$$
and $icode%py%.size() >= (%k% + 1) * %m%$$.
For $latex i = 0 , \ldots , m-1$$ and $latex \ell = 0 , \ldots , k$$,
$latex \[
	py[ i * (k + 1 ) + \ell ] = \partial G / \partial y_i^\ell
\] $$
If $icode%py%.size() > (%k% + 1) * %m%$$,
the other components of $icode py$$ are not specified and should not be used.

$subhead px$$
We define the function
$latex \[
H ( \{ x_j^\ell \} ) = G[ F( \{ x_j^\ell \} ) ]
\] $$
The $icode reverse$$ argument $icode px$$ has prototype
$codei%
	CppAD::vector<%Base%>& %px%
%$$
and $icode%px%.size() >= (%k% + 1) * %n%$$.
The input values of the elements of $icode px$$
are not specified (must not matter).
Upon return,
for $latex j = 0 , \ldots , n-1$$ and $latex p = 0 , \ldots , k$$,
$latex \[
\begin{array}{rcl}
px [ j * (k + 1) + p ] & = & \partial H / \partial x_j^p
\\
& = &
( \partial G / \partial \{ y_i^\ell \} )
	( \partial \{ y_i^\ell \} / \partial x_j^p )
\\
& = &
\sum_{i=0}^{m-1} \sum_{\ell=0}^k
( \partial G / \partial y_i^\ell ) ( \partial y_i^\ell / \partial x_j^p )
\\
& = &
\sum_{i=0}^{m-1} \sum_{\ell=p}^k
py[ i * (k + 1 ) + \ell ] ( \partial F_i^\ell / \partial x_j^p )
\end{array}
\] $$
Note that we have used the fact that for $latex \ell < p$$,
$latex \partial F_i^\ell / \partial x_j^p = 0$$.
If $icode%px%.size() > (%k% + 1) * %n%$$,
the other components of $icode px$$ are not specified and should not be used.

$head for_jac_sparse$$
The macro argument $icode for_jac_sparse$$
is a user defined function
$codei%
	%ok% = %for_jac_sparse%(%id%, %n%, %m%, %q%, %r%, %s%)
%$$
that is used to compute results during a forward Jacobian sparsity sweep.
For a fixed $latex n \times q$$ matrix $latex R$$,
the Jacobian of $latex f( x + R * u)$$ with respect to $latex u \in B^q$$ is
$latex \[
	S(x) = f^{(1)} (x) * R
\] $$
Given a $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for $latex R$$,
$icode for_jac_sparse$$ computes a sparsity pattern for $latex S(x)$$.

$subhead Usage$$
This routine is used by calls to $cref ForSparseJac$$.

$subhead q$$
The $icode for_jac_sparse$$ argument $icode q$$ has prototype
$codei%
     size_t %q%
%$$
It specifies the number of columns in
$latex R \in B^{n \times q}$$ and the Jacobian
$latex S(x) \in B^{m \times q}$$.

$subhead r$$
The $icode for_jac_sparse$$ argument $icode r$$ has prototype
$codei%
     const CppAD::vector< std::set<size_t> >& %r%
%$$
and $icode%r%.size() >= %n%$$.
For $latex j = 0 , \ldots , n-1$$,
all the elements of $icode%r%[%j%]%$$ are between
zero and $icode%q%-1%$$ inclusive.
This specifies a sparsity pattern for the matrix $latex R$$.

$subhead s$$
The $icode for_jac_sparse$$ return value $icode s$$ has prototype
$codei%
	CppAD::vector< std::set<size_t> >& %s%
%$$
and $icode%s%.size() >= %m%%$$.
The input values of its sets
are not specified (must not matter). Upon return
for $latex i = 0 , \ldots , m-1$$,
all the elements of $icode%s%[%i%]%$$ are between
zero and $icode%q%-1%$$ inclusive.
This represents a sparsity pattern for the matrix $latex S(x)$$.

$head rev_jac_sparse$$
The macro argument $icode rev_jac_sparse$$
is a user defined function
$codei%
	%ok% = %rev_jac_sparse%(%id%, %n%, %m%, %q%, %r%, %s%)
%$$
that is used to compute results during a reverse Jacobian sparsity sweep.
For a fixed $latex q \times m$$ matrix $latex S$$,
the Jacobian of $latex S * f( x )$$ with respect to $latex x \in B^n$$ is
$latex \[
	R(x) = S * f^{(1)} (x)
\] $$
Given a $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for $latex S$$,
$icode rev_jac_sparse$$ computes a sparsity pattern for $latex R(x)$$.

$subhead Usage$$
This routine is used by calls to $cref RevSparseJac$$
and to $cref optimize$$.


$subhead q$$
The $icode rev_jac_sparse$$ argument $icode q$$ has prototype
$codei%
     size_t %q%
%$$
It specifies the number of rows in
$latex S \in B^{q \times m}$$ and the Jacobian
$latex R(x) \in B^{q \times n}$$.

$subhead s$$
The $icode rev_jac_sparse$$ argument $icode s$$ has prototype
$codei%
     const CppAD::vector< std::set<size_t> >& %s%
%$$
and $icode%s%.size() >= %m%$$.
For $latex i = 0 , \ldots , m-1$$,
all the elements of $icode%s%[%i%]%$$
are between zero and $icode%q%-1%$$ inclusive.
This specifies a sparsity pattern for the matrix $latex S^\R{T}$$.

$subhead r$$
The $icode rev_jac_sparse$$ return value $icode r$$ has prototype
$codei%
	CppAD::vector< std::set<size_t> >& %r%
%$$
and $icode%r%.size() >= %n%$$.
The input values of its sets
are not specified (must not matter).
Upon return for $latex j = 0 , \ldots , n-1$$,
all the elements of $icode%r%[%j%]%$$
are between zero and $icode%q%-1%$$ inclusive.
This represents a sparsity pattern for the matrix $latex R(x)^\R{T}$$.

$head rev_hes_sparse$$
The macro argument $icode rev_hes_sparse$$
is a user defined function
$codei%
	%ok% = %rev_hes_sparse%(%id%, %n%, %m%, %q%, %r%, %s%, %t%, %u%, %v%)
%$$
There is an unspecified scalar valued function
$latex g : B^m \rightarrow B$$.
Given a sparsity pattern for $latex R$$
and information about the function $latex z = g(y)$$,
this routine computes the sparsity pattern for
$latex \[
	V(x) = (g \circ f)^{(2)}( x ) R
\] $$

$subhead Usage$$
This routine is used by calls to $cref RevSparseHes$$.

$subhead q$$
The $icode rev_hes_sparse$$ argument $icode q$$ has prototype
$codei%
     size_t %q%
%$$
It specifies the number of columns in the sparsity patterns.

$subhead r$$
The $icode rev_hes_sparse$$ argument $icode r$$ has prototype
$codei%
     const CppAD::vector< std::set<size_t> >& %r%
%$$
and $icode%r%.size() >= %n%$$.
For $latex j = 0 , \ldots , n-1$$,
all the elements of $icode%r%[%j%]%$$ are between
zero and $icode%q%-1%$$ inclusive.
This specifies a sparsity pattern for the matrix $latex R \in B^{n \times q}$$.

$subhead s$$
The $icode rev_hes_sparse$$ argument $icode s$$ has prototype
$codei%
     const CppAD::vector<bool>& %s%
%$$
and $icode%s%.size() >= %m%$$.
This specifies a sparsity pattern for the matrix
$latex S(x) = g^{(1)} (y) \in B^{1 \times m}$$.

$subhead t$$
The $icode rev_hes_sparse$$ argument $icode t$$ has prototype
$codei%
     CppAD::vector<bool>& %t%
%$$
and $icode%t%.size() >= %n%$$.
The input values of its elements
are not specified (must not matter).
Upon return it represents a sparsity pattern for the matrix
$latex T(x) \in B^{1 \times n}$$ defined by
$latex \[
T(x)  =  (g \circ f)^{(1)} (x) =  S(x) * f^{(1)} (x)
\] $$

$subhead u$$
The $icode rev_hes_sparse$$ argument $icode u$$ has prototype
$codei%
     const CppAD::vector< std::set<size_t> >& %u%
%$$
and $icode%u%.size() >= %m%$$.
For $latex i = 0 , \ldots , m-1$$,
all the elements of $icode%u%[%i%]%$$
are between zero and $icode%q%-1%$$ inclusive.
This specifies a sparsity pattern
for the matrix $latex U(x) \in B^{m \times q}$$ defined by
$latex \[
\begin{array}{rcl}
U(x)
& = &
\partial_u \{ \partial_y g[ y + f^{(1)} (x) R u ] \}_{u=0}
\\
& = &
\partial_u \{ g^{(1)} [ y + f^{(1)} (x) R u ] \}_{u=0}
\\
& = &
g^{(2)} (y) f^{(1)} (x) R
\end{array}
\] $$

$subhead v$$
The $icode rev_hes_sparse$$ argument $icode v$$ has prototype
$codei%
     CppAD::vector< std::set<size_t> >& %v%
%$$
and $icode%v%.size() >= %n%$$.
The input values of its elements
are not specified (must not matter).
Upon return, for $latex j = 0, \ldots , n-1$$,
all the elements of $icode%v%[%j%]%$$
are between zero and $icode%q%-1%$$ inclusive.
This represents a sparsity pattern for the matrix
$latex V(x) \in B^{n \times q}$$ defined by
$latex \[
\begin{array}{rcl}
V(x)
& = &
\partial_u [ \partial_x (g \circ f) ( x + R u )  ]_{u=0}
\\
& = &
\partial_u [ (g \circ f)^{(1)}( x + R u )  ]_{u=0}
\\
& = &
(g \circ f)^{(2)}( x ) R
\\
& = &
f^{(1)} (x)^\R{T} g^{(2)} ( y ) f^{(1)} (x)  R
+
\sum_{i=1}^m [ g^{(1)} (y) ]_i \; f_i^{(2)} (x) R
\\
& = &
f^{(1)} (x)^\R{T} U(x)
+
\sum_{i=1}^m S(x)_i \; f_i^{(2)} (x) R
\end{array}
\] $$

$head clear$$
User atomic functions hold onto static work space in order to
increase speed by avoiding system memory allocation calls.
The function call $codei%
	user_atomic<%Base%>::clear()
%$$
makes to work space $cref/available/ta_available/$$ to
for other uses by the same thread.
This should be called when you are done using the
user atomic functions for a specific value of $icode Base$$.

$subhead Restriction$$
The user atomic $code clear$$ routine cannot be called
while in $cref/parallel/ta_in_parallel/$$ execution mode.

$children%
	example/deprecated/old_reciprocal.cpp%
	example/deprecated/old_usead_1.cpp%
	example/deprecated/old_usead_2.cpp%
	example/deprecated/old_tan.cpp%
	example/deprecated/old_mat_mul.cpp
%$$
$head Example$$

$subhead Simple$$
The file $cref old_reciprocal.cpp$$ contains the simplest example and test
of a user atomic operation.

$subhead Use AD$$
The examples
$cref old_usead_1.cpp$$ and $cref old_usead_2.cpp$$
use AD to compute the derivatives
inside a user defined atomic function.
This may have the advantage of reducing the size of the tape, because
a repeated section of code would only be taped once.

$subhead Tangent Function$$
The file $cref old_tan.cpp$$ contains an example and test
implementation of the tangent function as a user atomic operation.

$subhead Matrix Multiplication$$
The file  $cref old_mat_mul.cpp$$ contains an example and test
implementation of matrix multiplication a a user atomic operation.

$end
------------------------------------------------------------------------------
*/
# include <set>
# include <cppad/core/cppad_assert.hpp>

// needed before one can use CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# include <cppad/utility/thread_alloc.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file old_atomic.hpp
user defined atomic operations.
*/

/*!
\def CPPAD_USER_ATOMIC(afun, Tvector,
	forward, reverse, for_jac_sparse, rev_jac_sparse, rev_hes_sparse
)
Defines the function <tt>afun(id, ax, ay)</tt>
where \c id is \c ax and \c ay are vectors with <tt>AD<Base></tt> elements.

\par Tvector
the Simple Vector template class for this function.

\par Base
the base type for the atomic operation.

\par afun
name of the CppAD defined function that corresponding to this operation.
Note that \c afun, preceeded by a pound sign,
is a version of \c afun with quotes arround it.

\par forward
name of the user defined function that computes corresponding
results during forward mode.

\par reverse
name of the user defined function that computes corresponding
results during reverse mode.

\par for_jac_sparse
name of the user defined routine that computes corresponding
results during forward mode jacobian sparsity sweeps.

\par rev_jac_sparse
name of the user defined routine that computes corresponding
results during reverse mode jacobian sparsity sweeps.

\par rev_hes_sparse
name of the user defined routine that computes corresponding
results during reverse mode Hessian sparsity sweeps.

\par memory allocation
Note that old_atomic is used as a static object, so its objects
do note get deallocated until the program terminates.
*/
# define CPPAD_USER_ATOMIC(                                           \
     afun            ,                                                \
     Tvector         ,                                                \
     Base            ,                                                \
	forward         ,                                                \
     reverse         ,                                                \
     for_jac_sparse  ,                                                \
     rev_jac_sparse  ,                                                \
     rev_hes_sparse                                                   \
)                                                                     \
inline void afun (                                                    \
     size_t                               id ,                        \
     const Tvector< CppAD::AD<Base> >&    ax ,                        \
     Tvector< CppAD::AD<Base> >&          ay                          \
)                                                                     \
{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;                            \
	static CppAD::old_atomic<Base> fun(                              \
          #afun          ,                                            \
          forward        ,                                            \
          reverse        ,                                            \
          for_jac_sparse ,                                            \
          rev_jac_sparse ,                                            \
          rev_hes_sparse                                              \
     );                                                               \
     fun(id, ax, ay);                                                 \
}

/// link so that user_atomic<Base>::clear() still works
template <class Base> class user_atomic : public atomic_base<Base> {
};

/*!
Class that actually implements the <tt>afun(id, ax, ay)</tt> calls.

A new old_atomic object is generated each time the user invokes
the CPPAD_USER_ATOMIC macro; see static object in that macro.
*/
template <class Base>
class old_atomic : public atomic_base<Base> {
public:
	/// disable old_atomic<Base>::clear(void)
	static void clear(void)
	{	CPPAD_ASSERT_KNOWN(
			false,
			"Depreacted API uses user_atomic<Base>::clear()"
		);
	}
	/// type for user routine that computes forward mode results
	typedef bool (*F) (
		size_t                  id ,
		size_t                   k ,
		size_t                   n ,
		size_t                   m ,
		const vector<bool>&     vx ,
		vector<bool>&           vy ,
		const vector<Base>&     tx ,
		vector<Base>&           ty
	);
	/// type for user routine that computes reverse mode results
	typedef bool (*R) (
		size_t                  id ,
		size_t                   k ,
		size_t                   n ,
		size_t                   m ,
		const vector<Base>&     tx ,
		const vector<Base>&     ty ,
		vector<Base>&           px ,
		const vector<Base>&     py
	);
	/// type for user routine that computes forward mode Jacobian sparsity
	typedef bool (*FJS) (
		size_t                           id ,
		size_t                            n ,
		size_t                            m ,
		size_t                            q ,
		const vector< std::set<size_t> >& r ,
		vector< std::set<size_t>  >&      s
	);
	/// type for user routine that computes reverse mode Jacobian sparsity
	typedef bool (*RJS) (
		size_t                           id ,
		size_t                            n ,
		size_t                            m ,
		size_t                            q ,
		vector< std::set<size_t> >&       r ,
		const vector< std::set<size_t> >& s
	);
	/// type for user routine that computes reverse mode Hessian sparsity
	typedef bool (*RHS) (
		size_t                           id ,
		size_t                            n ,
		size_t                            m ,
		size_t                            q ,
		const vector< std::set<size_t> >& r ,
		const vector<bool>&               s ,
		vector<bool>&                     t ,
		const vector< std::set<size_t> >& u ,
		vector< std::set<size_t> >&       v
	);
private:
	/// id value corresponding to next virtual callback
	size_t                  id_;
	/// user's implementation of forward mode
	const F                  f_;
	/// user's implementation of reverse mode
	const R                  r_;
	/// user's implementation of forward jacobian sparsity calculations
	const FJS              fjs_;
	/// user's implementation of reverse jacobian sparsity calculations
	const RJS              rjs_;
	/// user's implementation of reverse Hessian sparsity calculations
	const RHS              rhs_;

public:
	/*!
	Constructor called for each invocation of CPPAD_USER_ATOMIC.

	Put this object in the list of all objects for this class and set
	the constant private data f_, r_, fjs_, rjs_, rhs_.

	\param afun
	is the user's name for the AD version of this atomic operation.

	\param f
	user routine that does forward mode calculations for this operation.

	\param r
	user routine that does reverse mode calculations for this operation.

	\param fjs
	user routine that does forward Jacobian sparsity calculations.

	\param rjs
	user routine that does reverse Jacobian sparsity calculations.

	\param rhs
	user routine that does reverse Hessian sparsity calculations.

	\par
	This constructor can not be used in parallel mode because
	atomic_base has this restriction.
	*/
	old_atomic(const char* afun, F f, R r, FJS fjs, RJS rjs, RHS rhs) :
	atomic_base<Base>(afun) // name = afun
	, f_(f)
	, r_(r)
	, fjs_(fjs)
	, rjs_(rjs)
	, rhs_(rhs)
	{	this->option( atomic_base<Base>::set_sparsity_enum );
	}
	/*!
	Implement the user call to <tt>afun(id, ax, ay)</tt>.

	\tparam ADVector
	A simple vector class with elements of type <code>AD<Base></code>.

	\param id
	extra information vector that is just passed through by CppAD,
	and possibly used by user's routines.

	\param ax
	is the argument vector for this call,
	<tt>ax.size()</tt> determines the number of arguments.

	\param ay
	is the result vector for this call,
	<tt>ay.size()</tt> determines the number of results.
	*/
	template <class ADVector>
	void operator()(size_t id, const ADVector& ax, ADVector& ay)
	{	// call atomic_base function object
		this->atomic_base<Base>::operator()(ax, ay, id);
		return;
	}
	/*!
	Store id for next virtual function callback

	\param id
	id value corresponding to next virtual callback
	*/
	virtual void set_old(size_t id)
	{	id_ = id; }
	/*!
	Link from old_atomic to forward mode

	\copydetails atomic_base::forward
	*/
	virtual bool forward(
		size_t                    p ,
		size_t                    q ,
		const vector<bool>&      vx ,
		      vector<bool>&      vy ,
		const vector<Base>&      tx ,
		      vector<Base>&      ty )
	{	CPPAD_ASSERT_UNKNOWN( tx.size() % (q+1) == 0 );
		CPPAD_ASSERT_UNKNOWN( ty.size() % (q+1) == 0 );
		size_t n = tx.size() / (q+1);
		size_t m = ty.size() / (q+1);
		size_t i, j, k, ell;

		vector<Base> x(n * (q+1));
		vector<Base> y(m * (q+1));
		vector<bool> empty;

		// old_atomic interface can only handel one order at a time
		// so must just throuh hoops to get multiple orders at one time.
		bool ok = true;
		for(k = p; k <= q; k++)
		{	for(j = 0; j < n; j++)
				for(ell = 0; ell <= k; ell++)
					x[ j * (k+1) + ell ] = tx[ j * (q+1) + ell ];
			for(i = 0; i < m; i++)
				for(ell = 0; ell < k; ell++)
					y[ i * (k+1) + ell ] = ty[ i * (q+1) + ell ];
			if( k == 0 )
				ok &= f_(id_, k, n, m, vx, vy, x, y);
			else
				ok &= f_(id_, k, n, m, empty, empty, x, y);
			for(i = 0; i < m; i++)
				ty[ i * (q+1) + k ] = y[ i * (k+1) + k];
		}
		return ok;
	}
	/*!
	Link from old_atomic to reverse mode

	\copydetails atomic_base::reverse
	*/
	virtual bool reverse(
		size_t                   q ,
		const vector<Base>&     tx ,
		const vector<Base>&     ty ,
		      vector<Base>&     px ,
		const vector<Base>&     py )
	{	CPPAD_ASSERT_UNKNOWN( tx.size() % (q+1) == 0 );
		CPPAD_ASSERT_UNKNOWN( ty.size() % (q+1) == 0 );
		size_t n = tx.size() / (q+1);
		size_t m = ty.size() / (q+1);
		bool   ok = r_(id_, q, n, m, tx, ty, px, py);
		return ok;
	}
	/*!
	Link from forward Jacobian sparsity sweep to old_atomic

	\copydetails atomic_base::for_sparse_jac
	*/
	virtual bool for_sparse_jac(
		size_t                                q ,
		const vector< std::set<size_t> >&     r ,
		      vector< std::set<size_t> >&     s ,
		const vector<Base>&                   x )
	{	size_t n = r.size();
		size_t m = s.size();
		bool ok  = fjs_(id_, n, m, q, r, s);
		return ok;
	}

	/*!
	Link from reverse Jacobian sparsity sweep to old_atomic.

	\copydetails atomic_base::rev_sparse_jac
	*/
	virtual bool rev_sparse_jac(
		size_t                               q  ,
		const vector< std::set<size_t> >&    rt ,
		      vector< std::set<size_t> >&    st ,
		const vector<Base>&                   x )
	{	size_t n = st.size();
		size_t m = rt.size();
		bool ok  = rjs_(id_, n, m, q, st, rt);
		return ok;
	}
	/*!
	Link from reverse Hessian sparsity sweep to old_atomic

	\copydetails atomic_base::rev_sparse_hes
	*/
	virtual bool rev_sparse_hes(
		const vector<bool>&                   vx,
		const vector<bool>&                   s ,
		      vector<bool>&                   t ,
		size_t                                q ,
		const vector< std::set<size_t> >&     r ,
		const vector< std::set<size_t> >&     u ,
		      vector< std::set<size_t> >&     v ,
		const vector<Base>&                   x )
	{	size_t m = u.size();
		size_t n = v.size();
		CPPAD_ASSERT_UNKNOWN( r.size() == n );
		CPPAD_ASSERT_UNKNOWN( s.size() == m );
		CPPAD_ASSERT_UNKNOWN( t.size() == n );
		//
		// old interface used id instead of vx
		bool ok = rhs_(id_, n, m, q, r, s, t, u, v);
		return ok;
	}
};

} // END_CPPAD_NAMESPACE
# endif
