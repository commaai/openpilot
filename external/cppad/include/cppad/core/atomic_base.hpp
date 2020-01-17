# ifndef CPPAD_CORE_ATOMIC_BASE_HPP
# define CPPAD_CORE_ATOMIC_BASE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

# include <set>
# include <cppad/core/cppad_assert.hpp>
# include <cppad/local/sparse_internal.hpp>
// needed before one can use CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# include <cppad/utility/thread_alloc.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file atomic_base.hpp
Base class for atomic user operations.
*/

template <class Base>
class atomic_base {
// ===================================================================
public:
	enum option_enum {
		pack_sparsity_enum   ,
		bool_sparsity_enum   ,
		set_sparsity_enum
	};
private:
	// ------------------------------------------------------
	// constants
	//
	/// index of this object in class_object
	const size_t index_;

	// -----------------------------------------------------
	// variables
	//
	/// sparsity pattern this object is currently using
	/// (set by constructor and option member functions)
	option_enum sparsity_;

	/// temporary work space used by member functions, declared here to avoid
	// memory allocation/deallocation for each usage
	struct work_struct {
		vector<bool>               vx;
		vector<bool>               vy;
		vector<Base>               tx;
		vector<Base>               ty;
		//
		vector<bool>               bool_t;
		//
		vectorBool                 pack_h;
		vectorBool                 pack_r;
		vectorBool                 pack_s;
		vectorBool                 pack_u;
		//
		vector<bool>               bool_h;
		vector<bool>               bool_r;
		vector<bool>               bool_s;
		vector<bool>               bool_u;
		//
		vector< std::set<size_t> > set_h;
		vector< std::set<size_t> > set_r;
		vector< std::set<size_t> > set_s;
		vector< std::set<size_t> > set_u;
	};
	// Use pointers, to avoid false sharing between threads.
	// Not using: vector<work_struct*> work_;
	// so that deprecated atomic examples do not result in a memory leak.
	work_struct* work_[CPPAD_MAX_NUM_THREADS];
	// -----------------------------------------------------
	// static member functions
	//
	/// List of all the object in this class
	static std::vector<atomic_base *>& class_object(void)
	{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;
		static std::vector<atomic_base *> list_;
		return list_;
	}
	/// List of names for each object in this class
	static std::vector<std::string>& class_name(void)
	{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;
		static std::vector<std::string> list_;
		return list_;
	}
	// =====================================================================
public:
	// -----------------------------------------------------
	// member functions not in user API
	//
	/// current sparsity setting
	option_enum sparsity(void) const
	{	return sparsity_; }

	/// Name corresponding to a base_atomic object
	const std::string& afun_name(void) const
	{	return class_name()[index_]; }
/*
$begin atomic_ctor$$
$spell
	enum
	sq
	std
	afun
	arg
	CppAD
	bool
	ctor
	const
	mat_mul_xam.cpp
	hpp
$$

$section Atomic Function Constructor$$

$head Syntax$$
$icode%atomic_user afun%(%ctor_arg_list%)
%$$
$codei%atomic_base<%Base%>(%name%, %sparsity%)
%$$

$head atomic_user$$

$subhead ctor_arg_list$$
Is a list of arguments for the $icode atomic_user$$ constructor.

$subhead afun$$
The object $icode afun$$ must stay in scope for as long
as the corresponding atomic function is used.
This includes use by any $cref/ADFun<Base>/ADFun/$$ that
has this $icode atomic_user$$ operation in its
$cref/operation sequence/glossary/Operation/Sequence/$$.

$subhead Implementation$$
The user defined $icode atomic_user$$ class is a publicly derived class of
$codei%atomic_base<%Base%>%$$.
It should be declared as follows:
$codei%
	class %atomic_user% : public CppAD::atomic_base<%Base%> {
	public:
		%atomic_user%(%ctor_arg_list%) : atomic_base<%Base%>(%name%, %sparsity%)
	%...%
	};
%$$
where $icode ...$$
denotes the rest of the implementation of the derived class.
This includes completing the constructor and
all the virtual functions that have their
$code atomic_base$$ implementations replaced by
$icode atomic_user$$ implementations.

$head atomic_base$$

$subhead Restrictions$$
The $code atomic_base$$ constructor cannot be called in
$cref/parallel/ta_in_parallel/$$ mode.

$subhead Base$$
The template parameter determines the
$icode Base$$ type for this $codei%AD<%Base%>%$$ atomic operation.

$subhead name$$
This $code atomic_base$$ constructor argument has the following prototype
$codei%
	const std::string& %name%
%$$
It is the name for this atomic function and is used for error reporting.
The suggested value for $icode name$$ is $icode afun$$ or $icode atomic_user$$,
i.e., the name of the corresponding atomic object or class.

$subhead sparsity$$
This $code atomic_base$$ constructor argument has prototype
$codei%
	atomic_base<%Base%>::option_enum %sparsity%
%$$
The current $icode sparsity$$ for an $code atomic_base$$ object
determines which type of sparsity patterns it uses
and its value is one of the following:
$table
$icode sparsity$$   $cnext sparsity patterns $rnext
$codei%atomic_base<%Base%>::pack_sparsity_enum%$$ $pre  $$ $cnext
	$cref/vectorBool/CppAD_vector/vectorBool/$$
$rnext
$codei%atomic_base<%Base%>::bool_sparsity_enum%$$ $pre  $$ $cnext
	$cref/vector/CppAD_vector/$$$code <bool>$$
$rnext
$codei%atomic_base<%Base%>::set_sparsity_enum%$$ $pre  $$ $cnext
	$cref/vector/CppAD_vector/$$$code <std::set<std::size_t> >$$
$tend
There is a default value for $icode sparsity$$ if it is not
included in the constructor (which may be either the bool or set option).

$head Example$$

$subhead Define Constructor$$
The following is an example of a user atomic function constructor definitions:
$cref%get_started.cpp%atomic_get_started.cpp%Constructor%$$.

$subhead Use Constructor$$
The following is an example using a user atomic function constructor:
$cref%get_started.cpp%atomic_get_started.cpp%Use Atomic Function%Constructor%$$.

$end
*/
/*!
Base class for atomic_user functions.

\tparam Base
This class is used for defining an AD<Base> atomic operation y = f(x).
*/
/// make sure user does not invoke the default constructor
atomic_base(void)
{	CPPAD_ASSERT_KNOWN(false,
		"Attempt to use the atomic_base default constructor"
	);
}
/*!
Constructor

\param name
name used for error reporting

\param sparsity [in]
what type of sparsity patterns are computed by this function,
bool_sparsity_enum or set_sparsity_enum. Default value is
bool sparsity patterns.
*/
atomic_base(
		const std::string&     name,
		option_enum            sparsity = bool_sparsity_enum
) :
index_   ( class_object().size()  )  ,
sparsity_( sparsity               )
{	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel() ,
		"atomic_base: constructor cannot be called in parallel mode."
	);
	class_object().push_back(this);
	class_name().push_back(name);
	CPPAD_ASSERT_UNKNOWN( class_object().size() == class_name().size() );
	//
	// initialize work pointers as null;
	for(size_t thread = 0; thread < CPPAD_MAX_NUM_THREADS; thread++)
		work_[thread] = CPPAD_NULL;
}
/// destructor informs CppAD that this atomic function with this index
/// has dropped out of scope by setting its pointer to null
virtual ~atomic_base(void)
{	CPPAD_ASSERT_UNKNOWN( class_object().size() > index_ );
	// change object pointer to null, but leave name for error reporting
	class_object()[index_] = CPPAD_NULL;
	//
	// free temporary work memory
	for(size_t thread = 0; thread < CPPAD_MAX_NUM_THREADS; thread++)
		free_work(thread);
}
/// allocates work_ for a specified thread
void allocate_work(size_t thread)
{	if( work_[thread] == CPPAD_NULL )
	{	// allocate the raw memory
		size_t min_bytes = sizeof(work_struct);
		size_t num_bytes;
		void*  v_ptr     = thread_alloc::get_memory(min_bytes, num_bytes);
		// save in work_
		work_[thread]    = reinterpret_cast<work_struct*>( v_ptr );
		// call constructor
		new( work_[thread] ) work_struct;
	}
	return;
}
/// frees work_ for a specified thread
void free_work(size_t thread)
{	if( work_[thread] != CPPAD_NULL )
	{	// call destructor
		work_[thread]->~work_struct();
		// return memory to avialable pool for this thread
        thread_alloc::return_memory( reinterpret_cast<void*>(work_[thread]) );
		// mark this thread as not allocated
		work_[thread] = CPPAD_NULL;
	}
	return;
}
/// atomic_base function object corresponding to a certain index
static atomic_base* class_object(size_t index)
{	CPPAD_ASSERT_UNKNOWN( class_object().size() > index );
	return class_object()[index];
}
/// atomic_base function name corresponding to a certain index
static const std::string& class_name(size_t index)
{	CPPAD_ASSERT_UNKNOWN( class_name().size() > index );
	return class_name()[index];
}
/*
$begin atomic_option$$
$spell
	sq
	enum
	afun
	bool
	CppAD
	std
	typedef
$$

$section Set Atomic Function Options$$

$head Syntax$$
$icode%afun%.option(%option_value%)%$$
These settings do not apply to individual $icode afun$$ calls,
but rather all subsequent uses of the corresponding atomic operation
in an $cref ADFun$$ object.

$head atomic_sparsity$$
Note that, if you use $cref optimize$$, these sparsity patterns are used
to determine the $cref/dependency/dependency.cpp/$$ relationship between
argument and result variables.

$subhead pack_sparsity_enum$$
If $icode option_value$$ is $codei%atomic_base<%Base%>::pack_sparsity_enum%$$,
then the type used by $icode afun$$ for
$cref/sparsity patterns/glossary/Sparsity Pattern/$$,
(after the option is set) will be
$codei%
	typedef CppAD::vectorBool %atomic_sparsity%
%$$
If $icode r$$ is a sparsity pattern
for a matrix $latex R \in B^{p \times q}$$:
$icode%r%.size() == %p% * %q%$$.

$subhead bool_sparsity_enum$$
If $icode option_value$$ is $codei%atomic_base<%Base%>::bool_sparsity_enum%$$,
then the type used by $icode afun$$ for
$cref/sparsity patterns/glossary/Sparsity Pattern/$$,
(after the option is set) will be
$codei%
	typedef CppAD::vector<bool> %atomic_sparsity%
%$$
If $icode r$$ is a sparsity pattern
for a matrix $latex R \in B^{p \times q}$$:
$icode%r%.size() == %p% * %q%$$.

$subhead set_sparsity_enum$$
If $icode option_value$$ is $icode%atomic_base<%Base%>::set_sparsity_enum%$$,
then the type used by $icode afun$$ for
$cref/sparsity patterns/glossary/Sparsity Pattern/$$,
(after the option is set) will be
$codei%
	typedef CppAD::vector< std::set<size_t> > %atomic_sparsity%
%$$
If $icode r$$ is a sparsity pattern
for a matrix $latex R \in B^{p \times q}$$:
$icode%r%.size() == %p%$$, and for $latex i = 0 , \ldots , p-1$$,
the elements of $icode%r%[%i%]%$$ are between zero and $latex q-1$$ inclusive.

$end
*/
void option(enum option_enum option_value)
{	switch( option_value )
	{	case pack_sparsity_enum:
		case bool_sparsity_enum:
		case set_sparsity_enum:
		sparsity_ = option_value;
		break;

		default:
		CPPAD_ASSERT_KNOWN(
			false,
			"atoic_base::option: option_value is not valid"
		);
	}
	return;
}
/*
-----------------------------------------------------------------------------
$begin atomic_afun$$

$spell
	sq
	mul
	afun
	const
	CppAD
	mat_mul.cpp
$$

$section Using AD Version of Atomic Function$$

$head Syntax$$
$icode%afun%(%ax%, %ay%)%$$

$head Purpose$$
Given $icode ax$$,
this call computes the corresponding value of $icode ay$$.
If $codei%AD<%Base%>%$$ operations are being recorded,
it enters the computation as an atomic operation in the recording;
see $cref/start recording/Independent/Start Recording/$$.

$head ADVector$$
The type $icode ADVector$$ must be a
$cref/simple vector class/SimpleVector/$$ with elements of type
$codei%AD<%Base%>%$$; see $cref/Base/atomic_ctor/atomic_base/Base/$$.

$head afun$$
is a $cref/atomic_user/atomic_ctor/atomic_user/$$ object
and this $icode afun$$ function call is implemented by the
$cref/atomic_base/atomic_ctor/atomic_base/$$ class.

$head ax$$
This argument has prototype
$codei%
	const %ADVector%& %ax%
%$$
and size must be equal to $icode n$$.
It specifies vector $latex x \in B^n$$
at which an $codei%AD<%Base%>%$$ version of
$latex y = f(x)$$ is to be evaluated; see
$cref/Base/atomic_ctor/atomic_base/Base/$$.

$head ay$$
This argument has prototype
$codei%
	%ADVector%& %ay%
%$$
and size must be equal to $icode m$$.
The input values of its elements
are not specified (must not matter).
Upon return, it is an $codei%AD<%Base%>%$$ version of
$latex y = f(x)$$.

$head Examples$$
The following files contain example uses of
the AD version of atomic functions during recording:
$cref%get_started.cpp%atomic_get_started.cpp%Use Atomic Function%Recording%$$,
$cref%norm_sq.cpp%atomic_norm_sq.cpp%Use Atomic Function%Recording%$$,
$cref%reciprocal.cpp%atomic_reciprocal.cpp%Use Atomic Function%Recording%$$,
$cref%tangent.cpp%atomic_tangent.cpp%Use Atomic Function%Recording%$$,
$cref%mat_mul.cpp%atomic_mat_mul.cpp%Use Atomic Function%Recording%$$.

$end
-----------------------------------------------------------------------------
*/
/*!
Implement the user call to <tt>afun(ax, ay)</tt> and old_atomic call to
<tt>afun(ax, ay, id)</tt>.

\tparam ADVector
A simple vector class with elements of type <code>AD<Base></code>.

\param id
optional extra information vector that is just passed through by CppAD,
and used by old_atomic derived class (not other derived classes).
This is an extra parameter to the virtual callbacks for old_atomic;
see the set_old member function.

\param ax
is the argument vector for this call,
<tt>ax.size()</tt> determines the number of arguments.

\param ay
is the result vector for this call,
<tt>ay.size()</tt> determines the number of results.
*/
template <class ADVector>
void operator()(
	const ADVector&  ax     ,
	      ADVector&  ay     ,
	size_t           id = 0 )
{	size_t i, j;
	size_t n = ax.size();
	size_t m = ay.size();
# ifndef NDEBUG
	bool ok;
	std::string msg = "atomic_base: " + afun_name() + ".eval: ";
	if( (n == 0) | (m == 0) )
	{	msg += "ax.size() or ay.size() is zero";
		CPPAD_ASSERT_KNOWN(false, msg.c_str() );
	}
# endif
	size_t thread = thread_alloc::thread_num();
	allocate_work(thread);
	vector <Base>& tx  = work_[thread]->tx;
	vector <Base>& ty  = work_[thread]->ty;
	vector <bool>& vx  = work_[thread]->vx;
	vector <bool>& vy  = work_[thread]->vy;
	//
	if( vx.size() != n )
	{	vx.resize(n);
		tx.resize(n);
	}
	if( vy.size() != m )
	{	vy.resize(m);
		ty.resize(m);
	}
	//
	// Determine tape corresponding to variables in ax
	tape_id_t     tape_id  = 0;
	local::ADTape<Base>* tape     = CPPAD_NULL;
	for(j = 0; j < n; j++)
	{	tx[j]  = ax[j].value_;
		vx[j]  = Variable( ax[j] );
		if( vx[j] )
		{
			if( tape_id == 0 )
			{	tape    = ax[j].tape_this();
				tape_id = ax[j].tape_id_;
				CPPAD_ASSERT_UNKNOWN( tape != CPPAD_NULL );
			}
# ifndef NDEBUG
			if( tape_id != ax[j].tape_id_ )
			{	msg += afun_name() +
				": ax contains variables from different threads.";
				CPPAD_ASSERT_KNOWN(false, msg.c_str());
			}
# endif
		}
	}
	// Use zero order forward mode to compute values
	size_t p = 0, q = 0;
	set_old(id);
# ifdef NDEBUG
	forward(p, q, vx, vy, tx, ty);
# else
	ok = forward(p, q, vx, vy, tx, ty);
	if( ! ok )
	{	msg += afun_name() + ": ok is false for "
			"zero order forward mode calculation.";
		CPPAD_ASSERT_KNOWN(false, msg.c_str());
	}
# endif
	bool record_operation = false;
	for(i = 0; i < m; i++)
	{
		// pass back values
		ay[i].value_ = ty[i];

		// initialize entire vector parameters (not in tape)
		ay[i].tape_id_ = 0;
		ay[i].taddr_   = 0;

		// we need to record this operation if
		// any of the eleemnts of ay are variables,
		record_operation |= vy[i];
	}
# ifndef NDEBUG
	if( record_operation & (tape == CPPAD_NULL) )
	{	msg +=
		"all elements of vx are false but vy contains a true element";
		CPPAD_ASSERT_KNOWN(false, msg.c_str() );
	}
# endif
	// if tape is not null, ay is on the tape
	if( record_operation )
	{
		// Operator that marks beginning of this atomic operation
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::UserOp) == 0 );
		CPPAD_ASSERT_UNKNOWN( local::NumArg(local::UserOp) == 4 );
		CPPAD_ASSERT_KNOWN( std::numeric_limits<addr_t>::max() >=
			std::max( std::max( std::max(index_, id), n), m ),
			"atomic_base: cppad_tape_addr_type maximum not large enough"
		);
		tape->Rec_.PutArg(addr_t(index_), addr_t(id), addr_t(n), addr_t(m));
		tape->Rec_.PutOp(local::UserOp);

		// Now put n operators, one for each element of argument vector
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::UsravOp) == 0 );
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::UsrapOp) == 0 );
		CPPAD_ASSERT_UNKNOWN( local::NumArg(local::UsravOp) == 1 );
		CPPAD_ASSERT_UNKNOWN( local::NumArg(local::UsrapOp) == 1 );
		for(j = 0; j < n; j++)
		{	if( vx[j] )
			{	// information for an argument that is a variable
				tape->Rec_.PutArg(ax[j].taddr_);
				tape->Rec_.PutOp(local::UsravOp);
			}
			else
			{	// information for an argument that is parameter
				addr_t par = tape->Rec_.PutPar(ax[j].value_);
				tape->Rec_.PutArg(par);
				tape->Rec_.PutOp(local::UsrapOp);
			}
		}

		// Now put m operators, one for each element of result vector
		CPPAD_ASSERT_UNKNOWN( local::NumArg(local::UsrrpOp) == 1 );
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::UsrrpOp) == 0 );
		CPPAD_ASSERT_UNKNOWN( local::NumArg(local::UsrrvOp) == 0 );
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::UsrrvOp) == 1 );
		for(i = 0; i < m; i++)
		{	if( vy[i] )
			{	ay[i].taddr_    = tape->Rec_.PutOp(local::UsrrvOp);
				ay[i].tape_id_  = tape_id;
			}
			else
			{	addr_t par = tape->Rec_.PutPar(ay[i].value_);
				tape->Rec_.PutArg(par);
				tape->Rec_.PutOp(local::UsrrpOp);
			}
		}

		// Put a duplicate UserOp at end of UserOp sequence
		CPPAD_ASSERT_KNOWN( std::numeric_limits<addr_t>::max() >=
			std::max( std::max( std::max(index_, id), n), m ),
			"atomic_base: cppad_tape_addr_type maximum not large enough"
		);
		tape->Rec_.PutArg(addr_t(index_), addr_t(id), addr_t(n), addr_t(m));
		tape->Rec_.PutOp(local::UserOp);
	}
	return;
}
/*
-----------------------------------------------------------------------------
$begin atomic_forward$$
$spell
	sq
	mul.hpp
	hes
	afun
	vx
	vy
	ty
	Taylor
	const
	CppAD
	bool
$$

$section Atomic Forward Mode$$
$mindex callback virtual$$


$head Syntax$$
$icode%ok% = %afun%.forward(%p%, %q%, %vx%, %vy%, %tx%, %ty%)%$$

$head Purpose$$
This virtual function is used by $cref atomic_afun$$
to evaluate function values.
It is also used buy $cref/forward/Forward/$$
to compute function vales and derivatives.

$head Implementation$$
This virtual function must be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.
It can just return $icode%ok% == false%$$
(and not compute anything) for values
of $icode%q% > 0%$$ that are greater than those used by your
$cref/forward/Forward/$$ mode calculations.

$head p$$
The argument $icode p$$ has prototype
$codei%
	size_t %p%
%$$
It specifies the lowest order Taylor coefficient that we are evaluating.
During calls to $cref atomic_afun$$, $icode%p% == 0%$$.

$head q$$
The argument $icode q$$ has prototype
$codei%
	size_t %q%
%$$
It specifies the highest order Taylor coefficient that we are evaluating.
During calls to $cref atomic_afun$$, $icode%q% == 0%$$.

$head vx$$
The $code forward$$ argument $icode vx$$ has prototype
$codei%
	const CppAD::vector<bool>& %vx%
%$$
The case $icode%vx%.size() > 0%$$ only occurs while evaluating a call to
$cref atomic_afun$$.
In this case,
$icode%p% == %q% == 0%$$,
$icode%vx%.size() == %n%$$, and
for $latex j = 0 , \ldots , n-1$$,
$icode%vx%[%j%]%$$ is true if and only if
$icode%ax%[%j%]%$$ is a $cref/variable/glossary/Variable/$$
in the corresponding call to
$codei%
	%afun%(%ax%, %ay%)
%$$
If $icode%vx%.size() == 0%$$,
then $icode%vy%.size() == 0%$$ and neither of these vectors
should be used.

$head vy$$
The $code forward$$ argument $icode vy$$ has prototype
$codei%
	CppAD::vector<bool>& %vy%
%$$
If $icode%vy%.size() == 0%$$, it should not be used.
Otherwise,
$icode%q% == 0%$$ and $icode%vy%.size() == %m%$$.
The input values of the elements of $icode vy$$
are not specified (must not matter).
Upon return, for $latex j = 0 , \ldots , m-1$$,
$icode%vy%[%i%]%$$ is true if and only if
$icode%ay%[%i%]%$$ is a variable
(CppAD uses $icode vy$$ to reduce the necessary computations).

$head tx$$
The argument $icode tx$$ has prototype
$codei%
	const CppAD::vector<%Base%>& %tx%
%$$
and $icode%tx%.size() == (%q%+1)*%n%$$.
For $latex j = 0 , \ldots , n-1$$ and $latex k = 0 , \ldots , q$$,
we use the Taylor coefficient notation
$latex \[
\begin{array}{rcl}
	x_j^k    & = & tx [ j * ( q + 1 ) + k ]
	\\
	X_j (t)  & = & x_j^0 + x_j^1 t^1 + \cdots + x_j^q t^q
\end{array}
\] $$
Note that superscripts represent an index for $latex x_j^k$$
and an exponent for $latex t^k$$.
Also note that the Taylor coefficients for $latex X(t)$$ correspond
to the derivatives of $latex X(t)$$ at $latex t = 0$$ in the following way:
$latex \[
	x_j^k = \frac{1}{ k ! } X_j^{(k)} (0)
\] $$

$head ty$$
The argument $icode ty$$ has prototype
$codei%
	CppAD::vector<%Base%>& %ty%
%$$
and $icode%tx%.size() == (%q%+1)*%m%$$.
Upon return,
For $latex i = 0 , \ldots , m-1$$ and $latex k = 0 , \ldots , q$$,
$latex \[
\begin{array}{rcl}
	Y_i (t)  & = & f_i [ X(t) ]
	\\
	Y_i (t)  & = & y_i^0 + y_i^1 t^1 + \cdots + y_i^q t^q + o ( t^q )
	\\
	ty [ i * ( q + 1 ) + k ] & = & y_i^k
\end{array}
\] $$
where $latex o( t^q ) / t^q \rightarrow 0$$ as $latex t \rightarrow 0$$.
Note that superscripts represent an index for $latex y_j^k$$
and an exponent for $latex t^k$$.
Also note that the Taylor coefficients for $latex Y(t)$$ correspond
to the derivatives of $latex Y(t)$$ at $latex t = 0$$ in the following way:
$latex \[
	y_j^k = \frac{1}{ k ! } Y_j^{(k)} (0)
\] $$
If $latex p > 0$$,
for $latex i = 0 , \ldots , m-1$$ and $latex k = 0 , \ldots , p-1$$,
the input of $icode ty$$ satisfies
$latex \[
	ty [ i * ( q + 1 ) + k ] = y_i^k
\]$$
and hence the corresponding elements need not be recalculated.

$head ok$$
If the required results are calculated, $icode ok$$ should be true.
Otherwise, it should be false.

$head Discussion$$
For example, suppose that $icode%q% == 2%$$,
and you know how to compute the function $latex f(x)$$,
its first derivative $latex f^{(1)} (x)$$,
and it component wise Hessian $latex f_i^{(2)} (x)$$.
Then you can compute $icode ty$$ using the following formulas:
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
For $latex i = 0 , \ldots , m-1$$, and $latex k = 0 , 1 , 2$$,
$latex \[
	ty [ i * (q + 1) + k ] = y_i^k
\] $$

$children%
	example/atomic/forward.cpp
%$$
$head Examples$$
The file $cref atomic_forward.cpp$$ contains an example and test
that uses this routine.
It returns true if the test passes and false if it fails.

$end
-----------------------------------------------------------------------------
*/
/*!
Link from atomic_base to forward mode

\param p [in]
lowerest order for this forward mode calculation.

\param q [in]
highest order for this forward mode calculation.

\param vx [in]
if size not zero, which components of \c x are variables

\param vy [out]
if size not zero, which components of \c y are variables

\param tx [in]
Taylor coefficients corresponding to \c x for this calculation.

\param ty [out]
Taylor coefficient corresponding to \c y for this calculation

See the forward mode in user's documentation for base_atomic
*/
virtual bool forward(
	size_t                    p  ,
	size_t                    q  ,
	const vector<bool>&       vx ,
	      vector<bool>&       vy ,
	const vector<Base>&       tx ,
	      vector<Base>&       ty )
{	return false; }
/*
-----------------------------------------------------------------------------
$begin atomic_reverse$$
$spell
	sq
	mul.hpp
	afun
	ty
	px
	py
	Taylor
	const
	CppAD
$$

$section Atomic Reverse Mode$$
$spell
	bool
$$

$head Syntax$$
$icode%ok% = %afun%.reverse(%q%, %tx%, %ty%, %px%, %py%)%$$

$head Purpose$$
This function is used by $cref/reverse/Reverse/$$
to compute derivatives.

$head Implementation$$
If you are using
$cref/reverse/Reverse/$$ mode,
this virtual function must be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.
It can just return $icode%ok% == false%$$
(and not compute anything) for values
of $icode q$$ that are greater than those used by your
$cref/reverse/Reverse/$$ mode calculations.

$head q$$
The argument $icode q$$ has prototype
$codei%
	size_t %q%
%$$
It specifies the highest order Taylor coefficient that
computing the derivative of.

$head tx$$
The argument $icode tx$$ has prototype
$codei%
	const CppAD::vector<%Base%>& %tx%
%$$
and $icode%tx%.size() == (%q%+1)*%n%$$.
For $latex j = 0 , \ldots , n-1$$ and $latex k = 0 , \ldots , q$$,
we use the Taylor coefficient notation
$latex \[
\begin{array}{rcl}
	x_j^k    & = & tx [ j * ( q + 1 ) + k ]
	\\
	X_j (t)  & = & x_j^0 + x_j^1 t^1 + \cdots + x_j^q t^q
\end{array}
\] $$
Note that superscripts represent an index for $latex x_j^k$$
and an exponent for $latex t^k$$.
Also note that the Taylor coefficients for $latex X(t)$$ correspond
to the derivatives of $latex X(t)$$ at $latex t = 0$$ in the following way:
$latex \[
	x_j^k = \frac{1}{ k ! } X_j^{(k)} (0)
\] $$

$head ty$$
The argument $icode ty$$ has prototype
$codei%
	const CppAD::vector<%Base%>& %ty%
%$$
and $icode%tx%.size() == (%q%+1)*%m%$$.
For $latex i = 0 , \ldots , m-1$$ and $latex k = 0 , \ldots , q$$,
we use the Taylor coefficient notation
$latex \[
\begin{array}{rcl}
	Y_i (t)  & = & f_i [ X(t) ]
	\\
	Y_i (t)  & = & y_i^0 + y_i^1 t^1 + \cdots + y_i^q t^q + o ( t^q )
	\\
	y_i^k    & = & ty [ i * ( q + 1 ) + k ]
\end{array}
\] $$
where $latex o( t^q ) / t^q \rightarrow 0$$ as $latex t \rightarrow 0$$.
Note that superscripts represent an index for $latex y_j^k$$
and an exponent for $latex t^k$$.
Also note that the Taylor coefficients for $latex Y(t)$$ correspond
to the derivatives of $latex Y(t)$$ at $latex t = 0$$ in the following way:
$latex \[
	y_j^k = \frac{1}{ k ! } Y_j^{(k)} (0)
\] $$


$head F$$
We use the notation $latex \{ x_j^k \} \in B^{n \times (q+1)}$$ for
$latex \[
	\{ x_j^k \W{:} j = 0 , \ldots , n-1, k = 0 , \ldots , q \}
\]$$
We use the notation $latex \{ y_i^k \} \in B^{m \times (q+1)}$$ for
$latex \[
	\{ y_i^k \W{:} i = 0 , \ldots , m-1, k = 0 , \ldots , q \}
\]$$
We define the function
$latex F : B^{n \times (q+1)} \rightarrow B^{m \times (q+1)}$$ by
$latex \[
	y_i^k = F_i^k [ \{ x_j^k \} ]
\] $$
Note that
$latex \[
	F_i^0 ( \{ x_j^k \} ) = f_i ( X(0) )  = f_i ( x^0 )
\] $$
We also note that
$latex F_i^\ell ( \{ x_j^k \} )$$ is a function of
$latex x^0 , \ldots , x^\ell$$
and is determined by the derivatives of $latex f_i (x)$$
up to order $latex \ell$$.


$head G, H$$
We use $latex G : B^{m \times (q+1)} \rightarrow B$$
to denote an arbitrary scalar valued function of $latex \{ y_i^k \}$$.
We use $latex H : B^{n \times (q+1)} \rightarrow B$$
defined by
$latex \[
	H ( \{ x_j^k \} ) = G[ F( \{ x_j^k \} ) ]
\] $$

$head py$$
The argument $icode py$$ has prototype
$codei%
	const CppAD::vector<%Base%>& %py%
%$$
and $icode%py%.size() == m * (%q%+1)%$$.
For $latex i = 0 , \ldots , m-1$$, $latex k = 0 , \ldots , q$$,
$latex \[
	py[ i * (q + 1 ) + k ] = \partial G / \partial y_i^k
\] $$

$subhead px$$
The $icode px$$ has prototype
$codei%
	CppAD::vector<%Base%>& %px%
%$$
and $icode%px%.size() == n * (%q%+1)%$$.
The input values of the elements of $icode px$$
are not specified (must not matter).
Upon return,
for $latex j = 0 , \ldots , n-1$$ and $latex \ell = 0 , \ldots , q$$,
$latex \[
\begin{array}{rcl}
px [ j * (q + 1) + \ell ] & = & \partial H / \partial x_j^\ell
\\
& = &
( \partial G / \partial \{ y_i^k \} ) \cdot
	( \partial \{ y_i^k \} / \partial x_j^\ell )
\\
& = &
\sum_{k=0}^q
\sum_{i=0}^{m-1}
( \partial G / \partial y_i^k ) ( \partial y_i^k / \partial x_j^\ell )
\\
& = &
\sum_{k=\ell}^q
\sum_{i=0}^{m-1}
py[ i * (q + 1 ) + k ] ( \partial F_i^k / \partial x_j^\ell )
\end{array}
\] $$
Note that we have used the fact that for $latex k < \ell$$,
$latex \partial F_i^k / \partial x_j^\ell = 0$$.

$head ok$$
The return value $icode ok$$ has prototype
$codei%
	bool %ok%
%$$
If it is $code true$$, the corresponding evaluation succeeded,
otherwise it failed.

$children%
	example/atomic/reverse.cpp
%$$
$head Examples$$
The file $cref atomic_forward.cpp$$ contains an example and test
that uses this routine.
It returns true if the test passes and false if it fails.

$end
-----------------------------------------------------------------------------
*/
/*!
Link from reverse mode sweep to users routine.

\param q [in]
highest order for this reverse mode calculation.

\param tx [in]
Taylor coefficients corresponding to \c x for this calculation.

\param ty [in]
Taylor coefficient corresponding to \c y for this calculation

\param px [out]
Partials w.r.t. the \c x Taylor coefficients.

\param py [in]
Partials w.r.t. the \c y Taylor coefficients.

See atomic_reverse mode use documentation
*/
virtual bool reverse(
	size_t                    q  ,
	const vector<Base>&       tx ,
	const vector<Base>&       ty ,
	      vector<Base>&       px ,
	const vector<Base>&       py )
{	return false; }
/*
-------------------------------------- ---------------------------------------
$begin atomic_for_sparse_jac$$
$spell
	sq
	mul.hpp
	afun
	Jacobian
	jac
	const
	CppAD
	std
	bool
	std
$$

$section Atomic Forward Jacobian Sparsity Patterns$$

$head Syntax$$
$icode%ok% = %afun%.for_sparse_jac(%q%, %r%, %s%, %x%)
%$$

$head Deprecated 2016-06-27$$
$icode%ok% = %afun%.for_sparse_jac(%q%, %r%, %s%)
%$$

$head Purpose$$
This function is used by $cref ForSparseJac$$ to compute
Jacobian sparsity patterns.
For a fixed matrix $latex R \in B^{n \times q}$$,
the Jacobian of $latex f( x + R * u)$$ with respect to $latex u \in B^q$$ is
$latex \[
	S(x) = f^{(1)} (x) * R
\] $$
Given a $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for $latex R$$,
$code for_sparse_jac$$ computes a sparsity pattern for $latex S(x)$$.

$head Implementation$$
If you are using
$cref ForSparseJac$$,
$cref ForSparseHes$$, or
$cref RevSparseHes$$,
one of the versions of this
virtual function must be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.

$subhead q$$
The argument $icode q$$ has prototype
$codei%
     size_t %q%
%$$
It specifies the number of columns in
$latex R \in B^{n \times q}$$ and the Jacobian
$latex S(x) \in B^{m \times q}$$.

$subhead r$$
This argument has prototype
$codei%
     const %atomic_sparsity%& %r%
%$$
and is a $cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
$latex R \in B^{n \times q}$$.

$subhead s$$
This argument has prototype
$codei%
	%atomic_sparsity%& %s%
%$$
The input values of its elements
are not specified (must not matter).
Upon return, $icode s$$ is a
$cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
$latex S(x) \in B^{m \times q}$$.

$subhead x$$
$index deprecated$$
The argument has prototype
$codei%
	const CppAD::vector<%Base%>& %x%
%$$
and size is equal to the $icode n$$.
This is the $cref Value$$ value corresponding to the parameters in the
vector $cref/ax/atomic_afun/ax/$$ (when the atomic function was called).
To be specific, if
$codei%
	if( Parameter(%ax%[%i%]) == true )
		%x%[%i%] = Value( %ax%[%i%] );
	else
		%x%[%i%] = CppAD::numeric_limits<%Base%>::quiet_NaN();
%$$
The version of this function with out the $icode x$$ argument is deprecated;
i.e., you should include the argument even if you do not use it.

$head ok$$
The return value $icode ok$$ has prototype
$codei%
	bool %ok%
%$$
If it is $code true$$, the corresponding evaluation succeeded,
otherwise it failed.

$children%
	example/atomic/for_sparse_jac.cpp
%$$
$head Examples$$
The file $cref atomic_for_sparse_jac.cpp$$ contains an example and test
that uses this routine.
It returns true if the test passes and false if it fails.

$end
-----------------------------------------------------------------------------
*/
/*!
Link, after case split, from for_jac_sweep to atomic_base.

\param q
is the column dimension for the Jacobian sparsity partterns.

\param r
is the Jacobian sparsity pattern for the argument vector x

\param s
is the Jacobian sparsity pattern for the result vector y

\param x
is the integer value for x arguments that are parameters.
*/
virtual bool for_sparse_jac(
	size_t                                  q  ,
	const vector< std::set<size_t> >&       r  ,
	      vector< std::set<size_t> >&       s  ,
	const vector<Base>&                     x  )
{	return false; }
virtual bool for_sparse_jac(
	size_t                                  q  ,
	const vector<bool>&                     r  ,
	      vector<bool>&                     s  ,
	const vector<Base>&                     x  )
{	return false; }
virtual bool for_sparse_jac(
	size_t                                  q  ,
	const vectorBool&                       r  ,
	      vectorBool&                       s  ,
	const vector<Base>&                     x  )
{	return false; }
// deprecated versions
virtual bool for_sparse_jac(
	size_t                                  q  ,
	const vector< std::set<size_t> >&       r  ,
	      vector< std::set<size_t> >&       s  )
{	return false; }
virtual bool for_sparse_jac(
	size_t                                  q  ,
	const vector<bool>&                     r  ,
	      vector<bool>&                     s  )
{	return false; }
virtual bool for_sparse_jac(
	size_t                                  q  ,
	const vectorBool&                       r  ,
	      vectorBool&                       s  )
{	return false; }

/*!
Link, before case split, from for_jac_sweep to atomic_base.

\tparam InternalSparsity
Is the used internaly for sparsity calculations; i.e.,
sparse_pack or sparse_list.

\param x
is parameter arguments to the function, other components are nan.

\param x_index
is the variable index, on the tape, for the arguments to this function.
This size of x_index is n, the number of arguments to this function.

\param y_index
is the variable index, on the tape, for the results for this function.
This size of y_index is m, the number of results for this function.

\param var_sparsity
On input, for j = 0, ... , n-1, the sparsity pattern with index x_index[j],
is the sparsity for the j-th argument to this atomic function.
On input, for i = 0, ... , m-1, the sparsity pattern with index y_index[i],
is empty. On output, it is the sparsity
for the j-th result for this atomic function.
*/
template <class InternalSparsity>
void for_sparse_jac(
	const vector<Base>&        x            ,
	const vector<size_t>&      x_index      ,
	const vector<size_t>&      y_index      ,
	InternalSparsity&          var_sparsity )
{
	// intial results are empty during forward mode
	size_t q           = var_sparsity.end();
	bool   input_empty = true;
	bool   zero_empty  = true;
	bool   transpose   = false;
	size_t m           = y_index.size();
	bool   ok          = false;
	size_t thread      = thread_alloc::thread_num();
	allocate_work(thread);
	//
	std::string msg    = ": atomic_base.for_sparse_jac: returned false";
	if( sparsity_ == pack_sparsity_enum )
	{	vectorBool& pack_r ( work_[thread]->pack_r );
		vectorBool& pack_s ( work_[thread]->pack_s );
		local::get_internal_sparsity(
			transpose, x_index, var_sparsity, pack_r
		);
		//
		pack_s.resize(m * q );
		ok = for_sparse_jac(q, pack_r, pack_s, x);
		if( ! ok )
			ok = for_sparse_jac(q, pack_r, pack_s);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = pack_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, y_index, var_sparsity, pack_s
		);
	}
	else if( sparsity_ == bool_sparsity_enum )
	{	vector<bool>& bool_r ( work_[thread]->bool_r );
		vector<bool>& bool_s ( work_[thread]->bool_s );
		local::get_internal_sparsity(
			transpose, x_index, var_sparsity, bool_r
		);
		bool_s.resize(m * q );
		ok = for_sparse_jac(q, bool_r, bool_s, x);
		if( ! ok )
			ok = for_sparse_jac(q, bool_r, bool_s);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = bool_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, y_index, var_sparsity, bool_s
		);
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( sparsity_ == set_sparsity_enum );
		vector< std::set<size_t> >& set_r ( work_[thread]->set_r );
		vector< std::set<size_t> >& set_s ( work_[thread]->set_s );
		local::get_internal_sparsity(
			transpose, x_index, var_sparsity, set_r
		);
		//
		set_s.resize(m);
		ok = for_sparse_jac(q, set_r, set_s, x);
		if( ! ok )
			ok = for_sparse_jac(q, set_r, set_s);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = set_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, y_index, var_sparsity, set_s
		);
	}
	return;
}
/*
-------------------------------------- ---------------------------------------
$begin atomic_rev_sparse_jac$$
$spell
	sq
	mul.hpp
	rt
	afun
	Jacobian
	jac
	CppAD
	std
	bool
	const
	hes
$$

$section Atomic Reverse Jacobian Sparsity Patterns$$

$head Syntax$$
$icode%ok% = %afun%.rev_sparse_jac(%q%, %rt%, %st%, %x%)
%$$

$head Deprecated 2016-06-27$$
$icode%ok% = %afun%.rev_sparse_jac(%q%, %rt%, %st%)
%$$

$head Purpose$$
This function is used by
$cref RevSparseJac$$ to compute
Jacobian sparsity patterns.
If you are using $cref RevSparseJac$$,
one of the versions of this
virtual function must be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.
$pre

$$
For a fixed matrix $latex R \in B^{q \times m}$$,
the Jacobian of $latex R * f( x )$$ with respect to $latex x \in B^n$$ is
$latex \[
	S(x) = R * f^{(1)} (x)
\] $$
Given a $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for $latex R$$,
$code rev_sparse_jac$$ computes a sparsity pattern for $latex S(x)$$.

$head Implementation$$
If you are using
$cref RevSparseJac$$ or $cref ForSparseHes$$,
this virtual function must be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.

$subhead q$$
The argument $icode q$$ has prototype
$codei%
     size_t %q%
%$$
It specifies the number of rows in
$latex R \in B^{q \times m}$$ and the Jacobian
$latex S(x) \in B^{q \times n}$$.

$subhead rt$$
This argument has prototype
$codei%
     const %atomic_sparsity%& %rt%
%$$
and is a
$cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
$latex R^\R{T} \in B^{m \times q}$$.

$subhead st$$
This argument has prototype
$codei%
	%atomic_sparsity%& %st%
%$$
The input value of its elements
are not specified (must not matter).
Upon return, $icode s$$ is a
$cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
$latex S(x)^\R{T} \in B^{n \times q}$$.

$subhead x$$
$index deprecated$$
The argument has prototype
$codei%
	const CppAD::vector<%Base%>& %x%
%$$
and size is equal to the $icode n$$.
This is the $cref Value$$ corresponding to the parameters in the
vector $cref/ax/atomic_afun/ax/$$ (when the atomic function was called).
To be specific, if
$codei%
	if( Parameter(%ax%[%i%]) == true )
		%x%[%i%] = Value( %ax%[%i%] );
	else
		%x%[%i%] = CppAD::numeric_limits<%Base%>::quiet_NaN();
%$$
The version of this function with out the $icode x$$ argument is deprecated;
i.e., you should include the argument even if you do not use it.

$head ok$$
The return value $icode ok$$ has prototype
$codei%
	bool %ok%
%$$
If it is $code true$$, the corresponding evaluation succeeded,
otherwise it failed.

$children%
	example/atomic/rev_sparse_jac.cpp
%$$
$head Examples$$
The file $cref atomic_rev_sparse_jac.cpp$$ contains an example and test
that uses this routine.
It returns true if the test passes and false if it fails.

$end
-----------------------------------------------------------------------------
*/
/*!
Link, after case split, from rev_jac_sweep to atomic_base

\param q [in]
is the row dimension for the Jacobian sparsity partterns

\param rt [out]
is the tansposed Jacobian sparsity pattern w.r.t to range variables y

\param st [in]
is the tansposed Jacobian sparsity pattern for the argument variables x

\param x
is the integer value for x arguments that are parameters.
*/
virtual bool rev_sparse_jac(
	size_t                                  q  ,
	const vector< std::set<size_t> >&       rt ,
	      vector< std::set<size_t> >&       st ,
	const vector<Base>&                     x  )
{	return false; }
virtual bool rev_sparse_jac(
	size_t                                  q  ,
	const vector<bool>&                     rt ,
	      vector<bool>&                     st ,
	const vector<Base>&                     x  )
{	return false; }
virtual bool rev_sparse_jac(
	size_t                                  q  ,
	const vectorBool&                       rt ,
	      vectorBool&                       st ,
	const vector<Base>&                     x  )
{	return false; }
// deprecated versions
virtual bool rev_sparse_jac(
	size_t                                  q  ,
	const vector< std::set<size_t> >&       rt ,
	      vector< std::set<size_t> >&       st )
{	return false; }
virtual bool rev_sparse_jac(
	size_t                                  q  ,
	const vector<bool>&                     rt ,
	      vector<bool>&                     st )
{	return false; }
virtual bool rev_sparse_jac(
	size_t                                  q  ,
	const vectorBool&                       rt ,
	      vectorBool&                       st )
{	return false; }

/*!
Link, before case split, from rev_jac_sweep to atomic_base.

\tparam InternalSparsity
Is the used internaly for sparsity calculations; i.e.,
sparse_pack or sparse_list.

\param x
is parameter arguments to the function, other components are nan.

\param x_index
is the variable index, on the tape, for the arguments to this function.
This size of x_index is n, the number of arguments to this function.

\param y_index
is the variable index, on the tape, for the results for this function.
This size of y_index is m, the number of results for this function.

\param var_sparsity
On input, for i = 0, ... , m-1, the sparsity pattern with index y_index[i],
is the sparsity for the i-th argument to this atomic function.
On output, for j = 0, ... , n-1, the sparsity pattern with index x_index[j],
the sparsity has been updated to remove y as a function of x.
*/
template <class InternalSparsity>
void rev_sparse_jac(
	const vector<Base>&        x            ,
	const vector<size_t>&      x_index      ,
	const vector<size_t>&      y_index      ,
	InternalSparsity&          var_sparsity )
{
	// initial results may be non-empty during reverse mode
	size_t q           = var_sparsity.end();
	bool   input_empty = false;
	bool   zero_empty  = true;
	bool   transpose   = false;
	size_t n           = x_index.size();
	bool   ok          = false;
	size_t thread      = thread_alloc::thread_num();
	allocate_work(thread);
	//
	std::string msg    = ": atomic_base.rev_sparse_jac: returned false";
	if( sparsity_ == pack_sparsity_enum )
	{	vectorBool& pack_rt ( work_[thread]->pack_r );
		vectorBool& pack_st ( work_[thread]->pack_s );
		local::get_internal_sparsity(
			transpose, y_index, var_sparsity, pack_rt
		);
		//
		pack_st.resize(n * q );
		ok = rev_sparse_jac(q, pack_rt, pack_st, x);
		if( ! ok )
			ok = rev_sparse_jac(q, pack_rt, pack_st);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = pack_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, x_index, var_sparsity, pack_st
		);
	}
	else if( sparsity_ == bool_sparsity_enum )
	{	vector<bool>& bool_rt ( work_[thread]->bool_r );
		vector<bool>& bool_st ( work_[thread]->bool_s );
		local::get_internal_sparsity(
			transpose, y_index, var_sparsity, bool_rt
		);
		bool_st.resize(n * q );
		ok = rev_sparse_jac(q, bool_rt, bool_st, x);
		if( ! ok )
			ok = rev_sparse_jac(q, bool_rt, bool_st);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = bool_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, x_index, var_sparsity, bool_st
		);
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( sparsity_ == set_sparsity_enum );
		vector< std::set<size_t> >& set_rt ( work_[thread]->set_r );
		vector< std::set<size_t> >& set_st ( work_[thread]->set_s );
		local::get_internal_sparsity(
			transpose, y_index, var_sparsity, set_rt
		);
		set_st.resize(n);
		ok = rev_sparse_jac(q, set_rt, set_st, x);
		if( ! ok )
			ok = rev_sparse_jac(q, set_rt, set_st);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = set_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, x_index, var_sparsity, set_st
		);
	}
	return;
}
/*
-------------------------------------- ---------------------------------------
$begin atomic_for_sparse_hes$$
$spell
	sq
	mul.hpp
	vx
	afun
	Jacobian
	jac
	CppAD
	std
	bool
	hes
	const
$$

$section Atomic Forward Hessian Sparsity Patterns$$

$head Syntax$$
$icode%ok% = %afun%.for_sparse_hes(%vx%, %r%, %s%, %h%, %x%)%$$

$head Deprecated 2016-06-27$$
$icode%ok% = %afun%.for_sparse_hes(%vx%, %r%, %s%, %h%)%$$

$head Purpose$$
This function is used by $cref ForSparseHes$$ to compute
Hessian sparsity patterns.
If you are using $cref ForSparseHes$$,
one of the versions of this
virtual function must be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.
$pre

$$
Given a $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for
a diagonal matrix $latex R \in B^{n \times n}$$, and
a row vector $latex S \in B^{1 \times m}$$,
this routine computes the sparsity pattern for
$latex \[
	H(x) = R^\R{T} \cdot (S \cdot f)^{(2)}( x ) \cdot R
\] $$

$head Implementation$$
If you are using and $cref ForSparseHes$$,
this virtual function must be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.

$subhead vx$$
The argument $icode vx$$ has prototype
$codei%
     const CppAD:vector<bool>& %vx%
%$$
$icode%vx%.size() == %n%$$, and
for $latex j = 0 , \ldots , n-1$$,
$icode%vx%[%j%]%$$ is true if and only if
$icode%ax%[%j%]%$$ is a $cref/variable/glossary/Variable/$$
in the corresponding call to
$codei%
	%afun%(%ax%, %ay%)
%$$

$subhead r$$
This argument has prototype
$codei%
     const CppAD:vector<bool>& %r%
%$$
and is a $cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
the diagonal of $latex R \in B^{n \times n}$$.

$subhead s$$
The argument $icode s$$ has prototype
$codei%
     const CppAD:vector<bool>& %s%
%$$
and its size is $icode m$$.
It is a sparsity pattern for $latex S \in B^{1 \times m}$$.

$subhead h$$
This argument has prototype
$codei%
     %atomic_sparsity%& %h%
%$$
The input value of its elements
are not specified (must not matter).
Upon return, $icode h$$ is a
$cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
$latex H(x) \in B^{n \times n}$$ which is defined above.

$subhead x$$
$index deprecated$$
The argument has prototype
$codei%
	const CppAD::vector<%Base%>& %x%
%$$
and size is equal to the $icode n$$.
This is the $cref Value$$ value corresponding to the parameters in the
vector $cref/ax/atomic_afun/ax/$$ (when the atomic function was called).
To be specific, if
$codei%
	if( Parameter(%ax%[%i%]) == true )
		%x%[%i%] = Value( %ax%[%i%] );
	else
		%x%[%i%] = CppAD::numeric_limits<%Base%>::quiet_NaN();
%$$
The version of this function with out the $icode x$$ argument is deprecated;
i.e., you should include the argument even if you do not use it.

$children%
	example/atomic/for_sparse_hes.cpp
%$$
$head Examples$$
The file $cref atomic_for_sparse_hes.cpp$$ contains an example and test
that uses this routine.
It returns true if the test passes and false if it fails.

$end
-----------------------------------------------------------------------------
*/
/*!
Link, after case split, from for_hes_sweep to atomic_base.

\param vx [in]
which componens of x are variables.

\param r [in]
is the forward Jacobian sparsity pattern w.r.t the argument vector x.

\param s [in]
is the reverse Jacobian sparsity pattern w.r.t the result vector y.

\param h [out]
is the Hessian sparsity pattern w.r.t the argument vector x.

\param x
is the integer value of the x arguments that are parameters.
*/
virtual bool for_sparse_hes(
	const vector<bool>&             vx ,
	const vector<bool>&             r  ,
	const vector<bool>&             s  ,
	vector< std::set<size_t> >&     h  ,
	const vector<Base>&             x  )
{	return false; }
virtual bool for_sparse_hes(
	const vector<bool>&             vx ,
	const vector<bool>&             r  ,
	const vector<bool>&             s  ,
	vector<bool>&                   h  ,
	const vector<Base>&             x  )
{	return false; }
virtual bool for_sparse_hes(
	const vector<bool>&             vx ,
	const vector<bool>&             r  ,
	const vector<bool>&             s  ,
	vectorBool&                     h  ,
	const vector<Base>&             x  )
// deprecated
{	return false; }
virtual bool for_sparse_hes(
	const vector<bool>&             vx ,
	const vector<bool>&             r  ,
	const vector<bool>&             s  ,
	vector< std::set<size_t> >&     h  )
{	return false; }
// deprecated versions
virtual bool for_sparse_hes(
	const vector<bool>&             vx ,
	const vector<bool>&             r  ,
	const vector<bool>&             s  ,
	vector<bool>&                   h  )
{	return false; }
virtual bool for_sparse_hes(
	const vector<bool>&             vx ,
	const vector<bool>&             r  ,
	const vector<bool>&             s  ,
	vectorBool&                     h  )
{	return false; }
/*!
Link, before case split, from for_hes_sweep to atomic_base.

\tparam InternalSparsity
Is the used internaly for sparsity calculations; i.e.,
sparse_pack or sparse_list.

\param x
is parameter arguments to the function, other components are nan.

\param x_index
is the variable index, on the tape, for the arguments to this function.
This size of x_index is n, the number of arguments to this function.

\param y_index
is the variable index, on the tape, for the results for this function.
This size of y_index is m, the number of results for this function.

\param for_jac_sparsity
On input, for j = 0, ... , n-1, the sparsity pattern with index x_index[j],
is the forward Jacobian sparsity for the j-th argument to this atomic function.

\param rev_jac_sparsity
On input, for i = 0, ... , m-1, the sparsity pattern with index y_index[i],
is the reverse Jacobian sparsity for the i-th result to this atomic function.
This shows which components of the result affect the function we are
computing the Hessian of.

\param for_hes_sparsity
This is the sparsity pattern for the Hessian. On input, the non-linear
terms in the atomic fuction have not been included. Upon return, they
have been included.
*/
template <class InternalSparsity>
void for_sparse_hes(
	const vector<Base>&        x                ,
	const vector<size_t>&      x_index          ,
	const vector<size_t>&      y_index          ,
	const InternalSparsity&    for_jac_sparsity ,
	const InternalSparsity&    rev_jac_sparsity ,
	InternalSparsity&          for_hes_sparsity )
{	typedef typename InternalSparsity::const_iterator const_iterator;
	CPPAD_ASSERT_UNKNOWN( rev_jac_sparsity.end() == 1 );
	size_t n      = x_index.size();
	size_t m      = y_index.size();
	bool   ok     = false;
	size_t thread = thread_alloc::thread_num();
	allocate_work(thread);
	//
	// vx
	vector<bool> vx(n);
	for(size_t j = 0; j < n; j++)
		vx[j] = x_index[j] != 0;
	//
	// bool_r
	vector<bool>& bool_r( work_[thread]->bool_r );
	bool_r.resize(n);
	for(size_t j = 0; j < n; j++)
	{	// check if we must compute row and column j of h
		const_iterator itr(for_jac_sparsity, x_index[j]);
		size_t i = *itr;
		bool_r[j] = i < for_jac_sparsity.end();
	}
	//
	// bool s
	vector<bool>& bool_s( work_[thread]->bool_s );
	bool_s.resize(m);
	for(size_t i = 0; i < m; i++)
	{	// check if row i of result is included in h
		bool_s[i] = rev_jac_sparsity.is_element(y_index[i], 0);
	}
	//
	// h
	vectorBool&                 pack_h( work_[thread]->pack_h );
	vector<bool>&               bool_h( work_[thread]->bool_h );
	vector< std::set<size_t> >& set_h(  work_[thread]->set_h );
	//
	// call user's version of atomic function
	std::string msg    = ": atomic_base.for_sparse_hes: returned false";
	if( sparsity_ == pack_sparsity_enum )
	{	pack_h.resize(n * n);
		ok = for_sparse_hes(vx, bool_r, bool_s, pack_h, x);
		if( ! ok )
			ok = for_sparse_hes(vx, bool_r, bool_s, pack_h);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = pack_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
	}
	else if( sparsity_ == bool_sparsity_enum )
	{	bool_h.resize(n * n);
		ok = for_sparse_hes(vx, bool_r, bool_s, bool_h, x);
		if( ! ok )
			ok = for_sparse_hes(vx, bool_r, bool_s, bool_h);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = bool_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( sparsity_ == set_sparsity_enum )
		set_h.resize(n);
		ok = for_sparse_hes(vx, bool_r, bool_s, set_h, x);
		if( ! ok )
			ok = for_sparse_hes(vx, bool_r, bool_s, set_h);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = set_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
	}
	CPPAD_ASSERT_UNKNOWN( ok );
	//
	// modify hessian in calling routine
	for(size_t i = 0; i < n; i++)
	{	for(size_t j = 0; j < n; j++)
		{	if( (x_index[i] > 0) & (x_index[j] > 0) )
			{	bool flag = false;
				switch( sparsity_ )
				{	case pack_sparsity_enum:
					flag = pack_h[i * n + j];
					break;
					//
					case bool_sparsity_enum:
					flag = bool_h[i * n + j];
					break;
					//
					case set_sparsity_enum:
					flag = set_h[i].find(j) != set_h[i].end();
					break;
				}
				if( flag )
				{	const_iterator itr_i(for_jac_sparsity, x_index[i]);
					size_t i_x = *itr_i;
					while( i_x < for_jac_sparsity.end() )
					{	for_hes_sparsity.binary_union(
							i_x, i_x, x_index[j], for_jac_sparsity
						);
						i_x = *(++itr_i);
					}
					const_iterator itr_j(for_jac_sparsity, x_index[j]);
					size_t j_x = *itr_j;
					while( j_x < for_jac_sparsity.end() )
					{	for_hes_sparsity.binary_union(
							j_x, j_x, x_index[i], for_jac_sparsity
						);
						j_x = *(++itr_j);
					}
				}
			}
		}
	}
	return;
}
/*
-------------------------------------- ---------------------------------------
$begin atomic_rev_sparse_hes$$
$spell
	sq
	mul.hpp
	vx
	afun
	Jacobian
	jac
	CppAD
	std
	bool
	hes
	const
$$

$section Atomic Reverse Hessian Sparsity Patterns$$

$head Syntax$$
$icode%ok% = %afun%.rev_sparse_hes(%vx%, %s%, %t%, %q%, %r%, %u%, %v%, %x%)%$$

$head Deprecated 2016-06-27$$
$icode%ok% = %afun%.rev_sparse_hes(%vx%, %s%, %t%, %q%, %r%, %u%, %v%)%$$

$head Purpose$$
This function is used by $cref RevSparseHes$$ to compute
Hessian sparsity patterns.
If you are using $cref RevSparseHes$$ to compute
one of the versions of this
virtual function muse be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.
$pre

$$
There is an unspecified scalar valued function
$latex g : B^m \rightarrow B$$.
Given a $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for
$latex R \in B^{n \times q}$$,
and information about the function $latex z = g(y)$$,
this routine computes the sparsity pattern for
$latex \[
	V(x) = (g \circ f)^{(2)}( x ) R
\] $$

$head Implementation$$
If you are using and $cref RevSparseHes$$,
this virtual function must be defined by the
$cref/atomic_user/atomic_ctor/atomic_user/$$ class.

$subhead vx$$
The argument $icode vx$$ has prototype
$codei%
     const CppAD:vector<bool>& %vx%
%$$
$icode%vx%.size() == %n%$$, and
for $latex j = 0 , \ldots , n-1$$,
$icode%vx%[%j%]%$$ is true if and only if
$icode%ax%[%j%]%$$ is a $cref/variable/glossary/Variable/$$
in the corresponding call to
$codei%
	%afun%(%ax%, %ay%)
%$$

$subhead s$$
The argument $icode s$$ has prototype
$codei%
     const CppAD:vector<bool>& %s%
%$$
and its size is $icode m$$.
It is a sparsity pattern for
$latex S(x) = g^{(1)} [ f(x) ] \in B^{1 \times m}$$.

$subhead t$$
This argument has prototype
$codei%
     CppAD:vector<bool>& %t%
%$$
and its size is $icode m$$.
The input values of its elements
are not specified (must not matter).
Upon return, $icode t$$ is a
sparsity pattern for
$latex T(x) \in B^{1 \times n}$$ where
$latex \[
	T(x) = (g \circ f)^{(1)} (x) = S(x) * f^{(1)} (x)
\]$$

$subhead q$$
The argument $icode q$$ has prototype
$codei%
     size_t %q%
%$$
It specifies the number of columns in
$latex R \in B^{n \times q}$$,
$latex U(x) \in B^{m \times q}$$, and
$latex V(x) \in B^{n \times q}$$.

$subhead r$$
This argument has prototype
$codei%
     const %atomic_sparsity%& %r%
%$$
and is a $cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
$latex R \in B^{n \times q}$$.

$head u$$
This argument has prototype
$codei%
     const %atomic_sparsity%& %u%
%$$
and is a $cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
$latex U(x) \in B^{m \times q}$$ which is defined by
$latex \[
\begin{array}{rcl}
U(x)
& = &
\{ \partial_u \{ \partial_y g[ y + f^{(1)} (x) R u ] \}_{y=f(x)} \}_{u=0}
\\
& = &
\partial_u \{ g^{(1)} [ f(x) + f^{(1)} (x) R u ] \}_{u=0}
\\
& = &
g^{(2)} [ f(x) ] f^{(1)} (x) R
\end{array}
\] $$

$subhead v$$
This argument has prototype
$codei%
     %atomic_sparsity%& %v%
%$$
The input value of its elements
are not specified (must not matter).
Upon return, $icode v$$ is a
$cref/atomic_sparsity/atomic_option/atomic_sparsity/$$ pattern for
$latex V(x) \in B^{n \times q}$$ which is defined by
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
f^{(1)} (x)^\R{T} g^{(2)} [ f(x) ] f^{(1)} (x)  R
+
\sum_{i=1}^m g_i^{(1)} [ f(x) ] \; f_i^{(2)} (x) R
\\
& = &
f^{(1)} (x)^\R{T} U(x)
+
\sum_{i=1}^m S_i (x) \; f_i^{(2)} (x) R
\end{array}
\] $$

$subhead x$$
$index deprecated$$
The argument has prototype
$codei%
	const CppAD::vector<%Base%>& %x%
%$$
and size is equal to the $icode n$$.
This is the $cref Value$$ value corresponding to the parameters in the
vector $cref/ax/atomic_afun/ax/$$ (when the atomic function was called).
To be specific, if
$codei%
	if( Parameter(%ax%[%i%]) == true )
		%x%[%i%] = Value( %ax%[%i%] );
	else
		%x%[%i%] = CppAD::numeric_limits<%Base%>::quiet_NaN();
%$$
The version of this function with out the $icode x$$ argument is deprecated;
i.e., you should include the argument even if you do not use it.

$children%
	example/atomic/rev_sparse_hes.cpp
%$$
$head Examples$$
The file $cref atomic_rev_sparse_hes.cpp$$ contains an example and test
that uses this routine.
It returns true if the test passes and false if it fails.

$end
-----------------------------------------------------------------------------
*/
/*!
Link from reverse Hessian sparsity sweep to base_atomic

\param vx [in]
which componens of x are variables.

\param s [in]
is the reverse Jacobian sparsity pattern w.r.t the result vector y.

\param t [out]
is the reverse Jacobian sparsity pattern w.r.t the argument vector x.

\param q [in]
is the column dimension for the sparsity partterns.

\param r [in]
is the forward Jacobian sparsity pattern w.r.t the argument vector x

\param u [in]
is the Hessian sparsity pattern w.r.t the result vector y.

\param v [out]
is the Hessian sparsity pattern w.r.t the argument vector x.

\param x [in]
is the integer value of the x arguments that are parameters.
*/
virtual bool rev_sparse_hes(
	const vector<bool>&                     vx ,
	const vector<bool>&                     s  ,
	      vector<bool>&                     t  ,
	size_t                                  q  ,
	const vector< std::set<size_t> >&       r  ,
	const vector< std::set<size_t> >&       u  ,
	      vector< std::set<size_t> >&       v  ,
	const vector<Base>&                     x  )
{	return false; }
virtual bool rev_sparse_hes(
	const vector<bool>&                     vx ,
	const vector<bool>&                     s  ,
	      vector<bool>&                     t  ,
	size_t                                  q  ,
	const vector<bool>&                     r  ,
	const vector<bool>&                     u  ,
	      vector<bool>&                     v  ,
	const vector<Base>&                     x  )
{	return false; }
virtual bool rev_sparse_hes(
	const vector<bool>&                     vx ,
	const vector<bool>&                     s  ,
	      vector<bool>&                     t  ,
	size_t                                  q  ,
	const vectorBool&                       r  ,
	const vectorBool&                       u  ,
	      vectorBool&                       v  ,
	const vector<Base>&                     x  )
{	return false; }
// deprecated
virtual bool rev_sparse_hes(
	const vector<bool>&                     vx ,
	const vector<bool>&                     s  ,
	      vector<bool>&                     t  ,
	size_t                                  q  ,
	const vector< std::set<size_t> >&       r  ,
	const vector< std::set<size_t> >&       u  ,
	      vector< std::set<size_t> >&       v  )
{	return false; }
virtual bool rev_sparse_hes(
	const vector<bool>&                     vx ,
	const vector<bool>&                     s  ,
	      vector<bool>&                     t  ,
	size_t                                  q  ,
	const vector<bool>&                     r  ,
	const vector<bool>&                     u  ,
	      vector<bool>&                     v  )
{	return false; }
virtual bool rev_sparse_hes(
	const vector<bool>&                     vx ,
	const vector<bool>&                     s  ,
	      vector<bool>&                     t  ,
	size_t                                  q  ,
	const vectorBool&                       r  ,
	const vectorBool&                       u  ,
	      vectorBool&                       v  )
{	return false; }
/*!
Link, before case split, from rev_hes_sweep to atomic_base.

\tparam InternalSparsity
Is the used internaly for sparsity calculations; i.e.,
sparse_pack or sparse_list.

\param x
is parameter arguments to the function, other components are nan.

\param x_index
is the variable index, on the tape, for the arguments to this function.
This size of x_index is n, the number of arguments to this function.

\param y_index
is the variable index, on the tape, for the results for this function.
This size of y_index is m, the number of results for this function.

\param for_jac_sparsity
On input, for j = 0, ... , n-1, the sparsity pattern with index x_index[j],
is the forward Jacobian sparsity for the j-th argument to this atomic function.

\param rev_jac_flag
This shows which variables affect the function we are
computing the Hessian of.
On input, for i = 0, ... , m-1, the rev_jac_flag[ y_index[i] ] is true
if the Jacobian of function (we are computing sparsity for) is no-zero.
Upon return, for j = 0, ... , n-1, rev_jac_flag [ x_index[j] ]
as been adjusted to accound removing this atomic function.

\param rev_hes_sparsity
This is the sparsity pattern for the Hessian.
On input, for i = 0, ... , m-1, row y_index[i] is the reverse Hessian sparsity
with one of the partials with respect to to y_index[i].
*/
template <class InternalSparsity>
void rev_sparse_hes(
	const vector<Base>&        x                ,
	const vector<size_t>&      x_index          ,
	const vector<size_t>&      y_index          ,
	const InternalSparsity&    for_jac_sparsity ,
	bool*                      rev_jac_flag     ,
	InternalSparsity&          rev_hes_sparsity )
{	CPPAD_ASSERT_UNKNOWN( for_jac_sparsity.end() == rev_hes_sparsity.end() );
	size_t q           = rev_hes_sparsity.end();
	size_t n           = x_index.size();
	size_t m           = y_index.size();
	bool   ok          = false;
	size_t thread      = thread_alloc::thread_num();
	allocate_work(thread);
	bool   zero_empty  = true;
	bool   input_empty = false;
	bool   transpose   = false;
	//
	// vx
	vector<bool> vx(n);
	for(size_t j = 0; j < n; j++)
		vx[j] = x_index[j] != 0;
	//
	// note that s and t are vectors so transpose does not matter for bool case
	vector<bool> bool_s( work_[thread]->bool_s );
	vector<bool> bool_t( work_[thread]->bool_t );
	//
	bool_s.resize(m);
	bool_t.resize(n);
	//
	for(size_t i = 0; i < m; i++)
	{	if( y_index[i] > 0  )
			bool_s[i] = rev_jac_flag[ y_index[i] ];
	}
	//
	std::string msg = ": atomic_base.rev_sparse_hes: returned false";
	if( sparsity_ == pack_sparsity_enum )
	{	vectorBool&  pack_r( work_[thread]->pack_r );
		vectorBool&  pack_u( work_[thread]->pack_u );
		vectorBool&  pack_v( work_[thread]->pack_h );
		//
		pack_v.resize(n * q);
		//
		local::get_internal_sparsity(
			transpose, x_index, for_jac_sparsity, pack_r
		);
		local::get_internal_sparsity(
			transpose, y_index, rev_hes_sparsity, pack_u
		);
		//
		ok = rev_sparse_hes(vx, bool_s, bool_t, q, pack_r, pack_u, pack_v, x);
		if( ! ok )
			ok = rev_sparse_hes(vx, bool_s, bool_t, q, pack_r, pack_u, pack_v);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = pack_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, x_index, rev_hes_sparsity, pack_v
		);
	}
	else if( sparsity_ == bool_sparsity_enum )
	{	vector<bool>&  bool_r( work_[thread]->bool_r );
		vector<bool>&  bool_u( work_[thread]->bool_u );
		vector<bool>&  bool_v( work_[thread]->bool_h );
		//
		bool_v.resize(n * q);
		//
		local::get_internal_sparsity(
			transpose, x_index, for_jac_sparsity, bool_r
		);
		local::get_internal_sparsity(
			transpose, y_index, rev_hes_sparsity, bool_u
		);
		//
		ok = rev_sparse_hes(vx, bool_s, bool_t, q, bool_r, bool_u, bool_v, x);
		if( ! ok )
			ok = rev_sparse_hes(vx, bool_s, bool_t, q, bool_r, bool_u, bool_v);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = bool_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, x_index, rev_hes_sparsity, bool_v
		);
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( sparsity_ == set_sparsity_enum );
		vector< std::set<size_t> >&  set_r( work_[thread]->set_r );
		vector< std::set<size_t> >&  set_u( work_[thread]->set_u );
		vector< std::set<size_t> >&  set_v( work_[thread]->set_h );
		//
		set_v.resize(n);
		//
		local::get_internal_sparsity(
			transpose, x_index, for_jac_sparsity, set_r
		);
		local::get_internal_sparsity(
			transpose, y_index, rev_hes_sparsity, set_u
		);
		//
		ok = rev_sparse_hes(vx, bool_s, bool_t, q, set_r, set_u, set_v, x);
		if( ! ok )
			ok = rev_sparse_hes(vx, bool_s, bool_t, q, set_r, set_u, set_v);
		if( ! ok )
		{	msg = afun_name() + msg + " sparsity = set_sparsity_enum";
			CPPAD_ASSERT_KNOWN(false, msg.c_str());
		}
		local::set_internal_sparsity(zero_empty, input_empty,
			transpose, x_index, rev_hes_sparsity, set_v
		);
	}
	for(size_t j = 0; j < n; j++)
	{	if( x_index[j] > 0  )
			rev_jac_flag[ x_index[j] ] |= bool_t[j];
	}
	return;
}
/*
------------------------------------------------------------------------------
$begin atomic_base_clear$$
$spell
	sq
	alloc
$$

$section Free Static Variables$$
$mindex clear$$

$head Syntax$$
$codei%atomic_base<%Base%>::clear()%$$

$head Purpose$$
Each $code atomic_base$$ objects holds onto work space in order to
avoid repeated memory allocation calls and thereby increase speed
(until it is deleted).
If an the $code atomic_base$$ object is global or static because,
the it does not get deleted.
This is a problem when using
$code thread_alloc$$ $cref/free_all/ta_free_all/$$
to check that all allocated memory has been freed.
Calling this $code clear$$ function will free all the
memory currently being held onto by the
$codei%atomic_base<%Base%>%$$ class.

$head Future Use$$
If there is future use of an $code atomic_base$$ object,
after a call to $code clear$$,
the work space will be reallocated and held onto.

$head Restriction$$
This routine cannot be called
while in $cref/parallel/ta_in_parallel/$$ execution mode.

$end
------------------------------------------------------------------------------
*/
/*!
Free all thread_alloc static memory held by atomic_base (avoids reallocations).
(This does not include class_object() which is an std::vector.)
*/
/// Free vector memory used by this class (work space)
static void clear(void)
{	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel() ,
		"cannot use atomic_base clear during parallel execution"
	);
	size_t i = class_object().size();
	while(i--)
	{	atomic_base* op = class_object()[i];
		if( op != CPPAD_NULL )
		{	for(size_t thread = 0; thread < CPPAD_MAX_NUM_THREADS; thread++)
				op->free_work(thread);
		}
	}
	return;
}
// -------------------------------------------------------------------------
/*!
Set value of id (used by deprecated old_atomic class)

This function is called just before calling any of the virtual function
and has the corresponding id of the corresponding virtual call.
*/
virtual void set_old(size_t id)
{ }
// ---------------------------------------------------------------------------
};
} // END_CPPAD_NAMESPACE
# endif
