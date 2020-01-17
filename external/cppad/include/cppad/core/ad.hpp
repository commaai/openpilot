# ifndef CPPAD_CORE_AD_HPP
# define CPPAD_CORE_AD_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

// simple AD operations that must be defined for AD as well as base class
# include <cppad/core/ordered.hpp>
# include <cppad/core/identical.hpp>

// define the template classes that are used by the AD template class
# include <cppad/local/op_code.hpp>
# include <cppad/local/recorder.hpp>
# include <cppad/local/player.hpp>
# include <cppad/local/ad_tape.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE

typedef enum {
	tape_manage_new,
	tape_manage_delete,
	tape_manage_clear
} tape_manage_job;

template <class Base>
class AD {
private :
	// -----------------------------------------------------------------------
	// value_ corresponding to this object
	Base value_;

	// Tape identifier corresponding to taddr
	tape_id_t tape_id_;

	// taddr_ in tape for this variable
	addr_t taddr_;
	// -----------------------------------------------------------------------

	// enable use of AD<Base> in parallel mode
	template <class Type>
	friend void parallel_ad(void);

	// template friend functions where template parameter is not bound
	template <class VectorAD>
	friend void Independent(VectorAD &x, size_t abort_op_index);

	// one argument functions
	friend bool Parameter          <Base>
		(const AD<Base>    &u);
	friend bool Parameter          <Base>
		(const VecAD<Base> &u);
	friend bool Variable           <Base>
		(const AD<Base>    &u);
	friend bool Variable           <Base>
		(const VecAD<Base> &u);
	friend int  Integer            <Base>
		(const AD<Base>    &u);
	friend AD   Var2Par            <Base>
		(const AD<Base>    &u);

	// power function
	friend AD pow <Base>
		(const AD<Base> &x, const AD<Base> &y);

	// azmul function
	friend AD azmul <Base>
		(const AD<Base> &x, const AD<Base> &y);

	// order determining functions, see ordered.hpp
	friend bool GreaterThanZero   <Base> (const AD<Base> &x);
	friend bool GreaterThanOrZero <Base> (const AD<Base> &x);
	friend bool LessThanZero      <Base> (const AD<Base> &x);
	friend bool LessThanOrZero    <Base> (const AD<Base> &x);
	friend bool abs_geq           <Base>
		(const AD<Base>& x, const AD<Base>& y);

	// The identical property functions, see identical.hpp
	friend bool IdenticalPar      <Base> (const AD<Base> &x);
	friend bool IdenticalZero     <Base> (const AD<Base> &x);
	friend bool IdenticalOne      <Base> (const AD<Base> &x);
	friend bool IdenticalEqualPar <Base>
		(const AD<Base> &x, const AD<Base> &y);

	// EqualOpSeq function
	friend bool EqualOpSeq <Base>
		(const AD<Base> &u, const AD<Base> &v);

	// NearEqual function
	friend bool NearEqual <Base> (
	const AD<Base> &x, const AD<Base> &y, const Base &r, const Base &a);

	friend bool NearEqual <Base> (
	const Base &x, const AD<Base> &y, const Base &r, const Base &a);

	friend bool NearEqual <Base> (
	const AD<Base> &x, const Base &y, const Base &r, const Base &a);

	// CondExp function
	friend AD<Base> CondExpOp  <Base> (
		enum CompareOp  cop       ,
		const AD<Base> &left      ,
		const AD<Base> &right     ,
		const AD<Base> &trueCase  ,
		const AD<Base> &falseCase
	);

	// classes
	friend class local::ADTape<Base>;
	friend class ADFun<Base>;
	friend class atomic_base<Base>;
	friend class discrete<Base>;
	friend class VecAD<Base>;
	friend class VecAD_reference<Base>;

	// arithematic binary operators
	friend AD<Base> operator + <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend AD<Base> operator - <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend AD<Base> operator * <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend AD<Base> operator / <Base>
		(const AD<Base> &left, const AD<Base> &right);

	// comparison operators
	friend bool operator < <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool operator <= <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool operator > <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool operator >= <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool operator == <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool operator != <Base>
		(const AD<Base> &left, const AD<Base> &right);

	// input operator
	friend std::istream& operator >> <Base>
		(std::istream &is, AD<Base> &x);

	// output operations
	friend std::ostream& operator << <Base>
		(std::ostream &os, const AD<Base> &x);
	friend void PrintFor <Base> (
		const AD<Base>&    flag   ,
		const char*        before ,
		const AD<Base>&    var    ,
		const char*        after
	);
public:
	// type of value
	typedef Base value_type;

	// implicit default constructor
	inline AD(void);

	// use default implicit copy constructor and assignment operator
	// inline AD(const AD &x);
	// inline AD& operator=(const AD &x);

	// implicit construction and assingment from base type
	inline AD(const Base &b);
	inline AD& operator=(const Base &b);

	// implicit contructor and assignment from VecAD<Base>::reference
	inline AD(const VecAD_reference<Base> &x);
	inline AD& operator=(const VecAD_reference<Base> &x);

	// explicit construction from some other type (depricated)
	template <class T> inline explicit AD(const T &t);

	// assignment from some other type
	template <class T> inline AD& operator=(const T &right);

	// base type corresponding to an AD object
	friend Base Value <Base> (const AD<Base> &x);

	// compound assignment operators
	inline AD& operator += (const AD &right);
	inline AD& operator -= (const AD &right);
	inline AD& operator *= (const AD &right);
	inline AD& operator /= (const AD &right);

	// unary operators
	inline AD operator +(void) const;
	inline AD operator -(void) const;

	// destructor
	~AD(void)
	{ }

	// interface so these functions need not be friends
	inline AD abs_me(void) const;
	inline AD acos_me(void) const;
	inline AD asin_me(void) const;
	inline AD atan_me(void) const;
	inline AD cos_me(void) const;
	inline AD cosh_me(void) const;
	inline AD exp_me(void) const;
	inline AD fabs_me(void) const;
	inline AD log_me(void) const;
	inline AD sin_me(void) const;
	inline AD sign_me(void) const;
	inline AD sinh_me(void) const;
	inline AD sqrt_me(void) const;
	inline AD tan_me(void) const;
	inline AD tanh_me(void) const;
# if CPPAD_USE_CPLUSPLUS_2011
	inline AD erf_me(void) const;
	inline AD asinh_me(void) const;
	inline AD acosh_me(void) const;
	inline AD atanh_me(void) const;
	inline AD expm1_me(void) const;
	inline AD log1p_me(void) const;
# endif

	// ----------------------------------------------------------
	// static public member functions

	// abort current AD<Base> recording
	static void        abort_recording(void);

	// set the maximum number of OpenMP threads (deprecated)
	static void        omp_max_thread(size_t number);

	// These functions declared public so can be accessed by user through
	// a macro interface and are not intended for direct use.
	// The macro interface is documented in bool_fun.hpp.
	// Developer documentation for these fucntions is in  bool_fun.hpp
	static inline bool UnaryBool(
		bool FunName(const Base &x),
		const AD<Base> &x
	);
	static inline bool BinaryBool(
		bool FunName(const Base &x, const Base &y),
		const AD<Base> &x , const AD<Base> &y
	);

private:
	//
	// Make this variable a parameter
	//
	void make_parameter(void)
	{	CPPAD_ASSERT_UNKNOWN( Variable(*this) );  // currently a var
		tape_id_ = 0;
	}
	//
	// Make this parameter a new variable
	//
	void make_variable(tape_id_t id,  addr_t taddr)
	{	CPPAD_ASSERT_UNKNOWN( Parameter(*this) ); // currently a par
		CPPAD_ASSERT_UNKNOWN( taddr > 0 );        // sure valid taddr

		taddr_   = taddr;
		tape_id_ = id;
	}
	// ---------------------------------------------------------------
	// tape linking functions
	//
	// not static
	inline local::ADTape<Base>* tape_this(void) const;
	//
	// static
	inline static tape_id_t**    tape_id_handle(size_t thread);
	inline static tape_id_t*     tape_id_ptr(size_t thread);
	inline static local::ADTape<Base>** tape_handle(size_t thread);
	static local::ADTape<Base>*         tape_manage(tape_manage_job job);
	inline static local::ADTape<Base>*  tape_ptr(void);
	inline static local::ADTape<Base>*  tape_ptr(tape_id_t tape_id);
};
// ---------------------------------------------------------------------------

} // END_CPPAD_NAMESPACE

// tape linking private functions
# include <cppad/core/tape_link.hpp>

// operations that expect the AD template class to be defined


# endif
