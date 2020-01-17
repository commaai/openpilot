# ifndef CPPAD_CORE_AD_FUN_HPP
# define CPPAD_CORE_AD_FUN_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin ADFun$$
$spell
	xk
	Ind
	bool
	taylor_
	sizeof
	const
	std
	ind_taddr_
	dep_taddr_
$$

$spell
$$

$section ADFun Objects$$


$head Purpose$$
An AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$
is stored in an $code ADFun$$ object by its $cref FunConstruct$$.
The $code ADFun$$ object can then be used to calculate function values,
derivative values, and other values related to the corresponding function.

$childtable%
	omh/adfun.omh%
	cppad/core/optimize.hpp%
	example/abs_normal/abs_normal.omh%
	cppad/core/fun_check.hpp%
	cppad/core/check_for_nan.hpp
%$$

$end
*/

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file ad_fun.hpp
File used to define the ADFun<Base> class.
*/

/*!
Class used to hold function objects

\tparam Base
A function object has a recording of <tt>AD<Base></tt> operations.
It does it calculations using \c Base operations.
*/

template <class Base>
class ADFun {
// ------------------------------------------------------------
// Private member variables
private:
	/// Has this ADFun object been optmized
	bool has_been_optimized_;

	/// Check for nan's and report message to user (default value is true).
	bool check_for_nan_;

	/// If zero, ignoring comparison operators. Otherwise is the
	/// compare change count at which to store the operator index.
	size_t compare_change_count_;

	/// If compare_change_count_ is zero, compare_change_number_ is also zero.
	/// Otherwise, it is set to the number of comparison operations that had a
	/// different result during the subsequent zero order forward.
	size_t compare_change_number_;

	/// If compare_change_count is zero, compare_change_op_index_ is also
	/// zero. Otherwise it is the operator index for the comparison operator
	//// that corresponded to the number changing from count-1 to count.
	size_t compare_change_op_index_;

	/// number of orders stored in taylor_
	size_t num_order_taylor_;

	/// maximum number of orders that will fit in taylor_
	size_t cap_order_taylor_;

	/// number of directions stored in taylor_
	size_t num_direction_taylor_;

	/// number of variables in the recording (play_)
	size_t num_var_tape_;

	/// tape address for the independent variables
	CppAD::vector<size_t> ind_taddr_;

	/// tape address and parameter flag for the dependent variables
	CppAD::vector<size_t> dep_taddr_;

	/// which dependent variables are actually parameters
	CppAD::vector<bool>   dep_parameter_;

	/// results of the forward mode calculations
	local::pod_vector<Base> taylor_;

	/// which operations can be conditionally skipped
	/// Set during forward pass of order zero
	local::pod_vector<bool> cskip_op_;

	/// Variable on the tape corresponding to each vecad load operation
	/// (if zero, the operation corresponds to a parameter).
	local::pod_vector<addr_t> load_op_;

	/// the operation sequence corresponding to this object
	local::player<Base> play_;

	/// Packed results of the forward mode Jacobian sparsity calculations.
	/// for_jac_sparse_pack_.n_set() != 0  implies other sparsity results
	/// are empty
	local::sparse_pack      for_jac_sparse_pack_;

	/// Set results of the forward mode Jacobian sparsity calculations
	/// for_jac_sparse_set_.n_set() != 0  implies for_sparse_pack_ is empty.
	local::sparse_list         for_jac_sparse_set_;

// ------------------------------------------------------------
// Private member functions

	/// change the operation sequence corresponding to this object
	template <typename ADvector>
	void Dependent(local::ADTape<Base> *tape, const ADvector &y);

	// ------------------------------------------------------------
	// vector of bool version of ForSparseJac
	// (see doxygen in for_sparse_jac.hpp)
	template <class VectorSet>
	void ForSparseJacCase(
		bool               set_type  ,
		bool               transpose ,
		bool               dependency,
		size_t             q         ,
		const VectorSet&   r         ,
		VectorSet&         s
	);
	// vector of std::set<size_t> version of ForSparseJac
	// (see doxygen in for_sparse_jac.hpp)
	template <class VectorSet>
	void ForSparseJacCase(
		const std::set<size_t>&  set_type  ,
		bool                     transpose ,
		bool                     dependency,
		size_t                   q         ,
		const VectorSet&         r         ,
		VectorSet&               s
	);
	// ------------------------------------------------------------
	// vector of bool version of RevSparseJac
	// (see doxygen in rev_sparse_jac.hpp)
	template <class VectorSet>
	void RevSparseJacCase(
		bool               set_type  ,
		bool               transpose ,
		bool               dependency,
		size_t             p         ,
		const VectorSet&   s         ,
		VectorSet&         r
	);
	// vector of std::set<size_t> version of RevSparseJac
	// (see doxygen in rev_sparse_jac.hpp)
	template <class VectorSet>
	void RevSparseJacCase(
		const std::set<size_t>&  set_type  ,
		bool                     transpose ,
		bool                     dependency,
		size_t                   p         ,
		const VectorSet&         s         ,
		VectorSet&               r
	);
	// ------------------------------------------------------------
	// vector of bool version of ForSparseHes
	// (see doxygen in rev_sparse_hes.hpp)
	template <class VectorSet>
	void ForSparseHesCase(
		bool               set_type  ,
		const VectorSet&   r         ,
		const VectorSet&   s         ,
		VectorSet&         h
	);
	// vector of std::set<size_t> version of ForSparseHes
	// (see doxygen in rev_sparse_hes.hpp)
	template <class VectorSet>
	void ForSparseHesCase(
		const std::set<size_t>&  set_type  ,
		const VectorSet&         r         ,
		const VectorSet&         s         ,
		VectorSet&               h
	);
	// ------------------------------------------------------------
	// vector of bool version of RevSparseHes
	// (see doxygen in rev_sparse_hes.hpp)
	template <class VectorSet>
	void RevSparseHesCase(
		bool               set_type  ,
		bool               transpose ,
		size_t             q         ,
		const VectorSet&   s         ,
		VectorSet&         h
	);
	// vector of std::set<size_t> version of RevSparseHes
	// (see doxygen in rev_sparse_hes.hpp)
	template <class VectorSet>
	void RevSparseHesCase(
		const std::set<size_t>&  set_type  ,
		bool                     transpose ,
		size_t                   q         ,
		const VectorSet&         s         ,
		VectorSet&               h
	);
	// ------------------------------------------------------------
	// Forward mode version of SparseJacobian
	// (see doxygen in sparse_jacobian.hpp)
	template <class VectorBase, class VectorSet, class VectorSize>
	size_t SparseJacobianFor(
		const VectorBase&           x               ,
		      VectorSet&            p_transpose     ,
		const VectorSize&           row             ,
		const VectorSize&           col             ,
		      VectorBase&           jac             ,
		      sparse_jacobian_work& work
	);
	// Reverse mode version of SparseJacobian
	// (see doxygen in sparse_jacobian.hpp)
	template <class VectorBase, class VectorSet, class VectorSize>
	size_t SparseJacobianRev(
		const VectorBase&           x               ,
		      VectorSet&            p               ,
		const VectorSize&           row             ,
		const VectorSize&           col             ,
		      VectorBase&           jac             ,
		      sparse_jacobian_work& work
	);
	// ------------------------------------------------------------
	// combined sparse_list and sparse_pack version of
	// SparseHessian (see doxygen in sparse_hessian.hpp)
	template <class VectorBase, class VectorSet, class VectorSize>
	size_t SparseHessianCompute(
		const VectorBase&              x           ,
		const VectorBase&              w           ,
		      VectorSet&               sparsity    ,
		const VectorSize&              row         ,
		const VectorSize&              col         ,
		      VectorBase&              hes         ,
		      sparse_hessian_work&     work
	);
// ------------------------------------------------------------
public:
	/// copy constructor
	ADFun(const ADFun& g)
	: num_var_tape_(0)
	{	CppAD::ErrorHandler::Call(
		true,
		__LINE__,
		__FILE__,
		"ADFun(const ADFun& g)",
		"Attempting to use the ADFun<Base> copy constructor.\n"
		"Perhaps you are passing an ADFun<Base> object "
		"by value instead of by reference."
		);
	 }

	/// default constructor
	ADFun(void);

	// assignment operator
	// (see doxygen in fun_construct.hpp)
	void operator=(const ADFun& f);

	/// sequence constructor
	template <typename ADvector>
	ADFun(const ADvector &x, const ADvector &y);

	/// destructor
	~ADFun(void)
	{ }

	/// set value of check_for_nan_
	void check_for_nan(bool value)
	{	check_for_nan_ = value; }
	bool check_for_nan(void) const
	{	return check_for_nan_; }

	/// assign a new operation sequence
	template <typename ADvector>
	void Dependent(const ADvector &x, const ADvector &y);

	/// forward mode user API, one order multiple directions.
	template <typename VectorBase>
	VectorBase Forward(size_t q, size_t r, const VectorBase& x);

	/// forward mode user API, multiple directions one order.
	template <typename VectorBase>
	VectorBase Forward(size_t q,
		const VectorBase& x, std::ostream& s = std::cout
	);

	/// reverse mode sweep
	template <typename VectorBase>
	VectorBase Reverse(size_t p, const VectorBase &v);

	// ---------------------------------------------------------------------
	// Jacobian sparsity
	template <typename VectorSet>
	VectorSet ForSparseJac(
		size_t q, const VectorSet &r, bool transpose = false,
		bool dependency = false
	);
	template <typename VectorSet>
	VectorSet RevSparseJac(
		size_t q, const VectorSet &s, bool transpose = false,
		bool dependency = false
	);
	// ---------------------------------------------------------------------
	template <typename SizeVector, typename BaseVector>
	size_t sparse_jac_for(
		size_t                               group_max ,
		const BaseVector&                    x         ,
		sparse_rcv<SizeVector, BaseVector>&  subset    ,
		const sparse_rc<SizeVector>&         pattern   ,
		const std::string&                   coloring  ,
		sparse_jac_work&                     work
	);
	template <typename SizeVector, typename BaseVector>
	size_t sparse_jac_rev(
		const BaseVector&                    x        ,
		sparse_rcv<SizeVector, BaseVector>&  subset   ,
		const sparse_rc<SizeVector>&         pattern  ,
		const std::string&                   coloring ,
		sparse_jac_work&                     work
	);
	template <typename SizeVector, typename BaseVector>
	size_t sparse_hes(
		const BaseVector&                    x        ,
		const BaseVector&                    w        ,
		sparse_rcv<SizeVector, BaseVector>&  subset   ,
		const sparse_rc<SizeVector>&         pattern  ,
		const std::string&                   coloring ,
		sparse_hes_work&                     work
	);
	// ---------------------------------------------------------------------
	template <typename SizeVector>
	void for_jac_sparsity(
		const sparse_rc<SizeVector>& pattern_in       ,
		bool                         transpose        ,
		bool                         dependency       ,
		bool                         internal_bool    ,
		sparse_rc<SizeVector>&       pattern_out
	);
	template <typename SizeVector>
	void rev_jac_sparsity(
		const sparse_rc<SizeVector>& pattern_in       ,
		bool                         transpose        ,
		bool                         dependency       ,
		bool                         internal_bool    ,
		sparse_rc<SizeVector>&       pattern_out
	);
	template <typename BoolVector, typename SizeVector>
	void rev_hes_sparsity(
		const BoolVector&            select_range     ,
		bool                         transpose        ,
		bool                         internal_bool    ,
		sparse_rc<SizeVector>&       pattern_out
	);
	template <typename BoolVector, typename SizeVector>
	void for_hes_sparsity(
		const BoolVector&            select_domain    ,
		const BoolVector&            select_range     ,
		bool                         internal_bool    ,
		sparse_rc<SizeVector>&       pattern_out
	);
	// ---------------------------------------------------------------------
	// forward mode Hessian sparsity
	// (see doxygen documentation in rev_sparse_hes.hpp)
	template <typename VectorSet>
	VectorSet ForSparseHes(
		const VectorSet &r, const VectorSet &s
	);
	// internal set sparsity version of ForSparseHes
	// (used by checkpoint functions only)
	void ForSparseHesCheckpoint(
		vector<bool>&                 r         ,
		vector<bool>&                 s         ,
		local::sparse_list&                  h
	);
	// reverse mode Hessian sparsity
	// (see doxygen documentation in rev_sparse_hes.hpp)
	template <typename VectorSet>
	VectorSet RevSparseHes(
		size_t q, const VectorSet &s, bool transpose = false
	);
	// internal set sparsity version of RevSparseHes
	// (used by checkpoint functions only)
	void RevSparseHesCheckpoint(
		size_t                        q         ,
		vector<bool>&                 s         ,
		bool                          transpose ,
		local::sparse_list&                  h
	);
	// internal set sparsity version of RevSparseJac
	// (used by checkpoint functions only)
	void RevSparseJacCheckpoint(
		size_t                        q          ,
		const local::sparse_list&     r          ,
		bool                          transpose  ,
		bool                          dependency ,
		local::sparse_list&                  s
	);
    // internal set sparsity version of RevSparseJac
    // (used by checkpoint functions only)
	void ForSparseJacCheckpoint(
	size_t                        q          ,
	const local::sparse_list&     r          ,
	bool                          transpose  ,
	bool                          dependency ,
	local::sparse_list&                  s
	);

	/// amount of memory used for boolean Jacobain sparsity pattern
	size_t size_forward_bool(void) const
	{	return for_jac_sparse_pack_.memory(); }

	/// free memory used for Jacobain sparsity pattern
	void size_forward_bool(size_t zero)
	{	CPPAD_ASSERT_KNOWN(
			zero == 0,
			"size_forward_bool: argument not equal to zero"
		);
		for_jac_sparse_pack_.resize(0, 0);
	}

	/// amount of memory used for vector of set Jacobain sparsity pattern
	size_t size_forward_set(void) const
	{	return for_jac_sparse_set_.memory(); }

	/// free memory used for Jacobain sparsity pattern
	void size_forward_set(size_t zero)
	{	CPPAD_ASSERT_KNOWN(
			zero == 0,
			"size_forward_bool: argument not equal to zero"
		);
		for_jac_sparse_set_.resize(0, 0);
	}

	/// number of operators in the operation sequence
	size_t size_op(void) const
	{	return play_.num_op_rec(); }

	/// number of operator arguments in the operation sequence
	size_t size_op_arg(void) const
	{	return play_.num_op_arg_rec(); }

	/// amount of memory required for the operation sequence
	size_t size_op_seq(void) const
	{	return play_.Memory(); }

	/// number of parameters in the operation sequence
	size_t size_par(void) const
	{	return play_.num_par_rec(); }

	/// number taylor coefficient orders calculated
	size_t size_order(void) const
	{	return num_order_taylor_; }

	/// number taylor coefficient directions calculated
	size_t size_direction(void) const
	{	return num_direction_taylor_; }

	/// number of characters in the operation sequence
	size_t size_text(void) const
	{	return play_.num_text_rec(); }

	/// number of variables in opertion sequence
	size_t size_var(void) const
	{	return num_var_tape_; }

	/// number of VecAD indices in the operation sequence
	size_t size_VecAD(void) const
	{	return play_.num_vec_ind_rec(); }

	/// set number of orders currently allocated (user API)
	void capacity_order(size_t c);

	/// set number of orders and directions currently allocated
	void capacity_order(size_t c, size_t r);

	/// number of variables in conditional expressions that can be skipped
	size_t number_skip(void);

	/// number of independent variables
	size_t Domain(void) const
	{	return ind_taddr_.size(); }

	/// number of dependent variables
	size_t Range(void) const
	{	return dep_taddr_.size(); }

	/// is variable a parameter
	bool Parameter(size_t i)
	{	CPPAD_ASSERT_KNOWN(
			i < dep_taddr_.size(),
			"Argument to Parameter is >= dimension of range space"
		);
		return dep_parameter_[i];
	}

	/// Deprecated: number of comparison operations that changed
	/// for the previous zero order forward (than when function was recorded)
	size_t CompareChange(void) const
	{	return compare_change_number_; }

	/// count as which to store operator index
	void compare_change_count(size_t count)
	{	compare_change_count_    = count;
		compare_change_number_   = 0;
		compare_change_op_index_ = 0;
	}

	/// number of comparison operations that changed
	size_t compare_change_number(void) const
	{	return compare_change_number_; }

	/// operator index for the count-th  comparison change
	size_t compare_change_op_index(void) const
	{	if( has_been_optimized_ )
			return 0;
		return compare_change_op_index_;
	}

	/// calculate entire Jacobian
	template <typename VectorBase>
	VectorBase Jacobian(const VectorBase &x);

	/// calculate Hessian for one component of f
	template <typename VectorBase>
	VectorBase Hessian(const VectorBase &x, const VectorBase &w);
	template <typename VectorBase>
	VectorBase Hessian(const VectorBase &x, size_t i);

	/// forward mode calculation of partial w.r.t one domain component
	template <typename VectorBase>
	VectorBase ForOne(
		const VectorBase   &x ,
		size_t              j );

	/// reverse mode calculation of derivative of one range component
	template <typename VectorBase>
	VectorBase RevOne(
		const VectorBase   &x ,
		size_t              i );

	/// forward mode calculation of a subset of second order partials
	template <typename VectorBase, typename VectorSize_t>
	VectorBase ForTwo(
		const VectorBase   &x ,
		const VectorSize_t &J ,
		const VectorSize_t &K );

	/// reverse mode calculation of a subset of second order partials
	template <typename VectorBase, typename VectorSize_t>
	VectorBase RevTwo(
		const VectorBase   &x ,
		const VectorSize_t &I ,
		const VectorSize_t &J );

	/// calculate sparse Jacobians
	template <typename VectorBase>
	VectorBase SparseJacobian(
		const VectorBase &x
	);
	template <typename VectorBase, typename VectorSet>
	VectorBase SparseJacobian(
		const VectorBase &x ,
		const VectorSet  &p
	);
	template <class VectorBase, class VectorSet, class VectorSize>
	size_t SparseJacobianForward(
		const VectorBase&     x     ,
		const VectorSet&      p     ,
		const VectorSize&     r     ,
		const VectorSize&     c     ,
		VectorBase&           jac   ,
		sparse_jacobian_work& work
	);
	template <class VectorBase, class VectorSet, class VectorSize>
	size_t SparseJacobianReverse(
		const VectorBase&     x    ,
		const VectorSet&      p    ,
		const VectorSize&     r    ,
		const VectorSize&     c    ,
		VectorBase&           jac  ,
		sparse_jacobian_work& work
	);

	/// calculate sparse Hessians
	template <typename VectorBase>
	VectorBase SparseHessian(
		const VectorBase&    x  ,
		const VectorBase&    w
	);
	template <typename VectorBase, typename VectorBool>
	VectorBase SparseHessian(
		const VectorBase&    x  ,
		const VectorBase&    w  ,
		const VectorBool&    p
	);
	template <class VectorBase, class VectorSet, class VectorSize>
	size_t SparseHessian(
		const VectorBase&    x   ,
		const VectorBase&    w   ,
		const VectorSet&     p   ,
		const VectorSize&    r   ,
		const VectorSize&    c   ,
		VectorBase&          hes ,
		sparse_hessian_work& work
	);

	// Optimize the tape
	// (see doxygen documentation in optimize.hpp)
	void optimize( const std::string& options = "" );

	// create abs-normal representation of the function f(x)
	void abs_normal_fun( ADFun& g, ADFun& a );
	// ------------------- Deprecated -----------------------------

	/// deprecated: assign a new operation sequence
	template <typename ADvector>
	void Dependent(const ADvector &y);

	/// Deprecated: number of variables in opertion sequence
	size_t Size(void) const
	{	return num_var_tape_; }

	/// Deprecated: # taylor_ coefficients currently stored
	/// (per variable,direction)
	size_t Order(void) const
	{	return num_order_taylor_ - 1; }

	/// Deprecated: amount of memory for this object
	/// Note that an approximation is used for the std::set<size_t> memory
	size_t Memory(void) const
	{	size_t pervar  = cap_order_taylor_ * sizeof(Base)
		+ for_jac_sparse_pack_.memory()
		+ for_jac_sparse_set_.memory();
		size_t total   = num_var_tape_  * pervar + play_.Memory();
		return total;
	}

	/// Deprecated: # taylor_ coefficient orderss stored
	/// (per variable,direction)
	size_t taylor_size(void) const
	{	return num_order_taylor_; }

	/// Deprecated: Does this AD operation sequence use
	/// VecAD<Base>::reference operands
	bool use_VecAD(void) const
	{	return play_.num_vec_ind_rec() > 0; }

	/// Deprecated: # taylor_ coefficient orders calculated
	/// (per variable,direction)
	size_t size_taylor(void) const
	{	return num_order_taylor_; }

	/// Deprecated: set number of orders currently allocated
	/// (per variable,direction)
	void capacity_taylor(size_t per_var);
};
// ---------------------------------------------------------------------------

} // END_CPPAD_NAMESPACE

// non-user interfaces
# include <cppad/local/forward0sweep.hpp>
# include <cppad/local/forward1sweep.hpp>
# include <cppad/local/forward2sweep.hpp>
# include <cppad/local/reverse_sweep.hpp>
# include <cppad/local/for_jac_sweep.hpp>
# include <cppad/local/rev_jac_sweep.hpp>
# include <cppad/local/rev_hes_sweep.hpp>
# include <cppad/local/for_hes_sweep.hpp>

// user interfaces
# include <cppad/core/parallel_ad.hpp>
# include <cppad/core/independent.hpp>
# include <cppad/core/dependent.hpp>
# include <cppad/core/fun_construct.hpp>
# include <cppad/core/abort_recording.hpp>
# include <cppad/core/fun_eval.hpp>
# include <cppad/core/drivers.hpp>
# include <cppad/core/fun_check.hpp>
# include <cppad/core/omp_max_thread.hpp>
# include <cppad/core/optimize.hpp>
# include <cppad/core/abs_normal_fun.hpp>

# endif
