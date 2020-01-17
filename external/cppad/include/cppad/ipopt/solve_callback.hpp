// $Id: solve_callback.hpp 3804 2016-03-20 15:08:46Z bradbell $
# ifndef CPPAD_IPOPT_SOLVE_CALLBACK_HPP
# define CPPAD_IPOPT_SOLVE_CALLBACK_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

# include <cppad/cppad.hpp>
# include <coin/IpIpoptApplication.hpp>
# include <coin/IpTNLP.hpp>
# include <cppad/ipopt/solve_result.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
namespace ipopt {
/*!
\file solve_callback.hpp
\brief Class that connects ipopt::solve to Ipopt
*/

/*!
Class that Ipopt uses for obtaining information about this problem.


\section Evaluation_Methods Evaluation Methods
The set of evaluation methods for this class is
\verbatim
	{ eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h }
\endverbatim
Note that the bool return flag for the evaluations methods
does not appear in the Ipopt documentation.
Looking at the code, it seems to be a flag telling Ipopt to abort
when the flag is false.
*/
template <class Dvector, class ADvector, class FG_eval>
class solve_callback : public Ipopt::TNLP
{
private:
	// ------------------------------------------------------------------
	// Types used by this class
	// ------------------------------------------------------------------
	/// A Scalar value used by Ipopt
	typedef Ipopt::Number                         Number;
	/// An index value used by Ipopt
	typedef Ipopt::Index                          Index;
	/// Indexing style used in Ipopt sparsity structure
	typedef Ipopt::TNLP::IndexStyleEnum           IndexStyleEnum;
	// ------------------------------------------------------------------
	// Values directly passed in to constuctor
	// ------------------------------------------------------------------
	/// dimension of the range space for f(x).
	/// The objective is sum_i f_i (x).
	/// Note that, at this point, there is no advantage having nf_ > 1.
	const size_t                    nf_;
	/// dimension of the domain space for f(x) and g(x)
	const size_t                    nx_;
	/// dimension of the range space for g(x)
	const size_t                    ng_;
	/// initial value for x
	const Dvector&                  xi_;
	/// lower limit for x
	const Dvector&                  xl_;
	/// upper limit for x
	const Dvector&                  xu_;
	/// lower limit for g(x)
	const Dvector&                  gl_;
	/// upper limit for g(x)
	const Dvector&                  gu_;
	/// object that evaluates f(x) and g(x)
	FG_eval&                        fg_eval_;
	/// should operation sequence be retaped for each new x.
	bool                            retape_;
	/// Should sparse methods be used to compute Jacobians and Hessians
	/// with forward mode used for Jacobian.
	bool                            sparse_forward_;
	/// Should sparse methods be used to compute Jacobians and Hessians
	/// with reverse mode used for Jacobian.
	bool                            sparse_reverse_;
	/// final results are returned to this structure
	solve_result<Dvector>&          solution_;
	// ------------------------------------------------------------------
	// Values that are initilaized by the constructor
	// ------------------------------------------------------------------
	/// AD function object that evaluates x -> [ f(x) , g(x) ]
	/// If retape is false, this object is initialzed by constructor
	/// otherwise it is set by cache_new_x each time it is called.
	CppAD::ADFun<double>            adfun_;
	/// value of x corresponding to previous new_x
	Dvector                         x0_;
	/// value of fg corresponding to previous new_x
	Dvector                         fg0_;
	// ----------------------------------------------------------------------
	// Jacobian information
	// ----------------------------------------------------------------------
	/// Sparsity pattern for Jacobian of [f(x), g(x) ].
	/// If sparse is true, this pattern set by constructor and does not change.
	/// Otherwise this vector has size zero.
	CppAD::vectorBool               pattern_jac_;
	/// Row indices of [f(x), g(x)] for Jacobian of g(x) in row order.
	/// (Set by constructor and not changed.)
	CppAD::vector<size_t>           row_jac_;
	/// Column indices for Jacobian of g(x), same order as row_jac_.
	/// (Set by constructor and not changed.)
	CppAD::vector<size_t>           col_jac_;
	/// col_order_jac_ sorts row_jac_ and col_jac_ in column order.
	/// (Set by constructor and not changed.)
	CppAD::vector<size_t>           col_order_jac_;
	/// Work vector used by SparseJacobian, stored here to avoid recalculation.
	CppAD::sparse_jacobian_work     work_jac_;
	// ----------------------------------------------------------------------
	// Hessian information
	// ----------------------------------------------------------------------
	/// Sparsity pattern for Hessian of Lagragian
	/// \f[ L(x) = \sigma \sum_i f_i (x) + \sum_i \lambda_i  g_i (x) \f]
	/// If sparse is true, this pattern set by constructor and does not change.
	/// Otherwise this vector has size zero.
	CppAD::vectorBool               pattern_hes_;
	/// Row indices of Hessian lower left triangle in row order.
	/// (Set by constructor and not changed.)
	CppAD::vector<size_t>           row_hes_;
	/// Column indices of Hessian left triangle in same order as row_hes_.
	/// (Set by constructor and not changed.)
	CppAD::vector<size_t>           col_hes_;
	/// Work vector used by SparseJacobian, stored here to avoid recalculation.
	CppAD::sparse_hessian_work      work_hes_;
	// ------------------------------------------------------------------
	// Private member functions
	// ------------------------------------------------------------------
	/*!
	Cache information for a new value of x.

	\param x
	is the new value for x.

	\par x0_
	the elements of this vector are set to the new value for x.

	\par fg0_
	the elements of this vector are set to the new value for [f(x), g(x)]

	\par adfun_
	If retape is true, the operation sequence for this function
	is changes to correspond to the argument x.
	If retape is false, the operation sequence is not changed.
	The zero order Taylor coefficients for this function are set
	so they correspond to the argument x.
	*/
	void cache_new_x(const Number* x)
	{	size_t i;
		if( retape_ )
		{	// make adfun_, as well as x0_ and fg0_ correspond to this x
			ADvector a_x(nx_), a_fg(nf_ + ng_);
			for(i = 0; i < nx_; i++)
			{	x0_[i] = x[i];
				a_x[i] = x[i];
			}
			CppAD::Independent(a_x);
			fg_eval_(a_fg, a_x);
			adfun_.Dependent(a_x, a_fg);
		}
		else
		{	// make x0_ and fg0_ correspond to this x
			for(i = 0; i < nx_; i++)
				x0_[i] = x[i];
		}
		fg0_ = adfun_.Forward(0, x0_);
	}
public:
	// ----------------------------------------------------------------------
	/*!
	Constructor for the interface between ipopt::solve and Ipopt

	\param nf
	dimension of the range space for f(x)

	\param nx
	dimension of the domain space for f(x) and g(x).

	\param ng
	dimension of the range space for g(x)

	\param xi
	initial value of x during the optimization procedure (size nx).

	\param xl
	lower limit for x (size nx).

	\param xu
	upper limit for x (size nx).

	\param gl
	lower limit for g(x) (size ng).

	\param gu
	upper limit for g(x) (size ng).

	\param fg_eval
	function object that evaluations f(x) and g(x) using fg_eval(fg, x)

	\param retape
	should the operation sequence be retaped for each argument value.

	\param sparse_forward
	should sparse matrix computations be used for Jacobians and Hessians
	with forward mode for Jacobian.

	\param sparse_reverse
	should sparse matrix computations be used for Jacobians and Hessians
	with reverse mode for Jacobian.
	(sparse_forward and sparse_reverse cannot both be true).

	\param solution
	object where final results are stored.
	*/
	solve_callback(
		size_t                 nf              ,
		size_t                 nx              ,
		size_t                 ng              ,
		const Dvector&         xi              ,
		const Dvector&         xl              ,
		const Dvector&         xu              ,
		const Dvector&         gl              ,
		const Dvector&         gu              ,
		FG_eval&               fg_eval         ,
		bool                   retape          ,
		bool                   sparse_forward  ,
		bool                   sparse_reverse  ,
		solve_result<Dvector>& solution ) :
	nf_ ( nf ),
	nx_ ( nx ),
	ng_ ( ng ),
	xi_ ( xi ),
	xl_ ( xl ),
	xu_ ( xu ),
	gl_ ( gl ),
	gu_ ( gu ),
	fg_eval_ ( fg_eval ),
	retape_ ( retape ),
	sparse_forward_ ( sparse_forward ),
	sparse_reverse_ ( sparse_reverse ),
	solution_ ( solution )
	{	CPPAD_ASSERT_UNKNOWN( ! ( sparse_forward_ & sparse_reverse_ ) );

		size_t i, j;
		size_t nfg = nf_ + ng_;

		// initialize x0_ and fg0_ wih proper dimensions and value nan
		x0_.resize(nx);
		fg0_.resize(nfg);
		for(i = 0; i < nx_; i++)
			x0_[i] = CppAD::nan(0.0);
		for(i = 0; i < nfg; i++)
			fg0_[i] = CppAD::nan(0.0);

		if( ! retape_ )
		{	// make adfun_ correspond to x -> [ f(x), g(x) ]
			ADvector a_x(nx_), a_fg(nfg);
			for(i = 0; i < nx_; i++)
				a_x[i] = xi_[i];
			CppAD::Independent(a_x);
			fg_eval_(a_fg, a_x);
			adfun_.Dependent(a_x, a_fg);
			// optimize because we will make repeated use of this tape
			adfun_.optimize();
		}
		if( sparse_forward_ | sparse_reverse_ )
		{	CPPAD_ASSERT_UNKNOWN( ! retape );
			size_t m = nf_ + ng_;
			//
			// -----------------------------------------------------------
			// Jacobian
			pattern_jac_.resize( m * nx_ );
			if( nx_ <= m )
			{	// use forward mode to compute sparsity

				// number of bits that are packed into one unit in vectorBool
				size_t n_column = vectorBool::bit_per_unit();

				// sparsity patterns for current columns
				vectorBool r(nx_ * n_column), s(m * n_column);

				// compute the sparsity pattern n_column columns at a time
				size_t n_loop = (nx_ - 1) / n_column + 1;
				for(size_t i_loop = 0; i_loop < n_loop; i_loop++)
				{	// starting column index for this iteration
					size_t i_column = i_loop * n_column;

					// pattern that picks out the appropriate columns
					for(i = 0; i < nx_; i++)
					{	for(j = 0; j < n_column; j++)
							r[i * n_column + j] = (i == i_column + j);
					}
					s = adfun_.ForSparseJac(n_column, r);

					// fill in the corresponding columns of total_sparsity
					for(i = 0; i < m; i++)
					{	for(j = 0; j < n_column; j++)
						{	if( i_column + j < nx_ )
								pattern_jac_[i * nx_ + i_column + j] =
									s[i * n_column + j];
						}
					}
				}
			}
			else
			{	// use reverse mode to compute sparsity

				// number of bits that are packed into one unit in vectorBool
				size_t n_row = vectorBool::bit_per_unit();

				// sparsity patterns for current rows
				vectorBool r(n_row * m), s(n_row * nx_);

				// compute the sparsity pattern n_row row at a time
				size_t n_loop = (m - 1) / n_row + 1;
				for(size_t i_loop = 0; i_loop < n_loop; i_loop++)
				{	// starting row index for this iteration
					size_t i_row = i_loop * n_row;

					// pattern that picks out the appropriate rows
					for(i = 0; i < n_row; i++)
					{	for(j = 0; j < m; j++)
							r[i * m + j] = (i_row + i ==  j);
					}
					s = adfun_.RevSparseJac(n_row, r);

					// fill in correspoding rows of total sparsity
					for(i = 0; i < n_row; i++)
					{	for(j = 0; j < nx_; j++)
							if( i_row + i < m )
								pattern_jac_[ (i_row + i) * nx_ + j ] =
									s[ i  * nx_ + j];
					}
				}
			}
			/*
			{	// use reverse mode to compute sparsity
				CppAD::vectorBool s(m * m);
				for(i = 0; i < m; i++)
				{	for(j = 0; j < m; j++)
						s[i * m + j] = (i == j);
				}
				pattern_jac_ = adfun_.RevSparseJac(m, s);
			}
			*/
			// Set row and column indices in Jacoian of [f(x), g(x)]
			// for Jacobian of g(x). These indices are in row major order.
			for(i = nf_; i < nfg; i++)
			{	for(j = 0; j < nx_; j++)
				{	if( pattern_jac_[ i * nx_ + j ] )
					{	row_jac_.push_back(i);
						col_jac_.push_back(j);
					}
				}
			}
			// Set row and column indices in Jacoian of [f(x), g(x)]
			// for Jacobian of g(x). These indices are in row major order.
			// -----------------------------------------------------------
			// Hessian
			pattern_hes_.resize(nx_ * nx_);

			// number of bits that are packed into one unit in vectorBool
			size_t n_column = vectorBool::bit_per_unit();

			// sparsity patterns for current columns
			vectorBool r(nx_ * n_column), h(nx_ * n_column);

			// sparsity pattern for range space of function
			vectorBool s(m);
			for(i = 0; i < m; i++)
				s[i] = true;

			// compute the sparsity pattern n_column columns at a time
			size_t n_loop = (nx_ - 1) / n_column + 1;
			for(size_t i_loop = 0; i_loop < n_loop; i_loop++)
			{	// starting column index for this iteration
				size_t i_column = i_loop * n_column;

				// pattern that picks out the appropriate columns
				for(i = 0; i < nx_; i++)
				{	for(j = 0; j < n_column; j++)
						r[i * n_column + j] = (i == i_column + j);
				}
				adfun_.ForSparseJac(n_column, r);

				// sparsity pattern corresponding to paritls w.r.t. (theta, u)
				// of partial w.r.t. the selected columns
				bool transpose = true;
				h = adfun_.RevSparseHes(n_column, s, transpose);

				// fill in the corresponding columns of total_sparsity
				for(i = 0; i < nx_; i++)
				{	for(j = 0; j < n_column; j++)
					{	if( i_column + j < nx_ )
							pattern_hes_[i * nx_ + i_column + j] =
								h[i * n_column + j];
					}
				}
			}
			// Set row and column indices for Lower triangle of Hessian
			// of Lagragian.  These indices are in row major order.
			for(i = 0; i < nx_; i++)
			{	for(j = 0; j < nx_; j++)
				{	if( pattern_hes_[ i * nx_ + j ] )
					if( j <= i )
					{	row_hes_.push_back(i);
						col_hes_.push_back(j);
					}
				}
			}
		}
		else
		{	// Set row and column indices in Jacoian of [f(x), g(x)]
			// for Jacobian of g(x). These indices are in row major order.
			for(i = nf_; i < nfg; i++)
			{	for(j = 0; j < nx_; j++)
				{	row_jac_.push_back(i);
					col_jac_.push_back(j);
				}
			}
			// Set row and column indices for lower triangle of Hessian.
			// These indices are in row major order.
			for(i = 0; i < nx_; i++)
			{	for(j = 0; j <= i; j++)
				{	row_hes_.push_back(i);
					col_hes_.push_back(j);
				}
			}
		}

		// Column order indirect sort of the Jacobian indices
		col_order_jac_.resize( col_jac_.size() );
		index_sort( col_jac_, col_order_jac_ );
	}
	// -----------------------------------------------------------------------
	/*!
	Return dimension information about optimization problem.

	\param[out] n
	is set to the value nx_.

	\param[out] m
	is set to the value ng_.

	\param[out] nnz_jac_g
	is set to ng_ * nx_ (sparsity not yet implemented)

	\param[out] nnz_h_lag
	is set to nx_*(nx_+1)/2 (sparsity not yet implemented)

	\param[out] index_style
	is set to C_STYLE; i.e., zeoro based indexing is used in the
	information passed to Ipopt.
	*/
	virtual bool get_nlp_info(
		Index&          n            ,
		Index&          m            ,
		Index&          nnz_jac_g    ,
		Index&          nnz_h_lag    ,
		IndexStyleEnum& index_style  )
	{
		n         = static_cast<Index>(nx_);
		m         = static_cast<Index>(ng_);
		nnz_jac_g = static_cast<Index>(row_jac_.size());
		nnz_h_lag = static_cast<Index>(row_hes_.size());

# ifndef NDEBUG
		if( ! (sparse_forward_ | sparse_reverse_) )
		{	size_t nnz = static_cast<size_t>(nnz_jac_g);
			CPPAD_ASSERT_UNKNOWN( nnz == ng_ * nx_);
			//
			nnz = static_cast<size_t>(nnz_h_lag);
			CPPAD_ASSERT_UNKNOWN( nnz == (nx_ * (nx_ + 1)) / 2 );
		}
# endif

		// use the fortran index style for row/col entries
		index_style = C_STYLE;

		return true;
	}
	// -----------------------------------------------------------------------
	/*!
	Return bound information about optimization problem.

	\param[in] n
	is the dimension of the domain space for f(x) and g(x); i.e.,
	it must be equal to nx_.

	\param[out] x_l
	is a vector of size nx_.
	The input value of its elements does not matter.
	On output, it is a copy of the lower bound for \f$ x \f$; i.e.,
	xl_.

	\param[out] x_u
	is a vector of size nx_.
	The input value of its elements does not matter.
	On output, it is a copy of the upper bound for \f$ x \f$; i.e.,
	xu_.

	\param[in] m
	is the dimension of the range space for g(x). i.e.,
	it must be equal to ng_.

	\param[out] g_l
	is a vector of size ng_.
	The input value of its elements does not matter.
	On output, it is a copy of the lower bound for \f$ g(x) \f$; i.e., gl_.

	\param[out] g_u
	is a vector of size ng_.
	The input value of its elements does not matter.
	On output, it is a copy of the upper bound for \f$ g(x) \f$; i.e, gu_.
	*/
	virtual bool get_bounds_info(
		Index       n        ,
		Number*     x_l      ,
		Number*     x_u      ,
		Index       m        ,
		Number*     g_l      ,
		Number*     g_u      )
	{	size_t i;
		// here, the n and m we gave IPOPT in get_nlp_info are passed back
		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(n) == nx_);
		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(m) == ng_);

		// pass back bounds
		for(i = 0; i < nx_; i++)
		{	x_l[i] = xl_[i];
			x_u[i] = xu_[i];
		}
		for(i = 0; i < ng_; i++)
		{	g_l[i] = gl_[i];
			g_u[i] = gu_[i];
		}

		return true;
	}
	// -----------------------------------------------------------------------
	/*!
	Return initial x value where optimiation is started.

	\param[in] n
	must be equal to the domain dimension for f(x) and g(x); i.e.,
	it must be equal to nx_.

	\param[in] init_x
	must be equal to true.

	\param[out] x
	is a vector of size nx_.
	The input value of its elements does not matter.
	On output, it is a copy of the initial value for \f$ x \f$; i.e. xi_.

	\param[in] init_z
	must be equal to false.

	\param z_L
	is not used.

	\param z_U
	is not used.

	\param[in] m
	must be equal to the domain dimension for f(x) and g(x); i.e.,
	it must be equal to ng_.

	\param init_lambda
	must be equal to false.

	\param lambda
	is not used.
	*/
	virtual bool get_starting_point(
		Index           n            ,
		bool            init_x       ,
		Number*         x            ,
		bool            init_z       ,
		Number*         z_L          ,
		Number*         z_U          ,
		Index           m            ,
		bool            init_lambda  ,
		Number*         lambda       )
	{	size_t j;

		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(n) == nx_ );
		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(m) == ng_ );
		CPPAD_ASSERT_UNKNOWN(init_x == true);
		CPPAD_ASSERT_UNKNOWN(init_z == false);
		CPPAD_ASSERT_UNKNOWN(init_lambda == false);

		for(j = 0; j < nx_; j++)
			x[j] = xi_[j];

		return true;
	}
	// -----------------------------------------------------------------------
	/*!
	Evaluate the objective fucntion f(x).

	\param[in] n
	is the dimension of the argument space for f(x); i.e., must be equal nx_.

	\param[in] x
	is a vector of size nx_ containing the point at which to evaluate
	the function sum_i f_i (x).

	\param[in] new_x
	is false if the previous call to any one of the
	\ref Evaluation_Methods used the same value for x.

	\param[out] obj_value
	is the value of the objective sum_i f_i (x) at this value of x.

	\return
	The return value is always true; see \ref Evaluation_Methods.
	*/
	virtual bool eval_f(
		Index          n           ,
		const Number*  x           ,
		bool           new_x       ,
		Number&        obj_value   )
	{	size_t i;
		if( new_x )
			cache_new_x(x);
		//
		double sum = 0.0;
		for(i = 0; i < nf_; i++)
			sum += fg0_[i];
		obj_value = static_cast<Number>(sum);
		return true;
	}
	// -----------------------------------------------------------------------
	/*!
	Evaluate the gradient of f(x).

	\param[in] n
	is the dimension of the argument space for f(x); i.e., must be equal nx_.

	\param[in] x
	has a vector of size nx_ containing the point at which to evaluate
	the gradient of f(x).

	\param[in] new_x
	is false if the previous call to any one of the
	\ref Evaluation_Methods used the same value for x.

	\param[out] grad_f
	is a vector of size nx_.
	The input value of its elements does not matter.
	The output value of its elements is the gradient of f(x)
	at this value of.

	\return
	The return value is always true; see \ref Evaluation_Methods.
	*/
	virtual bool eval_grad_f(
		Index           n        ,
		const Number*   x        ,
		bool            new_x    ,
		Number*         grad_f   )
	{	size_t i;
		if( new_x )
			cache_new_x(x);
		//
		Dvector w(nf_ + ng_), dw(nx_);
		for(i = 0; i < nf_; i++)
			w[i] = 1.0;
		for(i = 0; i < ng_; i++)
			w[nf_ + i] = 0.0;
		dw = adfun_.Reverse(1, w);
		for(i = 0; i < nx_; i++)
			grad_f[i] = dw[i];
		return true;
	}
	// -----------------------------------------------------------------------
	/*!
	Evaluate the function g(x).

	\param[in] n
	is the dimension of the argument space for g(x); i.e., must be equal nx_.

	\param[in] x
	has a vector of size n containing the point at which to evaluate
	the gradient of g(x).

	\param[in] new_x
	is false if the previous call to any one of the
	\ref Evaluation_Methods used the same value for x.

	\param[in] m
	is the dimension of the range space for g(x); i.e., must be equal to ng_.

	\param[out] g
	is a vector of size ng_.
	The input value of its elements does not matter.
	The output value of its elements is
	the value of the function g(x) at this value of x.

	\return
	The return value is always true; see \ref Evaluation_Methods.
	*/
	virtual bool eval_g(
		Index   n            ,
		const   Number* x    ,
		bool    new_x        ,
		Index   m            ,
		Number* g            )
	{	size_t i;
		if( new_x )
			cache_new_x(x);
		//
		for(i = 0; i < ng_; i++)
			g[i] = fg0_[nf_ + i];
		return true;
	}
	// -----------------------------------------------------------------------
	/*!
	Evaluate the Jacobian of g(x).

	\param[in] n
	is the dimension of the argument space for g(x);
	i.e., must be equal nx_.

	\param x
	If values is not NULL,
	x is a vector of size nx_ containing the point at which to evaluate
	the gradient of g(x).

	\param[in] new_x
	is false if the previous call to any one of the
	\ref Evaluation_Methods used the same value for  x.

	\param[in] m
	is the dimension of the range space for g(x);
	i.e., must be equal to ng_.

	\param[in] nele_jac
	is the number of possibly non-zero elements in the Jacobian of g(x);
	i.e., must be equal to ng_ * nx_.

	\param iRow
	if values is not NULL, iRow is not defined.
	if values is NULL, iRow
	is a vector with size nele_jac.
	The input value of its elements does not matter.
	On output,
	For <tt>k = 0 , ... , nele_jac-1, iRow[k]</tt> is the
	base zero row index for the
	k-th possibly non-zero entry in the Jacobian of g(x).

	\param jCol
	if values is not NULL, jCol is not defined.
	if values is NULL, jCol
	is a vector with size nele_jac.
	The input value of its elements does not matter.
	On output,
	For <tt>k = 0 , ... , nele_jac-1, jCol[k]</tt> is the
	base zero column index for the
	k-th possibly non-zero entry in the Jacobian of g(x).

	\param values
	if \c values is not \c NULL, \c values
	is a vector with size \c nele_jac.
	The input value of its elements does not matter.
	On output,
	For <tt>k = 0 , ... , nele_jac-1, values[k]</tt> is the
	value for the
	k-th possibly non-zero entry in the Jacobian of g(x).

	\return
	The return value is always true; see \ref Evaluation_Methods.
	*/
	virtual bool eval_jac_g(
		Index n,
		const Number* x,
		bool new_x,

		Index m,
		Index nele_jac,
		Index* iRow,
		Index *jCol,

		Number* values)
	{	size_t i, j, k, ell;
		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(m)         == ng_ );
		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(n)         == nx_ );
		//
		size_t nk = row_jac_.size();
		CPPAD_ASSERT_UNKNOWN( static_cast<size_t>(nele_jac) == nk );
		//
		if( new_x )
			cache_new_x(x);

		if( values == NULL )
		{	for(k = 0; k < nk; k++)
			{	i = row_jac_[k];
				j = col_jac_[k];
				CPPAD_ASSERT_UNKNOWN( i >= nf_ );
				iRow[k] = static_cast<Index>(i - nf_);
				jCol[k] = static_cast<Index>(j);
			}
			return true;
		}
		//
		if( nk == 0 )
			return true;
		//
		if( sparse_forward_ )
		{	Dvector jac(nk);
			adfun_.SparseJacobianForward(
				x0_ , pattern_jac_, row_jac_, col_jac_, jac, work_jac_
			);
			for(k = 0; k < nk; k++)
				values[k] = jac[k];
		}
		else if( sparse_reverse_ )
		{	Dvector jac(nk);
			adfun_.SparseJacobianReverse(
				x0_ , pattern_jac_, row_jac_, col_jac_, jac, work_jac_
			);
			for(k = 0; k < nk; k++)
				values[k] = jac[k];
		}
		else if( nx_ < ng_ )
		{	// use forward mode
			Dvector x1(nx_), fg1(nf_ + ng_);
			for(j = 0; j < nx_; j++)
				x1[j] = 0.0;
			// index in col_order_jac_ of next entry
			ell = 0;
			k   = col_order_jac_[ell];
			for(j = 0; j < nx_; j++)
			{	// compute j-th column of Jacobian of g(x)
				x1[j] = 1.0;
				fg1 = adfun_.Forward(1, x1);
				while( ell < nk && col_jac_[k] <= j )
				{	CPPAD_ASSERT_UNKNOWN( col_jac_[k] == j );
					i = row_jac_[k];
					CPPAD_ASSERT_UNKNOWN( i >= nf_ )
					values[k] = fg1[i];
					ell++;
					if( ell < nk )
						k = col_order_jac_[ell];
				}
				x1[j] = 0.0;
			}
		}
		else
		{	// user reverse mode
			size_t nfg = nf_ + ng_;
			// user reverse mode
			Dvector w(nfg), dw(nx_);
			for(i = 0; i < nfg; i++)
				w[i] = 0.0;
			// index in row_jac_ of next entry
			k = 0;
			for(i = nf_; i < nfg; i++)
			{	// compute i-th row of Jacobian of g(x)
				w[i] = 1.0;
				dw = adfun_.Reverse(1, w);
				while( k < nk && row_jac_[k] <= i )
				{	CPPAD_ASSERT_UNKNOWN( row_jac_[k] == i );
					j = col_jac_[k];
					values[k] = dw[j];
					k++;
				}
				w[i] = 0.0;
			}
		}
		return true;
	}
	// -----------------------------------------------------------------------
	/*!
	Evaluate the Hessian of the Lagragian

	\section The_Hessian_of_the_Lagragian The Hessian of the Lagragian
	The Hessian of the Lagragian is defined as
	\f[
	H(x, \sigma, \lambda )
	=
	\sigma \nabla^2 f(x) + \sum_{i=0}^{m-1} \lambda_i \nabla^2 g(x)_i
	\f]

	\param[in] n
	is the dimension of the argument space for g(x);
	i.e., must be equal nx_.

	\param x
	if values is not NULL, x
	is a vector of size nx_ containing the point at which to evaluate
	the Hessian of the Lagragian.

	\param[in] new_x
	is false if the previous call to any one of the
	\ref Evaluation_Methods used the same value for x.

	\param[in] obj_factor
	the value \f$ \sigma \f$ multiplying the Hessian of
	f(x) in the expression for \ref The_Hessian_of_the_Lagragian.

	\param[in] m
	is the dimension of the range space for g(x);
	i.e., must be equal to ng_.

	\param[in] lambda
	if values is not NULL, lambda
	is a vector of size ng_ specifing the value of \f$ \lambda \f$
	in the expression for \ref The_Hessian_of_the_Lagragian.

	\param[in] new_lambda
	is true if the previous call to eval_h had the same value for
	lambda and false otherwise.
	(Not currently used.)

	\param[in] nele_hess
	is the number of possibly non-zero elements in the
	Hessian of the Lagragian;
	i.e., must be equal to nx_*(nx_+1)/2.

	\param iRow
	if values is not NULL, iRow is not defined.
	if values is NULL, iRow
	is a vector with size nele_hess.
	The input value of its elements does not matter.
	On output,
	For <tt>k = 0 , ... , nele_hess-1, iRow[k]</tt> is the
	base zero row index for the
	k-th possibly non-zero entry in the Hessian fo the Lagragian.

	\param jCol
	if values is not NULL, jCol is not defined.
	if values is NULL, jCol
	is a vector with size nele_hess.
	The input value of its elements does not matter.
	On output,
	For <tt>k = 0 , ... , nele_hess-1, jCol[k]</tt> is the
	base zero column index for the
	k-th possibly non-zero entry in the Hessian of the Lagragian.

	\param values
	if values is not NULL, it
	is a vector with size nele_hess.
	The input value of its elements does not matter.
	On output,
	For <tt>k = 0 , ... , nele_hess-1, values[k]</tt> is the
	value for the
	k-th possibly non-zero entry in the Hessian of the Lagragian.

	\return
	The return value is always true; see \ref Evaluation_Methods.
	*/
	virtual bool eval_h(
		Index         n              ,
		const Number* x              ,
		bool          new_x          ,
		Number        obj_factor     ,
		Index         m              ,
		const Number* lambda         ,
		bool          new_lambda     ,
		Index         nele_hess      ,
		Index*        iRow           ,
		Index*        jCol           ,
		Number*       values         )
	{	size_t i, j, k;
		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(m) == ng_ );
		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(n) == nx_ );
		//
		size_t nk = row_hes_.size();
		CPPAD_ASSERT_UNKNOWN( static_cast<size_t>(nele_hess) == nk );
		//
		if( new_x )
			cache_new_x(x);
		//
		if( values == NULL )
		{	for(k = 0; k < nk; k++)
			{	i = row_hes_[k];
				j = col_hes_[k];
				iRow[k] = static_cast<Index>(i);
				jCol[k] = static_cast<Index>(j);
			}
			return true;
		}
		//
		if( nk == 0 )
			return true;

		// weigting vector for Lagragian
		Dvector w(nf_ + ng_);
		for(i = 0; i < nf_; i++)
			w[i] = obj_factor;
		for(i = 0; i < ng_; i++)
			w[i + nf_] = lambda[i];
		//
		if( sparse_forward_ | sparse_reverse_ )
		{	Dvector hes(nk);
			adfun_.SparseHessian(
				x0_, w, pattern_hes_, row_hes_, col_hes_, hes, work_hes_
			);
			for(k = 0; k < nk; k++)
				values[k] = hes[k];
		}
		else
		{	Dvector hes(nx_ * nx_);
			hes = adfun_.Hessian(x0_, w);
			for(k = 0; k < nk; k++)
			{	i = row_hes_[k];
				j = col_hes_[k];
				values[k] = hes[i * nx_ + j];
			}
		}
		return true;
	}
	// ----------------------------------------------------------------------
	/*!
	Pass solution information from Ipopt to users solution structure.

	\param[in] status
	is value that the Ipopt solution status
	which gets mapped to a correponding value for
	\n
	<tt>solution_.status</tt>

	\param[in] n
	is the dimension of the domain space for f(x) and g(x); i.e.,
	it must be equal to nx_.

	\param[in] x
	is a vector with size nx_ specifing the final solution.
	This is the output value for
	\n
	<tt>solution_.x</tt>

	\param[in] z_L
	is a vector with size nx_ specifing the Lagragian multipliers for the
	constraint \f$ x^l \leq x \f$.
	This is the output value for
	\n
	<tt>solution_.zl</tt>

	\param[in] z_U
	is a vector with size nx_ specifing the Lagragian multipliers for the
	constraint \f$ x \leq x^u \f$.
	This is the output value for
	\n
	<tt>solution_.zu</tt>

	\param[in] m
	is the dimension of the range space for g(x). i.e.,
	it must be equal to ng_.

	\param[in] g
	is a vector with size ng_ containing the value of the constraint function
	g(x) at the final solution x.
	This is the output value for
	\n
	<tt>solution_.g</tt>

	\param[in] lambda
	is a vector with size ng_ specifing the Lagragian multipliers for the
	constraints \f$ g^l \leq g(x) \leq g^u \f$.
	This is the output value for
	\n
	<tt>solution_.lambda</tt>

	\param[in] obj_value
	is the value of the objective function f(x) at the final solution x.
	This is the output value for
	\n
	<tt>solution_.obj_value</tt>

	\param[in] ip_data
	is unspecified (by Ipopt) and hence not used.

	\param[in] ip_cq
	is unspecified (by Ipopt) and hence not used.

	\par solution_[out]
	this is a reference to the solution argument
	in the constructor for solve_callback.
	The results are stored here
	(see documentation above).
	*/
	virtual void finalize_solution(
		Ipopt::SolverReturn               status    ,
		Index                             n         ,
		const Number*                     x         ,
		const Number*                     z_L       ,
		const Number*                     z_U       ,
		Index                             m         ,
		const Number*                     g         ,
		const Number*                     lambda    ,
		Number                            obj_value ,
		const Ipopt::IpoptData*           ip_data   ,
		Ipopt::IpoptCalculatedQuantities* ip_cq
	)
	{	size_t i, j;

		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(n) == nx_ );
		CPPAD_ASSERT_UNKNOWN(static_cast<size_t>(m) == ng_ );

		switch(status)
		{	// convert status from Ipopt enum to solve_result<Dvector> enum
			case Ipopt::SUCCESS:
			solution_.status = solve_result<Dvector>::success;
			break;

			case Ipopt::MAXITER_EXCEEDED:
			solution_.status =
				solve_result<Dvector>::maxiter_exceeded;
			break;

			case Ipopt::STOP_AT_TINY_STEP:
			solution_.status =
				solve_result<Dvector>::stop_at_tiny_step;
			break;

			case Ipopt::STOP_AT_ACCEPTABLE_POINT:
			solution_.status =
				solve_result<Dvector>::stop_at_acceptable_point;
			break;

			case Ipopt::LOCAL_INFEASIBILITY:
			solution_.status =
				solve_result<Dvector>::local_infeasibility;
			break;

			case Ipopt::USER_REQUESTED_STOP:
			solution_.status =
				solve_result<Dvector>::user_requested_stop;
			break;

			case Ipopt::DIVERGING_ITERATES:
			solution_.status =
				solve_result<Dvector>::diverging_iterates;
			break;

			case Ipopt::RESTORATION_FAILURE:
			solution_.status =
				solve_result<Dvector>::restoration_failure;
			break;

			case Ipopt::ERROR_IN_STEP_COMPUTATION:
			solution_.status =
				solve_result<Dvector>::error_in_step_computation;
			break;

			case Ipopt::INVALID_NUMBER_DETECTED:
			solution_.status =
				solve_result<Dvector>::invalid_number_detected;
			break;

			case Ipopt::INTERNAL_ERROR:
			solution_.status =
				solve_result<Dvector>::internal_error;
			break;

			default:
			solution_.status =
				solve_result<Dvector>::unknown;
		}

		solution_.x.resize(nx_);
		solution_.zl.resize(nx_);
		solution_.zu.resize(nx_);
		for(j = 0; j < nx_; j++)
		{	solution_.x[j]   = x[j];
			solution_.zl[j]  = z_L[j];
			solution_.zu[j]  = z_U[j];
		}
		solution_.g.resize(ng_);
		solution_.lambda.resize(ng_);
		for(i = 0; i < ng_; i++)
		{	solution_.g[i]      = g[i];
			solution_.lambda[i] = lambda[i];
		}
		solution_.obj_value = obj_value;
		return;
	}
};

} // end namespace ipopt
} // END_CPPAD_NAMESPACE

# endif
