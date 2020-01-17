// $Id$
# ifndef CPPAD_EXAMPLE_MAT_MUL_HPP
# define CPPAD_EXAMPLE_MAT_MUL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin atomic_mat_mul.hpp$$
$spell
	Taylor
	ty
	px
	CppAD
	jac
	hes
	nr
	nc
$$

$section Matrix Multiply as an Atomic Operation$$

$head See Also$$
$cref atomic_eigen_mat_mul.hpp$$

$head Matrix Dimensions$$
This example puts the matrix dimensions in the atomic function arguments,
instead of the $cref/constructor/atomic_ctor/$$, so that they can
be different for different calls to the atomic function.
These dimensions are:
$table
$icode nr_left$$ $cnext number of rows in the left matrix $rend
$icode n_middle$$ $cnext rows in the left matrix and columns in right $rend
$icode nc_right$$ $cnext number of columns in the right matrix
$tend

$head Start Class Definition$$
$srccode%cpp% */
# include <cppad/cppad.hpp>
namespace { // Begin empty namespace
using CppAD::vector;
//
using CppAD::set_union;
//
// matrix result = left * right
class atomic_mat_mul : public CppAD::atomic_base<double> {
/* %$$
$head Constructor$$
$srccode%cpp% */
public:
	// ---------------------------------------------------------------------
	// constructor
	atomic_mat_mul(void) : CppAD::atomic_base<double>("mat_mul")
	{ }
private:
/* %$$
$head Left Operand Element Index$$
Index in the Taylor coefficient matrix $icode tx$$ of a left matrix element.
$srccode%cpp% */
	size_t left(
		size_t i        , // left matrix row index
		size_t j        , // left matrix column index
		size_t k        , // Taylor coeffocient order
		size_t nk       , // number of Taylor coefficients in tx
		size_t nr_left  , // rows in left matrix
		size_t n_middle , // rows in left and columns in right
		size_t nc_right ) // columns in right matrix
	{	assert( i < nr_left );
		assert( j < n_middle );
		return (3 + i * n_middle + j) * nk + k;
	}
/* %$$
$head Right Operand Element Index$$
Index in the Taylor coefficient matrix $icode tx$$ of a right matrix element.
$srccode%cpp% */
	size_t right(
		size_t i        , // right matrix row index
		size_t j        , // right matrix column index
		size_t k        , // Taylor coeffocient order
		size_t nk       , // number of Taylor coefficients in tx
		size_t nr_left  , // rows in left matrix
		size_t n_middle , // rows in left and columns in right
		size_t nc_right ) // columns in right matrix
	{	assert( i < n_middle );
		assert( j < nc_right );
		size_t offset = 3 + nr_left * n_middle;
		return (offset + i * nc_right + j) * nk + k;
	}
/* %$$
$head Result Element Index$$
Index in the Taylor coefficient matrix $icode ty$$ of a result matrix element.
$srccode%cpp% */
	size_t result(
		size_t i        , // result matrix row index
		size_t j        , // result matrix column index
		size_t k        , // Taylor coeffocient order
		size_t nk       , // number of Taylor coefficients in ty
		size_t nr_left  , // rows in left matrix
		size_t n_middle , // rows in left and columns in right
		size_t nc_right ) // columns in right matrix
	{	assert( i < nr_left  );
		assert( j < nc_right );
		return (i * nc_right + j) * nk + k;
	}
/* %$$
$head Forward Matrix Multiply$$
Forward mode multiply Taylor coefficients in $icode tx$$ and sum into
$icode ty$$ (for one pair of left and right orders)
$srccode%cpp% */
	void forward_multiply(
		size_t                 k_left   , // order for left coefficients
		size_t                 k_right  , // order for right coefficients
		const vector<double>&  tx       , // domain space Taylor coefficients
		      vector<double>&  ty       , // range space Taylor coefficients
		size_t                 nr_left  , // rows in left matrix
		size_t                 n_middle , // rows in left and columns in right
		size_t                 nc_right ) // columns in right matrix
	{
		size_t nx       = 3 + (nr_left + nc_right) * n_middle;
		size_t nk       = tx.size() / nx;
# ifndef NDEBUG
		size_t ny       = nr_left * nc_right;
		assert( nk == ty.size() / ny );
# endif
		//
		size_t k_result = k_left + k_right;
		assert( k_result < nk );
		//
		for(size_t i = 0; i < nr_left; i++)
		{	for(size_t j = 0; j < nc_right; j++)
			{	double sum = 0.0;
				for(size_t ell = 0; ell < n_middle; ell++)
				{	size_t i_left  = left(
						i, ell, k_left, nk, nr_left, n_middle, nc_right
					);
					size_t i_right = right(
						ell, j,  k_right, nk, nr_left, n_middle, nc_right
					);
					sum           += tx[i_left] * tx[i_right];
				}
				size_t i_result = result(
					i, j, k_result, nk, nr_left, n_middle, nc_right
				);
				ty[i_result]   += sum;
			}
		}
	}
/* %$$
$head Reverse Matrix Multiply$$
Reverse mode partials of Taylor coefficients and sum into $icode px$$
(for one pair of left and right orders)
$srccode%cpp% */
	void reverse_multiply(
		size_t                 k_left  , // order for left coefficients
		size_t                 k_right , // order for right coefficients
		const vector<double>&  tx      , // domain space Taylor coefficients
		const vector<double>&  ty      , // range space Taylor coefficients
		      vector<double>&  px      , // partials w.r.t. tx
		const vector<double>&  py      , // partials w.r.t. ty
		size_t                 nr_left  , // rows in left matrix
		size_t                 n_middle , // rows in left and columns in right
		size_t                 nc_right ) // columns in right matrix
	{
		size_t nx       = 3 + (nr_left + nc_right) * n_middle;
		size_t nk       = tx.size() / nx;
# ifndef NDEBUG
		size_t ny       = nr_left * nc_right;
		assert( nk == ty.size() / ny );
# endif
		assert( tx.size() == px.size() );
		assert( ty.size() == py.size() );
		//
		size_t k_result = k_left + k_right;
		assert( k_result < nk );
		//
		for(size_t i = 0; i < nr_left; i++)
		{	for(size_t j = 0; j < nc_right; j++)
			{	size_t i_result = result(
					i, j, k_result, nk, nr_left, n_middle, nc_right
				);
				for(size_t ell = 0; ell < n_middle; ell++)
				{	size_t i_left  = left(
						i, ell, k_left, nk, nr_left, n_middle, nc_right
					);
					size_t i_right = right(
						ell, j,  k_right, nk, nr_left, n_middle, nc_right
					);
					// sum        += tx[i_left] * tx[i_right];
					px[i_left]    += tx[i_right] * py[i_result];
					px[i_right]   += tx[i_left]  * py[i_result];
				}
			}
		}
		return;
	}
/* %$$
$head forward$$
Routine called by CppAD during $cref Forward$$ mode.
$srccode%cpp% */
	virtual bool forward(
		size_t                    q ,
		size_t                    p ,
		const vector<bool>&      vx ,
		      vector<bool>&      vy ,
		const vector<double>&    tx ,
		      vector<double>&    ty
	)
	{	size_t n_order  = p + 1;
		size_t nr_left  = size_t( tx[ 0 * n_order + 0 ] );
		size_t n_middle = size_t( tx[ 1 * n_order + 0 ] );
		size_t nc_right = size_t( tx[ 2 * n_order + 0 ] );
# ifndef NDEBUG
		size_t nx       = 3 + (nr_left + nc_right) * n_middle;
		size_t ny       = nr_left * nc_right;
# endif
		assert( vx.size() == 0 || nx == vx.size() );
		assert( vx.size() == 0 || ny == vy.size() );
		assert( nx * n_order == tx.size() );
		assert( ny * n_order == ty.size() );
		size_t i, j, ell;

		// check if we are computing vy information
		if( vx.size() > 0 )
		{	size_t nk = 1;
			size_t k  = 0;
			for(i = 0; i < nr_left; i++)
			{	for(j = 0; j < nc_right; j++)
				{	bool var = false;
					for(ell = 0; ell < n_middle; ell++)
					{	size_t i_left  = left(
							i, ell, k, nk, nr_left, n_middle, nc_right
						);
						size_t i_right = right(
							ell, j, k, nk, nr_left, n_middle, nc_right
						);
						bool   nz_left = vx[i_left] |(tx[i_left]  != 0.);
						bool  nz_right = vx[i_right]|(tx[i_right] != 0.);
						// if not multiplying by the constant zero
						if( nz_left & nz_right )
								var |= bool(vx[i_left]) | bool(vx[i_right]);
					}
					size_t i_result = result(
						i, j, k, nk, nr_left, n_middle, nc_right
					);
					vy[i_result] = var;
				}
			}
		}

		// initialize result as zero
		size_t k;
		for(i = 0; i < nr_left; i++)
		{	for(j = 0; j < nc_right; j++)
			{	for(k = q; k <= p; k++)
				{	size_t i_result = result(
						i, j, k, n_order, nr_left, n_middle, nc_right
					);
					ty[i_result] = 0.0;
				}
			}
		}
		for(k = q; k <= p; k++)
		{	// sum the produces that result in order k
			for(ell = 0; ell <= k; ell++)
				forward_multiply(
					ell, k - ell, tx, ty, nr_left, n_middle, nc_right
				);
		}

		// all orders are implented, so always return true
		return true;
	}
/* %$$
$head reverse$$
Routine called by CppAD during $cref Reverse$$ mode.
$srccode%cpp% */
	virtual bool reverse(
		size_t                     p ,
		const vector<double>&     tx ,
		const vector<double>&     ty ,
		      vector<double>&     px ,
		const vector<double>&     py
	)
	{	size_t n_order  = p + 1;
		size_t nr_left  = size_t( tx[ 0 * n_order + 0 ] );
		size_t n_middle = size_t( tx[ 1 * n_order + 0 ] );
		size_t nc_right = size_t( tx[ 2 * n_order + 0 ] );
# ifndef NDEBUG
		size_t nx       = 3 + (nr_left + nc_right) * n_middle;
		size_t ny       = nr_left * nc_right;
# endif
		assert( nx * n_order == tx.size() );
		assert( ny * n_order == ty.size() );
		assert( px.size() == tx.size() );
		assert( py.size() == ty.size() );

		// initialize summation
		for(size_t i = 0; i < px.size(); i++)
			px[i] = 0.0;

		// number of orders to differentiate
		size_t k = n_order;
		while(k--)
		{	// differentiate the produces that result in order k
			for(size_t ell = 0; ell <= k; ell++)
				reverse_multiply(
					ell, k - ell, tx, ty, px, py, nr_left, n_middle, nc_right
				);
		}

		// all orders are implented, so always return true
		return true;
	}
/* %$$
$head for_sparse_jac$$
Routines called by CppAD during $cref ForSparseJac$$.
$srccode%cpp% */
	// boolean sparsity patterns
	virtual bool for_sparse_jac(
		size_t                                q ,
		const vector<bool>&                   r ,
		      vector<bool>&                   s ,
		const vector<double>&                 x )
	{
		size_t nr_left  = size_t( CppAD::Integer( x[0] ) );
		size_t n_middle = size_t( CppAD::Integer( x[1] ) );
		size_t nc_right = size_t( CppAD::Integer( x[2] ) );
# ifndef NDEBUG
		size_t  nx      = 3 + (nr_left + nc_right) * n_middle;
		size_t  ny      = nr_left * nc_right;
# endif
		assert( nx     == x.size() );
		assert( nx * q == r.size() );
		assert( ny * q == s.size() );
		size_t p;

		// sparsity for S(x) = f'(x) * R
		size_t nk = 1;
		size_t k  = 0;
		for(size_t i = 0; i < nr_left; i++)
		{	for(size_t j = 0; j < nc_right; j++)
			{	size_t i_result = result(
					i, j, k, nk, nr_left, n_middle, nc_right
				);
				for(p = 0; p < q; p++)
					s[i_result * q + p] = false;
				for(size_t ell = 0; ell < n_middle; ell++)
				{	size_t i_left  = left(
						i, ell, k, nk, nr_left, n_middle, nc_right
					);
					size_t i_right = right(
						ell, j, k, nk, nr_left, n_middle, nc_right
					);
					for(p = 0; p < q; p++)
					{	// cast avoids Microsoft warning (should not be needed)
						s[i_result * q + p] |= bool( r[i_left * q + p ] );
						s[i_result * q + p] |= bool( r[i_right * q + p ] );
					}
				}
			}
		}
		return true;
	}
	// set sparsity patterns
	virtual bool for_sparse_jac(
		size_t                                q ,
		const vector< std::set<size_t> >&     r ,
		      vector< std::set<size_t> >&     s ,
		const vector<double>&                 x )
	{
		size_t nr_left  = size_t( CppAD::Integer( x[0] ) );
		size_t n_middle = size_t( CppAD::Integer( x[1] ) );
		size_t nc_right = size_t( CppAD::Integer( x[2] ) );
# ifndef NDEBUG
		size_t  nx      = 3 + (nr_left + nc_right) * n_middle;
		size_t  ny      = nr_left * nc_right;
# endif
		assert( nx == x.size() );
		assert( nx == r.size() );
		assert( ny == s.size() );

		// sparsity for S(x) = f'(x) * R
		size_t nk = 1;
		size_t k  = 0;
		for(size_t i = 0; i < nr_left; i++)
		{	for(size_t j = 0; j < nc_right; j++)
			{	size_t i_result = result(
					i, j, k, nk, nr_left, n_middle, nc_right
				);
				s[i_result].clear();
				for(size_t ell = 0; ell < n_middle; ell++)
				{	size_t i_left  = left(
						i, ell, k, nk, nr_left, n_middle, nc_right
					);
					size_t i_right = right(
						ell, j, k, nk, nr_left, n_middle, nc_right
					);
					//
					s[i_result] = set_union(s[i_result], r[i_left] );
					s[i_result] = set_union(s[i_result], r[i_right] );
				}
			}
		}
		return true;
	}
/* %$$
$head rev_sparse_jac$$
Routines called by CppAD during $cref RevSparseJac$$.
$srccode%cpp% */
	// boolean sparsity patterns
	virtual bool rev_sparse_jac(
		size_t                                q ,
		const vector<bool>&                  rt ,
		      vector<bool>&                  st ,
		const vector<double>&                 x )
	{
		size_t nr_left  = size_t( CppAD::Integer( x[0] ) );
		size_t n_middle = size_t( CppAD::Integer( x[1] ) );
		size_t nc_right = size_t( CppAD::Integer( x[2] ) );
		size_t  nx      = 3 + (nr_left + nc_right) * n_middle;
# ifndef NDEBUG
		size_t  ny      = nr_left * nc_right;
# endif
		assert( nx     == x.size() );
		assert( nx * q == st.size() );
		assert( ny * q == rt.size() );
		size_t i, j, p;

		// initialize
		for(i = 0; i < nx; i++)
		{	for(p = 0; p < q; p++)
				st[ i * q + p ] = false;
		}

		// sparsity for S(x)^T = f'(x)^T * R^T
		size_t nk = 1;
		size_t k  = 0;
		for(i = 0; i < nr_left; i++)
		{	for(j = 0; j < nc_right; j++)
			{	size_t i_result = result(
					i, j, k, nk, nr_left, n_middle, nc_right
				);
				for(size_t ell = 0; ell < n_middle; ell++)
				{	size_t i_left  = left(
						i, ell, k, nk, nr_left, n_middle, nc_right
					);
					size_t i_right = right(
						ell, j, k, nk, nr_left, n_middle, nc_right
					);
					for(p = 0; p < q; p++)
					{	st[i_left * q + p] |= bool( rt[i_result * q + p] );
						st[i_right* q + p] |= bool( rt[i_result * q + p] );
					}
				}
			}
		}
		return true;
	}
	// set sparsity patterns
	virtual bool rev_sparse_jac(
		size_t                                q ,
		const vector< std::set<size_t> >&    rt ,
		      vector< std::set<size_t> >&    st ,
		const vector<double>&                 x )
	{
		size_t nr_left  = size_t( CppAD::Integer( x[0] ) );
		size_t n_middle = size_t( CppAD::Integer( x[1] ) );
		size_t nc_right = size_t( CppAD::Integer( x[2] ) );
		size_t  nx      = 3 + (nr_left + nc_right) * n_middle;
# ifndef NDEBUG
		size_t  ny        = nr_left * nc_right;
# endif
		assert( nx == x.size() );
		assert( nx == st.size() );
		assert( ny == rt.size() );
		size_t i, j;

		// initialize
		for(i = 0; i < nx; i++)
			st[i].clear();

		// sparsity for S(x)^T = f'(x)^T * R^T
		size_t nk = 1;
		size_t k  = 0;
		for(i = 0; i < nr_left; i++)
		{	for(j = 0; j < nc_right; j++)
			{	size_t i_result = result(
					i, j, k, nk, nr_left, n_middle, nc_right
				);
				for(size_t ell = 0; ell < n_middle; ell++)
				{	size_t i_left  = left(
						i, ell, k, nk, nr_left, n_middle, nc_right
					);
					size_t i_right = right(
						ell, j, k, nk, nr_left, n_middle, nc_right
					);
					//
					st[i_left]  = set_union(st[i_left],  rt[i_result]);
					st[i_right] = set_union(st[i_right], rt[i_result]);
				}
			}
		}
		return true;
	}
/* %$$
$head rev_sparse_hes$$
Routines called by $cref RevSparseHes$$.
$srccode%cpp% */
	// set sparsity patterns
	virtual bool rev_sparse_hes(
		const vector<bool>&                   vx,
		const vector<bool>&                   s ,
		      vector<bool>&                   t ,
		size_t                                q ,
		const vector< std::set<size_t> >&     r ,
		const vector< std::set<size_t> >&     u ,
		      vector< std::set<size_t> >&     v ,
		const vector<double>&                 x )
	{
		size_t nr_left  = size_t( CppAD::Integer( x[0] ) );
		size_t n_middle = size_t( CppAD::Integer( x[1] ) );
		size_t nc_right = size_t( CppAD::Integer( x[2] ) );
		size_t  nx        = 3 + (nr_left + nc_right) * n_middle;
# ifndef NDEBUG
		size_t  ny        = nr_left * nc_right;
# endif
		assert( x.size()  == nx );
		assert( vx.size() == nx );
		assert( t.size()  == nx );
		assert( r.size()  == nx );
		assert( v.size()  == nx );
		assert( s.size()  == ny );
		assert( u.size()  == ny );
		//
		size_t i, j;
		//
		// initilaize sparsity patterns as false
		for(j = 0; j < nx; j++)
		{	t[j] = false;
			v[j].clear();
		}
		size_t nk = 1;
		size_t k  = 0;
		for(i = 0; i < nr_left; i++)
		{	for(j = 0; j < nc_right; j++)
			{	size_t i_result = result(
					i, j, k, nk, nr_left, n_middle, nc_right
				);
				for(size_t ell = 0; ell < n_middle; ell++)
				{	size_t i_left  = left(
						i, ell, k, nk, nr_left, n_middle, nc_right
					);
					size_t i_right = right(
						ell, j, k, nk, nr_left, n_middle, nc_right
					);
					//
					// Compute sparsity for T(x) = S(x) * f'(x).
					// We need not use vx with f'(x) back propagation.
					t[i_left]  |= bool( s[i_result] );
					t[i_right] |= bool( s[i_result] );

					// V(x) = f'(x)^T * U(x) +  S(x) * f''(x) * R
					// U(x) = g''(y) * f'(x) * R
					// S(x) = g'(y)

					// back propagate f'(x)^T * U(x)
					// (no need to use vx with f'(x) propogation)
					v[i_left]  = set_union(v[i_left],  u[i_result] );
					v[i_right] = set_union(v[i_right], u[i_result] );

					// back propagate S(x) * f''(x) * R
					// (here is where we must check for cross terms)
					if( s[i_result] & vx[i_left] & vx[i_right] )
					{	v[i_left]  = set_union(v[i_left],  r[i_right] );
						v[i_right] = set_union(v[i_right], r[i_left]  );
					}
				}
			}
		}
		return true;
	}
	// bool sparsity
	virtual bool rev_sparse_hes(
		const vector<bool>&                   vx,
		const vector<bool>&                   s ,
		      vector<bool>&                   t ,
		size_t                                q ,
		const vector<bool>&                   r ,
		const vector<bool>&                   u ,
		      vector<bool>&                   v ,
		const vector<double>&                 x )
	{
		size_t nr_left  = size_t( CppAD::Integer( x[0] ) );
		size_t n_middle = size_t( CppAD::Integer( x[1] ) );
		size_t nc_right = size_t( CppAD::Integer( x[2] ) );
		size_t  nx        = 3 + (nr_left + nc_right) * n_middle;
# ifndef NDEBUG
		size_t  ny        = nr_left * nc_right;
# endif
		assert( x.size()  == nx );
		assert( vx.size() == nx );
		assert( t.size()  == nx );
		assert( r.size()  == nx * q );
		assert( v.size()  == nx * q );
		assert( s.size()  == ny );
		assert( u.size()  == ny * q );
		size_t i, j, p;
		//
		// initilaize sparsity patterns as false
		for(j = 0; j < nx; j++)
		{	t[j] = false;
			for(p = 0; p < q; p++)
				v[j * q + p] = false;
		}
		size_t nk = 1;
		size_t k  = 0;
		for(i = 0; i < nr_left; i++)
		{	for(j = 0; j < nc_right; j++)
			{	size_t i_result = result(
					i, j, k, nk, nr_left, n_middle, nc_right
				);
				for(size_t ell = 0; ell < n_middle; ell++)
				{	size_t i_left  = left(
						i, ell, k, nk, nr_left, n_middle, nc_right
					);
					size_t i_right = right(
						ell, j, k, nk, nr_left, n_middle, nc_right
					);
					//
					// Compute sparsity for T(x) = S(x) * f'(x).
					// We so not need to use vx with f'(x) propagation.
					t[i_left]  |= bool( s[i_result] );
					t[i_right] |= bool( s[i_result] );

					// V(x) = f'(x)^T * U(x) +  S(x) * f''(x) * R
					// U(x) = g''(y) * f'(x) * R
					// S(x) = g'(y)

					// back propagate f'(x)^T * U(x)
					// (no need to use vx with f'(x) propogation)
					for(p = 0; p < q; p++)
					{	v[ i_left  * q + p] |= bool( u[ i_result * q + p] );
						v[ i_right * q + p] |= bool( u[ i_result * q + p] );
					}

					// back propagate S(x) * f''(x) * R
					// (here is where we must check for cross terms)
					if( s[i_result] & vx[i_left] & vx[i_right] )
					{	for(p = 0; p < q; p++)
						{	v[i_left * q + p]  |= bool( r[i_right * q + p] );
							v[i_right * q + p] |= bool( r[i_left * q + p] );
						}
					}
				}
			}
		}
		return true;
	}
/* %$$
$head End Class Definition$$
$srccode%cpp% */
}; // End of mat_mul class
}  // End empty namespace
/* %$$
$comment end nospell$$
$end
*/


# endif
