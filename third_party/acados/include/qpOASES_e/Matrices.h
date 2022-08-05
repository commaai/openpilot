/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2015 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file include/qpOASES_e/Matrices.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2009-2015
 *
 *  Various matrix classes: Abstract base matrix class, dense and sparse matrices,
 *  including symmetry exploiting specializations.
 */



#ifndef QPOASES_MATRICES_H
#define QPOASES_MATRICES_H

#ifdef __USE_SINGLE_PRECISION__

	// single precision
	#define GEMM sgemm_
	#define GEMV sgemv_
//	#define SYR ssyr_
//	#define SYR2 ssyr2_
	#define POTRF spotrf_

#else

	// double precision
	#define GEMM dgemm_
	#define GEMV dgemv_
//	#define SYR  dsyr_
//	#define SYR2 dsyr2_
	#define POTRF dpotrf_

#endif /* __USE_SINGLE_PRECISION__ */


#ifdef EXTERNAL_BLAS
	// double precision
	void dgemm_(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int ldb, double *beta, double *C, int *ldc);
	void dgemv_(char *ta, int *m, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
	void dpotrf_(char *uplo, int *m, double *A, int *lda, int *info);
	// single precision
	void sgemm_(char *ta, char *tb, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B, int ldb, float *beta, float *C, int *ldc);
	void sgemv_(char *ta, int *m, int *n, float *alpha, float *A, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
	void spotrf_(char *uplo, int *m, float *A, int *lda, int *info);
#else
	/** Performs one of the matrix-matrix operation in double precision. */
	void dgemm_ ( const char*, const char*, const unsigned long*, const unsigned long*, const unsigned long*,
			const double*, const double*, const unsigned long*, const double*, const unsigned long*,
			const double*, double*, const unsigned long* );
	/** Performs one of the matrix-matrix operation in single precision. */
	void sgemm_ ( const char*, const char*, const unsigned long*, const unsigned long*, const unsigned long*,
			const float*, const float*, const unsigned long*, const float*, const unsigned long*,
			const float*, float*, const unsigned long* );

	/** Calculates the Cholesky factorization of a real symmetric positive definite matrix in double precision. */
	void dpotrf_ ( const char *, const unsigned long *, double *, const unsigned long *, long * );
	/** Calculates the Cholesky factorization of a real symmetric positive definite matrix in single precision. */
	void spotrf_ ( const char *, const unsigned long *, float *, const unsigned long *, long * );

#endif

	/** Performs a symmetric rank 1 operation in double precision. */
//	void dsyr_ ( const char *, const unsigned long *, const double *, const double *,
//				 const unsigned long *, double *, const unsigned long *);
	/** Performs a symmetric rank 1 operation in single precision. */
//	void ssyr_ ( const char *, const unsigned long *, const float *, const float *,
//				 const unsigned long *, float *, const unsigned long *);

	/** Performs a symmetric rank 2 operation in double precision. */
//	void dsyr2_ ( const char *, const unsigned long *, const double *, const double *,
//				  const unsigned long *, const double *, const unsigned long *, double *, const unsigned long *);
	/** Performs a symmetric rank 2 operation in single precision. */
//	void ssyr2_ ( const char *, const unsigned long *, const float *, const float *,
//				  const unsigned long *, const float *, const unsigned long *, float *, const unsigned long *);


#include <qpOASES_e/Indexlist.h>


BEGIN_NAMESPACE_QPOASES


/**
 *	\brief Interfaces matrix-vector operations tailored to general dense matrices.
 *
 *	Dense matrix class (row major format).
 *
 *	\author Andreas Potschka, Christian Kirches, Hans Joachim Ferreau
 *	\version 3.1embedded
 *	\date 2011-2015
 */
typedef struct
{
	real_t *val;				/**< Vector of entries. */
	int nRows;					/**< Number of rows. */
	int nCols;					/**< Number of columns. */
	int leaDim;					/**< Leading dimension. */
} DenseMatrix;

int DenseMatrix_calculateMemorySize( int m, int n );

char *DenseMatrix_assignMemory( int m, int n, DenseMatrix **mem, void *raw_memory );

DenseMatrix *DenseMatrix_createMemory( int m, int n );

/** Constructor from vector of values.
 *  Caution: Data pointer must be valid throughout lifetime
 */
void DenseMatrixCON(	DenseMatrix* _THIS,
						int m,			/**< Number of rows. */
						int n,			/**< Number of columns. */
						int lD,			/**< Leading dimension. */
						real_t *v		/**< Values. */
						);

void DenseMatrixCPY(	DenseMatrix* FROM,
						DenseMatrix* TO
						);


/** Frees all internal memory. */
void DenseMatrix_free( DenseMatrix* _THIS );

/** Constructor from vector of values.
 *  Caution: Data pointer must be valid throughout lifetime
 */
returnValue DenseMatrix_init(	DenseMatrix* _THIS,
								int m,			/**< Number of rows. */
								int n,			/**< Number of columns. */
								int lD,			/**< Leading dimension. */
								real_t *v		/**< Values. */
								);


/** Returns i-th diagonal entry.
 *	\return i-th diagonal entry */
real_t DenseMatrix_diag(	DenseMatrix* _THIS,
							int i			/**< Index. */
							);

/** Checks whether matrix is square and diagonal.
 *	\return BT_TRUE  iff matrix is square and diagonal; \n
 *	        BT_FALSE otherwise. */
BooleanType DenseMatrix_isDiag( DenseMatrix* _THIS );

/** Get the N-norm of the matrix
 *  \return N-norm of the matrix
 */
real_t DenseMatrix_getNorm( DenseMatrix* _THIS,
							int type			/**< Norm type, 1: one-norm, 2: Euclidean norm. */
							);

/** Get the N-norm of a row
 *  \return N-norm of row \a rNum
 */
real_t DenseMatrix_getRowNorm(	DenseMatrix* _THIS,
								int rNum,			/**< Row number. */
								int type			/**< Norm type, 1: one-norm, 2: Euclidean norm. */
								);

/** Retrieve indexed entries of matrix row multiplied by alpha.
 *  \return SUCCESSFUL_RETURN */
returnValue DenseMatrix_getRow(	DenseMatrix* _THIS,
								int rNum,						/**< Row number. */
								const Indexlist* const icols,	/**< Index list specifying columns. */
								real_t alpha,					/**< Scalar factor. */
								real_t *row						/**< Output row vector. */
								);

/** Retrieve indexed entries of matrix column multiplied by alpha.
 *  \return SUCCESSFUL_RETURN */
 returnValue DenseMatrix_getCol(	DenseMatrix* _THIS,
									int cNum,						/**< Column number. */
									const Indexlist* const irows,	/**< Index list specifying rows. */
									real_t alpha,					/**< Scalar factor. */
									real_t *col						/**< Output column vector. */
									);

/** Evaluate Y=alpha*A*X + beta*Y.
 *  \return SUCCESSFUL_RETURN. */
returnValue DenseMatrix_times(	DenseMatrix* _THIS,
								int xN,					/**< Number of vectors to multiply. */
								real_t alpha,			/**< Scalar factor for matrix vector product. */
								const real_t *x,		/**< Input vector to be multiplied. */
								int xLD,				/**< Leading dimension of input x. */
								real_t beta,			/**< Scalar factor for y. */
								real_t *y,				/**< Output vector of results. */
								int yLD					/**< Leading dimension of output y. */
								);

/** Evaluate Y=alpha*A'*X + beta*Y.
 *  \return SUCCESSFUL_RETURN. */
returnValue DenseMatrix_transTimes(	DenseMatrix* _THIS,
									int xN,				/**< Number of vectors to multiply. */
									real_t alpha,		/**< Scalar factor for matrix vector product. */
									const real_t *x,	/**< Input vector to be multiplied. */
									int xLD,			/**< Leading dimension of input x. */
									real_t beta,		/**< Scalar factor for y. */
									real_t *y,			/**< Output vector of results. */
									int yLD				/**< Leading dimension of output y. */
									);

/** Evaluate matrix vector product with submatrix given by Indexlist.
 *	\return SUCCESSFUL_RETURN */
 returnValue DenseMatrix_subTimes(	DenseMatrix* _THIS,
									const Indexlist* const irows,	/**< Index list specifying rows. */
									const Indexlist* const icols,	/**< Index list specifying columns. */
									int xN,							/**< Number of vectors to multiply. */
									real_t alpha,					/**< Scalar factor for matrix vector product. */
									const real_t *x,				/**< Input vector to be multiplied. */
									int xLD,						/**< Leading dimension of input x. */
									real_t beta,					/**< Scalar factor for y. */
									real_t *y,						/**< Output vector of results. */
									int yLD,						/**< Leading dimension of output y. */
									BooleanType yCompr 				/**< Compressed storage for y. */
									);

/** Evaluate matrix transpose vector product.
 *	\return SUCCESSFUL_RETURN */
returnValue DenseMatrix_subTransTimes(	DenseMatrix* _THIS,
										const Indexlist* const irows,	/**< Index list specifying rows. */
										const Indexlist* const icols,	/**< Index list specifying columns. */
										int xN,							/**< Number of vectors to multiply. */
										real_t alpha,					/**< Scalar factor for matrix vector product. */
										const real_t *x,				/**< Input vector to be multiplied. */
										int xLD,						/**< Leading dimension of input x. */
										real_t beta,					/**< Scalar factor for y. */
										real_t *y,						/**< Output vector of results. */
										int yLD							/**< Leading dimension of output y. */
										);

/** Adds given offset to diagonal of matrix.
 *	\return SUCCESSFUL_RETURN \n
 			RET_NO_DIAGONAL_AVAILABLE */
returnValue DenseMatrix_addToDiag(	DenseMatrix* _THIS,
									real_t alpha		/**< Diagonal offset. */
									);

/** Prints matrix to screen.
 *	\return SUCCESSFUL_RETURN */
returnValue DenseMatrix_print(	DenseMatrix* _THIS
								);

static inline real_t* DenseMatrix_getVal( DenseMatrix* _THIS ) { return _THIS->val; }

/** Compute bilinear form y = x'*H*x using submatrix given by index list.
 *	\return SUCCESSFUL_RETURN */
returnValue DenseMatrix_bilinear(	DenseMatrix* _THIS,
									const Indexlist* const icols,	/**< Index list specifying columns of x. */
									int xN,							/**< Number of vectors to multiply. */
									const real_t *x,				/**< Input vector to be multiplied (uncompressed). */
									int xLD,						/**< Leading dimension of input x. */
									real_t *y,						/**< Output vector of results (compressed). */
									int yLD							/**< Leading dimension of output y. */
									);



END_NAMESPACE_QPOASES


#endif	/* QPOASES_MATRICES_H */
