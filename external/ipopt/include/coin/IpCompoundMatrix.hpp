// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpCompoundMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPCOMPOUNDMATRIX_HPP__
#define __IPCOMPOUNDMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpMatrix.hpp"

namespace Ipopt
{

  /* forward declarations */
  class CompoundMatrixSpace;

  /** Class for Matrices consisting of other matrices.  This matrix is
   *  a matrix that consists of zero, one or more Matrices's which are
   *  arranged like this: \f$ M_{\rm compound} =
   *  \left(\begin{array}{cccc}M_{00} & M_{01} & \ldots & M_{0,{\rm
   *  ncomp\_cols}-1} \\ \dots &&&\dots \\ M_{{\rm ncomp\_rows}-1,0} &
   *  M_{{\rm ncomp\_rows}-1,1} & \dots & M_{{\rm ncomp\_rows}-1,{\rm
   *  ncomp\_cols}-1}\end{array}\right)\f$.  The individual components
   *  can be associated to different MatrixSpaces.  The individual
   *  components can also be const and non-const Matrices.  If a
   *  component is not set (i.e., it's pointer is NULL), then this
   *  components is treated like a zero-matrix of appropriate
   *  dimensions.
   */
  class CompoundMatrix : public Matrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, taking the owner_space.  The owner_space has to
     *  be defined, so that at each block row and column contain at
     *  least one non-NULL component.  The individual components can
     *  be set afterwards with the SeteComp and SetCompNonConst
     *  methods.
     */
    CompoundMatrix(const CompoundMatrixSpace* owner_space);

    /** Destructor */
    virtual ~CompoundMatrix();
    //@}

    /** Method for setting an individual component at position (irow,
     *  icol) in the compound matrix.  The counting of indices starts
     *  at 0. */
    void SetComp(Index irow, Index jcol, const Matrix& matrix);

    /** Method to set a non-const Matrix entry */
    void SetCompNonConst(Index irow, Index jcol, Matrix& matrix);

    /** Method to create a new matrix from the space for this block */
    void CreateBlockFromSpace(Index irow, Index jcol);

    /** Method for retrieving one block from the compound matrix as a
     *  const Matrix.
     */
    SmartPtr<const Matrix> GetComp(Index irow, Index jcol) const
    {
      return ConstComp(irow, jcol);
    }

    /** Method for retrieving one block from the compound matrix as a
     *  non-const Matrix.  Note that calling this method with mark the
     *  CompoundMatrix as changed.  Therefore, only use this method if
     *  you are intending to change the Matrix that you receive. */
    SmartPtr<Matrix> GetCompNonConst(Index irow, Index jcol)
    {
      ObjectChanged();
      return Comp(irow, jcol);
    }

    /** Number of block rows of this compound matrix. */
    inline Index NComps_Rows() const;
    /** Number of block colmuns of this compound matrix. */
    inline Index NComps_Cols() const;

  protected:
    /**@name Methods overloaded from Matrix */
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector& x,
                                Number beta, Vector& y) const;

    virtual void TransMultVectorImpl(Number alpha, const Vector& x,
                                     Number beta, Vector& y) const;

    /** X = beta*X + alpha*(Matrix S^{-1} Z).  Specialized implementation.
     */
    virtual void AddMSinvZImpl(Number alpha, const Vector& S, const Vector& Z,
                               Vector& X) const;

    /** X = S^{-1} (r + alpha*Z*M^Td).  Specialized implementation.
     */
    virtual void SinvBlrmZMTdBrImpl(Number alpha, const Vector& S,
                                    const Vector& R, const Vector& Z,
                                    const Vector& D, Vector& X) const;

    /** Method for determining if all stored numbers are valid (i.e.,
     *  no Inf or Nan). */
    virtual bool HasValidNumbersImpl() const;

    virtual void ComputeRowAMaxImpl(Vector& rows_norms, bool init) const;

    virtual void ComputeColAMaxImpl(Vector& cols_norms, bool init) const;

    virtual void PrintImpl(const Journalist& jnlst,
                           EJournalLevel level,
                           EJournalCategory category,
                           const std::string& name,
                           Index indent,
                           const std::string& prefix) const;
    //@}

  private:
    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Default Constructor */
    CompoundMatrix();

    /** Copy Constructor */
    CompoundMatrix(const CompoundMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const CompoundMatrix&);
    //@}

    /** Matrix of matrix's containing the components */
    std::vector<std::vector<SmartPtr<Matrix> > > comps_;

    /** Matrix of const matrix's containing the components */
    std::vector<std::vector<SmartPtr<const Matrix> > > const_comps_;

    /** Copy of the owner_space ptr as a CompoundMatrixSpace instead
     *  of MatrixSpace */
    const CompoundMatrixSpace* owner_space_;

    /** boolean indicating if the compound matrix is in a "valid" state */
    mutable bool matrices_valid_;

    /** Method to check whether or not the matrices are valid */
    bool MatricesValid() const;

    inline const Matrix* ConstComp(Index irow, Index jcol) const;

    inline Matrix* Comp(Index irow, Index jcol);
  };

  /** This is the matrix space for CompoundMatrix.  Before a CompoundMatrix
   *  can be created, at least one MatrixSpace has to be set per block
   *  row and column.  Individual component MatrixSpace's can be set
   *  with the SetComp method.
   */
  class CompoundMatrixSpace : public MatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the number of row and columns blocks, as
     *  well as the totel number of rows and columns.
     */
    CompoundMatrixSpace(Index ncomps_rows,
                        Index ncomps_cols,
                        Index total_nRows,
                        Index total_nCols);

    /** Destructor */
    ~CompoundMatrixSpace()
    {}
    //@}

    /** @name Methods for setting information about the components. */
    //@{
    /** Set the number nrows of rows in row-block number irow. */
    void SetBlockRows(Index irow, Index nrows);

    /** Set the number ncols of columns in column-block number jcol. */
    void SetBlockCols(Index jcol, Index ncols);

    /** Get the number nrows of rows in row-block number irow. */
    Index GetBlockRows(Index irow) const;

    /** Set the number ncols of columns in column-block number jcol. */
    Index GetBlockCols(Index jcol) const;

    /** Set the component MatrixSpace.  If auto_allocate is true, then
     *  a new CompoundMatrix created later with MakeNew will have this
     *  component automatically created with the Matrix's MakeNew.
     *  Otherwise, the corresponding component will be NULL and has to
     *  be set with the SetComp methods of the CompoundMatrix.
     */
    void SetCompSpace(Index irow, Index jcol,
                      const MatrixSpace& mat_space,
                      bool auto_allocate = false);
    //@}

    /** Obtain the component MatrixSpace in block row irow and block
     *  column jcol.
     */
    SmartPtr<const MatrixSpace> GetCompSpace(Index irow, Index jcol) const
    {
      DBG_ASSERT(irow<NComps_Rows());
      DBG_ASSERT(jcol<NComps_Cols());
      return comp_spaces_[irow][jcol];
    }

    /** @name Accessor methods */
    //@{
    /** Number of block rows */
    Index NComps_Rows() const
    {
      return ncomps_rows_;
    }
    /** Number of block columns */
    Index NComps_Cols() const
    {
      return ncomps_cols_;
    }

    /** True if the blocks lie on the diagonal - can make some operations faster */
    bool Diagonal() const
    {
      return diagonal_;
    }
    //@}

    /** Method for creating a new matrix of this specific type. */
    CompoundMatrix* MakeNewCompoundMatrix() const;

    /** Overloaded MakeNew method for the MatrixSpace base class.
     */
    virtual Matrix* MakeNew() const
    {
      return MakeNewCompoundMatrix();
    }

  private:
    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Default constructor */
    CompoundMatrixSpace();

    /** Copy Constructor */
    CompoundMatrixSpace(const CompoundMatrixSpace&);

    /** Overloaded Equals Operator */
    CompoundMatrixSpace& operator=(const CompoundMatrixSpace&);
    //@}

    /** Number of block rows */
    Index ncomps_rows_;

    /** Number of block columns */
    Index ncomps_cols_;

    /** Store whether or not the dimensions are valid */
    mutable bool dimensions_set_;

    /** 2-dim std::vector of matrix spaces for the components */
    std::vector<std::vector<SmartPtr<const MatrixSpace> > > comp_spaces_;

    /** 2-dim std::vector of booleans deciding whether to
     *  allocate a new matrix for the blocks automagically */
    std::vector<std::vector< bool > > allocate_block_;

    /** Vector of the number of rows in each comp column */
    std::vector<Index> block_rows_;

    /** Vector of the number of cols in each comp row */
    std::vector<Index> block_cols_;

    /** true if the CompoundMatrixSpace only has Matrix spaces along the diagonal.
     *  this means that the CompoundMatrix will only have matrices along the 
     *  diagonal and it could make some operations more efficient
     */
    bool diagonal_;

    /** Auxilliary function for debugging to set if all block
     *  dimensions have been set. */
    bool DimensionsSet() const;
  };

  /* inline methods */
  inline
  Index CompoundMatrix::NComps_Rows() const
  {
    return owner_space_->NComps_Rows();
  }

  inline
  Index CompoundMatrix::NComps_Cols() const
  {
    return owner_space_->NComps_Cols();
  }

  inline
  const Matrix* CompoundMatrix::ConstComp(Index irow, Index jcol) const
  {
    DBG_ASSERT(irow < NComps_Rows());
    DBG_ASSERT(jcol < NComps_Cols());
    if (IsValid(comps_[irow][jcol])) {
      return GetRawPtr(comps_[irow][jcol]);
    }
    else if (IsValid(const_comps_[irow][jcol])) {
      return GetRawPtr(const_comps_[irow][jcol]);
    }

    return NULL;
  }

  inline
  Matrix* CompoundMatrix::Comp(Index irow, Index jcol)
  {
    DBG_ASSERT(irow < NComps_Rows());
    DBG_ASSERT(jcol < NComps_Cols());
    return GetRawPtr(comps_[irow][jcol]);
  }

} // namespace Ipopt
#endif
