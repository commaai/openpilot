// Copyright (C) 2004, 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpCompoundSymMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPCOMPOUNDSYMMATRIX_HPP__
#define __IPCOMPOUNDSYMMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpSymMatrix.hpp"

namespace Ipopt
{

  /* forward declarations */
  class CompoundSymMatrixSpace;

  /** Class for symmetric matrices consisting of other matrices.
   *  Here, the lower left block of the matrix is stored.
   */
  class CompoundSymMatrix : public SymMatrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, taking only the number for block components into the
     *  row and column direction.  The owner_space has to be defined, so
     *  that at each block row and column contain at least one non-NULL
     *  component.
     */
    CompoundSymMatrix(const CompoundSymMatrixSpace* owner_space);

    /** Destructor */
    ~CompoundSymMatrix();
    //@}

    /** Method for setting an individual component at position (irow,
     *  icol) in the compound matrix.  The counting of indices starts
     *  at 0. Since this only the lower left components are stored, we need
     *  to have jcol<=irow, and if irow==jcol, the matrix must be a SymMatrix */
    void SetComp(Index irow, Index jcol, const Matrix& matrix);

    /** Non const version of the same method */
    void SetCompNonConst(Index irow, Index jcol, Matrix& matrix);

    /** Method for retrieving one block from the compound matrix.
     *  Since this only the lower left components are stored, we need
     *  to have jcol<=irow */
    SmartPtr<const Matrix> GetComp(Index irow, Index jcol) const
    {
      return ConstComp(irow,jcol);
    }

    /** Non const version of GetComp.  You should only use this method
     *  if you are intending to change the matrix you receive, since
     *  this CompoundSymMatrix will be marked as changed. */
    SmartPtr<Matrix> GetCompNonConst(Index irow, Index jcol)
    {
      ObjectChanged();
      return Comp(irow,jcol);
    }

    /** Method for creating a new matrix of this specific type. */
    SmartPtr<CompoundSymMatrix> MakeNewCompoundSymMatrix() const;

    // The following don't seem to be necessary
    /* Number of block rows of this compound matrix. */
    //    Index NComps_NRows() const { return NComps_Dim(); }

    /* Number of block colmuns of this compound matrix. */
    //    Index NComps_NCols() const { return NComps_Dim(); }

    /** Number of block rows and columns */
    Index NComps_Dim() const;

  protected:
    /**@name Methods overloaded from matrix */
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector& x,
                                Number beta, Vector& y) const;

    /** Method for determining if all stored numbers are valid (i.e.,
     *  no Inf or Nan). */
    virtual bool HasValidNumbersImpl() const;

    virtual void ComputeRowAMaxImpl(Vector& rows_norms, bool init) const;

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
    CompoundSymMatrix();

    /** Copy Constructor */
    CompoundSymMatrix(const CompoundSymMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const CompoundSymMatrix&);
    //@}

    /** Vector of vectors containing the components */
    std::vector<std::vector<SmartPtr<Matrix> > > comps_;

    /** Vector of vectors containing the const components */
    std::vector<std::vector<SmartPtr<const Matrix> > > const_comps_;

    /** Copy of the owner_space ptr as a CompoundSymMatrixSpace */
    const CompoundSymMatrixSpace* owner_space_;

    /** boolean indicating if the compound matrix is in a "valid" state */
    mutable bool matrices_valid_;

    /** method to check wether or not the matrices are valid */
    bool MatricesValid() const;

    /** Internal method to return a const pointer to one of the comps */
    const Matrix* ConstComp(Index irow, Index jcol) const
    {
      DBG_ASSERT(irow < NComps_Dim());
      DBG_ASSERT(jcol <= irow);
      if (IsValid(comps_[irow][jcol])) {
        return GetRawPtr(comps_[irow][jcol]);
      }
      else if (IsValid(const_comps_[irow][jcol])) {
        return GetRawPtr(const_comps_[irow][jcol]);
      }

      return NULL;
    }

    /** Internal method to return a non-const pointer to one of the comps */
    Matrix* Comp(Index irow, Index jcol)
    {
      DBG_ASSERT(irow < NComps_Dim());
      DBG_ASSERT(jcol <= irow);
      // We shouldn't be asking for a non-const if this entry holds a
      // const one...
      DBG_ASSERT(IsNull(const_comps_[irow][jcol]));
      if (IsValid(comps_[irow][jcol])) {
        return GetRawPtr(comps_[irow][jcol]);
      }

      return NULL;
    }
  };

  /** This is the matrix space for CompoundSymMatrix.  Before a
   *  CompoundSymMatrix can be created, at least one SymMatrixSpace has
   *  to be set per block row and column.  Individual component
   *  SymMatrixSpace's can be set with the SetComp method.
   */
  class CompoundSymMatrixSpace : public SymMatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the number of blocks (same for rows and
     *  columns), as well as the total dimension of the matrix.
     */
    CompoundSymMatrixSpace(Index ncomp_spaces, Index total_dim);

    /** Destructor */
    ~CompoundSymMatrixSpace()
    {}
    //@}

    /** @name Methods for setting information about the components. */
    //@{
    /** Set the dimension dim for block row (or column) irow_jcol */
    void SetBlockDim(Index irow_jcol, Index dim);

    /** Get the dimension dim for block row (or column) irow_jcol */
    Index GetBlockDim(Index irow_jcol) const;

    /** Set the component SymMatrixSpace. If auto_allocate is true, then
     *  a new CompoundSymMatrix created later with MakeNew will have this
     *  component automatically created with the SymMatrix's MakeNew.
     *  Otherwise, the corresponding component will be NULL and has to
     *  be set with the SetComp methods of the CompoundSymMatrix.
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
      DBG_ASSERT(irow<ncomp_spaces_);
      DBG_ASSERT(jcol<=irow);
      return comp_spaces_[irow][jcol];
    }

    /** @name Accessor methods */
    //@{
    Index NComps_Dim() const
    {
      return ncomp_spaces_;
    }
    //@}

    /** Method for creating a new matrix of this specific type. */
    CompoundSymMatrix* MakeNewCompoundSymMatrix() const;

    /** Overloaded MakeNew method for the SymMatrixSpace base class.
     */
    virtual SymMatrix* MakeNewSymMatrix() const
    {
      return MakeNewCompoundSymMatrix();
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
    CompoundSymMatrixSpace();

    /** Copy Constructor */
    CompoundSymMatrixSpace(const CompoundSymMatrix&);

    /** Overloaded Equals Operator */
    CompoundSymMatrixSpace& operator=(const CompoundSymMatrixSpace&);
    //@}

    /** Number of components per row and column */
    Index ncomp_spaces_;

    /** Vector of the number of rows in each comp column,
     *  Since this is symmetric, this is also the number
     *  of columns in each row, hence, it is the dimension
     *  each of the diagonals */
    std::vector<Index> block_dim_;

    /** 2-dim std::vector of matrix spaces for the components.  Only
     *  the lower right part is stored. */
    std::vector<std::vector<SmartPtr<const MatrixSpace> > > comp_spaces_;

    /** 2-dim std::vector of booleans deciding whether to
     *  allocate a new matrix for the blocks automagically */
    std::vector<std::vector< bool > > allocate_block_;

    /** boolean indicating if the compound matrix space is in a "valid" state */
    mutable bool dimensions_set_;

    /** Method to check whether or not the spaces are valid */
    bool DimensionsSet() const;
  };

  inline
  SmartPtr<CompoundSymMatrix> CompoundSymMatrix::MakeNewCompoundSymMatrix() const
  {
    return owner_space_->MakeNewCompoundSymMatrix();
  }

} // namespace Ipopt
#endif
