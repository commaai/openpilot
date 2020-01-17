// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpGenTMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPGENTMATRIX_HPP__
#define __IPGENTMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpMatrix.hpp"

namespace Ipopt
{

  /* forward declarations */
  class GenTMatrixSpace;

  /** Class for general matrices stored in triplet format.  In the
   *  triplet format, the nonzeros elements of a general matrix is
   *  stored in three arrays, Irow, Jcol, and Values, all of length
   *  Nonzeros.  The first two arrays indicate the location of a
   *  non-zero element (row and column indices), and the last array
   *  stores the value at that location.  If nonzero elements are
   *  listed more than once, their values are added.
   *
   *  The structure of the nonzeros (i.e. the arrays Irow and Jcol)
   *  cannot be changed after the matrix can been initialized.  Only
   *  the values of the nonzero elements can be modified.
   *
   *  Note that the first row and column of a matrix has index 1, not
   *  0.
   */
  class GenTMatrix : public Matrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, taking the owner_space.
     */
    GenTMatrix(const GenTMatrixSpace* owner_space);

    /** Destructor */
    ~GenTMatrix();
    //@}

    /**@name Changing the Values.*/
    //@{
    /** Set values of nonzero elements.  The values of the nonzero
     *  elements are copied from the incoming Number array.  Important:
     *  It is assume that the order of the values in Values
     *  corresponds to the one of Irn and Jcn given to one of the
     *  constructors above. */
    void SetValues(const Number* Values);
    //@}

    /** @name Accessor Methods */
    //@{
    /** Number of nonzero entries */
    Index Nonzeros() const;

    /** Array with Row indices (counting starts at 1) */
    const Index* Irows() const;

    /** Array with Column indices (counting starts at 1) */
    const Index* Jcols() const;

    /** Array with nonzero values (const version).  */
    const Number* Values() const
    {
      return values_;
    }

    /** Array with the nonzero values of this matrix (non-const
     *  version).  Use this method only if you are intending to change
     *  the values, because the GenTMatrix will be marked as changed.
     */
    Number* Values()
    {
      ObjectChanged();
      initialized_ = true;
      return values_;
    }
    //@}

  protected:
    /**@name Overloaded methods from Matrix base class*/
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector &x, Number beta,
                                Vector &y) const;

    virtual void TransMultVectorImpl(Number alpha, const Vector& x, Number beta,
                                     Vector& y) const;

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
                           const std::string& prefix) const
    {
      PrintImplOffset(jnlst, level, category, name, indent, prefix, 0);
    }
    //@}

    void PrintImplOffset(const Journalist& jnlst,
                         EJournalLevel level,
                         EJournalCategory category,
                         const std::string& name,
                         Index indent,
                         const std::string& prefix,
                         Index offset) const;

    friend class ParGenMatrix;

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
    GenTMatrix();

    /** Copy Constructor */
    GenTMatrix(const GenTMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const GenTMatrix&);
    //@}

    /** Copy of the owner space as a GenTMatrixSpace instead of
     *  a MatrixSpace
     */
    const GenTMatrixSpace* owner_space_;

    /** Values of nonzeros */
    Number* values_;

    /** Flag for Initialization */
    bool initialized_;

  };

  /** This is the matrix space for a GenTMatrix with fixed sparsity
   *  structure.  The sparsity structure is stored here in the matrix
   *  space.
   */
  class GenTMatrixSpace : public MatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the number of rows and columns, as well as
     *  the number of nonzeros and the position of the nonzero
     *  elements.  Note that the counting of the nonzeros starts a 1,
     *  i.e., iRows[i]==1 and jCols[i]==1 refers to the first element
     *  in the first row.  This is in accordance with the HSL data
     *  structure.
     */
    GenTMatrixSpace(Index nRows, Index nCols,
                    Index nonZeros,
                    const Index* iRows, const Index* jCols);

    /** Destructor */
    ~GenTMatrixSpace()
    {
      delete [] iRows_;
      delete [] jCols_;
    }
    //@}

    /** Method for creating a new matrix of this specific type. */
    GenTMatrix* MakeNewGenTMatrix() const
    {
      return new GenTMatrix(this);
    }

    /** Overloaded MakeNew method for the MatrixSpace base class.
     */
    virtual Matrix* MakeNew() const
    {
      return MakeNewGenTMatrix();
    }

    /**@name Methods describing Matrix structure */
    //@{
    /** Number of non-zeros in the sparse matrix */
    Index Nonzeros() const
    {
      return nonZeros_;
    }

    /** Row index of each non-zero element (counting starts at 1) */
    const Index* Irows() const
    {
      return iRows_;
    }

    /** Column index of each non-zero element (counting starts at 1) */
    const Index* Jcols() const
    {
      return jCols_;
    }
    //@}

  private:
    /** @name Sparsity structure of matrices generated by this matrix
     *  space.
     */
    //@{
    const Index nonZeros_;
    Index* jCols_;
    Index* iRows_;
    //@}

    /** This method is only for the GenTMatrix to call in order
     *   to allocate internal storage */
    Number* AllocateInternalStorage() const;

    /** This method is only for the GenTMatrix to call in order
     *   to de-allocate internal storage */
    void FreeInternalStorage(Number* values) const;

    friend class GenTMatrix;
  };

  /* inline methods */
  inline
  Index GenTMatrix::Nonzeros() const
  {
    return owner_space_->Nonzeros();
  }

  inline
  const Index* GenTMatrix::Irows() const
  {
    return owner_space_->Irows();
  }

  inline
  const Index* GenTMatrix::Jcols() const
  {
    return owner_space_->Jcols();
  }


} // namespace Ipopt
#endif
