// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpSymTMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPSYMTMATRIX_HPP__
#define __IPSYMTMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpSymMatrix.hpp"

namespace Ipopt
{

  /* forward declarations */
  class SymTMatrixSpace;

  /** Class for symmetric matrices stored in triplet format.  In the
   *  triplet format, the nonzeros elements of a symmetric matrix is
   *  stored in three arrays, Irn, Jcn, and Values, all of length
   *  Nonzeros.  The first two arrays indicate the location of a
   *  non-zero element (as the row and column indices), and the last
   *  array stores the value at that location.  Off-diagonal elements
   *  need to be stored only once since the matrix is symmetric.  For
   *  example, the element \f$a_{1,2}=a_{2,1}\f$ would be stored only
   *  once, either with Irn[i]=1 and Jcn[i]=2, or with Irn[i]=2 and
   *  Jcn[i]=1.  Both representations are identical.  If nonzero
   *  elements (or their symmetric counter part) are listed more than
   *  once, their values are added.
   *
   *  The structure of the nonzeros (i.e. the arrays Irn and Jcn)
   *  cannot be changed after the matrix can been initialized.  Only
   *  the values of the nonzero elements can be modified.
   *
   *  Note that the first row and column of a matrix has index 1, not
   *  0.
   *
   */
  class SymTMatrix : public SymMatrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, taking the corresponding matrix space.
     */
    SymTMatrix(const SymTMatrixSpace* owner_space);

    /** Destructor */
    ~SymTMatrix();
    //@}

    /**@name Changing the Values.*/
    //@{
    /** Set values of nonzero elements.  The values of the nonzero
     *  elements is copied from the incoming Number array.  Important:
     *  It is assume that the order of the values in Values
     *  corresponds to the one of Irn and Jcn given to the matrix
     *  space. */
    void SetValues(const Number* Values);
    //@}

    /** @name Accessor Methods */
    //@{
    /** Number of nonzero entries */
    Index Nonzeros() const;

    /** Obtain pointer to the internal Index array irn_ without the
     *  intention to change the matrix data (USE WITH CARE!).  This
     *  does not produce a copy, and lifetime is not guaranteed!
     */
    const Index* Irows() const;

    /** Obtain pointer to the internal Index array jcn_ without the
     *  intention to change the matrix data (USE WITH CARE!).  This
     *  does not produce a copy, and lifetime is not guaranteed!
     */
    const Index* Jcols() const;

    /** Obtain pointer to the internal Number array values_ with the
     *  intention to change the matrix data (USE WITH CARE!).  This
     *  does not produce a copy, and lifetime is not guaranteed!
     */
    Number* Values();
    /** Obtain pointer to the internal Number array values_ without the
     *  intention to change the matrix data (USE WITH CARE!).  This
     *  does not produce a copy, and lifetime is not guaranteed!
     */
    const Number* Values() const;
    //@}

    /**@name Methods for providing copy of the matrix data */
    //@{
    /** Copy the nonzero structure into provided space */
    void FillStruct(ipfint* Irn, ipfint* Jcn) const;

    /** Copy the value data into provided space */
    void FillValues(Number* Values) const;
    //@}

  protected:
    /**@name Methods overloaded from matrix */
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector& x, Number beta,
                                Vector& y) const;

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
    SymTMatrix();

    /** Copy Constructor */
    SymTMatrix(const SymTMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const SymTMatrix&);
    //@}

    /** Copy of the owner_space ptr as a SymTMatrixSpace insteaqd
     *  of a MatrixSpace
     */
    const SymTMatrixSpace* owner_space_;

    /** Values of nonzeros */
    Number* values_;

    /** Flag for Initialization */
    bool initialized_;

  };

  /** This is the matrix space for a SymTMatrix with fixed sparsity
   *  structure.  The sparsity structure is stored here in the matrix
   *  space.
   */
  class SymTMatrixSpace : public SymMatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the number of rows and columns (both as
     *  dim), as well as the number of nonzeros and the position of
     *  the nonzero elements.  Note that the counting of the nonzeros
     *  starts a 1, i.e., iRows[i]==1 and jCols[i]==1 refers to the
     *  first element in the first row.  This is in accordance with
     *  the HSL data structure.  Off-diagonal elements are stored only
     *  once.
     */
    SymTMatrixSpace(Index dim, Index nonZeros, const Index* iRows,
                    const Index* jCols);

    /** Destructor */
    ~SymTMatrixSpace();
    //@}

    /** Overloaded MakeNew method for the sYMMatrixSpace base class.
     */
    virtual SymMatrix* MakeNewSymMatrix() const
    {
      return MakeNewSymTMatrix();
    }

    /** Method for creating a new matrix of this specific type. */
    SymTMatrix* MakeNewSymTMatrix() const
    {
      return new SymTMatrix(this);
    }

    /**@name Methods describing Matrix structure */
    //@{
    /** Number of non-zeros in the sparse matrix */
    Index Nonzeros() const
    {
      return nonZeros_;
    }

    /** Row index of each non-zero element */
    const Index* Irows() const
    {
      return iRows_;
    }

    /** Column index of each non-zero element */
    const Index* Jcols() const
    {
      return jCols_;
    }
    //@}

  private:
    /**@name Methods called by SymTMatrix for memory management */
    //@{
    /** Allocate internal storage for the SymTMatrix values */
    Number* AllocateInternalStorage() const;

    /** Deallocate internal storage for the SymTMatrix values */
    void FreeInternalStorage(Number* values) const;
    //@}

    const Index nonZeros_;
    Index* iRows_;
    Index* jCols_;

    friend class SymTMatrix;
  };

  /* Inline Methods */
  inline
  Index SymTMatrix::Nonzeros() const
  {
    return owner_space_->Nonzeros();
  }

  inline
  const Index* SymTMatrix::Irows() const
  {
    return owner_space_->Irows();
  }

  inline
  const Index* SymTMatrix::Jcols() const
  {
    return owner_space_->Jcols();
  }


} // namespace Ipopt
#endif
