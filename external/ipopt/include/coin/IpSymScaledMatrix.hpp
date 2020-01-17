// Copyright (C) 2004, 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpSymScaledMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPSYMSCALEDMATRIX_HPP__
#define __IPSYMSCALEDMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpSymMatrix.hpp"

namespace Ipopt
{

  /* forward declarations */
  class SymScaledMatrixSpace;

  /** Class for a Matrix in conjunction with its scaling factors for
   *  row and column scaling. Operations on the matrix are performed using
   *  the scaled matrix. You can pull out the pointer to the 
   *  unscaled matrix for unscaled calculations.
   */
  class SymScaledMatrix : public SymMatrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, taking the owner_space.
     */
    SymScaledMatrix(const SymScaledMatrixSpace* owner_space);

    /** Destructor */
    ~SymScaledMatrix();
    //@}

    /** Set the unscaled matrix */
    void SetUnscaledMatrix(const SmartPtr<const SymMatrix> unscaled_matrix);

    /** Set the unscaled matrix in a non-const version */
    void SetUnscaledMatrixNonConst(const SmartPtr<SymMatrix>& unscaled_matrix);

    /** Return the unscaled matrix in const form */
    SmartPtr<const SymMatrix> GetUnscaledMatrix() const;

    /** Return the unscaled matrix in non-const form */
    SmartPtr<SymMatrix> GetUnscaledMatrixNonConst();

    /** return the vector for the row and column scaling */
    SmartPtr<const Vector> RowColScaling() const;

  protected:
    /**@name Methods overloaded from Matrix */
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector& x,
                                Number beta, Vector& y) const;

    /** Method for determining if all stored numbers are valid (i.e.,
     *  no Inf or Nan).  It is assumed here that the scaling factors
     *  are always valid numbers. */
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
    SymScaledMatrix();

    /** Copy Constructor */
    SymScaledMatrix(const SymScaledMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const SymScaledMatrix&);
    //@}

    /** const version of the unscaled matrix */
    SmartPtr<const SymMatrix> matrix_;
    /** non-const version of the unscaled matrix */
    SmartPtr<SymMatrix> nonconst_matrix_;

    /** Matrix space stored as a SymScaledMatrixSpace */
    SmartPtr<const SymScaledMatrixSpace> owner_space_;
  };

  /** This is the matrix space for SymScaledMatrix.
   */
  class SymScaledMatrixSpace : public SymMatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the number of row and columns blocks, as
     *  well as the totel number of rows and columns.
     */
    SymScaledMatrixSpace(const SmartPtr<const Vector>& row_col_scaling,
                         bool row_col_scaling_reciprocal,
                         const SmartPtr<const SymMatrixSpace>& unscaled_matrix_space)
        :
        SymMatrixSpace(unscaled_matrix_space->Dim()),
        unscaled_matrix_space_(unscaled_matrix_space)
    {
      scaling_ = row_col_scaling->MakeNewCopy();
      if (row_col_scaling_reciprocal) {
        scaling_->ElementWiseReciprocal();
      }
    }

    /** Destructor */
    ~SymScaledMatrixSpace()
    {}
    //@}

    /** Method for creating a new matrix of this specific type. */
    SymScaledMatrix* MakeNewSymScaledMatrix(bool allocate_unscaled_matrix = false) const
    {
      SymScaledMatrix* ret = new SymScaledMatrix(this);
      if (allocate_unscaled_matrix) {
        SmartPtr<SymMatrix> unscaled_matrix = unscaled_matrix_space_->MakeNewSymMatrix();
        ret->SetUnscaledMatrixNonConst(unscaled_matrix);
      }
      return ret;
    }

    /** Overloaded method from SymMatrixSpace */
    virtual SymMatrix* MakeNewSymMatrix() const
    {
      return MakeNewSymScaledMatrix();
    }
    /** Overloaded MakeNew method for the MatrixSpace base class.
     */
    virtual Matrix* MakeNew() const
    {
      return MakeNewSymScaledMatrix();
    }

    /** return the vector for the row and column scaling */
    SmartPtr<const Vector> RowColScaling() const
    {
      return ConstPtr(scaling_);
    }

    /** return the matrix space for the unscaled matrix */
    SmartPtr<const SymMatrixSpace> UnscaledMatrixSpace() const
    {
      return unscaled_matrix_space_;
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
    SymScaledMatrixSpace();

    /** Copy Constructor */
    SymScaledMatrixSpace(const SymScaledMatrixSpace&);

    /** Overloaded Equals Operator */
    SymScaledMatrixSpace& operator=(const SymScaledMatrixSpace&);
    //@}

    /** Row scaling vector */
    SmartPtr<Vector> scaling_;
    /** unscaled matrix space */
    SmartPtr<const SymMatrixSpace> unscaled_matrix_space_;
  };

  inline
  void SymScaledMatrix::SetUnscaledMatrix(const SmartPtr<const SymMatrix> unscaled_matrix)
  {
    matrix_ = unscaled_matrix;
    nonconst_matrix_ = NULL;
    ObjectChanged();
  }

  inline
  void SymScaledMatrix::SetUnscaledMatrixNonConst(const SmartPtr<SymMatrix>& unscaled_matrix)
  {
    nonconst_matrix_ = unscaled_matrix;
    matrix_ = GetRawPtr(unscaled_matrix);
    ObjectChanged();
  }

  inline
  SmartPtr<const SymMatrix> SymScaledMatrix::GetUnscaledMatrix() const
  {
    return matrix_;
  }

  inline
  SmartPtr<SymMatrix> SymScaledMatrix::GetUnscaledMatrixNonConst()
  {
    DBG_ASSERT(IsValid(nonconst_matrix_));
    ObjectChanged();
    return nonconst_matrix_;
  }

  inline SmartPtr<const Vector> SymScaledMatrix::RowColScaling() const
  {
    return ConstPtr(owner_space_->RowColScaling());
  }

} // namespace Ipopt

#endif
