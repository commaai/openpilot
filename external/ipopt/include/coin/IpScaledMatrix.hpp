// Copyright (C) 2004, 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpScaledMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPSCALEDMATRIX_HPP__
#define __IPSCALEDMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpMatrix.hpp"

namespace Ipopt
{

  /* forward declarations */
  class ScaledMatrixSpace;

  /** Class for a Matrix in conjunction with its scaling factors for
   *  row and column scaling. Operations on the matrix are performed using
   *  the scaled matrix. You can pull out the pointer to the 
   *  unscaled matrix for unscaled calculations.
   */
  class ScaledMatrix : public Matrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, taking the owner_space.
     */
    ScaledMatrix(const ScaledMatrixSpace* owner_space);

    /** Destructor */
    ~ScaledMatrix();
    //@}

    /** Set the unscaled matrix */
    void SetUnscaledMatrix(const SmartPtr<const Matrix> unscaled_matrix);

    /** Set the unscaled matrix in a non-const version */
    void SetUnscaledMatrixNonConst(const SmartPtr<Matrix>& unscaled_matrix);

    /** Return the unscaled matrix in const form */
    SmartPtr<const Matrix> GetUnscaledMatrix() const;

    /** Return the unscaled matrix in non-const form */
    SmartPtr<Matrix> GetUnscaledMatrixNonConst();

    /** return the vector for the row scaling */
    SmartPtr<const Vector> RowScaling() const;

    /** return the vector for the column scaling */
    SmartPtr<const Vector> ColumnScaling() const;

  protected:
    /**@name Methods overloaded from Matrix */
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector& x,
                                Number beta, Vector& y) const;

    virtual void TransMultVectorImpl(Number alpha, const Vector& x,
                                     Number beta, Vector& y) const;

    /** Method for determining if all stored numbers are valid (i.e.,
     *  no Inf or Nan).  It is assumed that the scaling factors are
     *  valid. */
    virtual bool HasValidNumbersImpl() const;

    virtual void ComputeRowAMaxImpl(Vector& rows_norms, bool init) const;

    virtual void ComputeColAMaxImpl(Vector& cols_norms, bool init) const;

    virtual void PrintImpl(const Journalist& jnlst,
                           EJournalLevel level,
                           EJournalCategory category,
                           const std::string& name,
                           Index indent,
                           const std::string& prefix) const;

    /** X = beta*X + alpha*(Matrix S^{-1} Z).  Specialized
     *  implementation missing so far!
     */
    virtual void AddMSinvZImpl(Number alpha, const Vector& S, const Vector& Z,
                               Vector& X) const;

    /** X = S^{-1} (r + alpha*Z*M^Td).  Specialized implementation
     *  missing so far!
     */
    virtual void SinvBlrmZMTdBrImpl(Number alpha, const Vector& S,
                                    const Vector& R, const Vector& Z,
                                    const Vector& D, Vector& X) const;
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
    ScaledMatrix();

    /** Copy Constructor */
    ScaledMatrix(const ScaledMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const ScaledMatrix&);
    //@}

    /** const version of the unscaled matrix */
    SmartPtr<const Matrix> matrix_;
    /** non-const version of the unscaled matrix */
    SmartPtr<Matrix> nonconst_matrix_;

    /** Matrix space stored as a ScaledMatrixSpace */
    SmartPtr<const ScaledMatrixSpace> owner_space_;
  };

  /** This is the matrix space for ScaledMatrix.
   */
  class ScaledMatrixSpace : public MatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the number of row and columns blocks, as
     *  well as the totel number of rows and columns.
     */
    ScaledMatrixSpace(const SmartPtr<const Vector>& row_scaling,
                      bool row_scaling_reciprocal,
                      const SmartPtr<const MatrixSpace>& unscaled_matrix_space,
                      const SmartPtr<const Vector>& column_scaling,
                      bool column_scaling_reciprocal);

    /** Destructor */
    ~ScaledMatrixSpace()
    {}
    //@}

    /** Method for creating a new matrix of this specific type. */
    ScaledMatrix* MakeNewScaledMatrix(bool allocate_unscaled_matrix = false) const
    {
      ScaledMatrix* ret = new ScaledMatrix(this);
      if (allocate_unscaled_matrix) {
        SmartPtr<Matrix> unscaled_matrix = unscaled_matrix_space_->MakeNew();
        ret->SetUnscaledMatrixNonConst(unscaled_matrix);
      }
      return ret;
    }

    /** Overloaded MakeNew method for the MatrixSpace base class.
     */
    virtual Matrix* MakeNew() const
    {
      return MakeNewScaledMatrix();
    }

    /** return the vector for the row scaling */
    SmartPtr<const Vector> RowScaling() const
    {
      return ConstPtr(row_scaling_);
    }

    /** return the matrix space for the unscaled matrix */
    SmartPtr<const MatrixSpace> UnscaledMatrixSpace() const
    {
      return unscaled_matrix_space_;
    }

    /** return the vector for the column scaling */
    SmartPtr<const Vector> ColumnScaling() const
    {
      return ConstPtr(column_scaling_);
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
    ScaledMatrixSpace();

    /** Copy Constructor */
    ScaledMatrixSpace(const ScaledMatrixSpace&);

    /** Overloaded Equals Operator */
    ScaledMatrixSpace& operator=(const ScaledMatrixSpace&);
    //@}

    /** Row scaling vector */
    SmartPtr<Vector> row_scaling_;
    /** unscaled matrix space */
    SmartPtr<const MatrixSpace> unscaled_matrix_space_;
    /** column scaling vector */
    SmartPtr<Vector> column_scaling_;
  };

  inline
  void ScaledMatrix::SetUnscaledMatrix(const SmartPtr<const Matrix> unscaled_matrix)
  {
    matrix_ = unscaled_matrix;
    nonconst_matrix_ = NULL;
    ObjectChanged();
  }

  inline
  void ScaledMatrix::SetUnscaledMatrixNonConst(const SmartPtr<Matrix>& unscaled_matrix)
  {
    nonconst_matrix_ = unscaled_matrix;
    matrix_ = GetRawPtr(unscaled_matrix);
    ObjectChanged();
  }

  inline
  SmartPtr<const Matrix> ScaledMatrix::GetUnscaledMatrix() const
  {
    return matrix_;
  }

  inline
  SmartPtr<Matrix> ScaledMatrix::GetUnscaledMatrixNonConst()
  {
    DBG_ASSERT(IsValid(nonconst_matrix_));
    ObjectChanged();
    return nonconst_matrix_;
  }

  inline
  SmartPtr<const Vector> ScaledMatrix::RowScaling() const
  {
    return ConstPtr(owner_space_->RowScaling());
  }

  inline
  SmartPtr<const Vector> ScaledMatrix::ColumnScaling() const
  {
    return ConstPtr(owner_space_->ColumnScaling());
  }

} // namespace Ipopt

#endif
