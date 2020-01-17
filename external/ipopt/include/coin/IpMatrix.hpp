// Copyright (C) 2004, 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpMatrix.hpp 2472 2014-04-05 17:47:20Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPMATRIX_HPP__
#define __IPMATRIX_HPP__

#include "IpVector.hpp"

namespace Ipopt
{

  /* forward declarations */
  class MatrixSpace;

  /** Matrix Base Class. This is the base class for all derived matrix
   *  types.  All Matrices, such as Jacobian and Hessian matrices, as
   *  well as possibly the iteration matrices needed for the step
   *  computation, are of this type.
   *
   *  Deriving from Matrix:  Overload the protected XXX_Impl method.
   */
  class Matrix : public TaggedObject
  {
  public:
    /** @name Constructor/Destructor */
    //@{
    /** Constructor.  It has to be given a pointer to the
     *  corresponding MatrixSpace.
     */
    Matrix(const MatrixSpace* owner_space)
        :
        TaggedObject(),
        owner_space_(owner_space),
        valid_cache_tag_(0)
    {}

    /** Destructor */
    virtual ~Matrix()
    {}
    //@}

    /**@name Operations of the Matrix on a Vector */
    //@{
    /** Matrix-vector multiply.  Computes y = alpha * Matrix * x +
     *  beta * y.  Do not overload.  Overload MultVectorImpl instead.
     */
    void MultVector(Number alpha, const Vector& x, Number beta,
                    Vector& y) const
    {
      MultVectorImpl(alpha, x, beta, y);
    }

    /** Matrix(transpose) vector multiply.  Computes y = alpha *
     *  Matrix^T * x + beta * y.  Do not overload.  Overload
     *  TransMultVectorImpl instead.
     */
    void TransMultVector(Number alpha, const Vector& x, Number beta,
                         Vector& y) const
    {
      TransMultVectorImpl(alpha, x, beta, y);
    }
    //@}

    /** @name Methods for specialized operations.  A prototype
     *  implementation is provided, but for efficient implementation
     *  those should be specially implemented.
     */
    //@{
    /** X = X + alpha*(Matrix S^{-1} Z).  Should be implemented
     *  efficiently for the ExansionMatrix
     */
    void AddMSinvZ(Number alpha, const Vector& S, const Vector& Z,
                   Vector& X) const;

    /** X = S^{-1} (r + alpha*Z*M^Td).   Should be implemented
     *  efficiently for the ExansionMatrix
     */
    void SinvBlrmZMTdBr(Number alpha, const Vector& S,
                        const Vector& R, const Vector& Z,
                        const Vector& D, Vector& X) const;
    //@}

    /** Method for determining if all stored numbers are valid (i.e.,
     *  no Inf or Nan). */
    bool HasValidNumbers() const;

    /** @name Information about the size of the matrix */
    //@{
    /** Number of rows */
    inline
    Index  NRows() const;

    /** Number of columns */
    inline
    Index  NCols() const;
    //@}

    /** @name Norms of the individual rows and columns */
    //@{
    /** Compute the max-norm of the rows in the matrix.  The result is
     *  stored in rows_norms.  The vector is assumed to be initialized
     *  of init is false. */
    void ComputeRowAMax(Vector& rows_norms, bool init=true) const
    {
      DBG_ASSERT(NRows() == rows_norms.Dim());
      if (init) rows_norms.Set(0.);
      ComputeRowAMaxImpl(rows_norms, init);
    }
    /** Compute the max-norm of the columns in the matrix.  The result
     *  is stored in cols_norms  The vector is assumed to be initialized
     *  of init is false. */
    void ComputeColAMax(Vector& cols_norms, bool init=true) const
    {
      DBG_ASSERT(NCols() == cols_norms.Dim());
      if (init) cols_norms.Set(0.);
      ComputeColAMaxImpl(cols_norms, init);
    }
    //@}

    /** Print detailed information about the matrix. Do not overload.
     *  Overload PrintImpl instead.
     */
    //@{
    virtual void Print(SmartPtr<const Journalist> jnlst,
                       EJournalLevel level,
                       EJournalCategory category,
                       const std::string& name,
                       Index indent=0,
                       const std::string& prefix="") const;
    virtual void Print(const Journalist& jnlst,
                       EJournalLevel level,
                       EJournalCategory category,
                       const std::string& name,
                       Index indent=0,
                       const std::string& prefix="") const;
    //@}

    /** Return the owner MatrixSpace*/
    inline
    SmartPtr<const MatrixSpace> OwnerSpace() const;

  protected:
    /** @name implementation methods (derived classes MUST
     *  overload these pure virtual protected methods.
     */
    //@{
    /** Matrix-vector multiply.  Computes y = alpha * Matrix * x +
     *  beta * y
     */
    virtual void MultVectorImpl(Number alpha, const Vector& x, Number beta, Vector& y) const =0;

    /** Matrix(transpose) vector multiply.
     * Computes y = alpha * Matrix^T * x  +  beta * y
     */
    virtual void TransMultVectorImpl(Number alpha, const Vector& x, Number beta, Vector& y) const =0;

    /** X = X + alpha*(Matrix S^{-1} Z).  Prototype for this
     *  specialize method is provided, but for efficient
     *  implementation it should be overloaded for the expansion matrix.
     */
    virtual void AddMSinvZImpl(Number alpha, const Vector& S, const Vector& Z,
                               Vector& X) const;

    /** X = S^{-1} (r + alpha*Z*M^Td).   Should be implemented
     *  efficiently for the ExpansionMatrix.
     */
    virtual void SinvBlrmZMTdBrImpl(Number alpha, const Vector& S,
                                    const Vector& R, const Vector& Z,
                                    const Vector& D, Vector& X) const;

    /** Method for determining if all stored numbers are valid (i.e.,
     *  no Inf or Nan). A default implementation always returning true
     *  is provided, but if possible it should be implemented. */
    virtual bool HasValidNumbersImpl() const
    {
      return true;
    }

    /** Compute the max-norm of the rows in the matrix.  The result is
     *  stored in rows_norms.  The vector is assumed to be
     *  initialized. */
    virtual void ComputeRowAMaxImpl(Vector& rows_norms, bool init) const = 0;
    /** Compute the max-norm of the columns in the matrix.  The result
     *  is stored in cols_norms.  The vector is assumed to be
     *  initialized. */
    virtual void ComputeColAMaxImpl(Vector& cols_norms, bool init) const = 0;

    /** Print detailed information about the matrix. */
    virtual void PrintImpl(const Journalist& jnlst,
                           EJournalLevel level,
                           EJournalCategory category,
                           const std::string& name,
                           Index indent,
                           const std::string& prefix) const =0;
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
    /** default constructor */
    Matrix();

    /** Copy constructor */
    Matrix(const Matrix&);

    /** Overloaded Equals Operator */
    Matrix& operator=(const Matrix&);
    //@}

    const SmartPtr<const MatrixSpace> owner_space_;

    /**@name CachedResults data members */
    //@{
    mutable TaggedObject::Tag valid_cache_tag_;
    mutable bool cached_valid_;
    //@}
  };


  /** MatrixSpace base class, corresponding to the Matrix base class.
   *  For each Matrix implementation, a corresponding MatrixSpace has
   *  to be implemented.  A MatrixSpace is able to create new Matrices
   *  of a specific type.  The MatrixSpace should also store
   *  information that is common to all Matrices of that type.  For
   *  example, the dimensions of a Matrix is stored in the MatrixSpace
   *  base class.
   */
  class MatrixSpace : public ReferencedObject
  {
  public:
    /** @name Constructors/Destructors */
    //@{
    /** Constructor, given the number rows and columns of all matrices
     *  generated by this MatrixSpace.
     */
    MatrixSpace(Index nRows, Index nCols)
        :
        nRows_(nRows),
        nCols_(nCols)
    {}

    /** Destructor */
    virtual ~MatrixSpace()
    {}
    //@}

    /** Pure virtual method for creating a new Matrix of the
     *  corresponding type.
     */
    virtual Matrix* MakeNew() const=0;

    /** Accessor function for the number of rows. */
    Index NRows() const
    {
      return nRows_;
    }
    /** Accessor function for the number of columns. */
    Index NCols() const
    {
      return nCols_;
    }

    /** Method to test if a given matrix belongs to a particular
     *  matrix space.
     */
    bool IsMatrixFromSpace(const Matrix& matrix) const
    {
      return (matrix.OwnerSpace() == this);
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
    /** default constructor */
    MatrixSpace();

    /** Copy constructor */
    MatrixSpace(const MatrixSpace&);

    /** Overloaded Equals Operator */
    MatrixSpace& operator=(const MatrixSpace&);
    //@}

    /** Number of rows for all matrices of this type. */
    const Index nRows_;
    /** Number of columns for all matrices of this type. */
    const Index nCols_;
  };


  /* Inline Methods */
  inline
  Index  Matrix::NRows() const
  {
    return owner_space_->NRows();
  }

  inline
  Index  Matrix::NCols() const
  {
    return owner_space_->NCols();
  }

  inline
  SmartPtr<const MatrixSpace> Matrix::OwnerSpace() const
  {
    return owner_space_;
  }

} // namespace Ipopt

// Macro definitions for debugging matrices
#if COIN_IPOPT_VERBOSITY == 0
# define DBG_PRINT_MATRIX(__verbose_level, __mat_name, __mat)
#else
# define DBG_PRINT_MATRIX(__verbose_level, __mat_name, __mat) \
   if (dbg_jrnl.Verbosity() >= (__verbose_level)) { \
      if (dbg_jrnl.Jnlst()!=NULL) { \
        (__mat).Print(dbg_jrnl.Jnlst(), \
        J_ERROR, J_DBG, \
        __mat_name, \
        dbg_jrnl.IndentationLevel()*2, \
        "# "); \
      } \
   }
#endif // #if COIN_IPOPT_VERBOSITY == 0

#endif
