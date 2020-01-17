// Copyright (C) 2004, 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpSymMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPSYMMATRIX_HPP__
#define __IPSYMMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpMatrix.hpp"

namespace Ipopt
{

  /* forward declarations */
  class SymMatrixSpace;

  /** This is the base class for all derived symmetric matrix types.
   */
  class SymMatrix : public Matrix
  {
  public:
    /** @name Constructor/Destructor */
    //@{
    /** Constructor, taking the owner_space.
     */
    inline
    SymMatrix(const SymMatrixSpace* owner_space);

    /** Destructor */
    virtual ~SymMatrix()
    {}
    //@}

    /** @name Information about the size of the matrix */
    //@{
    /** Dimension of the matrix (number of rows and columns) */
    inline
    Index Dim() const;
    //@}

    inline
    SmartPtr<const SymMatrixSpace> OwnerSymMatrixSpace() const;

  protected:
    /** @name Overloaded methods from Matrix. */
    //@{
    /** Since the matrix is
     *  symmetric, it is only necessary to implement the
     *  MultVectorImpl method in a class that inherits from this base
     *  class.  If the TransMultVectorImpl is called, this base class
     *  automatically calls MultVectorImpl instead. */
    virtual void TransMultVectorImpl(Number alpha, const Vector& x, Number beta,
                                     Vector& y) const
    {
      // Since this matrix is symetric, this is the same operation as
      // MultVector
      MultVector(alpha, x, beta, y);
    }
    /** Since the matrix is symmetric, the row and column max norms
     *  are identical */
    virtual void ComputeColAMaxImpl(Vector& cols_norms, bool init) const
    {
      ComputeRowAMaxImpl(cols_norms, init);
    }
    //@}

  private:
    /** Copy of the owner space ptr as a SymMatrixSpace instead
     *  of a MatrixSpace 
     */
    const SymMatrixSpace* owner_space_;
  };


  /** SymMatrixSpace base class, corresponding to the SymMatrix base
   *  class. */
  class SymMatrixSpace : public MatrixSpace
  {
  public:
    /** @name Constructors/Destructors */
    //@{
    /** Constructor, given the dimension (identical to the number of
     *  rows and columns).
     */
    SymMatrixSpace(Index dim)
        :
        MatrixSpace(dim,dim)
    {}

    /** Destructor */
    virtual ~SymMatrixSpace()
    {}
    //@}

    /** Pure virtual method for creating a new matrix of this specific
     *  type. */
    virtual SymMatrix* MakeNewSymMatrix() const=0;

    /** Overloaded MakeNew method for the MatrixSpace base class.
     */
    virtual Matrix* MakeNew() const
    {
      return MakeNewSymMatrix();
    }

    /** Accessor method for the dimension of the matrices in this
     *  matrix space.
     */
    Index Dim() const
    {
      DBG_ASSERT(NRows() == NCols());
      return NRows();
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
    SymMatrixSpace();

    /* Copy constructor */
    SymMatrixSpace(const SymMatrixSpace&);

    /** Overloaded Equals Operator */
    SymMatrixSpace& operator=(const SymMatrixSpace&);
    //@}

  };

  /* inline methods */
  inline
  SymMatrix::SymMatrix(const SymMatrixSpace* owner_space)
      :
      Matrix(owner_space),
      owner_space_(owner_space)
  {}

  inline
  Index SymMatrix::Dim() const
  {
    return owner_space_->Dim();
  }

  inline
  SmartPtr<const SymMatrixSpace> SymMatrix::OwnerSymMatrixSpace() const
  {
    return owner_space_;
  }

} // namespace Ipopt

#endif
