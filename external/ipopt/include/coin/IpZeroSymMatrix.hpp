// Copyright (C) 2004, 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpZeroSymMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPZEROSYMMATRIX_HPP__
#define __IPZEROSYMMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpSymMatrix.hpp"

namespace Ipopt
{

  /** Class for Symmetric Matrices with only zero entries.
   */
  class ZeroSymMatrix : public SymMatrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, taking the corresponding matrix space.
     */
    ZeroSymMatrix(const SymMatrixSpace* owner_space);

    /** Destructor */
    ~ZeroSymMatrix();
    //@}

  protected:
    /**@name Methods overloaded from matrix */
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector& x,
                                Number beta, Vector& y) const;

    virtual void TransMultVectorImpl(Number alpha, const Vector& x,
                                     Number beta, Vector& y) const;

    virtual void ComputeRowAMaxImpl(Vector& rows_norms, bool init) const
      {}

    virtual void ComputeColAMaxImpl(Vector& cols_norms, bool init) const
      {}

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
    ZeroSymMatrix();

    /** Copy Constructor */
    ZeroSymMatrix(const ZeroSymMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const ZeroSymMatrix&);
    //@}
  };

  /** Class for matrix space for ZeroSymMatrix. */
  class ZeroSymMatrixSpace : public SymMatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the number of row and columns.
     */
    ZeroSymMatrixSpace(Index dim)
        :
        SymMatrixSpace(dim)
    {}

    /** Destructor */
    virtual ~ZeroSymMatrixSpace()
    {}
    //@}

    /** Overloaded MakeNew method for the MatrixSpace base class.
     */
    virtual Matrix* MakeNew() const
    {
      return MakeNewZeroSymMatrix();
    }

    /** Overloaded method from SymMatrixSpace base class
     */
    virtual SymMatrix* MakeNewSymMatrix() const
    {
      return MakeNewZeroSymMatrix();
    }

    /** Method for creating a new matrix of this specific type. */
    ZeroSymMatrix* MakeNewZeroSymMatrix() const
    {
      return new ZeroSymMatrix(this);
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
    /** Default Constructor */
    ZeroSymMatrixSpace();

    /** Copy Constructor */
    ZeroSymMatrixSpace(const ZeroSymMatrixSpace&);

    /** Overloaded Equals Operator */
    void operator=(const ZeroSymMatrixSpace&);
    //@}
  };
} // namespace Ipopt
#endif
