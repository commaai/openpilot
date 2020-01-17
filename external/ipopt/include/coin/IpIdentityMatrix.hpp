// Copyright (C) 2004, 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpIdentityMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPIDENTITYMATRIX_HPP__
#define __IPIDENTITYMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpSymMatrix.hpp"

namespace Ipopt
{

  /** Class for Matrices which are multiples of the identity matrix.
   *
   */
  class IdentityMatrix : public SymMatrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, initializing with dimensions of the matrix
     *  (true identity matrix).
     */
    IdentityMatrix(const SymMatrixSpace* owner_space);

    /** Destructor */
    ~IdentityMatrix();
    //@}

    /** Method for setting the factor for the identity matrix. */
    void SetFactor(Number factor)
    {
      factor_ = factor;
    }

    /** Method for getting the factor for the identity matrix. */
    Number GetFactor() const
    {
      return factor_;
    }

    /** Method for obtaining the dimention of the matrix. */
    Index Dim() const;

  protected:
    /**@name Methods overloaded from matrix */
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector& x,
                                Number beta, Vector& y) const;

    virtual void AddMSinvZImpl(Number alpha, const Vector& S,
                               const Vector& Z, Vector& X) const;

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
    IdentityMatrix();

    /** Copy Constructor */
    IdentityMatrix(const IdentityMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const IdentityMatrix&);
    //@}

    /** Scaling factor for this identity matrix */
    Number factor_;
  };

  /** This is the matrix space for IdentityMatrix. */
  class IdentityMatrixSpace : public SymMatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the dimension of the matrix. */
    IdentityMatrixSpace(Index dim)
        :
        SymMatrixSpace(dim)
    {}

    /** Destructor */
    virtual ~IdentityMatrixSpace()
    {}
    //@}

    /** Overloaded MakeNew method for the SymMatrixSpace base class.
     */
    virtual SymMatrix* MakeNewSymMatrix() const
    {
      return MakeNewIdentityMatrix();
    }

    /** Method for creating a new matrix of this specific type. */
    IdentityMatrix* MakeNewIdentityMatrix() const
    {
      return new IdentityMatrix(this);
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
    IdentityMatrixSpace();

    /** Copy Constructor */
    IdentityMatrixSpace(const IdentityMatrixSpace&);

    /** Overloaded Equals Operator */
    void operator=(const IdentityMatrixSpace&);
    //@}
  };

} // namespace Ipopt
#endif
