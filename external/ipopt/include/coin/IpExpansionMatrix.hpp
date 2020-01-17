// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpExpansionMatrix.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPEXPANSIONMATRIX_HPP__
#define __IPEXPANSIONMATRIX_HPP__

#include "IpUtils.hpp"
#include "IpMatrix.hpp"

namespace Ipopt
{

  /** forward declarations */
  class ExpansionMatrixSpace;

  /** Class for expansion/projection matrices.  These matrices allow
   *  to lift a vector to a vector with larger dimension, keeping
   *  some elements of the larger vector zero.  This operation is achieved
   *  by the MultVector operation.  The transpose operation then
   *  filters some elements from a large vector into a smaller vector.
   */
  class ExpansionMatrix : public Matrix
  {
  public:

    /**@name Constructors / Destructors */
    //@{

    /** Constructor, taking the owner_space.
     */
    ExpansionMatrix(const ExpansionMatrixSpace* owner_space);

    /** Destructor */
    ~ExpansionMatrix();
    //@}

    /** Return the vector of indices marking the expanded position.
     *  The result is the Index array (of length NSmallVec=NCols())
     *  that stores the mapping from the small vector to the large
     *  vector.  For each element i=0,..,NSmallVec in the small
     *  vector, ExpandedPosIndices()[i] give the corresponding index
     *  in the large vector.
     */
    const Index* ExpandedPosIndices() const;

    /** Return the vector of indices marking the compressed position.
     *  The result is the Index array (of length NLargeVec=NRows())
     *  that stores the mapping from the large vector to the small
     *  vector.  For each element i=0,..,NLargeVec in the large
     *  vector, CompressedPosIndices()[i] gives the corresponding
     *  index in the small vector, unless CompressedPosIndices()[i] is
     *  negative.
     */
    const Index* CompressedPosIndices() const;

  protected:
    /**@name Overloaded methods from Matrix base class*/
    //@{
    virtual void MultVectorImpl(Number alpha, const Vector &x, Number beta,
                                Vector &y) const;

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

    virtual void ComputeRowAMaxImpl(Vector& rows_norms, bool init) const;

    virtual void ComputeColAMaxImpl(Vector& cols_norms, bool init) const;

    virtual void PrintImpl(const Journalist& jnlst,
                           EJournalLevel level,
                           EJournalCategory category,
                           const std::string& name,
                           Index indent,
                           const std::string& prefix) const
    {
      PrintImplOffset(jnlst, level, category, name, indent, prefix, 1, 1);
    }
    //@}

    void PrintImplOffset(const Journalist& jnlst,
                         EJournalLevel level,
                         EJournalCategory category,
                         const std::string& name,
                         Index indent,
                         const std::string& prefix,
                         Index row_offset,
                         Index col_offset) const;

    friend class ParExpansionMatrix;

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
    ExpansionMatrix();

    /** Copy Constructor */
    ExpansionMatrix(const ExpansionMatrix&);

    /** Overloaded Equals Operator */
    void operator=(const ExpansionMatrix&);
    //@}

    const ExpansionMatrixSpace* owner_space_;

  };

  /** This is the matrix space for ExpansionMatrix.
   */
  class ExpansionMatrixSpace : public MatrixSpace
  {
  public:
    /** @name Constructors / Destructors */
    //@{
    /** Constructor, given the list of elements of the large vector
     *  (of size NLargeVec) to be filtered into the small vector (of
     *  size NSmallVec).  For each i=0..NSmallVec-1 the i-th element
     *  of the small vector will be put into the ExpPos[i] position of
     *  the large vector.  The position counting in the vector is
     *  assumed to start at 0 (C-like array notation).
     */
    ExpansionMatrixSpace(Index NLargeVec,
                         Index NSmallVec,
                         const Index *ExpPos,
                         const int offset = 0);

    /** Destructor */
    ~ExpansionMatrixSpace()
    {
      delete [] compressed_pos_;
      delete [] expanded_pos_;
    }
    //@}

    /** Method for creating a new matrix of this specific type. */
    ExpansionMatrix* MakeNewExpansionMatrix() const
    {
      return new ExpansionMatrix(this);
    }

    /** Overloaded MakeNew method for the MatrixSpace base class.
     */
    virtual Matrix* MakeNew() const
    {
      return MakeNewExpansionMatrix();
    }

    /** Accessor Method to obtain the Index array (of length
     *  NSmallVec=NCols()) that stores the mapping from the small
     *  vector to the large vector.  For each element i=0,..,NSmallVec
     *  in the small vector, ExpandedPosIndices()[i] give the
     *  corresponding index in the large vector.
     */
    const Index* ExpandedPosIndices() const
    {
      return expanded_pos_;
    }

    /** Accessor Method to obtain the Index array (of length
     *  NLargeVec=NRows()) that stores the mapping from the large
     *  vector to the small vector.  For each element i=0,..,NLargeVec
     *  in the large vector, CompressedPosIndices()[i] gives the
     *  corresponding index in the small vector, unless
     *  CompressedPosIndices()[i] is negative.
     */
    const Index* CompressedPosIndices() const
    {
      return compressed_pos_;
    }

  private:
    Index *expanded_pos_;
    Index *compressed_pos_;
  };

  /* inline methods */
  inline
  const Index* ExpansionMatrix::ExpandedPosIndices() const
  {
    return owner_space_->ExpandedPosIndices();
  }

  inline
  const Index* ExpansionMatrix::CompressedPosIndices() const
  {
    return owner_space_->CompressedPosIndices();
  }

} // namespace Ipopt
#endif
