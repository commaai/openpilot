// Copyright (C) 2004, 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpVector.hpp 2472 2014-04-05 17:47:20Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPVECTOR_HPP__
#define __IPVECTOR_HPP__

#include "IpTypes.hpp"
#include "IpTaggedObject.hpp"
#include "IpCachedResults.hpp"
#include "IpSmartPtr.hpp"
#include "IpJournalist.hpp"
#include "IpException.hpp"

#include <vector>

namespace Ipopt
{
  /** Exception that can be used to flag unimplemented linear algebra
   *  methods */
  DECLARE_STD_EXCEPTION(UNIMPLEMENTED_LINALG_METHOD_CALLED);

  /* forward declarations */
  class VectorSpace;

  /** Vector Base Class.
   * This is the base class for all derived vector types.  Those vectors
   * are meant to store entities like iterates, Lagrangian multipliers,
   * constraint values etc.  The implementation of a vector type depends
   * on the computational environment (e.g. just a double array on a shared
   * memory machine, or distributed double arrays for a distributed
   * memory machine.)
   * 
   * Deriving from Vector: This class inherits from tagged object to
   * implement an advanced caching scheme. Because of this, the
   * TaggedObject method ObjectChanged() must be called each time the
   * Vector changes. If you overload the XXXX_Impl protected methods,
   * this taken care of (along with caching if possible) for you. If
   * you have additional methods in your derived class that change the
   * underlying data (vector values), you MUST remember to call
   * ObjectChanged() AFTER making the change!
   */
  class Vector : public TaggedObject
  {
  public:
    /** @name Constructor/Destructor */
    //@{
    /** Constructor.  It has to be given a pointer to the
     *  corresponding VectorSpace.
     */
    inline
    Vector(const VectorSpace* owner_space);

    /** Destructor */
    inline
    virtual ~Vector();
    //@}

    /** Create new Vector of the same type with uninitialized data */
    inline
    Vector* MakeNew() const;

    /** Create new Vector of the same type and copy the data over */
    inline
    Vector* MakeNewCopy() const;

    /**@name Standard BLAS-1 Operations
     *  (derived classes do NOT overload these 
     *  methods, instead, overload the 
     *  protected versions of these methods). */
    //@{
    /** Copy the data of the vector x into this vector (DCOPY). */
    inline
    void Copy(const Vector& x);

    /** Scales the vector by scalar alpha (DSCAL) */
    void Scal(Number alpha);

    /** Add the multiple alpha of vector x to this vector (DAXPY) */
    inline
    void Axpy(Number alpha, const Vector &x);

    /** Computes inner product of vector x with this (DDOT) */
    inline
    Number Dot(const Vector &x) const;

    /** Computes the 2-norm of this vector (DNRM2) */
    inline
    Number Nrm2() const;

    /** Computes the 1-norm of this vector (DASUM) */
    inline
    Number Asum() const;

    /** Computes the max-norm of this vector (based on IDAMAX) */
    inline
    Number Amax() const;
    //@}

    /** @name Additional (Non-BLAS) Vector Methods
     *  (derived classes do NOT overload these 
     *  methods, instead, overload the 
     *  protected versions of these methods). */
    //@{
    /** Set each element in the vector to the scalar alpha. */
    inline
    void Set(Number alpha);

    /** Element-wise division  \f$y_i \gets y_i/x_i\f$*/
    inline
    void ElementWiseDivide(const Vector& x);

    /** Element-wise multiplication \f$y_i \gets y_i*x_i\f$ */
    inline
    void ElementWiseMultiply(const Vector& x);

    /** Element-wise max against entries in x */
    inline
    void ElementWiseMax(const Vector& x);

    /** Element-wise min against entries in x */
    inline
    void ElementWiseMin(const Vector& x);

    /** Reciprocates the entries in the vector */
    inline
    void ElementWiseReciprocal();

    /** Absolute values of the entries in the vector */
    inline
    void ElementWiseAbs();

    /** Element-wise square root of the entries in the vector */
    inline
    void ElementWiseSqrt();

    /** Replaces the vector values with their sgn values
    ( -1 if x_i < 0, 0 if x_i == 0, and 1 if x_i > 0)
    */
    inline
    void ElementWiseSgn();

    /** Add scalar to every vector component */
    inline
    void AddScalar(Number scalar);

    /** Returns the maximum value in the vector */
    inline
    Number Max() const;

    /** Returns the minimum value in the vector */
    inline
    Number Min() const;

    /** Returns the sum of the vector entries */
    inline
    Number Sum() const;

    /** Returns the sum of the logs of each vector entry */
    inline
    Number SumLogs() const;
    //@}

    /** @name Methods for specialized operations.  A prototype
     *  implementation is provided, but for efficient implementation
     *  those should be specially implemented.
     */
    //@{
    /** Add one vector, y = a * v1 + c * y.  This is automatically
     *  reduced to call AddTwoVectors.  */
    inline
    void AddOneVector(Number a, const Vector& v1, Number c);

    /** Add two vectors, y = a * v1 + b * v2 + c * y.  Here, this
     *  vector is y */
    inline void AddTwoVectors(Number a, const Vector& v1,
                       Number b, const Vector& v2, Number c);
    /** Fraction to the boundary parameter.  Computes \f$\alpha =
     *  \max\{\bar\alpha\in(0,1] : x + \bar\alpha \Delta \geq (1-\tau)x\}\f$
     */
    inline
    Number FracToBound(const Vector& delta, Number tau) const;
    /** Add the quotient of two vectors, y = a * z/s + c * y. */
    inline
    void AddVectorQuotient(Number a, const Vector& z, const Vector& s,
                           Number c);
    //@}

    /** Method for determining if all stored numbers are valid (i.e.,
     *  no Inf or Nan). */
    inline
    bool HasValidNumbers() const;

    /** @name Accessor methods */
    //@{
    /** Dimension of the Vector */
    inline
    Index Dim() const;

    /** Return the owner VectorSpace*/
    inline
    SmartPtr<const VectorSpace> OwnerSpace() const;
    //@}

    /** @name Output methods
     *  (derived classes do NOT overload these 
     *  methods, instead, overload the 
     *  protected versions of these methods). */
    //@{
    /** Print the entire vector */
    void Print(SmartPtr<const Journalist> jnlst,
               EJournalLevel level,
               EJournalCategory category,
               const std::string& name,
               Index indent=0,
               const std::string& prefix="") const;
    void Print(const Journalist& jnlst,
               EJournalLevel level,
               EJournalCategory category,
               const std::string& name,
               Index indent=0,
               const std::string& prefix="") const;
    //@}

  protected:
    /** @name implementation methods (derived classes MUST
     *  overload these pure virtual protected methods.)
     */
    //@{
    /** Copy the data of the vector x into this vector (DCOPY). */
    virtual void CopyImpl(const Vector& x)=0;

    /** Scales the vector by scalar alpha (DSCAL) */
    virtual void ScalImpl(Number alpha)=0;

    /** Add the multiple alpha of vector x to this vector (DAXPY) */
    virtual void AxpyImpl(Number alpha, const Vector &x)=0;

    /** Computes inner product of vector x with this (DDOT) */
    virtual Number DotImpl(const Vector &x) const =0;

    /** Computes the 2-norm of this vector (DNRM2) */
    virtual Number Nrm2Impl() const =0;

    /** Computes the 1-norm of this vector (DASUM) */
    virtual Number AsumImpl() const =0;

    /** Computes the max-norm of this vector (based on IDAMAX) */
    virtual Number AmaxImpl() const =0;

    /** Set each element in the vector to the scalar alpha. */
    virtual void SetImpl(Number alpha)=0;

    /** Element-wise division  \f$y_i \gets y_i/x_i\f$*/
    virtual void ElementWiseDivideImpl(const Vector& x)=0;

    /** Element-wise multiplication \f$y_i \gets y_i*x_i\f$ */
    virtual void ElementWiseMultiplyImpl(const Vector& x)=0;

    /** Element-wise max against entries in x */
    virtual void ElementWiseMaxImpl(const Vector& x)=0;

    /** Element-wise min against entries in x */
    virtual void ElementWiseMinImpl(const Vector& x)=0;

    /** Reciprocates the elements of the vector */
    virtual void ElementWiseReciprocalImpl()=0;

    /** Take elementwise absolute values of the elements of the vector */
    virtual void ElementWiseAbsImpl()=0;

    /** Take elementwise square-root of the elements of the vector */
    virtual void ElementWiseSqrtImpl()=0;

    /** Replaces entries with sgn of the entry */
    virtual void ElementWiseSgnImpl()=0;

    /** Add scalar to every component of vector */
    virtual void AddScalarImpl(Number scalar)=0;

    /** Max value in the vector */
    virtual Number MaxImpl() const=0;

    /** Min number in the vector */
    virtual Number MinImpl() const=0;

    /** Sum of entries in the vector */
    virtual Number SumImpl() const=0;

    /** Sum of logs of entries in the vector */
    virtual Number SumLogsImpl() const=0;

    /** Add two vectors (a * v1 + b * v2).  Result is stored in this
    vector. */
    virtual void AddTwoVectorsImpl(Number a, const Vector& v1,
                                   Number b, const Vector& v2, Number c);

    /** Fraction to boundary parameter. */
    virtual Number FracToBoundImpl(const Vector& delta, Number tau) const;

    /** Add the quotient of two vectors */
    virtual void AddVectorQuotientImpl(Number a, const Vector& z,
                                       const Vector& s, Number c);

    /** Method for determining if all stored numbers are valid (i.e.,
     *  no Inf or Nan). A default implementation using Asum is
     *  provided. */
    virtual bool HasValidNumbersImpl() const;

    /** Print the entire vector */
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
    /** Default constructor */
    Vector();

    /** Copy constructor */
    Vector(const Vector&);

    /** Overloaded Equals Operator */
    Vector& operator=(const Vector&);
    //@}

    /** Vector Space */
    const SmartPtr<const VectorSpace> owner_space_;

    /**@name CachedResults data members */
    //@{
    /** Cache for dot products */
    mutable CachedResults<Number> dot_cache_;

    mutable TaggedObject::Tag nrm2_cache_tag_;
    mutable Number cached_nrm2_;

    mutable TaggedObject::Tag asum_cache_tag_;
    mutable Number cached_asum_;

    mutable TaggedObject::Tag amax_cache_tag_;
    mutable Number cached_amax_;

    mutable TaggedObject::Tag max_cache_tag_;
    mutable Number cached_max_;

    mutable TaggedObject::Tag min_cache_tag_;
    mutable Number cached_min_;

    mutable TaggedObject::Tag sum_cache_tag_;
    mutable Number cached_sum_;

    mutable TaggedObject::Tag sumlogs_cache_tag_;
    mutable Number cached_sumlogs_;

    mutable TaggedObject::Tag valid_cache_tag_;
    mutable bool cached_valid_;

    //     AW: I removed this cache since it gets in the way for the
    //         quality function search
    //     /** Cache for FracToBound */
    //     mutable CachedResults<Number> frac_to_bound_cache_;
    //@}

  };

  /** VectorSpace base class, corresponding to the Vector base class.
   *  For each Vector implementation, a corresponding VectorSpace has
   *  to be implemented.  A VectorSpace is able to create new Vectors
   *  of a specific type.  The VectorSpace should also store
   *  information that is common to all Vectors of that type.  For
   *  example, the dimension of a Vector is stored in the VectorSpace
   *  base class.
   */
  class VectorSpace : public ReferencedObject
  {
  public:
    /** @name Constructors/Destructors */
    //@{
    /** Constructor, given the dimension of all vectors generated by
     *  this VectorSpace.
     */
    VectorSpace(Index dim);

    /** Destructor */
    virtual ~VectorSpace()
    {}
    //@}

    /** Pure virtual method for creating a new Vector of the
     *  corresponding type.
     */
    virtual Vector* MakeNew() const=0;

    /** Accessor function for the dimension of the vectors of this type.*/
    Index Dim() const
    {
      return dim_;
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
    VectorSpace();

    /** Copy constructor */
    VectorSpace(const VectorSpace&);

    /** Overloaded Equals Operator */
    VectorSpace& operator=(const VectorSpace&);
    //@}

    /** Dimension of the vectors in this vector space. */
    const Index dim_;
  };

  /* inline methods */
  inline
  Vector::~Vector()
  {}

  inline
  Vector::Vector(const VectorSpace* owner_space)
      :
      TaggedObject(),
      owner_space_(owner_space),
      dot_cache_(10),
      nrm2_cache_tag_(0),
      asum_cache_tag_(0),
      amax_cache_tag_(0),
      max_cache_tag_(0),
      min_cache_tag_(0),
      sum_cache_tag_(0),
      sumlogs_cache_tag_(0),
      cached_valid_(0)
  {
    DBG_ASSERT(IsValid(owner_space_));
  }

  inline
  Vector* Vector::MakeNew() const
  {
    return owner_space_->MakeNew();
  }

  inline
  Vector* Vector::MakeNewCopy() const
  {
    // ToDo: We can probably copy also the cached values for Norms etc here
    Vector* copy = MakeNew();
    copy->Copy(*this);
    return copy;
  }

  inline
  void Vector::Copy(const Vector& x)
  {
    CopyImpl(x);
    ObjectChanged();
    // Also copy any cached scalar values from the original vector
    // ToDo: Check if that is too much overhead
    TaggedObject::Tag x_tag = x.GetTag();
    if (x_tag == x.nrm2_cache_tag_) {
      nrm2_cache_tag_ = GetTag();
      cached_nrm2_ = x.cached_nrm2_;
    }
    if (x_tag == x.asum_cache_tag_) {
      asum_cache_tag_ = GetTag();
      cached_asum_ = x.cached_asum_;
    }
    if (x_tag == x.amax_cache_tag_) {
      amax_cache_tag_ = GetTag();
      cached_amax_ = x.cached_amax_;
    }
    if (x_tag == x.max_cache_tag_) {
      max_cache_tag_ = GetTag();
      cached_max_ = x.cached_max_;
    }
    if (x_tag == x.min_cache_tag_) {
      min_cache_tag_ = GetTag();
      cached_min_ = x.cached_min_;
    }
    if (x_tag == x.sum_cache_tag_) {
      sum_cache_tag_ = GetTag();
      cached_sum_ = x.cached_sum_;
    }
    if (x_tag == x.sumlogs_cache_tag_) {
      sumlogs_cache_tag_ = GetTag();
      cached_sumlogs_ = x.cached_sumlogs_;
    }
  }

  inline
  void Vector::Axpy(Number alpha, const Vector &x)
  {
    AxpyImpl(alpha, x);
    ObjectChanged();
  }

  inline
  Number Vector::Dot(const Vector &x) const
  {
    // The current implementation of the caching doesn't allow to have
    // a dependency of something with itself.  Therefore, we use the
    // Nrm2 method if the dot product is to be taken with the vector
    // itself.  Might be more efficient anyway.
    if (this==&x) {
      Number nrm2 = Nrm2();
      return nrm2*nrm2;
    }
    Number retValue;
    if (!dot_cache_.GetCachedResult2Dep(retValue, this, &x)) {
      retValue = DotImpl(x);
      dot_cache_.AddCachedResult2Dep(retValue, this, &x);
    }
    return retValue;
  }

  inline
  Number Vector::Nrm2() const
  {
    if (nrm2_cache_tag_ != GetTag()) {
      cached_nrm2_ = Nrm2Impl();
      nrm2_cache_tag_ = GetTag();
    }
    return cached_nrm2_;
  }

  inline
  Number Vector::Asum() const
  {
    if (asum_cache_tag_ != GetTag()) {
      cached_asum_ = AsumImpl();
      asum_cache_tag_ = GetTag();
    }
    return cached_asum_;
  }

  inline
  Number Vector::Amax() const
  {
    if (amax_cache_tag_ != GetTag()) {
      cached_amax_ = AmaxImpl();
      amax_cache_tag_ = GetTag();
    }
    return cached_amax_;
  }

  inline
  Number Vector::Sum() const
  {
    if (sum_cache_tag_ != GetTag()) {
      cached_sum_ = SumImpl();
      sum_cache_tag_ = GetTag();
    }
    return cached_sum_;
  }

  inline
  Number Vector::SumLogs() const
  {
    if (sumlogs_cache_tag_ != GetTag()) {
      cached_sumlogs_ = SumLogsImpl();
      sumlogs_cache_tag_ = GetTag();
    }
    return cached_sumlogs_;
  }

  inline
  void Vector::ElementWiseSgn()
  {
    ElementWiseSgnImpl();
    ObjectChanged();
  }

  inline
  void Vector::Set(Number alpha)
  {
    // Could initialize caches here
    SetImpl(alpha);
    ObjectChanged();
  }

  inline
  void Vector::ElementWiseDivide(const Vector& x)
  {
    ElementWiseDivideImpl(x);
    ObjectChanged();
  }

  inline
  void Vector::ElementWiseMultiply(const Vector& x)
  {
    ElementWiseMultiplyImpl(x);
    ObjectChanged();
  }

  inline
  void Vector::ElementWiseReciprocal()
  {
    ElementWiseReciprocalImpl();
    ObjectChanged();
  }

  inline
  void Vector::ElementWiseMax(const Vector& x)
  {
    // Could initialize some caches here
    ElementWiseMaxImpl(x);
    ObjectChanged();
  }

  inline
  void Vector::ElementWiseMin(const Vector& x)
  {
    // Could initialize some caches here
    ElementWiseMinImpl(x);
    ObjectChanged();
  }

  inline
  void Vector::ElementWiseAbs()
  {
    // Could initialize some caches here
    ElementWiseAbsImpl();
    ObjectChanged();
  }

  inline
  void Vector::ElementWiseSqrt()
  {
    ElementWiseSqrtImpl();
    ObjectChanged();
  }

  inline
  void Vector::AddScalar(Number scalar)
  {
    // Could initialize some caches here
    AddScalarImpl(scalar);
    ObjectChanged();
  }

  inline
  Number Vector::Max() const
  {
    if (max_cache_tag_ != GetTag()) {
      cached_max_ = MaxImpl();
      max_cache_tag_ = GetTag();
    }
    return cached_max_;
  }

  inline
  Number Vector::Min() const
  {
    if (min_cache_tag_ != GetTag()) {
      cached_min_ = MinImpl();
      min_cache_tag_ = GetTag();
    }
    return cached_min_;
  }

  inline
  void Vector::AddOneVector(Number a, const Vector& v1, Number c)
  {
    AddTwoVectors(a, v1, 0., v1, c);
  }

  inline
  void Vector::AddTwoVectors(Number a, const Vector& v1,
                             Number b, const Vector& v2, Number c)
  {
    AddTwoVectorsImpl(a, v1, b, v2, c);
    ObjectChanged();
  }

  inline
  Number Vector::FracToBound(const Vector& delta, Number tau) const
  {
    /* AW: I avoid the caching here, since it leads to overhead in the
       quality function search.  Caches for this are in
       CalculatedQuantities.
    Number retValue;
    std::vector<const TaggedObject*> tdeps(1);
    tdeps[0] = &delta;
    std::vector<Number> sdeps(1);
    sdeps[0] = tau;
    if (!frac_to_bound_cache_.GetCachedResult(retValue, tdeps, sdeps)) {
      retValue = FracToBoundImpl(delta, tau);
      frac_to_bound_cache_.AddCachedResult(retValue, tdeps, sdeps);
    }
    return retValue;
    */
    return FracToBoundImpl(delta, tau);
  }

  inline
  void Vector::AddVectorQuotient(Number a, const Vector& z,
                                 const Vector& s, Number c)
  {
    AddVectorQuotientImpl(a, z, s, c);
    ObjectChanged();
  }

  inline
  bool Vector::HasValidNumbers() const
  {
    if (valid_cache_tag_ != GetTag()) {
      cached_valid_ = HasValidNumbersImpl();
      valid_cache_tag_ = GetTag();
    }
    return cached_valid_;
  }

  inline
  Index Vector::Dim() const
  {
    return owner_space_->Dim();
  }

  inline
  SmartPtr<const VectorSpace> Vector::OwnerSpace() const
  {
    return owner_space_;
  }

  inline
  VectorSpace::VectorSpace(Index dim)
      :
      dim_(dim)
  {}

} // namespace Ipopt

// Macro definitions for debugging vectors
#if COIN_IPOPT_VERBOSITY == 0
# define DBG_PRINT_VECTOR(__verbose_level, __vec_name, __vec)
#else
# define DBG_PRINT_VECTOR(__verbose_level, __vec_name, __vec) \
   if (dbg_jrnl.Verbosity() >= (__verbose_level)) { \
      if (dbg_jrnl.Jnlst()!=NULL) { \
        (__vec).Print(dbg_jrnl.Jnlst(), \
        J_ERROR, J_DBG, \
        __vec_name, \
        dbg_jrnl.IndentationLevel()*2, \
        "# "); \
      } \
   }
#endif //if COIN_IPOPT_VERBOSITY == 0

#endif
