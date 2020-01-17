// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpDenseVector.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPDENSEVECTOR_HPP__
#define __IPDENSEVECTOR_HPP__

#include "IpUtils.hpp"
#include "IpVector.hpp"
#include <map>

namespace Ipopt
{

  /* forward declarations */
  class DenseVectorSpace;

  /** @name Exceptions */
  //@{
  DECLARE_STD_EXCEPTION(METADATA_ERROR);
  //@}

  /** Dense Vector Implementation.  This is the default Vector class
   *  in Ipopt.  It stores vectors in contiguous Number arrays, unless
   *  the vector has the same value in all entires.  In the latter
   *  case, we call the vector "homogeneous", and we store only the
   *  values that is repeated in all elements.  If you want to obtain
   *  the values of vector, use the IsHomogeneous() method to find out
   *  what status the vector is in, and then use either Values() const
   *  or Scalar() const methods to get the values.  To set the values
   *  of a homogeneous method, use the Set method.  To set the values
   *  of a non-homogeneous vector, use the SetValues method, or use
   *  the non-const Values method to get an array that you can
   *  overwrite.  In the latter case, storage is ensured.
   */
  class DenseVector : public Vector
  {
  public:

    /**@name Constructors / Destructors */
    //@{
    /** Default Constructor
     */
    DenseVector(const DenseVectorSpace* owner_space);

    /** Destructor
     */
    virtual ~DenseVector();
    //@}

    /** @name Additional public methods not in Vector base class. */
    //@{
    /** Create a new DenseVector from same VectorSpace */
    SmartPtr<DenseVector> MakeNewDenseVector() const;

    /** Set elements in the vector to the Number array x. */
    void SetValues(const Number *x);

    /** Obtain pointer to the internal Number array with vector
     *  elements with the indention to change the vector data (USE
     *  WITH CARE!). This does not produce a copy, and lifetime is not
     *  guaranteed!. 
     */
    inline Number* Values();

    /** Obtain pointer to the internal Number array with vector
     *  elements without the intention to change the vector data (USE
     *  WITH CARE!). This does not produce a copy, and lifetime is not
     *  guaranteed!  IMPORTANT: If this method is currently
     *  homogeneous (i.e. IsHomogeneous returns true), then you cannot
     *  call this method.  Instead, you need to use the Scalar()
     *  method.
     */
    inline const Number* Values() const;

    /** The same as the const version of Values, but we ensure that we
     *  always return a valid array, even if IsHomogeneous returns
     *  true. */
    const Number* ExpandedValues() const;

    /** This is the same as Values, but we add it here so that
     *  ExpandedValues can also be used for the non-const case. */
    inline Number* ExpandedValues()
    {
      return Values();
    }

    /** Indicates if the vector is homogeneous (i.e., all entries have
     *  the value Scalar() */
    bool IsHomogeneous() const
    {
      return homogeneous_;
    }

    /** Scalar value of all entries in a homogeneous vector */
    Number Scalar() const
    {
      DBG_ASSERT(homogeneous_);
      return scalar_;
    }
    //@}

    /** @name Modifying subranges of the vector. */
    //@{
    /** Copy the data in x into the subrange of this vector starting
     *  at position Pos in this vector.  Position count starts at 0.
     */
    void CopyToPos(Index Pos, const Vector& x);
    /** Copy a subrange of x, starting at Pos, into the full data of
     *  this vector.  Position count starts at 0.
     */
    void CopyFromPos(Index Pos, const Vector& x);
    //@}

  protected:
    /** @name Overloaded methods from Vector base class */
    //@{
    /** Copy the data of the vector x into this vector (DCOPY). */
    virtual void CopyImpl(const Vector& x);

    /** Scales the vector by scalar alpha (DSCAL) */
    virtual void ScalImpl(Number alpha);

    /** Add the multiple alpha of vector x to this vector (DAXPY) */
    virtual void AxpyImpl(Number alpha, const Vector &x);

    /** Computes inner product of vector x with this (DDOT) */
    virtual Number DotImpl(const Vector &x) const;

    /** Computes the 2-norm of this vector (DNRM2) */
    virtual Number Nrm2Impl() const;

    /** Computes the 1-norm of this vector (DASUM) */
    virtual Number AsumImpl() const;

    /** Computes the max-norm of this vector (based on IDAMAX) */
    virtual Number AmaxImpl() const;

    /** Set each element in the vector to the scalar alpha. */
    virtual void SetImpl(Number value);

    /** Element-wise division  \f$y_i \gets y_i/x_i\f$.*/
    virtual void ElementWiseDivideImpl(const Vector& x);

    /** Element-wise multiplication \f$y_i \gets y_i*x_i\f$.*/
    virtual void ElementWiseMultiplyImpl(const Vector& x);

    /** Set entry to max of itself and the corresponding element in x */
    virtual void ElementWiseMaxImpl(const Vector& x);

    /** Set entry to min of itself and the corresponding element in x */
    virtual void ElementWiseMinImpl(const Vector& x);

    /** reciprocates the elements of the vector */
    virtual void ElementWiseReciprocalImpl();

    /** take abs of the elements of the vector */
    virtual void ElementWiseAbsImpl();

    /** take square-root of the elements of the vector */
    virtual void ElementWiseSqrtImpl();

    /** Changes each entry in the vector to its sgn value */
    virtual void ElementWiseSgnImpl();

    /** Add scalar to every component of the vector.*/
    virtual void AddScalarImpl(Number scalar);

    /** Max value in the vector */
    virtual Number MaxImpl() const;

    /** Min value in the vector */
    virtual Number MinImpl() const;

    /** Computes the sum of the lements of vector */
    virtual Number SumImpl() const;

    /** Computes the sum of the logs of the elements of vector */
    virtual Number SumLogsImpl() const;

    /** @name Implemented specialized functions */
    //@{
    /** Add two vectors (a * v1 + b * v2).  Result is stored in this
    vector. */
    void AddTwoVectorsImpl(Number a, const Vector& v1,
                           Number b, const Vector& v2, Number c);
    /** Fraction to the boundary parameter. */
    Number FracToBoundImpl(const Vector& delta, Number tau) const;
    /** Add the quotient of two vectors, y = a * z/s + c * y. */
    void AddVectorQuotientImpl(Number a, const Vector& z, const Vector& s,
                               Number c);
    //@}

    /** @name Output methods */
    //@{
    /* Print the entire vector with padding */
    virtual void PrintImpl(const Journalist& jnlst,
                           EJournalLevel level,
                           EJournalCategory category,
                           const std::string& name,
                           Index indent,
                           const std::string& prefix) const
    {
      PrintImplOffset(jnlst, level, category, name, indent, prefix, 1);
    }
    /* Print the entire vector with padding, and start counting with
       an offset. */
    void PrintImplOffset(const Journalist& jnlst,
                         EJournalLevel level,
                         EJournalCategory category,
                         const std::string& name,
                         Index indent,
                         const std::string& prefix,
                         Index offset) const;
    //@}
    friend class ParVector;

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
    DenseVector();

    /** Copy Constructor */
    DenseVector(const DenseVector&);

    /** Overloaded Equals Operator */
    void operator=(const DenseVector&);
    //@}

    /** Copy of the owner_space ptr as a DenseVectorSpace instead
     *  of a VectorSpace
     */
    const DenseVectorSpace* owner_space_;

    /** Dense Number array of vector values. */
    Number* values_;

    /** Dense Number array pointer that is used for ExpandedValues */
    mutable Number* expanded_values_;

    /** Method of getting the internal values array, making sure that
     *  memory has been allocated */
    inline
    Number* values_allocated();

    /** Flag for Initialization.  This flag is false, if the data has
    not yet been initialized. */
    bool initialized_;

    /** Flag indicating whether the vector is currently homogeneous
     *  (that is, all elements have the same value). This flag is used
     *  to determine whether the elements of the vector are stored in
     *  values_ or in scalar_ */
    bool homogeneous_;

    /** Homogeneous value of all elements if the vector is currently
     *  homogenous */
    Number scalar_;

    /** Auxilliary method for setting explicitly all elements in
     *  values_ to the current scalar value. */
    void set_values_from_scalar();
  };

  /** typedefs for the map variables that define meta data for the
   *  DenseVectorSpace
   */
  typedef std::map<std::string, std::vector<std::string> > StringMetaDataMapType;
  typedef std::map<std::string, std::vector<Index> > IntegerMetaDataMapType;
  typedef std::map<std::string, std::vector<Number> > NumericMetaDataMapType;

  /** This vectors space is the vector space for DenseVector.
   */
  class DenseVectorSpace : public VectorSpace
  {
  public:
    /** @name Constructors/Destructors. */
    //@{
    /** Constructor, requires dimension of all vector for this
     *  VectorSpace
     */
    DenseVectorSpace(Index dim)
        :
        VectorSpace(dim)
    {}

    /** Destructor */
    ~DenseVectorSpace()
    {}
    //@}

    /** Method for creating a new vector of this specific type. */
    inline
    DenseVector* MakeNewDenseVector() const
    {
      return new DenseVector(this);
    }

    /** Instantiation of the generate MakeNew method for the
     *  VectorSpace base class.
     */
    virtual Vector* MakeNew() const
    {
      return MakeNewDenseVector();
    }

    /**@name Methods called by DenseVector for memory management.
     * This could allow to have sophisticated memory management in the
     * VectorSpace.
     */
    //@{
    /** Allocate internal storage for the DenseVector */
    inline
    Number* AllocateInternalStorage() const;

    /** Deallocate internal storage for the DenseVector */
    inline
    void FreeInternalStorage(Number* values) const;
    //@}

    /**@name Methods for dealing with meta data on the vector
     */
    //@{
    /** Check if string meta exists for tag */
    inline
    bool HasStringMetaData(const std::string tag) const;

    /** Check if Integer meta exists for tag */
    inline
    bool HasIntegerMetaData(const std::string tag) const;

    /** Check if Numeric meta exists for tag */
    inline
    bool HasNumericMetaData(const std::string tag) const;

    /** Get meta data of type std::string by tag */
    inline
    const std::vector<std::string>& GetStringMetaData(const std::string& tag) const;

    /** Get meta data of type Index by tag */
    inline
    const std::vector<Index>& GetIntegerMetaData(const std::string& tag) const;

    /** Get meta data of type Number by tag */
    inline
    const std::vector<Number>& GetNumericMetaData(const std::string& tag) const;

    /** Set meta data of type std::string by tag */
    inline
    void SetStringMetaData(std::string tag, std::vector<std::string> meta_data);

    /** Set meta data of type Index by tag */
    inline
    void SetIntegerMetaData(std::string tag, std::vector<Index> meta_data);

    /** Set meta data of type Number by tag */
    inline
    void SetNumericMetaData(std::string tag, std::vector<Number> meta_data);

    /** Get map of meta data of type Number */
    inline
    const StringMetaDataMapType& GetStringMetaData() const;

    /** Get map of meta data of type Number */
    inline
    const IntegerMetaDataMapType& GetIntegerMetaData() const;

    /** Get map of meta data of type Number */
    inline
    const NumericMetaDataMapType& GetNumericMetaData() const;
    //@}

  private:
    // variables to store vector meta data
    StringMetaDataMapType string_meta_data_;
    IntegerMetaDataMapType integer_meta_data_;
    NumericMetaDataMapType numeric_meta_data_;

  };

  // inline functions
  inline Number* DenseVector::Values()
  {
    // Here we assume that every time someone requests this direct raw
    // pointer, the data is going to change and the Tag for this
    // vector has to be updated.

    if (initialized_ && homogeneous_) {
      // If currently the vector is a homogeneous vector, set all elements
      // explicitly to this value
      set_values_from_scalar();
    }
    ObjectChanged();
    initialized_= true;
    homogeneous_ = false;
    return values_allocated();
  }

  inline const Number* DenseVector::Values() const
  {
    DBG_ASSERT(initialized_ && (Dim()==0 || values_));
    return values_;
  }

  inline Number* DenseVector::values_allocated()
  {
    if (values_==NULL) {
      values_ = owner_space_->AllocateInternalStorage();
    }
    return values_;
  }

  inline
  Number* DenseVectorSpace::AllocateInternalStorage() const
  {
    if (Dim()>0) {
      return new Number[Dim()];
    }
    else {
      return NULL;
    }
  }

  inline
  void DenseVectorSpace::FreeInternalStorage(Number* values) const
  {
    delete [] values;
  }

  inline
  SmartPtr<DenseVector> DenseVector::MakeNewDenseVector() const
  {
    return owner_space_->MakeNewDenseVector();
  }

  inline
  bool DenseVectorSpace::HasStringMetaData(const std::string tag) const
  {
    StringMetaDataMapType::const_iterator iter;
    iter = string_meta_data_.find(tag);

    if (iter != string_meta_data_.end()) {
      return true;
    }

    return false;
  }

  inline
  bool DenseVectorSpace::HasIntegerMetaData(const std::string tag) const
  {
    IntegerMetaDataMapType::const_iterator iter;
    iter = integer_meta_data_.find(tag);

    if (iter != integer_meta_data_.end()) {
      return true;
    }

    return false;
  }

  inline
  bool DenseVectorSpace::HasNumericMetaData(const std::string tag) const
  {
    NumericMetaDataMapType::const_iterator iter;
    iter = numeric_meta_data_.find(tag);

    if (iter != numeric_meta_data_.end()) {
      return true;
    }

    return false;
  }

  inline
  const std::vector<std::string>& DenseVectorSpace::GetStringMetaData(const std::string& tag) const
  {
    DBG_ASSERT(HasStringMetaData(tag));
    StringMetaDataMapType::const_iterator iter;
    iter = string_meta_data_.find(tag);
    return iter->second;
  }

  inline
  const std::vector<Index>& DenseVectorSpace::GetIntegerMetaData(const std::string& tag) const
  {
    DBG_ASSERT(HasIntegerMetaData(tag));
    IntegerMetaDataMapType::const_iterator iter;
    iter = integer_meta_data_.find(tag);
    return iter->second;
  }

  inline
  const std::vector<Number>& DenseVectorSpace::GetNumericMetaData(const std::string& tag) const
  {
    DBG_ASSERT(HasNumericMetaData(tag));
    NumericMetaDataMapType::const_iterator iter;
    iter = numeric_meta_data_.find(tag);
    return iter->second;
  }

  inline
  void DenseVectorSpace::SetStringMetaData(std::string tag, std::vector<std::string> meta_data)
  {
    string_meta_data_[tag] = meta_data;
  }

  inline
  void DenseVectorSpace::SetIntegerMetaData(std::string tag, std::vector<Index> meta_data)
  {
    integer_meta_data_[tag] = meta_data;
  }

  inline
  void DenseVectorSpace::SetNumericMetaData(std::string tag, std::vector<Number> meta_data)
  {
    numeric_meta_data_[tag] = meta_data;
  }

  inline
  const StringMetaDataMapType& DenseVectorSpace::GetStringMetaData() const
  {
    return string_meta_data_;
  }

  inline
  const IntegerMetaDataMapType& DenseVectorSpace::GetIntegerMetaData() const
  {
    return integer_meta_data_;
  }

  inline
  const NumericMetaDataMapType& DenseVectorSpace::GetNumericMetaData() const
  {
    return numeric_meta_data_;
  }

} // namespace Ipopt
#endif
