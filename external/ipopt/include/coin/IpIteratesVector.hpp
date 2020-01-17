// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpIteratesVector.hpp 2472 2014-04-05 17:47:20Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-06-06

#ifndef __IPITERATESVECTOR_HPP__
#define __IPITERATESVECTOR_HPP__

#include "IpCompoundVector.hpp"

namespace Ipopt
{
  /* forward declarations */
  class IteratesVectorSpace;

  /** Specialized CompoundVector class specifically for the algorithm
   *  iterates.  This class inherits from CompoundVector and is a
   *  specialized class for handling the iterates of the Ipopt
   *  Algorithm, that is, x, s, y_c, y_d, z_L, z_U, v_L, and v_U. It
   *  inherits from CompoundVector so it can behave like a CV in most
   *  calculations, but it has fixed dimensions and cannot be
   *  customized
   */
  class IteratesVector : public CompoundVector
  {
  public:
    /** Constructors / Destructors */
    //@{
    IteratesVector(const IteratesVectorSpace* owner_space, bool create_new);

    virtual ~IteratesVector();
    //@}

    /** Make New methods */
    //@{
    /** Use this method to create a new iterates vector. The MakeNew
     *  method on the Vector class also works, but it does not give
     *  the create_new option.
     */
    SmartPtr<IteratesVector> MakeNewIteratesVector(bool create_new = true) const;

    /** Use this method to create a new iterates vector with a copy of
     *  all the data.
     */
    SmartPtr<IteratesVector> MakeNewIteratesVectorCopy() const
    {
      SmartPtr<IteratesVector> ret = MakeNewIteratesVector(true);
      ret->Copy(*this);
      return ret;
    }

    /** Use this method to create a new iterates vector
     *  container. This creates a new NonConst container, but the
     *  elements inside the iterates vector may be const. Therefore,
     *  the container can be modified to point to new entries, but the
     *  existing entries may or may not be modifiable.
     */
    SmartPtr<IteratesVector> MakeNewContainer() const;
    //@}

    /** Iterates Set/Get Methods */
    //@{
    /** Get the x iterate (const) */
    SmartPtr<const Vector> x() const
    {
      return GetIterateFromComp(0);
    }

    /** Get the x iterate (non-const) - this can only be called if the
     *  vector was created intenally, or the Set_x_NonConst method was
     *  used. */
    SmartPtr<Vector> x_NonConst()
    {
      return GetNonConstIterateFromComp(0);
    }

    /** Create a new vector in the x entry */
    inline
    SmartPtr<Vector> create_new_x();

    /** Create a new vector in the x entry and copy the current values
     *  into it. */
    SmartPtr<Vector> create_new_x_copy()
    {
      SmartPtr<const Vector> curr_x = GetComp(0);
      Set_x_NonConst(*curr_x->MakeNew());
      x_NonConst()->Copy(*curr_x);
      return x_NonConst();
    }

    /** Set the x iterate (const). Sets the pointer, does NOT copy
     *  data. */
    void Set_x(const Vector& vec)
    {
      SetComp(0, vec);
    }

    /** Set the x iterate (non-const). Sets the pointer, does NOT copy
     *  data. */
    void Set_x_NonConst(Vector& vec)
    {
      SetCompNonConst(0, vec);
    }

    /** Get the s iterate (const) */
    SmartPtr<const Vector> s() const
    {
      return GetIterateFromComp(1);
    }

    /** Get the s iterate (non-const) - this can only be called if the
     *  vector was created intenally, or the Set_s_NonConst method was
     *  used. */
    SmartPtr<Vector> s_NonConst()
    {
      return GetNonConstIterateFromComp(1);
    }

    /** Create a new vector in the s entry */
    inline
    SmartPtr<Vector> create_new_s();

    /** Create a new vector in the s entry and copy the current values
     *  into it. */
    SmartPtr<Vector> create_new_s_copy()
    {
      SmartPtr<const Vector> curr_s = GetComp(1);
      Set_s_NonConst(*curr_s->MakeNew());
      s_NonConst()->Copy(*curr_s);
      return s_NonConst();
    }

    /** Set the s iterate (const). Sets the pointer, does NOT copy
     *  data. */
    void Set_s(const Vector& vec)
    {
      SetComp(1, vec);
    }

    /** Set the s iterate (non-const). Sets the pointer, does NOT copy
     *  data. */
    void Set_s_NonConst(Vector& vec)
    {
      SetCompNonConst(1, vec);
    }

    /** Get the y_c iterate (const) */
    SmartPtr<const Vector> y_c() const
    {
      return GetIterateFromComp(2);
    }

    /** Get the y_c iterate (non-const) - this can only be called if
     *  the vector was created intenally, or the Set_y_c_NonConst
     *  method was used. */
    SmartPtr<Vector> y_c_NonConst()
    {
      return GetNonConstIterateFromComp(2);
    }

    /** Create a new vector in the y_c entry */
    inline
    SmartPtr<Vector> create_new_y_c();

    /** Create a new vector in the y_c entry and copy the current
     *  values into it. */
    SmartPtr<Vector> create_new_y_c_copy()
    {
      SmartPtr<const Vector> curr_y_c = GetComp(2);
      Set_y_c_NonConst(*curr_y_c->MakeNew());
      y_c_NonConst()->Copy(*curr_y_c);
      return y_c_NonConst();
    }

    /** Set the y_c iterate (const). Sets the pointer, does NOT copy
     *  data. */
    void Set_y_c(const Vector& vec)
    {
      SetComp(2, vec);
    }

    /** Set the y_c iterate (non-const). Sets the pointer, does NOT
     *  copy data. */
    void Set_y_c_NonConst(Vector& vec)
    {
      SetCompNonConst(2, vec);
    }

    /** Get the y_d iterate (const) */
    SmartPtr<const Vector> y_d() const
    {
      return GetIterateFromComp(3);
    }

    /** Get the y_d iterate (non-const) - this can only be called if
     *  the vector was created intenally, or the Set_y_d_NonConst
     *  method was used. */
    SmartPtr<Vector> y_d_NonConst()
    {
      return GetNonConstIterateFromComp(3);
    }

    /** Create a new vector in the y_d entry */
    inline
    SmartPtr<Vector> create_new_y_d();

    /** Create a new vector in the y_d entry and copy the current
     *  values into it. */
    SmartPtr<Vector> create_new_y_d_copy()
    {
      SmartPtr<const Vector> curr_y_d = GetComp(3);
      Set_y_d_NonConst(*curr_y_d->MakeNew());
      y_d_NonConst()->Copy(*curr_y_d);
      return y_d_NonConst();
    }

    /** Set the y_d iterate (const). Sets the pointer, does NOT copy
     *  data. */
    void Set_y_d(const Vector& vec)
    {
      SetComp(3, vec);
    }

    /** Set the y_d iterate (non-const). Sets the pointer, does NOT
     *  copy data. */
    void Set_y_d_NonConst(Vector& vec)
    {
      SetCompNonConst(3, vec);
    }

    /** Get the z_L iterate (const) */
    SmartPtr<const Vector> z_L() const
    {
      return GetIterateFromComp(4);
    }

    /** Get the z_L iterate (non-const) - this can only be called if
     *  the vector was created intenally, or the Set_z_L_NonConst
     *  method was used. */
    SmartPtr<Vector> z_L_NonConst()
    {
      return GetNonConstIterateFromComp(4);
    }

    /** Create a new vector in the z_L entry */
    inline
    SmartPtr<Vector> create_new_z_L();

    /** Create a new vector in the z_L entry and copy the current
     *  values into it. */
    SmartPtr<Vector> create_new_z_L_copy()
    {
      SmartPtr<const Vector> curr_z_L = GetComp(4);
      Set_z_L_NonConst(*curr_z_L->MakeNew());
      z_L_NonConst()->Copy(*curr_z_L);
      return z_L_NonConst();
    }

    /** Set the z_L iterate (const). Sets the pointer, does NOT copy
     *  data. */
    void Set_z_L(const Vector& vec)
    {
      SetComp(4, vec);
    }

    /** Set the z_L iterate (non-const). Sets the pointer, does NOT
     *  copy data. */
    void Set_z_L_NonConst(Vector& vec)
    {
      SetCompNonConst(4, vec);
    }

    /** Get the z_U iterate (const) */
    SmartPtr<const Vector> z_U() const
    {
      return GetIterateFromComp(5);
    }

    /** Get the z_U iterate (non-const) - this can only be called if
     *  the vector was created intenally, or the Set_z_U_NonConst
     *  method was used. */
    SmartPtr<Vector> z_U_NonConst()
    {
      return GetNonConstIterateFromComp(5);
    }

    /** Create a new vector in the z_U entry */
    inline
    SmartPtr<Vector> create_new_z_U();

    /** Create a new vector in the z_U entry and copy the current
     *  values into it. */
    SmartPtr<Vector> create_new_z_U_copy()
    {
      SmartPtr<const Vector> curr_z_U = GetComp(5);
      Set_z_U_NonConst(*curr_z_U->MakeNew());
      z_U_NonConst()->Copy(*curr_z_U);
      return z_U_NonConst();
    }

    /** Set the z_U iterate (const). Sets the pointer, does NOT copy
     *  data. */
    void Set_z_U(const Vector& vec)
    {
      SetComp(5, vec);
    }

    /** Set the z_U iterate (non-const). Sets the pointer, does NOT
     *  copy data. */
    void Set_z_U_NonConst(Vector& vec)
    {
      SetCompNonConst(5, vec);
    }

    /** Get the v_L iterate (const) */
    SmartPtr<const Vector> v_L() const
    {
      return GetIterateFromComp(6);
    }

    /** Get the v_L iterate (non-const) - this can only be called if
     *  the vector was created intenally, or the Set_v_L_NonConst
     *  method was used. */
    SmartPtr<Vector> v_L_NonConst()
    {
      return GetNonConstIterateFromComp(6);
    }

    /** Create a new vector in the v_L entry */
    inline
    SmartPtr<Vector> create_new_v_L();

    /** Create a new vector in the v_L entry and copy the current
     *  values into it. */
    SmartPtr<Vector> create_new_v_L_copy()
    {
      SmartPtr<const Vector> curr_v_L = GetComp(6);
      Set_v_L_NonConst(*curr_v_L->MakeNew());
      v_L_NonConst()->Copy(*curr_v_L);
      return v_L_NonConst();
    }

    /** Set the v_L iterate (const). Sets the pointer, does NOT copy
     *  data. */
    void Set_v_L(const Vector& vec)
    {
      SetComp(6, vec);
    }

    /** Set the v_L iterate (non-const). Sets the pointer, does NOT
     *  copy data. */
    void Set_v_L_NonConst(Vector& vec)
    {
      SetCompNonConst(6, vec);
    }

    /** Get the v_U iterate (const) */
    SmartPtr<const Vector> v_U() const
    {
      return GetIterateFromComp(7);
    }

    /** Get the v_U iterate (non-const) - this can only be called if
     *  the vector was created intenally, or the Set_v_U_NonConst
     *  method was used. */
    SmartPtr<Vector> v_U_NonConst()
    {
      return GetNonConstIterateFromComp(7);
    }

    /** Create a new vector in the v_U entry */
    inline
    SmartPtr<Vector> create_new_v_U();

    /** Create a new vector in the v_U entry and copy the current
     *  values into it. */
    SmartPtr<Vector> create_new_v_U_copy()
    {
      SmartPtr<const Vector> curr_v_U = GetComp(7);
      Set_v_U_NonConst(*curr_v_U->MakeNew());
      v_U_NonConst()->Copy(*curr_v_U);
      return v_U_NonConst();
    }

    /** Set the v_U iterate (const). Sets the pointer, does NOT copy
     *  data. */
    void Set_v_U(const Vector& vec)
    {
      SetComp(7, vec);
    }

    /** Set the v_U iterate (non-const). Sets the pointer, does NOT
     *  copy data. */
    void Set_v_U_NonConst(Vector& vec)
    {
      SetCompNonConst(7, vec);
    }

    /** Set the primal variables all in one shot. Sets the pointers,
     *  does NOT copy data */
    void Set_primal(const Vector& x, const Vector& s)
    {
      SetComp(0, x);
      SetComp(1, s);
    }
    void Set_primal_NonConst(Vector& x, Vector& s)
    {
      SetCompNonConst(0, x);
      SetCompNonConst(1, s);
    }

    /** Set the eq multipliers all in one shot. Sets the pointers,
     *  does not copy data. */
    void Set_eq_mult(const Vector& y_c, const Vector& y_d)
    {
      SetComp(2, y_c);
      SetComp(3, y_d);
    }
    void Set_eq_mult_NonConst(Vector& y_c, Vector& y_d)
    {
      SetCompNonConst(2, y_c);
      SetCompNonConst(3, y_d);
    }

    /** Set the bound multipliers all in one shot. Sets the pointers,
     *  does not copy data. */
    void Set_bound_mult(const Vector& z_L, const Vector& z_U, const Vector& v_L, const Vector& v_U)
    {
      SetComp(4, z_L);
      SetComp(5, z_U);
      SetComp(6, v_L);
      SetComp(7, v_U);
    }
    void Set_bound_mult_NonConst(Vector& z_L, Vector& z_U, Vector& v_L, Vector& v_U)
    {
      SetCompNonConst(4, z_L);
      SetCompNonConst(5, z_U);
      SetCompNonConst(6, v_L);
      SetCompNonConst(7, v_U);
    }

    /** Get a sum of the tags of the contained items. There is no
     *  guarantee that this is unique, but there is a high chance it
     *  is unique and it can be used for debug checks relatively
     *  reliably.
     */
    TaggedObject::Tag GetTagSum() const
    {
      TaggedObject::Tag tag = 0;

      if (IsValid(x())) {
        tag += x()->GetTag();
      }
      if (IsValid(s())) {
        tag += s()->GetTag();
      }
      if (IsValid(y_c())) {
        tag += y_c()->GetTag();
      }
      if (IsValid(y_d())) {
        tag += y_d()->GetTag();
      }
      if (IsValid(z_L())) {
        tag += z_L()->GetTag();
      }
      if (IsValid(z_U())) {
        tag += z_U()->GetTag();
      }
      if (IsValid(v_L())) {
        tag += v_L()->GetTag();
      }
      if (IsValid(v_U())) {
        tag += v_U()->GetTag();
      }

      return tag;
    }
    //@}

  private:
    /**@name Default Compiler Generated Methods (Hidden to avoid
     * implicit creation/calling).  These methods are not implemented
     * and we do not want the compiler to implement them for us, so we
     * declare them private and do not define them. This ensures that
     * they will not be implicitly created/called.
     */
    //@{
    /** Default Constructor */
    IteratesVector();

    /** Copy Constructor */
    IteratesVector(const IteratesVector&);

    /** Overloaded Equals Operator */
    void operator=(const IteratesVector&);
    //@}

    const IteratesVectorSpace* owner_space_;

    /** private method to return the const element from the compound
     *  vector.  This method will return NULL if none is currently
     *  set.
     */
    SmartPtr<const Vector> GetIterateFromComp(Index i) const
    {
      if (IsCompNull(i)) {
        return NULL;
      }
      return GetComp(i);
    }

    /** private method to return the non-const element from the
     *  compound vector.  This method will return NULL if none is
     *  currently set.
     */
    SmartPtr<Vector> GetNonConstIterateFromComp(Index i)
    {
      if (IsCompNull(i)) {
        return NULL;
      }
      return GetCompNonConst(i);
    }

  };

  /** Vector Space for the IteratesVector class.  This is a
   *  specialized vector space for the IteratesVector class.
   */
  class IteratesVectorSpace : public CompoundVectorSpace
  {
  public:
    /** @name Constructors/Destructors. */
    //@{
    /** Constructor that takes the spaces for each of the iterates.
     *  Warning! None of these can be NULL ! 
     */
    IteratesVectorSpace(const VectorSpace& x_space, const VectorSpace& s_space,
                        const VectorSpace& y_c_space, const VectorSpace& y_d_space,
                        const VectorSpace& z_L_space, const VectorSpace& z_U_space,
                        const VectorSpace& v_L_space, const VectorSpace& v_U_space
                       );

    virtual ~IteratesVectorSpace();
    //@}

    /** Method for creating vectors . */
    //@{
    /** Use this to create a new IteratesVector. You can pass-in
     *  create_new = false if you only want a container and do not
     *  want vectors allocated.
     */
    virtual IteratesVector* MakeNewIteratesVector(bool create_new = true) const
    {
      return new IteratesVector(this, create_new);
    }

    /** Use this method to create a new const IteratesVector. You must pass in
     *  valid pointers for all of the entries.
     */
    const SmartPtr<const IteratesVector> MakeNewIteratesVector(const Vector& x, const Vector& s,
        const Vector& y_c, const Vector& y_d,
        const Vector& z_L, const Vector& z_U,
        const Vector& v_L, const Vector& v_U)
    {
      SmartPtr<IteratesVector> newvec = MakeNewIteratesVector(false);
      newvec->Set_x(x);
      newvec->Set_s(s);
      newvec->Set_y_c(y_c);
      newvec->Set_y_d(y_d);
      newvec->Set_z_L(z_L);
      newvec->Set_z_U(z_U);
      newvec->Set_v_L(v_L);
      newvec->Set_v_U(v_U);
      return ConstPtr(newvec);
    }


    /** This method overloads
     *  ComooundVectorSpace::MakeNewCompoundVector to make sure that
     *  we get a vector of the correct type
     */
    virtual CompoundVector* MakeNewCompoundVector(bool create_new = true) const
    {
      return MakeNewIteratesVector(create_new);
    }

    /** This method creates a new vector (and allocates space in all
     *  the contained vectors. This is really only used for code that
     *  does not know what type of vector it is dealing with - for
     *  example, this method is called from Vector::MakeNew()
     */
    virtual Vector* MakeNew() const
    {
      return MakeNewIteratesVector();
    }
    //@}

    /** This method hides the CompoundVectorSpace::SetCompSpace method
     *  since the components of the Iterates are fixed at
     *  construction.
     */
    virtual void SetCompSpace(Index icomp, const VectorSpace& vec_space)
    {
      DBG_ASSERT(false && "This is an IteratesVectorSpace - a special compound vector for Ipopt iterates. The contained spaces should not be modified.");
    }

  private:
    /**@name Default Compiler Generated Methods (Hidden to avoid
    * implicit creation/calling).  These methods are not implemented
    * and we do not want the compiler to implement them for us, so we
    * declare them private and do not define them. This ensures that
    * they will not be implicitly created/called. */
    //@{
    /** Default constructor */
    IteratesVectorSpace();

    /** Copy Constructor */
    IteratesVectorSpace(const IteratesVectorSpace&);

    /** Overloaded Equals Operator */
    IteratesVectorSpace& operator=(const IteratesVectorSpace&);
    //@}

    /** Contained Spaces */
    SmartPtr<const VectorSpace> x_space_;
    SmartPtr<const VectorSpace> s_space_;
    SmartPtr<const VectorSpace> y_c_space_;
    SmartPtr<const VectorSpace> y_d_space_;
    SmartPtr<const VectorSpace> z_L_space_;
    SmartPtr<const VectorSpace> z_U_space_;
    SmartPtr<const VectorSpace> v_L_space_;
    SmartPtr<const VectorSpace> v_U_space_;
  };


  inline
  SmartPtr<Vector> IteratesVector::create_new_x()
  {
    Set_x_NonConst(*owner_space_->GetCompSpace(0)->MakeNew());
    return x_NonConst();
  }
  inline
  SmartPtr<Vector> IteratesVector::create_new_s()
  {
    Set_s_NonConst(*owner_space_->GetCompSpace(1)->MakeNew());
    return s_NonConst();
  }
  inline
  SmartPtr<Vector> IteratesVector::create_new_y_c()
  {
    Set_y_c_NonConst(*owner_space_->GetCompSpace(2)->MakeNew());
    return y_c_NonConst();
  }
  inline
  SmartPtr<Vector> IteratesVector::create_new_y_d()
  {
    Set_y_d_NonConst(*owner_space_->GetCompSpace(3)->MakeNew());
    return y_d_NonConst();
  }
  inline
  SmartPtr<Vector> IteratesVector::create_new_z_L()
  {
    Set_z_L_NonConst(*owner_space_->GetCompSpace(4)->MakeNew());
    return z_L_NonConst();
  }
  inline
  SmartPtr<Vector> IteratesVector::create_new_z_U()
  {
    Set_z_U_NonConst(*owner_space_->GetCompSpace(5)->MakeNew());
    return z_U_NonConst();
  }
  inline
  SmartPtr<Vector> IteratesVector::create_new_v_L()
  {
    Set_v_L_NonConst(*owner_space_->GetCompSpace(6)->MakeNew());
    return v_L_NonConst();
  }
  inline
  SmartPtr<Vector> IteratesVector::create_new_v_U()
  {
    Set_v_U_NonConst(*owner_space_->GetCompSpace(7)->MakeNew());
    return v_U_NonConst();
  }
} // namespace Ipopt

#endif
