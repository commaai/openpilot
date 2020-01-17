// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpReferenced.hpp 2182 2013-03-30 20:02:18Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPREFERENCED_HPP__
#define __IPREFERENCED_HPP__

#include "IpTypes.hpp"
#include "IpDebug.hpp"

#include <list>

#if COIN_IPOPT_CHECKLEVEL > 3
  #define IP_DEBUG_REFERENCED
#endif

namespace Ipopt
{

  /** Psydo-class, from which everything has to inherit that wants to
   *  use be registered as a Referencer for a ReferencedObject.
   */
  class Referencer
    {}
  ;

  /** ReferencedObject class.
   * This is part of the implementation of an intrusive smart pointer 
   * design. This class stores the reference count of all the smart
   * pointers that currently reference it. See the documentation for
   * the SmartPtr class for more details.
   * 
   * A SmartPtr behaves much like a raw pointer, but manages the lifetime 
   * of an object, deleting the object automatically. This class implements
   * a reference-counting, intrusive smart pointer design, where all
   * objects pointed to must inherit off of ReferencedObject, which
   * stores the reference count. Although this is intrusive (native types
   * and externally authored classes require wrappers to be referenced
   * by smart pointers), it is a safer design. A more detailed discussion of
   * these issues follows after the usage information.
   * 
   * Usage Example:
   * Note: to use the SmartPtr, all objects to which you point MUST
   * inherit off of ReferencedObject.
   * 
   * \verbatim
   * 
   * In MyClass.hpp...
   * 
   * #include "IpReferenced.hpp"

   * namespace Ipopt {
   * 
   *  class MyClass : public ReferencedObject // must derive from ReferencedObject
   *    {
   *      ...
   *    }
   * } // namespace Ipopt
   * 
   * 
   * In my_usage.cpp...
   * 
   * #include "IpSmartPtr.hpp"
   * #include "MyClass.hpp"
   * 
   * void func(AnyObject& obj)
   *  {
   *    SmartPtr<MyClass> ptr_to_myclass = new MyClass(...);
   *    // ptr_to_myclass now points to a new MyClass,
   *    // and the reference count is 1
   *  
   *    ...
   * 
   *    obj.SetMyClass(ptr_to_myclass);
   *    // Here, let's assume that AnyObject uses a
   *    // SmartPtr<MyClass> internally here.
   *    // Now, both ptr_to_myclass and the internal
   *    // SmartPtr in obj point to the same MyClass object
   *    // and its reference count is 2.
   * 
   *    ...
   * 
   *    // No need to delete ptr_to_myclass, this
   *    // will be done automatically when the
   *    // reference count drops to zero.
   * 
   *  }  
   *  
   * \endverbatim
   * 
   * Other Notes:
   *  The SmartPtr implements both dereference operators -> & *.
   *  The SmartPtr does NOT implement a conversion operator to
   *    the raw pointer. Use the GetRawPtr() method when this
   *    is necessary. Make sure that the raw pointer is NOT
   *    deleted. 
   *  The SmartPtr implements the comparison operators == & !=
   *    for a variety of types. Use these instead of
   *    \verbatim
   *    if (GetRawPtr(smrt_ptr) == ptr) // Don't use this
   *    \endverbatim
   *  SmartPtr's, as currently implemented, do NOT handle circular references.
   *    For example: consider a higher level object using SmartPtrs to point to 
   *    A and B, but A and B also point to each other (i.e. A has a SmartPtr 
   *    to B and B has a SmartPtr to A). In this scenario, when the higher
   *    level object is finished with A and B, their reference counts will 
   *    never drop to zero (since they reference each other) and they
   *    will not be deleted. This can be detected by memory leak tools like
   *    valgrind. If the circular reference is necessary, the problem can be
   *    overcome by a number of techniques:
   *  
   *    1) A and B can have a method that "releases" each other, that is
   *        they set their internal SmartPtrs to NULL.
   *        \verbatim
   *        void AClass::ReleaseCircularReferences()
   *          {
   *          smart_ptr_to_B = NULL;
   *          }
   *        \endverbatim
   *        Then, the higher level class can call these methods before
   *        it is done using A & B.
   * 
   *    2) Raw pointers can be used in A and B to reference each other.
   *        Here, an implicit assumption is made that the lifetime is
   *        controlled by the higher level object and that A and B will
   *        both exist in a controlled manner. Although this seems 
   *        dangerous, in many situations, this type of referencing
   *        is very controlled and this is reasonably safe.
   * 
   *    3) This SmartPtr class could be redesigned with the Weak/Strong
   *        design concept. Here, the SmartPtr is identified as being
   *        Strong (controls lifetime of the object) or Weak (merely
   *        referencing the object). The Strong SmartPtr increments 
   *        (and decrements) the reference count in ReferencedObject
   *        but the Weak SmartPtr does not. In the example above,
   *        the higher level object would have Strong SmartPtrs to
   *        A and B, but A and B would have Weak SmartPtrs to each
   *        other. Then, when the higher level object was done with
   *        A and B, they would be deleted. The Weak SmartPtrs in A
   *        and B would not decrement the reference count and would,
   *        of course, not delete the object. This idea is very similar
   *        to item (2), where it is implied that the sequence of events 
   *        is controlled such that A and B will not call anything using
   *        their pointers following the higher level delete (i.e. in
   *        their destructors!). This is somehow safer, however, because
   *        code can be written (however expensive) to perform run-time 
   *        detection of this situation. For example, the ReferencedObject
   *        could store pointers to all Weak SmartPtrs that are referencing
   *        it and, in its destructor, tell these pointers that it is
   *        dying. They could then set themselves to NULL, or set an
   *        internal flag to detect usage past this point.
   *
   *   For every most derived object only one ReferencedObject may exist,
   *   that is multiple inheritance requires virtual inheritance, see also
   *   the 2nd point in ticket #162.
   * 
   * Comments on Non-Intrusive Design:
   * In a non-intrusive design, the reference count is stored somewhere other
   * than the object being referenced. This means, unless the reference
   * counting pointer is the first referencer, it must get a pointer to the 
   * referenced object from another smart pointer (so it has access to the 
   * reference count location). In this non-intrusive design, if we are 
   * pointing to an object with a smart pointer (or a number of smart
   * pointers), and we then give another smart pointer the address through
   * a RAW pointer, we will have two independent, AND INCORRECT, reference
   * counts. To avoid this pitfall, we use an intrusive reference counting
   * technique where the reference count is stored in the object being
   * referenced. 
   */
  class ReferencedObject
  {
  public:
    ReferencedObject()
        :
        reference_count_(0)
    {}

    virtual ~ReferencedObject()
    {
      DBG_ASSERT(reference_count_ == 0);
    }

    inline
    Index ReferenceCount() const;

    inline
    void AddRef(const Referencer* referencer) const;

    inline
    void ReleaseRef(const Referencer* referencer) const;

  private:
    mutable Index reference_count_;

#   ifdef IP_DEBUG_REFERENCED
    mutable std::list<const Referencer*> referencers_;
#   endif

  };

  /* inline methods */
  inline
  Index ReferencedObject::ReferenceCount() const
  {
    //    DBG_START_METH("ReferencedObject::ReferenceCount()", 0);
    //    DBG_PRINT((1,"Returning reference_count_ = %d\n", reference_count_));
    return reference_count_;
  }

  inline
  void ReferencedObject::AddRef(const Referencer* referencer) const
  {
    //    DBG_START_METH("ReferencedObject::AddRef(const Referencer* referencer)", 0);
    reference_count_++;
    //    DBG_PRINT((1, "New reference_count_ = %d\n", reference_count_));
#   ifdef IP_DEBUG_REFERENCED
    referencers_.push_back(referencer);
#   endif

  }

  inline
  void ReferencedObject::ReleaseRef(const Referencer* referencer) const
  {
    //    DBG_START_METH("ReferencedObject::ReleaseRef(const Referencer* referencer)",
    //                   0);
    reference_count_--;
    //    DBG_PRINT((1, "New reference_count_ = %d\n", reference_count_));

#   ifdef IP_DEBUG_REFERENCED

    bool found = false;
    std::list<const Referencer*>::iterator iter;
    for (iter = referencers_.begin(); iter != referencers_.end(); iter++) {
      if ((*iter) == referencer) {
        found = true;
        break;
      }
    }

    // cannot call release on a reference that was never added...
    DBG_ASSERT(found);

    if (found) {
      referencers_.erase(iter);
    }
#   endif

  }


} // namespace Ipopt

#endif
