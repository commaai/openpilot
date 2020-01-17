// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpTaggedObject.hpp 2613 2015-11-04 14:42:02Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPTAGGEDOBJECT_HPP__
#define __IPTAGGEDOBJECT_HPP__

#include "IpUtils.hpp"
#include "IpDebug.hpp"
#include "IpReferenced.hpp"
#include "IpObserver.hpp"
#include <limits>

/* keyword to declare a thread-local variable according to http://en.wikipedia.org/wiki/Thread-local_storage
 * GCC < 4.5 on MacOS X does not support TLS
 * With Intel compiler on MacOS X, problems with TLS were reported.
 */
#ifndef IPOPT_THREAD_LOCAL

#if defined(_MSC_VER)
#define IPOPT_THREAD_LOCAL __declspec(thread)
#elif defined(__APPLE__) && ((defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ < 405)) || defined(__INTEL_COMPILER))
#define IPOPT_THREAD_LOCAL
#else
#define IPOPT_THREAD_LOCAL __thread
#endif

#endif

namespace Ipopt
{

  /** TaggedObject class.
   * Often, certain calculations or operations are expensive,
   * and it can be very inefficient to perform these calculations
   * again if the input to the calculation has not changed
   * since the result was last stored. 
   * This base class provides an efficient mechanism to update
   * a tag, indicating that the object has changed.
   * Users of a TaggedObject class, need their own Tag data
   * member to keep track of the state of the TaggedObject, the
   * last time they performed a calculation. A basic use case for 
   * users of a class inheriting from TaggedObject follows like
   * this:
   * 
   *  1. Initialize your own Tag to zero in constructor.
   *  
   *  2. Before an expensive calculation,
   *      check if the TaggedObject has changed, passing in
   *      your own Tag, indicating the last time you used
   *      the object for the calculation. If it has changed,
   *      perform the calculation again, and store the result.
   *      If it has not changed, simply return the stored result.
   * 
   *      Here is a simple example:
   \verbatim
          if (vector.HasChanged(my_vector_tag_)) {
            my_vector_tag_ = vector.GetTag();
            result = PerformExpensiveCalculation(vector);
            return result;
          }
          else {
            return result;
          }
   \endverbatim
   * 
   *  Objects derived from TaggedObject must indicate that they have changed to
   *  the base class using the protected member function ObjectChanged(). For
   *  example, a Vector class, inside its own set method, MUST call 
   *  ObjectChanged() to update the internally stored tag for comparison.
   */
  class TaggedObject : public ReferencedObject, public Subject
  {
  public:
    /** Type for the Tag values */
    typedef unsigned int Tag;

    /** Constructor. */
    TaggedObject()
        :
        Subject()
    {
      ObjectChanged();
    }

    /** Destructor. */
    virtual ~TaggedObject()
    {}

    /** Users of TaggedObjects call this to
     *  update their own internal tags every time
     *  they perform the expensive operation.
     */
    Tag GetTag() const
    {
      return tag_;
    }

    /** Users of TaggedObjects call this to
     *  check if the object HasChanged since
     *  they last updated their own internal
     *  tag.
     */
    bool HasChanged(const Tag comparison_tag) const
    {
      return (comparison_tag == tag_) ? false : true;
    }
  protected:
    /** Objects derived from TaggedObject MUST call this
     *  method every time their internal state changes to 
     *  update the internal tag for comparison
     */
    void ObjectChanged()
    {
      DBG_START_METH("TaggedObject::ObjectChanged()", 0);
      tag_ = unique_tag_;
      unique_tag_++;
      DBG_ASSERT(unique_tag_ < std::numeric_limits<Tag>::max());
      // The Notify method from the Subject base class notifies all
      // registered Observers that this subject has changed.
      Notify(Observer::NT_Changed);
    }
  private:
    /**@name Default Compiler Generated Methods (Hidden to avoid
     * implicit creation/calling).  These methods are not implemented
     * and we do not want the compiler to implement them for us, so we
     * declare them private and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Copy Constructor */
    TaggedObject(const TaggedObject&);

    /** Overloaded Equals Operator */
    void operator=(const TaggedObject&);
    //@}

    /** static data member that is incremented every
     *  time ANY TaggedObject changes. This allows us
     *  to obtain a unique Tag when the object changes
     */
    static IPOPT_THREAD_LOCAL Tag unique_tag_;

    /** The tag indicating the current state of the object.
     *  We use this to compare against the comparison_tag
     *  in the HasChanged method. This member is updated
     *  from the unique_tag_ every time the object changes.
     */
    Tag tag_;

    /** The index indicating the cache priority for this
     * TaggedObject. If a result that depended on this 
     * TaggedObject is cached, it will be cached with this
     * priority
     */
    Index cache_priority_;
  };
} // namespace Ipopt
#endif
