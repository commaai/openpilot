// Copyright (C) 2004, 2011 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpCachedResults.hpp 2472 2014-04-05 17:47:20Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPCACHEDRESULTS_HPP__
#define __IPCACHEDRESULTS_HPP__

#include "IpTaggedObject.hpp"
#include "IpObserver.hpp"
#include <algorithm>
#include <vector>
#include <list>

namespace Ipopt
{

#if COIN_IPOPT_CHECKLEVEL > 2
# define IP_DEBUG_CACHE
#endif
#ifdef IP_DEBUG_CACHE
# include "IpDebug.hpp"
#endif

  // Forward Declarations

  template <class T>
  class DependentResult;

  //  AW: I'm taking this out, since this is by far the most used
  //  class.  We should keep it as simple as possible.
  //   /** Cache Priority Enum */
  //   enum CachePriority
  //   {
  //     CP_Lowest,
  //     CP_Standard,
  //     CP_Trial,
  //     CP_Iterate
  //   };

  /** Templated class for Cached Results.  This class stores up to a
   *  given number of "results", entities that are stored here
   *  together with identifiers, that can be used to later retrieve the
   *  information again.
   *
   *  Typically, T is a SmartPtr for some calculated quantity that
   *  should be stored (such as a Vector).  The identifiers (or
   *  dependencies) are a (possibly varying) number of Tags from
   *  TaggedObjects, and a number of Numbers.  Results are added to
   *  the cache using the AddCachedResults methods, and the can be
   *  retrieved with the GetCachedResults methods. The second set of
   *  methods checks whether a result has been cached for the given
   *  identifiers.  If a corresponding result is found, a copy of it
   *  is returned and the method evaluates to true, otherwise it
   *  evaluates to false.
   *
   *  Note that cached results can become "stale", namely when a
   *  TaggedObject that is used to identify this CachedResult is
   *  changed.  When this happens, the cached result can never be
   *  asked for again, so that there is no point in storing it any
   *  longer.  For this purpose, a cached result, which is stored as a
   *  DependentResult, inherits off an Observer.  This Observer
   *  retrieves notification whenever a TaggedObject dependency has
   *  changed.  Stale results are later removed from the cache.
   */
  template <class T>
  class CachedResults
  {
  public:
#ifdef IP_DEBUG_CACHE
    /** (Only if compiled in DEBUG mode): debug verbosity level */
    static const Index dbg_verbosity;
#endif

    /** @name Constructors and Destructors. */
    //@{
    /** Constructor, where max_cache_size is the maximal number of
     *  results that should be cached.  If max_cache_size is negative,
     *  we allow an infinite amount of cache.
     */
    CachedResults(Int max_cache_size);

    /** Destructor */
    virtual ~CachedResults();
    //@}

    /** @name Generic methods for adding and retrieving cached results. */
    //@{
    /** Generic method for adding a result to the cache, given a
     *  std::vector of TaggesObjects and a std::vector of Numbers.
     */
    void AddCachedResult(const T& result,
                         const std::vector<const TaggedObject*>& dependents,
                         const std::vector<Number>& scalar_dependents);

    /** Generic method for retrieving a cached results, given the
     *  dependencies as a std::vector of TaggesObjects and a
     *  std::vector of Numbers.
     */
    bool GetCachedResult(T& retResult,
                         const std::vector<const TaggedObject*>& dependents,
                         const std::vector<Number>& scalar_dependents) const;

    /** Method for adding a result, providing only a std::vector of
     *  TaggedObjects.
     */
    void AddCachedResult(const T& result,
                         const std::vector<const TaggedObject*>& dependents);

    /** Method for retrieving a cached result, providing only a
     * std::vector of TaggedObjects.
     */
    bool GetCachedResult(T& retResult,
                         const std::vector<const TaggedObject*>& dependents) const;
    //@}

    /** @name Pointer-based methods for adding and retrieving cached
     *  results, providing dependencies explicitly.
     */
    //@{
    /** Method for adding a result to the cache, proving one
     *  dependency as a TaggedObject explicitly.
     */
    void AddCachedResult1Dep(const T& result,
                             const TaggedObject* dependent1);

    /** Method for retrieving a cached result, proving one dependency
     *  as a TaggedObject explicitly.
     */
    bool GetCachedResult1Dep(T& retResult, const TaggedObject* dependent1);

    /** Method for adding a result to the cache, proving two
     *  dependencies as a TaggedObject explicitly.
     */
    void AddCachedResult2Dep(const T& result,
                             const TaggedObject* dependent1,
                             const TaggedObject* dependent2);

    /** Method for retrieving a cached result, proving two
     *  dependencies as a TaggedObject explicitly.
     */
    bool GetCachedResult2Dep(T& retResult,
                             const TaggedObject* dependent1,
                             const TaggedObject* dependent2);

    /** Method for adding a result to the cache, proving three
     *  dependencies as a TaggedObject explicitly.
     */
    void AddCachedResult3Dep(const T& result,
                             const TaggedObject* dependent1,
                             const TaggedObject* dependent2,
                             const TaggedObject* dependent3);

    /** Method for retrieving a cached result, proving three
     *  dependencies as a TaggedObject explicitly.
     */
    bool GetCachedResult3Dep(T& retResult,
                             const TaggedObject* dependent1,
                             const TaggedObject* dependent2,
                             const TaggedObject* dependent3);

    /** @name Pointer-free version of the Add and Get methods */
    //@{
    bool GetCachedResult1Dep(T& retResult, const TaggedObject& dependent1)
    {
      return GetCachedResult1Dep(retResult, &dependent1);
    }
    bool GetCachedResult2Dep(T& retResult,
                             const TaggedObject& dependent1,
                             const TaggedObject& dependent2)
    {
      return GetCachedResult2Dep(retResult, &dependent1, &dependent2);
    }
    bool GetCachedResult3Dep(T& retResult,
                             const TaggedObject& dependent1,
                             const TaggedObject& dependent2,
                             const TaggedObject& dependent3)
    {
      return GetCachedResult3Dep(retResult, &dependent1, &dependent2, &dependent3);
    }
    void AddCachedResult1Dep(const T& result,
                             const TaggedObject& dependent1)
    {
      AddCachedResult1Dep(result, &dependent1);
    }
    void AddCachedResult2Dep(const T& result,
                             const TaggedObject& dependent1,
                             const TaggedObject& dependent2)
    {
      AddCachedResult2Dep(result, &dependent1, &dependent2);
    }
    void AddCachedResult3Dep(const T& result,
                             const TaggedObject& dependent1,
                             const TaggedObject& dependent2,
                             const TaggedObject& dependent3)
    {
      AddCachedResult3Dep(result, &dependent1, &dependent2, &dependent3);
    }
    //@}

    /** Invalidates the result for given dependencies. Sets the stale
     *  flag for the corresponding cached result to true if it is
     *  found.  Returns true, if the result was found. */
    bool InvalidateResult(const std::vector<const TaggedObject*>& dependents,
                          const std::vector<Number>& scalar_dependents);

    /** Invalidates all cached results */
    void Clear();

    /** Invalidate all cached results and changes max_cache_size */
    void Clear(Int max_cache_size);

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
    CachedResults();

    /** Copy Constructor */
    CachedResults(const CachedResults&);

    /** Overloaded Equals Operator */
    void operator=(const CachedResults&);
    //@}

    /** maximum number of cached results */
    Int max_cache_size_;

    /** list of currently cached results. */
    mutable std::list<DependentResult<T>*>* cached_results_;

    /** internal method for removing stale DependentResults from the
     *  list.  It is called at the beginning of every
     *  GetDependentResult method.
     */
    void CleanupInvalidatedResults() const;

    /** Print list of currently cached results */
    void DebugPrintCachedResults() const;
  };

  /** Templated class which stores one entry for the CachedResult
   *  class.  It stores the result (of type T), together with its
   *  dependencies (vector of TaggedObjects and vector of Numbers).
   *  It also stores a priority.
   */
  template <class T>
  class DependentResult : public Observer
  {
  public:

#ifdef IP_DEBUG_CACHE
    static const Index dbg_verbosity;
#endif

    /** @name Constructor, Destructors */
    //@{
    /** Constructor, given all information about the result. */
    DependentResult(const T& result, const std::vector<const TaggedObject*>& dependents,
                    const std::vector<Number>& scalar_dependents);

    /** Destructor. */
    ~DependentResult();
    //@}

    /** @name Accessor method. */
    //@{
    /** This returns true, if the DependentResult is no longer valid. */
    bool IsStale() const;

    /** Invalidates the cached result. */
    void Invalidate();

    /** Returns the cached result. */
    const T& GetResult() const;
    //@}

    /** This method returns true if the dependencies provided to this
     *  function are identical to the ones stored with the
     *  DependentResult.
     */
    bool DependentsIdentical(const std::vector<const TaggedObject*>& dependents,
                             const std::vector<Number>& scalar_dependents) const;

    /** Print information about this DependentResults. */
    void DebugPrint() const;

  protected:
    /** This method is overloading the pure virtual method from the
     *  Observer base class.  This method is called when a Subject
     *  registered for this Observer sends a notification.  In this
     *  particular case, if this method is called with
     *  notify_type==NT_Changed or NT_BeingDeleted, then this results
     *  is marked as stale.
     */
    virtual void RecieveNotification(NotifyType notify_type, const Subject* subject);

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
    DependentResult();

    /** Copy Constructor */
    DependentResult(const DependentResult&);

    /** Overloaded Equals Operator */
    void operator=(const DependentResult&);
    //@}

    /** Flag indicating, if the cached result is still valid.  A
    result becomes invalid, if the RecieveNotification method is
    called with NT_Changed */
    bool stale_;
    /** The value of the dependent results */
    const T result_;
    /** Dependencies in form of TaggedObjects */
    std::vector<TaggedObject::Tag> dependent_tags_;
    /** Dependencies in form a Numbers */
    std::vector<Number> scalar_dependents_;
  };

#ifdef IP_DEBUG_CACHE
  template <class T>
  const Index CachedResults<T>::dbg_verbosity = 0;

  template <class T>
  const Index DependentResult<T>::dbg_verbosity = 0;
#endif

  template <class T>
  DependentResult<T>::DependentResult(
    const T& result,
    const std::vector<const TaggedObject*>& dependents,
    const std::vector<Number>& scalar_dependents)
      :
      stale_(false),
      result_(result),
      dependent_tags_(dependents.size()),
      scalar_dependents_(scalar_dependents)
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("DependentResult<T>::DependentResult()", dbg_verbosity);
#endif

    for (Index i=0; i<(Index)dependents.size(); i++) {
      if (dependents[i]) {
        // Call the RequestAttach method of the Observer base class.
        // This will add this dependent result in the Observer list
        // for the Subject dependents[i].  As a consequence, the
        // RecieveNotification method of this DependentResult will be
        // called with notify_type=NT_Changed, whenever the
        // TaggedResult dependents[i] is changed (i.e. its HasChanged
        // method is called).
        RequestAttach(NT_Changed, dependents[i]);
        dependent_tags_[i] = dependents[i]->GetTag();
      }
      else {
        dependent_tags_[i] = 0;
      }
    }
  }

  template <class T>
  DependentResult<T>::~DependentResult()
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("DependentResult<T>::~DependentResult()", dbg_verbosity);
    //DBG_ASSERT(stale_ == true);
#endif
    // Nothing to be done here, destructor
    // of T should sufficiently remove
    // any memory, etc.
  }

  template <class T>
  bool DependentResult<T>::IsStale() const
  {
    return stale_;
  }

  template <class T>
  void DependentResult<T>::Invalidate()
  {
    stale_ = true;
  }

  template <class T>
  void DependentResult<T>::RecieveNotification(NotifyType notify_type, const Subject* subject)
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("DependentResult<T>::RecieveNotification", dbg_verbosity);
#endif

    if (notify_type == NT_Changed || notify_type==NT_BeingDestroyed) {
      stale_ = true;
      // technically, I could unregister the notifications here, but they
      // aren't really hurting anything
    }
  }

  template <class T>
  bool DependentResult<T>::DependentsIdentical(const std::vector<const TaggedObject*>& dependents,
      const std::vector<Number>& scalar_dependents) const
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("DependentResult<T>::DependentsIdentical", dbg_verbosity);
    DBG_ASSERT(stale_ == false);
    DBG_ASSERT(dependents.size() == dependent_tags_.size());
#endif

    bool retVal = true;

    if (dependents.size() != dependent_tags_.size()
        || scalar_dependents.size() != scalar_dependents_.size()) {
      retVal = false;
    }
    else {
      for (Index i=0; i<(Index)dependents.size(); i++) {
        if ( (dependents[i] && dependents[i]->GetTag() != dependent_tags_[i])
             || (!dependents[i] && dependent_tags_[i] != 0) ) {
          retVal = false;
          break;
        }
      }
      if (retVal) {
        for (Index i=0; i<(Index)scalar_dependents.size(); i++) {
          if (scalar_dependents[i] != scalar_dependents_[i]) {
            retVal = false;
            break;
          }
        }
      }
    }

    return retVal;
  }

  template <class T>
  const T& DependentResult<T>::GetResult() const
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("DependentResult<T>::GetResult()", dbg_verbosity);
    DBG_ASSERT(stale_ == false);
#endif

    return result_;
  }

  template <class T>
  void DependentResult<T>::DebugPrint() const
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("DependentResult<T>::DebugPrint", dbg_verbosity);
#endif

  }

  template <class T>
  CachedResults<T>::CachedResults(Int max_cache_size)
      :
      max_cache_size_(max_cache_size),
      cached_results_(NULL)
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::CachedResults", dbg_verbosity);
#endif

  }

  template <class T>
  CachedResults<T>::~CachedResults()
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::!CachedResults()", dbg_verbosity);
#endif

    if (cached_results_) {
      for (typename std::list< DependentResult<T>* >::iterator iter = cached_results_->
           begin();
           iter != cached_results_->end();
           iter++) {
        delete *iter;
      }
      delete cached_results_;
    }
    /*
    while (!cached_results_.empty()) {
      DependentResult<T>* result = cached_results_.back();
      cached_results_.pop_back();
      delete result;
    }
    */
  }

  template <class T>
  void CachedResults<T>::AddCachedResult(const T& result,
                                         const std::vector<const TaggedObject*>& dependents,
                                         const std::vector<Number>& scalar_dependents)
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::AddCachedResult", dbg_verbosity);
#endif

    CleanupInvalidatedResults();

    // insert the new one here
    DependentResult<T>* newResult = new DependentResult<T>(result, dependents, scalar_dependents);
    if (!cached_results_) {
      cached_results_ = new std::list<DependentResult<T>*>;
    }
    cached_results_->push_front(newResult);

    // keep the list small enough
    if (max_cache_size_ >= 0) { // if negative, allow infinite cache
      // non-negative - limit size of list to max_cache_size
      DBG_ASSERT((Int)cached_results_->size()<=max_cache_size_+1);
      if ((Int)cached_results_->size() > max_cache_size_) {
        delete cached_results_->back();
        cached_results_->pop_back();
      }
    }

#ifdef IP_DEBUG_CACHE
    DBG_EXEC(2, DebugPrintCachedResults());
#endif

  }

  template <class T>
  void CachedResults<T>::AddCachedResult(const T& result,
                                         const std::vector<const TaggedObject*>& dependents)
  {
    std::vector<Number> scalar_dependents;
    AddCachedResult(result, dependents, scalar_dependents);
  }

  template <class T>
  bool CachedResults<T>::GetCachedResult(T& retResult, const std::vector<const TaggedObject*>& dependents,
                                         const std::vector<Number>& scalar_dependents) const
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::GetCachedResult", dbg_verbosity);
#endif

    if (!cached_results_)
      return false;

    CleanupInvalidatedResults();

    bool retValue = false;
    typename std::list< DependentResult<T>* >::const_iterator iter;
    for (iter = cached_results_->begin(); iter != cached_results_->end(); iter++) {
      if ((*iter)->DependentsIdentical(dependents, scalar_dependents)) {
        retResult = (*iter)->GetResult();
        retValue = true;
        break;
      }
    }

#ifdef IP_DEBUG_CACHE
    DBG_EXEC(2, DebugPrintCachedResults());
#endif

    return retValue;
  }

  template <class T>
  bool CachedResults<T>::GetCachedResult(
    T& retResult, const std::vector<const TaggedObject*>& dependents) const
  {
    std::vector<Number> scalar_dependents;
    return GetCachedResult(retResult, dependents, scalar_dependents);
  }

  template <class T>
  void CachedResults<T>::AddCachedResult1Dep(const T& result,
      const TaggedObject* dependent1)
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::AddCachedResult1Dep", dbg_verbosity);
#endif

    std::vector<const TaggedObject*> dependents(1);
    dependents[0] = dependent1;

    AddCachedResult(result, dependents);
  }

  template <class T>
  bool CachedResults<T>::GetCachedResult1Dep(T& retResult, const TaggedObject* dependent1)
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::GetCachedResult1Dep", dbg_verbosity);
#endif

    std::vector<const TaggedObject*> dependents(1);
    dependents[0] = dependent1;

    return GetCachedResult(retResult, dependents);
  }

  template <class T>
  void CachedResults<T>::AddCachedResult2Dep(const T& result, const TaggedObject* dependent1,
      const TaggedObject* dependent2)

  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::AddCachedResult2dDep", dbg_verbosity);
#endif

    std::vector<const TaggedObject*> dependents(2);
    dependents[0] = dependent1;
    dependents[1] = dependent2;

    AddCachedResult(result, dependents);
  }

  template <class T>
  bool CachedResults<T>::GetCachedResult2Dep(T& retResult, const TaggedObject* dependent1, const TaggedObject* dependent2)
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::GetCachedResult2Dep", dbg_verbosity);
#endif

    std::vector<const TaggedObject*> dependents(2);
    dependents[0] = dependent1;
    dependents[1] = dependent2;

    return GetCachedResult(retResult, dependents);
  }

  template <class T>
  void CachedResults<T>::AddCachedResult3Dep(const T& result, const TaggedObject* dependent1,
      const TaggedObject* dependent2,
      const TaggedObject* dependent3)

  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::AddCachedResult2dDep", dbg_verbosity);
#endif

    std::vector<const TaggedObject*> dependents(3);
    dependents[0] = dependent1;
    dependents[1] = dependent2;
    dependents[2] = dependent3;

    AddCachedResult(result, dependents);
  }

  template <class T>
  bool CachedResults<T>::GetCachedResult3Dep(T& retResult, const TaggedObject* dependent1,
      const TaggedObject* dependent2,
      const TaggedObject* dependent3)
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::GetCachedResult2Dep", dbg_verbosity);
#endif

    std::vector<const TaggedObject*> dependents(3);
    dependents[0] = dependent1;
    dependents[1] = dependent2;
    dependents[2] = dependent3;

    return GetCachedResult(retResult, dependents);
  }

  template <class T>
  bool CachedResults<T>::InvalidateResult(const std::vector<const TaggedObject*>& dependents,
                                          const std::vector<Number>& scalar_dependents)
  {
    if (!cached_results_)
      return false;

    CleanupInvalidatedResults();

    bool retValue = false;
    typename std::list< DependentResult<T>* >::const_iterator iter;
    for (iter = cached_results_->begin(); iter != cached_results_->end();
         iter++) {
      if ((*iter)->DependentsIdentical(dependents, scalar_dependents)) {
        (*iter)->Invalidate();
        retValue = true;
        break;
      }
    }

    return retValue;
  }

  template <class T>
  void CachedResults<T>::Clear()
  {
    if (!cached_results_)
      return;

    typename std::list< DependentResult<T>* >::const_iterator iter;
    for (iter = cached_results_->begin(); iter != cached_results_->end();
         iter++) {
      (*iter)->Invalidate();
    }

    CleanupInvalidatedResults();
  }

  template <class T>
  void CachedResults<T>::Clear(Int max_cache_size)
  {
    Clear();
    max_cache_size_ = max_cache_size;
  }

  template <class T>
  void CachedResults<T>::CleanupInvalidatedResults() const
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::CleanupInvalidatedResults", dbg_verbosity);
#endif

    if (!cached_results_)
      return;

    typename std::list< DependentResult<T>* >::iterator iter;
    iter = cached_results_->begin();
    while (iter != cached_results_->end()) {
      if ((*iter)->IsStale()) {
        typename std::list< DependentResult<T>* >::iterator
        iter_to_remove = iter;
        iter++;
        DependentResult<T>* result_to_delete = (*iter_to_remove);
        cached_results_->erase(iter_to_remove);
        delete result_to_delete;
      }
      else {
        iter++;
      }
    }
  }

  template <class T>
  void CachedResults<T>::DebugPrintCachedResults() const
  {
#ifdef IP_DEBUG_CACHE
    DBG_START_METH("CachedResults<T>::DebugPrintCachedResults", dbg_verbosity);
    if (DBG_VERBOSITY()>=2 ) {
      if (!cached_results_) {
        DBG_PRINT((2,"Currentlt no cached results:\n"));
      }
      else {
        typename std::list< DependentResult<T>* >::const_iterator iter;
        DBG_PRINT((2,"Current set of cached results:\n"));
        for (iter = cached_results_->begin(); iter != cached_results_->end(); iter++) {
          DBG_PRINT((2,"  DependentResult:0x%x\n", (*iter)));
        }
      }
    }
#endif

  }

} // namespace Ipopt

#endif
