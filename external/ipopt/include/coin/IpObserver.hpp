// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpObserver.hpp 2161 2013-01-01 20:39:05Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPOBSERVER_HPP__
#define __IPOBSERVER_HPP__

#include "IpUtils.hpp"
#include <vector>
#include <algorithm>

//#define IP_DEBUG_OBSERVER
#if COIN_IPOPT_CHECKLEVEL > 2
# define IP_DEBUG_OBSERVER
#endif
#ifdef IP_DEBUG_OBSERVER
# include "IpDebug.hpp"
#endif

namespace Ipopt
{
  /** Forward declarations */
  class Subject;

  /** Slight Variation of the Observer Design Pattern.
   *  This class implements the Observer class of the
   *  Observer Design Pattern. An Observer "Attach"es
   *  to a Subject, indicating that it would like to
   *  be notified of changes in the Subject.
   *  Any derived class wishing to recieve notifications
   *  from a Subject should inherit off of 
   *  Observer and overload the protected method,
   *  RecieveNotification_(...).
   */
  class Observer
  {
  public:
#ifdef IP_DEBUG_OBSERVER

    static const Index dbg_verbosity;
#endif

    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor */
    Observer()
    {}

    /** Default destructor */
    inline
    virtual ~Observer();
    //@}

    /** Enumeration specifying the type of notification */
    enum NotifyType
    {
      NT_All,
      NT_BeingDestroyed,
      NT_Changed
    };

  protected:
    /** Derived classes should call this method
     * to request an "Attach" to a Subject. Do 
     * not call "Attach" explicitly on the Subject
     * since further processing is done here
     */
    inline
    void RequestAttach(NotifyType notify_type, const Subject* subject);

    /** Derived classes should call this method
     * to request a "Detach" to a Subject. Do 
     * not call "Detach" explicitly on the Subject
     * since further processing is done here
     */
    inline
    void RequestDetach(NotifyType notify_type, const Subject* subject);

    /** Derived classes should overload this method to
     * recieve the requested notification from 
     * attached Subjects
     */
    virtual void RecieveNotification(NotifyType notify_type, const Subject* subject)=0;

  private:
    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Copy Constructor */
    Observer(const Observer&);

    /** Overloaded Equals Operator */
    void operator=(const Observer&);
    //@}

    /** A list of the subjects currently being
     *  observed. */
    std::vector<const Subject*> subjects_;

    /** Private Method for Recieving Notification
     *  should only be called by the friend class
     *  Subject. This method will, in turn, call
     *  the overloaded RecieveNotification method
     *  for the derived class to process.
     */
    inline
    void ProcessNotification(NotifyType notify_type, const Subject* subject);

    friend class Subject;
  };

  /** Slight Variation of the Observer Design Pattern (Subject part).
   *  This class implements the Subject class of the Observer Design
   *  Pattern. An Observer "Attach"es to a Subject, indicating that it
   *  would like to be notified of changes in the Subject.  Any
   *  derived class that is to be observed has to inherit off the
   *  Subject base class.  If the subject needs to notify the
   *  Observer, it calls the Notify method.
   */
  class Subject
  {
  public:
#ifdef IP_DEBUG_OBSERVER

    static const Index dbg_verbosity;
#endif

    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor */
    Subject()
    {}

    /** Default destructor */
    inline
    virtual ~Subject();
    //@}

    /**@name Methods to Add and Remove Observers.
     *  Currently, the notify_type flags are not used,
     *  and Observers are attached in general and will
     *  recieve all notifications (of the type requested
     *  and possibly of types not requested). It is 
     *  up to the observer to ignore the types they
     *  are not interested in. The NotifyType in the
     *  parameter list is so a more efficient mechanism
     *  depending on type could be implemented later if
     *  necessary.*/
    //@{

    /** Attach the specified observer
     *  (i.e., begin recieving notifications). */
    inline
    void AttachObserver(Observer::NotifyType notify_type, Observer* observer) const;

    /** Detach the specified observer
     *  (i.e., no longer recieve notifications). */
    inline
    void DetachObserver(Observer::NotifyType notify_type, Observer* observer) const;
    //@}

  protected:

    inline
    void Notify(Observer::NotifyType notify_type) const;

  private:
    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Copy Constructor */
    Subject(const Subject&);

    /** Overloaded Equals Operator */
    void operator=(const Subject&);
    //@}

    mutable std::vector<Observer*> observers_;

  };

  /* inline methods */
  inline
  Observer::~Observer()
  {
#ifdef IP_DEBUG_OBSERVER
    DBG_START_METH("Observer::~Observer", dbg_verbosity);
    if (DBG_VERBOSITY()>=1) {
      for (Index i=0; i<(Index)subjects_.size(); i++) {
        DBG_PRINT((1,"subjects_[%d] = 0x%x\n", i, subjects_[i]));
      }
    }
#endif
    // Detach all subjects
    for (Int i=(Int)(subjects_.size()-1); i>=0; i--) {
#ifdef IP_DEBUG_OBSERVER
      DBG_PRINT((1,"About to detach subjects_[%d] = 0x%x\n", i, subjects_[i]));
#endif

      RequestDetach(NT_All, subjects_[i]);
    }
  }

  inline
  void Observer::RequestAttach(NotifyType notify_type, const Subject* subject)
  {
#ifdef IP_DEBUG_OBSERVER
    DBG_START_METH("Observer::RequestAttach", dbg_verbosity);

    // Add the subject to the list if it does not already exist
    std::vector<const Subject*>::iterator attached_subject;
    attached_subject = std::find(subjects_.begin(), subjects_.end(), subject);
    DBG_ASSERT(attached_subject == subjects_.end());
    DBG_ASSERT(subject);
#endif

    // add the subject to the list
    subjects_.push_back(subject);
    // Attach the observer to the subject
    subject->AttachObserver(notify_type, this);
  }

  inline
  void Observer::RequestDetach(NotifyType notify_type, const Subject* subject)
  {
#ifdef IP_DEBUG_OBSERVER
    DBG_START_METH("Observer::RequestDetach", dbg_verbosity);
    DBG_PRINT((1, "Requesting detach of subject: 0x%x\n", subject));
    DBG_ASSERT(subject);
#endif

    if (subject) {
      std::vector<const Subject*>::iterator attached_subject;
      attached_subject = std::find(subjects_.begin(), subjects_.end(), subject);
#ifdef IP_DEBUG_OBSERVER

      DBG_ASSERT(attached_subject != subjects_.end());
#endif

      if (attached_subject != subjects_.end()) {
#ifdef IP_DEBUG_OBSERVER
        DBG_PRINT((1, "Removing subject: 0x%x from the list\n", subject));
#endif

        subjects_.erase(attached_subject);
      }

      // Detach the observer from the subject
      subject->DetachObserver(notify_type, this);
    }
  }

  inline
  void Observer::ProcessNotification(NotifyType notify_type, const Subject* subject)
  {
#ifdef IP_DEBUG_OBSERVER
    DBG_START_METH("Observer::ProcessNotification", dbg_verbosity);
    DBG_ASSERT(subject);
#endif

    if (subject) {
      std::vector<const Subject*>::iterator attached_subject;
      attached_subject = std::find(subjects_.begin(), subjects_.end(), subject);

      // We must be processing a notification for a
      // subject that was previously attached.
#ifdef IP_DEBUG_OBSERVER

      DBG_ASSERT(attached_subject != subjects_.end());
#endif

      this->RecieveNotification(notify_type, subject);

      if (notify_type == NT_BeingDestroyed) {
        // the subject is going away, remove it from our list
        subjects_.erase(attached_subject);
      }
    }
  }

  inline
  Subject::~Subject()
  {
#ifdef IP_DEBUG_OBSERVER
    DBG_START_METH("Subject::~Subject", dbg_verbosity);
#endif

    std::vector<Observer*>::iterator iter;
    for (iter = observers_.begin(); iter != observers_.end(); iter++) {
      (*iter)->ProcessNotification(Observer::NT_BeingDestroyed, this);
    }
  }

  inline
  void Subject::AttachObserver(Observer::NotifyType notify_type, Observer* observer) const
  {
#ifdef IP_DEBUG_OBSERVER
    DBG_START_METH("Subject::AttachObserver", dbg_verbosity);
    // current implementation notifies all observers of everything
    // they must filter the notifications that they are not interested
    // in (i.e. a hub, not a router)
    DBG_ASSERT(observer);

    std::vector<Observer*>::iterator attached_observer;
    attached_observer = std::find(observers_.begin(), observers_.end(), observer);
    DBG_ASSERT(attached_observer == observers_.end());

    DBG_ASSERT(observer);
#endif

    observers_.push_back(observer);
  }

  inline
  void Subject::DetachObserver(Observer::NotifyType notify_type, Observer* observer) const
  {
#ifdef IP_DEBUG_OBSERVER
    DBG_START_METH("Subject::DetachObserver", dbg_verbosity);
    DBG_ASSERT(observer);
#endif

    if (observer) {
      std::vector<Observer*>::iterator attached_observer;
      attached_observer = std::find(observers_.begin(), observers_.end(), observer);
#ifdef IP_DEBUG_OBSERVER

      DBG_ASSERT(attached_observer != observers_.end());
#endif

      if (attached_observer != observers_.end()) {
        observers_.erase(attached_observer);
      }
    }
  }

  inline
  void Subject::Notify(Observer::NotifyType notify_type) const
  {
#ifdef IP_DEBUG_OBSERVER
    DBG_START_METH("Subject::Notify", dbg_verbosity);
#endif

    std::vector<Observer*>::iterator iter;
    for (iter = observers_.begin(); iter != observers_.end(); iter++) {
      (*iter)->ProcessNotification(notify_type, this);
    }
  }


} // namespace Ipopt

#endif
