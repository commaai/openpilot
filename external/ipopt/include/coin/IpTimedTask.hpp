// Copyright (C) 2006, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpTimedTask.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Andreas Waechter               IBM    2005-09-19

#ifndef __IPTIMEDTASK_HPP__
#define __IPTIMEDTASK_HPP__

#include "IpUtils.hpp"

namespace Ipopt
{
  /** This class is used to collect timing information for a
   *  particular task. */
  class TimedTask
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default constructor. */
    TimedTask()
        :
        total_cputime_(0.),
        total_systime_(0.),
        total_walltime_(0.),
        start_called_(false),
        end_called_(true)
    {}

    /** Default destructor */
    ~TimedTask()
    {}
    //@}

    /** Method for resetting time to zero. */
    void Reset()
    {
      total_cputime_ = 0.;
      total_systime_ = 0.;
      total_walltime_ = 0.;
      start_called_ = false;
      end_called_ = true;
    }

    /** Method that is called before execution of the task. */
    void Start()
    {
      DBG_ASSERT(end_called_);
      DBG_ASSERT(!start_called_);
      end_called_ = false;
      start_called_ = true;
      start_cputime_ = CpuTime();
      start_systime_ = SysTime();
      start_walltime_ = WallclockTime();
    }

    /** Method that is called after execution of the task. */
    void End()
    {
      DBG_ASSERT(!end_called_);
      DBG_ASSERT(start_called_);
      end_called_ = true;
      start_called_ = false;
      total_cputime_ += CpuTime() - start_cputime_;
      total_systime_ += SysTime() - start_systime_;
      total_walltime_ += WallclockTime() - start_walltime_;
    }

    /** Method that is called after execution of the task for which
     *  timing might have been started.  This only updates the timing
     *  if the timing has indeed been conducted. This is useful to
     *  stop timing after catching exceptions. */
    void EndIfStarted()
    {
      if (start_called_) {
        end_called_ = true;
        start_called_ = false;
        total_cputime_ += CpuTime() - start_cputime_;
        total_systime_ += SysTime() - start_systime_;
        total_walltime_ += WallclockTime() - start_walltime_;
      }
      DBG_ASSERT(end_called_);
    }

    /** Method returning total CPU time spend for task so far. */
    Number TotalCpuTime() const
    {
      DBG_ASSERT(end_called_);
      return total_cputime_;
    }

    /** Method returning total system time spend for task so far. */
    Number TotalSysTime() const
    {
      DBG_ASSERT(end_called_);
      return total_systime_;
    }

    /** Method returning total wall clock time spend for task so far. */
    Number TotalWallclockTime() const
    {
      DBG_ASSERT(end_called_);
      return total_walltime_;
    }

  private:
    /**@name Default Compiler Generated Methods (Hidden to avoid
     * implicit creation/calling).  These methods are not
     * implemented and we do not want the compiler to implement them
     * for us, so we declare them private and do not define
     * them. This ensures that they will not be implicitly
     * created/called. */
    //@{
    /** Copy Constructor */
    TimedTask(const TimedTask&);

    /** Overloaded Equals Operator */
    void operator=(const TimedTask&);
    //@}

    /** CPU time at beginning of task. */
    Number start_cputime_;
    /** Total CPU time for task measured so far. */
    Number total_cputime_;
    /** System time at beginning of task. */
    Number start_systime_;
    /** Total system time for task measured so far. */
    Number total_systime_;
    /** Wall clock time at beginning of task. */
    Number start_walltime_;
    /** Total wall clock time for task measured so far. */
    Number total_walltime_;

    /** @name fields for debugging */
    //@{
    bool start_called_;
    bool end_called_;
    //@}

  };
} // namespace Ipopt

#endif
