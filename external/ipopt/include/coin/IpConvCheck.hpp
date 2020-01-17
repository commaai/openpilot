// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpConvCheck.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPCONVCHECK_HPP__
#define __IPCONVCHECK_HPP__

#include "IpAlgStrategy.hpp"

namespace Ipopt
{

  /** Base class for checking the algorithm
   *  termination criteria.
   */
  class ConvergenceCheck : public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Constructor */
    ConvergenceCheck()
    {}

    /** Default destructor */
    virtual ~ConvergenceCheck()
    {}
    //@}

    /** Convergence return enum */
    enum ConvergenceStatus {
      CONTINUE,
      CONVERGED,
      CONVERGED_TO_ACCEPTABLE_POINT,
      MAXITER_EXCEEDED,
      CPUTIME_EXCEEDED,
      DIVERGING,
      USER_STOP,
      FAILED
    };

    /** overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** Pure virtual method for performing the convergence test.  If
     *  call_intermediate_callback is true, the user callback method
     *  in the NLP should be called in order to see if the user
     *  requests an early termination. */
    virtual ConvergenceStatus
    CheckConvergence(bool call_intermediate_callback = true) = 0;

    /** Method for testing if the current iterate is considered to
     *  satisfy the "accptable level" of accuracy.  The idea is that
     *  if the desired convergence tolerance cannot be achieved, the
     *  algorithm might stop after a number of acceptable points have
     *  been encountered. */
    virtual bool CurrentIsAcceptable()=0;

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
    //    ConvergenceCheck();

    /** Copy Constructor */
    ConvergenceCheck(const ConvergenceCheck&);

    /** Overloaded Equals Operator */
    void operator=(const ConvergenceCheck&);
    //@}

  };

} // namespace Ipopt

#endif
