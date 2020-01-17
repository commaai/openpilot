// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpLineSearch.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPLINESEARCH_HPP__
#define __IPLINESEARCH_HPP__

#include "IpAlgStrategy.hpp"
#include "IpIpoptCalculatedQuantities.hpp"

namespace Ipopt
{

  /** Base class for line search objects.
   */
  class LineSearch : public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor */
    LineSearch()
    {}

    /** Default destructor */
    virtual ~LineSearch()
    {}
    //@}

    /** Perform the line search.  As search direction the delta
     *  in the data object is used
     */
    virtual void FindAcceptableTrialPoint() = 0;

    /** Reset the line search.
     *  This function should be called if all previous information
     *  should be discarded when the line search is performed the
     *  next time.  For example, this method should be called after
     *  the barrier parameter is changed.
     */
    virtual void Reset() = 0;

    /** Set flag indicating whether a very rigorous line search should
     *  be performed.  If this flag is set to true, the line search
     *  algorithm might decide to abort the line search and not to
     *  accept a new iterate.  If the line search decided not to
     *  accept a new iterate, the return value of
     *  CheckSkippedLineSearch() is true at the next call.  For
     *  example, in the non-monotone barrier parameter update
     *  procedure, the filter algorithm should not switch to the
     *  restoration phase in the free mode; instead, the algorithm
     *  should swtich to the fixed mode.
     */
    virtual void SetRigorousLineSearch(bool rigorous) = 0;

    /** Check if the line search procedure didn't accept a new iterate
     *  during the last call of FindAcceptableTrialPoint().
     *  
     */
    virtual bool CheckSkippedLineSearch() = 0;

    /** This method should be called if the optimization process
     *  requires the line search object to switch to some fallback
     *  mechanism (like the restoration phase), when the regular
     *  optimization procedure cannot be continued (for example,
     *  because the search direction could not be computed).  This
     *  will cause the line search object to immediately proceed with
     *  this mechanism when FindAcceptableTrialPoint() is call.  This
     *  method returns false if no fallback mechanism is available. */
    virtual bool ActivateFallbackMechanism() = 0;

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
    LineSearch(const LineSearch&);

    /** Overloaded Equals Operator */
    void operator=(const LineSearch&);
    //@}

  };

} // namespace Ipopt

#endif
