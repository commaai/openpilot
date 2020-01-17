// Copyright (C) 2005, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpHessianUpdater.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Andreas Waechter            IBM    2005-12-26

#ifndef __IPHESSIANUPDATER_HPP__
#define __IPHESSIANUPDATER_HPP__

#include "IpAlgStrategy.hpp"

namespace Ipopt
{

  /** Abstract base class for objects responsible for updating the
   *  Hessian information.  This can be done using exact second
   *  derivatives from the NLP, or by a quasi-Newton Option.  The
   *  result is put into the W field in IpData.
   */
  class HessianUpdater : public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor */
    HessianUpdater()
    {}

    /** Default destructor */
    virtual ~HessianUpdater()
    {}
    //@}

    /** overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** Update the Hessian based on the current information in IpData,
     *  and possibly on information from previous calls.
     */
    virtual void UpdateHessian() = 0;

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
    HessianUpdater(const HessianUpdater&);

    /** Overloaded Equals Operator */
    void operator=(const HessianUpdater&);
    //@}

  };

} // namespace Ipopt

#endif
