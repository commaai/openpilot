// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpMuUpdate.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPMUUPDATE_HPP__
#define __IPMUUPDATE_HPP__

#include "IpAlgStrategy.hpp"

namespace Ipopt
{
  /** Abstract Base Class for classes that implement methods for computing
   *  the barrier and fraction-to-the-boundary rule parameter for the
   *  current iteration.
   */
  class MuUpdate : public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor */
    MuUpdate()
    {}

    /** Default destructor */
    virtual ~MuUpdate()
    {}
    //@}

    /** Initialize method - overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** Method for determining the barrier parameter for the next
     *  iteration.  A LineSearch object is passed, so that this method
     *  can call the Reset method in the LineSearch object, for
     *  example when then barrier parameter is changed. This method is
     *  also responsible for setting the fraction-to-the-boundary
     *  parameter tau.  This method returns false if the update could
     *  not be performed and the algorithm should revert to an
     *  emergency fallback mechanism. */
    virtual bool UpdateBarrierParameter() = 0;

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
    MuUpdate(const MuUpdate&);

    /** Overloaded Equals Operator */
    void operator=(const MuUpdate&);
    //@}

  };

} // namespace Ipopt

#endif
