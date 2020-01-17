// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpEqMultCalculator.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Carl Laird, Andreas Waechter              IBM    2004-09-23

#ifndef __IPEQMULTCALCULATOR_HPP__
#define __IPEQMULTCALCULATOR_HPP__

#include "IpUtils.hpp"
#include "IpAlgStrategy.hpp"

namespace Ipopt
{
  /** Base Class for objects that compute estimates for the equality
   *  constraint multipliers y_c and y_d.  For example, this is the
   *  base class for objects for computing least square multipliers or
   *  coordinate multipliers. */
  class EqMultiplierCalculator: public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor. */
    EqMultiplierCalculator()
    {}
    /** Default destructor */
    virtual ~EqMultiplierCalculator()
    {}
    //@}

    /** overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** This method computes the estimates for y_c and y_d at the
     *  current point.  If the estimates cannot be computed (e.g. some
     *  linear system is singular), the return value of this method is
     *  false. */
    virtual bool CalculateMultipliers(Vector& y_c,
                                      Vector& y_d) = 0;

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
    EqMultiplierCalculator(const EqMultiplierCalculator&);

    /** Overloaded Equals Operator */
    void operator=(const EqMultiplierCalculator&);
    //@}
  };

} // namespace Ipopt

#endif
