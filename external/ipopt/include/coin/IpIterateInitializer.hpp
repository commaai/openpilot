// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpIterateInitializer.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Carl Laird, Andreas Waechter              IBM    2004-09-24

#ifndef __IPITERATEINITIALIZER_HPP__
#define __IPITERATEINITIALIZER_HPP__

#include "IpAlgStrategy.hpp"
#include "IpIpoptNLP.hpp"
#include "IpIpoptData.hpp"
#include "IpIpoptCalculatedQuantities.hpp"

namespace Ipopt
{

  /** Base class for all methods for initializing the iterates.
   */
  class IterateInitializer: public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor */
    IterateInitializer()
    {}

    /** Default destructor */
    virtual ~IterateInitializer()
    {}
    //@}

    /** overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** Compute the initial iterates and set the into the curr field
     *  of the ip_data object. */
    virtual bool SetInitialIterates() = 0;

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
    IterateInitializer(const IterateInitializer&);

    /** Overloaded Equals Operator */
    void operator=(const IterateInitializer&);
    //@}

  };

} // namespace Ipopt

#endif
