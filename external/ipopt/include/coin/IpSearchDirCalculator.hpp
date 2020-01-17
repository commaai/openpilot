// Copyright (C) 2005, 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpSearchDirCalculator.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Andreas Waechter            IBM    2005-10-13

#ifndef __IPSEARCHDIRCALCULATOR_HPP__
#define __IPSEARCHDIRCALCULATOR_HPP__

#include "IpAlgStrategy.hpp"

namespace Ipopt
{

  /** Base class for computing the search direction for the line
   *  search.
   */
  class SearchDirectionCalculator : public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Constructor */
    SearchDirectionCalculator()
    {}

    /** Default destructor */
    virtual ~SearchDirectionCalculator()
    {}
    //@}

    /** overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** Pure virtual method for computing the search direction. The
     *  computed direction is stored in IpData().delta().*/
    virtual bool ComputeSearchDirection()=0;

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
    //    SearchDirectionCalculator();

    /** Copy Constructor */
    SearchDirectionCalculator(const SearchDirectionCalculator&);

    /** Overloaded Equals Operator */
    void operator=(const SearchDirectionCalculator&);
    //@}

  };

} // namespace Ipopt

#endif
