// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpAlgStrategy.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPALGSTRATEGY_HPP__
#define __IPALGSTRATEGY_HPP__

#include "IpOptionsList.hpp"
#include "IpJournalist.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptNLP.hpp"
#include "IpIpoptData.hpp"

namespace Ipopt
{

  /** This is the base class for all algorithm strategy objects.  The
   *  AlgorithmStrategyObject base class implements a common interface
   *  for all algorithm strategy objects.  A strategy object is a
   *  component of the algorithm for which different alternatives or
   *  implementations exists.  It allows to compose the algorithm
   *  before execution for a particular configuration, without the
   *  need to call alternatives based on enums. For example, the
   *  LineSearch object is a strategy object, since different line
   *  search options might be used for different runs.
   *
   *  This interface is used for
   *  things that are done to all strategy objects, like
   *  initialization and setting options.
   */
  class AlgorithmStrategyObject : public ReferencedObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor */
    AlgorithmStrategyObject()
        :
        initialize_called_(false)
    {}

    /** Default Destructor */
    virtual ~AlgorithmStrategyObject()
    {}
    //@}

    /** This method is called every time the algorithm starts again -
     *  it is used to reset any internal state.  The pointers to the
     *  Journalist, as well as to the IpoptNLP, IpoptData, and
     *  IpoptCalculatedQuantities objects should be stored in the
     *  instanciation of this base class.  This method is also used to
     *  get all required user options from the OptionsList.  Here, if
     *  prefix is given, each tag (identifying the options) is first
     *  looked for with the prefix in front, and if not found, without
     *  the prefix.  Note: you should not cue off of the iteration
     *  count to indicate the "start" of an algorithm!
     *
     *  Do not overload this method, since it does some general
     *  initialization that is common for all strategy objects.
     *  Overload the protected InitializeImpl method instead.
     */
    bool Initialize(const Journalist& jnlst,
                    IpoptNLP& ip_nlp,
                    IpoptData& ip_data,
                    IpoptCalculatedQuantities& ip_cq,
                    const OptionsList& options,
                    const std::string& prefix)
    {
      initialize_called_ = true;
      // Copy the pointers for the problem defining objects
      jnlst_ = &jnlst;
      ip_nlp_ = &ip_nlp;
      ip_data_ = &ip_data;
      ip_cq_ = &ip_cq;

      bool retval = InitializeImpl(options, prefix);
      if (!retval) {
        initialize_called_ = false;
      }

      return retval;
    }

    /** Reduced version of the Initialize method, which does not
     *  require special Ipopt information.  This is useful for
     *  algorithm objects that could be used outside Ipopt, such as
     *  linear solvers. */
    bool ReducedInitialize(const Journalist& jnlst,
                           const OptionsList& options,
                           const std::string& prefix)
    {
      initialize_called_ = true;
      // Copy the pointers for the problem defining objects
      jnlst_ = &jnlst;
      ip_nlp_ = NULL;
      ip_data_ = NULL;
      ip_cq_ = NULL;

      bool retval = InitializeImpl(options, prefix);
      if (!retval) {
        initialize_called_ = false;
      }

      return retval;
    }

  protected:
    /** Implementation of the initialization method that has to be
     *  overloaded by for each derived class. */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix)=0;

    /** @name Accessor methods for the problem defining objects.
     *  Those should be used by the derived classes. */
    //@{
    const Journalist& Jnlst() const
    {
      DBG_ASSERT(initialize_called_);
      return *jnlst_;
    }
    IpoptNLP& IpNLP() const
    {
      DBG_ASSERT(initialize_called_);
      DBG_ASSERT(IsValid(ip_nlp_));
      return *ip_nlp_;
    }
    IpoptData& IpData() const
    {
      DBG_ASSERT(initialize_called_);
      DBG_ASSERT(IsValid(ip_data_));
      return *ip_data_;
    }
    IpoptCalculatedQuantities& IpCq() const
    {
      DBG_ASSERT(initialize_called_);
      DBG_ASSERT(IsValid(ip_cq_));
      return *ip_cq_;
    }
    bool HaveIpData() const
    {
      return IsValid(ip_data_);
    }
    //@}

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
    //AlgorithmStrategyObject();


    /** Copy Constructor */
    AlgorithmStrategyObject(const AlgorithmStrategyObject&);

    /** Overloaded Equals Operator */
    void operator=(const AlgorithmStrategyObject&);
    //@}

    /** @name Pointers to objects defining a particular optimization
     *  problem */
    //@{
    SmartPtr<const Journalist> jnlst_;
    SmartPtr<IpoptNLP> ip_nlp_;
    SmartPtr<IpoptData> ip_data_;
    SmartPtr<IpoptCalculatedQuantities> ip_cq_;
    //@}

    /** flag indicating if Initialize method has been called (for
     *  debugging) */
    bool initialize_called_;
  };

} // namespace Ipopt

#endif
