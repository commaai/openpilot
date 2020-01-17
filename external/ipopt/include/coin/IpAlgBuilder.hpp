// Copyright (C) 2004, 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpAlgBuilder.hpp 2666 2016-07-20 16:02:55Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-09-29

#ifndef __IPALGBUILDER_HPP__
#define __IPALGBUILDER_HPP__

#include "IpIpoptAlg.hpp"
#include "IpReferenced.hpp"
#include "IpAugSystemSolver.hpp"
#include "IpPDSystemSolver.hpp"

namespace Ipopt
{

  // forward declarations
  class IterationOutput;
  class HessianUpdater;
  class ConvergenceCheck;
  class SearchDirectionCalculator;
  class EqMultiplierCalculator;
  class IterateInitializer;
  class LineSearch;
  class MuUpdate;

  /** Builder for creating a complete IpoptAlg object.  This object
   *  contains all subelements (such as line search objects etc).  How
   *  the resulting IpoptAlg object is built can be influenced by the
   *  options.
   *
   *  More advanced customization can be achieved by subclassing this
   *  class and overloading the virtual methods that build the
   *  individual parts. The advantage of doing this is that it allows
   *  one to reuse the extensive amount of options processing that
   *  takes place, for instance, when generating the symmetric linear
   *  system solver. Another method for customizing the algorithm is
   *  using the optional custom_solver argument, which allows the
   *  expert user to provide a specialized linear solver for the
   *  augmented system (e.g., type GenAugSystemSolver), possibly for
   *  user-defined matrix objects. The optional custom_solver constructor
   *  argument is likely obsolete, however, as more control over this
   *  this process can be achieved by implementing a subclass of this
   *  AlgBuilder (e.g., by overloading the AugSystemSolverFactory method).
   */
  class AlgorithmBuilder : public ReferencedObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Constructor */
    AlgorithmBuilder(SmartPtr<AugSystemSolver> custom_solver=NULL);

    /** Destructor */
    virtual ~AlgorithmBuilder()
    {}

    //@}

    /** Methods for IpoptTypeInfo */
    //@{
    /** register the options used by the algorithm builder */
    static void RegisterOptions(SmartPtr<RegisteredOptions> roptions);
    //@}

    /** @name Convenience methods for building solvers without having
     *  to duplicate the significant amount of preprocessor flag and
     *  option checking that takes place. These solvers are used to
     *  create a number of core algorithm components across the
     *  different Build* methods, but depending on what options are
     *  chosen, the first method requiring the solver to be used can
     *  vary. Therefore, each of the Factory methods below is paired
     *  with a Getter method, which is called by all parts of this
     *  algorithm builder to ensure the Factory is only called once. */
    //@{

    /** Create a solver that can be used to solve a symmetric linear
     *  system.
     *  Dependencies: None
     */
    virtual SmartPtr<SymLinearSolver>
        SymLinearSolverFactory(const Journalist& jnlst,
                               const OptionsList& options,
                               const std::string& prefix);

    /** Get the symmetric linear system solver for this
     *  algorithm. This method will call the SymLinearSolverFactory
     *  exactly once (the first time it is used), and store its
     *  instance on SymSolver_ for use in subsequent calls.
     */
    SmartPtr<SymLinearSolver> GetSymLinearSolver(const Journalist& jnlst,
                                                 const OptionsList& options,
                                                 const std::string& prefix);

    /** Create a solver that can be used to solve an
     *  augmented system.
     *  Dependencies:
     *     -> GetSymLinearSolver()
     *         -> SymLinearSolverFactory()
     *     -> custom_solver_
     */
    virtual SmartPtr<AugSystemSolver>
        AugSystemSolverFactory(const Journalist& jnlst,
                               const OptionsList& options,
                               const std::string& prefix);

    /** Get the augmented system solver for this algorithm. This
     *  method will call the AugSystemSolverFactory exactly once (the
     *  first time it is used), and store its instance on AugSolver_
     *  for use in subsequent calls.
     */
    SmartPtr<AugSystemSolver> GetAugSystemSolver(const Journalist& jnlst,
                                                 const OptionsList& options,
                                                 const std::string& prefix);

    /** Create a solver that can be used to solve a
     *  primal-dual system.
     *  Dependencies:
     *     -> GetAugSystemSolver()
     *         -> AugSystemSolverFactory()
     *             -> GetSymLinearSolver()
     *                 -> SymLinearSolverFactory()
     *             -> custom_solver_
     */
    virtual SmartPtr<PDSystemSolver>
        PDSystemSolverFactory(const Journalist& jnlst,
                              const OptionsList& options,
                              const std::string& prefix);

    /** Get the primal-dual system solver for this algorithm. This
     *  method will call the PDSystemSolverFactory exactly once (the
     *  first time it is used), and store its instance on PDSolver_
     *  for use in subsequent calls.
     */
    SmartPtr<PDSystemSolver> GetPDSystemSolver(const Journalist& jnlst,
                                               const OptionsList& options,
                                               const std::string& prefix);
    //@}

    /** @name Methods to build parts of the algorithm */
    //@{
    /** Allocates memory for the IpoptNLP, IpoptData, and
     *  IpoptCalculatedQuanties arguments.
     *  Dependencies: None
     */
    virtual void BuildIpoptObjects(const Journalist& jnlst,
                                   const OptionsList& options,
                                   const std::string& prefix,
                                   const SmartPtr<NLP>& nlp,
                                   SmartPtr<IpoptNLP>& ip_nlp,
                                   SmartPtr<IpoptData>& ip_data,
                                   SmartPtr<IpoptCalculatedQuantities>& ip_cq);

    /** Creates an instance of the IpoptAlgorithm class by building
     *  each of its required constructor arguments piece-by-piece. The
     *  default algorithm can be customized by overloading this method
     *  or by overloading one or more of the Build* methods called in
     *  this method's default implementation. Additional control can
     *  be achieved by overloading any of the *SolverFactory methods.
     *  This method will call (in this order):
     *     -> BuildIterationOutput()
     *     -> BuildHessianUpdater()
     *     -> BuildConvergenceCheck()
     *     -> BuildSearchDirectionCalculator()
     *     -> BuildEqMultiplierCalculator()
     *     -> BuildIterateInitializer()
     *     -> BuildLineSearch()
     *     -> BuildMuUpdate()
     */
    virtual SmartPtr<IpoptAlgorithm> BuildBasicAlgorithm(const Journalist& jnlst,
        const OptionsList& options,
        const std::string& prefix);

    /** Creates an instance of the IterationOutput class. This method
     *  is called in the default implementation of
     *  BuildBasicAlgorithm. It can be overloaded to customize that
     *  portion the default algorithm.
     *  Dependencies: None
     */
    virtual SmartPtr<IterationOutput>
        BuildIterationOutput(const Journalist& jnlst,
                             const OptionsList& options,
                             const std::string& prefix);

    /** Creates an instance of the HessianUpdater class. This method
     *  is called in the default implementation of
     *  BuildBasicAlgorithm.  It can be overloaded to customize that
     *  portion the default algorithm.
     *  Dependencies: None
     */
    virtual SmartPtr<HessianUpdater>
        BuildHessianUpdater(const Journalist& jnlst,
                            const OptionsList& options,
                            const std::string& prefix);

    /** Creates an instance of the ConvergenceCheck class. This method
     *  is called in the default implementation of
     *  BuildBasicAlgorithm.  It can be overloaded to customize that
     *  portion the default algorithm.
     *  Dependencies: None
     */
    virtual SmartPtr<ConvergenceCheck>
        BuildConvergenceCheck(const Journalist& jnlst,
                              const OptionsList& options,
                              const std::string& prefix);

    /** Creates an instance of the SearchDirectionCalculator
     *  class. This method is called in the default implementation of
     *  BuildBasicAlgorithm.  It can be overloaded to customize that
     *  portion the default algorithm.
     *  Dependencies:
     *     -> GetPDSystemSolver()
     *         -> PDSystemSolverFactory()
     *             -> GetAugSystemSolver()
     *                 -> AugSystemSolverFactory()
     *                     -> GetSymLinearSolver()
     *                         -> SymLinearSolverFactory()
     *                     -> custom_solver_
     */
     virtual SmartPtr<SearchDirectionCalculator>
        BuildSearchDirectionCalculator(const Journalist& jnlst,
                                       const OptionsList& options,
                                       const std::string& prefix);

    /** Creates an instance of the EqMultiplierCalculator class. This
     *  method is called in the default implementation of
     *  BuildBasicAlgorithm.  It can be overloaded to customize that
     *  portion the default algorithm.
     *  Dependencies:
     *     -> GetAugSystemSolver()
     *         -> AugSystemSolverFactory()
     *             -> GetSymLinearSolver()
     *                 -> SymLinearSolverFactory()
     *             -> custom_solver_
     */
    virtual SmartPtr<EqMultiplierCalculator>
        BuildEqMultiplierCalculator(const Journalist& jnlst,
                                    const OptionsList& options,
                                    const std::string& prefix);

    /** Creates an instance of the IterateInitializer class. This
     *  method is called in the default implementation of
     *  BuildBasicAlgorithm.  It can be overloaded to customize that
     *  portion the default algorithm.
     *  Dependencies:
     *     -> EqMultCalculator_
     *     -> GetAugSystemSolver()
     *         -> AugSystemSolverFactory()
     *             -> GetSymLinearSolver()
     *                 -> SymLinearSolverFactory()
     *             -> custom_solver_
     */
    virtual SmartPtr<IterateInitializer>
        BuildIterateInitializer(const Journalist& jnlst,
                                const OptionsList& options,
                                const std::string& prefix);

    /** Creates an instance of the LineSearch class. This method is
     *  called in the default implementation of BuildBasicAlgorithm.
     *  It can be overloaded to customize that portion the default
     *  algorithm.
     *  Dependencies:
     *     -> EqMultCalculator_
     *     -> ConvCheck_
     *     -> GetAugSystemSolver()
     *         -> AugSystemSolverFactory()
     *             -> GetSymLinearSolver()
     *                 -> SymLinearSolverFactory()
     *             -> custom_solver_
     *     -> GetPDSystemSolver()
     *         -> PDSystemSolverFactory()
     *             -> GetAugSystemSolver()
     *                 -> AugSystemSolverFactory()
     *                     -> GetSymLinearSolver()
     *                         -> SymLinearSolverFactory()
     *                     -> custom_solver_
     */
    virtual SmartPtr<LineSearch> BuildLineSearch(const Journalist& jnlst,
                                                 const OptionsList& options,
                                                 const std::string& prefix);

    /** Creates an instance of the MuUpdate class. This method is
     *  called in the default implementation of BuildBasicAlgorithm.
     *  It can be overloaded to customize that portion the default
     *  algorithm.
     *  Dependencies:
     *     -> LineSearch_
     *         -> EqMultCalculator_
     *         -> ConvCheck_
     *     -> GetPDSystemSolver()
     *         -> PDSystemSolverFactory()
     *             -> GetAugSystemSolver()
     *                 -> AugSystemSolverFactory()
     *                     -> GetSymLinearSolver()
     *                         -> SymLinearSolverFactory()
     *                     -> custom_solver_
     */
    virtual SmartPtr<MuUpdate> BuildMuUpdate(const Journalist& jnlst,
                                             const OptionsList& options,
                                             const std::string& prefix);
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
    //AlgorithmBuilder();

    /** Copy Constructor */
    AlgorithmBuilder(const AlgorithmBuilder&);

    /** Overloaded Equals Operator */
    void operator=(const AlgorithmBuilder&);
    //@}

    /** @name IpoptAlgorithm constructor arguments.
     *  These components are built in separate Build
     *  methods in the order defined by BuildBasicAlgorithm.
     *  A single core component may require one or more
     *  other core components in its constructor, so the
     *  this class holds pointers to each component for use
     *  between the separate Build methods. */
    //@{
    SmartPtr<IterationOutput> IterOutput_;
    SmartPtr<HessianUpdater> HessUpdater_;
    SmartPtr<ConvergenceCheck> ConvCheck_;
    SmartPtr<SearchDirectionCalculator> SearchDirCalc_;
    SmartPtr<EqMultiplierCalculator> EqMultCalculator_;
    SmartPtr<IterateInitializer> IterInitializer_;
    SmartPtr<LineSearch> LineSearch_;
    SmartPtr<MuUpdate> MuUpdate_;
    //@}

    /** @name Commonly used solver components
     *  for building core algorithm components. Each
     *  of these members is paired with a Factory/Getter
     *  method. */
    //@{
    SmartPtr<SymLinearSolver> SymSolver_;
    SmartPtr<AugSystemSolver> AugSolver_;
    SmartPtr<PDSystemSolver> PDSolver_;
    //@}

    /** Optional pointer to AugSystemSolver.  If this is set in the
     *  contructor, we will use this to solve the linear systems. */
    SmartPtr<AugSystemSolver> custom_solver_;

  };
} // namespace Ipopt

#endif
