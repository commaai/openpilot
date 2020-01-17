// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpIpoptData.hpp 2472 2014-04-05 17:47:20Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPIPOPTDATA_HPP__
#define __IPIPOPTDATA_HPP__

#include "IpSymMatrix.hpp"
#include "IpOptionsList.hpp"
#include "IpIteratesVector.hpp"
#include "IpRegOptions.hpp"
#include "IpTimingStatistics.hpp"

namespace Ipopt
{

  /* Forward declaration */
  class IpoptNLP;

  /** Base class for additional data that is special to a particular
   *  type of algorithm, such as the CG penalty function, or using
   *  iterative linear solvers.  The regular IpoptData object should
   *  be given a derivation of this base class when it is created. */
  class IpoptAdditionalData : public ReferencedObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default Constructor */
    IpoptAdditionalData()
    {}

    /** Default destructor */
    virtual ~IpoptAdditionalData()
    {}
    //@}

    /** This method is called to initialize the global algorithmic
     *  parameters.  The parameters are taken from the OptionsList
     *  object. */
    virtual bool Initialize(const Journalist& jnlst,
                            const OptionsList& options,
                            const std::string& prefix) = 0;

    /** Initialize Data Structures at the beginning. */
    virtual bool InitializeDataStructures() = 0;

    /** Do whatever is necessary to accept a trial point as current
     *  iterate.  This is also used to finish an iteration, i.e., to
     *  release memory, and to reset any flags for a new iteration. */
    virtual void AcceptTrialPoint() = 0;

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
    IpoptAdditionalData(const IpoptAdditionalData&);

    /** Overloaded Equals Operator */
    void operator=(const IpoptAdditionalData&);
    //@}
  };

  /** Class to organize all the data required by the algorithm.
   *  Internally, once this Data object has been initialized, all
   *  internal curr_ vectors must always be set (so that prototyes are
   *  available).  The current values can only be set from the trial
   *  values.  The trial values can be set by copying from a vector or
   *  by adding some fraction of a step to the current values.  This
   *  object also stores steps, which allows to easily communicate the
   *  step from the step computation object to the line search object.
   */
  class IpoptData : public ReferencedObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Constructor */
    IpoptData(SmartPtr<IpoptAdditionalData> add_data = NULL,
              Number cpu_time_start = -1.);

    /** Default destructor */
    virtual ~IpoptData();
    //@}

    /** Initialize Data Structures */
    bool InitializeDataStructures(IpoptNLP& ip_nlp,
                                  bool want_x,
                                  bool want_y_c,
                                  bool want_y_d,
                                  bool want_z_L,
                                  bool want_z_U);

    /** This method must be called to initialize the global
     *  algorithmic parameters.  The parameters are taken from the
     *  OptionsList object. */
    bool Initialize(const Journalist& jnlst,
                    const OptionsList& options,
                    const std::string& prefix);

    /** @name Get Methods for Iterates */
    //@{
    /** Current point */
    inline
    SmartPtr<const IteratesVector> curr() const;

    /** Get the current point in a copied container that is non-const.
    The entries in the container cannot be modified, but 
    the container can be modified to point to new entries.
    */
    //    SmartPtr<IteratesVector> curr_container() const;

    /** Get Trial point */
    inline
    SmartPtr<const IteratesVector> trial() const;

    /** Get Trial point in a copied container that is non-const.
     *  The entries in the container can not be modified, but
     *  the container can be modified to point to new entries. 
     */
    //SmartPtr<IteratesVector> trial_container() const;

    /** Set the trial point - this method copies the pointer for
     *  efficiency (no copy and to keep cache tags the same) so
     *  after you call set you cannot modify the data again
     */
    inline
    void set_trial(SmartPtr<IteratesVector>& trial);

    /** Set the values of the primal trial variables (x and s) from
     *  provided Step with step length alpha.
     */
    void SetTrialPrimalVariablesFromStep(Number alpha,
                                         const Vector& delta_x,
                                         const Vector& delta_s);
    /** Set the values of the trial values for the equality constraint
     *  multipliers (y_c and y_d) from provided step with step length
     *  alpha.
     */
    void SetTrialEqMultipliersFromStep(Number alpha,
                                       const Vector& delta_y_c,
                                       const Vector& delta_y_d);
    /** Set the value of the trial values for the bound multipliers
     *  (z_L, z_U, v_L, v_U) from provided step with step length
     *  alpha.
     */
    void SetTrialBoundMultipliersFromStep(Number alpha,
                                          const Vector& delta_z_L,
                                          const Vector& delta_z_U,
                                          const Vector& delta_v_L,
                                          const Vector& delta_v_U);

    /** ToDo: I may need to add versions of set_trial like the
     *  following, but I am not sure 
     */
    // void set_trial(const SmartPtr<IteratesVector>& trial_iterates);
    // void set_trial(SmartPtr<const IteratesVector>& trial_iterates);

    /** get the current delta */
    inline
    SmartPtr<const IteratesVector> delta() const;

    /** Set the current delta - like the trial point, this method copies
     *  the pointer for efficiency (no copy and to keep cache tags the
     *  same) so after you call set, you cannot modify the data
     */
    inline
    void set_delta(SmartPtr<IteratesVector>& delta);

    /** Set the current delta - like the trial point, this method
     *  copies the pointer for efficiency (no copy and to keep cache
     *  tags the same) so after you call set, you cannot modify the
     *  data.  This is the version that is happy with a pointer to
     *  const IteratesVector.
     */
    inline
    void set_delta(SmartPtr<const IteratesVector>& delta);

    /** Affine Delta */
    inline
    SmartPtr<const IteratesVector> delta_aff() const;

    /** Set the affine delta - like the trial point, this method copies
     *  the pointer for efficiency (no copy and to keep cache tags the
     *  same) so after you call set, you cannot modify the data
     */
    inline
    void set_delta_aff(SmartPtr<IteratesVector>& delta_aff);

    /** Hessian or Hessian approximation (do not hold on to it, it might be changed) */
    SmartPtr<const SymMatrix> W()
    {
      DBG_ASSERT(IsValid(W_));
      return W_;
    }

    /** Set Hessian approximation */
    void Set_W(SmartPtr<const SymMatrix> W)
    {
      W_ = W;
    }

    /** @name ("Main") Primal-dual search direction.  Those fields are
     *  used to store the search directions computed from solving the
     *  primal-dual system, and can be used in the line search.  They
     *  are overwritten in every iteration, so do not hold on to the
     *  pointers (make copies instead) */
    //@{

    /** Returns true, if the primal-dual step have been already
     *  computed for the current iteration.  This flag is reset after
     *  every call of AcceptTrialPoint().  If the search direction is
     *  computed during the computation of the barrier parameter, the
     *  method computing the barrier parameter should call
     *  SetHaveDeltas(true) to tell the IpoptAlgorithm object that it
     *  doesn't need to recompute the primal-dual step. */
    bool HaveDeltas() const
    {
      return have_deltas_;
    }

    /** Method for setting the HaveDeltas flag.  This method should be
     *  called if some method computes the primal-dual step (and
     *  stores it in the delta_ fields of IpoptData) at an early part
     *  of the iteration.  If that flag is set to true, the
     *  IpoptAlgorithm object will not recompute the step. */
    void SetHaveDeltas(bool have_deltas)
    {
      have_deltas_ = have_deltas;
    }
    //@}

    /** @name Affine-scaling step.  Those fields can be used to store
     *  the affine scaling step.  For example, if the method for
     *  computing the current barrier parameter computes the affine
     *  scaling steps, then the corrector step in the line search does
     *  not have to recompute those solutions of the linear system. */
    //@{

    /** Returns true, if the affine-scaling step have been already
     *  computed for the current iteration.  This flag is reset after
     *  every call of AcceptTrialPoint().  If the search direction is
     *  computed during the computation of the barrier parameter, the
     *  method computing the barrier parameter should call
     *  SetHaveDeltas(true) to tell the line search does not have to
     *  recompute them in case it wants to do a corrector step. */
    bool HaveAffineDeltas() const
    {
      return have_affine_deltas_;
    }

    /** Method for setting the HaveDeltas flag.  This method should be
     *  called if some method computes the primal-dual step (and
     *  stores it in the delta_ fields of IpoptData) at an early part
     *  of the iteration.  If that flag is set to true, the
     *  IpoptAlgorithm object will not recompute the step. */
    void SetHaveAffineDeltas(bool have_affine_deltas)
    {
      have_affine_deltas_ = have_affine_deltas;
    }
    //@}

    /** @name Public Methods for updating iterates */
    //@{
    /** Copy the trial values to the current values */
    inline
    void CopyTrialToCurrent();

    /** Set the current iterate values from the
     *  trial values. */
    void AcceptTrialPoint();
    //@}

    /** @name General algorithmic data */
    //@{
    Index iter_count() const
    {
      return iter_count_;
    }
    void Set_iter_count(Index iter_count)
    {
      iter_count_ = iter_count;
    }

    Number curr_mu() const
    {
      DBG_ASSERT(mu_initialized_);
      return curr_mu_;
    }
    void Set_mu(Number mu)
    {
      curr_mu_ = mu;
      mu_initialized_ = true;
    }
    bool MuInitialized() const
    {
      return mu_initialized_;
    }

    Number curr_tau() const
    {
      DBG_ASSERT(tau_initialized_);
      return curr_tau_;
    }
    void Set_tau(Number tau)
    {
      curr_tau_ = tau;
      tau_initialized_ = true;
    }
    bool TauInitialized() const
    {
      return tau_initialized_;
    }

    void SetFreeMuMode(bool free_mu_mode)
    {
      free_mu_mode_ = free_mu_mode;
    }
    bool FreeMuMode() const
    {
      return free_mu_mode_;
    }

    /** Setting the flag that indicates if a tiny step (below machine
     *  precision) has been detected */
    void Set_tiny_step_flag(bool flag)
    {
      tiny_step_flag_ = flag;
    }
    bool tiny_step_flag()
    {
      return tiny_step_flag_;
    }
    //@}

    /** Overall convergence tolerance.  It is used in the convergence
     *  test, but also in some other parts of the algorithm that
     *  depend on the specified tolerance, such as the minimum value
     *  for the barrier parameter. */
    //@{
    /** Obtain the tolerance. */
    Number tol() const
    {
      DBG_ASSERT(initialize_called_);
      return tol_;
    }
    /** Set a new value for the tolerance.  One should be very careful
     *  when using this, since changing the predefined tolerance might
     *  have unexpected consequences.  This method is for example used
     *  in the restoration convergence checker to tighten the
     *  restoration phase convergence tolerance, if the restoration
     *  phase converged to a point that has not a large value for the
     *  constraint violation. */
    void Set_tol(Number tol)
    {
      tol_ = tol;
    }
    //@}

    /** Cpu time counter at the beginning of the optimization.  This
     *  is useful to see how much CPU time has been spent in this
     *  optimization run. */
    Number cpu_time_start() const
    {
      return cpu_time_start_;
    }

    /** @name Information gathered for iteration output */
    //@{
    Number info_regu_x() const
    {
      return info_regu_x_;
    }
    void Set_info_regu_x(Number regu_x)
    {
      info_regu_x_ = regu_x;
    }
    Number info_alpha_primal() const
    {
      return info_alpha_primal_;
    }
    void Set_info_alpha_primal(Number alpha_primal)
    {
      info_alpha_primal_ = alpha_primal;
    }
    char info_alpha_primal_char() const
    {
      return info_alpha_primal_char_;
    }
    void Set_info_alpha_primal_char(char info_alpha_primal_char)
    {
      info_alpha_primal_char_ = info_alpha_primal_char;
    }
    Number info_alpha_dual() const
    {
      return info_alpha_dual_;
    }
    void Set_info_alpha_dual(Number alpha_dual)
    {
      info_alpha_dual_ = alpha_dual;
    }
    Index info_ls_count() const
    {
      return info_ls_count_;
    }
    void Set_info_ls_count(Index ls_count)
    {
      info_ls_count_ = ls_count;
    }
    bool info_skip_output() const
    {
      return info_skip_output_;
    }
    void Append_info_string(const std::string& add_str)
    {
      info_string_ += add_str;
    }
    const std::string& info_string() const
    {
      return info_string_;
    }
    /** Set this to true, if the next time when output is written, the
     *  summary line should not be printed. */
    void Set_info_skip_output(bool info_skip_output)
    {
      info_skip_output_ = info_skip_output;
    }

    /** gives time when the last summary output line was printed */
    Number info_last_output()
    {
       return info_last_output_;
    }
    /** sets time when the last summary output line was printed */
    void Set_info_last_output(Number info_last_output)
    {
       info_last_output_ = info_last_output;
    }

    /** gives number of iteration summaries actually printed
     * since last summary header was printed */
    int info_iters_since_header()
    {
       return info_iters_since_header_;
    }
    /** increases number of iteration summaries actually printed
     * since last summary header was printed */
    void Inc_info_iters_since_header()
    {
       info_iters_since_header_++;
    }
    /** sets number of iteration summaries actually printed
     * since last summary header was printed */
    void Set_info_iters_since_header(int info_iters_since_header)
    {
       info_iters_since_header_ = info_iters_since_header;
    }

    /** Reset all info fields */
    void ResetInfo()
    {
      info_regu_x_ = 0;
      info_alpha_primal_ = 0;
      info_alpha_dual_ = 0.;
      info_alpha_primal_char_ = ' ';
      info_skip_output_ = false;
      info_string_.erase();
    }
    //@}

    /** Return Timing Statistics Object */
    TimingStatistics& TimingStats()
    {
      return timing_statistics_;
    }

    /** Check if additional data has been set */
    bool HaveAddData()
    {
      return IsValid(add_data_);
    }

    /** Get access to additional data object */
    IpoptAdditionalData& AdditionalData()
    {
      return *add_data_;
    }

    /** Set a new pointer for additional Ipopt data */
    void SetAddData(SmartPtr<IpoptAdditionalData> add_data)
    {
      DBG_ASSERT(!HaveAddData());
      add_data_ = add_data;
    }

    /** Set the perturbation of the primal-dual system */
    void setPDPert(Number pd_pert_x, Number pd_pert_s,
                   Number pd_pert_c, Number pd_pert_d)
    {
      pd_pert_x_ = pd_pert_x;
      pd_pert_s_ = pd_pert_s;
      pd_pert_c_ = pd_pert_c;
      pd_pert_d_ = pd_pert_d;
    }

    /** Get the current perturbation of the primal-dual system */
    void getPDPert(Number& pd_pert_x, Number& pd_pert_s,
                   Number& pd_pert_c, Number& pd_pert_d)
    {
      pd_pert_x = pd_pert_x_;
      pd_pert_s = pd_pert_s_;
      pd_pert_c = pd_pert_c_;
      pd_pert_d = pd_pert_d_;
    }

    /** Methods for IpoptType */
    //@{
    static void RegisterOptions(const SmartPtr<RegisteredOptions>& roptions);
    //@}

  private:
    /** @name Iterates */
    //@{
    /** Main iteration variables
     * (current iteration) */
    SmartPtr<const IteratesVector> curr_;

    /** Main iteration variables
     *  (trial calculations) */
    SmartPtr<const IteratesVector> trial_;

    /** Hessian (approximation) - might be changed elsewhere! */
    SmartPtr<const SymMatrix> W_;

    /** @name Primal-dual Step */
    //@{
    SmartPtr<const IteratesVector> delta_;
    /** The following flag is set to true, if some other part of the
     *  algorithm (like the method for computing the barrier
     *  parameter) has already computed the primal-dual search
     *  direction.  This flag is reset when the AcceptTrialPoint
     *  method is called.
     * ToDo: we could cue off of a null delta_;
     */
    bool have_deltas_;
    //@}

    /** @name Affine-scaling step.  This used to transfer the
     *  information about the affine-scaling step from the computation
     *  of the barrier parameter to the corrector (in the line
     *  search). */
    //@{
    SmartPtr<const IteratesVector> delta_aff_;
    /** The following flag is set to true, if some other part of the
     *  algorithm (like the method for computing the barrier
     *  parameter) has already computed the affine-scaling step.  This
     *  flag is reset when the AcceptTrialPoint method is called.
     * ToDo: we could cue off of a null delta_aff_;
     */
    bool have_affine_deltas_;
    //@}

    /** iteration count */
    Index iter_count_;

    /** current barrier parameter */
    Number curr_mu_;
    bool mu_initialized_;

    /** current fraction to the boundary parameter */
    Number curr_tau_;
    bool tau_initialized_;

    /** flag indicating if Initialize method has been called (for
     *  debugging) */
    bool initialize_called_;

    /** flag for debugging whether we have already curr_ values
     *  available (from which new Vectors can be generated */
    bool have_prototypes_;

    /** @name Global algorithm parameters.  Those are options that can
     *  be modified by the user and appear at different places in the
     *  algorithm.  They are set using an OptionsList object in the
     *  Initialize method.  */
    //@{
    /** Overall convergence tolerance */
    Number tol_;
    //@}

    /** @name Status data **/
    //@{
    /** flag indicating whether the algorithm is in the free mu mode */
    bool free_mu_mode_;
    /** flag indicating if a tiny step has been detected */
    bool tiny_step_flag_;
    //@}

    /** @name Gathered information for iteration output */
    //@{
    /** Size of regularization for the Hessian */
    Number info_regu_x_;
    /** Primal step size */
    Number info_alpha_primal_;
    /** Info character for primal step size */
    char info_alpha_primal_char_;
    /** Dual step size */
    Number info_alpha_dual_;
    /** Number of backtracking trial steps */
    Index info_ls_count_;
    /** true, if next summary output line should not be printed (eg
     *  after restoration phase. */
    bool info_skip_output_;
    /** any string of characters for the end of the output line */
    std::string info_string_;
    /** time when the last summary output line was printed */
    Number info_last_output_;
    /** number of iteration summaries actually printed since last
     * summary header was printed */
    int info_iters_since_header_;
    //@}

    /** VectorSpace for all the iterates */
    SmartPtr<IteratesVectorSpace> iterates_space_;

    /** TimingStatistics object collecting all Ipopt timing
     *  statistics */
    TimingStatistics timing_statistics_;

    /** CPU time counter at initialization. */
    Number cpu_time_start_;

    /** Object for the data specific for the Chen-Goldfarb penalty
     *  method algorithm */
    SmartPtr<IpoptAdditionalData> add_data_;

    /** @name Information about the perturbation of the primal-dual
     *  system */
    //@{
    Number pd_pert_x_;
    Number pd_pert_s_;
    Number pd_pert_c_;
    Number pd_pert_d_;
    //@}

    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Copy Constructor */
    IpoptData(const IpoptData&);

    /** Overloaded Equals Operator */
    void operator=(const IpoptData&);
    //@}

#if COIN_IPOPT_CHECKLEVEL > 0
    /** Some debug flags to make sure vectors are not changed
     *  behind the IpoptData's back
     */
    //@{
    TaggedObject::Tag debug_curr_tag_;
    TaggedObject::Tag debug_trial_tag_;
    TaggedObject::Tag debug_delta_tag_;
    TaggedObject::Tag debug_delta_aff_tag_;
    TaggedObject::Tag debug_curr_tag_sum_;
    TaggedObject::Tag debug_trial_tag_sum_;
    TaggedObject::Tag debug_delta_tag_sum_;
    TaggedObject::Tag debug_delta_aff_tag_sum_;
    //@}
#endif

  };

  inline
  SmartPtr<const IteratesVector> IpoptData::curr() const
  {
    DBG_ASSERT(IsNull(curr_) || (curr_->GetTag() == debug_curr_tag_ && curr_->GetTagSum() == debug_curr_tag_sum_) );

    return curr_;
  }

  inline
  SmartPtr<const IteratesVector> IpoptData::trial() const
  {
    DBG_ASSERT(IsNull(trial_) || (trial_->GetTag() == debug_trial_tag_ && trial_->GetTagSum() == debug_trial_tag_sum_) );

    return trial_;
  }

  inline
  SmartPtr<const IteratesVector> IpoptData::delta() const
  {
    DBG_ASSERT(IsNull(delta_) || (delta_->GetTag() == debug_delta_tag_ && delta_->GetTagSum() == debug_delta_tag_sum_) );

    return delta_;
  }

  inline
  SmartPtr<const IteratesVector> IpoptData::delta_aff() const
  {
    DBG_ASSERT(IsNull(delta_aff_) || (delta_aff_->GetTag() == debug_delta_aff_tag_ && delta_aff_->GetTagSum() == debug_delta_aff_tag_sum_) );

    return delta_aff_;
  }

  inline
  void IpoptData::CopyTrialToCurrent()
  {
    curr_ = trial_;
#if COIN_IPOPT_CHECKLEVEL > 0

    if (IsValid(curr_)) {
      debug_curr_tag_ = curr_->GetTag();
      debug_curr_tag_sum_ = curr_->GetTagSum();
    }
    else {
      debug_curr_tag_ = 0;
      debug_curr_tag_sum_ = 0;
    }
#endif

  }

  inline
  void IpoptData::set_trial(SmartPtr<IteratesVector>& trial)
  {
    trial_ = ConstPtr(trial);

#if COIN_IPOPT_CHECKLEVEL > 0
    // verify the correct space
    DBG_ASSERT(trial_->OwnerSpace() == (VectorSpace*)GetRawPtr(iterates_space_));
    if (IsValid(trial)) {
      debug_trial_tag_ = trial->GetTag();
      debug_trial_tag_sum_ = trial->GetTagSum();
    }
    else {
      debug_trial_tag_ = 0;
      debug_trial_tag_sum_ = 0;
    }
#endif

    trial = NULL;
  }

  inline
  void IpoptData::set_delta(SmartPtr<IteratesVector>& delta)
  {
    delta_ = ConstPtr(delta);
#if COIN_IPOPT_CHECKLEVEL > 0

    if (IsValid(delta)) {
      debug_delta_tag_ = delta->GetTag();
      debug_delta_tag_sum_ = delta->GetTagSum();
    }
    else {
      debug_delta_tag_ = 0;
      debug_delta_tag_sum_ = 0;
    }
#endif

    delta = NULL;
  }

  inline
  void IpoptData::set_delta(SmartPtr<const IteratesVector>& delta)
  {
    delta_ = delta;
#if COIN_IPOPT_CHECKLEVEL > 0

    if (IsValid(delta)) {
      debug_delta_tag_ = delta->GetTag();
      debug_delta_tag_sum_ = delta->GetTagSum();
    }
    else {
      debug_delta_tag_ = 0;
      debug_delta_tag_sum_ = 0;
    }
#endif

    delta = NULL;
  }

  inline
  void IpoptData::set_delta_aff(SmartPtr<IteratesVector>& delta_aff)
  {
    delta_aff_ = ConstPtr(delta_aff);
#if COIN_IPOPT_CHECKLEVEL > 0

    if (IsValid(delta_aff)) {
      debug_delta_aff_tag_ = delta_aff->GetTag();
      debug_delta_aff_tag_sum_ = delta_aff->GetTagSum();
    }
    else {
      debug_delta_aff_tag_ = 0;
      debug_delta_aff_tag_sum_ = delta_aff->GetTagSum();
    }
#endif

    delta_aff = NULL;
  }

} // namespace Ipopt

#endif
