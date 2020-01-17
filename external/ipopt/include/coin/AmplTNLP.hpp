// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: AmplTNLP.hpp 2242 2013-04-24 19:26:30Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPAMPLTNLP_HPP__
#define __IPAMPLTNLP_HPP__

#include "IpUtils.hpp"
#include "IpTNLP.hpp"
#include "IpJournalist.hpp"
#include "IpOptionsList.hpp"

#include <map>
#include <string>

/* non Ipopt forward declaration */
struct ASL_pfgh;
struct SufDecl;
struct SufDesc;

namespace Ipopt
{
  class AmplSuffixHandler : public ReferencedObject
  {
  public:
    AmplSuffixHandler();

    ~AmplSuffixHandler();

    enum Suffix_Type
    {
      Index_Type,
      Number_Type
    };

    enum Suffix_Source
    {
      Variable_Source,
      Constraint_Source,
      Objective_Source,
      Problem_Source
    };

    void AddAvailableSuffix(std::string suffix_string, Suffix_Source source, Suffix_Type type)
    {
      suffix_ids_.push_back(suffix_string);
      suffix_types_.push_back(type);
      suffix_sources_.push_back(source);
      //      suffix_values_.push_back();
    }

    const Index* GetIntegerSuffixValues(std::string suffix_string, Suffix_Source source) const;

    const Number* GetNumberSuffixValues(std::string suffix_string, Suffix_Source source) const;

    std::vector<Index> GetIntegerSuffixValues(Index n, std::string suffix_string, Suffix_Source source) const;

    std::vector<Number> GetNumberSuffixValues(Index n, std::string suffix_string, Suffix_Source source) const;

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
    //AmplSuffixHandler();

    /** Copy Constructor */
    AmplSuffixHandler(const AmplSuffixHandler&);

    /** Overloaded Equals Operator */
    void operator=(const AmplSuffixHandler&);
    //@}

    mutable ASL_pfgh* asl_;

    SufDecl* suftab_;

    std::vector<std::string> suffix_ids_;
    std::vector<Suffix_Type> suffix_types_;
    std::vector<Suffix_Source> suffix_sources_;

    /** Method called by AmplTNLP to prepare the asl for the suffixes */
    void PrepareAmplForSuffixes(ASL_pfgh* asl);

    /** Method called by AmplTNLP to retrieve the suffixes from asl */
    //    void RetrieveSuffixesFromAmpl(ASL_pfgh* asl);

    friend class AmplTNLP;
  };

  /** Class for storing a number of AMPL options that should be
   *  registered to the AMPL Solver library interface */
  class AmplOptionsList : public ReferencedObject
  {
  public:
    enum AmplOptionType {
      String_Option,
      Number_Option,
      Integer_Option,
      WS_Option,  /* this is for AMPL's internal wantsol callback */
      HaltOnError_Option /* this is for our setting of the nerror_ member */
    };

    /** Ampl Option class, contains name, type and description for an
     *  AMPL option */
  class AmplOption : public ReferencedObject
    {
    public:
      AmplOption(const std::string ipopt_option_name,
                 AmplOptionType type,
                 const std::string description);

      ~AmplOption()
      {
        delete [] description_;
      }

      const std::string& IpoptOptionName() const
      {
        return ipopt_option_name_;
      }
      AmplOptionType Type() const
      {
        return type_;
      }
      char* Description() const
      {
        return description_;
      }
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
      AmplOption();

      /** Copy Constructor */
      AmplOption(const AmplOption&);

      /** Overloaded Equals Operator */
      void operator=(const AmplOption&);
      //@}

      const std::string ipopt_option_name_;
      const AmplOptionType type_;
      char* description_;
    };

    class PrivatInfo
    {
    public:
      PrivatInfo(const std::string ipopt_name,
                 SmartPtr<OptionsList> options,
                 SmartPtr<const Journalist> jnlst,
                 void** nerror = NULL)
          :
          ipopt_name_(ipopt_name),
          options_(options),
          jnlst_(jnlst),
          nerror_(nerror)
      {}
      const std::string& IpoptName() const
      {
        return ipopt_name_;
      }
      const SmartPtr<OptionsList>& Options() const
      {
        return options_;
      }
      const SmartPtr<const Journalist>& Jnlst() const
      {
        return jnlst_;
      }
      void** NError()
      {
        return nerror_;
      }
    private:
      const std::string ipopt_name_;
      const SmartPtr<OptionsList> options_;
      const SmartPtr<const Journalist> jnlst_;
      void** nerror_;
    };

  public:
    /** Default Constructor */
    AmplOptionsList()
        :
        keywds_(NULL),
        nkeywds_(0)
    {}

    /** Destructor */
    ~AmplOptionsList();

    /** Adding a new AMPL Option */
    void AddAmplOption(const std::string ampl_option_name,
                       const std::string ipopt_option_name,
                       AmplOptionsList::AmplOptionType type,
                       const std::string description)
    {
      SmartPtr<AmplOption> new_option =
        new AmplOption(ipopt_option_name, type, description);
      ampl_options_map_[ampl_option_name] = ConstPtr(new_option);
    }

    /** Number of AMPL Options */
    Index NumberOfAmplOptions()
    {
      return (Index)ampl_options_map_.size();
    }

    /** ASL keywords list for the stored options. */
    void* Keywords(const SmartPtr<OptionsList>& options,
                   SmartPtr<const Journalist> jnlst,
                   void** nerror);

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
    //AmplOptionsList();

    /** Copy Constructor */
    AmplOptionsList(const AmplOptionsList&);

    /** Overloaded Equals Operator */
    void operator=(const AmplOptionsList&);
    //@}

    void MakeValidLatexString(std::string source, std::string& dest) const;

    void PrintLatex(SmartPtr<const Journalist> jnlst);

    /** map for storing registered AMPL options */
    std::map<std::string, SmartPtr<const AmplOption> > ampl_options_map_;
    // AW: I think it should be with const like in the following line
    //     but with const the AIX compiler fails
    // std::map<const std::string, SmartPtr<const AmplOption> > ampl_options_map_;

    /** pointer to the keywords */
    void* keywds_;

    /** Number of entries stored in keywds_ */
    Index nkeywds_;
  };

  /** Ampl Interface.
   *  Ampl Interface, implemented as a TNLP.
   */
  class AmplTNLP : public TNLP
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Constructor. */
    AmplTNLP(const SmartPtr<const Journalist>& jnlst,
             const SmartPtr<OptionsList> options,
             char**& argv, SmartPtr<AmplSuffixHandler>
             suffix_handler = NULL, bool allow_discrete = false,
             SmartPtr<AmplOptionsList> ampl_options_list = NULL,
             const char* ampl_option_string = NULL,
             const char* ampl_invokation_string = NULL,
             const char* ampl_banner_string = NULL,
             std::string* nl_file_content = NULL);

    /** Default destructor */
    virtual ~AmplTNLP();
    //@}

    /** Exceptions */
    DECLARE_STD_EXCEPTION(NONPOSITIVE_SCALING_FACTOR);

    /**@name methods to gather information about the NLP. These
    * methods are overloaded from TNLP. See TNLP for their more
    * detailed documentation. */
    //@{
    /** returns dimensions of the nlp. Overloaded from TNLP */
    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                              Index& nnz_h_lag, IndexStyleEnum& index_style);

    /** returns names and other meta data for the variables and constraints
     *  Overloaded from TNLP */
    virtual bool get_var_con_metadata(Index n,
                                      StringMetaDataMapType& var_string_md,
                                      IntegerMetaDataMapType& var_integer_md,
                                      NumericMetaDataMapType& var_numeric_md,
                                      Index m,
                                      StringMetaDataMapType& con_string_md,
                                      IntegerMetaDataMapType& con_integer_md,
                                      NumericMetaDataMapType& con_numeric_md);

    /** returns bounds of the nlp. Overloaded from TNLP */
    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                                 Index m, Number* g_l, Number* g_u);

    /** Returns the constraint linearity.
     * array will be alocated with length n. (default implementation
     *  just return false and does not fill the array). */
    virtual bool get_constraints_linearity(Index m,
                                           LinearityType* const_types);

    /** provides a starting point for the nlp variables. Overloaded
    from TNLP */
    virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda, Number* lambda);

    /** evaluates the objective value for the nlp. Overloaded from TNLP */
    virtual bool eval_f(Index n, const Number* x, bool new_x,
                        Number& obj_value);

    /** evaluates the gradient of the objective for the
    nlp. Overloaded from TNLP */
    virtual bool eval_grad_f(Index n, const Number* x, bool new_x,
                             Number* grad_f);

    /** evaluates the constraint residuals for the nlp. Overloaded from TNLP */
    virtual bool eval_g(Index n, const Number* x, bool new_x,
                        Index m, Number* g);

    /** specifies the jacobian structure (if values is NULL) and
     *  evaluates the jacobian values (if values is not NULL) for the
     *  nlp. Overloaded from TNLP */
    virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow,
                            Index *jCol, Number* values);

    /** specifies the structure of the hessian of the lagrangian (if
     *  values is NULL) and evaluates the values (if values is not
     *  NULL). Overloaded from TNLP */
    virtual bool eval_h(Index n, const Number* x, bool new_x,
                        Number obj_factor, Index m, const Number* lambda,
                        bool new_lambda, Index nele_hess, Index* iRow,
                        Index* jCol, Number* values);

    /** retrieve the scaling parameters for the variables, objective
     *  function, and constraints. */
    virtual bool get_scaling_parameters(Number& obj_scaling,
                                        bool& use_x_scaling, Index n,
                                        Number* x_scaling,
                                        bool& use_g_scaling, Index m,
                                        Number* g_scaling);
    //@}

    /** @name Solution Methods */
    //@{
    virtual void finalize_solution(SolverReturn status,
                                   Index n, const Number* x, const Number* z_L, const Number* z_U,
                                   Index m, const Number* g, const Number* lambda,
                                   Number obj_value,
                                   const IpoptData* ip_data,
                                   IpoptCalculatedQuantities* ip_cq);
    //@}

    /** @name Method for quasi-Newton approximation information. */
    //@{
    virtual Index get_number_of_nonlinear_variables();
    virtual bool get_list_of_nonlinear_variables(Index num_nonlin_vars,
        Index* pos_nonlin_vars);
    //@}


    /**@name Ampl specific methods */
    //@{
    /** Return the ampl solver object (ASL*) */
    ASL_pfgh* AmplSolverObject()
    {
      return asl_;
    }

    /** Write the solution file.  This is a wrapper for AMPL's
     *  write_sol.  TODO Maybe this should be at a different place, or
     *  collect the numbers itself? */
    void write_solution_file(const std::string& message) const;

    /** ampl orders the variables like (continuous, binary, integer).
     *  This method gives the number of binary and integer variables.
     *  For details, see Tables 3 and 4 in "Hooking Your Solver to
     *  AMPL"
     */
    void get_discrete_info(Index& nlvb_,
                           Index& nlvbi_,
                           Index& nlvc_,
                           Index& nlvci_,
                           Index& nlvo_,
                           Index& nlvoi_,
                           Index& nbv_,
                           Index& niv_) const;
    //@}

    /** A method for setting the index of the objective function to be
     *  considered.  This method must be called after the constructor,
     *  and before anything else is called.  It can only be called
     *  once, and if there is more than one objective function in the
     *  AMPL model, it MUST be called. */
    void set_active_objective(Index obj_no);

    /**@name Methods to set meta data for the variables
     * and constraints. These values will be passed on
     * to the TNLP in get_var_con_meta_data
     */
    //@{
    void set_string_metadata_for_var(std::string tag, std::vector<std::string> meta_data)
    {
      var_string_md_[tag] = meta_data;
    }

    void set_integer_metadata_for_var(std::string tag, std::vector<Index> meta_data)
    {
      var_integer_md_[tag] = meta_data;
    }

    void set_numeric_metadata_for_var(std::string tag, std::vector<Number> meta_data)
    {
      var_numeric_md_[tag] = meta_data;
    }

    void set_string_metadata_for_con(std::string tag, std::vector<std::string> meta_data)
    {
      con_string_md_[tag] = meta_data;
    }

    void set_integer_metadata_for_con(std::string tag, std::vector<Index> meta_data)
    {
      con_integer_md_[tag] = meta_data;
    }

    void set_numeric_metadata_for_con(std::string tag, std::vector<Number> meta_data)
    {
      con_numeric_md_[tag] = meta_data;
    }
    //@}

    /** Method for returning the suffix handler */
    SmartPtr<AmplSuffixHandler> get_suffix_handler()
    {
      return suffix_handler_;
    }

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
    AmplTNLP();

    /** Copy Constructor */
    AmplTNLP(const AmplTNLP&);

    /** Overloaded Equals Operator */
    void operator=(const AmplTNLP&);
    //@}

    /** Journlist */
    SmartPtr<const Journalist> jnlst_;

    /** pointer to the main ASL structure */
    ASL_pfgh* asl_;

    /** Sign of the objective fn (1 for min, -1 for max) */
    double obj_sign_;

    /**@name Problem Size Data*/
    //@{
    Index nz_h_full_; // number of nonzeros in the full_x hessian
    /* the rest of the problem size data is available easily through the ampl variables */
    //@}

    /**@name Internal copies of data */
    //@{
    /** Solution Vectors */
    Number* x_sol_;
    Number* z_L_sol_;
    Number* z_U_sol_;
    Number* g_sol_;
    Number* lambda_sol_;
    Number obj_sol_;
    //@}

    /**@name Flags to track internal state */
    //@{
    /** true when the objective value has been calculated with the
     *  current x, set to false in apply_new_x, and set to true in
     *  internal_objval */
    bool objval_called_with_current_x_;
    /** true when the constraint values have been calculated with the
     *  current x, set to false in apply_new_x, and set to true in
     *  internal_conval */
    bool conval_called_with_current_x_;
    /** true when we have called hesset */
    bool hesset_called_;
    /** true when set_active_objective has been called */
    bool set_active_objective_called_;
    //@}

    /** Pointer to the Oinfo structure */
    void* Oinfo_ptr_;

    /** nerror flag passed to ampl calls - set to NULL to halt on error */
    void* nerror_;

    /** Suffix Handler */
    SmartPtr<AmplSuffixHandler> suffix_handler_;

    /** Make the objective call to ampl */
    bool internal_objval(const Number* x, Number& obj_val);

    /** Make the constraint call to ampl*/
    bool internal_conval(const Number* x, Index m, Number* g=NULL);

    /** Internal function to update the internal and ampl state if the
     *  x value changes */
    bool apply_new_x(bool new_x, Index n, const Number* x);

    /** Method for obtaining the name of the NL file and the options
     *  set from AMPL.  Returns a pointer to a char* with the name of
     *  the stub */
    char* get_options(const SmartPtr<OptionsList>& options,
                      SmartPtr<AmplOptionsList>& ampl_options_list,
                      const char* ampl_option_string,
                      const char* ampl_invokation_string,
                      const char* ampl_banner_string, char**& argv);

    /** returns true if the ampl nerror code is ok */
    bool nerror_ok(void* nerror);

    /** calls hesset ASL function */
    void call_hesset();

    /** meta data to pass on to TNLP */
    StringMetaDataMapType var_string_md_;
    IntegerMetaDataMapType var_integer_md_;
    NumericMetaDataMapType var_numeric_md_;
    StringMetaDataMapType con_string_md_;
    IntegerMetaDataMapType con_integer_md_;
    NumericMetaDataMapType con_numeric_md_;
  };



} // namespace Ipopt

#endif
