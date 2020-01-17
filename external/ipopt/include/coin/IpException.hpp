// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpException.hpp 2023 2011-06-18 18:49:49Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPEXCEPTION_HPP__
#define __IPEXCEPTION_HPP__

#include "IpUtils.hpp"
#include "IpJournalist.hpp"

/*  This file contains a base class for all exceptions
 *  and a set of macros to help with exceptions
 */

namespace Ipopt
{

  /** This is the base class for all exceptions.  The easiest way to
   *   use this class is by means of the following macros:
   *
   * \verbatim

     DECLARE_STD_EXCEPTION(ExceptionType);
     \endverbatim
   *
   * This macro defines a new class with the name ExceptionType,
   * inherited from the base class IpoptException.  After this,
   * exceptions of this type can be thrown using
   *
   * \verbatim

     THROW_EXCEPTION(ExceptionType, Message);
     \endverbatim
   *
   * where Message is a std::string with a message that gives an
   * indication of what caused the exception.  Exceptions can also be
   * thrown using the macro
   *
   * \verbatim

     ASSERT_EXCEPTION(Condition, ExceptionType, Message);
     \endverbatim
   *
   * where Conditions is an expression.  If Condition evaluates to
   * false, then the exception of the type ExceptionType is thrown
   * with Message.
   *
   * When an exception is caught, the method ReportException can be
   * used to write the information about the exception to the
   * Journalist, using the level J_ERROR and the category J_MAIN.
   *
   */
  class IpoptException
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Constructor */
    IpoptException(std::string msg, std::string file_name, Index line_number, std::string type="IpoptException")
        :
        msg_(msg),
        file_name_(file_name),
        line_number_(line_number),
        type_(type)
    {}

    /** Copy Constructor */
    IpoptException(const IpoptException& copy)
        :
        msg_(copy.msg_),
        file_name_(copy.file_name_),
        line_number_(copy.line_number_),
        type_(copy.type_)
    {}

    /** Default destructor */
    virtual ~IpoptException()
    {}
    //@}

    /** Method to report the exception to a journalist */
    void ReportException(const Journalist& jnlst,
                         EJournalLevel level = J_ERROR) const
    {
      jnlst.Printf(level, J_MAIN,
                   "Exception of type: %s in file \"%s\" at line %d:\n Exception message: %s\n",
                   type_.c_str(), file_name_.c_str(),  line_number_, msg_.c_str());
    }

    const std::string& Message() const
    {
      return msg_;
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
    IpoptException();

    /** Overloaded Equals Operator */
    void operator=(const IpoptException&);
    //@}

    std::string msg_;
    std::string file_name_;
    Index line_number_;
    std::string type_;
  };

} // namespace Ipopt

#define THROW_EXCEPTION(__except_type, __msg) \
  throw __except_type( (__msg), (__FILE__), (__LINE__) );

#define ASSERT_EXCEPTION(__condition, __except_type, __msg) \
  if (! (__condition) ) { \
    std::string newmsg = #__condition; \
    newmsg += " evaluated false: "; \
    newmsg += __msg; \
    throw __except_type( (newmsg), (__FILE__), (__LINE__) ); \
  }

#define DECLARE_STD_EXCEPTION(__except_type) \
    class __except_type : public Ipopt::IpoptException \
    { \
    public: \
      __except_type(std::string msg, std::string fname, Ipopt::Index line) \
 : Ipopt::IpoptException(msg,fname,line, #__except_type) {} \
      __except_type(const __except_type& copy) \
 : Ipopt::IpoptException(copy) {} \
    private: \
       __except_type(); \
       void operator=(const __except_type&); \
    }

#endif
