// Copyright (C) 2005, 2010 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpAlgTypes.hpp 2551 2015-02-13 02:51:47Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2005-07-19

#ifndef __IPALGTYPES_HPP__
#define __IPALGTYPES_HPP__

#include "IpTypes.hpp"
#include "IpException.hpp"

namespace Ipopt
{

  /**@name Enumerations */
  //@{
  /** enum for the return from the optimize algorithm
   *  (obviously we need to add more) */
  enum SolverReturn {
    SUCCESS,
    MAXITER_EXCEEDED,
    CPUTIME_EXCEEDED,
    STOP_AT_TINY_STEP,
    STOP_AT_ACCEPTABLE_POINT,
    LOCAL_INFEASIBILITY,
    USER_REQUESTED_STOP,
    FEASIBLE_POINT_FOUND,
    DIVERGING_ITERATES,
    RESTORATION_FAILURE,
    ERROR_IN_STEP_COMPUTATION,
    INVALID_NUMBER_DETECTED,
    TOO_FEW_DEGREES_OF_FREEDOM,
    INVALID_OPTION,
    OUT_OF_MEMORY,
    INTERNAL_ERROR,
    UNASSIGNED
  };
  //@}

  /** @name Some exceptions used in multiple places */
  //@{
  DECLARE_STD_EXCEPTION(LOCALLY_INFEASIBLE);
  DECLARE_STD_EXCEPTION(TOO_FEW_DOF);
  DECLARE_STD_EXCEPTION(TINY_STEP_DETECTED);
  DECLARE_STD_EXCEPTION(ACCEPTABLE_POINT_REACHED);
  DECLARE_STD_EXCEPTION(FEASIBILITY_PROBLEM_SOLVED);
  DECLARE_STD_EXCEPTION(INVALID_WARMSTART);
  DECLARE_STD_EXCEPTION(INTERNAL_ABORT);
  DECLARE_STD_EXCEPTION(NO_FREE_VARIABLES_BUT_FEASIBLE);
  DECLARE_STD_EXCEPTION(NO_FREE_VARIABLES_AND_INFEASIBLE);
  DECLARE_STD_EXCEPTION(INCONSISTENT_BOUNDS);
  /** Exception FAILED_INITIALIZATION for problem during
   *  initialization of a strategy object (or other problems).  This
   *  is thrown by a strategy object, if a problem arises during
   *  initialization, such as a value out of a feasible range.
   */
  DECLARE_STD_EXCEPTION(FAILED_INITIALIZATION);
  //@}


}

#endif
