// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpUtils.hpp 2167 2013-03-08 11:15:38Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPUTILS_HPP__
#define __IPUTILS_HPP__

// Standard Ip Include Files
#include "IpTypes.hpp"
#include "IpDebug.hpp"

namespace Ipopt
{

  inline Index Max(Index a, Index b)
  {
    return ((a) > (b) ? (a) : (b));
  }

  inline Index Max(Index a, Index b, Index c)
  {
    Index max = Max(a,b);
    max = Max(max, c);
    return max;
  }

  inline Index Max(Index a, Index b, Index c, Index d)
  {
    Index max = Max(a, b, c);
    max = Max(max, d);
    return max;
  }

  inline Index Min(Index a, Index b)
  {
    return ((a) < (b) ? (a) : (b));
  }

  inline Index Min(Index a, Index b, Index c)
  {
    Index min = Min(a,b);
    min = Min(min, c);
    return min;
  }

  inline Index Min(Index a, Index b, Index c, Index d)
  {
    Index min = Min(a, b, c);
    min = Min(min, d);
    return min;
  }

  ///////////////////////////////////////////

  inline Number Max(Number a, Number b)
  {
    return ((a) > (b) ? (a) : (b));
  }

  inline Number Max(Number a, Number b, Number c)
  {
    Number max = Max(a,b);
    max = Max(max, c);
    return max;
  }

  inline Number Max(Number a, Number b, Number c, Number d)
  {
    Number max = Max(a, b, c);
    max = Max(max, d);
    return max;
  }

  inline Number Min(Number a, Number b)
  {
    return ((a) < (b) ? (a) : (b));
  }

  inline Number Min(Number a, Number b, Number c)
  {
    Number min = Min(a,b);
    min = Min(min, c);
    return min;
  }

  inline Number Min(Number a, Number b, Number c, Number d)
  {
    Number min = Min(a, b, c);
    min = Min(min, d);
    return min;
  }

  /** Function returning true iff the argument is a valid double number
   *  (not NaN or Inf). */
  bool IsFiniteNumber(Number val);

  /** Function returning a random number between 0 and 1 */
  Number IpRandom01();

  /** Function resetting the random number generator */
  void IpResetRandom01();

  /** method determining CPU time */
  Number CpuTime();

  /** method determining system time */
  Number SysTime();

  /** method determining wallclock time since first call */
  Number WallclockTime();

  /** Method for comparing two numbers within machine precision.  The
   *  return value is true if lhs is less or equal the rhs, relaxing
   *  this inequality by something a little larger than machine
   *  precision relative to the absolute value of BasVal. */
  bool Compare_le(Number lhs, Number rhs, Number BasVal);

  /** Method for printing a formatted output to a string with given size.
  */
  int Snprintf(char* str, long size, const char* format, ...);

} //namespace Ipopt

#endif
