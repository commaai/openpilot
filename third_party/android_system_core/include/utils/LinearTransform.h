/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _LIBS_UTILS_LINEAR_TRANSFORM_H
#define _LIBS_UTILS_LINEAR_TRANSFORM_H

#include <stdint.h>

namespace android {

// LinearTransform defines a structure which hold the definition of a
// transformation from single dimensional coordinate system A into coordinate
// system B (and back again).  Values in A and in B are 64 bit, the linear
// scale factor is expressed as a rational number using two 32 bit values.
//
// Specifically, let
// f(a) = b
// F(b) = f^-1(b) = a
// then
//
// f(a) = (((a - a_zero) * a_to_b_numer) / a_to_b_denom) + b_zero;
//
// and
//
// F(b) = (((b - b_zero) * a_to_b_denom) / a_to_b_numer) + a_zero;
//
struct LinearTransform {
  int64_t  a_zero;
  int64_t  b_zero;
  int32_t  a_to_b_numer;
  uint32_t a_to_b_denom;

  // Transform from A->B
  // Returns true on success, or false in the case of a singularity or an
  // overflow.
  bool doForwardTransform(int64_t a_in, int64_t* b_out) const;

  // Transform from B->A
  // Returns true on success, or false in the case of a singularity or an
  // overflow.
  bool doReverseTransform(int64_t b_in, int64_t* a_out) const;

  // Helpers which will reduce the fraction N/D using Euclid's method.
  template <class T> static void reduce(T* N, T* D);
  static void reduce(int32_t* N, uint32_t* D);
};


}

#endif  // _LIBS_UTILS_LINEAR_TRANSFORM_H
