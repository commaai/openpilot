/*
 * Copyright (C) 2005 The Android Open Source Project
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

#ifndef ANDROID_UTILS_DEBUG_H
#define ANDROID_UTILS_DEBUG_H

#include <stdint.h>
#include <sys/types.h>

namespace android {
// ---------------------------------------------------------------------------

#ifdef __cplusplus
template<bool> struct CompileTimeAssert;
template<> struct CompileTimeAssert<true> {};
#define COMPILE_TIME_ASSERT(_exp) \
    template class CompileTimeAssert< (_exp) >;
#endif
#define COMPILE_TIME_ASSERT_FUNCTION_SCOPE(_exp) \
    CompileTimeAssert<( _exp )>();

// ---------------------------------------------------------------------------

#ifdef __cplusplus
template<bool C, typename LSH, typename RHS> struct CompileTimeIfElse;
template<typename LHS, typename RHS> 
struct CompileTimeIfElse<true,  LHS, RHS> { typedef LHS TYPE; };
template<typename LHS, typename RHS> 
struct CompileTimeIfElse<false, LHS, RHS> { typedef RHS TYPE; };
#endif

// ---------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_UTILS_DEBUG_H
