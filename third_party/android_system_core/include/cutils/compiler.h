/*
 * Copyright (C) 2009 The Android Open Source Project
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

#ifndef ANDROID_CUTILS_COMPILER_H
#define ANDROID_CUTILS_COMPILER_H

/*
 * helps the compiler's optimizer predicting branches
 */

#ifdef __cplusplus
#   define CC_LIKELY( exp )    (__builtin_expect( !!(exp), true ))
#   define CC_UNLIKELY( exp )  (__builtin_expect( !!(exp), false ))
#else
#   define CC_LIKELY( exp )    (__builtin_expect( !!(exp), 1 ))
#   define CC_UNLIKELY( exp )  (__builtin_expect( !!(exp), 0 ))
#endif

/**
 * exports marked symbols
 *
 * if used on a C++ class declaration, this macro must be inserted
 * after the "class" keyword. For instance:
 *
 * template <typename TYPE>
 * class ANDROID_API Singleton { }
 */

#define ANDROID_API __attribute__((visibility("default")))

#endif // ANDROID_CUTILS_COMPILER_H
