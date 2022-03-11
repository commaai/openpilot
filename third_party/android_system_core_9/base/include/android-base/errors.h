/*
 * Copyright (C) 2016 The Android Open Source Project
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

// Portable error handling functions. This is only necessary for host-side
// code that needs to be cross-platform; code that is only run on Unix should
// just use errno and strerror() for simplicity.
//
// There is some complexity since Windows has (at least) three different error
// numbers, not all of which share the same type:
//   * errno: for C runtime errors.
//   * GetLastError(): Windows non-socket errors.
//   * WSAGetLastError(): Windows socket errors.
// errno can be passed to strerror() on all platforms, but the other two require
// special handling to get the error string. Refer to Microsoft documentation
// to determine which error code to check for each function.

#ifndef ANDROID_BASE_ERRORS_H
#define ANDROID_BASE_ERRORS_H

#include <string>

namespace android {
namespace base {

// Returns a string describing the given system error code. |error_code| must
// be errno on Unix or GetLastError()/WSAGetLastError() on Windows. Passing
// errno on Windows has undefined behavior.
std::string SystemErrorCodeToString(int error_code);

}  // namespace base
}  // namespace android

#endif  // ANDROID_BASE_ERRORS_H
