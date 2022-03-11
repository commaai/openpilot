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

#ifndef ANDROID_BASE_PARSENETADDRESS_H
#define ANDROID_BASE_PARSENETADDRESS_H

#include <string>

namespace android {
namespace base {

// Parses |address| into |host| and |port|.
//
// If |address| doesn't contain a port number, the default value is taken from
// |port|. If |canonical_address| is non-null it will be set to "host:port" or
// "[host]:port" as appropriate.
//
// On failure, returns false and fills |error|.
bool ParseNetAddress(const std::string& address, std::string* host, int* port,
                     std::string* canonical_address, std::string* error);

}  // namespace base
}  // namespace android

#endif  // ANDROID_BASE_PARSENETADDRESS_H
