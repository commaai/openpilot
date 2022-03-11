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

#ifndef ANDROID_BASE_PROPERTIES_H
#define ANDROID_BASE_PROPERTIES_H

#include <sys/cdefs.h>

#if !defined(__BIONIC__)
#error Only bionic supports system properties.
#endif

#include <chrono>
#include <limits>
#include <string>

namespace android {
namespace base {

// Returns the current value of the system property `key`,
// or `default_value` if the property is empty or doesn't exist.
std::string GetProperty(const std::string& key, const std::string& default_value);

// Returns true if the system property `key` has the value "1", "y", "yes", "on", or "true",
// false for "0", "n", "no", "off", or "false", or `default_value` otherwise.
bool GetBoolProperty(const std::string& key, bool default_value);

// Returns the signed integer corresponding to the system property `key`.
// If the property is empty, doesn't exist, doesn't have an integer value, or is outside
// the optional bounds, returns `default_value`.
template <typename T> T GetIntProperty(const std::string& key,
                                       T default_value,
                                       T min = std::numeric_limits<T>::min(),
                                       T max = std::numeric_limits<T>::max());

// Returns the unsigned integer corresponding to the system property `key`.
// If the property is empty, doesn't exist, doesn't have an integer value, or is outside
// the optional bound, returns `default_value`.
template <typename T> T GetUintProperty(const std::string& key,
                                        T default_value,
                                        T max = std::numeric_limits<T>::max());

// Sets the system property `key` to `value`.
// Note that system property setting is inherently asynchronous so a return value of `true`
// isn't particularly meaningful, and immediately reading back the value won't necessarily
// tell you whether or not your call succeeded. A `false` return value definitely means failure.
bool SetProperty(const std::string& key, const std::string& value);

// Waits for the system property `key` to have the value `expected_value`.
// Times out after `relative_timeout`.
// Returns true on success, false on timeout.
bool WaitForProperty(const std::string& key, const std::string& expected_value,
                     std::chrono::milliseconds relative_timeout = std::chrono::milliseconds::max());

// Waits for the system property `key` to be created.
// Times out after `relative_timeout`.
// Returns true on success, false on timeout.
bool WaitForPropertyCreation(const std::string& key, std::chrono::milliseconds relative_timeout =
                                                         std::chrono::milliseconds::max());

} // namespace base
} // namespace android

#endif  // ANDROID_BASE_PROPERTIES_H
