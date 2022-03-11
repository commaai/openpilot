/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef ANDROID_BASE_PARSEINT_H
#define ANDROID_BASE_PARSEINT_H

#include <errno.h>
#include <stdlib.h>

#include <limits>
#include <string>

namespace android {
namespace base {

// Parses the unsigned decimal integer in the string 's' and sets 'out' to
// that value. Optionally allows the caller to define a 'max' beyond which
// otherwise valid values will be rejected. Returns boolean success; 'out'
// is untouched if parsing fails.
template <typename T>
bool ParseUint(const char* s, T* out,
               T max = std::numeric_limits<T>::max()) {
  int base = (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ? 16 : 10;
  errno = 0;
  char* end;
  unsigned long long int result = strtoull(s, &end, base);
  if (errno != 0 || s == end || *end != '\0') {
    return false;
  }
  if (max < result) {
    return false;
  }
  *out = static_cast<T>(result);
  return true;
}

// TODO: string_view
template <typename T>
bool ParseUint(const std::string& s, T* out,
               T max = std::numeric_limits<T>::max()) {
  return ParseUint(s.c_str(), out, max);
}

// Parses the signed decimal integer in the string 's' and sets 'out' to
// that value. Optionally allows the caller to define a 'min' and 'max
// beyond which otherwise valid values will be rejected. Returns boolean
// success; 'out' is untouched if parsing fails.
template <typename T>
bool ParseInt(const char* s, T* out,
              T min = std::numeric_limits<T>::min(),
              T max = std::numeric_limits<T>::max()) {
  int base = (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ? 16 : 10;
  errno = 0;
  char* end;
  long long int result = strtoll(s, &end, base);
  if (errno != 0 || s == end || *end != '\0') {
    return false;
  }
  if (result < min || max < result) {
    return false;
  }
  *out = static_cast<T>(result);
  return true;
}

// TODO: string_view
template <typename T>
bool ParseInt(const std::string& s, T* out,
              T min = std::numeric_limits<T>::min(),
              T max = std::numeric_limits<T>::max()) {
  return ParseInt(s.c_str(), out, min, max);
}

}  // namespace base
}  // namespace android

#endif  // ANDROID_BASE_PARSEINT_H
