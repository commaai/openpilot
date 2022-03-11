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

#ifndef ANDROID_BASE_PARSEDOUBLE_H
#define ANDROID_BASE_PARSEDOUBLE_H

#include <errno.h>
#include <stdlib.h>

#include <limits>

namespace android {
namespace base {

// Parse double value in the string 's' and sets 'out' to that value.
// Optionally allows the caller to define a 'min' and 'max' beyond which
// otherwise valid values will be rejected. Returns boolean success.
static inline bool ParseDouble(const char* s, double* out,
                               double min = std::numeric_limits<double>::lowest(),
                               double max = std::numeric_limits<double>::max()) {
  errno = 0;
  char* end;
  double result = strtod(s, &end);
  if (errno != 0 || s == end || *end != '\0') {
    return false;
  }
  if (result < min || max < result) {
    return false;
  }
  *out = result;
  return true;
}

}  // namespace base
}  // namespace android

#endif  // ANDROID_BASE_PARSEDOUBLE_H
