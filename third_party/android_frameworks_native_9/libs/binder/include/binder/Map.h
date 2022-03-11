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

#ifndef ANDROID_MAP_H
#define ANDROID_MAP_H

#include <map>
#include <string>

// ---------------------------------------------------------------------------
namespace android {
namespace binder {

class Value;

/**
 * Convenience typedef for ::std::map<::std::string,::android::binder::Value>
 */
typedef ::std::map<::std::string, Value> Map;

} // namespace binder
} // namespace android

// ---------------------------------------------------------------------------

#endif // ANDROID_MAP_H
