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

#ifndef ANDROID_BASE_UTF8_H
#define ANDROID_BASE_UTF8_H

#ifdef _WIN32
#include <string>
#else
// Bring in prototypes for standard APIs so that we can import them into the utf8 namespace.
#include <fcntl.h>      // open
#include <stdio.h>      // fopen
#include <sys/stat.h>   // mkdir
#include <unistd.h>     // unlink
#endif

namespace android {
namespace base {

// Only available on Windows because this is only needed on Windows.
#ifdef _WIN32
// Convert size number of UTF-16 wchar_t's to UTF-8. Returns whether the
// conversion was done successfully.
bool WideToUTF8(const wchar_t* utf16, const size_t size, std::string* utf8);

// Convert a NULL-terminated string of UTF-16 characters to UTF-8. Returns
// whether the conversion was done successfully.
bool WideToUTF8(const wchar_t* utf16, std::string* utf8);

// Convert a UTF-16 std::wstring (including any embedded NULL characters) to
// UTF-8. Returns whether the conversion was done successfully.
bool WideToUTF8(const std::wstring& utf16, std::string* utf8);

// Convert size number of UTF-8 char's to UTF-16. Returns whether the conversion
// was done successfully.
bool UTF8ToWide(const char* utf8, const size_t size, std::wstring* utf16);

// Convert a NULL-terminated string of UTF-8 characters to UTF-16. Returns
// whether the conversion was done successfully.
bool UTF8ToWide(const char* utf8, std::wstring* utf16);

// Convert a UTF-8 std::string (including any embedded NULL characters) to
// UTF-16. Returns whether the conversion was done successfully.
bool UTF8ToWide(const std::string& utf8, std::wstring* utf16);

// Convert a file system path, represented as a NULL-terminated string of
// UTF-8 characters, to a UTF-16 string representing the same file system
// path using the Windows extended-lengh path representation.
//
// See https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx#MAXPATH:
//   ```The Windows API has many functions that also have Unicode versions to
//   permit an extended-length path for a maximum total path length of 32,767
//   characters. To specify an extended-length path, use the "\\?\" prefix.
//   For example, "\\?\D:\very long path".```
//
// Returns whether the conversion was done successfully.
bool UTF8PathToWindowsLongPath(const char* utf8, std::wstring* utf16);
#endif

// The functions in the utf8 namespace take UTF-8 strings. For Windows, these
// are wrappers, for non-Windows these just expose existing APIs. To call these
// functions, use:
//
// // anonymous namespace to avoid conflict with existing open(), unlink(), etc.
// namespace {
//   // Import functions into anonymous namespace.
//   using namespace android::base::utf8;
//
//   void SomeFunction(const char* name) {
//     int fd = open(name, ...);  // Calls android::base::utf8::open().
//     ...
//     unlink(name);              // Calls android::base::utf8::unlink().
//   }
// }
namespace utf8 {

#ifdef _WIN32
FILE* fopen(const char* name, const char* mode);
int mkdir(const char* name, mode_t mode);
int open(const char* name, int flags, ...);
int unlink(const char* name);
#else
using ::fopen;
using ::mkdir;
using ::open;
using ::unlink;
#endif

}  // namespace utf8
}  // namespace base
}  // namespace android

#endif  // ANDROID_BASE_UTF8_H
