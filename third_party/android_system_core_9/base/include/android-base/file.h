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

#ifndef ANDROID_BASE_FILE_H
#define ANDROID_BASE_FILE_H

#include <sys/stat.h>
#include <sys/types.h>
#include <string>

#if !defined(_WIN32) && !defined(O_BINARY)
#define O_BINARY 0
#endif

#if defined(__APPLE__)
/* Mac OS has always had a 64-bit off_t, so it doesn't have off64_t. */
typedef off_t off64_t;
#endif

namespace android {
namespace base {

bool ReadFdToString(int fd, std::string* content);
bool ReadFileToString(const std::string& path, std::string* content,
                      bool follow_symlinks = false);

bool WriteStringToFile(const std::string& content, const std::string& path,
                       bool follow_symlinks = false);
bool WriteStringToFd(const std::string& content, int fd);

#if !defined(_WIN32)
bool WriteStringToFile(const std::string& content, const std::string& path,
                       mode_t mode, uid_t owner, gid_t group,
                       bool follow_symlinks = false);
#endif

bool ReadFully(int fd, void* data, size_t byte_count);

// Reads `byte_count` bytes from the file descriptor at the specified offset.
// Returns false if there was an IO error or EOF was reached before reading `byte_count` bytes.
//
// NOTE: On Linux/Mac, this function wraps pread, which provides atomic read support without
// modifying the read pointer of the file descriptor. On Windows, however, the read pointer does
// get modified. This means that ReadFullyAtOffset can be used concurrently with other calls to the
// same function, but concurrently seeking or reading incrementally can lead to unexpected
// behavior.
bool ReadFullyAtOffset(int fd, void* data, size_t byte_count, off64_t offset);

bool WriteFully(int fd, const void* data, size_t byte_count);

bool RemoveFileIfExists(const std::string& path, std::string* err = nullptr);

#if !defined(_WIN32)
bool Realpath(const std::string& path, std::string* result);
bool Readlink(const std::string& path, std::string* result);
#endif

std::string GetExecutablePath();
std::string GetExecutableDirectory();

// Like the regular basename and dirname, but thread-safe on all
// platforms and capable of correctly handling exotic Windows paths.
std::string Basename(const std::string& path);
std::string Dirname(const std::string& path);

}  // namespace base
}  // namespace android

#endif // ANDROID_BASE_FILE_H
