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

#ifndef ANDROID_BASE_TEST_UTILS_H
#define ANDROID_BASE_TEST_UTILS_H

#include <regex>
#include <string>

#include <android-base/macros.h>

class TemporaryFile {
 public:
  TemporaryFile();
  explicit TemporaryFile(const std::string& tmp_dir);
  ~TemporaryFile();

  // Release the ownership of fd, caller is reponsible for closing the
  // fd or stream properly.
  int release();

  int fd;
  char path[1024];

 private:
  void init(const std::string& tmp_dir);

  DISALLOW_COPY_AND_ASSIGN(TemporaryFile);
};

class TemporaryDir {
 public:
  TemporaryDir();
  ~TemporaryDir();

  char path[1024];

 private:
  bool init(const std::string& tmp_dir);

  DISALLOW_COPY_AND_ASSIGN(TemporaryDir);
};

class CapturedStderr {
 public:
  CapturedStderr();
  ~CapturedStderr();

  int fd() const;

 private:
  void init();
  void reset();

  TemporaryFile temp_file_;
  int old_stderr_;

  DISALLOW_COPY_AND_ASSIGN(CapturedStderr);
};

#define ASSERT_MATCH(str, pattern)                                             \
  do {                                                                         \
    if (!std::regex_search((str), std::regex((pattern)))) {                    \
      FAIL() << "regex mismatch: expected " << (pattern) << " in:\n" << (str); \
    }                                                                          \
  } while (0)

#define ASSERT_NOT_MATCH(str, pattern)                                                     \
  do {                                                                                     \
    if (std::regex_search((str), std::regex((pattern)))) {                                 \
      FAIL() << "regex mismatch: expected to not find " << (pattern) << " in:\n" << (str); \
    }                                                                                      \
  } while (0)

#define EXPECT_MATCH(str, pattern)                                                    \
  do {                                                                                \
    if (!std::regex_search((str), std::regex((pattern)))) {                           \
      ADD_FAILURE() << "regex mismatch: expected " << (pattern) << " in:\n" << (str); \
    }                                                                                 \
  } while (0)

#define EXPECT_NOT_MATCH(str, pattern)                                                            \
  do {                                                                                            \
    if (std::regex_search((str), std::regex((pattern)))) {                                        \
      ADD_FAILURE() << "regex mismatch: expected to not find " << (pattern) << " in:\n" << (str); \
    }                                                                                             \
  } while (0)

#endif  // ANDROID_BASE_TEST_UTILS_H
