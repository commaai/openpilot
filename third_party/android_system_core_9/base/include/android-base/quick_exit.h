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

#pragma once

#include <stdlib.h>

// Provide emulation for at_quick_exit/quick_exit on platforms that don't have it.
namespace android {
namespace base {

// Bionic and glibc have quick_exit, Darwin and Windows don't.
#if !defined(__linux__)
  void quick_exit(int exit_code) __attribute__((noreturn));
  int at_quick_exit(void (*func)());
#else
  using ::at_quick_exit;
  using ::quick_exit;
#endif
}
}
