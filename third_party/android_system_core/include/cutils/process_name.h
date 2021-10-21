/*
 * Copyright (C) 2008 The Android Open Source Project
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

/**
 * Gives the current process a name.
 */

#ifndef __PROCESS_NAME_H
#define __PROCESS_NAME_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Sets the current process name.
 *
 * Warning: This leaks a string every time you call it. Use judiciously!
 */
void set_process_name(const char* process_name);

/** Gets the current process name. */
const char* get_process_name(void);

#ifdef __cplusplus
}
#endif

#endif /* __PROCESS_NAME_H */ 
