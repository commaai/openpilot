/*
 *
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LIBDISKUSAGE_DIRSIZE_H
#define __LIBDISKUSAGE_DIRSIZE_H

#include <stdint.h>

__BEGIN_DECLS

int64_t stat_size(struct stat *s);
int64_t calculate_dir_size(int dfd);

__END_DECLS

#endif /* __LIBDISKUSAGE_DIRSIZE_H */
