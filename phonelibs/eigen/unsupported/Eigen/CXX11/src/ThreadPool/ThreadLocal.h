// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_THREAD_LOCAL_H
#define EIGEN_CXX11_THREADPOOL_THREAD_LOCAL_H

// Try to come up with a portable implementation of thread local variables
#if EIGEN_COMP_GNUC && EIGEN_GNUC_AT_MOST(4, 7)
#define EIGEN_THREAD_LOCAL static __thread
#elif EIGEN_COMP_CLANG
#define EIGEN_THREAD_LOCAL static __thread
#else
#define EIGEN_THREAD_LOCAL static thread_local
#endif

#endif  // EIGEN_CXX11_THREADPOOL_THREAD_LOCAL_H
