// Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
// Licensed under the MIT License:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// This exposes IndexingIterator as something compatible with std::iterator so that things like
// std::copy work with List::begin/List::end.

// Make sure that if this header is before list.h by the user it includes it to make
// IndexingIterator visible to avoid brittle header problems.
#include "../list.h"
#include <iterator>

CAPNP_BEGIN_HEADER

namespace std {

template <typename Container, typename Element>
struct iterator_traits<capnp::_::IndexingIterator<Container, Element>> {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = Element;
  using difference_type	= int;
  using pointer = Element*;
  using reference = Element;
};

}  // namespace std

CAPNP_END_HEADER
