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

#include "dynamic.h"
#include <kj/string-tree.h>

CAPNP_BEGIN_HEADER

namespace capnp {

kj::StringTree prettyPrint(DynamicStruct::Reader value);
kj::StringTree prettyPrint(DynamicStruct::Builder value);
kj::StringTree prettyPrint(DynamicList::Reader value);
kj::StringTree prettyPrint(DynamicList::Builder value);
// Print the given Cap'n Proto struct or list with nice indentation.  Note that you can pass any
// struct or list reader or builder type to this method, since they can be implicitly converted
// to one of the dynamic types.
//
// If you don't want indentation, just use the value's KJ stringifier (e.g. pass it to kj::str(),
// any of the KJ debug macros, etc.).

}  // namespace capnp

CAPNP_END_HEADER
