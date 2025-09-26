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

#include <capnp/schema.capnp.h>
#include "message.h"

CAPNP_BEGIN_HEADER

namespace capnp {

template <typename T, typename CapnpPrivate = typename T::_capnpPrivate>
inline schema::Node::Reader schemaProto() {
  // Get the schema::Node for this type's schema. This function works even in lite mode.
  return readMessageUnchecked<schema::Node>(CapnpPrivate::encodedSchema());
}

template <typename T, uint64_t id = schemas::EnumInfo<T>::typeId>
inline schema::Node::Reader schemaProto() {
  // Get the schema::Node for this type's schema. This function works even in lite mode.
  return readMessageUnchecked<schema::Node>(schemas::EnumInfo<T>::encodedSchema());
}

}  // namespace capnp

CAPNP_END_HEADER
