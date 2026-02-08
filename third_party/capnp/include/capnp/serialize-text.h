// Copyright (c) 2015 Philip Quinn.
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

#include <kj/string.h>
#include "dynamic.h"
#include "orphan.h"
#include "schema.h"

CAPNP_BEGIN_HEADER

namespace capnp {

class TextCodec {
  // Reads and writes Cap'n Proto objects in a plain text format (as used in the schema
  // language for constants, and read/written by the 'decode' and 'encode' commands of
  // the capnp tool).
  //
  // This format is useful for debugging or human input, but it is not a robust alternative
  // to the binary format. Changes to a schema's types or names that are permitted in a
  // schema's binary evolution will likely break messages stored in this format.
  //
  // Note that definitions or references (to constants, other fields, or files) are not
  // permitted in this format. To evaluate declarations with the full expressiveness of the
  // schema language, see `capnp::SchemaParser`.
  //
  // Requires linking with the capnpc library.

public:
  TextCodec();
  ~TextCodec() noexcept(true);

  void setPrettyPrint(bool enabled);
  // If enabled, pads the output of `encode()` with spaces and newlines to make it more
  // human-readable.

  template <typename T>
  kj::String encode(T&& value) const;
  kj::String encode(DynamicValue::Reader value) const;
  // Encode any Cap'n Proto value.

  template <typename T>
  Orphan<T> decode(kj::StringPtr input, Orphanage orphanage) const;
  // Decode a text message into a Cap'n Proto object of type T, allocated in the given
  // orphanage. Any errors parsing the input or assigning the fields of T are thrown as
  // exceptions.

  void decode(kj::StringPtr input, DynamicStruct::Builder output) const;
  // Decode a text message for a struct into the given builder. Any errors parsing the
  // input or assigning the fields of the output are thrown as exceptions.

  // TODO(someday): expose some control over the error handling?
private:
  Orphan<DynamicValue> decode(kj::StringPtr input, Type type, Orphanage orphanage) const;

  bool prettyPrint;
};

// =======================================================================================
// inline stuff

template <typename T>
inline kj::String TextCodec::encode(T&& value) const {
  return encode(DynamicValue::Reader(ReaderFor<FromAny<T>>(kj::fwd<T>(value))));
}

template <typename T>
inline Orphan<T> TextCodec::decode(kj::StringPtr input, Orphanage orphanage) const {
  return decode(input, Type::from<T>(), orphanage).template releaseAs<T>();
}

}  // namespace capnp

CAPNP_END_HEADER
