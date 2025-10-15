// Copyright (c) 2014 Sandstorm Development Group, Inc. and contributors
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

/*
 * Compatibility layer for stdlib iostream
 */

#pragma once

#include "../io.h"
#include <iostream>

KJ_BEGIN_HEADER

namespace kj {
namespace std {

class StdOutputStream: public kj::OutputStream {

public:
  explicit StdOutputStream(::std::ostream& stream) : stream_(stream) {}
  ~StdOutputStream() noexcept(false) {}

  virtual void write(const void* src, size_t size) override {
    // Always writes the full size.

    stream_.write((char*)src, size);
  }

  virtual void write(ArrayPtr<const ArrayPtr<const byte>> pieces) override {
    // Equivalent to write()ing each byte array in sequence, which is what the
    // default implementation does. Override if you can do something better,
    // e.g. use writev() to do the write in a single syscall.

    for (auto piece : pieces) {
      write(piece.begin(), piece.size());
    }
  }

private:
  ::std::ostream& stream_;

};

class StdInputStream: public kj::InputStream {

public:
  explicit StdInputStream(::std::istream& stream) : stream_(stream) {}
  ~StdInputStream() noexcept(false) {}

  virtual size_t tryRead(
      void* buffer, size_t minBytes, size_t maxBytes) override {
    // Like read(), but may return fewer than minBytes on EOF.

    stream_.read((char*)buffer, maxBytes);
    return stream_.gcount();
  }

private:
  ::std::istream& stream_;

};

}  // namespace std
}  // namespace kj

KJ_END_HEADER
