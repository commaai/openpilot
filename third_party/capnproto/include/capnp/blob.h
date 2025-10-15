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

#include <kj/common.h>
#include <kj/string.h>
#include "common.h"
#include <string.h>

CAPNP_BEGIN_HEADER

namespace capnp {

struct Data {
  Data() = delete;
  class Reader;
  class Builder;
  class Pipeline {};
};

struct Text {
  Text() = delete;
  class Reader;
  class Builder;
  class Pipeline {};
};

class Data::Reader: public kj::ArrayPtr<const byte> {
  // Points to a blob of bytes.  The usual Reader rules apply -- Data::Reader behaves like a simple
  // pointer which does not own its target, can be passed by value, etc.

public:
  typedef Data Reads;

  Reader() = default;
  inline Reader(decltype(nullptr)): ArrayPtr<const byte>(nullptr) {}
  inline Reader(const byte* value, size_t size): ArrayPtr<const byte>(value, size) {}
  inline Reader(const kj::Array<const byte>& value): ArrayPtr<const byte>(value) {}
  inline Reader(const ArrayPtr<const byte>& value): ArrayPtr<const byte>(value) {}
  inline Reader(const kj::Array<byte>& value): ArrayPtr<const byte>(value) {}
  inline Reader(const ArrayPtr<byte>& value): ArrayPtr<const byte>(value) {}
};

class Text::Reader: public kj::StringPtr {
  // Like Data::Reader, but points at NUL-terminated UTF-8 text.  The NUL terminator is not counted
  // in the size but must be present immediately after the last byte.
  //
  // Text::Reader's interface contract is that its data MUST be NUL-terminated.  The producer of
  // the Text::Reader must guarantee this, so that the consumer need not check.  The data SHOULD
  // also be valid UTF-8, but this is NOT guaranteed -- the consumer must verify if it cares.

public:
  typedef Text Reads;

  Reader() = default;
  inline Reader(decltype(nullptr)): StringPtr(nullptr) {}
  inline Reader(const char* value): StringPtr(value) {}
  inline Reader(const char* value, size_t size): StringPtr(value, size) {}
  inline Reader(const kj::String& value): StringPtr(value) {}
  inline Reader(const StringPtr& value): StringPtr(value) {}

#if KJ_COMPILER_SUPPORTS_STL_STRING_INTEROP
  template <typename T, typename = decltype(kj::instance<T>().c_str())>
  inline Reader(const T& t): StringPtr(t) {}
  // Allow implicit conversion from any class that has a c_str() method (namely, std::string).
  // We use a template trick to detect std::string in order to avoid including the header for
  // those who don't want it.
#endif
};

class Data::Builder: public kj::ArrayPtr<byte> {
  // Like Data::Reader except the pointers aren't const.

public:
  typedef Data Builds;

  Builder() = default;
  inline Builder(decltype(nullptr)): ArrayPtr<byte>(nullptr) {}
  inline Builder(byte* value, size_t size): ArrayPtr<byte>(value, size) {}
  inline Builder(kj::Array<byte>& value): ArrayPtr<byte>(value) {}
  inline Builder(ArrayPtr<byte> value): ArrayPtr<byte>(value) {}

  inline Data::Reader asReader() const {
    return Data::Reader(kj::implicitCast<const kj::ArrayPtr<byte>&>(*this));
  }
  inline operator Reader() const { return asReader(); }
};

class Text::Builder: public kj::DisallowConstCopy {
  // Basically identical to kj::StringPtr, except that the contents are non-const.

public:
  inline Builder(): content(nulstr, 1) {}
  inline Builder(decltype(nullptr)): content(nulstr, 1) {}
  inline Builder(char* value): content(value, strlen(value) + 1) {}
  inline Builder(char* value, size_t size): content(value, size + 1) {
    KJ_IREQUIRE(value[size] == '\0', "StringPtr must be NUL-terminated.");
  }

  inline Reader asReader() const { return Reader(content.begin(), content.size() - 1); }
  inline operator Reader() const { return asReader(); }

  inline operator kj::ArrayPtr<char>();
  inline kj::ArrayPtr<char> asArray();
  inline operator kj::ArrayPtr<const char>() const;
  inline kj::ArrayPtr<const char> asArray() const;
  inline kj::ArrayPtr<byte> asBytes() { return asArray().asBytes(); }
  inline kj::ArrayPtr<const byte> asBytes() const { return asArray().asBytes(); }
  // Result does not include NUL terminator.

  inline operator kj::StringPtr() const;
  inline kj::StringPtr asString() const;

  inline const char* cStr() const { return content.begin(); }
  // Returns NUL-terminated string.

  inline size_t size() const { return content.size() - 1; }
  // Result does not include NUL terminator.

  inline char operator[](size_t index) const { return content[index]; }
  inline char& operator[](size_t index) { return content[index]; }

  inline char* begin() { return content.begin(); }
  inline char* end() { return content.end() - 1; }
  inline const char* begin() const { return content.begin(); }
  inline const char* end() const { return content.end() - 1; }

  inline bool operator==(decltype(nullptr)) const { return content.size() <= 1; }
  inline bool operator!=(decltype(nullptr)) const { return content.size() > 1; }

  inline bool operator==(Builder other) const { return asString() == other.asString(); }
  inline bool operator!=(Builder other) const { return asString() != other.asString(); }
  inline bool operator< (Builder other) const { return asString() <  other.asString(); }
  inline bool operator> (Builder other) const { return asString() >  other.asString(); }
  inline bool operator<=(Builder other) const { return asString() <= other.asString(); }
  inline bool operator>=(Builder other) const { return asString() >= other.asString(); }

  inline kj::StringPtr slice(size_t start) const;
  inline kj::ArrayPtr<const char> slice(size_t start, size_t end) const;
  inline Builder slice(size_t start);
  inline kj::ArrayPtr<char> slice(size_t start, size_t end);
  // A string slice is only NUL-terminated if it is a suffix, so slice() has a one-parameter
  // version that assumes end = size().

private:
  inline explicit Builder(kj::ArrayPtr<char> content): content(content) {}

  kj::ArrayPtr<char> content;

  static char nulstr[1];
};

inline kj::StringPtr KJ_STRINGIFY(Text::Builder builder) {
  return builder.asString();
}

inline bool operator==(const char* a, const Text::Builder& b) { return b.asString() == a; }
inline bool operator!=(const char* a, const Text::Builder& b) { return b.asString() != a; }

inline Text::Builder::operator kj::StringPtr() const {
  return kj::StringPtr(content.begin(), content.size() - 1);
}

inline kj::StringPtr Text::Builder::asString() const {
  return kj::StringPtr(content.begin(), content.size() - 1);
}

inline Text::Builder::operator kj::ArrayPtr<char>() {
  return content.slice(0, content.size() - 1);
}

inline kj::ArrayPtr<char> Text::Builder::asArray() {
  return content.slice(0, content.size() - 1);
}

inline Text::Builder::operator kj::ArrayPtr<const char>() const {
  return content.slice(0, content.size() - 1);
}

inline kj::ArrayPtr<const char> Text::Builder::asArray() const {
  return content.slice(0, content.size() - 1);
}

inline kj::StringPtr Text::Builder::slice(size_t start) const {
  return asReader().slice(start);
}
inline kj::ArrayPtr<const char> Text::Builder::slice(size_t start, size_t end) const {
  return content.slice(start, end);
}

inline Text::Builder Text::Builder::slice(size_t start) {
  return Text::Builder(content.slice(start, content.size()));
}
inline kj::ArrayPtr<char> Text::Builder::slice(size_t start, size_t end) {
  return content.slice(start, end);
}

}  // namespace capnp

CAPNP_END_HEADER
