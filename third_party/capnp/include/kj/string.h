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

#include <initializer_list>
#include "array.h"
#include "kj/common.h"
#include <string.h>

KJ_BEGIN_HEADER

namespace kj {
  class StringPtr;
  class LiteralStringConst;
  class String;
  class ConstString;

  class StringTree;   // string-tree.h
}

constexpr kj::StringPtr operator "" _kj(const char* str, size_t n);
// You can append _kj to a string literal to make its type be StringPtr. There are a few cases
// where you must do this for correctness:
// - When you want to declare a constexpr StringPtr. Without _kj, this is a compile error.
// - When you want to initialize a static/global StringPtr from a string literal without forcing
//   global constructor code to run at dynamic initialization time.
// - When you have a string literal that contains NUL characters. Without _kj, the string will
//   be considered to end at the first NUL.
// - When you want to initialize an ArrayPtr<const char> from a string literal, without including
//   the NUL terminator in the data. (Initializing an ArrayPtr from a regular string literal is
//   a compile error specifically due to this ambiguity.)
//
// In other cases, there should be no difference between initializing a StringPtr from a regular
// string literal vs. one with _kj (assuming the compiler is able to optimize away strlen() on a
// string literal).

constexpr kj::LiteralStringConst operator "" _kjc(const char* str, size_t n);

namespace kj {

// Our STL string SFINAE trick does not work with GCC 4.7, but it works with Clang and GCC 4.8, so
// we'll just preprocess it out if not supported.
#if __clang__ || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || _MSC_VER
#define KJ_COMPILER_SUPPORTS_STL_STRING_INTEROP 1
#endif

// =======================================================================================
// StringPtr -- A NUL-terminated ArrayPtr<const char> containing UTF-8 text.
//
// NUL bytes are allowed to appear before the end of the string.  The only requirement is that
// a NUL byte appear immediately after the last byte of the content.  This terminator byte is not
// counted in the string's size.

class StringPtr {
public:
  inline StringPtr(): content("", 1) {}
  inline StringPtr(decltype(nullptr)): content("", 1) {}
  inline StringPtr(const char* value KJ_LIFETIMEBOUND): content(value, strlen(value) + 1) {}
  inline StringPtr(const char* value KJ_LIFETIMEBOUND, size_t size): content(value, size + 1) {
    KJ_IREQUIRE(value[size] == '\0', "StringPtr must be NUL-terminated.");
  }
  inline StringPtr(const char* begin KJ_LIFETIMEBOUND, const char* end KJ_LIFETIMEBOUND): StringPtr(begin, end - begin) {}
  inline StringPtr(String&& value KJ_LIFETIMEBOUND) : StringPtr(value) {}
  inline StringPtr(const String& value KJ_LIFETIMEBOUND);
  inline StringPtr(const ConstString& value KJ_LIFETIMEBOUND);
  StringPtr& operator=(String&& value) = delete;
  inline StringPtr& operator=(decltype(nullptr)) {
    content = ArrayPtr<const char>("", 1);
    return *this;
  }

#if __cpp_char8_t
  inline StringPtr(const char8_t* value KJ_LIFETIMEBOUND): StringPtr(reinterpret_cast<const char*>(value)) {}
  inline StringPtr(const char8_t* value KJ_LIFETIMEBOUND, size_t size)
      : StringPtr(reinterpret_cast<const char*>(value), size) {}
  inline StringPtr(const char8_t* begin KJ_LIFETIMEBOUND, const char8_t* end KJ_LIFETIMEBOUND)
      : StringPtr(reinterpret_cast<const char*>(begin), reinterpret_cast<const char*>(end)) {}
  // KJ strings are and always have been UTF-8, so screw this C++20 char8_t stuff.
#endif

#if KJ_COMPILER_SUPPORTS_STL_STRING_INTEROP
  template <
    typename T,
    typename = decltype(instance<T>().c_str()),
    typename = decltype(instance<T>().size())>
  inline StringPtr(const T& t KJ_LIFETIMEBOUND): StringPtr(t.c_str(), t.size()) {}
  // Allow implicit conversion from any class that has a c_str() and a size() method (namely, std::string).
  // We use a template trick to detect std::string in order to avoid including the header for
  // those who don't want it.
  template <
    typename T,
    typename = decltype(instance<T>().c_str()),
    typename = decltype(instance<T>().size())>
  inline operator T() const { return {cStr(), size()}; }
  // Allow implicit conversion to any class that has a c_str() method and a size() method (namely, std::string).
  // We use a template trick to detect std::string in order to avoid including the header for
  // those who don't want it.
#endif

  inline constexpr operator ArrayPtr<const char>() const;
  inline constexpr ArrayPtr<const char> asArray() const;
  inline ArrayPtr<const byte> asBytes() const { return asArray().asBytes(); }
  // Result does not include NUL terminator.

  inline const char* cStr() const { return content.begin(); }
  // Returns NUL-terminated string.

  inline size_t size() const { return content.size() - 1; }
  // Result does not include NUL terminator.

  inline char operator[](size_t index) const { return content[index]; }

  inline constexpr const char* begin() const { return content.begin(); }
  inline constexpr const char* end() const { return content.end() - 1; }

  inline constexpr bool operator==(decltype(nullptr)) const { return content.size() <= 1; }
#if !__cpp_impl_three_way_comparison
  inline constexpr bool operator!=(decltype(nullptr)) const { return content.size() > 1; }
#endif

  inline bool operator==(const StringPtr& other) const;
#if !__cpp_impl_three_way_comparison
  inline bool operator!=(const StringPtr& other) const { return !(*this == other); }
#endif
  inline bool operator< (const StringPtr& other) const;
  inline bool operator> (const StringPtr& other) const { return other < *this; }
  inline bool operator<=(const StringPtr& other) const { return !(other < *this); }
  inline bool operator>=(const StringPtr& other) const { return !(*this < other); }

  inline StringPtr slice(size_t start) const;
  inline ArrayPtr<const char> slice(size_t start, size_t end) const;
  // A string slice is only NUL-terminated if it is a suffix, so slice() has a one-parameter
  // version that assumes end = size().

  inline bool startsWith(const StringPtr& other) const { return asArray().startsWith(other);}
  inline bool endsWith(const StringPtr& other) const { return asArray().endsWith(other); }

  inline Maybe<size_t> findFirst(char c) const { return asArray().findFirst(c); }
  inline Maybe<size_t> findLast(char c) const { return asArray().findLast(c); }

  template <typename T>
  T parseAs() const;
  // Parse string as template number type.
  // Integer numbers prefixed by "0x" and "0X" are parsed in base 16 (like strtoi with base 0).
  // Integer numbers prefixed by "0" are parsed in base 10 (unlike strtoi with base 0).
  // Overflowed integer numbers throw exception.
  // Overflowed floating numbers return inf.
  template <typename T>
  Maybe<T> tryParseAs() const;
  // Same as parseAs, but rather than throwing an exception we return NULL.

  template <typename... Attachments>
  ConstString attach(Attachments&&... attachments) const KJ_WARN_UNUSED_RESULT;
  ConstString attach() const KJ_WARN_UNUSED_RESULT;
  // Like ArrayPtr<T>::attach(), but instead promotes a StringPtr into a ConstString. Generally the
  // attachment should be an object that somehow owns the String that the StringPtr is pointing at.

private:
  inline explicit constexpr StringPtr(ArrayPtr<const char> content): content(content) {}
  friend constexpr StringPtr (::operator "" _kj)(const char* str, size_t n);
  friend class LiteralStringConst;

  ArrayPtr<const char> content;
  friend class SourceLocation;
};

#if !__cpp_impl_three_way_comparison
inline bool operator==(const char* a, const StringPtr& b) { return b == a; }
inline bool operator!=(const char* a, const StringPtr& b) { return b != a; }
#endif

template <> char StringPtr::parseAs<char>() const;
template <> signed char StringPtr::parseAs<signed char>() const;
template <> unsigned char StringPtr::parseAs<unsigned char>() const;
template <> short StringPtr::parseAs<short>() const;
template <> unsigned short StringPtr::parseAs<unsigned short>() const;
template <> int StringPtr::parseAs<int>() const;
template <> unsigned StringPtr::parseAs<unsigned>() const;
template <> long StringPtr::parseAs<long>() const;
template <> unsigned long StringPtr::parseAs<unsigned long>() const;
template <> long long StringPtr::parseAs<long long>() const;
template <> unsigned long long StringPtr::parseAs<unsigned long long>() const;
template <> float StringPtr::parseAs<float>() const;
template <> double StringPtr::parseAs<double>() const;

template <> Maybe<char> StringPtr::tryParseAs<char>() const;
template <> Maybe<signed char> StringPtr::tryParseAs<signed char>() const;
template <> Maybe<unsigned char> StringPtr::tryParseAs<unsigned char>() const;
template <> Maybe<short> StringPtr::tryParseAs<short>() const;
template <> Maybe<unsigned short> StringPtr::tryParseAs<unsigned short>() const;
template <> Maybe<int> StringPtr::tryParseAs<int>() const;
template <> Maybe<unsigned> StringPtr::tryParseAs<unsigned>() const;
template <> Maybe<long> StringPtr::tryParseAs<long>() const;
template <> Maybe<unsigned long> StringPtr::tryParseAs<unsigned long>() const;
template <> Maybe<long long> StringPtr::tryParseAs<long long>() const;
template <> Maybe<unsigned long long> StringPtr::tryParseAs<unsigned long long>() const;
template <> Maybe<float> StringPtr::tryParseAs<float>() const;
template <> Maybe<double> StringPtr::tryParseAs<double>() const;

class LiteralStringConst: public StringPtr {
public:
  inline operator ConstString() const;

private:
  inline explicit constexpr LiteralStringConst(ArrayPtr<const char> content): StringPtr(content) {}
  friend constexpr LiteralStringConst (::operator "" _kjc)(const char* str, size_t n);
};

// =======================================================================================
// String -- A NUL-terminated Array<char> containing UTF-8 text.
//
// NUL bytes are allowed to appear before the end of the string.  The only requirement is that
// a NUL byte appear immediately after the last byte of the content.  This terminator byte is not
// counted in the string's size.
//
// To allocate a String, you must call kj::heapString().  We do not implement implicit copying to
// the heap because this hides potential inefficiency from the developer.

class String {
public:
  String() = default;
  inline String(decltype(nullptr)): content(nullptr) {}
  inline String(char* value, size_t size, const ArrayDisposer& disposer);
  // Does not copy.  `size` does not include NUL terminator, but `value` must be NUL-terminated.
  inline explicit String(Array<char> buffer);
  // Does not copy.  Requires `buffer` ends with `\0`.

  inline operator ArrayPtr<char>() KJ_LIFETIMEBOUND;
  inline operator ArrayPtr<const char>() const KJ_LIFETIMEBOUND;
  inline ArrayPtr<char> asArray() KJ_LIFETIMEBOUND;
  inline ArrayPtr<const char> asArray() const KJ_LIFETIMEBOUND;
  inline ArrayPtr<byte> asBytes() KJ_LIFETIMEBOUND { return asArray().asBytes(); }
  inline ArrayPtr<const byte> asBytes() const KJ_LIFETIMEBOUND { return asArray().asBytes(); }
  // Result does not include NUL terminator.

  inline StringPtr asPtr() const KJ_LIFETIMEBOUND {
    // Convenience operator to return a StringPtr.
    return StringPtr{*this};
  }

  inline Array<char> releaseArray() { return kj::mv(content); }
  // Disowns the backing array (which includes the NUL terminator) and returns it. The String value
  // is clobbered (as if moved away).

  inline const char* cStr() const KJ_LIFETIMEBOUND;

  inline size_t size() const;
  // Result does not include NUL terminator.

  inline char operator[](size_t index) const;
  inline char& operator[](size_t index) KJ_LIFETIMEBOUND;

  inline char* begin() KJ_LIFETIMEBOUND;
  inline char* end() KJ_LIFETIMEBOUND;
  inline const char* begin() const KJ_LIFETIMEBOUND;
  inline const char* end() const KJ_LIFETIMEBOUND;

  inline bool operator==(decltype(nullptr)) const { return content.size() <= 1; }
  inline bool operator!=(decltype(nullptr)) const { return content.size() > 1; }

  inline bool operator==(const StringPtr& other) const { return StringPtr(*this) == other; }
#if !__cpp_impl_three_way_comparison
  inline bool operator!=(const StringPtr& other) const { return StringPtr(*this) != other; }
#endif
  inline bool operator< (const StringPtr& other) const { return StringPtr(*this) <  other; }
  inline bool operator> (const StringPtr& other) const { return StringPtr(*this) >  other; }
  inline bool operator<=(const StringPtr& other) const { return StringPtr(*this) <= other; }
  inline bool operator>=(const StringPtr& other) const { return StringPtr(*this) >= other; }

  inline bool operator==(const String& other) const { return StringPtr(*this) == StringPtr(other); }
#if !__cpp_impl_three_way_comparison
  inline bool operator!=(const String& other) const { return StringPtr(*this) != StringPtr(other); }
#endif
  inline bool operator< (const String& other) const { return StringPtr(*this) <  StringPtr(other); }
  inline bool operator> (const String& other) const { return StringPtr(*this) >  StringPtr(other); }
  inline bool operator<=(const String& other) const { return StringPtr(*this) <= StringPtr(other); }
  inline bool operator>=(const String& other) const { return StringPtr(*this) >= StringPtr(other); }
  // Note that if we don't overload for `const String&` specifically, then C++20 will decide that
  // comparisons between two strings are ambiguous. (Clang turns this into a warning,
  // -Wambiguous-reversed-operator, due to the stupidity...)

  inline bool operator==(const ConstString& other) const { return StringPtr(*this) == StringPtr(other); }
#if !__cpp_impl_three_way_comparison
  inline bool operator!=(const ConstString& other) const { return StringPtr(*this) != StringPtr(other); }
#endif
  inline bool operator< (const ConstString& other) const { return StringPtr(*this) <  StringPtr(other); }
  inline bool operator> (const ConstString& other) const { return StringPtr(*this) >  StringPtr(other); }
  inline bool operator<=(const ConstString& other) const { return StringPtr(*this) <= StringPtr(other); }
  inline bool operator>=(const ConstString& other) const { return StringPtr(*this) >= StringPtr(other); }

  inline bool startsWith(const StringPtr& other) const { return asArray().startsWith(other);}
  inline bool endsWith(const StringPtr& other) const { return asArray().endsWith(other); }

  inline StringPtr slice(size_t start) const KJ_LIFETIMEBOUND {
    return StringPtr(*this).slice(start);
  }
  inline ArrayPtr<const char> slice(size_t start, size_t end) const KJ_LIFETIMEBOUND {
    return StringPtr(*this).slice(start, end);
  }

  inline Maybe<size_t> findFirst(char c) const { return asArray().findFirst(c); }
  inline Maybe<size_t> findLast(char c) const { return asArray().findLast(c); }

  template <typename T>
  T parseAs() const { return StringPtr(*this).parseAs<T>(); }
  // Parse as number

  template <typename T>
  Maybe<T> tryParseAs() const { return StringPtr(*this).tryParseAs<T>(); }

private:
  Array<char> content;
};

// =======================================================================================
// ConstString -- Same as String, but the backing buffer is const.
//
// This has the useful property that it can reference a string literal without allocating
// a copy. Any String can also convert (by move) to ConstString, transferring ownership of
// the buffer.

class ConstString {
public:
  ConstString() = default;
  inline ConstString(decltype(nullptr)): content(nullptr) {}
  inline ConstString(const char* value, size_t size, const ArrayDisposer& disposer);
  // Does not copy.  `size` does not include NUL terminator, but `value` must be NUL-terminated.
  inline explicit ConstString(Array<const char> buffer);
  // Does not copy.  Requires `buffer` ends with `\0`.
  inline explicit ConstString(String&& string): content(string.releaseArray()) {}
  // Does not copy. Ownership is transfered.

  inline operator ArrayPtr<const char>() const KJ_LIFETIMEBOUND;
  inline ArrayPtr<const char> asArray() const KJ_LIFETIMEBOUND;
  inline ArrayPtr<const byte> asBytes() const KJ_LIFETIMEBOUND { return asArray().asBytes(); }
  // Result does not include NUL terminator.

  inline StringPtr asPtr() const KJ_LIFETIMEBOUND {
    // Convenience operator to return a StringPtr.
    return StringPtr{*this};
  }

  inline Array<const char> releaseArray() { return kj::mv(content); }
  // Disowns the backing array (which includes the NUL terminator) and returns it. The ConstString value
  // is clobbered (as if moved away).

  inline const char* cStr() const KJ_LIFETIMEBOUND;

  inline size_t size() const;
  // Result does not include NUL terminator.

  inline char operator[](size_t index) const;
  inline char& operator[](size_t index) KJ_LIFETIMEBOUND;

  inline const char* begin() const KJ_LIFETIMEBOUND;
  inline const char* end() const KJ_LIFETIMEBOUND;

  inline bool operator==(decltype(nullptr)) const { return content.size() <= 1; }
  inline bool operator!=(decltype(nullptr)) const { return content.size() > 1; }

  inline bool operator==(const StringPtr& other) const { return StringPtr(*this) == other; }
#if !__cpp_impl_three_way_comparison
  inline bool operator!=(const StringPtr& other) const { return StringPtr(*this) != other; }
#endif
  inline bool operator< (const StringPtr& other) const { return StringPtr(*this) <  other; }
  inline bool operator> (const StringPtr& other) const { return StringPtr(*this) >  other; }
  inline bool operator<=(const StringPtr& other) const { return StringPtr(*this) <= other; }
  inline bool operator>=(const StringPtr& other) const { return StringPtr(*this) >= other; }

  inline bool operator==(const String& other) const { return StringPtr(*this) == StringPtr(other); }
#if !__cpp_impl_three_way_comparison
  inline bool operator!=(const String& other) const { return StringPtr(*this) != StringPtr(other); }
#endif
  inline bool operator< (const String& other) const { return StringPtr(*this) <  StringPtr(other); }
  inline bool operator> (const String& other) const { return StringPtr(*this) >  StringPtr(other); }
  inline bool operator<=(const String& other) const { return StringPtr(*this) <= StringPtr(other); }
  inline bool operator>=(const String& other) const { return StringPtr(*this) >= StringPtr(other); }

  inline bool operator==(const ConstString& other) const { return StringPtr(*this) == StringPtr(other); }
#if !__cpp_impl_three_way_comparison
  inline bool operator!=(const ConstString& other) const { return StringPtr(*this) != StringPtr(other); }
#endif
  inline bool operator< (const ConstString& other) const { return StringPtr(*this) <  StringPtr(other); }
  inline bool operator> (const ConstString& other) const { return StringPtr(*this) >  StringPtr(other); }
  inline bool operator<=(const ConstString& other) const { return StringPtr(*this) <= StringPtr(other); }
  inline bool operator>=(const ConstString& other) const { return StringPtr(*this) >= StringPtr(other); }
  // Note that if we don't overload for `const ConstString&` specifically, then C++20 will decide that
  // comparisons between two strings are ambiguous. (Clang turns this into a warning,
  // -Wambiguous-reversed-operator, due to the stupidity...)

  inline bool startsWith(const StringPtr& other) const { return asArray().startsWith(other);}
  inline bool endsWith(const StringPtr& other) const { return asArray().endsWith(other); }

  inline StringPtr slice(size_t start) const KJ_LIFETIMEBOUND {
    return StringPtr(*this).slice(start);
  }
  inline ArrayPtr<const char> slice(size_t start, size_t end) const KJ_LIFETIMEBOUND {
    return StringPtr(*this).slice(start, end);
  }

  inline Maybe<size_t> findFirst(char c) const { return asArray().findFirst(c); }
  inline Maybe<size_t> findLast(char c) const { return asArray().findLast(c); }

  template <typename T>
  T parseAs() const { return StringPtr(*this).parseAs<T>(); }
  // Parse as number

  template <typename T>
  Maybe<T> tryParseAs() const { return StringPtr(*this).tryParseAs<T>(); }

private:
  Array<const char> content;
};

#if !__cpp_impl_three_way_comparison
inline bool operator==(const char* a, const String& b) { return b == a; }
inline bool operator!=(const char* a, const String& b) { return b != a; }
#endif

String heapString(size_t size);
// Allocate a String of the given size on the heap, not including NUL terminator.  The NUL
// terminator will be initialized automatically but the rest of the content is not initialized.

String heapString(const char* value);
String heapString(const char* value, size_t size);
String heapString(StringPtr value);
String heapString(const String& value);
String heapString(ArrayPtr<const char> value);
// Allocates a copy of the given value on the heap.

// =======================================================================================
// Magic str() function which transforms parameters to text and concatenates them into one big
// String.

namespace _ {  // private

inline size_t sum(std::initializer_list<size_t> nums) {
  size_t result = 0;
  for (auto num: nums) {
    result += num;
  }
  return result;
}

inline char* fill(char* ptr) { return ptr; }
inline char* fillLimited(char* ptr, char* limit) { return ptr; }

template <typename... Rest>
char* fill(char* __restrict__ target, const StringTree& first, Rest&&... rest);
template <typename... Rest>
char* fillLimited(char* __restrict__ target, char* limit, const StringTree& first, Rest&&... rest);
// Make str() work with stringifiers that return StringTree by patching fill().
//
// Defined in string-tree.h.

template <typename First, typename... Rest>
char* fill(char* __restrict__ target, const First& first, Rest&&... rest) {
  auto i = first.begin();
  auto end = first.end();
  while (i != end) {
    *target++ = *i++;
  }
  return fill(target, kj::fwd<Rest>(rest)...);
}

template <typename... Params>
String concat(Params&&... params) {
  // Concatenate a bunch of containers into a single Array.  The containers can be anything that
  // is iterable and whose elements can be converted to `char`.

  String result = heapString(sum({params.size()...}));
  fill(result.begin(), kj::fwd<Params>(params)...);
  return result;
}

inline String concat(String&& arr) {
  return kj::mv(arr);
}

template <typename First, typename... Rest>
char* fillLimited(char* __restrict__ target, char* limit, const First& first, Rest&&... rest) {
  auto i = first.begin();
  auto end = first.end();
  while (i != end) {
    if (target == limit) return target;
    *target++ = *i++;
  }
  return fillLimited(target, limit, kj::fwd<Rest>(rest)...);
}

template <typename T>
class Delimited;
// Delimits a sequence of type T with a string delimiter. Implements kj::delimited().

template <typename T, typename... Rest>
char* fill(char* __restrict__ target, Delimited<T>&& first, Rest&&... rest);
template <typename T, typename... Rest>
char* fillLimited(char* __restrict__ target, char* limit, Delimited<T>&& first,Rest&&... rest);
template <typename T, typename... Rest>
char* fill(char* __restrict__ target, Delimited<T>& first, Rest&&... rest);
template <typename T, typename... Rest>
char* fillLimited(char* __restrict__ target, char* limit, Delimited<T>& first,Rest&&... rest);
// As with StringTree, we special-case Delimited<T>.

struct Stringifier {
  // This is a dummy type with only one instance: STR (below).  To make an arbitrary type
  // stringifiable, define `operator*(Stringifier, T)` to return an iterable container of `char`.
  // The container type must have a `size()` method.  Be sure to declare the operator in the same
  // namespace as `T` **or** in the global scope.
  //
  // A more usual way to accomplish what we're doing here would be to require that you define
  // a function like `toString(T)` and then rely on argument-dependent lookup.  However, this has
  // the problem that it pollutes other people's namespaces and even the global namespace.  For
  // example, some other project may already have functions called `toString` which do something
  // different.  Declaring `operator*` with `Stringifier` as the left operand cannot conflict with
  // anything.

  inline ArrayPtr<const char> operator*(ArrayPtr<const char> s) const { return s; }
  inline ArrayPtr<const char> operator*(ArrayPtr<char> s) const { return s; }
  inline ArrayPtr<const char> operator*(const Array<const char>& s) const KJ_LIFETIMEBOUND {
    return s;
  }
  inline ArrayPtr<const char> operator*(const Array<char>& s) const KJ_LIFETIMEBOUND { return s; }
  template<size_t n>
  inline ArrayPtr<const char> operator*(const CappedArray<char, n>& s) const KJ_LIFETIMEBOUND {
    return s;
  }
  template<size_t n>
  inline ArrayPtr<const char> operator*(const FixedArray<char, n>& s) const KJ_LIFETIMEBOUND {
    return s;
  }
  inline ArrayPtr<const char> operator*(const char* s) const KJ_LIFETIMEBOUND {
    return arrayPtr(s, strlen(s));
  }
#if __cpp_char8_t
  inline ArrayPtr<const char> operator*(const char8_t* s) const KJ_LIFETIMEBOUND {
    return operator*(reinterpret_cast<const char*>(s));
  }
#endif
  inline ArrayPtr<const char> operator*(const String& s) const KJ_LIFETIMEBOUND {
    return s.asArray();
  }
  inline ArrayPtr<const char> operator*(const StringPtr& s) const { return s.asArray(); }
  inline ArrayPtr<const char> operator*(const ConstString& s) const { return s.asArray(); }

  inline Range<char> operator*(const Range<char>& r) const { return r; }
  inline Repeat<char> operator*(const Repeat<char>& r) const { return r; }

  inline FixedArray<char, 1> operator*(char c) const {
    FixedArray<char, 1> result;
    result[0] = c;
    return result;
  }

  StringPtr operator*(decltype(nullptr)) const;
  StringPtr operator*(bool b) const;

  CappedArray<char, 5> operator*(signed char i) const;
  CappedArray<char, 5> operator*(unsigned char i) const;
  CappedArray<char, sizeof(short) * 3 + 2> operator*(short i) const;
  CappedArray<char, sizeof(unsigned short) * 3 + 2> operator*(unsigned short i) const;
  CappedArray<char, sizeof(int) * 3 + 2> operator*(int i) const;
  CappedArray<char, sizeof(unsigned int) * 3 + 2> operator*(unsigned int i) const;
  CappedArray<char, sizeof(long) * 3 + 2> operator*(long i) const;
  CappedArray<char, sizeof(unsigned long) * 3 + 2> operator*(unsigned long i) const;
  CappedArray<char, sizeof(long long) * 3 + 2> operator*(long long i) const;
  CappedArray<char, sizeof(unsigned long long) * 3 + 2> operator*(unsigned long long i) const;
  CappedArray<char, 24> operator*(float f) const;
  CappedArray<char, 32> operator*(double f) const;
  CappedArray<char, sizeof(const void*) * 2 + 1> operator*(const void* s) const;

#if KJ_COMPILER_SUPPORTS_STL_STRING_INTEROP  // supports expression SFINAE?
  template <typename T, typename Result = decltype(instance<T>().toString())>
  inline Result operator*(T&& value) const { return kj::fwd<T>(value).toString(); }
#endif
};
static KJ_CONSTEXPR(const) Stringifier STR = Stringifier();

}  // namespace _ (private)

template <typename T>
auto toCharSequence(T&& value) -> decltype(_::STR * kj::fwd<T>(value)) {
  // Returns an iterable of chars that represent a textual representation of the value, suitable
  // for debugging.
  //
  // Most users should use str() instead, but toCharSequence() may occasionally be useful to avoid
  // heap allocation overhead that str() implies.
  //
  // To specialize this function for your type, see KJ_STRINGIFY.

  return _::STR * kj::fwd<T>(value);
}

CappedArray<char, sizeof(unsigned char) * 2 + 1> hex(unsigned char i);
CappedArray<char, sizeof(unsigned short) * 2 + 1> hex(unsigned short i);
CappedArray<char, sizeof(unsigned int) * 2 + 1> hex(unsigned int i);
CappedArray<char, sizeof(unsigned long) * 2 + 1> hex(unsigned long i);
CappedArray<char, sizeof(unsigned long long) * 2 + 1> hex(unsigned long long i);

template <typename... Params>
String str(Params&&... params) {
  // Magic function which builds a string from a bunch of arbitrary values.  Example:
  //     str(1, " / ", 2, " = ", 0.5)
  // returns:
  //     "1 / 2 = 0.5"
  // To teach `str` how to stringify a type, see `Stringifier`.

  return _::concat(toCharSequence(kj::fwd<Params>(params))...);
}

inline String str(String&& s) { return mv(s); }
// Overload to prevent redundant allocation.

template <typename T>
_::Delimited<T> delimited(T&& arr, kj::StringPtr delim);
// Use to stringify an array.

template <typename T>
String strArray(T&& arr, const char* delim) {
  size_t delimLen = strlen(delim);
  KJ_STACK_ARRAY(decltype(_::STR * arr[0]), pieces, kj::size(arr), 8, 32);
  size_t size = 0;
  for (size_t i = 0; i < kj::size(arr); i++) {
    if (i > 0) size += delimLen;
    pieces[i] = _::STR * arr[i];
    size += pieces[i].size();
  }

  String result = heapString(size);
  char* pos = result.begin();
  for (size_t i = 0; i < kj::size(arr); i++) {
    if (i > 0) {
      memcpy(pos, delim, delimLen);
      pos += delimLen;
    }
    pos = _::fill(pos, pieces[i]);
  }
  return result;
}

template <typename... Params>
StringPtr strPreallocated(ArrayPtr<char> buffer, Params&&... params) {
  // Like str() but writes into a preallocated buffer. If the buffer is not long enough, the result
  // is truncated (but still NUL-terminated).
  //
  // This can be used like:
  //
  //     char buffer[256];
  //     StringPtr text = strPreallocated(buffer, params...);
  //
  // This is useful for optimization. It can also potentially be used safely in async signal
  // handlers. HOWEVER, to use in an async signal handler, all of the stringifiers for the inputs
  // must also be signal-safe. KJ guarantees signal safety when stringifying any built-in integer
  // type (but NOT floating-points), basic char/byte sequences (ArrayPtr<byte>, String, etc.), as
  // well as Array<T> as long as T can also be stringified safely. To safely stringify a delimited
  // array, you must use kj::delimited(arr, delim) rather than the deprecated
  // kj::strArray(arr, delim).

  char* end = _::fillLimited(buffer.begin(), buffer.end() - 1,
      toCharSequence(kj::fwd<Params>(params))...);
  *end = '\0';
  return StringPtr(buffer.begin(), end);
}

template <typename T, typename = decltype(toCharSequence(kj::instance<T&>()))>
inline _::Delimited<ArrayPtr<T>> operator*(const _::Stringifier&, ArrayPtr<T> arr) {
  return _::Delimited<ArrayPtr<T>>(arr, ", ");
}

template <typename T, typename = decltype(toCharSequence(kj::instance<const T&>()))>
inline _::Delimited<ArrayPtr<const T>> operator*(const _::Stringifier&, const Array<T>& arr) {
  return _::Delimited<ArrayPtr<const T>>(arr, ", ");
}

#define KJ_STRINGIFY(...) operator*(::kj::_::Stringifier, __VA_ARGS__)
// Defines a stringifier for a custom type.  Example:
//
//    class Foo {...};
//    inline StringPtr KJ_STRINGIFY(const Foo& foo) { return foo.name(); }
//      // or perhaps
//    inline String KJ_STRINGIFY(const Foo& foo) { return kj::str(foo.fld1(), ",", foo.fld2()); }
//
// This allows Foo to be passed to str().
//
// The function should be declared either in the same namespace as the target type or in the global
// namespace.  It can return any type which is an iterable container of chars.

// =======================================================================================
// Inline implementation details.

inline StringPtr::StringPtr(const String& value): content(value.cStr(), value.size() + 1) {}
inline StringPtr::StringPtr(const ConstString& value): content(value.cStr(), value.size() + 1) {}

inline constexpr StringPtr::operator ArrayPtr<const char>() const {
  return ArrayPtr<const char>(content.begin(), content.size() - 1);
}

inline constexpr ArrayPtr<const char> StringPtr::asArray() const {
  return ArrayPtr<const char>(content.begin(), content.size() - 1);
}

inline bool StringPtr::operator==(const StringPtr& other) const {
  return content.size() == other.content.size() &&
      memcmp(content.begin(), other.content.begin(), content.size() - 1) == 0;
}

inline bool StringPtr::operator<(const StringPtr& other) const {
  bool shorter = content.size() < other.content.size();
  int cmp = memcmp(content.begin(), other.content.begin(),
                   shorter ? content.size() : other.content.size());
  return cmp < 0 || (cmp == 0 && shorter);
}

inline StringPtr StringPtr::slice(size_t start) const {
  return StringPtr(content.slice(start, content.size()));
}
inline ArrayPtr<const char> StringPtr::slice(size_t start, size_t end) const {
  return content.slice(start, end);
}

inline LiteralStringConst::operator ConstString() const {
  return ConstString(begin(), size(), NullArrayDisposer::instance);
}

inline ConstString StringPtr::attach() const {
  // This is meant as a roundabout way to make a ConstString from a StringPtr
  return ConstString(begin(), size(), NullArrayDisposer::instance);
}

template <typename... Attachments>
inline ConstString StringPtr::attach(Attachments&&... attachments) const {
  return ConstString { .content = content.attach(kj::fwd<Attachments>(attachments)...) };
}

inline String::operator ArrayPtr<char>() {
  return content == nullptr ? ArrayPtr<char>(nullptr) : content.slice(0, content.size() - 1);
}
inline String::operator ArrayPtr<const char>() const {
  return content == nullptr ? ArrayPtr<const char>(nullptr) : content.slice(0, content.size() - 1);
}
inline ConstString::operator ArrayPtr<const char>() const {
  return content == nullptr ? ArrayPtr<const char>(nullptr) : content.slice(0, content.size() - 1);
}

inline ArrayPtr<char> String::asArray() {
  return content == nullptr ? ArrayPtr<char>(nullptr) : content.slice(0, content.size() - 1);
}
inline ArrayPtr<const char> String::asArray() const {
  return content == nullptr ? ArrayPtr<const char>(nullptr) : content.slice(0, content.size() - 1);
}
inline ArrayPtr<const char> ConstString::asArray() const {
  return content == nullptr ? ArrayPtr<const char>(nullptr) : content.slice(0, content.size() - 1);
}

inline const char* String::cStr() const { return content == nullptr ? "" : content.begin(); }
inline const char* ConstString::cStr() const { return content == nullptr ? "" : content.begin(); }

inline size_t String::size() const { return content == nullptr ? 0 : content.size() - 1; }
inline size_t ConstString::size() const { return content == nullptr ? 0 : content.size() - 1; }

inline char String::operator[](size_t index) const { return content[index]; }
inline char& String::operator[](size_t index) { return content[index]; }
inline char ConstString::operator[](size_t index) const { return content[index]; }

inline char* String::begin() { return content == nullptr ? nullptr : content.begin(); }
inline char* String::end() { return content == nullptr ? nullptr : content.end() - 1; }
inline const char* String::begin() const { return content == nullptr ? nullptr : content.begin(); }
inline const char* String::end() const { return content == nullptr ? nullptr : content.end() - 1; }
inline const char* ConstString::begin() const { return content == nullptr ? nullptr : content.begin(); }
inline const char* ConstString::end() const { return content == nullptr ? nullptr : content.end() - 1; }

inline String::String(char* value, size_t size, const ArrayDisposer& disposer)
    : content(value, size + 1, disposer) {
  KJ_IREQUIRE(value[size] == '\0', "String must be NUL-terminated.");
}
inline ConstString::ConstString(const char* value, size_t size, const ArrayDisposer& disposer)
    : content(value, size + 1, disposer) {
  KJ_IREQUIRE(value[size] == '\0', "String must be NUL-terminated.");
}

inline String::String(Array<char> buffer): content(kj::mv(buffer)) {
  KJ_IREQUIRE(content.size() > 0 && content.back() == '\0', "String must be NUL-terminated.");
}
inline ConstString::ConstString(Array<const char> buffer): content(kj::mv(buffer)) {
  KJ_IREQUIRE(content.size() > 0 && content.back() == '\0', "String must be NUL-terminated.");
}

inline String heapString(const char* value) {
  return heapString(value, strlen(value));
}
inline String heapString(StringPtr value) {
  return heapString(value.begin(), value.size());
}
inline String heapString(const String& value) {
  return heapString(value.begin(), value.size());
}
inline String heapString(ArrayPtr<const char> value) {
  return heapString(value.begin(), value.size());
}

namespace _ {  // private

template <typename T>
class Delimited {
public:
  Delimited(T array, kj::StringPtr delimiter)
      : array(kj::fwd<T>(array)), delimiter(delimiter) {}

  // TODO(someday): In theory we should support iteration as a character sequence, but the iterator
  //   will be pretty complicated.

  size_t size() {
    ensureStringifiedInitialized();

    size_t result = 0;
    bool first = true;
    for (auto& e: stringified) {
      if (first) {
        first = false;
      } else {
        result += delimiter.size();
      }
      result += e.size();
    }
    return result;
  }

  char* flattenTo(char* __restrict__ target) {
    ensureStringifiedInitialized();

    bool first = true;
    for (auto& elem: stringified) {
      if (first) {
        first = false;
      } else {
        target = fill(target, delimiter);
      }
      target = fill(target, elem);
    }
    return target;
  }

  char* flattenTo(char* __restrict__ target, char* limit) {
    // This is called in the strPreallocated(). We want to avoid allocation. size() will not have
    // been called in this case, so hopefully `stringified` is still uninitialized. We will
    // stringify each item and immediately use it.
    bool first = true;
    for (auto&& elem: array) {
      if (target == limit) return target;
      if (first) {
        first = false;
      } else {
        target = fillLimited(target, limit, delimiter);
      }
      target = fillLimited(target, limit, kj::toCharSequence(elem));
    }
    return target;
  }

private:
  typedef decltype(toCharSequence(*instance<T>().begin())) StringifiedItem;
  T array;
  kj::StringPtr delimiter;
  Array<StringifiedItem> stringified;

  void ensureStringifiedInitialized() {
    if (array.size() > 0 && stringified.size() == 0) {
      stringified = KJ_MAP(e, array) { return toCharSequence(e); };
    }
  }
};

template <typename T, typename... Rest>
char* fill(char* __restrict__ target, Delimited<T>&& first, Rest&&... rest) {
  target = first.flattenTo(target);
  return fill(target, kj::fwd<Rest>(rest)...);
}
template <typename T, typename... Rest>
char* fillLimited(char* __restrict__ target, char* limit, Delimited<T>&& first, Rest&&... rest) {
  target = first.flattenTo(target, limit);
  return fillLimited(target, limit, kj::fwd<Rest>(rest)...);
}
template <typename T, typename... Rest>
char* fill(char* __restrict__ target, Delimited<T>& first, Rest&&... rest) {
  target = first.flattenTo(target);
  return fill(target, kj::fwd<Rest>(rest)...);
}
template <typename T, typename... Rest>
char* fillLimited(char* __restrict__ target, char* limit, Delimited<T>& first, Rest&&... rest) {
  target = first.flattenTo(target, limit);
  return fillLimited(target, limit, kj::fwd<Rest>(rest)...);
}

template <typename T>
inline Delimited<T>&& KJ_STRINGIFY(Delimited<T>&& delimited) { return kj::mv(delimited); }
template <typename T>
inline const Delimited<T>& KJ_STRINGIFY(const Delimited<T>& delimited) { return delimited; }

}  // namespace _ (private)

template <typename T>
_::Delimited<T> delimited(T&& arr, kj::StringPtr delim) {
  return _::Delimited<T>(kj::fwd<T>(arr), delim);
}

}  // namespace kj

constexpr kj::StringPtr operator "" _kj(const char* str, size_t n) {
  return kj::StringPtr(kj::ArrayPtr<const char>(str, n + 1));
};

constexpr kj::LiteralStringConst operator "" _kjc(const char* str, size_t n) {
  return kj::LiteralStringConst(kj::ArrayPtr<const char>(str, n + 1));
};

KJ_END_HEADER
