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

#ifndef KJ_STRING_H_
#define KJ_STRING_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include <initializer_list>
#include "array.h"
#include <string.h>

namespace kj {

class StringPtr;
class String;

class StringTree;   // string-tree.h

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
  inline StringPtr(const char* value): content(value, strlen(value) + 1) {}
  inline StringPtr(const char* value, size_t size): content(value, size + 1) {
    KJ_IREQUIRE(value[size] == '\0', "StringPtr must be NUL-terminated.");
  }
  inline StringPtr(const char* begin, const char* end): StringPtr(begin, end - begin) {}
  inline StringPtr(const String& value);

#if KJ_COMPILER_SUPPORTS_STL_STRING_INTEROP
  template <typename T, typename = decltype(instance<T>().c_str())>
  inline StringPtr(const T& t): StringPtr(t.c_str()) {}
  // Allow implicit conversion from any class that has a c_str() method (namely, std::string).
  // We use a template trick to detect std::string in order to avoid including the header for
  // those who don't want it.

  template <typename T, typename = decltype(instance<T>().c_str())>
  inline operator T() const { return cStr(); }
  // Allow implicit conversion to any class that has a c_str() method (namely, std::string).
  // We use a template trick to detect std::string in order to avoid including the header for
  // those who don't want it.
#endif

  inline operator ArrayPtr<const char>() const;
  inline ArrayPtr<const char> asArray() const;
  inline ArrayPtr<const byte> asBytes() const { return asArray().asBytes(); }
  // Result does not include NUL terminator.

  inline const char* cStr() const { return content.begin(); }
  // Returns NUL-terminated string.

  inline size_t size() const { return content.size() - 1; }
  // Result does not include NUL terminator.

  inline char operator[](size_t index) const { return content[index]; }

  inline const char* begin() const { return content.begin(); }
  inline const char* end() const { return content.end() - 1; }

  inline bool operator==(decltype(nullptr)) const { return content.size() <= 1; }
  inline bool operator!=(decltype(nullptr)) const { return content.size() > 1; }

  inline bool operator==(const StringPtr& other) const;
  inline bool operator!=(const StringPtr& other) const { return !(*this == other); }
  inline bool operator< (const StringPtr& other) const;
  inline bool operator> (const StringPtr& other) const { return other < *this; }
  inline bool operator<=(const StringPtr& other) const { return !(other < *this); }
  inline bool operator>=(const StringPtr& other) const { return !(*this < other); }

  inline StringPtr slice(size_t start) const;
  inline ArrayPtr<const char> slice(size_t start, size_t end) const;
  // A string slice is only NUL-terminated if it is a suffix, so slice() has a one-parameter
  // version that assumes end = size().

  inline bool startsWith(const StringPtr& other) const;
  inline bool endsWith(const StringPtr& other) const;

  inline Maybe<size_t> findFirst(char c) const;
  inline Maybe<size_t> findLast(char c) const;

  template <typename T>
  T parseAs() const;
  // Parse string as template number type.
  // Integer numbers prefixed by "0x" and "0X" are parsed in base 16 (like strtoi with base 0).
  // Integer numbers prefixed by "0" are parsed in base 10 (unlike strtoi with base 0).
  // Overflowed integer numbers throw exception.
  // Overflowed floating numbers return inf.

private:
  inline StringPtr(ArrayPtr<const char> content): content(content) {}

  ArrayPtr<const char> content;
};

inline bool operator==(const char* a, const StringPtr& b) { return b == a; }
inline bool operator!=(const char* a, const StringPtr& b) { return b != a; }

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

  inline operator ArrayPtr<char>();
  inline operator ArrayPtr<const char>() const;
  inline ArrayPtr<char> asArray();
  inline ArrayPtr<const char> asArray() const;
  inline ArrayPtr<byte> asBytes() { return asArray().asBytes(); }
  inline ArrayPtr<const byte> asBytes() const { return asArray().asBytes(); }
  // Result does not include NUL terminator.

  inline Array<char> releaseArray() { return kj::mv(content); }
  // Disowns the backing array (which includes the NUL terminator) and returns it. The String value
  // is clobbered (as if moved away).

  inline const char* cStr() const;

  inline size_t size() const;
  // Result does not include NUL terminator.

  inline char operator[](size_t index) const;
  inline char& operator[](size_t index);

  inline char* begin();
  inline char* end();
  inline const char* begin() const;
  inline const char* end() const;

  inline bool operator==(decltype(nullptr)) const { return content.size() <= 1; }
  inline bool operator!=(decltype(nullptr)) const { return content.size() > 1; }

  inline bool operator==(const StringPtr& other) const { return StringPtr(*this) == other; }
  inline bool operator!=(const StringPtr& other) const { return StringPtr(*this) != other; }
  inline bool operator< (const StringPtr& other) const { return StringPtr(*this) <  other; }
  inline bool operator> (const StringPtr& other) const { return StringPtr(*this) >  other; }
  inline bool operator<=(const StringPtr& other) const { return StringPtr(*this) <= other; }
  inline bool operator>=(const StringPtr& other) const { return StringPtr(*this) >= other; }

  inline bool startsWith(const StringPtr& other) const { return StringPtr(*this).startsWith(other);}
  inline bool endsWith(const StringPtr& other) const { return StringPtr(*this).endsWith(other); }

  inline StringPtr slice(size_t start) const { return StringPtr(*this).slice(start); }
  inline ArrayPtr<const char> slice(size_t start, size_t end) const {
    return StringPtr(*this).slice(start, end);
  }

  inline Maybe<size_t> findFirst(char c) const { return StringPtr(*this).findFirst(c); }
  inline Maybe<size_t> findLast(char c) const { return StringPtr(*this).findLast(c); }

  template <typename T>
  T parseAs() const { return StringPtr(*this).parseAs<T>(); }
  // Parse as number

private:
  Array<char> content;
};

inline bool operator==(const char* a, const String& b) { return b == a; }
inline bool operator!=(const char* a, const String& b) { return b != a; }

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

template <typename... Rest>
char* fill(char* __restrict__ target, const StringTree& first, Rest&&... rest);
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
  inline ArrayPtr<const char> operator*(const Array<const char>& s) const { return s; }
  inline ArrayPtr<const char> operator*(const Array<char>& s) const { return s; }
  template<size_t n>
  inline ArrayPtr<const char> operator*(const CappedArray<char, n>& s) const { return s; }
  template<size_t n>
  inline ArrayPtr<const char> operator*(const FixedArray<char, n>& s) const { return s; }
  inline ArrayPtr<const char> operator*(const char* s) const { return arrayPtr(s, strlen(s)); }
  inline ArrayPtr<const char> operator*(const String& s) const { return s.asArray(); }
  inline ArrayPtr<const char> operator*(const StringPtr& s) const { return s.asArray(); }

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
  CappedArray<char, sizeof(const void*) * 3 + 2> operator*(const void* s) const;

  template <typename T>
  String operator*(ArrayPtr<T> arr) const;
  template <typename T>
  String operator*(const Array<T>& arr) const;

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

namespace _ {  // private

template <typename T>
inline String Stringifier::operator*(ArrayPtr<T> arr) const {
  return strArray(arr, ", ");
}

template <typename T>
inline String Stringifier::operator*(const Array<T>& arr) const {
  return strArray(arr, ", ");
}

}  // namespace _ (private)

#define KJ_STRINGIFY(...) operator*(::kj::_::Stringifier, __VA_ARGS__)
// Defines a stringifier for a custom type.  Example:
//
//    class Foo {...};
//    inline StringPtr KJ_STRINGIFY(const Foo& foo) { return foo.name(); }
//
// This allows Foo to be passed to str().
//
// The function should be declared either in the same namespace as the target type or in the global
// namespace.  It can return any type which is an iterable container of chars.

// =======================================================================================
// Inline implementation details.

inline StringPtr::StringPtr(const String& value): content(value.begin(), value.size() + 1) {}

inline StringPtr::operator ArrayPtr<const char>() const {
  return content.slice(0, content.size() - 1);
}

inline ArrayPtr<const char> StringPtr::asArray() const {
  return content.slice(0, content.size() - 1);
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

inline bool StringPtr::startsWith(const StringPtr& other) const {
  return other.content.size() <= content.size() &&
      memcmp(content.begin(), other.content.begin(), other.size()) == 0;
}
inline bool StringPtr::endsWith(const StringPtr& other) const {
  return other.content.size() <= content.size() &&
      memcmp(end() - other.size(), other.content.begin(), other.size()) == 0;
}

inline Maybe<size_t> StringPtr::findFirst(char c) const {
  const char* pos = reinterpret_cast<const char*>(memchr(content.begin(), c, size()));
  if (pos == nullptr) {
    return nullptr;
  } else {
    return pos - content.begin();
  }
}

inline Maybe<size_t> StringPtr::findLast(char c) const {
  for (size_t i = size(); i > 0; --i) {
    if (content[i-1] == c) {
      return i-1;
    }
  }
  return nullptr;
}

inline String::operator ArrayPtr<char>() {
  return content == nullptr ? ArrayPtr<char>(nullptr) : content.slice(0, content.size() - 1);
}
inline String::operator ArrayPtr<const char>() const {
  return content == nullptr ? ArrayPtr<const char>(nullptr) : content.slice(0, content.size() - 1);
}

inline ArrayPtr<char> String::asArray() {
  return content == nullptr ? ArrayPtr<char>(nullptr) : content.slice(0, content.size() - 1);
}
inline ArrayPtr<const char> String::asArray() const {
  return content == nullptr ? ArrayPtr<const char>(nullptr) : content.slice(0, content.size() - 1);
}

inline const char* String::cStr() const { return content == nullptr ? "" : content.begin(); }

inline size_t String::size() const { return content == nullptr ? 0 : content.size() - 1; }

inline char String::operator[](size_t index) const { return content[index]; }
inline char& String::operator[](size_t index) { return content[index]; }

inline char* String::begin() { return content == nullptr ? nullptr : content.begin(); }
inline char* String::end() { return content == nullptr ? nullptr : content.end() - 1; }
inline const char* String::begin() const { return content == nullptr ? nullptr : content.begin(); }
inline const char* String::end() const { return content == nullptr ? nullptr : content.end() - 1; }

inline String::String(char* value, size_t size, const ArrayDisposer& disposer)
    : content(value, size + 1, disposer) {
  KJ_IREQUIRE(value[size] == '\0', "String must be NUL-terminated.");
}

inline String::String(Array<char> buffer): content(kj::mv(buffer)) {
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

}  // namespace kj

#endif  // KJ_STRING_H_
