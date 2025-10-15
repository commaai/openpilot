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

// This file contains parsers useful for character stream inputs, including parsers to parse
// common kinds of tokens like identifiers, numbers, and quoted strings.

#pragma once

#include "common.h"
#include "../string.h"
#include <inttypes.h>

KJ_BEGIN_HEADER

namespace kj {
namespace parse {

// =======================================================================================
// Exact char/string.

class ExactString_ {
public:
  constexpr inline ExactString_(const char* str): str(str) {}

  template <typename Input>
  Maybe<Tuple<>> operator()(Input& input) const {
    const char* ptr = str;

    while (*ptr != '\0') {
      if (input.atEnd() || input.current() != *ptr) return nullptr;
      input.next();
      ++ptr;
    }

    return Tuple<>();
  }

private:
  const char* str;
};

constexpr inline ExactString_ exactString(const char* str) {
  return ExactString_(str);
}

template <char c>
constexpr ExactlyConst_<char, c> exactChar() {
  // Returns a parser that matches exactly the character given by the template argument (returning
  // no result).
  return ExactlyConst_<char, c>();
}

// =======================================================================================
// Char ranges / sets

class CharGroup_ {
public:
  constexpr inline CharGroup_(): bits{0, 0, 0, 0} {}

  constexpr inline CharGroup_ orRange(unsigned char first, unsigned char last) const {
    return CharGroup_(bits[0] | (oneBits(last +   1) & ~oneBits(first      )),
                      bits[1] | (oneBits(last -  63) & ~oneBits(first -  64)),
                      bits[2] | (oneBits(last - 127) & ~oneBits(first - 128)),
                      bits[3] | (oneBits(last - 191) & ~oneBits(first - 192)));
  }

  constexpr inline CharGroup_ orAny(const char* chars) const {
    return *chars == 0 ? *this : orChar(*chars).orAny(chars + 1);
  }

  constexpr inline CharGroup_ orChar(unsigned char c) const {
    return CharGroup_(bits[0] | bit(c),
                      bits[1] | bit(c - 64),
                      bits[2] | bit(c - 128),
                      bits[3] | bit(c - 256));
  }

  constexpr inline CharGroup_ orGroup(CharGroup_ other) const {
    return CharGroup_(bits[0] | other.bits[0],
                      bits[1] | other.bits[1],
                      bits[2] | other.bits[2],
                      bits[3] | other.bits[3]);
  }

  constexpr inline CharGroup_ invert() const {
    return CharGroup_(~bits[0], ~bits[1], ~bits[2], ~bits[3]);
  }

  constexpr inline bool contains(unsigned char c) const {
    return (bits[c / 64] & (1ll << (c % 64))) != 0;
  }

  inline bool containsAll(ArrayPtr<const char> text) const {
    for (char c: text) {
      if (!contains(c)) return false;
    }
    return true;
  }

  template <typename Input>
  Maybe<char> operator()(Input& input) const {
    if (input.atEnd()) return nullptr;
    unsigned char c = input.current();
    if (contains(c)) {
      input.next();
      return c;
    } else {
      return nullptr;
    }
  }

private:
  typedef unsigned long long Bits64;

  constexpr inline CharGroup_(Bits64 a, Bits64 b, Bits64 c, Bits64 d): bits{a, b, c, d} {}
  Bits64 bits[4];

  static constexpr inline Bits64 oneBits(int count) {
    return count <= 0 ? 0ll : count >= 64 ? -1ll : ((1ll << count) - 1);
  }
  static constexpr inline Bits64 bit(int index) {
    return index < 0 ? 0 : index >= 64 ? 0 : (1ll << index);
  }
};

constexpr inline CharGroup_ charRange(char first, char last) {
  // Create a parser which accepts any character in the range from `first` to `last`, inclusive.
  // For example: `charRange('a', 'z')` matches all lower-case letters.  The parser's result is the
  // character matched.
  //
  // The returned object has methods which can be used to match more characters.  The following
  // produces a parser which accepts any letter as well as '_', '+', '-', and '.'.
  //
  //     charRange('a', 'z').orRange('A', 'Z').orChar('_').orAny("+-.")
  //
  // You can also use `.invert()` to match the opposite set of characters.

  return CharGroup_().orRange(first, last);
}

#if _MSC_VER && !defined(__clang__)
#define anyOfChars(chars) CharGroup_().orAny(chars)
// TODO(msvc): MSVC ICEs on the proper definition of `anyOfChars()`, which in turn prevents us from
//   building the compiler or schema parser. We don't know why this happens, but Harris found that
//   this horrible, horrible hack makes things work. This is awful, but it's better than nothing.
//   Hopefully, MSVC will get fixed soon and we'll be able to remove this.
#else
constexpr inline CharGroup_ anyOfChars(const char* chars) {
  // Returns a parser that accepts any of the characters in the given string (which should usually
  // be a literal).  The returned parser is of the same type as returned by `charRange()` -- see
  // that function for more info.

  return CharGroup_().orAny(chars);
}
#endif

// =======================================================================================

namespace _ {  // private

struct ArrayToString {
  inline String operator()(const Array<char>& arr) const {
    return heapString(arr);
  }
};

}  // namespace _ (private)

template <typename SubParser>
constexpr inline auto charsToString(SubParser&& subParser)
    -> decltype(transform(kj::fwd<SubParser>(subParser), _::ArrayToString())) {
  // Wraps a parser that returns Array<char> such that it returns String instead.
  return parse::transform(kj::fwd<SubParser>(subParser), _::ArrayToString());
}

// =======================================================================================
// Basic character classes.

constexpr auto alpha = charRange('a', 'z').orRange('A', 'Z');
constexpr auto digit = charRange('0', '9');
constexpr auto alphaNumeric = alpha.orGroup(digit);
constexpr auto nameStart = alpha.orChar('_');
constexpr auto nameChar = alphaNumeric.orChar('_');
constexpr auto hexDigit = charRange('0', '9').orRange('a', 'f').orRange('A', 'F');
constexpr auto octDigit = charRange('0', '7');
constexpr auto whitespaceChar = anyOfChars(" \f\n\r\t\v");
constexpr auto controlChar = charRange(0, 0x1f).invert().orGroup(whitespaceChar).invert();

constexpr auto whitespace = many(anyOfChars(" \f\n\r\t\v"));

constexpr auto discardWhitespace = discard(many(discard(anyOfChars(" \f\n\r\t\v"))));
// Like discard(whitespace) but avoids some memory allocation.

// =======================================================================================
// Identifiers

namespace _ { // private

struct IdentifierToString {
  inline String operator()(char first, const Array<char>& rest) const {
    if (rest.size() == 0) return heapString(&first, 1);
    String result = heapString(rest.size() + 1);
    result[0] = first;
    memcpy(result.begin() + 1, rest.begin(), rest.size());
    return result;
  }
};

}  // namespace _ (private)

constexpr auto identifier = transform(sequence(nameStart, many(nameChar)), _::IdentifierToString());
// Parses an identifier (e.g. a C variable name).

// =======================================================================================
// Integers

namespace _ {  // private

inline char parseDigit(char c) {
  if (c < 'A') return c - '0';
  if (c < 'a') return c - 'A' + 10;
  return c - 'a' + 10;
}

template <uint base>
struct ParseInteger {
  inline uint64_t operator()(const Array<char>& digits) const {
    return operator()('0', digits);
  }
  uint64_t operator()(char first, const Array<char>& digits) const {
    uint64_t result = parseDigit(first);
    for (char digit: digits) {
      result = result * base + parseDigit(digit);
    }
    return result;
  }
};


}  // namespace _ (private)

constexpr auto integer = sequence(
    oneOf(
      transform(sequence(exactChar<'0'>(), exactChar<'x'>(), oneOrMore(hexDigit)), _::ParseInteger<16>()),
      transform(sequence(exactChar<'0'>(), many(octDigit)), _::ParseInteger<8>()),
      transform(sequence(charRange('1', '9'), many(digit)), _::ParseInteger<10>())),
    notLookingAt(alpha.orAny("_.")));

// =======================================================================================
// Numbers (i.e. floats)

namespace _ {  // private

struct ParseFloat {
  double operator()(const Array<char>& digits,
                    const Maybe<Array<char>>& fraction,
                    const Maybe<Tuple<Maybe<char>, Array<char>>>& exponent) const;
};

}  // namespace _ (private)

constexpr auto number = transform(
    sequence(
        oneOrMore(digit),
        optional(sequence(exactChar<'.'>(), many(digit))),
        optional(sequence(discard(anyOfChars("eE")), optional(anyOfChars("+-")), many(digit))),
        notLookingAt(alpha.orAny("_."))),
    _::ParseFloat());

// =======================================================================================
// Quoted strings

namespace _ {  // private

struct InterpretEscape {
  char operator()(char c) const {
    switch (c) {
      case 'a': return '\a';
      case 'b': return '\b';
      case 'f': return '\f';
      case 'n': return '\n';
      case 'r': return '\r';
      case 't': return '\t';
      case 'v': return '\v';
      default: return c;
    }
  }
};

struct ParseHexEscape {
  inline char operator()(char first, char second) const {
    return (parseDigit(first) << 4) | parseDigit(second);
  }
};

struct ParseHexByte {
  inline byte operator()(char first, char second) const {
    return (parseDigit(first) << 4) | parseDigit(second);
  }
};

struct ParseOctEscape {
  inline char operator()(char first, Maybe<char> second, Maybe<char> third) const {
    char result = first - '0';
    KJ_IF_MAYBE(digit1, second) {
      result = (result << 3) | (*digit1 - '0');
      KJ_IF_MAYBE(digit2, third) {
        result = (result << 3) | (*digit2 - '0');
      }
    }
    return result;
  }
};

}  // namespace _ (private)

constexpr auto escapeSequence =
    sequence(exactChar<'\\'>(), oneOf(
        transform(anyOfChars("abfnrtv'\"\\\?"), _::InterpretEscape()),
        transform(sequence(exactChar<'x'>(), hexDigit, hexDigit), _::ParseHexEscape()),
        transform(sequence(octDigit, optional(octDigit), optional(octDigit)),
                  _::ParseOctEscape())));
// A parser that parses a C-string-style escape sequence (starting with a backslash).  Returns
// a char.

constexpr auto doubleQuotedString = charsToString(sequence(
    exactChar<'\"'>(),
    many(oneOf(anyOfChars("\\\n\"").invert(), escapeSequence)),
    exactChar<'\"'>()));
// Parses a C-style double-quoted string.

constexpr auto singleQuotedString = charsToString(sequence(
    exactChar<'\''>(),
    many(oneOf(anyOfChars("\\\n\'").invert(), escapeSequence)),
    exactChar<'\''>()));
// Parses a C-style single-quoted string.

constexpr auto doubleQuotedHexBinary = sequence(
    exactChar<'0'>(), exactChar<'x'>(), exactChar<'\"'>(),
    oneOrMore(transform(sequence(discardWhitespace, hexDigit, hexDigit), _::ParseHexByte())),
    discardWhitespace,
    exactChar<'\"'>());
// Parses a double-quoted hex binary literal. Returns Array<byte>.

}  // namespace parse
}  // namespace kj

KJ_END_HEADER
