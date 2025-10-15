// Copyright (c) 2017 Cloudflare, Inc. and contributors
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
// Functions for encoding/decoding bytes and text in common formats, including:
// - UTF-{8,16,32}
// - Hex
// - URI encoding
// - Base64

#include "string.h"

KJ_BEGIN_HEADER

namespace kj {

template <typename ResultType>
struct EncodingResult: public ResultType {
  // Equivalent to ResultType (a String or wide-char array) for all intents and purposes, except
  // that the bool `hadErrors` can be inspected to see if any errors were encountered in the input.
  // Each encoding/decoding function that returns this type will "work around" errors in some way,
  // so an application doesn't strictly have to check for errors. E.g. the Unicode functions
  // replace errors with U+FFFD in the output.
  //
  // Through magic, KJ_IF_MAYBE() and KJ_{REQUIRE,ASSERT}_NONNULL() work on EncodingResult<T>
  // exactly if it were a Maybe<T> that is null in case of errors.

  inline EncodingResult(ResultType&& result, bool hadErrors)
      : ResultType(kj::mv(result)), hadErrors(hadErrors) {}

  const bool hadErrors;
};

template <typename T>
inline auto KJ_STRINGIFY(const EncodingResult<T>& value)
    -> decltype(toCharSequence(implicitCast<const T&>(value))) {
  return toCharSequence(implicitCast<const T&>(value));
}

EncodingResult<Array<char16_t>> encodeUtf16(ArrayPtr<const char> text, bool nulTerminate = false);
EncodingResult<Array<char32_t>> encodeUtf32(ArrayPtr<const char> text, bool nulTerminate = false);
// Convert UTF-8 text (which KJ strings use) to UTF-16 or UTF-32.
//
// If `nulTerminate` is true, an extra NUL character will be added to the end of the output.
//
// The returned arrays are in platform-native endianness (otherwise they wouldn't really be
// char16_t / char32_t).
//
// Note that the KJ Unicode encoding and decoding functions actually implement
// [WTF-8 encoding](http://simonsapin.github.io/wtf-8/), which affects how invalid input is
// handled. See comments on decodeUtf16() for more info.

EncodingResult<String> decodeUtf16(ArrayPtr<const char16_t> utf16);
EncodingResult<String> decodeUtf32(ArrayPtr<const char32_t> utf32);
// Convert UTF-16 or UTF-32 to UTF-8 (which KJ strings use).
//
// The input should NOT include a NUL terminator; any NUL characters in the input array will be
// preserved in the output.
//
// The input must be in platform-native endianness. BOMs are NOT recognized by these functions.
//
// Note that the KJ Unicode encoding and decoding functions actually implement
// [WTF-8 encoding](http://simonsapin.github.io/wtf-8/). This means that if you start with an array
// of char16_t and you pass it through any number of conversions to other Unicode encodings,
// eventually returning it to UTF-16, all the while ignoring `hadErrors`, you will end up with
// exactly the same char16_t array you started with, *even if* the array is not valid UTF-16. This
// is useful because many real-world systems that were designed for UCS-2 (plain 16-bit Unicode)
// and later "upgraded" to UTF-16 do not enforce that their UTF-16 is well-formed. For example,
// file names on Windows NT are encoded using 16-bit characters, without enforcing that the
// character sequence is valid UTF-16. It is important that programs on Windows be able to handle
// such filenames, even if they choose to convert the name to UTF-8 for internal processing.
//
// Specifically, KJ's Unicode handling allows unpaired surrogate code points to round-trip through
// UTF-8 and UTF-32. Unpaired surrogates will be flagged as an error (setting `hadErrors` in the
// result), but will NOT be replaced with the Unicode replacement character as other erroneous
// sequences would be, but rather encoded as an invalid surrogate codepoint in the target encoding.
//
// KJ makes the following guarantees about invalid input:
// - A round trip from UTF-16 to other encodings and back will produce exactly the original input,
//   with every leg of the trip raising the `hadErrors` flag if the original input was not valid.
// - A round trip from UTF-8 or UTF-32 to other encodings and back will either produce exactly
//   the original input, or will have replaced some invalid sequences with the Unicode replacement
//   character, U+FFFD. No code units will ever be removed unless they are replaced with U+FFFD,
//   and no code units will ever be added except to encode U+FFFD. If the original input was not
//   valid, the `hadErrors` flag will be raised on the first leg of the trip, and will also be
//   raised on subsequent legs unless all invalid sequences were replaced with U+FFFD (which, after
//   all, is a valid code point).

EncodingResult<Array<wchar_t>> encodeWideString(
    ArrayPtr<const char> text, bool nulTerminate = false);
EncodingResult<String> decodeWideString(ArrayPtr<const wchar_t> wide);
// Encode / decode strings of wchar_t, aka "wide strings". Unfortunately, different platforms have
// different definitions for wchar_t. For example, on Windows they are 16-bit and encode UTF-16,
// but on Linux they are 32-bit and encode UTF-32. Some platforms even define wchar_t as 8-bit,
// encoding UTF-8 (e.g. BeOS did this).
//
// KJ assumes that wide strings use the UTF encoding that corresponds to the size of wchar_t on
// the target platform. So, these functions are simple aliases for encodeUtf*/decodeUtf*, above
// (or simply make a copy if wchar_t is 8 bits).

String encodeHex(ArrayPtr<const byte> bytes);
EncodingResult<Array<byte>> decodeHex(ArrayPtr<const char> text);
// Encode/decode bytes as hex strings.

String encodeUriComponent(ArrayPtr<const byte> bytes);
String encodeUriComponent(ArrayPtr<const char> bytes);
EncodingResult<String> decodeUriComponent(ArrayPtr<const char> text);
// Encode/decode URI components using % escapes for characters listed as "reserved" in RFC 2396.
// This is the same behavior as JavaScript's `encodeURIComponent()`.
//
// See https://tools.ietf.org/html/rfc2396#section-2.3

String encodeUriFragment(ArrayPtr<const byte> bytes);
String encodeUriFragment(ArrayPtr<const char> bytes);
// Encode URL fragment components using the fragment percent encode set defined by the WHATWG URL
// specification. Use decodeUriComponent() to decode.
//
// Quirk: We also percent-encode the '%' sign itself, because we expect to be called on percent-
//   decoded data. In other words, this function is not idempotent, in contrast to the URL spec.
//
// See https://url.spec.whatwg.org/#fragment-percent-encode-set

String encodeUriPath(ArrayPtr<const byte> bytes);
String encodeUriPath(ArrayPtr<const char> bytes);
// Encode URL path components (not entire paths!) using the path percent encode set defined by the
// WHATWG URL specification. Use decodeUriComponent() to decode.
//
// Quirk: We also percent-encode the '%' sign itself, because we expect to be called on percent-
//   decoded data. In other words, this function is not idempotent, in contrast to the URL spec.
//
// Quirk: This percent-encodes '/' and '\' characters as well, which are not actually in the set
//   defined by the WHATWG URL spec. Since a conforming URL implementation will only ever call this
//   function on individual path components, and never entire paths, augmenting the character set to
//   include these separators allows this function to be used to implement a URL class that stores
//   its path components in percent-decoded form.
//
// See https://url.spec.whatwg.org/#path-percent-encode-set

String encodeUriUserInfo(ArrayPtr<const byte> bytes);
String encodeUriUserInfo(ArrayPtr<const char> bytes);
// Encode URL userinfo components using the userinfo percent encode set defined by the WHATWG URL
// specification. Use decodeUriComponent() to decode.
//
// Quirk: We also percent-encode the '%' sign itself, because we expect to be called on percent-
//   decoded data. In other words, this function is not idempotent, in contrast to the URL spec.
//
// See https://url.spec.whatwg.org/#userinfo-percent-encode-set

String encodeWwwForm(ArrayPtr<const byte> bytes);
String encodeWwwForm(ArrayPtr<const char> bytes);
EncodingResult<String> decodeWwwForm(ArrayPtr<const char> text);
// Encode/decode URI components using % escapes and '+' (for spaces) according to the
// application/x-www-form-urlencoded format defined by the WHATWG URL specification.
//
// Note: Like the fragment, path, and userinfo percent-encoding functions above, this function is
//   not idempotent: we percent-encode '%' signs. However, in this particular case the spec happens
//   to agree with us!
//
// See https://url.spec.whatwg.org/#concept-urlencoded-byte-serializer

struct DecodeUriOptions {
  // Parameter to `decodeBinaryUriComponent()`.

  // This struct is intentionally convertible from bool, in order to maintain backwards
  // compatibility with code written when `decodeBinaryUriComponent()` took a boolean second
  // parameter.
  DecodeUriOptions(bool nulTerminate = false, bool plusToSpace = false)
      : nulTerminate(nulTerminate), plusToSpace(plusToSpace) {}

  bool nulTerminate;
  // Append a terminal NUL byte.

  bool plusToSpace;
  // Convert '+' to ' ' characters before percent decoding. Used to decode
  // application/x-www-form-urlencoded text, such as query strings.
};
EncodingResult<Array<byte>> decodeBinaryUriComponent(
    ArrayPtr<const char> text, DecodeUriOptions options = DecodeUriOptions());
// Decode URI components using % escapes. This is a lower-level interface used to implement both
// `decodeUriComponent()` and `decodeWwwForm()`

String encodeCEscape(ArrayPtr<const byte> bytes);
String encodeCEscape(ArrayPtr<const char> bytes);
EncodingResult<Array<byte>> decodeBinaryCEscape(
    ArrayPtr<const char> text, bool nulTerminate = false);
EncodingResult<String> decodeCEscape(ArrayPtr<const char> text);

String encodeBase64(ArrayPtr<const byte> bytes, bool breakLines = false);
// Encode the given bytes as base64 text. If `breakLines` is true, line breaks will be inserted
// into the output every 72 characters (e.g. for encoding e-mail bodies).

EncodingResult<Array<byte>> decodeBase64(ArrayPtr<const char> text);
// Decode base64 text. This function reports errors required by the WHATWG HTML/Infra specs: see
// https://html.spec.whatwg.org/multipage/webappapis.html#atob for details.

String encodeBase64Url(ArrayPtr<const byte> bytes);
// Encode the given bytes as URL-safe base64 text. (RFC 4648, section 5)

// =======================================================================================
// inline implementation details

namespace _ {  // private

template <typename T>
NullableValue<T> readMaybe(EncodingResult<T>&& value) {
  if (value.hadErrors) {
    return nullptr;
  } else {
    return kj::mv(value);
  }
}

template <typename T>
T* readMaybe(EncodingResult<T>& value) {
  if (value.hadErrors) {
    return nullptr;
  } else {
    return &value;
  }
}

template <typename T>
const T* readMaybe(const EncodingResult<T>& value) {
  if (value.hadErrors) {
    return nullptr;
  } else {
    return &value;
  }
}

String encodeCEscapeImpl(ArrayPtr<const byte> bytes, bool isBinary);

}  // namespace _ (private)

inline String encodeUriComponent(ArrayPtr<const char> text) {
  return encodeUriComponent(text.asBytes());
}
inline EncodingResult<String> decodeUriComponent(ArrayPtr<const char> text) {
  auto result = decodeBinaryUriComponent(text, DecodeUriOptions { /*.nulTerminate=*/true });
  return { String(result.releaseAsChars()), result.hadErrors };
}

inline String encodeUriFragment(ArrayPtr<const char> text) {
  return encodeUriFragment(text.asBytes());
}
inline String encodeUriPath(ArrayPtr<const char> text) {
  return encodeUriPath(text.asBytes());
}
inline String encodeUriUserInfo(ArrayPtr<const char> text) {
  return encodeUriUserInfo(text.asBytes());
}

inline String encodeWwwForm(ArrayPtr<const char> text) {
  return encodeWwwForm(text.asBytes());
}
inline EncodingResult<String> decodeWwwForm(ArrayPtr<const char> text) {
  auto result = decodeBinaryUriComponent(text, DecodeUriOptions { /*.nulTerminate=*/true,
                                                                  /*.plusToSpace=*/true });
  return { String(result.releaseAsChars()), result.hadErrors };
}

inline String encodeCEscape(ArrayPtr<const char> text) {
  return _::encodeCEscapeImpl(text.asBytes(), false);
}

inline String encodeCEscape(ArrayPtr<const byte> bytes) {
  return _::encodeCEscapeImpl(bytes, true);
}

inline EncodingResult<String> decodeCEscape(ArrayPtr<const char> text) {
  auto result = decodeBinaryCEscape(text, true);
  return { String(result.releaseAsChars()), result.hadErrors };
}

// If you pass a string literal to a function taking ArrayPtr<const char>, it'll include the NUL
// termintator, which is surprising. Let's add overloads that avoid that. In practice this probably
// only even matters for encoding-test.c++.

template <size_t s>
inline EncodingResult<Array<char16_t>> encodeUtf16(const char (&text)[s], bool nulTerminate=false) {
  return encodeUtf16(arrayPtr(text, s - 1), nulTerminate);
}
template <size_t s>
inline EncodingResult<Array<char32_t>> encodeUtf32(const char (&text)[s], bool nulTerminate=false) {
  return encodeUtf32(arrayPtr(text, s - 1), nulTerminate);
}
template <size_t s>
inline EncodingResult<Array<wchar_t>> encodeWideString(
    const char (&text)[s], bool nulTerminate=false) {
  return encodeWideString(arrayPtr(text, s - 1), nulTerminate);
}
template <size_t s>
inline EncodingResult<String> decodeUtf16(const char16_t (&utf16)[s]) {
  return decodeUtf16(arrayPtr(utf16, s - 1));
}
template <size_t s>
inline EncodingResult<String> decodeUtf32(const char32_t (&utf32)[s]) {
  return decodeUtf32(arrayPtr(utf32, s - 1));
}
template <size_t s>
inline EncodingResult<String> decodeWideString(const wchar_t (&utf32)[s]) {
  return decodeWideString(arrayPtr(utf32, s - 1));
}
template <size_t s>
inline EncodingResult<Array<byte>> decodeHex(const char (&text)[s]) {
  return decodeHex(arrayPtr(text, s - 1));
}
template <size_t s>
inline String encodeUriComponent(const char (&text)[s]) {
  return encodeUriComponent(arrayPtr(text, s - 1));
}
template <size_t s>
inline Array<byte> decodeBinaryUriComponent(const char (&text)[s]) {
  return decodeBinaryUriComponent(arrayPtr(text, s - 1));
}
template <size_t s>
inline EncodingResult<String> decodeUriComponent(const char (&text)[s]) {
  return decodeUriComponent(arrayPtr(text, s-1));
}
template <size_t s>
inline String encodeUriFragment(const char (&text)[s]) {
  return encodeUriFragment(arrayPtr(text, s - 1));
}
template <size_t s>
inline String encodeUriPath(const char (&text)[s]) {
  return encodeUriPath(arrayPtr(text, s - 1));
}
template <size_t s>
inline String encodeUriUserInfo(const char (&text)[s]) {
  return encodeUriUserInfo(arrayPtr(text, s - 1));
}
template <size_t s>
inline String encodeWwwForm(const char (&text)[s]) {
  return encodeWwwForm(arrayPtr(text, s - 1));
}
template <size_t s>
inline EncodingResult<String> decodeWwwForm(const char (&text)[s]) {
  return decodeWwwForm(arrayPtr(text, s-1));
}
template <size_t s>
inline String encodeCEscape(const char (&text)[s]) {
  return encodeCEscape(arrayPtr(text, s - 1));
}
template <size_t s>
inline EncodingResult<Array<byte>> decodeBinaryCEscape(const char (&text)[s]) {
  return decodeBinaryCEscape(arrayPtr(text, s - 1));
}
template <size_t s>
inline EncodingResult<String> decodeCEscape(const char (&text)[s]) {
  return decodeCEscape(arrayPtr(text, s-1));
}
template <size_t s>
EncodingResult<Array<byte>> decodeBase64(const char (&text)[s]) {
  return decodeBase64(arrayPtr(text, s - 1));
}

#if __cpp_char8_t
template <size_t s>
inline EncodingResult<Array<char16_t>> encodeUtf16(const char8_t (&text)[s], bool nulTerminate=false) {
  return encodeUtf16(arrayPtr(reinterpret_cast<const char*>(text), s - 1), nulTerminate);
}
template <size_t s>
inline EncodingResult<Array<char32_t>> encodeUtf32(const char8_t (&text)[s], bool nulTerminate=false) {
  return encodeUtf32(arrayPtr(reinterpret_cast<const char*>(text), s - 1), nulTerminate);
}
template <size_t s>
inline EncodingResult<Array<wchar_t>> encodeWideString(
    const char8_t (&text)[s], bool nulTerminate=false) {
  return encodeWideString(arrayPtr(reinterpret_cast<const char*>(text), s - 1), nulTerminate);
}
template <size_t s>
inline EncodingResult<Array<byte>> decodeHex(const char8_t (&text)[s]) {
  return decodeHex(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline String encodeUriComponent(const char8_t (&text)[s]) {
  return encodeUriComponent(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline Array<byte> decodeBinaryUriComponent(const char8_t (&text)[s]) {
  return decodeBinaryUriComponent(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline EncodingResult<String> decodeUriComponent(const char8_t (&text)[s]) {
  return decodeUriComponent(arrayPtr(reinterpret_cast<const char*>(text), s-1));
}
template <size_t s>
inline String encodeUriFragment(const char8_t (&text)[s]) {
  return encodeUriFragment(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline String encodeUriPath(const char8_t (&text)[s]) {
  return encodeUriPath(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline String encodeUriUserInfo(const char8_t (&text)[s]) {
  return encodeUriUserInfo(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline String encodeWwwForm(const char8_t (&text)[s]) {
  return encodeWwwForm(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline EncodingResult<String> decodeWwwForm(const char8_t (&text)[s]) {
  return decodeWwwForm(arrayPtr(reinterpret_cast<const char*>(text), s-1));
}
template <size_t s>
inline String encodeCEscape(const char8_t (&text)[s]) {
  return encodeCEscape(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline EncodingResult<Array<byte>> decodeBinaryCEscape(const char8_t (&text)[s]) {
  return decodeBinaryCEscape(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
template <size_t s>
inline EncodingResult<String> decodeCEscape(const char8_t (&text)[s]) {
  return decodeCEscape(arrayPtr(reinterpret_cast<const char*>(text), s-1));
}
template <size_t s>
EncodingResult<Array<byte>> decodeBase64(const char8_t (&text)[s]) {
  return decodeBase64(arrayPtr(reinterpret_cast<const char*>(text), s - 1));
}
#endif

} // namespace kj

KJ_END_HEADER
