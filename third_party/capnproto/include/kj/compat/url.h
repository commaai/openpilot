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

#include <kj/string.h>
#include <kj/vector.h>
#include <inttypes.h>

KJ_BEGIN_HEADER

namespace kj {

struct UrlOptions {
  // A bag of options that you can pass to Url::parse()/tryParse() to customize the parser's
  // behavior.
  //
  // A copy of this options struct will be stored in the parsed Url object, at which point it
  // controls the behavior of the serializer in Url::toString().

  bool percentDecode = true;
  // True if URL components should be automatically percent-decoded during parsing, and
  // percent-encoded during serialization.

  bool allowEmpty = false;
  // Whether or not to allow empty path and query components when parsing; otherwise, they are
  // silently removed. In other words, setting this false causes consecutive slashes in the path or
  // consecutive ampersands in the query to be collapsed into one, whereas if true then they
  // produce empty components.
};

struct Url {
  // Represents a URL (or, more accurately, a URI, but whatever).
  //
  // Can be parsed from a string and composed back into a string.

  String scheme;
  // E.g. "http", "https".

  struct UserInfo {
    String username;
    Maybe<String> password;
  };

  Maybe<UserInfo> userInfo;
  // Username / password.

  String host;
  // Hostname, including port if specified. We choose not to parse out the port because KJ's
  // network address parsing functions already accept addresses containing port numbers, and
  // because most web standards don't actually want to separate host and port.

  Vector<String> path;
  bool hasTrailingSlash = false;
  // Path, split on '/' characters. Note that the individual components of `path` could contain
  // '/' characters if they were percent-encoded in the original URL.
  //
  // No component of the path is allowed to be "", ".", nor ".."; if such components are present,
  // toString() will throw. Note that parse() and parseRelative() automatically resolve such
  // components.

  struct QueryParam {
    String name;
    String value;
  };
  Vector<QueryParam> query;
  // Query, e.g. from "?key=value&key2=value2". If a component of the query contains no '=' sign,
  // it will be parsed as a key with a null value, and later serialized with no '=' sign if you call
  // Url::toString().
  //
  // To distinguish between null-valued and empty-valued query parameters, we test whether
  // QueryParam::value is an allocated or unallocated string. For example:
  //
  //     QueryParam { kj::str("name"), nullptr }      // Null-valued; will not have an '=' sign.
  //     QueryParam { kj::str("name"), kj::str("") }  // Empty-valued; WILL have an '=' sign.

  Maybe<String> fragment;
  // The stuff after the '#' character (not including the '#' character itself), if present.

  using Options = UrlOptions;
  Options options;

  // ---------------------------------------------------------------------------

  Url() = default;
  Url(Url&&) = default;
  ~Url() noexcept(false);
  Url& operator=(Url&&) = default;

  inline Url(String&& scheme, Maybe<UserInfo>&& userInfo, String&& host, Vector<String>&& path,
             bool hasTrailingSlash, Vector<QueryParam>&& query, Maybe<String>&& fragment,
             UrlOptions options)
      : scheme(kj::mv(scheme)), userInfo(kj::mv(userInfo)), host(kj::mv(host)), path(kj::mv(path)),
        hasTrailingSlash(hasTrailingSlash), query(kj::mv(query)), fragment(kj::mv(fragment)),
        options(options) {}
  // This constructor makes brace initialization work in C++11 and C++20 -- but is technically not
  // needed in C++14 nor C++17. Go figure.

  Url clone() const;

  enum Context {
    REMOTE_HREF,
    // A link to a remote resource. Requires an authority (hostname) section, hence this will
    // reject things like "mailto:" and "data:". This is the default context.

    HTTP_PROXY_REQUEST,
    // The URL to place in the first line of an HTTP proxy request. This includes scheme, host,
    // path, and query, but omits userInfo (which should be used to construct the Authorization
    // header) and fragment (which should not be transmitted).

    HTTP_REQUEST
    // The path to place in the first line of a regular HTTP request. This includes only the path
    // and query. Scheme, user, host, and fragment are omitted.

    // TODO(someday): Add context(s) that supports things like "mailto:", "data:", "blob:". These
    //   don't have an authority section.
  };

  kj::String toString(Context context = REMOTE_HREF) const;
  // Convert the URL to a string.

  static Url parse(StringPtr text, Context context = REMOTE_HREF, Options options = {});
  static Maybe<Url> tryParse(StringPtr text, Context context = REMOTE_HREF, Options options = {});
  // Parse an absolute URL.

  Url parseRelative(StringPtr relative) const;
  Maybe<Url> tryParseRelative(StringPtr relative) const;
  // Parse a relative URL string with this URL as the base.
};

} // namespace kj

KJ_END_HEADER
