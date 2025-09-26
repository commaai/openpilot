// Copyright (c) 2017 Sandstorm Development Group, Inc. and contributors
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
// The KJ HTTP client/server library.
//
// This is a simple library which can be used to implement an HTTP client or server. Properties
// of this library include:
// - Uses KJ async framework.
// - Agnostic to transport layer -- you can provide your own.
// - Header parsing is zero-copy -- it results in strings that point directly into the buffer
//   received off the wire.
// - Application code which reads and writes headers refers to headers by symbolic names, not by
//   string literals, with lookups being array-index-based, not map-based. To make this possible,
//   the application announces what headers it cares about in advance, in order to assign numeric
//   values to them.
// - Methods are identified by an enum.

#include <kj/string.h>
#include <kj/vector.h>
#include <kj/memory.h>
#include <kj/one-of.h>
#include <kj/async-io.h>
#include <kj/debug.h>

KJ_BEGIN_HEADER

namespace kj {

#define KJ_HTTP_FOR_EACH_METHOD(MACRO) \
  MACRO(GET) \
  MACRO(HEAD) \
  MACRO(POST) \
  MACRO(PUT) \
  MACRO(DELETE) \
  MACRO(PATCH) \
  MACRO(PURGE) \
  MACRO(OPTIONS) \
  MACRO(TRACE) \
  /* standard methods */ \
  /* */ \
  /* (CONNECT is intentionally omitted since it should be handled specially in HttpServer) */ \
  \
  MACRO(COPY) \
  MACRO(LOCK) \
  MACRO(MKCOL) \
  MACRO(MOVE) \
  MACRO(PROPFIND) \
  MACRO(PROPPATCH) \
  MACRO(SEARCH) \
  MACRO(UNLOCK) \
  MACRO(ACL) \
  /* WebDAV */ \
  \
  MACRO(REPORT) \
  MACRO(MKACTIVITY) \
  MACRO(CHECKOUT) \
  MACRO(MERGE) \
  /* Subversion */ \
  \
  MACRO(MSEARCH) \
  MACRO(NOTIFY) \
  MACRO(SUBSCRIBE) \
  MACRO(UNSUBSCRIBE)
  /* UPnP */

enum class HttpMethod {
  // Enum of known HTTP methods.
  //
  // We use an enum rather than a string to allow for faster parsing and switching and to reduce
  // ambiguity.

#define DECLARE_METHOD(id) id,
KJ_HTTP_FOR_EACH_METHOD(DECLARE_METHOD)
#undef DECLARE_METHOD
};

struct HttpConnectMethod {};
// CONNECT is handled specially and separately from the other HttpMethods.

kj::StringPtr KJ_STRINGIFY(HttpMethod method);
kj::StringPtr KJ_STRINGIFY(HttpConnectMethod method);
kj::Maybe<HttpMethod> tryParseHttpMethod(kj::StringPtr name);
kj::Maybe<kj::OneOf<HttpMethod, HttpConnectMethod>> tryParseHttpMethodAllowingConnect(
    kj::StringPtr name);
// Like tryParseHttpMethod but, as the name suggests, explicitly allows for the CONNECT
// method. Added as a separate function instead of modifying tryParseHttpMethod to avoid
// breaking API changes in existing uses of tryParseHttpMethod.

class HttpHeaderTable;

class HttpHeaderId {
  // Identifies an HTTP header by numeric ID that indexes into an HttpHeaderTable.
  //
  // The KJ HTTP API prefers that headers be identified by these IDs for a few reasons:
  // - Integer lookups are much more efficient than string lookups.
  // - Case-insensitivity is awkward to deal with when const strings are being passed to the lookup
  //   method.
  // - Writing out strings less often means fewer typos.
  //
  // See HttpHeaderTable for usage hints.

public:
  HttpHeaderId() = default;

  inline bool operator==(const HttpHeaderId& other) const { return id == other.id; }
  inline bool operator!=(const HttpHeaderId& other) const { return id != other.id; }
  inline bool operator< (const HttpHeaderId& other) const { return id <  other.id; }
  inline bool operator> (const HttpHeaderId& other) const { return id >  other.id; }
  inline bool operator<=(const HttpHeaderId& other) const { return id <= other.id; }
  inline bool operator>=(const HttpHeaderId& other) const { return id >= other.id; }

  inline size_t hashCode() const { return id; }
  // Returned value is guaranteed to be small and never collide with other headers on the same
  // table.

  kj::StringPtr toString() const;

  void requireFrom(const HttpHeaderTable& table) const;
  // In debug mode, throws an exception if the HttpHeaderId is not from the given table.
  //
  // In opt mode, no-op.

#define KJ_HTTP_FOR_EACH_BUILTIN_HEADER(MACRO) \
  /* Headers that are always read-only. */ \
  MACRO(CONNECTION, "Connection") \
  MACRO(KEEP_ALIVE, "Keep-Alive") \
  MACRO(TE, "TE") \
  MACRO(TRAILER, "Trailer") \
  MACRO(UPGRADE, "Upgrade") \
  \
  /* Headers that are read-only except in the case of a response to a HEAD request. */ \
  MACRO(CONTENT_LENGTH, "Content-Length") \
  MACRO(TRANSFER_ENCODING, "Transfer-Encoding") \
  \
  /* Headers that are read-only for WebSocket handshakes. */ \
  MACRO(SEC_WEBSOCKET_KEY, "Sec-WebSocket-Key") \
  MACRO(SEC_WEBSOCKET_VERSION, "Sec-WebSocket-Version") \
  MACRO(SEC_WEBSOCKET_ACCEPT, "Sec-WebSocket-Accept") \
  MACRO(SEC_WEBSOCKET_EXTENSIONS, "Sec-WebSocket-Extensions") \
  \
  /* Headers that you can write. */ \
  MACRO(HOST, "Host") \
  MACRO(DATE, "Date") \
  MACRO(LOCATION, "Location") \
  MACRO(CONTENT_TYPE, "Content-Type")
  // For convenience, these headers are valid for all HttpHeaderTables. You can refer to them like:
  //
  //     HttpHeaderId::HOST
  //
  // TODO(someday): Fill this out with more common headers.

#define DECLARE_HEADER(id, name) \
  static const HttpHeaderId id;
  // Declare a constant for each builtin header, e.g.: HttpHeaderId::CONNECTION

  KJ_HTTP_FOR_EACH_BUILTIN_HEADER(DECLARE_HEADER);
#undef DECLARE_HEADER

private:
  const HttpHeaderTable* table;
  uint id;

  inline explicit constexpr HttpHeaderId(const HttpHeaderTable* table, uint id)
      : table(table), id(id) {}
  friend class HttpHeaderTable;
  friend class HttpHeaders;
};

class HttpHeaderTable {
  // Construct an HttpHeaderTable to declare which headers you'll be interested in later on, and
  // to manufacture IDs for them.
  //
  // Example:
  //
  //     // Build a header table with the headers we are interested in.
  //     kj::HttpHeaderTable::Builder builder;
  //     const HttpHeaderId accept = builder.add("Accept");
  //     const HttpHeaderId contentType = builder.add("Content-Type");
  //     kj::HttpHeaderTable table(kj::mv(builder));
  //
  //     // Create an HTTP client.
  //     auto client = kj::newHttpClient(table, network);
  //
  //     // Get http://example.com.
  //     HttpHeaders headers(table);
  //     headers.set(accept, "text/html");
  //     auto response = client->send(kj::HttpMethod::GET, "http://example.com", headers)
  //         .wait(waitScope);
  //     auto msg = kj::str("Response content type: ", response.headers.get(contentType));

  struct IdsByNameMap;

public:
  HttpHeaderTable();
  // Constructs a table that only contains the builtin headers.

  class Builder {
  public:
    Builder();
    HttpHeaderId add(kj::StringPtr name);
    Own<HttpHeaderTable> build();

    HttpHeaderTable& getFutureTable();
    // Get the still-unbuilt header table. You cannot actually use it until build() has been
    // called.
    //
    // This method exists to help when building a shared header table -- the Builder may be passed
    // to several components, each of which will register the headers they need and get a reference
    // to the future table.

  private:
    kj::Own<HttpHeaderTable> table;
  };

  KJ_DISALLOW_COPY_AND_MOVE(HttpHeaderTable);  // Can't copy because HttpHeaderId points to the table.
  ~HttpHeaderTable() noexcept(false);

  uint idCount() const;
  // Return the number of IDs in the table.

  kj::Maybe<HttpHeaderId> stringToId(kj::StringPtr name) const;
  // Try to find an ID for the given name. The matching is case-insensitive, per the HTTP spec.
  //
  // Note: if `name` contains characters that aren't allowed in HTTP header names, this may return
  //   a bogus value rather than null, due to optimizations used in case-insensitive matching.

  kj::StringPtr idToString(HttpHeaderId id) const;
  // Get the canonical string name for the given ID.

  bool isReady() const;
  // Returns true if this HttpHeaderTable either was default constructed or its Builder has
  // invoked `build()` and released it.

private:
  kj::Vector<kj::StringPtr> namesById;
  kj::Own<IdsByNameMap> idsByName;

  enum class BuildStatus {
    UNSTARTED = 0,
    BUILDING = 1,
    FINISHED = 2,
  };
  BuildStatus buildStatus = BuildStatus::UNSTARTED;
};

class HttpHeaders {
  // Represents a set of HTTP headers.
  //
  // This class guards against basic HTTP header injection attacks: Trying to set a header name or
  // value containing a newline, carriage return, or other invalid character will throw an
  // exception.

public:
  explicit HttpHeaders(const HttpHeaderTable& table);

  static bool isValidHeaderValue(kj::StringPtr value);
  // This returns whether the value is a valid parameter to the set call. While the HTTP spec
  // suggests that only printable ASCII characters are allowed in header values, in practice that
  // turns out to not be the case. We follow the browser's lead in disallowing \r and \n.
  // https://github.com/httpwg/http11bis/issues/19
  // Use this if you want to validate the value before supplying it to set() if you want to avoid
  // an exception being thrown (e.g. you have custom error reporting). NOTE that set will still
  // validate the value. If performance is a problem this API needs to be adjusted to a
  // `validateHeaderValue` function that returns a special type that set can be confident has
  // already passed through the validation routine.

  KJ_DISALLOW_COPY(HttpHeaders);
  HttpHeaders(HttpHeaders&&) = default;
  HttpHeaders& operator=(HttpHeaders&&) = default;

  size_t size() const;
  // Returns the number of headers that forEach() would iterate over.

  void clear();
  // Clears all contents, as if the object was freshly-allocated. However, calling this rather
  // than actually re-allocating the object may avoid re-allocation of internal objects.

  HttpHeaders clone() const;
  // Creates a deep clone of the HttpHeaders. The returned object owns all strings it references.

  HttpHeaders cloneShallow() const;
  // Creates a shallow clone of the HttpHeaders. The returned object references the same strings
  // as the original, owning none of them.

  bool isWebSocket() const;
  // Convenience method that checks for the presence of the header `Upgrade: websocket`.
  //
  // Note that this does not actually validate that the request is a complete WebSocket handshake
  // with the correct version number -- such validation will occur if and when you call
  // acceptWebSocket().

  kj::Maybe<kj::StringPtr> get(HttpHeaderId id) const;
  // Read a header.
  //
  // Note that there is intentionally no method to look up a header by string name rather than
  // header ID. The intent is that you should always allocate a header ID for any header that you
  // care about, so that you can get() it by ID. Headers with registered IDs are stored in an array
  // indexed by ID, making lookup fast. Headers without registered IDs are stored in a separate list
  // that is optimized for re-transmission of the whole list, but not for lookup.

  template <typename Func>
  void forEach(Func&& func) const;
  // Calls `func(name, value)` for each header in the set -- including headers that aren't mapped
  // to IDs in the header table. Both inputs are of type kj::StringPtr.

  template <typename Func1, typename Func2>
  void forEach(Func1&& func1, Func2&& func2) const;
  // Calls `func1(id, value)` for each header in the set that has a registered HttpHeaderId, and
  // `func2(name, value)` for each header that does not. All calls to func1() precede all calls to
  // func2().

  void set(HttpHeaderId id, kj::StringPtr value);
  void set(HttpHeaderId id, kj::String&& value);
  // Sets a header value, overwriting the existing value.
  //
  // The String&& version is equivalent to calling the other version followed by takeOwnership().
  //
  // WARNING: It is the caller's responsibility to ensure that `value` remains valid until the
  //   HttpHeaders object is destroyed. This allows string literals to be passed without making a
  //   copy, but complicates the use of dynamic values. Hint: Consider using `takeOwnership()`.

  void add(kj::StringPtr name, kj::StringPtr value);
  void add(kj::StringPtr name, kj::String&& value);
  void add(kj::String&& name, kj::String&& value);
  // Append a header. `name` will be looked up in the header table, but if it's not mapped, the
  // header will be added to the list of unmapped headers.
  //
  // The String&& versions are equivalent to calling the other version followed by takeOwnership().
  //
  // WARNING: It is the caller's responsibility to ensure that `name` and `value` remain valid
  //   until the HttpHeaders object is destroyed. This allows string literals to be passed without
  //   making a copy, but complicates the use of dynamic values. Hint: Consider using
  //   `takeOwnership()`.

  void unset(HttpHeaderId id);
  // Removes a header.
  //
  // It's not possible to remove a header by string name because non-indexed headers would take
  // O(n) time to remove. Instead, construct a new HttpHeaders object and copy contents.

  void takeOwnership(kj::String&& string);
  void takeOwnership(kj::Array<char>&& chars);
  void takeOwnership(HttpHeaders&& otherHeaders);
  // Takes ownership of a string so that it lives until the HttpHeaders object is destroyed. Useful
  // when you've passed a dynamic value to set() or add() or parse*().

  struct Request {
    HttpMethod method;
    kj::StringPtr url;
  };
  struct ConnectRequest {
    kj::StringPtr authority;
  };
  struct Response {
    uint statusCode;
    kj::StringPtr statusText;
  };

  struct ProtocolError {
    // Represents a protocol error, such as a bad request method or invalid headers. Debugging such
    // errors is difficult without a copy of the data which we tried to parse, but this data is
    // sensitive, so we can't just lump it into the error description directly. ProtocolError
    // provides this sensitive data separate from the error description.
    //
    // TODO(cleanup): Should maybe not live in HttpHeaders? HttpServerErrorHandler::ProtocolError?
    //   Or HttpProtocolError? Or maybe we need a more general way of attaching sensitive context to
    //   kj::Exceptions?

    uint statusCode;
    // Suggested HTTP status code that should be used when returning an error to the client.
    //
    // Most errors are 400. An unrecognized method will be 501.

    kj::StringPtr statusMessage;
    // HTTP status message to go with `statusCode`, e.g. "Bad Request".

    kj::StringPtr description;
    // An error description safe for all the world to see.

    kj::ArrayPtr<char> rawContent;
    // Unredacted data which led to the error condition. This may contain anything transported over
    // HTTP, to include sensitive PII, so you must take care to sanitize this before using it in any
    // error report that may leak to unprivileged eyes.
    //
    // This ArrayPtr is merely a copy of the `content` parameter passed to `tryParseRequest()` /
    // `tryParseResponse()`, thus it remains valid for as long as a successfully-parsed HttpHeaders
    // object would remain valid.
  };

  using RequestOrProtocolError = kj::OneOf<Request, ProtocolError>;
  using ResponseOrProtocolError = kj::OneOf<Response, ProtocolError>;
  using RequestConnectOrProtocolError = kj::OneOf<Request, ConnectRequest, ProtocolError>;

  RequestOrProtocolError tryParseRequest(kj::ArrayPtr<char> content);
  RequestConnectOrProtocolError tryParseRequestOrConnect(kj::ArrayPtr<char> content);
  ResponseOrProtocolError tryParseResponse(kj::ArrayPtr<char> content);

  // Parse an HTTP header blob and add all the headers to this object.
  //
  // `content` should be all text from the start of the request to the first occurrence of two
  // newlines in a row -- including the first of these two newlines, but excluding the second.
  //
  // The parse is performed with zero copies: The callee clobbers `content` with '\0' characters
  // to split it into a bunch of shorter strings. The caller must keep `content` valid until the
  // `HttpHeaders` is destroyed, or pass it to `takeOwnership()`.

  bool tryParse(kj::ArrayPtr<char> content);
  // Like tryParseRequest()/tryParseResponse(), but don't expect any request/response line.

  kj::String serializeRequest(HttpMethod method, kj::StringPtr url,
                              kj::ArrayPtr<const kj::StringPtr> connectionHeaders = nullptr) const;
  kj::String serializeConnectRequest(kj::StringPtr authority,
                              kj::ArrayPtr<const kj::StringPtr> connectionHeaders = nullptr) const;
  kj::String serializeResponse(uint statusCode, kj::StringPtr statusText,
                               kj::ArrayPtr<const kj::StringPtr> connectionHeaders = nullptr) const;
  // **Most applications will not use these methods; they are called by the HTTP client and server
  // implementations.**
  //
  // Serialize the headers as a complete request or response blob. The blob uses '\r\n' newlines
  // and includes the double-newline to indicate the end of the headers.
  //
  // `connectionHeaders`, if provided, contains connection-level headers supplied by the HTTP
  // implementation, in the order specified by the KJ_HTTP_FOR_EACH_BUILTIN_HEADER macro. These
  // headers values override any corresponding header value in the HttpHeaders object. The
  // CONNECTION_HEADERS_COUNT constants below can help you construct this `connectionHeaders` array.

  enum class BuiltinIndicesEnum {
  #define HEADER_ID(id, name) id,
    KJ_HTTP_FOR_EACH_BUILTIN_HEADER(HEADER_ID)
  #undef HEADER_ID
  };

  struct BuiltinIndices {
  #define HEADER_ID(id, name) static constexpr uint id = static_cast<uint>(BuiltinIndicesEnum::id);
    KJ_HTTP_FOR_EACH_BUILTIN_HEADER(HEADER_ID)
  #undef HEADER_ID
  };

  static constexpr uint HEAD_RESPONSE_CONNECTION_HEADERS_COUNT = BuiltinIndices::CONTENT_LENGTH;
  static constexpr uint CONNECTION_HEADERS_COUNT = BuiltinIndices::SEC_WEBSOCKET_KEY;
  static constexpr uint WEBSOCKET_CONNECTION_HEADERS_COUNT = BuiltinIndices::HOST;
  // Constants for use with HttpHeaders::serialize*().

  kj::String toString() const;

private:
  const HttpHeaderTable* table;

  kj::Array<kj::StringPtr> indexedHeaders;
  // Size is always table->idCount().

  struct Header {
    kj::StringPtr name;
    kj::StringPtr value;
  };
  kj::Vector<Header> unindexedHeaders;

  kj::Vector<kj::Array<char>> ownedStrings;

  void addNoCheck(kj::StringPtr name, kj::StringPtr value);

  kj::StringPtr cloneToOwn(kj::StringPtr str);

  kj::String serialize(kj::ArrayPtr<const char> word1,
                       kj::ArrayPtr<const char> word2,
                       kj::ArrayPtr<const char> word3,
                       kj::ArrayPtr<const kj::StringPtr> connectionHeaders) const;

  bool parseHeaders(char* ptr, char* end);

  // TODO(perf): Arguably we should store a map, but header sets are never very long
  // TODO(perf): We could optimize for common headers by storing them directly as fields. We could
  //   also add direct accessors for those headers.
};

class HttpInputStream {
  // Low-level interface to receive HTTP-formatted messages (headers followed by body) from an
  // input stream, without a paired output stream.
  //
  // Most applications will not use this. Regular HTTP clients and servers don't need this. This
  // is mainly useful for apps implementing various protocols that look like HTTP but aren't
  // really.

public:
  struct Request {
    HttpMethod method;
    kj::StringPtr url;
    const HttpHeaders& headers;
    kj::Own<kj::AsyncInputStream> body;
  };
  virtual kj::Promise<Request> readRequest() = 0;
  // Reads one HTTP request from the input stream.
  //
  // The returned struct contains pointers directly into a buffer that is invalidated on the next
  // message read.

  struct Connect {
    kj::StringPtr authority;
    const HttpHeaders& headers;
    kj::Own<kj::AsyncInputStream> body;
  };
  virtual kj::Promise<kj::OneOf<Request, Connect>> readRequestAllowingConnect() = 0;
  // Reads one HTTP request from the input stream.
  //
  // The returned struct contains pointers directly into a buffer that is invalidated on the next
  // message read.

  struct Response {
    uint statusCode;
    kj::StringPtr statusText;
    const HttpHeaders& headers;
    kj::Own<kj::AsyncInputStream> body;
  };
  virtual kj::Promise<Response> readResponse(HttpMethod requestMethod) = 0;
  // Reads one HTTP response from the input stream.
  //
  // You must provide the request method because responses to HEAD requests require special
  // treatment.
  //
  // The returned struct contains pointers directly into a buffer that is invalidated on the next
  // message read.

  struct Message {
    const HttpHeaders& headers;
    kj::Own<kj::AsyncInputStream> body;
  };
  virtual kj::Promise<Message> readMessage() = 0;
  // Reads an HTTP header set followed by a body, with no request or response line. This is not
  // useful for HTTP but may be useful for other protocols that make the unfortunate choice to
  // mimic HTTP message format, such as Visual Studio Code's JSON-RPC transport.
  //
  // The returned struct contains pointers directly into a buffer that is invalidated on the next
  // message read.

  virtual kj::Promise<bool> awaitNextMessage() = 0;
  // Waits until more data is available, but doesn't consume it. Returns false on EOF.
};

class EntropySource {
  // Interface for an object that generates entropy. Typically, cryptographically-random entropy
  // is expected.
  //
  // TODO(cleanup): Put this somewhere more general.

public:
  virtual void generate(kj::ArrayPtr<byte> buffer) = 0;
};

struct CompressionParameters {
  // These are the parameters for `Sec-WebSocket-Extensions` permessage-deflate extension.
  // Since we cannot distinguish the client/server in `upgradeToWebSocket`, we use the prefixes
  // `inbound` and `outbound` instead.
  bool outboundNoContextTakeover = false;
  bool inboundNoContextTakeover = false;
  kj::Maybe<size_t> outboundMaxWindowBits = nullptr;
  kj::Maybe<size_t> inboundMaxWindowBits = nullptr;
};

class WebSocket {
  // Interface representincg an open WebSocket session.
  //
  // Each side can send and receive data and "close" messages.
  //
  // Ping/Pong and message fragmentation are not exposed through this interface. These features of
  // the underlying WebSocket protocol are not exposed by the browser-level JavaScript API either,
  // and thus applications typically need to implement these features at the application protocol
  // level instead. The implementation is, however, expected to reply to Ping messages it receives.

public:
  virtual kj::Promise<void> send(kj::ArrayPtr<const byte> message) = 0;
  virtual kj::Promise<void> send(kj::ArrayPtr<const char> message) = 0;
  // Send a message (binary or text). The underlying buffer must remain valid, and you must not
  // call send() again, until the returned promise resolves.

  virtual kj::Promise<void> close(uint16_t code, kj::StringPtr reason) = 0;
  // Send a Close message.
  //
  // Note that the returned Promise resolves once the message has been sent -- it does NOT wait
  // for the other end to send a Close reply. The application should await a reply before dropping
  // the WebSocket object.

  virtual kj::Promise<void> disconnect() = 0;
  // Sends EOF on the underlying connection without sending a "close" message. This is NOT a clean
  // shutdown, but is sometimes useful when you want the other end to trigger whatever behavior
  // it normally triggers when a connection is dropped.

  virtual void abort() = 0;
  // Forcefully close this WebSocket, such that the remote end should get a DISCONNECTED error if
  // it continues to write. This differs from disconnect(), which only closes the sending
  // direction, but still allows receives.

  virtual kj::Promise<void> whenAborted() = 0;
  // Resolves when the remote side aborts the connection such that send() would throw DISCONNECTED,
  // if this can be detected without actually writing a message. (If not, this promise never
  // resolves, but send() or receive() will throw DISCONNECTED when appropriate. See also
  // kj::AsyncOutputStream::whenWriteDisconnected().)

  struct ProtocolError {
    // Represents a protocol error, such as a bad opcode or oversize message.

    uint statusCode;
    // Suggested WebSocket status code that should be used when returning an error to the client.
    //
    // Most errors are 1002; an oversize message will be 1009.

    kj::StringPtr description;
    // An error description safe for all the world to see. This should be at most 123 bytes so that
    // it can be used as the body of a Close frame (RFC 6455 sections 5.5 and 5.5.1).
  };

  struct Close {
    uint16_t code;
    kj::String reason;
  };

  typedef kj::OneOf<kj::String, kj::Array<byte>, Close> Message;

  static constexpr size_t SUGGESTED_MAX_MESSAGE_SIZE = 1u << 20;  // 1MB

  virtual kj::Promise<Message> receive(size_t maxSize = SUGGESTED_MAX_MESSAGE_SIZE) = 0;
  // Read one message from the WebSocket and return it. Can only call once at a time. Do not call
  // again after Close is received.

  virtual kj::Promise<void> pumpTo(WebSocket& other);
  // Continuously receives messages from this WebSocket and send them to `other`.
  //
  // On EOF, calls other.disconnect(), then resolves.
  //
  // On other read errors, calls other.close() with the error, then resolves.
  //
  // On write error, rejects with the error.

  virtual kj::Maybe<kj::Promise<void>> tryPumpFrom(WebSocket& other);
  // Either returns null, or performs the equivalent of other.pumpTo(*this). Only returns non-null
  // if this WebSocket implementation is able to perform the pump in an optimized way, better than
  // the default implementation of pumpTo(). The default implementation of pumpTo() always tries
  // calling this first, and the default implementation of tryPumpFrom() always returns null.

  virtual uint64_t sentByteCount() = 0;
  virtual uint64_t receivedByteCount() = 0;

  enum ExtensionsContext {
    // Indicate whether a Sec-WebSocket-Extension header should be rendered for use in request
    // headers or response headers.
    REQUEST,
    RESPONSE
  };
  virtual kj::Maybe<kj::String> getPreferredExtensions(ExtensionsContext ctx) { return nullptr; }
  // If pumpTo() / tryPumpFrom() is able to be optimized only if the other WebSocket is using
  // certain extensions (e.g. compression settings), then this method returns what those extensions
  // are. For example, matching extensions between standard WebSockets allows pumping to be
  // implemented by pumping raw bytes between network connections, without reading individual frames.
  //
  // A null return value indicates that there is no preference. A non-null return value containing
  // an empty string indicates a preference for no extensions to be applied.
};

using TlsStarterCallback = kj::Maybe<kj::Function<kj::Promise<void>(kj::StringPtr)>>;
struct HttpConnectSettings {
  bool useTls = false;
  // Requests to automatically establish a TLS session over the connection. The remote party
  // will be expected to present a valid certificate matching the requested hostname.
  kj::Maybe<TlsStarterCallback&> tlsStarter;
  // This is an output parameter. It doesn't need to be set. But if it is set, then it may get
  // filled with a callback function. It will get filled with `nullptr` if any of the following
  // are true:
  //
  // * kj is not built with TLS support
  // * the underlying HttpClient does not support the startTls mechanism
  // * `useTls` has been set to `true` and so TLS has already been started
  //
  // The callback function itself can be called to initiate a TLS handshake on the connection in
  // between write() operations. It is not allowed to initiate a TLS handshake while a write
  // operation or a pump operation to the connection exists. Read operations are not subject to
  // the same constraint, however: implementations are required to be able to handle TLS
  // initiation while a read operation or pump operation from the connection exists. Once the
  // promise returned from the callback is fulfilled, the connection has become a secure stream,
  // and write operations are once again permitted. The StringPtr parameter to the callback,
  // expectedServerHostname may be dropped after the function synchronously returns.
  //
  // The PausableReadAsyncIoStream class defined below can be used to ensure that read operations
  // are not pending when the tlsStarter is invoked.
  //
  // This mechanism is required for certain protocols, more info can be found on
  // https://en.wikipedia.org/wiki/Opportunistic_TLS.
};


class PausableReadAsyncIoStream final: public kj::AsyncIoStream {
  // A custom AsyncIoStream which can pause pending reads. This is used by startTls to pause a
  // a read before TLS is initiated.
  //
  // TODO(cleanup): this class should be rewritten to use a CRTP mixin approach so that pumps
  // can be optimised once startTls is invoked.
  class PausableRead;
public:
  PausableReadAsyncIoStream(kj::Own<kj::AsyncIoStream> stream)
      : inner(kj::mv(stream)), currentlyWriting(false), currentlyReading(false) {}

  _::Deferred<kj::Function<void()>> trackRead();

  _::Deferred<kj::Function<void()>> trackWrite();

  kj::Promise<size_t> tryRead(void* buffer, size_t minBytes, size_t maxBytes) override;

  kj::Promise<size_t> tryReadImpl(void* buffer, size_t minBytes, size_t maxBytes);

  kj::Maybe<uint64_t> tryGetLength() override;

  kj::Promise<uint64_t> pumpTo(kj::AsyncOutputStream& output, uint64_t amount) override;

  kj::Promise<void> write(const void* buffer, size_t size) override;

  kj::Promise<void> write(kj::ArrayPtr<const kj::ArrayPtr<const byte>> pieces) override;

  kj::Maybe<kj::Promise<uint64_t>> tryPumpFrom(
      kj::AsyncInputStream& input, uint64_t amount = kj::maxValue) override;

  kj::Promise<void> whenWriteDisconnected() override;

  void shutdownWrite() override;

  void abortRead() override;

  kj::Maybe<int> getFd() const override;

  void pause();

  void unpause();

  bool getCurrentlyReading();

  bool getCurrentlyWriting();

  kj::Own<kj::AsyncIoStream> takeStream();

  void replaceStream(kj::Own<kj::AsyncIoStream> stream);

  void reject(kj::Exception&& exc);

private:
  kj::Own<kj::AsyncIoStream> inner;
  kj::Maybe<PausableRead&> maybePausableRead;
  bool currentlyWriting;
  bool currentlyReading;
};

class HttpClient {
  // Interface to the client end of an HTTP connection.
  //
  // There are two kinds of clients:
  // * Host clients are used when talking to a specific host. The `url` specified in a request
  //   is actually just a path. (A `Host` header is still required in all requests.)
  // * Proxy clients are used when the target could be any arbitrary host on the internet.
  //   The `url` specified in a request is a full URL including protocol and hostname.

public:
  struct Response {
    uint statusCode;
    kj::StringPtr statusText;
    const HttpHeaders* headers;
    kj::Own<kj::AsyncInputStream> body;
    // `statusText` and `headers` remain valid until `body` is dropped or read from.
  };

  struct Request {
    kj::Own<kj::AsyncOutputStream> body;
    // Write the request entity body to this stream, then drop it when done.
    //
    // May be null for GET and HEAD requests (which have no body) and requests that have
    // Content-Length: 0.

    kj::Promise<Response> response;
    // Promise for the eventual response.
  };

  virtual Request request(HttpMethod method, kj::StringPtr url, const HttpHeaders& headers,
                          kj::Maybe<uint64_t> expectedBodySize = nullptr) = 0;
  // Perform an HTTP request.
  //
  // `url` may be a full URL (with protocol and host) or it may be only the path part of the URL,
  // depending on whether the client is a proxy client or a host client.
  //
  // `url` and `headers` need only remain valid until `request()` returns (they can be
  // stack-allocated).
  //
  // `expectedBodySize`, if provided, must be exactly the number of bytes that will be written to
  // the body. This will trigger use of the `Content-Length` connection header. Otherwise,
  // `Transfer-Encoding: chunked` will be used.

  struct WebSocketResponse {
    uint statusCode;
    kj::StringPtr statusText;
    const HttpHeaders* headers;
    kj::OneOf<kj::Own<kj::AsyncInputStream>, kj::Own<WebSocket>> webSocketOrBody;
    // `statusText` and `headers` remain valid until `webSocketOrBody` is dropped or read from.
  };
  virtual kj::Promise<WebSocketResponse> openWebSocket(
      kj::StringPtr url, const HttpHeaders& headers);
  // Tries to open a WebSocket. Default implementation calls send() and never returns a WebSocket.
  //
  // `url` and `headers` need only remain valid until `openWebSocket()` returns (they can be
  // stack-allocated).

  struct ConnectRequest {
    struct Status {
      uint statusCode;
      kj::String statusText;
      kj::Own<HttpHeaders> headers;
      kj::Maybe<kj::Own<kj::AsyncInputStream>> errorBody;
      // If the connect request is rejected, the statusCode can be any HTTP status code
      // outside the 200-299 range and errorBody *may* be specified if there is a rejection
      // payload.

      // TODO(perf): Having Status own the statusText and headers is a bit unfortunate.
      // Ideally we could have these be non-owned so that the headers object could just
      // point directly into HttpOutputStream's buffer and not be copied. That's a bit
      // more difficult to with CONNECT since the lifetimes of the buffers are a little
      // different than with regular HTTP requests. It should still be possible but for
      // now copying and owning the status text and headers is easier.

      Status(uint statusCode,
             kj::String statusText,
             kj::Own<HttpHeaders> headers,
             kj::Maybe<kj::Own<kj::AsyncInputStream>> errorBody = nullptr)
          : statusCode(statusCode),
            statusText(kj::mv(statusText)),
            headers(kj::mv(headers)),
            errorBody(kj::mv(errorBody)) {}
    };

    kj::Promise<Status> status;
    kj::Own<kj::AsyncIoStream> connection;
  };

  virtual ConnectRequest connect(
      kj::StringPtr host, const HttpHeaders& headers, HttpConnectSettings settings);
  // Handles CONNECT requests.
  //
  // `host` must specify both the host and port (e.g. "example.org:1234").
  //
  // The `host` and `headers` need only remain valid until `connect()` returns (it can be
  // stack-allocated).
};

class HttpService {
  // Interface which HTTP services should implement.
  //
  // This interface is functionally equivalent to HttpClient, but is intended for applications to
  // implement rather than call. The ergonomics and performance of the method signatures are
  // optimized for the serving end.
  //
  // As with clients, there are two kinds of services:
  // * Host services are used when talking to a specific host. The `url` specified in a request
  //   is actually just a path. (A `Host` header is still required in all requests, and the service
  //   may in fact serve multiple origins via this header.)
  // * Proxy services are used when the target could be any arbitrary host on the internet, i.e. to
  //   implement an HTTP proxy. The `url` specified in a request is a full URL including protocol
  //   and hostname.

public:
  class Response {
  public:
    virtual kj::Own<kj::AsyncOutputStream> send(
        uint statusCode, kj::StringPtr statusText, const HttpHeaders& headers,
        kj::Maybe<uint64_t> expectedBodySize = nullptr) = 0;
    // Begin the response.
    //
    // `statusText` and `headers` need only remain valid until send() returns (they can be
    // stack-allocated).
    //
    // `send()` may only be called a single time. Calling it a second time will cause an exception
    // to be thrown.

    virtual kj::Own<WebSocket> acceptWebSocket(const HttpHeaders& headers) = 0;
    // If headers.isWebSocket() is true then you can call acceptWebSocket() instead of send().
    //
    // If the request is an invalid WebSocket request (e.g., it has an Upgrade: websocket header,
    // but other WebSocket-related headers are invalid), `acceptWebSocket()` will throw an
    // exception, and the HttpServer will return a 400 Bad Request response and close the
    // connection. In this circumstance, the HttpServer will ignore any exceptions which propagate
    // from the `HttpService::request()` promise. `HttpServerErrorHandler::handleApplicationError()`
    // will not be invoked, and the HttpServer's listen task will be fulfilled normally.
    //
    // `acceptWebSocket()` may only be called a single time. Calling it a second time will cause an
    // exception to be thrown.

    kj::Promise<void> sendError(uint statusCode, kj::StringPtr statusText,
                                const HttpHeaders& headers);
    kj::Promise<void> sendError(uint statusCode, kj::StringPtr statusText,
                                const HttpHeaderTable& headerTable);
    // Convenience wrapper around send() which sends a basic error. A generic error page specifying
    // the error code is sent as the body.
    //
    // You must provide headers or a header table because downstream service wrappers may be
    // expecting response headers built with a particular table so that they can insert additional
    // headers.
  };

  virtual kj::Promise<void> request(
      HttpMethod method, kj::StringPtr url, const HttpHeaders& headers,
      kj::AsyncInputStream& requestBody, Response& response) = 0;
  // Perform an HTTP request.
  //
  // `url` may be a full URL (with protocol and host) or it may be only the path part of the URL,
  // depending on whether the service is a proxy service or a host service.
  //
  // `url` and `headers` are invalidated on the first read from `requestBody` or when the returned
  // promise resolves, whichever comes first.
  //
  // Request processing can be canceled by dropping the returned promise. HttpServer may do so if
  // the client disconnects prematurely.
  //
  // The implementation of `request()` should usually not try to use `response` in any way in
  // exception-handling code, because it is often not possible to tell whether `Response::send()` or
  // `Response::acceptWebSocket()` has already been called. Instead, to generate error HTTP
  // responses for the client, implement an HttpServerErrorHandler and pass it to the HttpServer via
  // HttpServerSettings. If the `HttpService::request()` promise rejects and no response has yet
  // been sent, `HttpServerErrorHandler::handleApplicationError()` will be passed a non-null
  // `Maybe<Response&>` parameter.

  class ConnectResponse {
  public:
    virtual void accept(
        uint statusCode,
        kj::StringPtr statusText,
        const HttpHeaders& headers) = 0;
    // Signals acceptance of the CONNECT tunnel.

    virtual kj::Own<kj::AsyncOutputStream> reject(
        uint statusCode,
        kj::StringPtr statusText,
        const HttpHeaders& headers,
        kj::Maybe<uint64_t> expectedBodySize = nullptr) = 0;
    // Signals rejection of the CONNECT tunnel.
  };

  virtual kj::Promise<void> connect(kj::StringPtr host,
                                    const HttpHeaders& headers,
                                    kj::AsyncIoStream& connection,
                                    ConnectResponse& response,
                                    HttpConnectSettings settings);
  // Handles CONNECT requests.
  //
  // The `host` must include host and port.
  //
  // `host` and `headers` are invalidated when accept or reject is called on the ConnectResponse
  // or when the returned promise resolves, whichever comes first.
  //
  // The connection is provided to support pipelining. Writes to the connection will be blocked
  // until one of either accept() or reject() is called on tunnel. Reads from the connection are
  // permitted at any time.
  //
  // Request processing can be canceled by dropping the returned promise. HttpServer may do so if
  // the client disconnects prematurely.
};

class HttpClientErrorHandler {
public:
  virtual HttpClient::Response handleProtocolError(HttpHeaders::ProtocolError protocolError);
  // Override this function to customize error handling when the client receives an HTTP message
  // that fails to parse. The default implementations throws an exception.
  //
  // There are two main use cases for overriding this:
  // 1. `protocolError` contains the actual header content that failed to parse, giving you the
  //    opportunity to log it for debugging purposes. The default implementation throws away this
  //    content.
  // 2. You could potentially convert protocol errors into HTTP error codes, e.g. 502 Bad Gateway.
  //
  // Note that `protocolError` may contain pointers into buffers that are no longer valid once
  // this method returns; you will have to make copies if you want to keep them.

  virtual HttpClient::WebSocketResponse handleWebSocketProtocolError(
      HttpHeaders::ProtocolError protocolError);
  // Like handleProtocolError() but for WebSocket requests. The default implementation calls
  // handleProtocolError() and converts the Response to WebSocketResponse. There is probably very
  // little reason to override this.
};

struct HttpClientSettings {
  kj::Duration idleTimeout = 5 * kj::SECONDS;
  // For clients which automatically create new connections, any connection idle for at least this
  // long will be closed. Set this to 0 to prevent connection reuse entirely.

  kj::Maybe<EntropySource&> entropySource = nullptr;
  // Must be provided in order to use `openWebSocket`. If you don't need WebSockets, this can be
  // omitted. The WebSocket protocol uses random values to avoid triggering flaws (including
  // security flaws) in certain HTTP proxy software. Specifically, entropy is used to generate the
  // `Sec-WebSocket-Key` header and to generate frame masks. If you know that there are no broken
  // or vulnerable proxies between you and the server, you can provide a dummy entropy source that
  // doesn't generate real entropy (e.g. returning the same value every time). Otherwise, you must
  // provide a cryptographically-random entropy source.

  kj::Maybe<HttpClientErrorHandler&> errorHandler = nullptr;
  // Customize how protocol errors are handled by the HttpClient. If null, HttpClientErrorHandler's
  // default implementation will be used.

  enum WebSocketCompressionMode {
    NO_COMPRESSION,
    MANUAL_COMPRESSION,    // Lets the application decide the compression configuration (if any).
    AUTOMATIC_COMPRESSION, // Automatically includes the compression header in the WebSocket request.
  };
  WebSocketCompressionMode webSocketCompressionMode = NO_COMPRESSION;

  kj::Maybe<SecureNetworkWrapper&> tlsContext;
  // A reference to a TLS context that will be used when tlsStarter is invoked.
};

class WebSocketErrorHandler {
public:
  virtual kj::Exception handleWebSocketProtocolError(WebSocket::ProtocolError protocolError);
  // Handles low-level protocol errors in received WebSocket data.
  //
  // This is called when the WebSocket peer sends us bad data *after* a successful WebSocket
  // upgrade, e.g. a continuation frame without a preceding start frame, a frame with an unknown
  // opcode, or similar.
  //
  // You would override this method in order to customize the exception. You cannot prevent the
  // exception from being thrown.
};

kj::Own<HttpClient> newHttpClient(kj::Timer& timer, const HttpHeaderTable& responseHeaderTable,
                                  kj::Network& network, kj::Maybe<kj::Network&> tlsNetwork,
                                  HttpClientSettings settings = HttpClientSettings());
// Creates a proxy HttpClient that connects to hosts over the given network. The URL must always
// be an absolute URL; the host is parsed from the URL. This implementation will automatically
// add an appropriate Host header (and convert the URL to just a path) once it has connected.
//
// Note that if you wish to route traffic through an HTTP proxy server rather than connect to
// remote hosts directly, you should use the form of newHttpClient() that takes a NetworkAddress,
// and supply the proxy's address.
//
// `responseHeaderTable` is used when parsing HTTP responses. Requests can use any header table.
//
// `tlsNetwork` is required to support HTTPS destination URLs. If null, only HTTP URLs can be
// fetched.

kj::Own<HttpClient> newHttpClient(kj::Timer& timer, const HttpHeaderTable& responseHeaderTable,
                                  kj::NetworkAddress& addr,
                                  HttpClientSettings settings = HttpClientSettings());
// Creates an HttpClient that always connects to the given address no matter what URL is requested.
// The client will open and close connections as needed. It will attempt to reuse connections for
// multiple requests but will not send a new request before the previous response on the same
// connection has completed, as doing so can result in head-of-line blocking issues. The client may
// be used as a proxy client or a host client depending on whether the peer is operating as
// a proxy. (Hint: This is the best kind of client to use when routing traffic through an HTTP
// proxy. `addr` should be the address of the proxy, and the proxy itself will resolve remote hosts
// based on the URLs passed to it.)
//
// `responseHeaderTable` is used when parsing HTTP responses. Requests can use any header table.

kj::Own<HttpClient> newHttpClient(const HttpHeaderTable& responseHeaderTable,
                                  kj::AsyncIoStream& stream,
                                  HttpClientSettings settings = HttpClientSettings());
// Creates an HttpClient that speaks over the given pre-established connection. The client may
// be used as a proxy client or a host client depending on whether the peer is operating as
// a proxy.
//
// Note that since this client has only one stream to work with, it will try to pipeline all
// requests on this stream. If one request or response has an I/O failure, all subsequent requests
// fail as well. If the destination server chooses to close the connection after a response,
// subsequent requests will fail. If a response takes a long time, it blocks subsequent responses.
// If a WebSocket is opened successfully, all subsequent requests fail.

kj::Own<HttpClient> newConcurrencyLimitingHttpClient(
    HttpClient& inner, uint maxConcurrentRequests,
    kj::Function<void(uint runningCount, uint pendingCount)> countChangedCallback);
// Creates an HttpClient that is limited to a maximum number of concurrent requests.  Additional
// requests are queued, to be opened only after an open request completes.  `countChangedCallback`
// is called when a new connection is opened or enqueued and when an open connection is closed,
// passing the number of open and pending connections.

kj::Own<HttpClient> newHttpClient(HttpService& service);
kj::Own<HttpService> newHttpService(HttpClient& client);
// Adapts an HttpClient to an HttpService and vice versa.

kj::Own<HttpInputStream> newHttpInputStream(
    kj::AsyncInputStream& input, const HttpHeaderTable& headerTable);
// Create an HttpInputStream on top of the given stream. Normally applications would not call this
// directly, but it can be useful for implementing protocols that aren't quite HTTP but use similar
// message delimiting.
//
// The HttpInputStream implementation does read-ahead buffering on `input`. Therefore, when the
// HttpInputStream is destroyed, some data read from `input` may be lost, so it's not possible to
// continue reading from `input` in a reliable way.

kj::Own<WebSocket> newWebSocket(kj::Own<kj::AsyncIoStream> stream,
                                kj::Maybe<EntropySource&> maskEntropySource,
                                kj::Maybe<CompressionParameters> compressionConfig = nullptr,
                                kj::Maybe<WebSocketErrorHandler&> errorHandler = nullptr);
// Create a new WebSocket on top of the given stream. It is assumed that the HTTP -> WebSocket
// upgrade handshake has already occurred (or is not needed), and messages can immediately be
// sent and received on the stream. Normally applications would not call this directly.
//
// `maskEntropySource` is used to generate cryptographically-random frame masks. If null, outgoing
// frames will not be masked. Servers are required NOT to mask their outgoing frames, but clients
// ARE required to do so. So, on the client side, you MUST specify an entropy source. The mask
// must be crytographically random if the data being sent on the WebSocket may be malicious. The
// purpose of the mask is to prevent badly-written HTTP proxies from interpreting "things that look
// like HTTP requests" in a message as being actual HTTP requests, which could result in cache
// poisoning. See RFC6455 section 10.3.
//
// `compressionConfig` is an optional argument that allows us to specify how the WebSocket should
// compress and decompress messages. The configuration is determined by the
// `Sec-WebSocket-Extensions` header during WebSocket negotiation.
//
// `errorHandler` is an optional argument that lets callers throw custom exceptions for WebSocket
// protocol errors.

struct WebSocketPipe {
  kj::Own<WebSocket> ends[2];
};

WebSocketPipe newWebSocketPipe();
// Create a WebSocket pipe. Messages written to one end of the pipe will be readable from the other
// end. No buffering occurs -- a message send does not complete until a corresponding receive
// accepts the message.

class HttpServerErrorHandler;
class HttpServerCallbacks;

struct HttpServerSettings {
  kj::Duration headerTimeout = 15 * kj::SECONDS;
  // After initial connection open, or after receiving the first byte of a pipelined request,
  // the client must send the complete request within this time.

  kj::Duration pipelineTimeout = 5 * kj::SECONDS;
  // After one request/response completes, we'll wait up to this long for a pipelined request to
  // arrive.

  kj::Duration canceledUploadGracePeriod = 1 * kj::SECONDS;
  size_t canceledUploadGraceBytes = 65536;
  // If the HttpService sends a response and returns without having read the entire request body,
  // then we have to decide whether to close the connection or wait for the client to finish the
  // request so that it can pipeline the next one. We'll give them a grace period defined by the
  // above two values -- if they hit either one, we'll close the socket, but if the request
  // completes, we'll let the connection stay open to handle more requests.

  kj::Maybe<HttpServerErrorHandler&> errorHandler = nullptr;
  // Customize how client protocol errors and service application exceptions are handled by the
  // HttpServer. If null, HttpServerErrorHandler's default implementation will be used.

  kj::Maybe<HttpServerCallbacks&> callbacks = nullptr;
  // Additional optional callbacks used to control some server behavior.

  kj::Maybe<WebSocketErrorHandler&> webSocketErrorHandler = nullptr;
  // Customize exceptions thrown on WebSocket protocol errors.

  enum WebSocketCompressionMode {
    NO_COMPRESSION,
    MANUAL_COMPRESSION,    // Gives the application more control when considering whether to compress.
    AUTOMATIC_COMPRESSION, // Will perform compression parameter negotiation if client requests it.
  };
  WebSocketCompressionMode webSocketCompressionMode = NO_COMPRESSION;
};

class HttpServerErrorHandler {
public:
  virtual kj::Promise<void> handleClientProtocolError(
      HttpHeaders::ProtocolError protocolError, kj::HttpService::Response& response);
  virtual kj::Promise<void> handleApplicationError(
      kj::Exception exception, kj::Maybe<kj::HttpService::Response&> response);
  virtual kj::Promise<void> handleNoResponse(kj::HttpService::Response& response);
  // Override these functions to customize error handling during the request/response cycle.
  //
  // Client protocol errors arise when the server receives an HTTP message that fails to parse. As
  // such, HttpService::request() will not have been called yet, and the handler is always
  // guaranteed an opportunity to send a response. The default implementation of
  // handleClientProtocolError() replies with a 400 Bad Request response.
  //
  // Application errors arise when HttpService::request() throws an exception. The default
  // implementation of handleApplicationError() maps the following exception types to HTTP statuses,
  // and generates bodies from the stringified exceptions:
  //
  //   - OVERLOADED: 503 Service Unavailable
  //   - UNIMPLEMENTED: 501 Not Implemented
  //   - DISCONNECTED: (no response)
  //   - FAILED: 500 Internal Server Error
  //
  // No-response errors occur when HttpService::request() allows its promise to settle before
  // sending a response. The default implementation of handleNoResponse() replies with a 500
  // Internal Server Error response.
  //
  // Unlike `HttpService::request()`, when calling `response.send()` in the context of one of these
  // functions, a "Connection: close" header will be added, and the connection will be closed.
  //
  // Also unlike `HttpService::request()`, it is okay to return kj::READY_NOW without calling
  // `response.send()`. In this case, no response will be sent, and the connection will be closed.

  virtual void handleListenLoopException(kj::Exception&& exception);
  // Override this function to customize error handling for individual connections in the
  // `listenHttp()` overload which accepts a ConnectionReceiver reference.
  //
  // The default handler uses KJ_LOG() to log the exception as an error.
};

class HttpServerCallbacks {
public:
  virtual bool shouldClose() { return false; }
  // Whenever the HttpServer begins response headers, it will check `shouldClose()` to decide
  // whether to send a `Connection: close` header and close the connection.
  //
  // This can be useful e.g. if the server has too many connections open and wants to shed some
  // of them. Note that to implement graceful shutdown of a server, you should use
  // `HttpServer::drain()` instead.
};

class HttpServer final: private kj::TaskSet::ErrorHandler {
  // Class which listens for requests on ports or connections and sends them to an HttpService.

public:
  typedef HttpServerSettings Settings;
  typedef kj::Function<kj::Own<HttpService>(kj::AsyncIoStream&)> HttpServiceFactory;
  class SuspendableRequest;
  typedef kj::Function<kj::Maybe<kj::Own<HttpService>>(SuspendableRequest&)>
      SuspendableHttpServiceFactory;

  HttpServer(kj::Timer& timer, const HttpHeaderTable& requestHeaderTable, HttpService& service,
             Settings settings = Settings());
  // Set up an HttpServer that directs incoming connections to the given service. The service
  // may be a host service or a proxy service depending on whether you are intending to implement
  // an HTTP server or an HTTP proxy.

  HttpServer(kj::Timer& timer, const HttpHeaderTable& requestHeaderTable,
             HttpServiceFactory serviceFactory, Settings settings = Settings());
  // Like the other constructor, but allows a new HttpService object to be used for each
  // connection, based on the connection object. This is particularly useful for capturing the
  // client's IP address and injecting it as a header.

  kj::Promise<void> drain();
  // Stop accepting new connections or new requests on existing connections. Finish any requests
  // that are already executing, then close the connections. Returns once no more requests are
  // in-flight.

  kj::Promise<void> listenHttp(kj::ConnectionReceiver& port);
  // Accepts HTTP connections on the given port and directs them to the handler.
  //
  // The returned promise never completes normally. It may throw if port.accept() throws. Dropping
  // the returned promise will cause the server to stop listening on the port, but already-open
  // connections will continue to be served. Destroy the whole HttpServer to cancel all I/O.

  kj::Promise<void> listenHttp(kj::Own<kj::AsyncIoStream> connection);
  // Reads HTTP requests from the given connection and directs them to the handler. A successful
  // completion of the promise indicates that all requests received on the connection resulted in
  // a complete response, and the client closed the connection gracefully or drain() was called.
  // The promise throws if an unparsable request is received or if some I/O error occurs. Dropping
  // the returned promise will cancel all I/O on the connection and cancel any in-flight requests.

  kj::Promise<bool> listenHttpCleanDrain(kj::AsyncIoStream& connection);
  // Like listenHttp(), but allows you to potentially drain the server without closing connections.
  // The returned promise resolves to `true` if the connection has been left in a state where a
  // new HttpServer could potentially accept further requests from it. If `false`, then the
  // connection is either in an inconsistent state or already completed a closing handshake; the
  // caller should close it without any further reads/writes. Note this only ever returns `true`
  // if you called `drain()` -- otherwise this server would keep handling the connection.

  class SuspendedRequest {
    // SuspendedRequest is a representation of a request immediately after parsing the method line and
    // headers. You can obtain one of these by suspending a request by calling
    // SuspendableRequest::suspend(), then later resume the request with another call to
    // listenHttpCleanDrain().

  public:
    // Nothing, this is an opaque type.

  private:
    SuspendedRequest(kj::Array<byte>, kj::ArrayPtr<byte>, kj::OneOf<HttpMethod, HttpConnectMethod>, kj::StringPtr, HttpHeaders);

    kj::Array<byte> buffer;
    // A buffer containing at least the request's method, URL, and headers, and possibly content
    // thereafter.

    kj::ArrayPtr<byte> leftover;
    // Pointer to the end of the request headers. If this has a non-zero length, then our buffer
    // contains additional content, presumably the head of the request body.

    kj::OneOf<HttpMethod, HttpConnectMethod> method;
    kj::StringPtr url;
    HttpHeaders headers;
    // Parsed request front matter. `url` and `headers` both store pointers into `buffer`.

    friend class HttpServer;
  };

  kj::Promise<bool> listenHttpCleanDrain(kj::AsyncIoStream& connection,
      SuspendableHttpServiceFactory factory,
      kj::Maybe<SuspendedRequest> suspendedRequest = nullptr);
  // Like listenHttpCleanDrain(), but allows you to suspend requests.
  //
  // When this overload is in use, the HttpServer's default HttpService or HttpServiceFactory is not
  // used. Instead, the HttpServer reads the request method line and headers, then calls `factory`
  // with a SuspendableRequest representing the request parsed so far. The factory may then return
  // a kj::Own<HttpService> for that specific request, or it may call SuspendableRequest::suspend()
  // and return nullptr. (It is an error for the factory to return nullptr without also calling
  // suspend(); this will result in a rejected listenHttpCleanDrain() promise.)
  //
  // If the factory chooses to suspend, the listenHttpCleanDrain() promise is resolved with false
  // at the earliest opportunity.
  //
  // SuspendableRequest::suspend() returns a SuspendedRequest. You can resume this request later by
  // calling this same listenHttpCleanDrain() overload with the original connection stream, and the
  // SuspendedRequest in question.
  //
  // This overload of listenHttpCleanDrain() implements draining, as documented above. Note that the
  // returned promise will resolve to false (not clean) if a request is suspended.

private:
  class Connection;

  kj::Timer& timer;
  const HttpHeaderTable& requestHeaderTable;
  kj::OneOf<HttpService*, HttpServiceFactory> service;
  Settings settings;

  bool draining = false;
  kj::ForkedPromise<void> onDrain;
  kj::Own<kj::PromiseFulfiller<void>> drainFulfiller;

  uint connectionCount = 0;
  kj::Maybe<kj::Own<kj::PromiseFulfiller<void>>> zeroConnectionsFulfiller;

  kj::TaskSet tasks;

  HttpServer(kj::Timer& timer, const HttpHeaderTable& requestHeaderTable,
             kj::OneOf<HttpService*, HttpServiceFactory> service,
             Settings settings, kj::PromiseFulfillerPair<void> paf);

  kj::Promise<void> listenLoop(kj::ConnectionReceiver& port);

  void taskFailed(kj::Exception&& exception) override;

  kj::Promise<bool> listenHttpImpl(kj::AsyncIoStream& connection, bool wantCleanDrain);
  kj::Promise<bool> listenHttpImpl(kj::AsyncIoStream& connection,
      SuspendableHttpServiceFactory factory,
      kj::Maybe<SuspendedRequest> suspendedRequest,
      bool wantCleanDrain);
};

class HttpServer::SuspendableRequest {
  // Interface passed to the SuspendableHttpServiceFactory parameter of listenHttpCleanDrain().

public:
  kj::OneOf<HttpMethod,HttpConnectMethod> method;
  kj::StringPtr url;
  const HttpHeaders& headers;
  // Parsed request front matter, so the implementer can decide whether to suspend the request.

  SuspendedRequest suspend();
  // Signal to the HttpServer that the current request loop should be exited. Return a
  // SuspendedRequest, containing HTTP method, URL, and headers access, along with the actual header
  // buffer. The request can be later resumed with a call to listenHttpCleanDrain() using the same
  // connection.

private:
  explicit SuspendableRequest(
      Connection& connection, kj::OneOf<HttpMethod, HttpConnectMethod> method, kj::StringPtr url, const HttpHeaders& headers)
      : method(method), url(url), headers(headers), connection(connection) {}
  KJ_DISALLOW_COPY_AND_MOVE(SuspendableRequest);

  Connection& connection;

  friend class Connection;
};

// =======================================================================================
// inline implementation

inline void HttpHeaderId::requireFrom(const HttpHeaderTable& table) const {
  KJ_IREQUIRE(this->table == nullptr || this->table == &table,
      "the provided HttpHeaderId is from the wrong HttpHeaderTable");
}

inline kj::Own<HttpHeaderTable> HttpHeaderTable::Builder::build() {
  table->buildStatus = BuildStatus::FINISHED;
  return kj::mv(table);
}
inline HttpHeaderTable& HttpHeaderTable::Builder::getFutureTable() { return *table; }

inline uint HttpHeaderTable::idCount() const { return namesById.size(); }
inline bool HttpHeaderTable::isReady() const {
  switch (buildStatus) {
    case BuildStatus::UNSTARTED: return true;
    case BuildStatus::BUILDING: return false;
    case BuildStatus::FINISHED: return true;
  }

  KJ_UNREACHABLE;
}

inline kj::StringPtr HttpHeaderTable::idToString(HttpHeaderId id) const {
  id.requireFrom(*this);
  return namesById[id.id];
}

inline kj::Maybe<kj::StringPtr> HttpHeaders::get(HttpHeaderId id) const {
  id.requireFrom(*table);
  auto result = indexedHeaders[id.id];
  return result == nullptr ? kj::Maybe<kj::StringPtr>(nullptr) : result;
}

inline void HttpHeaders::unset(HttpHeaderId id) {
  id.requireFrom(*table);
  indexedHeaders[id.id] = nullptr;
}

template <typename Func>
inline void HttpHeaders::forEach(Func&& func) const {
  for (auto i: kj::indices(indexedHeaders)) {
    if (indexedHeaders[i] != nullptr) {
      func(table->idToString(HttpHeaderId(table, i)), indexedHeaders[i]);
    }
  }

  for (auto& header: unindexedHeaders) {
    func(header.name, header.value);
  }
}

template <typename Func1, typename Func2>
inline void HttpHeaders::forEach(Func1&& func1, Func2&& func2) const {
  for (auto i: kj::indices(indexedHeaders)) {
    if (indexedHeaders[i] != nullptr) {
      func1(HttpHeaderId(table, i), indexedHeaders[i]);
    }
  }

  for (auto& header: unindexedHeaders) {
    func2(header.name, header.value);
  }
}

// =======================================================================================
namespace _ { // private implementation details for WebSocket compression

kj::ArrayPtr<const char> splitNext(kj::ArrayPtr<const char>& cursor, char delimiter);

void stripLeadingAndTrailingSpace(ArrayPtr<const char>& str);

kj::Vector<kj::ArrayPtr<const char>> splitParts(kj::ArrayPtr<const char> input, char delim);

struct KeyMaybeVal {
  ArrayPtr<const char> key;
  kj::Maybe<ArrayPtr<const char>> val;
};

kj::Array<KeyMaybeVal> toKeysAndVals(const kj::ArrayPtr<kj::ArrayPtr<const char>>& params);

struct UnverifiedConfig {
  // An intermediate representation of the final `CompressionParameters` struct; used during parsing.
  // We use it to ensure the structure of an offer is generally correct, see
  // `populateUnverifiedConfig()` for details.
  bool clientNoContextTakeover = false;
  bool serverNoContextTakeover = false;
  kj::Maybe<ArrayPtr<const char>> clientMaxWindowBits = nullptr;
  kj::Maybe<ArrayPtr<const char>> serverMaxWindowBits = nullptr;
};

kj::Maybe<UnverifiedConfig> populateUnverifiedConfig(kj::Array<KeyMaybeVal>& params);

kj::Maybe<CompressionParameters> validateCompressionConfig(UnverifiedConfig&& config,
    bool isAgreement);

kj::Vector<CompressionParameters> findValidExtensionOffers(StringPtr offers);

kj::String generateExtensionRequest(const ArrayPtr<CompressionParameters>& extensions);

kj::Maybe<CompressionParameters> tryParseExtensionOffers(StringPtr offers);

kj::Maybe<CompressionParameters> tryParseAllExtensionOffers(StringPtr offers,
    CompressionParameters manualConfig);

kj::Maybe<CompressionParameters> compareClientAndServerConfigs(CompressionParameters requestConfig,
    CompressionParameters manualConfig);

kj::String generateExtensionResponse(const CompressionParameters& parameters);

kj::OneOf<CompressionParameters, kj::Exception> tryParseExtensionAgreement(
    const Maybe<CompressionParameters>& clientOffer,
    StringPtr agreedParameters);

}; // namespace _ (private)

}  // namespace kj

KJ_END_HEADER
