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

#include <kj/async-io.h>
#include <kj/io.h>
#include "message.h"

CAPNP_BEGIN_HEADER

namespace capnp {

struct MessageReaderAndFds {
  kj::Own<MessageReader> reader;
  kj::ArrayPtr<kj::AutoCloseFd> fds;
};

struct MessageAndFds {
  kj::ArrayPtr<const kj::ArrayPtr<const word>> segments;
  kj::ArrayPtr<const int> fds;
};

class MessageStream {
  // Interface over which messages can be sent and received; virtualizes
  // the functionality above.
public:
  virtual kj::Promise<kj::Maybe<MessageReaderAndFds>> tryReadMessage(
      kj::ArrayPtr<kj::AutoCloseFd> fdSpace,
      ReaderOptions options = ReaderOptions(), kj::ArrayPtr<word> scratchSpace = nullptr) = 0;
  // Read a message that may also have file descriptors attached, e.g. from a Unix socket with
  // SCM_RIGHTS. Returns null on EOF.
  //
  // `scratchSpace`, if provided, must remain valid until the returned MessageReader is destroyed.

  kj::Promise<kj::Maybe<kj::Own<MessageReader>>> tryReadMessage(
      ReaderOptions options = ReaderOptions(),
      kj::ArrayPtr<word> scratchSpace = nullptr);
  // Equivalent to the above with fdSpace = nullptr.

  kj::Promise<MessageReaderAndFds> readMessage(
      kj::ArrayPtr<kj::AutoCloseFd> fdSpace,
      ReaderOptions options = ReaderOptions(), kj::ArrayPtr<word> scratchSpace = nullptr);
  kj::Promise<kj::Own<MessageReader>> readMessage(
      ReaderOptions options = ReaderOptions(),
      kj::ArrayPtr<word> scratchSpace = nullptr);
  // Like tryReadMessage, but throws an exception on EOF.

  virtual kj::Promise<void> writeMessage(
      kj::ArrayPtr<const int> fds,
      kj::ArrayPtr<const kj::ArrayPtr<const word>> segments)
    KJ_WARN_UNUSED_RESULT = 0;
  kj::Promise<void> writeMessage(
      kj::ArrayPtr<const int> fds,
      MessageBuilder& builder)
    KJ_WARN_UNUSED_RESULT;
  // Write a message with FDs attached, e.g. to a Unix socket with SCM_RIGHTS.
  // The parameters must remain valid until the returned promise resolves.

  kj::Promise<void> writeMessage(
      kj::ArrayPtr<const kj::ArrayPtr<const word>> segments)
    KJ_WARN_UNUSED_RESULT;
  kj::Promise<void> writeMessage(MessageBuilder& builder)
      KJ_WARN_UNUSED_RESULT;
  // Equivalent to the above with fds = nullptr.

  kj::Promise<void> writeMessages(
      kj::ArrayPtr<MessageAndFds> messages)
    KJ_WARN_UNUSED_RESULT;
  virtual kj::Promise<void> writeMessages(
      kj::ArrayPtr<kj::ArrayPtr<const kj::ArrayPtr<const word>>> messages)
    KJ_WARN_UNUSED_RESULT = 0;
  kj::Promise<void> writeMessages(kj::ArrayPtr<MessageBuilder*> builders)
      KJ_WARN_UNUSED_RESULT;
  // Similar to the above, but for writing multiple messages at a time in a batch.

  virtual kj::Maybe<int> getSendBufferSize() = 0;
  // Get the size of the underlying send buffer, if applicable. The RPC
  // system uses this as a hint for flow control purposes; see:
  //
  // https://capnproto.org/news/2020-04-23-capnproto-0.8.html#multi-stream-flow-control
  //
  // ...for a more thorough explanation of how this is used. Implementations
  // may return nullptr if they do not have access to this information, or if
  // the underlying transport does not use a congestion window.

  virtual kj::Promise<void> end() = 0;
  // Cleanly shut down just the write end of the transport, while keeping the read end open.

};

class AsyncIoMessageStream final: public MessageStream {
  // A MessageStream that wraps an AsyncIoStream.
public:
  explicit AsyncIoMessageStream(kj::AsyncIoStream& stream);

  // Implements MessageStream
  kj::Promise<kj::Maybe<MessageReaderAndFds>> tryReadMessage(
      kj::ArrayPtr<kj::AutoCloseFd> fdSpace,
      ReaderOptions options = ReaderOptions(), kj::ArrayPtr<word> scratchSpace = nullptr) override;
  kj::Promise<void> writeMessage(
      kj::ArrayPtr<const int> fds,
      kj::ArrayPtr<const kj::ArrayPtr<const word>> segments) override;
  kj::Promise<void> writeMessages(
      kj::ArrayPtr<kj::ArrayPtr<const kj::ArrayPtr<const word>>> messages) override;
  kj::Maybe<int> getSendBufferSize() override;

  kj::Promise<void> end() override;

  // Make sure the overridden virtual methods don't hide the non-virtual methods.
  using MessageStream::tryReadMessage;
  using MessageStream::writeMessage;
private:
  kj::AsyncIoStream& stream;
};

class AsyncCapabilityMessageStream final: public MessageStream {
  // A MessageStream that wraps an AsyncCapabilityStream.
public:
  explicit AsyncCapabilityMessageStream(kj::AsyncCapabilityStream& stream);

  // Implements MessageStream
  kj::Promise<kj::Maybe<MessageReaderAndFds>> tryReadMessage(
      kj::ArrayPtr<kj::AutoCloseFd> fdSpace,
      ReaderOptions options = ReaderOptions(), kj::ArrayPtr<word> scratchSpace = nullptr) override;
  kj::Promise<void> writeMessage(
      kj::ArrayPtr<const int> fds,
      kj::ArrayPtr<const kj::ArrayPtr<const word>> segments) override;
  kj::Promise<void> writeMessages(
      kj::ArrayPtr<kj::ArrayPtr<const kj::ArrayPtr<const word>>> messages) override;
  kj::Maybe<int> getSendBufferSize() override;
  kj::Promise<void> end() override;

  // Make sure the overridden virtual methods don't hide the non-virtual methods.
  using MessageStream::tryReadMessage;
  using MessageStream::writeMessage;
private:
  kj::AsyncCapabilityStream& stream;
};

class BufferedMessageStream final: public MessageStream {
  // A MessageStream that reads into a buffer in the hopes of receiving multiple messages in a
  // single system call. Compared to the other implementations, this implementation is expected
  // to be faster when reading from an OS stream (but probably not when reading from an in-memory
  // async pipe). It has the down sides of using more memory (for the buffer) and requiring extra
  // copies.

public:
  using IsShortLivedCallback = kj::Function<bool(MessageReader&)>;
  // Callback function which decides whether a message will be "short-lived", meaning that it is
  // guaranteed to be dropped before the next message is read. The stream uses this as an
  // optimization to decide whether it can return a MessageReader pointing into the buffer, which
  // will be reused for future reads. For long-lived messages, the stream must copy the content
  // into a separate buffer.

  explicit BufferedMessageStream(
      kj::AsyncIoStream& stream, IsShortLivedCallback isShortLivedCallback,
      size_t bufferSizeInWords = 8192);
  explicit BufferedMessageStream(
      kj::AsyncCapabilityStream& stream, IsShortLivedCallback isShortLivedCallback,
      size_t bufferSizeInWords = 8192);

  // Implements MessageStream
  kj::Promise<kj::Maybe<MessageReaderAndFds>> tryReadMessage(
      kj::ArrayPtr<kj::AutoCloseFd> fdSpace,
      ReaderOptions options = ReaderOptions(), kj::ArrayPtr<word> scratchSpace = nullptr) override;
  kj::Promise<void> writeMessage(
      kj::ArrayPtr<const int> fds,
      kj::ArrayPtr<const kj::ArrayPtr<const word>> segments) override;
  kj::Promise<void> writeMessages(
      kj::ArrayPtr<kj::ArrayPtr<const kj::ArrayPtr<const word>>> messages) override;
  kj::Maybe<int> getSendBufferSize() override;
  kj::Promise<void> end() override;

  // Make sure the overridden virtual methods don't hide the non-virtual methods.
  using MessageStream::tryReadMessage;
  using MessageStream::writeMessage;

private:
  kj::AsyncIoStream& stream;
  kj::Maybe<kj::AsyncCapabilityStream&> capStream;
  IsShortLivedCallback isShortLivedCallback;

  kj::Array<word> buffer;

  word* beginData;
  // Pointer to location in `buffer` where the next message starts. This is always on a word
  // boundray since messages are always a whole number of words.

  kj::byte* beginAvailable;
  // Pointer to the location in `buffer` where unused buffer space begins, i.e. immediately after
  // the last byte read.

  kj::Vector<kj::AutoCloseFd> leftoverFds;
  // FDs which were accidentally read too early. These are always connected to the last message
  // in the buffer, since the OS would not have allowed us to read past that point.

  bool hasOutstandingShortLivedMessage = false;

  kj::Promise<kj::Maybe<MessageReaderAndFds>> tryReadMessageImpl(
      kj::ArrayPtr<kj::AutoCloseFd> fdSpace, size_t fdsSoFar,
      ReaderOptions options, kj::ArrayPtr<word> scratchSpace);

  kj::Promise<kj::Maybe<MessageReaderAndFds>> readEntireMessage(
      kj::ArrayPtr<const byte> prefix, size_t expectedSizeInWords,
      kj::ArrayPtr<kj::AutoCloseFd> fdSpace, size_t fdsSoFar,
      ReaderOptions options);
  // Given a message prefix and expected size of the whole message, read the entire message into
  // a single array and return it.

  kj::Promise<kj::AsyncCapabilityStream::ReadResult> tryReadWithFds(
      void* buffer, size_t minBytes, size_t maxBytes, kj::AutoCloseFd* fdBuffer, size_t maxFds);
  // Executes AsyncCapabilityStream::tryReadWithFds() on the underlying stream, or falls back to
  // AsyncIoStream::tryRead() if it's not a capability stream.

  class MessageReaderImpl;
};

// -----------------------------------------------------------------------------
// Stand-alone functions for reading & writing messages on AsyncInput/AsyncOutputStreams.
//
// In general, foo(stream, ...) is equivalent to
// AsyncIoMessageStream(stream).foo(...), whenever the latter would type check.
//
// The first argument must remain valid until the returned promise resolves
// (or is canceled).

kj::Promise<kj::Own<MessageReader>> readMessage(
    kj::AsyncInputStream& input, ReaderOptions options = ReaderOptions(),
    kj::ArrayPtr<word> scratchSpace = nullptr);

kj::Promise<kj::Maybe<kj::Own<MessageReader>>> tryReadMessage(
    kj::AsyncInputStream& input, ReaderOptions options = ReaderOptions(),
    kj::ArrayPtr<word> scratchSpace = nullptr);

kj::Promise<void> writeMessage(kj::AsyncOutputStream& output,
                               kj::ArrayPtr<const kj::ArrayPtr<const word>> segments)
    KJ_WARN_UNUSED_RESULT;

kj::Promise<void> writeMessage(kj::AsyncOutputStream& output, MessageBuilder& builder)
    KJ_WARN_UNUSED_RESULT;

// -----------------------------------------------------------------------------
// Stand-alone versions that support FD passing.
//
// For each of these, `foo(stream, ...)` is equivalent to
// `AsyncCapabilityMessageStream(stream).foo(...)`.

kj::Promise<MessageReaderAndFds> readMessage(
    kj::AsyncCapabilityStream& input, kj::ArrayPtr<kj::AutoCloseFd> fdSpace,
    ReaderOptions options = ReaderOptions(), kj::ArrayPtr<word> scratchSpace = nullptr);

kj::Promise<kj::Maybe<MessageReaderAndFds>> tryReadMessage(
    kj::AsyncCapabilityStream& input, kj::ArrayPtr<kj::AutoCloseFd> fdSpace,
    ReaderOptions options = ReaderOptions(), kj::ArrayPtr<word> scratchSpace = nullptr);

kj::Promise<void> writeMessage(kj::AsyncCapabilityStream& output, kj::ArrayPtr<const int> fds,
                               kj::ArrayPtr<const kj::ArrayPtr<const word>> segments)
    KJ_WARN_UNUSED_RESULT;
kj::Promise<void> writeMessage(kj::AsyncCapabilityStream& output, kj::ArrayPtr<const int> fds,
                               MessageBuilder& builder)
    KJ_WARN_UNUSED_RESULT;


// -----------------------------------------------------------------------------
// Stand-alone functions for writing multiple messages at once on AsyncOutputStreams.

kj::Promise<void> writeMessages(kj::AsyncOutputStream& output,
                                kj::ArrayPtr<kj::ArrayPtr<const kj::ArrayPtr<const word>>> messages)
    KJ_WARN_UNUSED_RESULT;

kj::Promise<void> writeMessages(
    kj::AsyncOutputStream& output, kj::ArrayPtr<MessageBuilder*> builders)
    KJ_WARN_UNUSED_RESULT;

// =======================================================================================
// inline implementation details

inline kj::Promise<void> writeMessage(kj::AsyncOutputStream& output, MessageBuilder& builder) {
  return writeMessage(output, builder.getSegmentsForOutput());
}
inline kj::Promise<void> writeMessage(
    kj::AsyncCapabilityStream& output, kj::ArrayPtr<const int> fds, MessageBuilder& builder) {
  return writeMessage(output, fds, builder.getSegmentsForOutput());
}

inline kj::Promise<void> MessageStream::writeMessage(kj::ArrayPtr<const kj::ArrayPtr<const word>> segments) {
  return writeMessage(nullptr, segments);
}

inline kj::Promise<void> MessageStream::writeMessage(MessageBuilder& builder) {
  return writeMessage(builder.getSegmentsForOutput());
}

inline kj::Promise<void> MessageStream::writeMessage(
    kj::ArrayPtr<const int> fds, MessageBuilder& builder) {
  return writeMessage(fds, builder.getSegmentsForOutput());
}

}  // namespace capnp

CAPNP_END_HEADER
