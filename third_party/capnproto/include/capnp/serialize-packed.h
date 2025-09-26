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

#include "serialize.h"

CAPNP_BEGIN_HEADER

namespace capnp {

namespace _ {  // private

class PackedInputStream: public kj::InputStream {
  // An input stream that unpacks packed data with a picky constraint:  The caller must read data
  // in the exact same size and sequence as the data was written to PackedOutputStream.

public:
  explicit PackedInputStream(kj::BufferedInputStream& inner);
  KJ_DISALLOW_COPY_AND_MOVE(PackedInputStream);
  ~PackedInputStream() noexcept(false);

  // implements InputStream ------------------------------------------
  size_t tryRead(void* buffer, size_t minBytes, size_t maxBytes) override;
  void skip(size_t bytes) override;

private:
  kj::BufferedInputStream& inner;
};

class PackedOutputStream: public kj::OutputStream {
  // An output stream that packs data. Buffers passed to `write()` must be word-aligned.
public:
  explicit PackedOutputStream(kj::BufferedOutputStream& inner);
  KJ_DISALLOW_COPY_AND_MOVE(PackedOutputStream);
  ~PackedOutputStream() noexcept(false);

  // implements OutputStream -----------------------------------------
  void write(const void* buffer, size_t bytes) override;

private:
  kj::BufferedOutputStream& inner;
};

}  // namespace _ (private)

class PackedMessageReader: private _::PackedInputStream, public InputStreamMessageReader {
public:
  PackedMessageReader(kj::BufferedInputStream& inputStream, ReaderOptions options = ReaderOptions(),
                      kj::ArrayPtr<word> scratchSpace = nullptr);
  KJ_DISALLOW_COPY_AND_MOVE(PackedMessageReader);
  ~PackedMessageReader() noexcept(false);
};

class PackedFdMessageReader: private kj::FdInputStream, private kj::BufferedInputStreamWrapper,
                             public PackedMessageReader {
public:
  PackedFdMessageReader(int fd, ReaderOptions options = ReaderOptions(),
                        kj::ArrayPtr<word> scratchSpace = nullptr);
  // Read message from a file descriptor, without taking ownership of the descriptor.
  // Note that if you want to reuse the descriptor after the reader is destroyed, you'll need to
  // seek it, since otherwise the position is unspecified.

  PackedFdMessageReader(kj::AutoCloseFd fd, ReaderOptions options = ReaderOptions(),
                        kj::ArrayPtr<word> scratchSpace = nullptr);
  // Read a message from a file descriptor, taking ownership of the descriptor.

  KJ_DISALLOW_COPY_AND_MOVE(PackedFdMessageReader);

  ~PackedFdMessageReader() noexcept(false);
};

void writePackedMessage(kj::BufferedOutputStream& output, MessageBuilder& builder);
void writePackedMessage(kj::BufferedOutputStream& output,
                        kj::ArrayPtr<const kj::ArrayPtr<const word>> segments);
// Write a packed message to a buffered output stream.

void writePackedMessage(kj::OutputStream& output, MessageBuilder& builder);
void writePackedMessage(kj::OutputStream& output,
                        kj::ArrayPtr<const kj::ArrayPtr<const word>> segments);
// Write a packed message to an unbuffered output stream.  If you intend to write multiple messages
// in succession, consider wrapping your output in a buffered stream in order to reduce system
// call overhead.

void writePackedMessageToFd(int fd, MessageBuilder& builder);
void writePackedMessageToFd(int fd, kj::ArrayPtr<const kj::ArrayPtr<const word>> segments);
// Write a single packed message to the file descriptor.

size_t computeUnpackedSizeInWords(kj::ArrayPtr<const byte> packedBytes);
// Computes the number of words to which the given packed bytes will unpack. Not intended for use
// in performance-sensitive situations.

// =======================================================================================
// inline stuff

inline void writePackedMessage(kj::BufferedOutputStream& output, MessageBuilder& builder) {
  writePackedMessage(output, builder.getSegmentsForOutput());
}

inline void writePackedMessage(kj::OutputStream& output, MessageBuilder& builder) {
  writePackedMessage(output, builder.getSegmentsForOutput());
}

inline void writePackedMessageToFd(int fd, MessageBuilder& builder) {
  writePackedMessageToFd(fd, builder.getSegmentsForOutput());
}

}  // namespace capnp

CAPNP_END_HEADER
