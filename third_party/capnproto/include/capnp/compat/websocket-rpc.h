// Copyright (c) 2021 Ian Denhardt and contributors
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

#include <kj/compat/http.h>
#include <capnp/serialize-async.h>

CAPNP_BEGIN_HEADER

namespace capnp {

class WebSocketMessageStream final : public MessageStream {
  // An implementation of MessageStream that sends messages over a websocket.
  //
  // Each capnproto message is sent in a single binary websocket frame.
public:
  WebSocketMessageStream(kj::WebSocket& socket);

  // Implements MessageStream
  kj::Promise<kj::Maybe<MessageReaderAndFds>> tryReadMessage(
      kj::ArrayPtr<kj::AutoCloseFd> fdSpace,
      ReaderOptions options = ReaderOptions(), kj::ArrayPtr<word> scratchSpace = nullptr) override;
  kj::Promise<void> writeMessage(
      kj::ArrayPtr<const int> fds,
      kj::ArrayPtr<const kj::ArrayPtr<const word>> segments) override
    KJ_WARN_UNUSED_RESULT;
  kj::Promise<void> writeMessages(
      kj::ArrayPtr<kj::ArrayPtr<const kj::ArrayPtr<const word>>> messages) override
    KJ_WARN_UNUSED_RESULT;
  kj::Maybe<int> getSendBufferSize() override;
  kj::Promise<void> end() override;
private:
  kj::WebSocket& socket;
};

}  // namespace capnp

CAPNP_END_HEADER
