# Copyright (c) 2019 Cloudflare, Inc. and contributors
# Licensed under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

@0x86c366a91393f3f8;
# Defines placeholder types used to provide backwards-compatibility while introducing streaming
# to the language. The goal is that old code generators that don't know about streaming can still
# generate code that functions, leaving it up to the application to implement flow control
# manually.

$import "/capnp/c++.capnp".namespace("capnp");

struct StreamResult @0x995f9a3377c0b16e {
  # Empty struct that serves as the return type for "streaming" methods.
  #
  # Defining a method like:
  #
  #     write @0 (bytes :Data) -> stream;
  #
  # Is equivalent to:
  #
  #     write @0 (bytes :Data) -> import "/capnp/stream.capnp".StreamResult;
  #
  # However, implementations that recognize streaming will elide the reference to StreamResult
  # and instead give write() a different signature appropriate for streaming.
  #
  # Streaming methods do not return a result -- that is, they return Promise<void>. This promise
  # resolves not to indicate that the call was actually delivered, but instead to provide
  # backpressure. When the previous call's promise resolves, it is time to make another call. On
  # the client side, the RPC system will resolve promises immediately until an appropriate number
  # of requests are in-flight, and then will delay promise resolution to apply back-pressure.
  # On the server side, the RPC system will deliver one call at a time.
}
