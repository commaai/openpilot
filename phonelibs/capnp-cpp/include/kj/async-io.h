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

#ifndef KJ_ASYNC_IO_H_
#define KJ_ASYNC_IO_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "async.h"
#include "function.h"
#include "thread.h"
#include "time.h"

struct sockaddr;

namespace kj {

#if _WIN32
class Win32EventPort;
#else
class UnixEventPort;
#endif

class NetworkAddress;
class AsyncOutputStream;

// =======================================================================================
// Streaming I/O

class AsyncInputStream {
  // Asynchronous equivalent of InputStream (from io.h).

public:
  virtual Promise<size_t> read(void* buffer, size_t minBytes, size_t maxBytes);
  virtual Promise<size_t> tryRead(void* buffer, size_t minBytes, size_t maxBytes) = 0;

  Promise<void> read(void* buffer, size_t bytes);

  virtual Maybe<uint64_t> tryGetLength();
  // Get the remaining number of bytes that will be produced by this stream, if known.
  //
  // This is used e.g. to fill in the Content-Length header of an HTTP message. If unknown, the
  // HTTP implementation may need to fall back to Transfer-Encoding: chunked.
  //
  // The default implementation always returns null.

  virtual Promise<uint64_t> pumpTo(
      AsyncOutputStream& output, uint64_t amount = kj::maxValue);
  // Read `amount` bytes from this stream (or to EOF) and write them to `output`, returning the
  // total bytes actually pumped (which is only less than `amount` if EOF was reached).
  //
  // Override this if your stream type knows how to pump itself to certain kinds of output
  // streams more efficiently than via the naive approach. You can use
  // kj::dynamicDowncastIfAvailable() to test for stream types you recognize, and if none match,
  // delegate to the default implementation.
  //
  // The default implementation first tries calling output.tryPumpFrom(), but if that fails, it
  // performs a naive pump by allocating a buffer and reading to it / writing from it in a loop.

  Promise<Array<byte>> readAllBytes();
  Promise<String> readAllText();
  // Read until EOF and return as one big byte array or string.
};

class AsyncOutputStream {
  // Asynchronous equivalent of OutputStream (from io.h).

public:
  virtual Promise<void> write(const void* buffer, size_t size) KJ_WARN_UNUSED_RESULT = 0;
  virtual Promise<void> write(ArrayPtr<const ArrayPtr<const byte>> pieces)
      KJ_WARN_UNUSED_RESULT = 0;

  virtual Maybe<Promise<uint64_t>> tryPumpFrom(
      AsyncInputStream& input, uint64_t amount = kj::maxValue);
  // Implements double-dispatch for AsyncInputStream::pumpTo().
  //
  // This method should only be called from within an implementation of pumpTo().
  //
  // This method examines the type of `input` to find optimized ways to pump data from it to this
  // output stream. If it finds one, it performs the pump. Otherwise, it returns null.
  //
  // The default implementation always returns null.
};

class AsyncIoStream: public AsyncInputStream, public AsyncOutputStream {
  // A combination input and output stream.

public:
  virtual void shutdownWrite() = 0;
  // Cleanly shut down just the write end of the stream, while keeping the read end open.

  virtual void abortRead() {}
  // Similar to shutdownWrite, but this will shut down the read end of the stream, and should only
  // be called when an error has occurred.

  virtual void getsockopt(int level, int option, void* value, uint* length);
  virtual void setsockopt(int level, int option, const void* value, uint length);
  // Corresponds to getsockopt() and setsockopt() syscalls. Will throw an "unimplemented" exception
  // if the stream is not a socket or the option is not appropriate for the socket type. The
  // default implementations always throw "unimplemented".

  virtual void getsockname(struct sockaddr* addr, uint* length);
  virtual void getpeername(struct sockaddr* addr, uint* length);
  // Corresponds to getsockname() and getpeername() syscalls. Will throw an "unimplemented"
  // exception if the stream is not a socket. The default implementations always throw
  // "unimplemented".
  //
  // Note that we don't provide methods that return NetworkAddress because it usually wouldn't
  // be useful. You can't connect() to or listen() on these addresses, obviously, because they are
  // ephemeral addresses for a single connection.
};

struct OneWayPipe {
  // A data pipe with an input end and an output end.  (Typically backed by pipe() system call.)

  Own<AsyncInputStream> in;
  Own<AsyncOutputStream> out;
};

struct TwoWayPipe {
  // A data pipe that supports sending in both directions.  Each end's output sends data to the
  // other end's input.  (Typically backed by socketpair() system call.)

  Own<AsyncIoStream> ends[2];
};

class ConnectionReceiver {
  // Represents a server socket listening on a port.

public:
  virtual Promise<Own<AsyncIoStream>> accept() = 0;
  // Accept the next incoming connection.

  virtual uint getPort() = 0;
  // Gets the port number, if applicable (i.e. if listening on IP).  This is useful if you didn't
  // specify a port when constructing the NetworkAddress -- one will have been assigned
  // automatically.

  virtual void getsockopt(int level, int option, void* value, uint* length);
  virtual void setsockopt(int level, int option, const void* value, uint length);
  // Same as the methods of AsyncIoStream.
};

// =======================================================================================
// Datagram I/O

class AncillaryMessage {
  // Represents an ancillary message (aka control message) received using the recvmsg() system
  // call (or equivalent). Most apps will not use this.

public:
  inline AncillaryMessage(int level, int type, ArrayPtr<const byte> data);
  AncillaryMessage() = default;

  inline int getLevel() const;
  // Originating protocol / socket level.

  inline int getType() const;
  // Protocol-specific message type.

  template <typename T>
  inline Maybe<const T&> as();
  // Interpret the ancillary message as the given struct type. Most ancillary messages are some
  // sort of struct, so this is a convenient way to access it. Returns nullptr if the message
  // is smaller than the struct -- this can happen if the message was truncated due to
  // insufficient ancillary buffer space.

  template <typename T>
  inline ArrayPtr<const T> asArray();
  // Interpret the ancillary message as an array of items. If the message size does not evenly
  // divide into elements of type T, the remainder is discarded -- this can happen if the message
  // was truncated due to insufficient ancillary buffer space.

private:
  int level;
  int type;
  ArrayPtr<const byte> data;
  // Message data. In most cases you should use `as()` or `asArray()`.
};

class DatagramReceiver {
  // Class encapsulating the recvmsg() system call. You must specify the DatagramReceiver's
  // capacity in advance; if a received packet is larger than the capacity, it will be truncated.

public:
  virtual Promise<void> receive() = 0;
  // Receive a new message, overwriting this object's content.
  //
  // receive() may reuse the same buffers for content and ancillary data with each call.

  template <typename T>
  struct MaybeTruncated {
    T value;

    bool isTruncated;
    // True if the Receiver's capacity was insufficient to receive the value and therefore the
    // value is truncated.
  };

  virtual MaybeTruncated<ArrayPtr<const byte>> getContent() = 0;
  // Get the content of the datagram.

  virtual MaybeTruncated<ArrayPtr<const AncillaryMessage>> getAncillary() = 0;
  // Ancilarry messages received with the datagram. See the recvmsg() system call and the cmsghdr
  // struct. Most apps don't need this.
  //
  // If the returned value is truncated, then the last message in the array may itself be
  // truncated, meaning its as<T>() method will return nullptr or its asArray<T>() method will
  // return fewer elements than expected. Truncation can also mean that additional messages were
  // available but discarded.

  virtual NetworkAddress& getSource() = 0;
  // Get the datagram sender's address.

  struct Capacity {
    size_t content = 8192;
    // How much space to allocate for the datagram content. If a datagram is received that is
    // larger than this, it will be truncated, with no way to recover the tail.

    size_t ancillary = 0;
    // How much space to allocate for ancillary messages. As with content, if the ancillary data
    // is larger than this, it will be truncated.
  };
};

class DatagramPort {
public:
  virtual Promise<size_t> send(const void* buffer, size_t size, NetworkAddress& destination) = 0;
  virtual Promise<size_t> send(ArrayPtr<const ArrayPtr<const byte>> pieces,
                               NetworkAddress& destination) = 0;

  virtual Own<DatagramReceiver> makeReceiver(
      DatagramReceiver::Capacity capacity = DatagramReceiver::Capacity()) = 0;
  // Create a new `Receiver` that can be used to receive datagrams. `capacity` specifies how much
  // space to allocate for the received message. The `DatagramPort` must outlive the `Receiver`.

  virtual uint getPort() = 0;
  // Gets the port number, if applicable (i.e. if listening on IP).  This is useful if you didn't
  // specify a port when constructing the NetworkAddress -- one will have been assigned
  // automatically.

  virtual void getsockopt(int level, int option, void* value, uint* length);
  virtual void setsockopt(int level, int option, const void* value, uint length);
  // Same as the methods of AsyncIoStream.
};

// =======================================================================================
// Networks

class NetworkAddress {
  // Represents a remote address to which the application can connect.

public:
  virtual Promise<Own<AsyncIoStream>> connect() = 0;
  // Make a new connection to this address.
  //
  // The address must not be a wildcard ("*").  If it is an IP address, it must have a port number.

  virtual Own<ConnectionReceiver> listen() = 0;
  // Listen for incoming connections on this address.
  //
  // The address must be local.

  virtual Own<DatagramPort> bindDatagramPort();
  // Open this address as a datagram (e.g. UDP) port.
  //
  // The address must be local.

  virtual Own<NetworkAddress> clone() = 0;
  // Returns an equivalent copy of this NetworkAddress.

  virtual String toString() = 0;
  // Produce a human-readable string which hopefully can be passed to Network::parseAddress()
  // to reproduce this address, although whether or not that works of course depends on the Network
  // implementation.  This should be called only to display the address to human users, who will
  // hopefully know what they are able to do with it.
};

class Network {
  // Factory for NetworkAddress instances, representing the network services offered by the
  // operating system.
  //
  // This interface typically represents broad authority, and well-designed code should limit its
  // use to high-level startup code and user interaction.  Low-level APIs should accept
  // NetworkAddress instances directly and work from there, if at all possible.

public:
  virtual Promise<Own<NetworkAddress>> parseAddress(StringPtr addr, uint portHint = 0) = 0;
  // Construct a network address from a user-provided string.  The format of the address
  // strings is not specified at the API level, and application code should make no assumptions
  // about them.  These strings should always be provided by humans, and said humans will know
  // what format to use in their particular context.
  //
  // `portHint`, if provided, specifies the "standard" IP port number for the application-level
  // service in play.  If the address turns out to be an IP address (v4 or v6), and it lacks a
  // port number, this port will be used.  If `addr` lacks a port number *and* `portHint` is
  // omitted, then the returned address will only support listen() and bindDatagramPort()
  // (not connect()), and an unused port will be chosen each time one of those methods is called.

  virtual Own<NetworkAddress> getSockaddr(const void* sockaddr, uint len) = 0;
  // Construct a network address from a legacy struct sockaddr.
};

// =======================================================================================
// I/O Provider

class AsyncIoProvider {
  // Class which constructs asynchronous wrappers around the operating system's I/O facilities.
  //
  // Generally, the implementation of this interface must integrate closely with a particular
  // `EventLoop` implementation.  Typically, the EventLoop implementation itself will provide
  // an AsyncIoProvider.

public:
  virtual OneWayPipe newOneWayPipe() = 0;
  // Creates an input/output stream pair representing the ends of a one-way pipe (e.g. created with
  // the pipe(2) system call).

  virtual TwoWayPipe newTwoWayPipe() = 0;
  // Creates two AsyncIoStreams representing the two ends of a two-way pipe (e.g. created with
  // socketpair(2) system call).  Data written to one end can be read from the other.

  virtual Network& getNetwork() = 0;
  // Creates a new `Network` instance representing the networks exposed by the operating system.
  //
  // DO NOT CALL THIS except at the highest levels of your code, ideally in the main() function.  If
  // you call this from low-level code, then you are preventing higher-level code from injecting an
  // alternative implementation.  Instead, if your code needs to use network functionality, it
  // should ask for a `Network` as a constructor or method parameter, so that higher-level code can
  // chose what implementation to use.  The system network is essentially a singleton.  See:
  //     http://www.object-oriented-security.org/lets-argue/singletons
  //
  // Code that uses the system network should not make any assumptions about what kinds of
  // addresses it will parse, as this could differ across platforms.  String addresses should come
  // strictly from the user, who will know how to write them correctly for their system.
  //
  // With that said, KJ currently supports the following string address formats:
  // - IPv4: "1.2.3.4", "1.2.3.4:80"
  // - IPv6: "1234:5678::abcd", "[1234:5678::abcd]:80"
  // - Local IP wildcard (covers both v4 and v6):  "*", "*:80"
  // - Symbolic names:  "example.com", "example.com:80", "example.com:http", "1.2.3.4:http"
  // - Unix domain: "unix:/path/to/socket"

  struct PipeThread {
    // A combination of a thread and a two-way pipe that communicates with that thread.
    //
    // The fields are intentionally ordered so that the pipe will be destroyed (and therefore
    // disconnected) before the thread is destroyed (and therefore joined).  Thus if the thread
    // arranges to exit when it detects disconnect, destruction should be clean.

    Own<Thread> thread;
    Own<AsyncIoStream> pipe;
  };

  virtual PipeThread newPipeThread(
      Function<void(AsyncIoProvider&, AsyncIoStream&, WaitScope&)> startFunc) = 0;
  // Create a new thread and set up a two-way pipe (socketpair) which can be used to communicate
  // with it.  One end of the pipe is passed to the thread's start function and the other end of
  // the pipe is returned.  The new thread also gets its own `AsyncIoProvider` instance and will
  // already have an active `EventLoop` when `startFunc` is called.
  //
  // TODO(someday):  I'm not entirely comfortable with this interface.  It seems to be doing too
  //   much at once but I'm not sure how to cleanly break it down.

  virtual Timer& getTimer() = 0;
  // Returns a `Timer` based on real time.  Time does not pass while event handlers are running --
  // it only updates when the event loop polls for system events.  This means that calling `now()`
  // on this timer does not require a system call.
  //
  // This timer is not affected by changes to the system date.  It is unspecified whether the timer
  // continues to count while the system is suspended.
};

class LowLevelAsyncIoProvider {
  // Similar to `AsyncIoProvider`, but represents a lower-level interface that may differ on
  // different operating systems.  You should prefer to use `AsyncIoProvider` over this interface
  // whenever possible, as `AsyncIoProvider` is portable and friendlier to dependency-injection.
  //
  // On Unix, this interface can be used to import native file descriptors into the async framework.
  // Different implementations of this interface might work on top of different event handling
  // primitives, such as poll vs. epoll vs. kqueue vs. some higher-level event library.
  //
  // On Windows, this interface can be used to import native HANDLEs into the async framework.
  // Different implementations of this interface might work on top of different event handling
  // primitives, such as I/O completion ports vs. completion routines.
  //
  // TODO(port):  Actually implement Windows support.

public:
  // ---------------------------------------------------------------------------
  // Unix-specific stuff

  enum Flags {
    // Flags controlling how to wrap a file descriptor.

    TAKE_OWNERSHIP = 1 << 0,
    // The returned object should own the file descriptor, automatically closing it when destroyed.
    // The close-on-exec flag will be set on the descriptor if it is not already.
    //
    // If this flag is not used, then the file descriptor is not automatically closed and the
    // close-on-exec flag is not modified.

#if !_WIN32
    ALREADY_CLOEXEC = 1 << 1,
    // Indicates that the close-on-exec flag is known already to be set, so need not be set again.
    // Only relevant when combined with TAKE_OWNERSHIP.
    //
    // On Linux, all system calls which yield new file descriptors have flags or variants which
    // set the close-on-exec flag immediately.  Unfortunately, other OS's do not.

    ALREADY_NONBLOCK = 1 << 2
    // Indicates that the file descriptor is known already to be in non-blocking mode, so the flag
    // need not be set again.  Otherwise, all wrap*Fd() methods will enable non-blocking mode
    // automatically.
    //
    // On Linux, all system calls which yield new file descriptors have flags or variants which
    // enable non-blocking mode immediately.  Unfortunately, other OS's do not.
#endif
  };

#if _WIN32
  typedef uintptr_t Fd;
  // On Windows, the `fd` parameter to each of these methods must be a SOCKET, and must have the
  // flag WSA_FLAG_OVERLAPPED (which socket() uses by default, but WSASocket() wants you to specify
  // explicitly).
#else
  typedef int Fd;
  // On Unix, any arbitrary file descriptor is supported.
#endif

  virtual Own<AsyncInputStream> wrapInputFd(Fd fd, uint flags = 0) = 0;
  // Create an AsyncInputStream wrapping a file descriptor.
  //
  // `flags` is a bitwise-OR of the values of the `Flags` enum.

  virtual Own<AsyncOutputStream> wrapOutputFd(Fd fd, uint flags = 0) = 0;
  // Create an AsyncOutputStream wrapping a file descriptor.
  //
  // `flags` is a bitwise-OR of the values of the `Flags` enum.

  virtual Own<AsyncIoStream> wrapSocketFd(Fd fd, uint flags = 0) = 0;
  // Create an AsyncIoStream wrapping a socket file descriptor.
  //
  // `flags` is a bitwise-OR of the values of the `Flags` enum.

  virtual Promise<Own<AsyncIoStream>> wrapConnectingSocketFd(
      Fd fd, const struct sockaddr* addr, uint addrlen, uint flags = 0) = 0;
  // Create an AsyncIoStream wrapping a socket and initiate a connection to the given address.
  // The returned promise does not resolve until connection has completed.
  //
  // `flags` is a bitwise-OR of the values of the `Flags` enum.

  virtual Own<ConnectionReceiver> wrapListenSocketFd(Fd fd, uint flags = 0) = 0;
  // Create an AsyncIoStream wrapping a listen socket file descriptor.  This socket should already
  // have had `bind()` and `listen()` called on it, so it's ready for `accept()`.
  //
  // `flags` is a bitwise-OR of the values of the `Flags` enum.

  virtual Own<DatagramPort> wrapDatagramSocketFd(Fd fd, uint flags = 0);

  virtual Timer& getTimer() = 0;
  // Returns a `Timer` based on real time.  Time does not pass while event handlers are running --
  // it only updates when the event loop polls for system events.  This means that calling `now()`
  // on this timer does not require a system call.
  //
  // This timer is not affected by changes to the system date.  It is unspecified whether the timer
  // continues to count while the system is suspended.
};

Own<AsyncIoProvider> newAsyncIoProvider(LowLevelAsyncIoProvider& lowLevel);
// Make a new AsyncIoProvider wrapping a `LowLevelAsyncIoProvider`.

struct AsyncIoContext {
  Own<LowLevelAsyncIoProvider> lowLevelProvider;
  Own<AsyncIoProvider> provider;
  WaitScope& waitScope;

#if _WIN32
  Win32EventPort& win32EventPort;
#else
  UnixEventPort& unixEventPort;
  // TEMPORARY: Direct access to underlying UnixEventPort, mainly for waiting on signals. This
  //   field will go away at some point when we have a chance to improve these interfaces.
#endif
};

AsyncIoContext setupAsyncIo();
// Convenience method which sets up the current thread with everything it needs to do async I/O.
// The returned objects contain an `EventLoop` which is wrapping an appropriate `EventPort` for
// doing I/O on the host system, so everything is ready for the thread to start making async calls
// and waiting on promises.
//
// You would typically call this in your main() loop or in the start function of a thread.
// Example:
//
//     int main() {
//       auto ioContext = kj::setupAsyncIo();
//
//       // Now we can call an async function.
//       Promise<String> textPromise = getHttp(*ioContext.provider, "http://example.com");
//
//       // And we can wait for the promise to complete.  Note that you can only use `wait()`
//       // from the top level, not from inside a promise callback.
//       String text = textPromise.wait(ioContext.waitScope);
//       print(text);
//       return 0;
//     }
//
// WARNING: An AsyncIoContext can only be used in the thread and process that created it. In
//   particular, note that after a fork(), an AsyncIoContext created in the parent process will
//   not work correctly in the child, even if the parent ceases to use its copy. In particular
//   note that this means that server processes which daemonize themselves at startup must wait
//   until after daemonization to create an AsyncIoContext.

// =======================================================================================
// inline implementation details

inline AncillaryMessage::AncillaryMessage(
    int level, int type, ArrayPtr<const byte> data)
    : level(level), type(type), data(data) {}

inline int AncillaryMessage::getLevel() const { return level; }
inline int AncillaryMessage::getType() const { return type; }

template <typename T>
inline Maybe<const T&> AncillaryMessage::as() {
  if (data.size() >= sizeof(T)) {
    return *reinterpret_cast<const T*>(data.begin());
  } else {
    return nullptr;
  }
}

template <typename T>
inline ArrayPtr<const T> AncillaryMessage::asArray() {
  return arrayPtr(reinterpret_cast<const T*>(data.begin()), data.size() / sizeof(T));
}

}  // namespace kj

#endif  // KJ_ASYNC_IO_H_
