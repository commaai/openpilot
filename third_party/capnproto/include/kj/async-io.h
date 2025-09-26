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

#include "async.h"
#include <kj/function.h>
#include <kj/thread.h>
#include <kj/timer.h>

KJ_BEGIN_HEADER

struct sockaddr;

namespace kj {

#if _WIN32
class Win32EventPort;
class AutoCloseHandle;
#else
class UnixEventPort;
#endif

class AutoCloseFd;
class NetworkAddress;
class AsyncOutputStream;
class AsyncIoStream;
class AncillaryMessage;

class ReadableFile;
class File;

// =======================================================================================
// Streaming I/O

class AsyncInputStream: private AsyncObject {
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

  Promise<Array<byte>> readAllBytes(uint64_t limit = kj::maxValue);
  Promise<String> readAllText(uint64_t limit = kj::maxValue);
  // Read until EOF and return as one big byte array or string. Throw an exception if EOF is not
  // seen before reading `limit` bytes.
  //
  // To prevent runaway memory allocation, consider using a more conservative value for `limit` than
  // the default, particularly on untrusted data streams which may never see EOF.

  virtual void registerAncillaryMessageHandler(Function<void(ArrayPtr<AncillaryMessage>)> fn);
  // Register interest in checking for ancillary messages (aka control messages) when reading.
  // The provided callback will be called whenever any are encountered. The messages passed to
  // the function do not live beyond when function returns.
  // Only supported on Unix (the default impl throws UNIMPLEMENTED). Most apps will not use this.

  virtual Maybe<Own<AsyncInputStream>> tryTee(uint64_t limit = kj::maxValue);
  // Primarily intended as an optimization for the `tee` call. Returns an input stream whose state
  // is independent from this one but which will return the exact same set of bytes read going
  // forward. limit is a total limit on the amount of memory, in bytes, which a tee implementation
  // may use to buffer stream data. An implementation must throw an exception if a read operation
  // would cause the limit to be exceeded. If tryTee() can see that the new limit is impossible to
  // satisfy, it should return nullptr so that the pessimized path is taken in newTee. This is
  // likely to arise if tryTee() is called twice with different limits on the same stream.
};

class AsyncOutputStream: private AsyncObject {
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

  virtual Promise<void> whenWriteDisconnected() = 0;
  // Returns a promise that resolves when the stream has become disconnected such that new write()s
  // will fail with a DISCONNECTED exception. This is particularly useful, for example, to cancel
  // work early when it is detected that no one will receive the result.
  //
  // Note that not all streams are able to detect this condition without actually performing a
  // write(); such stream implementations may return a promise that never resolves. (In particular,
  // as of this writing, whenWriteDisconnected() is not implemented on Windows. Also, for TCP
  // streams, not all disconnects are detectable -- a power or network failure may lead the
  // connection to hang forever, or until configured socket options lead to a timeout.)
  //
  // Unlike most other asynchronous stream methods, it is safe to call whenWriteDisconnected()
  // multiple times without canceling the previous promises.
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

  virtual kj::Maybe<int> getFd() const { return nullptr; }
  // Get the underlying Unix file descriptor, if any. Returns nullptr if this object actually
  // isn't wrapping a file descriptor.
};

Promise<uint64_t> unoptimizedPumpTo(
    AsyncInputStream& input, AsyncOutputStream& output, uint64_t amount,
    uint64_t completedSoFar = 0);
// Performs a pump using read() and write(), without calling the stream's pumpTo() nor
// tryPumpFrom() methods. This is intended to be used as a fallback by implementations of pumpTo()
// and tryPumpFrom() when they want to give up on optimization, but can't just call pumpTo() again
// because this would recursively retry the optimization. unoptimizedPumpTo() should only be called
// inside implementations of streams, never by the caller of a stream -- use the pumpTo() method
// instead.
//
// `completedSoFar` is the number of bytes out of `amount` that have already been pumped. This is
// provided for convenience for cases where the caller has already done some pumping before they
// give up. Otherwise, a `.then()` would need to be used to add the bytes to the final result.

class AsyncCapabilityStream: public AsyncIoStream {
  // An AsyncIoStream that also allows transmitting new stream objects and file descriptors
  // (capabilities, in the object-capability model sense), in addition to bytes.
  //
  // Capabilities can be attached to bytes when they are written. On the receiving end, the read()
  // that receives the first byte of such a message will also receive the capabilities.
  //
  // Note that AsyncIoStream's regular byte-oriented methods can be used on AsyncCapabilityStream,
  // with the effect of silently dropping any capabilities attached to the respective bytes. E.g.
  // using `AsyncIoStream::tryRead()` to read bytes that had been sent with `writeWithFds()` will
  // silently drop the FDs (closing them if appropriate). Also note that pumping a stream with
  // `pumpTo()` always drops all capabilities attached to the pumped data. (TODO(someday): Do we
  // want a version of pumpTo() that preserves capabilities?)
  //
  // On Unix, KJ provides an implementation based on Unix domain sockets and file descriptor
  // passing via SCM_RIGHTS. Due to the nature of SCM_RIGHTS, if the application accidentally
  // read()s when it should have called receiveStream(), it will observe a NUL byte in the data
  // and the capability will be discarded. Of course, an application should not depend on this
  // behavior; it should avoid read()ing through a capability.
  //
  // KJ does not provide any inter-process implementation of this type on Windows, as there's no
  // obvious implementation there. Handle passing on Windows requires at least one of the processes
  // involved to have permission to modify the other's handle table, which is effectively full
  // control. Handle passing between mutually non-trusting processes would require a trusted
  // broker process to facilitate. One could possibly implement this type in terms of such a
  // broker, or in terms of direct handle passing if at least one process trusts the other.

public:
  virtual Promise<void> writeWithFds(ArrayPtr<const byte> data,
                                     ArrayPtr<const ArrayPtr<const byte>> moreData,
                                     ArrayPtr<const int> fds) = 0;
  Promise<void> writeWithFds(ArrayPtr<const byte> data,
                             ArrayPtr<const ArrayPtr<const byte>> moreData,
                             ArrayPtr<const AutoCloseFd> fds);
  // Write some data to the stream with some file descriptors attached to it.
  //
  // The maximum number of FDs that can be sent at a time is usually subject to an OS-imposed
  // limit. On Linux, this is 253. In practice, sending more than a handful of FDs at once is
  // probably a bad idea.

  struct ReadResult {
    size_t byteCount;
    size_t capCount;
  };

  virtual Promise<ReadResult> tryReadWithFds(void* buffer, size_t minBytes, size_t maxBytes,
                                             AutoCloseFd* fdBuffer, size_t maxFds) = 0;
  // Read data from the stream that may have file descriptors attached. Any attached descriptors
  // will be placed in `fdBuffer`. If multiple bundles of FDs are encountered in the course of
  // reading the amount of data requested by minBytes/maxBytes, then they will be concatenated. If
  // more FDs are received than fit in the buffer, then the excess will be discarded and closed --
  // this behavior, while ugly, is important to defend against denial-of-service attacks that may
  // fill up the FD table with garbage. Applications must think carefully about how many FDs they
  // really need to receive at once and set a well-defined limit.

  virtual Promise<void> writeWithStreams(ArrayPtr<const byte> data,
                                         ArrayPtr<const ArrayPtr<const byte>> moreData,
                                         Array<Own<AsyncCapabilityStream>> streams) = 0;
  virtual Promise<ReadResult> tryReadWithStreams(
      void* buffer, size_t minBytes, size_t maxBytes,
      Own<AsyncCapabilityStream>* streamBuffer, size_t maxStreams) = 0;
  // Like above, but passes AsyncCapabilityStream objects. The stream implementations must be from
  // the same AsyncIoProvider.

  // ---------------------------------------------------------------------------
  // Helpers for sending individual capabilities.
  //
  // These are equivalent to the above methods with the constraint that only one FD is
  // sent/received at a time and the corresponding data is a single zero-valued byte.

  Promise<Own<AsyncCapabilityStream>> receiveStream();
  Promise<Maybe<Own<AsyncCapabilityStream>>> tryReceiveStream();
  Promise<void> sendStream(Own<AsyncCapabilityStream> stream);
  // Transfer a single stream.

  Promise<AutoCloseFd> receiveFd();
  Promise<Maybe<AutoCloseFd>> tryReceiveFd();
  Promise<void> sendFd(int fd);
  // Transfer a single raw file descriptor.
};

struct OneWayPipe {
  // A data pipe with an input end and an output end.  (Typically backed by pipe() system call.)

  Own<AsyncInputStream> in;
  Own<AsyncOutputStream> out;
};

OneWayPipe newOneWayPipe(kj::Maybe<uint64_t> expectedLength = nullptr);
// Constructs a OneWayPipe that operates in-process. The pipe does not do any buffering -- it waits
// until both a read() and a write() call are pending, then resolves both.
//
// If `expectedLength` is non-null, then the pipe will be expected to transmit exactly that many
// bytes. The input end's `tryGetLength()` will return the number of bytes left.

struct TwoWayPipe {
  // A data pipe that supports sending in both directions.  Each end's output sends data to the
  // other end's input.  (Typically backed by socketpair() system call.)

  Own<AsyncIoStream> ends[2];
};

TwoWayPipe newTwoWayPipe();
// Constructs a TwoWayPipe that operates in-process. The pipe does not do any buffering -- it waits
// until both a read() and a write() call are pending, then resolves both.

struct CapabilityPipe {
  // Like TwoWayPipe but allowing capability-passing.

  Own<AsyncCapabilityStream> ends[2];
};

CapabilityPipe newCapabilityPipe();
// Like newTwoWayPipe() but creates a capability pipe.
//
// The requirement of `writeWithStreams()` that "The stream implementations must be from the same
// AsyncIoProvider." does not apply to this pipe; any kind of AsyncCapabilityStream implementation
// is supported.
//
// This implementation does not know how to convert streams to FDs or vice versa; if you write FDs
// you must read FDs, and if you write streams you must read streams.

struct Tee {
  // Two AsyncInputStreams which each read the same data from some wrapped inner AsyncInputStream.

  Own<AsyncInputStream> branches[2];
};

Tee newTee(Own<AsyncInputStream> input, uint64_t limit = kj::maxValue);
// Constructs a Tee that operates in-process. The tee buffers data if any read or pump operations is
// called on one of the two input ends. If a read or pump operation is subsequently called on the
// other input end, the buffered data is consumed.
//
// `pumpTo()` operations on the input ends will proactively read from the inner stream and block
// while writing to the output stream. While one branch has an active `pumpTo()` operation, any
// `tryRead()` operation on the other branch will not be allowed to read faster than allowed by the
// pump's backpressure. (In other words, it will never cause buffering on the pump.) Similarly, if
// there are `pumpTo()` operations active on both branches, the greater of the two backpressures is
// respected -- the two pumps progress in lockstep, and there is no buffering.
//
// At no point will a branch's buffer be allowed to grow beyond `limit` bytes. If the buffer would
// grow beyond the limit, an exception is generated, which both branches see once they have
// exhausted their buffers.
//
// It is recommended that you use a more conservative value for `limit` than the default.

Own<AsyncOutputStream> newPromisedStream(Promise<Own<AsyncOutputStream>> promise);
Own<AsyncIoStream> newPromisedStream(Promise<Own<AsyncIoStream>> promise);
// Constructs an Async*Stream which waits for a promise to resolve, then forwards all calls to the
// promised stream.

// =======================================================================================
// Authenticated streams

class PeerIdentity {
  // PeerIdentity provides information about a connecting client. Various subclasses exist to
  // address different network types.
public:
  virtual kj::String toString() = 0;
  // Returns a human-readable string identifying the peer. Where possible, this string will be
  // in the same format as the addresses you could pass to `kj::Network::parseAddress()`. However,
  // only certain subclasses of `PeerIdentity` guarantee this property.
};

struct AuthenticatedStream {
  // A pair of an `AsyncIoStream` and a `PeerIdentity`. This is used as the return type of
  // `NetworkAddress::connectAuthenticated()` and `ConnectionReceiver::acceptAuthenticated()`.

  Own<AsyncIoStream> stream;
  // The byte stream.

  Own<PeerIdentity> peerIdentity;
  // An object indicating who is at the other end of the stream.
  //
  // Different subclasses of `PeerIdentity` are used in different situations:
  // - TCP connections will use NetworkPeerIdentity, which gives the network address of the client.
  // - Local (unix) socket connections will use LocalPeerIdentity, which identifies the UID
  //   and PID of the process that initiated the connection.
  // - TLS connections will use TlsPeerIdentity which provides details of the client certificate,
  //   if any was provided.
  // - When no meaningful peer identity can be provided, `UnknownPeerIdentity` is returned.
  //
  // Implementations of `Network`, `ConnectionReceiver`, `NetworkAddress`, etc. should document the
  // specific assumptions the caller can make about the type of `PeerIdentity`s used, allowing for
  // identities to be statically downcast if the right conditions are met. In the absence of
  // documented promises, RTTI may be needed to query the type.
};

class NetworkPeerIdentity: public PeerIdentity {
  // PeerIdentity used for network protocols like TCP/IP. This identifies the remote peer.
  //
  // This is only "authenticated" to the extent that we know data written to the stream will be
  // routed to the given address. This does not preclude the possibility of man-in-the-middle
  // attacks by attackers who are able to manipulate traffic along the route.
public:
  virtual NetworkAddress& getAddress() = 0;
  // Obtain the peer's address as a NetworkAddress object. The returned reference's lifetime is the
  // same as the `NetworkPeerIdentity`, but you can always call `clone()` on it to get a copy that
  // lives longer.

  static kj::Own<NetworkPeerIdentity> newInstance(kj::Own<NetworkAddress> addr);
  // Construct an instance of this interface wrapping the given address.
};

class LocalPeerIdentity: public PeerIdentity {
  // PeerIdentity used for connections between processes on the local machine -- in particular,
  // Unix sockets.
  //
  // (This interface probably isn't useful on Windows.)
public:
  struct Credentials {
    kj::Maybe<int> pid;
    kj::Maybe<uint> uid;

    // We don't cover groups at present because some systems produce a list of groups while others
    // only provide the peer's main group, the latter being pretty useless.
  };

  virtual Credentials getCredentials() = 0;
  // Get the PID and UID of the peer process, if possible.
  //
  // Either ID may be null if the peer could not be identified. Some operating systems do not
  // support retrieving these credentials, or can only provide one or the other. Some situations
  // (like user and PID namespaces on Linux) may also make it impossible to represent the peer's
  // credentials accurately.
  //
  // Note the meaning here can be subtle. Multiple processes can potentially have the socket in
  // their file descriptor tables. The identified process is the one who called `connect()` or
  // `listen()`.
  //
  // On Linux this is implemented with SO_PEERCRED.

  static kj::Own<LocalPeerIdentity> newInstance(Credentials creds);
  // Construct an instance of this interface wrapping the given credentials.
};

class UnknownPeerIdentity: public PeerIdentity {
public:
  static kj::Own<UnknownPeerIdentity> newInstance();
  // Get an instance of this interface. This actually always returns the same instance with no
  // memory allocation.
};

// =======================================================================================
// Accepting connections

class ConnectionReceiver: private AsyncObject {
  // Represents a server socket listening on a port.

public:
  virtual Promise<Own<AsyncIoStream>> accept() = 0;
  // Accept the next incoming connection.

  virtual Promise<AuthenticatedStream> acceptAuthenticated();
  // Accept the next incoming connection, and also provide a PeerIdentity with any information
  // about the client.
  //
  // For backwards-compatibility, the default implementation of this method calls `accept()` and
  // then adds `UnknownPeerIdentity`.

  virtual uint getPort() = 0;
  // Gets the port number, if applicable (i.e. if listening on IP).  This is useful if you didn't
  // specify a port when constructing the NetworkAddress -- one will have been assigned
  // automatically.

  virtual void getsockopt(int level, int option, void* value, uint* length);
  virtual void setsockopt(int level, int option, const void* value, uint length);
  virtual void getsockname(struct sockaddr* addr, uint* length);
  // Same as the methods of AsyncIoStream.
};

Own<ConnectionReceiver> newAggregateConnectionReceiver(Array<Own<ConnectionReceiver>> receivers);
// Create a ConnectionReceiver that listens on several other ConnectionReceivers and returns
// sockets from any of them.

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
  inline Maybe<const T&> as() const;
  // Interpret the ancillary message as the given struct type. Most ancillary messages are some
  // sort of struct, so this is a convenient way to access it. Returns nullptr if the message
  // is smaller than the struct -- this can happen if the message was truncated due to
  // insufficient ancillary buffer space.

  template <typename T>
  inline ArrayPtr<const T> asArray() const;
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
  // Ancillary messages received with the datagram. See the recvmsg() system call and the cmsghdr
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

class NetworkAddress: private AsyncObject {
  // Represents a remote address to which the application can connect.

public:
  virtual Promise<Own<AsyncIoStream>> connect() = 0;
  // Make a new connection to this address.
  //
  // The address must not be a wildcard ("*").  If it is an IP address, it must have a port number.

  virtual Promise<AuthenticatedStream> connectAuthenticated();
  // Connect to the address and return both the connection and information about the peer identity.
  // This is especially useful when using TLS, to get certificate details.
  //
  // For backwards-compatibility, the default implementation of this method calls `connect()` and
  // then uses a `NetworkPeerIdentity` wrapping a clone of this `NetworkAddress` -- which is not
  // particularly useful.

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

  virtual Own<Network> restrictPeers(
      kj::ArrayPtr<const kj::StringPtr> allow,
      kj::ArrayPtr<const kj::StringPtr> deny = nullptr) KJ_WARN_UNUSED_RESULT = 0;
  // Constructs a new Network instance wrapping this one which restricts which peer addresses are
  // permitted (both for outgoing and incoming connections).
  //
  // Communication will be allowed only with peers whose addresses match one of the patterns
  // specified in the `allow` array. If a `deny` array is specified, then any address which matches
  // a pattern in `deny` and *does not* match any more-specific pattern in `allow` will also be
  // denied.
  //
  // The syntax of address patterns depends on the network, except that three special patterns are
  // defined for all networks:
  // - "private": Matches network addresses that are reserved by standards for private networks,
  //   such as "10.0.0.0/8" or "192.168.0.0/16". This is a superset of "local".
  // - "public": Opposite of "private".
  // - "local": Matches network addresses that are defined by standards to only be accessible from
  //   the local machine, such as "127.0.0.0/8" or Unix domain addresses.
  // - "network": Opposite of "local".
  //
  // For the standard KJ network implementation, the following patterns are also recognized:
  // - Network blocks specified in CIDR notation (ipv4 and ipv6), such as "192.0.2.0/24" or
  //   "2001:db8::/32".
  // - "unix" to match all Unix domain addresses. (In the future, we may support specifying a
  //   glob.)
  // - "unix-abstract" to match Linux's "abstract unix domain" addresses. (In the future, we may
  //   support specifying a glob.)
  //
  // Network restrictions apply *after* DNS resolution (otherwise they'd be useless).
  //
  // It is legal to parseAddress() a restricted address. An exception won't be thrown until
  // connect() is called.
  //
  // It's possible to listen() on a restricted address. However, connections will only be accepted
  // from non-restricted addresses; others will be dropped. If a particular listen address has no
  // valid peers (e.g. because it's a unix socket address and unix sockets are not allowed) then
  // listen() may throw (or may simply never receive any connections).
  //
  // Examples:
  //
  //     auto restricted = network->restrictPeers({"public"});
  //
  // Allows connections only to/from public internet addresses. Use this when connecting to an
  // address specified by a third party that is not trusted and is not themselves already on your
  // private network.
  //
  //     auto restricted = network->restrictPeers({"private"});
  //
  // Allows connections only to/from the private network. Use this on the server side to reject
  // connections from the public internet.
  //
  //     auto restricted = network->restrictPeers({"192.0.2.0/24"}, {"192.0.2.3/32"});
  //
  // Allows connections only to/from 192.0.2.*, except 192.0.2.3 which is blocked.
  //
  //     auto restricted = network->restrictPeers({"10.0.0.0/8", "10.1.2.3/32"}, {"10.1.2.0/24"});
  //
  // Allows connections to/from 10.*.*.*, with the exception of 10.1.2.* (which is denied), with an
  // exception to the exception of 10.1.2.3 (which is allowed, because it is matched by an allow
  // rule that is more specific than the deny rule).
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

  virtual CapabilityPipe newCapabilityPipe();
  // Creates two AsyncCapabilityStreams representing the two ends of a two-way capability pipe.
  //
  // The default implementation throws an unimplemented exception. In particular this is not
  // implemented by the default AsyncIoProvider on Windows, since Windows lacks any sane way to
  // pass handles over a stream.

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
  // On Windows, this interface can be used to import native SOCKETs into the async framework.
  // Different implementations of this interface might work on top of different event handling
  // primitives, such as I/O completion ports vs. completion routines.

public:
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
  typedef AutoCloseHandle OwnFd;
  // On Windows, the `fd` parameter to each of these methods must be a SOCKET, and must have the
  // flag WSA_FLAG_OVERLAPPED (which socket() uses by default, but WSASocket() wants you to specify
  // explicitly).
#else
  typedef int Fd;
  typedef AutoCloseFd OwnFd;
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

#if !_WIN32
  virtual Own<AsyncCapabilityStream> wrapUnixSocketFd(Fd fd, uint flags = 0);
  // Like wrapSocketFd() but also support capability passing via SCM_RIGHTS. The socket must be
  // a Unix domain socket.
  //
  // The default implementation throws UNIMPLEMENTED, for backwards-compatibility with
  // LowLevelAsyncIoProvider implementations written before this method was added.
#endif

  virtual Promise<Own<AsyncIoStream>> wrapConnectingSocketFd(
      Fd fd, const struct sockaddr* addr, uint addrlen, uint flags = 0) = 0;
  // Create an AsyncIoStream wrapping a socket and initiate a connection to the given address.
  // The returned promise does not resolve until connection has completed.
  //
  // `flags` is a bitwise-OR of the values of the `Flags` enum.

  class NetworkFilter {
  public:
    virtual bool shouldAllow(const struct sockaddr* addr, uint addrlen) = 0;
    // Returns true if incoming connections or datagrams from the given peer should be accepted.
    // If false, they will be dropped. This is used to implement kj::Network::restrictPeers().

    static NetworkFilter& getAllAllowed();
  };

  virtual Own<ConnectionReceiver> wrapListenSocketFd(
      Fd fd, NetworkFilter& filter, uint flags = 0) = 0;
  inline Own<ConnectionReceiver> wrapListenSocketFd(Fd fd, uint flags = 0) {
    return wrapListenSocketFd(fd, NetworkFilter::getAllAllowed(), flags);
  }
  // Create an AsyncIoStream wrapping a listen socket file descriptor.  This socket should already
  // have had `bind()` and `listen()` called on it, so it's ready for `accept()`.
  //
  // `flags` is a bitwise-OR of the values of the `Flags` enum.

  virtual Own<DatagramPort> wrapDatagramSocketFd(Fd fd, NetworkFilter& filter, uint flags = 0);
  inline Own<DatagramPort> wrapDatagramSocketFd(Fd fd, uint flags = 0) {
    return wrapDatagramSocketFd(fd, NetworkFilter::getAllAllowed(), flags);
  }

  virtual Timer& getTimer() = 0;
  // Returns a `Timer` based on real time.  Time does not pass while event handlers are running --
  // it only updates when the event loop polls for system events.  This means that calling `now()`
  // on this timer does not require a system call.
  //
  // This timer is not affected by changes to the system date.  It is unspecified whether the timer
  // continues to count while the system is suspended.

  Own<AsyncInputStream> wrapInputFd(OwnFd&& fd, uint flags = 0);
  Own<AsyncOutputStream> wrapOutputFd(OwnFd&& fd, uint flags = 0);
  Own<AsyncIoStream> wrapSocketFd(OwnFd&& fd, uint flags = 0);
#if !_WIN32
  Own<AsyncCapabilityStream> wrapUnixSocketFd(OwnFd&& fd, uint flags = 0);
#endif
  Promise<Own<AsyncIoStream>> wrapConnectingSocketFd(
      OwnFd&& fd, const struct sockaddr* addr, uint addrlen, uint flags = 0);
  Own<ConnectionReceiver> wrapListenSocketFd(
      OwnFd&& fd, NetworkFilter& filter, uint flags = 0);
  Own<ConnectionReceiver> wrapListenSocketFd(OwnFd&& fd, uint flags = 0);
  Own<DatagramPort> wrapDatagramSocketFd(OwnFd&& fd, NetworkFilter& filter, uint flags = 0);
  Own<DatagramPort> wrapDatagramSocketFd(OwnFd&& fd, uint flags = 0);
  // Convenience wrappers which transfer ownership via AutoCloseFd (Unix) or AutoCloseHandle
  // (Windows). TAKE_OWNERSHIP will be implicitly added to `flags`.
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
// Convenience adapters.

class CapabilityStreamConnectionReceiver final: public ConnectionReceiver {
  // Trivial wrapper which allows an AsyncCapabilityStream to act as a ConnectionReceiver. accept()
  // calls receiveStream().

public:
  CapabilityStreamConnectionReceiver(AsyncCapabilityStream& inner)
      : inner(inner) {}

  Promise<Own<AsyncIoStream>> accept() override;
  uint getPort() override;

  Promise<AuthenticatedStream> acceptAuthenticated() override;
  // Always produces UnknownIdentity. Capability-based security patterns should not rely on
  // authenticating peers; the other end of the capability stream should only be given to
  // authorized parties in the first place.

private:
  AsyncCapabilityStream& inner;
};

class CapabilityStreamNetworkAddress final: public NetworkAddress {
  // Trivial wrapper which allows an AsyncCapabilityStream to act as a NetworkAddress.
  //
  // connect() is implemented by calling provider.newCapabilityPipe(), sending one end over the
  // original capability stream, and returning the other end. If `provider` is null, then the
  // global kj::newCapabilityPipe() will be used, but this ONLY works if `inner` itself is agnostic
  // to the type of streams it receives, e.g. because it was also created using
  // kj::NewCapabilityPipe().
  //
  // listen().accept() is implemented by receiving new streams over the original stream.
  //
  // Note that clone() doesn't work (due to ownership issues) and toString() returns a static
  // string.

public:
  CapabilityStreamNetworkAddress(kj::Maybe<AsyncIoProvider&> provider, AsyncCapabilityStream& inner)
      : provider(provider), inner(inner) {}

  Promise<Own<AsyncIoStream>> connect() override;
  Own<ConnectionReceiver> listen() override;

  Own<NetworkAddress> clone() override;
  String toString() override;

  Promise<AuthenticatedStream> connectAuthenticated() override;
  // Always produces UnknownIdentity. Capability-based security patterns should not rely on
  // authenticating peers; the other end of the capability stream should only be given to
  // authorized parties in the first place.

private:
  kj::Maybe<AsyncIoProvider&> provider;
  AsyncCapabilityStream& inner;
};

class FileInputStream: public AsyncInputStream {
  // InputStream that reads from a disk file -- and enables sendfile() optimization.
  //
  // Reads are performed synchronously -- no actual attempt is made to use asynchronous file I/O.
  // True asynchronous file I/O is complicated and is mostly unnecessary in the presence of
  // caching. Only certain niche programs can expect to benefit from it. For the rest, it's better
  // to use regular syrchronous disk I/O, so that's what this class does.
  //
  // The real purpose of this class, aside from general convenience, is to enable sendfile()
  // optimization. When you use this class's pumpTo() method, and the destination is a socket,
  // the system will detect this and optimize to sendfile(), so that the file data never needs to
  // be read into userspace.
  //
  // NOTE: As of this writing, sendfile() optimization is only implemented on Linux.

public:
  FileInputStream(const ReadableFile& file, uint64_t offset = 0)
      : file(file), offset(offset) {}

  const ReadableFile& getUnderlyingFile() { return file; }
  uint64_t getOffset() { return offset; }
  void seek(uint64_t newOffset) { offset = newOffset; }

  Promise<size_t> tryRead(void* buffer, size_t minBytes, size_t maxBytes);
  Maybe<uint64_t> tryGetLength();

  // (pumpTo() is not actually overridden here, but AsyncStreamFd's tryPumpFrom() will detect when
  // the source is a file.)

private:
  const ReadableFile& file;
  uint64_t offset;
};

class FileOutputStream: public AsyncOutputStream {
  // OutputStream that writes to a disk file.
  //
  // As with FileInputStream, calls are not actually async. Async would be even less useful here
  // because writes should usually land in cache anyway.
  //
  // sendfile() optimization does not apply when writing to a file, but on Linux, splice() can
  // be used to achieve a similar effect.
  //
  // NOTE: As of this writing, splice() optimization is not implemented.

public:
  FileOutputStream(const File& file, uint64_t offset = 0)
      : file(file), offset(offset) {}

  const File& getUnderlyingFile() { return file; }
  uint64_t getOffset() { return offset; }
  void seek(uint64_t newOffset) { offset = newOffset; }

  Promise<void> write(const void* buffer, size_t size);
  Promise<void> write(ArrayPtr<const ArrayPtr<const byte>> pieces);
  Promise<void> whenWriteDisconnected();

private:
  const File& file;
  uint64_t offset;
};

// =======================================================================================
// inline implementation details

inline AncillaryMessage::AncillaryMessage(
    int level, int type, ArrayPtr<const byte> data)
    : level(level), type(type), data(data) {}

inline int AncillaryMessage::getLevel() const { return level; }
inline int AncillaryMessage::getType() const { return type; }

template <typename T>
inline Maybe<const T&> AncillaryMessage::as() const {
  if (data.size() >= sizeof(T)) {
    return *reinterpret_cast<const T*>(data.begin());
  } else {
    return nullptr;
  }
}

template <typename T>
inline ArrayPtr<const T> AncillaryMessage::asArray() const {
  return arrayPtr(reinterpret_cast<const T*>(data.begin()), data.size() / sizeof(T));
}

class SecureNetworkWrapper {
  // Abstract interface for a class which implements a "secure" network as a wrapper around an
  // insecure one. "secure" means:
  // * Connections to a server will only succeed if it can be verified that the requested hostname
  //   actually belongs to the responding server.
  // * No man-in-the-middle attacker can potentially see the bytes sent and received.
  //
  // The typical implementation uses TLS. The object in this case could be configured to use cerain
  // keys, certificates, etc. See kj/compat/tls.h for such an implementation.
  //
  // However, an implementation could use some other form of encryption, or might not need to use
  // encryption at all. For example, imagine a kj::Network that exists only on a single machine,
  // providing communications between various processes using unix sockets. Perhaps the "hostnames"
  // are actually PIDs in this case. An implementation of such a network could verify the other
  // side's identity using an `SCM_CREDENTIALS` auxiliary message, which cannot be forged. Once
  // verified, there is no need to encrypt since unix sockets cannot be intercepted.

public:
  virtual kj::Promise<kj::Own<kj::AsyncIoStream>> wrapServer(kj::Own<kj::AsyncIoStream> stream) = 0;
  // Act as the server side of a connection. The given stream is already connected to a client, but
  // no authentication has occurred. The returned stream represents the secure transport once
  // established.

  virtual kj::Promise<kj::Own<kj::AsyncIoStream>> wrapClient(
      kj::Own<kj::AsyncIoStream> stream, kj::StringPtr expectedServerHostname) = 0;
  // Act as the client side of a connection. The given stream is already connecetd to a server, but
  // no authentication has occurred. This method will verify that the server actually is the given
  // hostname, then return the stream representing a secure transport to that server.

  virtual kj::Promise<kj::AuthenticatedStream> wrapServer(kj::AuthenticatedStream stream) = 0;
  virtual kj::Promise<kj::AuthenticatedStream> wrapClient(
      kj::AuthenticatedStream stream, kj::StringPtr expectedServerHostname) = 0;
  // Same as above, but implementing kj::AuthenticatedStream, which provides PeerIdentity objects
  // with more details about the peer. The SecureNetworkWrapper will provide its own implementation
  // of PeerIdentity with the specific details it is able to authenticate.

  virtual kj::Own<kj::ConnectionReceiver> wrapPort(kj::Own<kj::ConnectionReceiver> port) = 0;
  // Wrap a connection listener. This is equivalent to calling wrapServer() on every connection
  // received.

  virtual kj::Own<kj::NetworkAddress> wrapAddress(
      kj::Own<kj::NetworkAddress> address, kj::StringPtr expectedServerHostname) = 0;
  // Wrap a NetworkAddress. This is equivalent to calling `wrapClient()` on every connection
  // formed by calling `connect()` on the address.

  virtual kj::Own<kj::Network> wrapNetwork(kj::Network& network) = 0;
  // Wrap a whole `kj::Network`. This automatically wraps everything constructed using the network.
  // The network will only accept address strings that can be authenticated, and will automatically
  // authenticate servers against those addresses when connecting to them.
};

}  // namespace kj

KJ_END_HEADER
