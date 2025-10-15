# Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
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

@0xb312981b2552a250;
# Recall that Cap'n Proto RPC allows messages to contain references to remote objects that
# implement interfaces.  These references are called "capabilities", because they both designate
# the remote object to use and confer permission to use it.
#
# Recall also that Cap'n Proto RPC has the feature that when a method call itself returns a
# capability, the caller can begin calling methods on that capability _before the first call has
# returned_.  The caller essentially sends a message saying "Hey server, as soon as you finish
# that previous call, do this with the result!".  Cap'n Proto's RPC protocol makes this possible.
#
# The protocol is significantly more complicated than most RPC protocols.  However, this is
# implementation complexity that underlies an easy-to-grasp higher-level model of object oriented
# programming.  That is, just like TCP is a surprisingly complicated protocol that implements a
# conceptually-simple byte stream abstraction, Cap'n Proto is a surprisingly complicated protocol
# that implements a conceptually-simple object abstraction.
#
# Cap'n Proto RPC is based heavily on CapTP, the object-capability protocol used by the E
# programming language:
#     http://www.erights.org/elib/distrib/captp/index.html
#
# Cap'n Proto RPC takes place between "vats".  A vat hosts some set of objects and talks to other
# vats through direct bilateral connections.  Typically, there is a 1:1 correspondence between vats
# and processes (in the unix sense of the word), although this is not strictly always true (one
# process could run multiple vats, or a distributed virtual vat might live across many processes).
#
# Cap'n Proto does not distinguish between "clients" and "servers" -- this is up to the application.
# Either end of any connection can potentially hold capabilities pointing to the other end, and
# can call methods on those capabilities.  In the doc comments below, we use the words "sender"
# and "receiver".  These refer to the sender and receiver of an instance of the struct or field
# being documented.  Sometimes we refer to a "third-party" that is neither the sender nor the
# receiver.  Documentation is generally written from the point of view of the sender.
#
# It is generally up to the vat network implementation to securely verify that connections are made
# to the intended vat as well as to encrypt transmitted data for privacy and integrity.  See the
# `VatNetwork` example interface near the end of this file.
#
# When a new connection is formed, the only interesting things that can be done are to send a
# `Bootstrap` (level 0) or `Accept` (level 3) message.
#
# Unless otherwise specified, messages must be delivered to the receiving application in the same
# order in which they were initiated by the sending application.  The goal is to support "E-Order",
# which states that two calls made on the same reference must be delivered in the order which they
# were made:
#     http://erights.org/elib/concurrency/partial-order.html
#
# Since the full protocol is complicated, we define multiple levels of support that an
# implementation may target.  For many applications, level 1 support will be sufficient.
# Comments in this file indicate which level requires the corresponding feature to be
# implemented.
#
# * **Level 0:** The implementation does not support object references. Only the bootstrap interface
#   can be called. At this level, the implementation does not support object-oriented protocols and
#   is similar in complexity to JSON-RPC or Protobuf services. This level should be considered only
#   a temporary stepping-stone toward level 1 as the lack of object references drastically changes
#   how protocols are designed. Applications _should not_ attempt to design their protocols around
#   the limitations of level 0 implementations.
#
# * **Level 1:** The implementation supports simple bilateral interaction with object references
#   and promise pipelining, but interactions between three or more parties are supported only via
#   proxying of objects.  E.g. if Alice (in Vat A) wants to send Bob (in Vat B) a capability
#   pointing to Carol (in Vat C), Alice must create a proxy of Carol within Vat A and send Bob a
#   reference to that; Bob cannot form a direct connection to Carol.  Level 1 implementations do
#   not support checking if two capabilities received from different vats actually point to the
#   same object ("join"), although they should be able to do this check on capabilities received
#   from the same vat.
#
# * **Level 2:** The implementation supports saving persistent capabilities -- i.e. capabilities
#   that remain valid even after disconnect, and can be restored on a future connection. When a
#   capability is saved, the requester receives a `SturdyRef`, which is a token that can be used
#   to restore the capability later.
#
# * **Level 3:** The implementation supports three-way interactions.  That is, if Alice (in Vat A)
#   sends Bob (in Vat B) a capability pointing to Carol (in Vat C), then Vat B will automatically
#   form a direct connection to Vat C rather than have requests be proxied through Vat A.
#
# * **Level 4:** The entire protocol is implemented, including joins (checking if two capabilities
#   are equivalent).
#
# Note that an implementation must also support specific networks (transports), as described in
# the "Network-specific Parameters" section below.  An implementation might have different levels
# depending on the network used.
#
# New implementations of Cap'n Proto should start out targeting the simplistic two-party network
# type as defined in `rpc-twoparty.capnp`.  With this network type, level 3 is irrelevant and
# levels 2 and 4 are much easier than usual to implement.  When such an implementation is paired
# with a container proxy, the contained app effectively gets to make full use of the proxy's
# network at level 4.  And since Cap'n Proto IPC is extremely fast, it may never make sense to
# bother implementing any other vat network protocol -- just use the correct container type and get
# it for free.

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("capnp::rpc");

# ========================================================================================
# The Four Tables
#
# Cap'n Proto RPC connections are stateful (although an application built on Cap'n Proto could
# export a stateless interface).  As in CapTP, for each open connection, a vat maintains four state
# tables: questions, answers, imports, and exports.  See the diagram at:
#     http://www.erights.org/elib/distrib/captp/4tables.html
#
# The question table corresponds to the other end's answer table, and the imports table corresponds
# to the other end's exports table.
#
# The entries in each table are identified by ID numbers (defined below as 32-bit integers).  These
# numbers are always specific to the connection; a newly-established connection starts with no
# valid IDs.  Since low-numbered IDs will pack better, it is suggested that IDs be assigned like
# Unix file descriptors -- prefer the lowest-number ID that is currently available.
#
# IDs in the questions/answers tables are chosen by the questioner and generally represent method
# calls that are in progress.
#
# IDs in the imports/exports tables are chosen by the exporter and generally represent objects on
# which methods may be called.  Exports may be "settled", meaning the exported object is an actual
# object living in the exporter's vat, or they may be "promises", meaning the exported object is
# the as-yet-unknown result of an ongoing operation and will eventually be resolved to some other
# object once that operation completes.  Calls made to a promise will be forwarded to the eventual
# target once it is known.  The eventual replacement object does *not* get the same ID as the
# promise, as it may turn out to be an object that is already exported (so already has an ID) or
# may even live in a completely different vat (and so won't get an ID on the same export table
# at all).
#
# IDs can be reused over time.  To make this safe, we carefully define the lifetime of IDs.  Since
# messages using the ID could be traveling in both directions simultaneously, we must define the
# end of life of each ID _in each direction_.  The ID is only safe to reuse once it has been
# released by both sides.
#
# When a Cap'n Proto connection is lost, everything on the four tables is lost.  All questions are
# canceled and throw exceptions.  All imports become broken (all future calls to them throw
# exceptions).  All exports and answers are implicitly released.  The only things not lost are
# persistent capabilities (`SturdyRef`s).  The application must plan for this and should respond by
# establishing a new connection and restoring from these persistent capabilities.

using QuestionId = UInt32;
# **(level 0)**
#
# Identifies a question in the sender's question table (which corresponds to the receiver's answer
# table).  The questioner (caller) chooses an ID when making a call.  The ID remains valid in
# caller -> callee messages until a Finish message is sent, and remains valid in callee -> caller
# messages until a Return message is sent.

using AnswerId = QuestionId;
# **(level 0)**
#
# Identifies an answer in the sender's answer table (which corresponds to the receiver's question
# table).
#
# AnswerId is physically equivalent to QuestionId, since the question and answer tables correspond,
# but we define a separate type for documentation purposes:  we always use the type representing
# the sender's point of view.

using ExportId = UInt32;
# **(level 1)**
#
# Identifies an exported capability or promise in the sender's export table (which corresponds
# to the receiver's import table).  The exporter chooses an ID before sending a capability over the
# wire.  If the capability is already in the table, the exporter should reuse the same ID.  If the
# ID is a promise (as opposed to a settled capability), this must be indicated at the time the ID
# is introduced (e.g. by using `senderPromise` instead of `senderHosted` in `CapDescriptor`); in
# this case, the importer shall expect a later `Resolve` message that replaces the promise.
#
# ExportId/ImportIds are subject to reference counting.  Whenever an `ExportId` is sent over the
# wire (from the exporter to the importer), the export's reference count is incremented (unless
# otherwise specified).  The reference count is later decremented by a `Release` message.  Since
# the `Release` message can specify an arbitrary number by which to reduce the reference count, the
# importer should usually batch reference decrements and only send a `Release` when it believes the
# reference count has hit zero.  Of course, it is possible that a new reference to the export is
# in-flight at the time that the `Release` message is sent, so it is necessary for the exporter to
# keep track of the reference count on its end as well to avoid race conditions.
#
# When a connection is lost, all exports are implicitly released.  It is not possible to restore
# a connection state after disconnect (although a transport layer could implement a concept of
# persistent connections if it is transparent to the RPC layer).

using ImportId = ExportId;
# **(level 1)**
#
# Identifies an imported capability or promise in the sender's import table (which corresponds to
# the receiver's export table).
#
# ImportId is physically equivalent to ExportId, since the export and import tables correspond,
# but we define a separate type for documentation purposes:  we always use the type representing
# the sender's point of view.
#
# An `ImportId` remains valid in importer -> exporter messages until the importer has sent
# `Release` messages that (it believes) have reduced the reference count to zero.

# ========================================================================================
# Messages

struct Message {
  # An RPC connection is a bi-directional stream of Messages.

  union {
    unimplemented @0 :Message;
    # The sender previously received this message from the peer but didn't understand it or doesn't
    # yet implement the functionality that was requested.  So, the sender is echoing the message
    # back.  In some cases, the receiver may be able to recover from this by pretending the sender
    # had taken some appropriate "null" action.
    #
    # For example, say `resolve` is received by a level 0 implementation (because a previous call
    # or return happened to contain a promise).  The level 0 implementation will echo it back as
    # `unimplemented`.  The original sender can then simply release the cap to which the promise
    # had resolved, thus avoiding a leak.
    #
    # For any message type that introduces a question, if the message comes back unimplemented,
    # the original sender may simply treat it as if the question failed with an exception.
    #
    # In cases where there is no sensible way to react to an `unimplemented` message (without
    # resource leaks or other serious problems), the connection may need to be aborted.  This is
    # a gray area; different implementations may take different approaches.

    abort @1 :Exception;
    # Sent when a connection is being aborted due to an unrecoverable error.  This could be e.g.
    # because the sender received an invalid or nonsensical message or because the sender had an
    # internal error.  The sender will shut down the outgoing half of the connection after `abort`
    # and will completely close the connection shortly thereafter (it's up to the sender how much
    # of a time buffer they want to offer for the client to receive the `abort` before the
    # connection is reset).

    # Level 0 features -----------------------------------------------

    bootstrap @8 :Bootstrap;  # Request the peer's bootstrap interface.
    call @2 :Call;            # Begin a method call.
    return @3 :Return;        # Complete a method call.
    finish @4 :Finish;        # Release a returned answer / cancel a call.

    # Level 1 features -----------------------------------------------

    resolve @5 :Resolve;   # Resolve a previously-sent promise.
    release @6 :Release;   # Release a capability so that the remote object can be deallocated.
    disembargo @13 :Disembargo;  # Lift an embargo used to enforce E-order over promise resolution.

    # Level 2 features -----------------------------------------------

    obsoleteSave @7 :AnyPointer;
    # Obsolete request to save a capability, resulting in a SturdyRef. This has been replaced
    # by the `Persistent` interface defined in `persistent.capnp`. This operation was never
    # implemented.

    obsoleteDelete @9 :AnyPointer;
    # Obsolete way to delete a SturdyRef. This operation was never implemented.

    # Level 3 features -----------------------------------------------

    provide @10 :Provide;  # Provide a capability to a third party.
    accept @11 :Accept;    # Accept a capability provided by a third party.

    # Level 4 features -----------------------------------------------

    join @12 :Join;        # Directly connect to the common root of two or more proxied caps.
  }
}

# Level 0 message types ----------------------------------------------

struct Bootstrap {
  # **(level 0)**
  #
  # Get the "bootstrap" interface exported by the remote vat.
  #
  # For level 0, 1, and 2 implementations, the "bootstrap" interface is simply the main interface
  # exported by a vat. If the vat acts as a server fielding connections from clients, then the
  # bootstrap interface defines the basic functionality available to a client when it connects.
  # The exact interface definition obviously depends on the application.
  #
  # We call this a "bootstrap" because in an ideal Cap'n Proto world, bootstrap interfaces would
  # never be used. In such a world, any time you connect to a new vat, you do so because you
  # received an introduction from some other vat (see `ThirdPartyCapId`). Thus, the first message
  # you send is `Accept`, and further communications derive from there. `Bootstrap` is not used.
  #
  # In such an ideal world, DNS itself would support Cap'n Proto -- performing a DNS lookup would
  # actually return a new Cap'n Proto capability, thus introducing you to the target system via
  # level 3 RPC. Applications would receive the capability to talk to DNS in the first place as
  # an initial endowment or part of a Powerbox interaction. Therefore, an app can form arbitrary
  # connections without ever using `Bootstrap`.
  #
  # Of course, in the real world, DNS is not Cap'n-Proto-based, and we don't want Cap'n Proto to
  # require a whole new internet infrastructure to be useful. Therefore, we offer bootstrap
  # interfaces as a way to get up and running without a level 3 introduction. Thus, bootstrap
  # interfaces are used to "bootstrap" from other, non-Cap'n-Proto-based means of service discovery,
  # such as legacy DNS.
  #
  # Note that a vat need not provide a bootstrap interface, and in fact many vats (especially those
  # acting as clients) do not. In this case, the vat should either reply to `Bootstrap` with a
  # `Return` indicating an exception, or should return a dummy capability with no methods.

  questionId @0 :QuestionId;
  # A new question ID identifying this request, which will eventually receive a Return message
  # containing the restored capability.

  deprecatedObjectId @1 :AnyPointer;
  # ** DEPRECATED **
  #
  # A Vat may export multiple bootstrap interfaces. In this case, `deprecatedObjectId` specifies
  # which one to return. If this pointer is null, then the default bootstrap interface is returned.
  #
  # As of version 0.5, use of this field is deprecated. If a service wants to export multiple
  # bootstrap interfaces, it should instead define a single bootstrap interface that has methods
  # that return each of the other interfaces.
  #
  # **History**
  #
  # In the first version of Cap'n Proto RPC (0.4.x) the `Bootstrap` message was called `Restore`.
  # At the time, it was thought that this would eventually serve as the way to restore SturdyRefs
  # (level 2). Meanwhile, an application could offer its "main" interface on a well-known
  # (non-secret) SturdyRef.
  #
  # Since level 2 RPC was not implemented at the time, the `Restore` message was in practice only
  # used to obtain the main interface. Since most applications had only one main interface that
  # they wanted to restore, they tended to designate this with a null `objectId`.
  #
  # Unfortunately, the earliest version of the EZ RPC interfaces set a precedent of exporting
  # multiple main interfaces by allowing them to be exported under string names. In this case,
  # `objectId` was a Text value specifying the name.
  #
  # All of this proved problematic for several reasons:
  #
  # - The arrangement assumed that a client wishing to restore a SturdyRef would know exactly what
  #   machine to connect to and would be able to immediately restore a SturdyRef on connection.
  #   However, in practice, the ability to restore SturdyRefs is itself a capability that may
  #   require going through an authentication process to obtain. Thus, it makes more sense to
  #   define a "restorer service" as a full Cap'n Proto interface. If this restorer interface is
  #   offered as the vat's bootstrap interface, then this is equivalent to the old arrangement.
  #
  # - Overloading "Restore" for the purpose of obtaining well-known capabilities encouraged the
  #   practice of exporting singleton services with string names. If singleton services are desired,
  #   it is better to have one main interface that has methods that can be used to obtain each
  #   service, in order to get all the usual benefits of schemas and type checking.
  #
  # - Overloading "Restore" also had a security problem: Often, "main" or "well-known"
  #   capabilities exported by a vat are in fact not public: they are intended to be accessed only
  #   by clients who are capable of forming a connection to the vat. This can lead to trouble if
  #   the client itself has other clients and wishes to forward some `Restore` requests from those
  #   external clients -- it has to be very careful not to allow through `Restore` requests
  #   addressing the default capability.
  #
  #   For example, consider the case of a sandboxed Sandstorm application and its supervisor. The
  #   application exports a default capability to its supervisor that provides access to
  #   functionality that only the supervisor is supposed to access. Meanwhile, though, applications
  #   may publish other capabilities that may be persistent, in which case the application needs
  #   to field `Restore` requests that could come from anywhere. These requests of course have to
  #   pass through the supervisor, as all communications with the outside world must. But, the
  #   supervisor has to be careful not to honor an external request addressing the application's
  #   default capability, since this capability is privileged. Unfortunately, the default
  #   capability cannot be given an unguessable name, because then the supervisor itself would not
  #   be able to address it!
  #
  # As of Cap'n Proto 0.5, `Restore` has been renamed to `Bootstrap` and is no longer planned for
  # use in restoring SturdyRefs.
  #
  # Note that 0.4 also defined a message type called `Delete` that, like `Restore`, addressed a
  # SturdyRef, but indicated that the client would not restore the ref again in the future. This
  # operation was never implemented, so it was removed entirely. If a "delete" operation is desired,
  # it should exist as a method on the same interface that handles restoring SturdyRefs. However,
  # the utility of such an operation is questionable. You wouldn't be able to rely on it for
  # garbage collection since a client could always disappear permanently without remembering to
  # delete all its SturdyRefs, thus leaving them dangling forever. Therefore, it is advisable to
  # design systems such that SturdyRefs never represent "owned" pointers.
  #
  # For example, say a SturdyRef points to an image file hosted on some server. That image file
  # should also live inside a collection (a gallery, perhaps) hosted on the same server, owned by
  # a user who can delete the image at any time. If the user deletes the image, the SturdyRef
  # stops working. On the other hand, if the SturdyRef is discarded, this has no effect on the
  # existence of the image in its collection.
}

struct Call {
  # **(level 0)**
  #
  # Message type initiating a method call on a capability.

  questionId @0 :QuestionId;
  # A number, chosen by the caller, that identifies this call in future messages.  This number
  # must be different from all other calls originating from the same end of the connection (but
  # may overlap with question IDs originating from the opposite end).  A fine strategy is to use
  # sequential question IDs, but the recipient should not assume this.
  #
  # A question ID can be reused once both:
  # - A matching Return has been received from the callee.
  # - A matching Finish has been sent from the caller.

  target @1 :MessageTarget;
  # The object that should receive this call.

  interfaceId @2 :UInt64;
  # The type ID of the interface being called.  Each capability may implement multiple interfaces.

  methodId @3 :UInt16;
  # The ordinal number of the method to call within the requested interface.

  allowThirdPartyTailCall @8 :Bool = false;
  # Indicates whether or not the receiver is allowed to send a `Return` containing
  # `acceptFromThirdParty`.  Level 3 implementations should set this true.  Otherwise, the callee
  # will have to proxy the return in the case of a tail call to a third-party vat.

  noPromisePipelining @9 :Bool = false;
  # If true, the sender promises that it won't make any promise-pipelined calls on the results of
  # this call. If it breaks this promise, the receiver may throw an arbitrary error from such
  # calls.
  #
  # The receiver may use this as an optimization, by skipping the bookkeeping needed for pipelining
  # when no pipelined calls are expected. The sender typically sets this to false when the method's
  # schema does not specify any return capabilities.

  onlyPromisePipeline @10 :Bool = false;
  # If true, the sender only plans to use this call to make pipelined calls. The receiver need not
  # send a `Return` message (but is still allowed to do so).
  #
  # Since the sender does not know whether a `Return` will be sent, it must release all state
  # related to the call when it sends `Finish`. However, in the case that the callee does not
  # recognize this hint and chooses to send a `Return`, then technically the caller is not allowed
  # to reuse the question ID until it receives said `Return`. This creates a conundrum: How does
  # the caller decide when it's OK to reuse the ID? To sidestep the problem, the C++ implementation
  # uses high-numbered IDs (with the high-order bit set) for such calls, and cycles through the
  # IDs in order. If all 2^31 IDs in this space are used without ever seeing a `Return`, then the
  # implementation assumes that the other end is in fact honoring the hint, and the ID counter is
  # allowed to loop around. If a `Return` is ever seen when `onlyPromisePipeline` was set, then
  # the implementation stops using this hint.

  params @4 :Payload;
  # The call parameters.  `params.content` is a struct whose fields correspond to the parameters of
  # the method.

  sendResultsTo :union {
    # Where should the return message be sent?

    caller @5 :Void;
    # Send the return message back to the caller (the usual).

    yourself @6 :Void;
    # **(level 1)**
    #
    # Don't actually return the results to the sender.  Instead, hold on to them and await
    # instructions from the sender regarding what to do with them.  In particular, the sender
    # may subsequently send a `Return` for some other call (which the receiver had previously made
    # to the sender) with `takeFromOtherQuestion` set.  The results from this call are then used
    # as the results of the other call.
    #
    # When `yourself` is used, the receiver must still send a `Return` for the call, but sets the
    # field `resultsSentElsewhere` in that `Return` rather than including the results.
    #
    # This feature can be used to implement tail calls in which a call from Vat A to Vat B ends up
    # returning the result of a call from Vat B back to Vat A.
    #
    # In particular, the most common use case for this feature is when Vat A makes a call to a
    # promise in Vat B, and then that promise ends up resolving to a capability back in Vat A.
    # Vat B must forward all the queued calls on that promise back to Vat A, but can set `yourself`
    # in the calls so that the results need not pass back through Vat B.
    #
    # For example:
    # - Alice, in Vat A, calls foo() on Bob in Vat B.
    # - Alice makes a pipelined call bar() on the promise returned by foo().
    # - Later on, Bob resolves the promise from foo() to point at Carol, who lives in Vat A (next
    #   to Alice).
    # - Vat B dutifully forwards the bar() call to Carol.  Let us call this forwarded call bar'().
    #   Notice that bar() and bar'() are travelling in opposite directions on the same network
    #   link.
    # - The `Call` for bar'() has `sendResultsTo` set to `yourself`.
    # - Vat B sends a `Return` for bar() with `takeFromOtherQuestion` set in place of the results,
    #   with the value set to the question ID of bar'().  Vat B does not wait for bar'() to return,
    #   as doing so would introduce unnecessary round trip latency.
    # - Vat A receives bar'() and delivers it to Carol.
    # - When bar'() returns, Vat A sends a `Return` for bar'() to Vat B, with `resultsSentElsewhere`
    #   set in place of results.
    # - Vat A sends a `Finish` for the bar() call to Vat B.
    # - Vat B receives the `Finish` for bar() and sends a `Finish` for bar'().

    thirdParty @7 :RecipientId;
    # **(level 3)**
    #
    # The call's result should be returned to a different vat.  The receiver (the callee) expects
    # to receive an `Accept` message from the indicated vat, and should return the call's result
    # to it, rather than to the sender of the `Call`.
    #
    # This operates much like `yourself`, above, except that Carol is in a separate Vat C.  `Call`
    # messages are sent from Vat A -> Vat B and Vat B -> Vat C.  A `Return` message is sent from
    # Vat B -> Vat A that contains `acceptFromThirdParty` in place of results.  When Vat A sends
    # an `Accept` to Vat C, it receives back a `Return` containing the call's actual result.  Vat C
    # also sends a `Return` to Vat B with `resultsSentElsewhere`.
  }
}

struct Return {
  # **(level 0)**
  #
  # Message type sent from callee to caller indicating that the call has completed.

  answerId @0 :AnswerId;
  # Equal to the QuestionId of the corresponding `Call` message.

  releaseParamCaps @1 :Bool = true;
  # If true, all capabilities that were in the params should be considered released.  The sender
  # must not send separate `Release` messages for them.  Level 0 implementations in particular
  # should always set this true.  This defaults true because if level 0 implementations forget to
  # set it they'll never notice (just silently leak caps), but if level >=1 implementations forget
  # to set it to false they'll quickly get errors.
  #
  # The receiver should act as if the sender had sent a release message with count=1 for each
  # CapDescriptor in the original Call message.

  noFinishNeeded @8 :Bool = false;
  # If true, the sender does not need the receiver to send a `Finish` message; its answer table
  # entry has already been cleaned up. This implies that the results do not contain any
  # capabilities, since the `Finish` message would normally release those capabilities from
  # promise pipelining responsibility. The caller may still send a `Finish` message if it wants,
  # which will be silently ignored by the callee.

  union {
    results @2 :Payload;
    # The result.
    #
    # For regular method calls, `results.content` points to the result struct.
    #
    # For a `Return` in response to an `Accept` or `Bootstrap`, `results` contains a single
    # capability (rather than a struct), and `results.content` is just a capability pointer with
    # index 0.  A `Finish` is still required in this case.

    exception @3 :Exception;
    # Indicates that the call failed and explains why.

    canceled @4 :Void;
    # Indicates that the call was canceled due to the caller sending a Finish message
    # before the call had completed.

    resultsSentElsewhere @5 :Void;
    # This is set when returning from a `Call` that had `sendResultsTo` set to something other
    # than `caller`.
    #
    # It doesn't matter too much when this is sent, as the receiver doesn't need to do anything
    # with it, but the C++ implementation appears to wait for the call to finish before sending
    # this.

    takeFromOtherQuestion @6 :QuestionId;
    # The sender has also sent (before this message) a `Call` with the given question ID and with
    # `sendResultsTo.yourself` set, and the results of that other call should be used as the
    # results here.  `takeFromOtherQuestion` can only used once per question.

    acceptFromThirdParty @7 :ThirdPartyCapId;
    # **(level 3)**
    #
    # The caller should contact a third-party vat to pick up the results.  An `Accept` message
    # sent to the vat will return the result.  This pairs with `Call.sendResultsTo.thirdParty`.
    # It should only be used if the corresponding `Call` had `allowThirdPartyTailCall` set.
  }
}

struct Finish {
  # **(level 0)**
  #
  # Message type sent from the caller to the callee to indicate:
  # 1) The questionId will no longer be used in any messages sent by the callee (no further
  #    pipelined requests).
  # 2) If the call has not returned yet, the caller no longer cares about the result.  If nothing
  #    else cares about the result either (e.g. there are no other outstanding calls pipelined on
  #    the result of this one) then the callee may wish to immediately cancel the operation and
  #    send back a Return message with "canceled" set.  However, implementations are not required
  #    to support premature cancellation -- instead, the implementation may wait until the call
  #    actually completes and send a normal `Return` message.
  #
  # TODO(someday): Should we separate (1) and implicitly releasing result capabilities?  It would be
  #   possible and useful to notify the server that it doesn't need to keep around the response to
  #   service pipeline requests even though the caller still wants to receive it / hasn't yet
  #   finished processing it.  It could also be useful to notify the server that it need not marshal
  #   the results because the caller doesn't want them anyway, even if the caller is still sending
  #   pipelined calls, although this seems less useful (just saving some bytes on the wire).

  questionId @0 :QuestionId;
  # ID of the call whose result is to be released.

  releaseResultCaps @1 :Bool = true;
  # If true, all capabilities that were in the results should be considered released.  The sender
  # must not send separate `Release` messages for them.  Level 0 implementations in particular
  # should always set this true.  This defaults true because if level 0 implementations forget to
  # set it they'll never notice (just silently leak caps), but if level >=1 implementations forget
  # set it false they'll quickly get errors.

  requireEarlyCancellationWorkaround @2 :Bool = true;
  # If true, if the RPC system receives this Finish message before the original call has even been
  # delivered, it should defer cancellation util after delivery. In particular, this gives the
  # destination object a chance to opt out of cancellation, e.g. as controlled by the
  # `allowCancellation` annotation defined in `c++.capnp`.
  #
  # This is a work-around. Versions 1.0 and up of Cap'n Proto always set this to false. However,
  # older versions of Cap'n Proto unintentionally exhibited this errant behavior by default, and
  # as a result programs built with older versions could be inadvertently relying on their peers
  # to implement the behavior. The purpose of this flag is to let newer versions know when the
  # peer is an older version, so that it can attempt to work around the issue.
  #
  # See also comments in handleFinish() in rpc.c++ for more details.
}

# Level 1 message types ----------------------------------------------

struct Resolve {
  # **(level 1)**
  #
  # Message type sent to indicate that a previously-sent promise has now been resolved to some other
  # object (possibly another promise) -- or broken, or canceled.
  #
  # Keep in mind that it's possible for a `Resolve` to be sent to a level 0 implementation that
  # doesn't implement it.  For example, a method call or return might contain a capability in the
  # payload.  Normally this is fine even if the receiver is level 0, because they will implicitly
  # release all such capabilities on return / finish.  But if the cap happens to be a promise, then
  # a follow-up `Resolve` may be sent regardless of this release.  The level 0 receiver will reply
  # with an `unimplemented` message, and the sender (of the `Resolve`) can respond to this as if the
  # receiver had immediately released any capability to which the promise resolved.
  #
  # When implementing promise resolution, it's important to understand how embargos work and the
  # tricky case of the Tribble 4-way race condition. See the comments for the Disembargo message,
  # below.

  promiseId @0 :ExportId;
  # The ID of the promise to be resolved.
  #
  # Unlike all other instances of `ExportId` sent from the exporter, the `Resolve` message does
  # _not_ increase the reference count of `promiseId`.  In fact, it is expected that the receiver
  # will release the export soon after receiving `Resolve`, and the sender will not send this
  # `ExportId` again until it has been released and recycled.
  #
  # When an export ID sent over the wire (e.g. in a `CapDescriptor`) is indicated to be a promise,
  # this indicates that the sender will follow up at some point with a `Resolve` message.  If the
  # same `promiseId` is sent again before `Resolve`, still only one `Resolve` is sent.  If the
  # same ID is sent again later _after_ a `Resolve`, it can only be because the export's
  # reference count hit zero in the meantime and the ID was re-assigned to a new export, therefore
  # this later promise does _not_ correspond to the earlier `Resolve`.
  #
  # If a promise ID's reference count reaches zero before a `Resolve` is sent, the `Resolve`
  # message may or may not still be sent (the `Resolve` may have already been in-flight when
  # `Release` was sent, but if the `Release` is received before `Resolve` then there is no longer
  # any reason to send a `Resolve`).  Thus a `Resolve` may be received for a promise of which
  # the receiver has no knowledge, because it already released it earlier.  In this case, the
  # receiver should simply release the capability to which the promise resolved.

  union {
    cap @1 :CapDescriptor;
    # The object to which the promise resolved.
    #
    # The sender promises that from this point forth, until `promiseId` is released, it shall
    # simply forward all messages to the capability designated by `cap`.  This is true even if
    # `cap` itself happens to designate another promise, and that other promise later resolves --
    # messages sent to `promiseId` shall still go to that other promise, not to its resolution.
    # This is important in the case that the receiver of the `Resolve` ends up sending a
    # `Disembargo` message towards `promiseId` in order to control message ordering -- that
    # `Disembargo` really needs to reflect back to exactly the object designated by `cap` even
    # if that object is itself a promise.

    exception @2 :Exception;
    # Indicates that the promise was broken.
  }
}

struct Release {
  # **(level 1)**
  #
  # Message type sent to indicate that the sender is done with the given capability and the receiver
  # can free resources allocated to it.

  id @0 :ImportId;
  # What to release.

  referenceCount @1 :UInt32;
  # The amount by which to decrement the reference count.  The export is only actually released
  # when the reference count reaches zero.
}

struct Disembargo {
  # **(level 1)**
  #
  # Message sent to indicate that an embargo on a recently-resolved promise may now be lifted.
  #
  # Embargos are used to enforce E-order in the presence of promise resolution.  That is, if an
  # application makes two calls foo() and bar() on the same capability reference, in that order,
  # the calls should be delivered in the order in which they were made.  But if foo() is called
  # on a promise, and that promise happens to resolve before bar() is called, then the two calls
  # may travel different paths over the network, and thus could arrive in the wrong order.  In
  # this case, the call to `bar()` must be embargoed, and a `Disembargo` message must be sent along
  # the same path as `foo()` to ensure that the `Disembargo` arrives after `foo()`.  Once the
  # `Disembargo` arrives, `bar()` can then be delivered.
  #
  # There are two particular cases where embargos are important.  Consider object Alice, in Vat A,
  # who holds a promise P, pointing towards Vat B, that eventually resolves to Carol.  The two
  # cases are:
  # - Carol lives in Vat A, i.e. next to Alice.  In this case, Vat A needs to send a `Disembargo`
  #   message that echos through Vat B and back, to ensure that all pipelined calls on the promise
  #   have been delivered.
  # - Carol lives in a different Vat C.  When the promise resolves, a three-party handoff occurs
  #   (see `Provide` and `Accept`, which constitute level 3 of the protocol).  In this case, we
  #   piggyback on the state that has already been set up to handle the handoff:  the `Accept`
  #   message (from Vat A to Vat C) is embargoed, as are all pipelined messages sent to it, while
  #   a `Disembargo` message is sent from Vat A through Vat B to Vat C.  See `Accept.embargo` for
  #   an example.
  #
  # Note that in the case where Carol actually lives in Vat B (i.e., the same vat that the promise
  # already pointed at), no embargo is needed, because the pipelined calls are delivered over the
  # same path as the later direct calls.
  #
  # Keep in mind that promise resolution happens both in the form of Resolve messages as well as
  # Return messages (which resolve PromisedAnswers). Embargos apply in both cases.
  #
  # An alternative strategy for enforcing E-order over promise resolution could be for Vat A to
  # implement the embargo internally.  When Vat A is notified of promise resolution, it could
  # send a dummy no-op call to promise P and wait for it to complete.  Until that call completes,
  # all calls to the capability are queued locally.  This strategy works, but is pessimistic:
  # in the three-party case, it requires an A -> B -> C -> B -> A round trip before calls can start
  # being delivered directly to from Vat A to Vat C.  The `Disembargo` message allows latency to be
  # reduced.  (In the two-party loopback case, the `Disembargo` message is just a more explicit way
  # of accomplishing the same thing as a no-op call, but isn't any faster.)
  #
  # *The Tribble 4-way Race Condition*
  #
  # Any implementation of promise resolution and embargos must be aware of what we call the
  # "Tribble 4-way race condition", after Dean Tribble, who explained the problem in a lively
  # Friam meeting.
  #
  # Embargos are designed to work in the case where a two-hop path is being shortened to one hop.
  # But sometimes there are more hops. Imagine that Alice has a reference to a remote promise P1
  # that eventually resolves to _another_ remote promise P2 (in a third vat), which _at the same
  # time_ happens to resolve to Bob (in a fourth vat). In this case, we're shortening from a 3-hop
  # path (with four parties) to a 1-hop path (Alice -> Bob).
  #
  # Extending the embargo/disembargo protocol to be able to shorted multiple hops at once seems
  # difficult. Instead, we make a rule that prevents this case from coming up:
  #
  # One a promise P has been resolved to a remote object reference R, then all further messages
  # received addressed to P will be forwarded strictly to R. Even if it turns out later that R is
  # itself a promise, and has resolved to some other object Q, messages sent to P will still be
  # forwarded to R, not directly to Q (R will of course further forward the messages to Q).
  #
  # This rule does not cause a significant performance burden because once P has resolved to R, it
  # is expected that people sending messages to P will shortly start sending them to R instead and
  # drop P. P is at end-of-life anyway, so it doesn't matter if it ignores chances to further
  # optimize its path.
  #
  # Note well: the Tribble 4-way race condition does not require each vat to be *distinct*; as long
  # as each resolution crosses a network boundary the race can occur -- so this concerns even level
  # 1 implementations, not just level 3 implementations.

  target @0 :MessageTarget;
  # What is to be disembargoed.

  using EmbargoId = UInt32;
  # Used in `senderLoopback` and `receiverLoopback`, below.

  context :union {
    senderLoopback @1 :EmbargoId;
    # The sender is requesting a disembargo on a promise that is known to resolve back to a
    # capability hosted by the sender.  As soon as the receiver has echoed back all pipelined calls
    # on this promise, it will deliver the Disembargo back to the sender with `receiverLoopback`
    # set to the same value as `senderLoopback`.  This value is chosen by the sender, and since
    # it is also consumed be the sender, the sender can use whatever strategy it wants to make sure
    # the value is unambiguous.
    #
    # The receiver must verify that the target capability actually resolves back to the sender's
    # vat.  Otherwise, the sender has committed a protocol error and should be disconnected.

    receiverLoopback @2 :EmbargoId;
    # The receiver previously sent a `senderLoopback` Disembargo towards a promise resolving to
    # this capability, and that Disembargo is now being echoed back.

    accept @3 :Void;
    # **(level 3)**
    #
    # The sender is requesting a disembargo on a promise that is known to resolve to a third-party
    # capability that the sender is currently in the process of accepting (using `Accept`).
    # The receiver of this `Disembargo` has an outstanding `Provide` on said capability.  The
    # receiver should now send a `Disembargo` with `provide` set to the question ID of that
    # `Provide` message.
    #
    # See `Accept.embargo` for an example.

    provide @4 :QuestionId;
    # **(level 3)**
    #
    # The sender is requesting a disembargo on a capability currently being provided to a third
    # party.  The question ID identifies the `Provide` message previously sent by the sender to
    # this capability.  On receipt, the receiver (the capability host) shall release the embargo
    # on the `Accept` message that it has received from the third party.  See `Accept.embargo` for
    # an example.
  }
}

# Level 2 message types ----------------------------------------------

# See persistent.capnp.

# Level 3 message types ----------------------------------------------

struct Provide {
  # **(level 3)**
  #
  # Message type sent to indicate that the sender wishes to make a particular capability implemented
  # by the receiver available to a third party for direct access (without the need for the third
  # party to proxy through the sender).
  #
  # (In CapTP, `Provide` and `Accept` are methods of the global `NonceLocator` object exported by
  # every vat.  In Cap'n Proto, we bake this into the core protocol.)

  questionId @0 :QuestionId;
  # Question ID to be held open until the recipient has received the capability.  A result will be
  # returned once the third party has successfully received the capability.  The sender must at some
  # point send a `Finish` message as with any other call, and that message can be used to cancel the
  # whole operation.

  target @1 :MessageTarget;
  # What is to be provided to the third party.

  recipient @2 :RecipientId;
  # Identity of the third party that is expected to pick up the capability.
}

struct Accept {
  # **(level 3)**
  #
  # Message type sent to pick up a capability hosted by the receiving vat and provided by a third
  # party.  The third party previously designated the capability using `Provide`.
  #
  # This message is also used to pick up a redirected return -- see `Return.acceptFromThirdParty`.

  questionId @0 :QuestionId;
  # A new question ID identifying this accept message, which will eventually receive a Return
  # message containing the provided capability (or the call result in the case of a redirected
  # return).

  provision @1 :ProvisionId;
  # Identifies the provided object to be picked up.

  embargo @2 :Bool;
  # If true, this accept shall be temporarily embargoed.  The resulting `Return` will not be sent,
  # and any pipelined calls will not be delivered, until the embargo is released.  The receiver
  # (the capability host) will expect the provider (the vat that sent the `Provide` message) to
  # eventually send a `Disembargo` message with the field `context.provide` set to the question ID
  # of the original `Provide` message.  At that point, the embargo is released and the queued
  # messages are delivered.
  #
  # For example:
  # - Alice, in Vat A, holds a promise P, which currently points toward Vat B.
  # - Alice calls foo() on P.  The `Call` message is sent to Vat B.
  # - The promise P in Vat B ends up resolving to Carol, in Vat C.
  # - Vat B sends a `Provide` message to Vat C, identifying Vat A as the recipient.
  # - Vat B sends a `Resolve` message to Vat A, indicating that the promise has resolved to a
  #   `ThirdPartyCapId` identifying Carol in Vat C.
  # - Vat A sends an `Accept` message to Vat C to pick up the capability.  Since Vat A knows that
  #   it has an outstanding call to the promise, it sets `embargo` to `true` in the `Accept`
  #   message.
  # - Vat A sends a `Disembargo` message to Vat B on promise P, with `context.accept` set.
  # - Alice makes a call bar() to promise P, which is now pointing towards Vat C.  Alice doesn't
  #   know anything about the mechanics of promise resolution happening under the hood, but she
  #   expects that bar() will be delivered after foo() because that is the order in which she
  #   initiated the calls.
  # - Vat A sends the bar() call to Vat C, as a pipelined call on the result of the `Accept` (which
  #   hasn't returned yet, due to the embargo).  Since calls to the newly-accepted capability
  #   are embargoed, Vat C does not deliver the call yet.
  # - At some point, Vat B forwards the foo() call from the beginning of this example on to Vat C.
  # - Vat B forwards the `Disembargo` from Vat A on to vat C.  It sets `context.provide` to the
  #   question ID of the `Provide` message it had sent previously.
  # - Vat C receives foo() before `Disembargo`, thus allowing it to correctly deliver foo()
  #   before delivering bar().
  # - Vat C receives `Disembargo` from Vat B.  It can now send a `Return` for the `Accept` from
  #   Vat A, as well as deliver bar().
}

# Level 4 message types ----------------------------------------------

struct Join {
  # **(level 4)**
  #
  # Message type sent to implement E.join(), which, given a number of capabilities that are
  # expected to be equivalent, finds the underlying object upon which they all agree and forms a
  # direct connection to it, skipping any proxies that may have been constructed by other vats
  # while transmitting the capability.  See:
  #     http://erights.org/elib/equality/index.html
  #
  # Note that this should only serve to bypass fully-transparent proxies -- proxies that were
  # created merely for convenience, without any intention of hiding the underlying object.
  #
  # For example, say Bob holds two capabilities hosted by Alice and Carol, but he expects that both
  # are simply proxies for a capability hosted elsewhere.  He then issues a join request, which
  # operates as follows:
  # - Bob issues Join requests on both Alice and Carol.  Each request contains a different piece
  #   of the JoinKey.
  # - Alice is proxying a capability hosted by Dana, so forwards the request to Dana's cap.
  # - Dana receives the first request and sees that the JoinKeyPart is one of two.  She notes that
  #   she doesn't have the other part yet, so she records the request and responds with a
  #   JoinResult.
  # - Alice relays the JoinAnswer back to Bob.
  # - Carol is also proxying a capability from Dana, and so forwards her Join request to Dana as
  #   well.
  # - Dana receives Carol's request and notes that she now has both parts of a JoinKey.  She
  #   combines them in order to form information needed to form a secure connection to Bob.  She
  #   also responds with another JoinResult.
  # - Bob receives the responses from Alice and Carol.  He uses the returned JoinResults to
  #   determine how to connect to Dana and attempts to form the connection.  Since Bob and Dana now
  #   agree on a secret key that neither Alice nor Carol ever saw, this connection can be made
  #   securely even if Alice or Carol is conspiring against the other.  (If Alice and Carol are
  #   conspiring _together_, they can obviously reproduce the key, but this doesn't matter because
  #   the whole point of the join is to verify that Alice and Carol agree on what capability they
  #   are proxying.)
  #
  # If the two capabilities aren't actually proxies of the same object, then the join requests
  # will come back with conflicting `hostId`s and the join will fail before attempting to form any
  # connection.

  questionId @0 :QuestionId;
  # Question ID used to respond to this Join.  (Note that this ID only identifies one part of the
  # request for one hop; each part has a different ID and relayed copies of the request have
  # (probably) different IDs still.)
  #
  # The receiver will reply with a `Return` whose `results` is a JoinResult.  This `JoinResult`
  # is relayed from the joined object's host, possibly with transformation applied as needed
  # by the network.
  #
  # Like any return, the result must be released using a `Finish`.  However, this release
  # should not occur until the joiner has either successfully connected to the joined object.
  # Vats relaying a `Join` message similarly must not release the result they receive until the
  # return they relayed back towards the joiner has itself been released.  This allows the
  # joined object's host to detect when the Join operation is canceled before completing -- if
  # it receives a `Finish` for one of the join results before the joiner successfully
  # connects.  It can then free any resources it had allocated as part of the join.

  target @1 :MessageTarget;
  # The capability to join.

  keyPart @2 :JoinKeyPart;
  # A part of the join key.  These combine to form the complete join key, which is used to establish
  # a direct connection.

  # TODO(before implementing):  Change this so that multiple parts can be sent in a single Join
  # message, so that if multiple join parts are going to cross the same connection they can be sent
  # together, so that the receive can potentially optimize its handling of them.  In the case where
  # all parts are bundled together, should the recipient be expected to simply return a cap, so
  # that the caller can immediately start pipelining to it?
}

# ========================================================================================
# Common structures used in messages

struct MessageTarget {
  # The target of a `Call` or other messages that target a capability.

  union {
    importedCap @0 :ImportId;
    # This message is to a capability or promise previously imported by the caller (exported by
    # the receiver).

    promisedAnswer @1 :PromisedAnswer;
    # This message is to a capability that is expected to be returned by another call that has not
    # yet been completed.
    #
    # At level 0, this is supported only for addressing the result of a previous `Bootstrap`, so
    # that initial startup doesn't require a round trip.
  }
}

struct Payload {
  # Represents some data structure that might contain capabilities.

  content @0 :AnyPointer;
  # Some Cap'n Proto data structure.  Capability pointers embedded in this structure index into
  # `capTable`.

  capTable @1 :List(CapDescriptor);
  # Descriptors corresponding to the cap pointers in `content`.
}

struct CapDescriptor {
  # **(level 1)**
  #
  # When an application-defined type contains an interface pointer, that pointer contains an index
  # into the message's capability table -- i.e. the `capTable` part of the `Payload`.  Each
  # capability in the table is represented as a `CapDescriptor`.  The runtime API should not reveal
  # the CapDescriptor directly to the application, but should instead wrap it in some kind of
  # callable object with methods corresponding to the interface that the capability implements.
  #
  # Keep in mind that `ExportIds` in a `CapDescriptor` are subject to reference counting.  See the
  # description of `ExportId`.
  #
  # Note that it is currently not possible to include a broken capability in the CapDescriptor
  # table.  Instead, create a new export (`senderPromise`) for each broken capability and then
  # immediately follow the payload-bearing Call or Return message with one Resolve message for each
  # broken capability, resolving it to an exception.

  union {
    none @0 :Void;
    # There is no capability here.  This `CapDescriptor` should not appear in the payload content.
    # A `none` CapDescriptor can be generated when an application inserts a capability into a
    # message and then later changes its mind and removes it -- rewriting all of the other
    # capability pointers may be hard, so instead a tombstone is left, similar to the way a removed
    # struct or list instance is zeroed out of the message but the space is not reclaimed.
    # Hopefully this is unusual.

    senderHosted @1 :ExportId;
    # The ID of a capability in the sender's export table (receiver's import table).  It may be a
    # newly allocated table entry, or an existing entry (increments the reference count).

    senderPromise @2 :ExportId;
    # A promise that the sender will resolve later.  The sender will send exactly one Resolve
    # message at a future point in time to replace this promise.  Note that even if the same
    # `senderPromise` is received multiple times, only one `Resolve` is sent to cover all of
    # them.  If `senderPromise` is released before the `Resolve` is sent, the sender (of this
    # `CapDescriptor`) may choose not to send the `Resolve` at all.

    receiverHosted @3 :ImportId;
    # A capability (or promise) previously exported by the receiver (imported by the sender).

    receiverAnswer @4 :PromisedAnswer;
    # A capability expected to be returned in the results of a currently-outstanding call posed
    # by the sender.

    thirdPartyHosted @5 :ThirdPartyCapDescriptor;
    # **(level 3)**
    #
    # A capability that lives in neither the sender's nor the receiver's vat.  The sender needs
    # to form a direct connection to a third party to pick up the capability.
    #
    # Level 1 and 2 implementations that receive a `thirdPartyHosted` may simply send calls to its
    # `vine` instead.
  }

  attachedFd @6 :UInt8 = 0xff;
  # If the RPC message in which this CapDescriptor was delivered also had file descriptors
  # attached, and `fd` is a valid index into the list of attached file descriptors, then
  # that file descriptor should be attached to this capability. If `attachedFd` is out-of-bounds
  # for said list, then no FD is attached.
  #
  # For example, if the RPC message arrived over a Unix socket, then file descriptors may be
  # attached by sending an SCM_RIGHTS ancillary message attached to the data bytes making up the
  # raw message. Receivers who wish to opt into FD passing should arrange to receive SCM_RIGHTS
  # whenever receiving an RPC message. Senders who wish to send FDs need not verify whether the
  # receiver knows how to receive them, because the operating system will automatically discard
  # ancillary messages like SCM_RIGHTS if the receiver doesn't ask to receive them, including
  # automatically closing any FDs.
  #
  # It is up to the application protocol to define what capabilities are expected to have file
  # descriptors attached, and what those FDs mean. But, for example, an application could use this
  # to open a file on disk and then transmit the open file descriptor to a sandboxed process that
  # does not otherwise have permission to access the filesystem directly. This is usually an
  # optimization: the sending process could instead provide an RPC interface supporting all the
  # operations needed (such as reading and writing a file), but by passing the file descriptor
  # directly, the recipient can often perform operations much more efficiently. Application
  # designers are encouraged to provide such RPC interfaces and automatically fall back to them
  # when FD passing is not available, so that the application can still work when the parties are
  # remote over a network.
  #
  # An attached FD is most often associated with a `senderHosted` descriptor. It could also make
  # sense in the case of `thirdPartyHosted`: in this case, the sender is forwarding the FD that
  # they received from the third party, so that the receiver can start using it without first
  # interacting with the third party. This is an optional optimization -- the middleman may choose
  # not to forward capabilities, in which case the receiver will need to complete the handshake
  # with the third party directly before receiving the FD. If an implementation receives a second
  # attached FD after having already received one previously (e.g. both in a `thirdPartyHosted`
  # CapDescriptor and then later again when receiving the final capability directly from the
  # third party), the implementation should discard the later FD and stick with the original. At
  # present, there is no known reason why other capability types (e.g. `receiverHosted`) would want
  # to carry an attached FD, but we reserve the right to define a meaning for this in the future.
  #
  # Each file descriptor attached to the message must be used in no more than one CapDescriptor,
  # so that the receiver does not need to use dup() or refcounting to handle the possibility of
  # multiple capabilities using the same descriptor. If multiple CapDescriptors do point to the
  # same FD index, then the receiver can arbitrarily choose which capability ends up having the
  # FD attached.
  #
  # To mitigate DoS attacks, RPC implementations should limit the number of FDs they are willing to
  # receive in a single message to a small value. If a message happens to contain more than that,
  # the list is truncated. Moreover, in some cases, FD passing needs to be blocked entirely for
  # security or implementation reasons, in which case the list may be truncated to zero. Hence,
  # `attachedFd` might point past the end of the list, which the implementation should treat as if
  # no FD was attached at all.
  #
  # The type of this field was chosen to be UInt8 because Linux supports sending only a maximum
  # of 253 file descriptors in an SCM_RIGHTS message anyway, and CapDescriptor had two bytes of
  # padding left -- so after adding this, there is still one byte for a future feature.
  # Conveniently, this also means we're able to use 0xff as the default value, which will always
  # be out-of-range (of course, the implementation should explicitly enforce that 255 descriptors
  # cannot be sent at once, rather than relying on Linux to do so).
}

struct PromisedAnswer {
  # **(mostly level 1)**
  #
  # Specifies how to derive a promise from an unanswered question, by specifying the path of fields
  # to follow from the root of the eventual result struct to get to the desired capability.  Used
  # to address method calls to a not-yet-returned capability or to pass such a capability as an
  # input to some other method call.
  #
  # Level 0 implementations must support `PromisedAnswer` only for the case where the answer is
  # to a `Bootstrap` message.  In this case, `path` is always empty since `Bootstrap` always returns
  # a raw capability.

  questionId @0 :QuestionId;
  # ID of the question (in the sender's question table / receiver's answer table) whose answer is
  # expected to contain the capability.

  transform @1 :List(Op);
  # Operations / transformations to apply to the result in order to get the capability actually
  # being addressed.  E.g. if the result is a struct and you want to call a method on a capability
  # pointed to by a field of the struct, you need a `getPointerField` op.

  struct Op {
    union {
      noop @0 :Void;
      # Does nothing.  This member is mostly defined so that we can make `Op` a union even
      # though (as of this writing) only one real operation is defined.

      getPointerField @1 :UInt16;
      # Get a pointer field within a struct.  The number is an index into the pointer section, NOT
      # a field ordinal, so that the receiver does not need to understand the schema.

      # TODO(someday):  We could add:
      # - For lists, the ability to address every member of the list, or a slice of the list, the
      #   result of which would be another list.  This is useful for implementing the equivalent of
      #   a SQL table join (not to be confused with the `Join` message type).
      # - Maybe some ability to test a union.
      # - Probably not a good idea:  the ability to specify an arbitrary script to run on the
      #   result.  We could define a little stack-based language where `Op` specifies one
      #   "instruction" or transformation to apply.  Although this is not a good idea
      #   (over-engineered), any narrower additions to `Op` should be designed as if this
      #   were the eventual goal.
    }
  }
}

struct ThirdPartyCapDescriptor {
  # **(level 3)**
  #
  # Identifies a capability in a third-party vat that the sender wants the receiver to pick up.

  id @0 :ThirdPartyCapId;
  # Identifies the third-party host and the specific capability to accept from it.

  vineId @1 :ExportId;
  # A proxy for the third-party object exported by the sender.  In CapTP terminology this is called
  # a "vine", because it is an indirect reference to the third-party object that snakes through the
  # sender vat.  This serves two purposes:
  #
  # * Level 1 and 2 implementations that don't understand how to connect to a third party may
  #   simply send calls to the vine.  Such calls will be forwarded to the third-party by the
  #   sender.
  #
  # * Level 3 implementations must release the vine only once they have successfully picked up the
  #   object from the third party.  This ensures that the capability is not released by the sender
  #   prematurely.
  #
  # The sender will close the `Provide` request that it has sent to the third party as soon as
  # it receives either a `Call` or a `Release` message directed at the vine.
}

struct Exception {
  # **(level 0)**
  #
  # Describes an arbitrary error that prevented an operation (e.g. a call) from completing.
  #
  # Cap'n Proto exceptions always indicate that something went wrong. In other words, in a fantasy
  # world where everything always works as expected, no exceptions would ever be thrown. Clients
  # should only ever catch exceptions as a means to implement fault-tolerance, where "fault" can
  # mean:
  # - Bugs.
  # - Invalid input.
  # - Configuration errors.
  # - Network problems.
  # - Insufficient resources.
  # - Version skew (unimplemented functionality).
  # - Other logistical problems.
  #
  # Exceptions should NOT be used to flag application-specific conditions that a client is expected
  # to handle in an application-specific way. Put another way, in the Cap'n Proto world,
  # "checked exceptions" (where an interface explicitly defines the exceptions it throws and
  # clients are forced by the type system to handle those exceptions) do NOT make sense.

  reason @0 :Text;
  # Human-readable failure description.

  type @3 :Type;
  # The type of the error. The purpose of this enum is not to describe the error itself, but
  # rather to describe how the client might want to respond to the error.

  enum Type {
    failed @0;
    # A generic problem occurred, and it is believed that if the operation were repeated without
    # any change in the state of the world, the problem would occur again.
    #
    # A client might respond to this error by logging it for investigation by the developer and/or
    # displaying it to the user.

    overloaded @1;
    # The request was rejected due to a temporary lack of resources.
    #
    # Examples include:
    # - There's not enough CPU time to keep up with incoming requests, so some are rejected.
    # - The server ran out of RAM or disk space during the request.
    # - The operation timed out (took significantly longer than it should have).
    #
    # A client might respond to this error by scheduling to retry the operation much later. The
    # client should NOT retry again immediately since this would likely exacerbate the problem.

    disconnected @2;
    # The method failed because a connection to some necessary capability was lost.
    #
    # Examples include:
    # - The client introduced the server to a third-party capability, the connection to that third
    #   party was subsequently lost, and then the client requested that the server use the dead
    #   capability for something.
    # - The client previously requested that the server obtain a capability from some third party.
    #   The server returned a capability to an object wrapping the third-party capability. Later,
    #   the server's connection to the third party was lost.
    # - The capability has been revoked. Revocation does not necessarily mean that the client is
    #   no longer authorized to use the capability; it is often used simply as a way to force the
    #   client to repeat the setup process, perhaps to efficiently move them to a new back-end or
    #   get them to recognize some other change that has occurred.
    #
    # A client should normally respond to this error by releasing all capabilities it is currently
    # holding related to the one it called and then re-creating them by restoring SturdyRefs and/or
    # repeating the method calls used to create them originally. In other words, disconnect and
    # start over. This should in turn cause the server to obtain a new copy of the capability that
    # it lost, thus making everything work.
    #
    # If the client receives another `disconnected` error in the process of rebuilding the
    # capability and retrying the call, it should treat this as an `overloaded` error: the network
    # is currently unreliable, possibly due to load or other temporary issues.

    unimplemented @3;
    # The server doesn't implement the requested method. If there is some other method that the
    # client could call (perhaps an older and/or slower interface), it should try that instead.
    # Otherwise, this should be treated like `failed`.
  }

  obsoleteIsCallersFault @1 :Bool;
  # OBSOLETE. Ignore.

  obsoleteDurability @2 :UInt16;
  # OBSOLETE. See `type` instead.

  trace @4 :Text;
  # Stack trace text from the remote server. The format is not specified. By default,
  # implementations do not provide stack traces; the application must explicitly enable them
  # when desired.
}

# ========================================================================================
# Network-specific Parameters
#
# Some parts of the Cap'n Proto RPC protocol are not specified here because different vat networks
# may wish to use different approaches to solving them.  For example, on the public internet, you
# may want to authenticate vats using public-key cryptography, but on a local intranet with trusted
# infrastructure, you may be happy to authenticate based on network address only, or some other
# lightweight mechanism.
#
# To accommodate this, we specify several "parameter" types.  Each type is defined here as an
# alias for `AnyPointer`, but a specific network will want to define a specific set of types to use.
# All vats in a vat network must agree on these parameters in order to be able to communicate.
# Inter-network communication can be accomplished through "gateways" that perform translation
# between the primitives used on each network; these gateways may need to be deeply stateful,
# depending on the translations they perform.
#
# For interaction over the global internet between parties with no other prior arrangement, a
# particular set of bindings for these types is defined elsewhere.  (TODO(someday): Specify where
# these common definitions live.)
#
# Another common network type is the two-party network, in which one of the parties typically
# interacts with the outside world entirely through the other party.  In such a connection between
# Alice and Bob, all objects that exist on Bob's other networks appear to Alice as if they were
# hosted by Bob himself, and similarly all objects on Alice's network (if she even has one) appear
# to Bob as if they were hosted by Alice.  This network type is interesting because from the point
# of view of a simple application that communicates with only one other party via the two-party
# protocol, there are no three-party interactions at all, and joins are unusually simple to
# implement, so implementing at level 4 is barely more complicated than implementing at level 1.
# Moreover, if you pair an app implementing the two-party network with a container that implements
# some other network, the app can then participate on the container's network just as if it
# implemented that network directly.  The types used by the two-party network are defined in
# `rpc-twoparty.capnp`.
#
# The things that we need to parameterize are:
# - How to store capabilities long-term without holding a connection open (mostly level 2).
# - How to authenticate vats in three-party introductions (level 3).
# - How to implement `Join` (level 4).
#
# Persistent references
# ---------------------
#
# **(mostly level 2)**
#
# We want to allow some capabilities to be stored long-term, even if a connection is lost and later
# recreated.  ExportId is a short-term identifier that is specific to a connection, so it doesn't
# help here.  We need a way to specify long-term identifiers, as well as a strategy for
# reconnecting to a referenced capability later.
#
# Three-party interactions
# ------------------------
#
# **(level 3)**
#
# In cases where more than two vats are interacting, we have situations where VatA holds a
# capability hosted by VatB and wants to send that capability to VatC.  This can be accomplished
# by VatA proxying requests on the new capability, but doing so has two big problems:
# - It's inefficient, requiring an extra network hop.
# - If VatC receives another capability to the same object from VatD, it is difficult for VatC to
#   detect that the two capabilities are really the same and to implement the E "join" operation,
#   which is necessary for certain four-or-more-party interactions, such as the escrow pattern.
#   See:  http://www.erights.org/elib/equality/grant-matcher/index.html
#
# Instead, we want a way for VatC to form a direct, authenticated connection to VatB.
#
# Join
# ----
#
# **(level 4)**
#
# The `Join` message type and corresponding operation arranges for a direct connection to be formed
# between the joiner and the host of the joined object, and this connection must be authenticated.
# Thus, the details are network-dependent.

using SturdyRef = AnyPointer;
# **(level 2)**
#
# Identifies a persisted capability that can be restored in the future. How exactly a SturdyRef
# is restored to a live object is specified along with the SturdyRef definition (i.e. not by
# rpc.capnp).
#
# Generally a SturdyRef needs to specify three things:
# - How to reach the vat that can restore the ref (e.g. a hostname or IP address).
# - How to authenticate the vat after connecting (e.g. a public key fingerprint).
# - The identity of a specific object hosted by the vat. Generally, this is an opaque pointer whose
#   format is defined by the specific vat -- the client has no need to inspect the object ID.
#   It is important that the object ID be unguessable if the object is not public (and objects
#   should almost never be public).
#
# The above are only suggestions. Some networks might work differently. For example, a private
# network might employ a special restorer service whose sole purpose is to restore SturdyRefs.
# In this case, the entire contents of SturdyRef might be opaque, because they are intended only
# to be forwarded to the restorer service.

using ProvisionId = AnyPointer;
# **(level 3)**
#
# The information that must be sent in an `Accept` message to identify the object being accepted.
#
# In a network where each vat has a public/private key pair, this could simply be the public key
# fingerprint of the provider vat along with a nonce matching the one in the `RecipientId` used
# in the `Provide` message sent from that provider.

using RecipientId = AnyPointer;
# **(level 3)**
#
# The information that must be sent in a `Provide` message to identify the recipient of the
# capability.
#
# In a network where each vat has a public/private key pair, this could simply be the public key
# fingerprint of the recipient along with a nonce matching the one in the `ProvisionId`.
#
# As another example, when communicating between processes on the same machine over Unix sockets,
# RecipientId could simply refer to a file descriptor attached to the message via SCM_RIGHTS.
# This file descriptor would be one end of a newly-created socketpair, with the other end having
# been sent to the capability's recipient in ThirdPartyCapId.

using ThirdPartyCapId = AnyPointer;
# **(level 3)**
#
# The information needed to connect to a third party and accept a capability from it.
#
# In a network where each vat has a public/private key pair, this could be a combination of the
# third party's public key fingerprint, hints on how to connect to the third party (e.g. an IP
# address), and the nonce used in the corresponding `Provide` message's `RecipientId` as sent
# to that third party (used to identify which capability to pick up).
#
# As another example, when communicating between processes on the same machine over Unix sockets,
# ThirdPartyCapId could simply refer to a file descriptor attached to the message via SCM_RIGHTS.
# This file descriptor would be one end of a newly-created socketpair, with the other end having
# been sent to the process hosting the capability in RecipientId.

using JoinKeyPart = AnyPointer;
# **(level 4)**
#
# A piece of a secret key.  One piece is sent along each path that is expected to lead to the same
# place.  Once the pieces are combined, a direct connection may be formed between the sender and
# the receiver, bypassing any men-in-the-middle along the paths.  See the `Join` message type.
#
# The motivation for Joins is discussed under "Supporting Equality" in the "Unibus" protocol
# sketch: http://www.erights.org/elib/distrib/captp/unibus.html
#
# In a network where each vat has a public/private key pair and each vat forms no more than one
# connection to each other vat, Joins will rarely -- perhaps never -- be needed, as objects never
# need to be transparently proxied and references to the same object sent over the same connection
# have the same export ID.  Thus, a successful join requires only checking that the two objects
# come from the same connection and have the same ID, and then completes immediately.
#
# However, in networks where two vats may form more than one connection between each other, or
# where proxying of objects occurs, joins are necessary.
#
# Typically, each JoinKeyPart would include a fixed-length data value such that all value parts
# XOR'd together forms a shared secret that can be used to form an encrypted connection between
# the joiner and the joined object's host.  Each JoinKeyPart should also include an indication of
# how many parts to expect and a hash of the shared secret (used to match up parts).

using JoinResult = AnyPointer;
# **(level 4)**
#
# Information returned as the result to a `Join` message, needed by the joiner in order to form a
# direct connection to a joined object.  This might simply be the address of the joined object's
# host vat, since the `JoinKey` has already been communicated so the two vats already have a shared
# secret to use to authenticate each other.
#
# The `JoinResult` should also contain information that can be used to detect when the Join
# requests ended up reaching different objects, so that this situation can be detected easily.
# This could be a simple matter of including a sequence number -- if the joiner receives two
# `JoinResult`s with sequence number 0, then they must have come from different objects and the
# whole join is a failure.

# ========================================================================================
# Network interface sketch
#
# The interfaces below are meant to be pseudo-code to illustrate how the details of a particular
# vat network might be abstracted away.  They are written like Cap'n Proto interfaces, but in
# practice you'd probably define these interfaces manually in the target programming language.  A
# Cap'n Proto RPC implementation should be able to use these interfaces without knowing the
# definitions of the various network-specific parameters defined above.

# interface VatNetwork {
#   # Represents a vat network, with the ability to connect to particular vats and receive
#   # connections from vats.
#   #
#   # Note that methods returning a `Connection` may return a pre-existing `Connection`, and the
#   # caller is expected to find and share state with existing users of the connection.
#
#   # Level 0 features -----------------------------------------------
#
#   connect(vatId :VatId) :Connection;
#   # Connect to the given vat.  The transport should return a promise that does not
#   # resolve until authentication has completed, but allows messages to be pipelined in before
#   # that; the transport either queues these messages until authenticated, or sends them encrypted
#   # such that only the authentic vat would be able to decrypt them.  The latter approach avoids a
#   # round trip for authentication.
#
#   accept() :Connection;
#   # Wait for the next incoming connection and return it.  Only connections formed by
#   # connect() are returned by this method.
#
#   # Level 4 features -----------------------------------------------
#
#   newJoiner(count :UInt32) :NewJoinerResponse;
#   # Prepare a new Join operation, which will eventually lead to forming a new direct connection
#   # to the host of the joined capability.  `count` is the number of capabilities to join.
#
#   struct NewJoinerResponse {
#     joinKeyParts :List(JoinKeyPart);
#     # Key parts to send in Join messages to each capability.
#
#     joiner :Joiner;
#     # Used to establish the final connection.
#   }
#
#   interface Joiner {
#     addJoinResult(result :JoinResult) :Void;
#     # Add a JoinResult received in response to one of the `Join` messages.  All `JoinResult`s
#     # returned from all paths must be added before trying to connect.
#
#     connect() :ConnectionAndProvisionId;
#     # Try to form a connection to the joined capability's host, verifying that it has received
#     # all of the JoinKeyParts.  Once the connection is formed, the caller should send an `Accept`
#     # message on it with the specified `ProvisionId` in order to receive the final capability.
#   }
#
#   acceptConnectionFromJoiner(parts :List(JoinKeyPart), paths :List(VatPath))
#       :ConnectionAndProvisionId;
#   # Called on a joined capability's host to receive the connection from the joiner, once all
#   # key parts have arrived.  The caller should expect to receive an `Accept` message over the
#   # connection with the given ProvisionId.
# }
#
# interface Connection {
#   # Level 0 features -----------------------------------------------
#
#   send(message :Message) :Void;
#   # Send the message.  Returns successfully when the message (and all preceding messages) has
#   # been acknowledged by the recipient.
#
#   receive() :Message;
#   # Receive the next message, and acknowledges receipt to the sender.  Messages are received in
#   # the order in which they are sent.
#
#   # Level 3 features -----------------------------------------------
#
#   introduceTo(recipient :Connection) :IntroductionInfo;
#   # Call before starting a three-way introduction, assuming a `Provide` message is to be sent on
#   # this connection and a `ThirdPartyCapId` is to be sent to `recipient`.
#
#   struct IntroductionInfo {
#     sendToRecipient :ThirdPartyCapId;
#     sendToTarget :RecipientId;
#   }
#
#   connectToIntroduced(capId :ThirdPartyCapId) :ConnectionAndProvisionId;
#   # Given a ThirdPartyCapId received over this connection, connect to the third party.  The
#   # caller should then send an `Accept` message over the new connection.
#
#   acceptIntroducedConnection(recipientId :RecipientId) :Connection;
#   # Given a RecipientId received in a `Provide` message on this `Connection`, wait for the
#   # recipient to connect, and return the connection formed.  Usually, the first message received
#   # on the new connection will be an `Accept` message.
# }
#
# struct ConnectionAndProvisionId {
#   # **(level 3)**
#
#   connection :Connection;
#   # Connection on which to issue `Accept` message.
#
#   provision :ProvisionId;
#   # `ProvisionId` to send in the `Accept` message.
# }
