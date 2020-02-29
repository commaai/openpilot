// Copyright (c) 2013-2016 Sandstorm Development Group, Inc. and contributors
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

#include <kj/common.h>
#include <kj/memory.h>
#include <kj/mutex.h>
#include <kj/debug.h>
#include "common.h"
#include "layout.h"
#include "any.h"

#ifndef CAPNP_MESSAGE_H_
#define CAPNP_MESSAGE_H_

#if defined(__GNUC__) && !defined(CAPNP_HEADER_WARNINGS)
#pragma GCC system_header
#endif

namespace capnp {

namespace _ {  // private
  class ReaderArena;
  class BuilderArena;
}

class StructSchema;
class Orphanage;
template <typename T>
class Orphan;

// =======================================================================================

struct ReaderOptions {
  // Options controlling how data is read.

  uint64_t traversalLimitInWords = 8 * 1024 * 1024;
  // Limits how many total words of data are allowed to be traversed.  Traversal is counted when
  // a new struct or list builder is obtained, e.g. from a get() accessor.  This means that calling
  // the getter for the same sub-struct multiple times will cause it to be double-counted.  Once
  // the traversal limit is reached, an error will be reported.
  //
  // This limit exists for security reasons.  It is possible for an attacker to construct a message
  // in which multiple pointers point at the same location.  This is technically invalid, but hard
  // to detect.  Using such a message, an attacker could cause a message which is small on the wire
  // to appear much larger when actually traversed, possibly exhausting server resources leading to
  // denial-of-service.
  //
  // It makes sense to set a traversal limit that is much larger than the underlying message.
  // Together with sensible coding practices (e.g. trying to avoid calling sub-object getters
  // multiple times, which is expensive anyway), this should provide adequate protection without
  // inconvenience.
  //
  // The default limit is 64 MiB.  This may or may not be a sensible number for any given use case,
  // but probably at least prevents easy exploitation while also avoiding causing problems in most
  // typical cases.

  int nestingLimit = 64;
  // Limits how deeply-nested a message structure can be, e.g. structs containing other structs or
  // lists of structs.
  //
  // Like the traversal limit, this limit exists for security reasons.  Since it is common to use
  // recursive code to traverse recursive data structures, an attacker could easily cause a stack
  // overflow by sending a very-deeply-nested (or even cyclic) message, without the message even
  // being very large.  The default limit of 64 is probably low enough to prevent any chance of
  // stack overflow, yet high enough that it is never a problem in practice.
};

class MessageReader {
  // Abstract interface for an object used to read a Cap'n Proto message.  Subclasses of
  // MessageReader are responsible for reading the raw, flat message content.  Callers should
  // usually call `messageReader.getRoot<MyStructType>()` to get a `MyStructType::Reader`
  // representing the root of the message, then use that to traverse the message content.
  //
  // Some common subclasses of `MessageReader` include `SegmentArrayMessageReader`, whose
  // constructor accepts pointers to the raw data, and `StreamFdMessageReader` (from
  // `serialize.h`), which reads the message from a file descriptor.  One might implement other
  // subclasses to handle things like reading from shared memory segments, mmap()ed files, etc.

public:
  MessageReader(ReaderOptions options);
  // It is suggested that subclasses take ReaderOptions as a constructor parameter, but give it a
  // default value of "ReaderOptions()".  The base class constructor doesn't have a default value
  // in order to remind subclasses that they really need to give the user a way to provide this.

  virtual ~MessageReader() noexcept(false);

  virtual kj::ArrayPtr<const word> getSegment(uint id) = 0;
  // Gets the segment with the given ID, or returns null if no such segment exists. This method
  // will be called at most once for each segment ID.

  inline const ReaderOptions& getOptions();
  // Get the options passed to the constructor.

  template <typename RootType>
  typename RootType::Reader getRoot();
  // Get the root struct of the message, interpreting it as the given struct type.

  template <typename RootType, typename SchemaType>
  typename RootType::Reader getRoot(SchemaType schema);
  // Dynamically interpret the root struct of the message using the given schema (a StructSchema).
  // RootType in this case must be DynamicStruct, and you must #include <capnp/dynamic.h> to
  // use this.

  bool isCanonical();
  // Returns whether the message encoded in the reader is in canonical form.

private:
  ReaderOptions options;

  // Space in which we can construct a ReaderArena.  We don't use ReaderArena directly here
  // because we don't want clients to have to #include arena.h, which itself includes a bunch of
  // big STL headers.  We don't use a pointer to a ReaderArena because that would require an
  // extra malloc on every message which could be expensive when processing small messages.
  void* arenaSpace[15 + sizeof(kj::MutexGuarded<void*>) / sizeof(void*)];
  bool allocatedArena;

  _::ReaderArena* arena() { return reinterpret_cast<_::ReaderArena*>(arenaSpace); }
  AnyPointer::Reader getRootInternal();
};

class MessageBuilder {
  // Abstract interface for an object used to allocate and build a message.  Subclasses of
  // MessageBuilder are responsible for allocating the space in which the message will be written.
  // The most common subclass is `MallocMessageBuilder`, but other subclasses may be used to do
  // tricky things like allocate messages in shared memory or mmap()ed files.
  //
  // Creating a new message ususually means allocating a new MessageBuilder (ideally on the stack)
  // and then calling `messageBuilder.initRoot<MyStructType>()` to get a `MyStructType::Builder`.
  // That, in turn, can be used to fill in the message content.  When done, you can call
  // `messageBuilder.getSegmentsForOutput()` to get a list of flat data arrays containing the
  // message.

public:
  MessageBuilder();
  virtual ~MessageBuilder() noexcept(false);
  KJ_DISALLOW_COPY(MessageBuilder);

  struct SegmentInit {
    kj::ArrayPtr<word> space;

    size_t wordsUsed;
    // Number of words in `space` which are used; the rest are free space in which additional
    // objects may be allocated.
  };

  explicit MessageBuilder(kj::ArrayPtr<SegmentInit> segments);
  // Create a MessageBuilder backed by existing memory. This is an advanced interface that most
  // people should not use. THIS METHOD IS INSECURE; see below.
  //
  // This allows a MessageBuilder to be constructed to modify an in-memory message without first
  // making a copy of the content. This is especially useful in conjunction with mmap().
  //
  // The contents of each segment must outlive the MessageBuilder, but the SegmentInit array itself
  // only need outlive the constructor.
  //
  // SECURITY: Do not use this in conjunction with untrusted data. This constructor assumes that
  //   the input message is valid. This constructor is designed to be used with data you control,
  //   e.g. an mmap'd file which is owned and accessed by only one program. When reading data you
  //   do not trust, you *must* load it into a Reader and then copy into a Builder as a means of
  //   validating the content.
  //
  // WARNING: It is NOT safe to initialize a MessageBuilder in this way from memory that is
  //   currently in use by another MessageBuilder or MessageReader. Other readers/builders will
  //   not observe changes to the segment sizes nor newly-allocated segments caused by allocating
  //   new objects in this message.

  virtual kj::ArrayPtr<word> allocateSegment(uint minimumSize) = 0;
  // Allocates an array of at least the given number of words, throwing an exception or crashing if
  // this is not possible.  It is expected that this method will usually return more space than
  // requested, and the caller should use that extra space as much as possible before allocating
  // more.  The returned space remains valid at least until the MessageBuilder is destroyed.
  //
  // Cap'n Proto will only call this once at a time, so the subclass need not worry about
  // thread-safety.

  template <typename RootType>
  typename RootType::Builder initRoot();
  // Initialize the root struct of the message as the given struct type.

  template <typename Reader>
  void setRoot(Reader&& value);
  // Set the root struct to a deep copy of the given struct.

  template <typename RootType>
  typename RootType::Builder getRoot();
  // Get the root struct of the message, interpreting it as the given struct type.

  template <typename RootType, typename SchemaType>
  typename RootType::Builder getRoot(SchemaType schema);
  // Dynamically interpret the root struct of the message using the given schema (a StructSchema).
  // RootType in this case must be DynamicStruct, and you must #include <capnp/dynamic.h> to
  // use this.

  template <typename RootType, typename SchemaType>
  typename RootType::Builder initRoot(SchemaType schema);
  // Dynamically init the root struct of the message using the given schema (a StructSchema).
  // RootType in this case must be DynamicStruct, and you must #include <capnp/dynamic.h> to
  // use this.

  template <typename T>
  void adoptRoot(Orphan<T>&& orphan);
  // Like setRoot() but adopts the orphan without copying.

  kj::ArrayPtr<const kj::ArrayPtr<const word>> getSegmentsForOutput();
  // Get the raw data that makes up the message.

  Orphanage getOrphanage();

  bool isCanonical();
  // Check whether the message builder is in canonical form

private:
  void* arenaSpace[22];
  // Space in which we can construct a BuilderArena.  We don't use BuilderArena directly here
  // because we don't want clients to have to #include arena.h, which itself includes a bunch of
  // big STL headers.  We don't use a pointer to a BuilderArena because that would require an
  // extra malloc on every message which could be expensive when processing small messages.

  bool allocatedArena = false;
  // We have to initialize the arena lazily because when we do so we want to allocate the root
  // pointer immediately, and this will allocate a segment, which requires a virtual function
  // call on the MessageBuilder.  We can't do such a call in the constructor since the subclass
  // isn't constructed yet.  This is kind of annoying because it means that getOrphanage() is
  // not thread-safe, but that shouldn't be a huge deal...

  _::BuilderArena* arena() { return reinterpret_cast<_::BuilderArena*>(arenaSpace); }
  _::SegmentBuilder* getRootSegment();
  AnyPointer::Builder getRootInternal();
};

template <typename RootType>
typename RootType::Reader readMessageUnchecked(const word* data);
// IF THE INPUT IS INVALID, THIS MAY CRASH, CORRUPT MEMORY, CREATE A SECURITY HOLE IN YOUR APP,
// MURDER YOUR FIRST-BORN CHILD, AND/OR BRING ABOUT ETERNAL DAMNATION ON ALL OF HUMANITY.  DO NOT
// USE UNLESS YOU UNDERSTAND THE CONSEQUENCES.
//
// Given a pointer to a known-valid message located in a single contiguous memory segment,
// returns a reader for that message.  No bounds-checking will be done while traversing this
// message.  Use this only if you have already verified that all pointers are valid and in-bounds,
// and there are no far pointers in the message.
//
// To create a message that can be passed to this function, build a message using a MallocAllocator
// whose preferred segment size is larger than the message size.  This guarantees that the message
// will be allocated as a single segment, meaning getSegmentsForOutput() returns a single word
// array.  That word array is your message; you may pass a pointer to its first word into
// readMessageUnchecked() to read the message.
//
// This can be particularly handy for embedding messages in generated code:  you can
// embed the raw bytes (using AlignedData) then make a Reader for it using this.  This is the way
// default values are embedded in code generated by the Cap'n Proto compiler.  E.g., if you have
// a message MyMessage, you can read its default value like so:
//    MyMessage::Reader reader = Message<MyMessage>::readMessageUnchecked(MyMessage::DEFAULT.words);
//
// To sanitize a message from an untrusted source such that it can be safely passed to
// readMessageUnchecked(), use copyToUnchecked().

template <typename Reader>
void copyToUnchecked(Reader&& reader, kj::ArrayPtr<word> uncheckedBuffer);
// Copy the content of the given reader into the given buffer, such that it can safely be passed to
// readMessageUnchecked().  The buffer's size must be exactly reader.totalSizeInWords() + 1,
// otherwise an exception will be thrown.  The buffer must be zero'd before calling.

template <typename RootType>
typename RootType::Reader readDataStruct(kj::ArrayPtr<const word> data);
// Interprets the given data as a single, data-only struct. Only primitive fields (booleans,
// numbers, and enums) will be readable; all pointers will be null. This is useful if you want
// to use Cap'n Proto as a language/platform-neutral way to pack some bits.
//
// The input is a word array rather than a byte array to enforce alignment. If you have a byte
// array which you know is word-aligned (or if your platform supports unaligned reads and you don't
// mind the performance penalty), then you can use `reinterpret_cast` to convert a byte array into
// a word array:
//
//     kj::arrayPtr(reinterpret_cast<const word*>(bytes.begin()),
//                  reinterpret_cast<const word*>(bytes.end()))

template <typename BuilderType>
typename kj::ArrayPtr<const word> writeDataStruct(BuilderType builder);
// Given a struct builder, get the underlying data section as a word array, suitable for passing
// to `readDataStruct()`.
//
// Note that you may call `.toBytes()` on the returned value to convert to `ArrayPtr<const byte>`.

template <typename Type>
static typename Type::Reader defaultValue();
// Get a default instance of the given struct or list type.
//
// TODO(cleanup):  Find a better home for this function?

// =======================================================================================

class SegmentArrayMessageReader: public MessageReader {
  // A simple MessageReader that reads from an array of word arrays representing all segments.
  // In particular you can read directly from the output of MessageBuilder::getSegmentsForOutput()
  // (although it would probably make more sense to call builder.getRoot().asReader() in that case).

public:
  SegmentArrayMessageReader(kj::ArrayPtr<const kj::ArrayPtr<const word>> segments,
                            ReaderOptions options = ReaderOptions());
  // Creates a message pointing at the given segment array, without taking ownership of the
  // segments.  All arrays passed in must remain valid until the MessageReader is destroyed.

  KJ_DISALLOW_COPY(SegmentArrayMessageReader);
  ~SegmentArrayMessageReader() noexcept(false);

  virtual kj::ArrayPtr<const word> getSegment(uint id) override;

private:
  kj::ArrayPtr<const kj::ArrayPtr<const word>> segments;
};

enum class AllocationStrategy: uint8_t {
  FIXED_SIZE,
  // The builder will prefer to allocate the same amount of space for each segment with no
  // heuristic growth.  It will still allocate larger segments when the preferred size is too small
  // for some single object.  This mode is generally not recommended, but can be particularly useful
  // for testing in order to force a message to allocate a predictable number of segments.  Note
  // that you can force every single object in the message to be located in a separate segment by
  // using this mode with firstSegmentWords = 0.

  GROW_HEURISTICALLY
  // The builder will heuristically decide how much space to allocate for each segment.  Each
  // allocated segment will be progressively larger than the previous segments on the assumption
  // that message sizes are exponentially distributed.  The total number of segments that will be
  // allocated for a message of size n is O(log n).
};

constexpr uint SUGGESTED_FIRST_SEGMENT_WORDS = 1024;
constexpr AllocationStrategy SUGGESTED_ALLOCATION_STRATEGY = AllocationStrategy::GROW_HEURISTICALLY;

class MallocMessageBuilder: public MessageBuilder {
  // A simple MessageBuilder that uses malloc() (actually, calloc()) to allocate segments.  This
  // implementation should be reasonable for any case that doesn't require writing the message to
  // a specific location in memory.

public:
  explicit MallocMessageBuilder(uint firstSegmentWords = SUGGESTED_FIRST_SEGMENT_WORDS,
      AllocationStrategy allocationStrategy = SUGGESTED_ALLOCATION_STRATEGY);
  // Creates a BuilderContext which allocates at least the given number of words for the first
  // segment, and then uses the given strategy to decide how much to allocate for subsequent
  // segments.  When choosing a value for firstSegmentWords, consider that:
  // 1) Reading and writing messages gets slower when multiple segments are involved, so it's good
  //    if most messages fit in a single segment.
  // 2) Unused bytes will not be written to the wire, so generally it is not a big deal to allocate
  //    more space than you need.  It only becomes problematic if you are allocating many messages
  //    in parallel and thus use lots of memory, or if you allocate so much extra space that just
  //    zeroing it out becomes a bottleneck.
  // The defaults have been chosen to be reasonable for most people, so don't change them unless you
  // have reason to believe you need to.

  explicit MallocMessageBuilder(kj::ArrayPtr<word> firstSegment,
      AllocationStrategy allocationStrategy = SUGGESTED_ALLOCATION_STRATEGY);
  // This version always returns the given array for the first segment, and then proceeds with the
  // allocation strategy.  This is useful for optimization when building lots of small messages in
  // a tight loop:  you can reuse the space for the first segment.
  //
  // firstSegment MUST be zero-initialized.  MallocMessageBuilder's destructor will write new zeros
  // over any space that was used so that it can be reused.

  KJ_DISALLOW_COPY(MallocMessageBuilder);
  virtual ~MallocMessageBuilder() noexcept(false);

  virtual kj::ArrayPtr<word> allocateSegment(uint minimumSize) override;

private:
  uint nextSize;
  AllocationStrategy allocationStrategy;

  bool ownFirstSegment;
  bool returnedFirstSegment;

  void* firstSegment;

  struct MoreSegments;
  kj::Maybe<kj::Own<MoreSegments>> moreSegments;
};

class FlatMessageBuilder: public MessageBuilder {
  // THIS IS NOT THE CLASS YOU'RE LOOKING FOR.
  //
  // If you want to write a message into already-existing scratch space, use `MallocMessageBuilder`
  // and pass the scratch space to its constructor.  It will then only fall back to malloc() if
  // the scratch space is not large enough.
  //
  // Do NOT use this class unless you really know what you're doing.  This class is problematic
  // because it requires advance knowledge of the size of your message, which is usually impossible
  // to determine without actually building the message.  The class was created primarily to
  // implement `copyToUnchecked()`, which itself exists only to support other internal parts of
  // the Cap'n Proto implementation.

public:
  explicit FlatMessageBuilder(kj::ArrayPtr<word> array);
  KJ_DISALLOW_COPY(FlatMessageBuilder);
  virtual ~FlatMessageBuilder() noexcept(false);

  void requireFilled();
  // Throws an exception if the flat array is not exactly full.

  virtual kj::ArrayPtr<word> allocateSegment(uint minimumSize) override;

private:
  kj::ArrayPtr<word> array;
  bool allocated;
};

// =======================================================================================
// implementation details

inline const ReaderOptions& MessageReader::getOptions() {
  return options;
}

template <typename RootType>
inline typename RootType::Reader MessageReader::getRoot() {
  return getRootInternal().getAs<RootType>();
}

template <typename RootType>
inline typename RootType::Builder MessageBuilder::initRoot() {
  return getRootInternal().initAs<RootType>();
}

template <typename Reader>
inline void MessageBuilder::setRoot(Reader&& value) {
  getRootInternal().setAs<FromReader<Reader>>(value);
}

template <typename RootType>
inline typename RootType::Builder MessageBuilder::getRoot() {
  return getRootInternal().getAs<RootType>();
}

template <typename T>
void MessageBuilder::adoptRoot(Orphan<T>&& orphan) {
  return getRootInternal().adopt(kj::mv(orphan));
}

template <typename RootType, typename SchemaType>
typename RootType::Reader MessageReader::getRoot(SchemaType schema) {
  return getRootInternal().getAs<RootType>(schema);
}

template <typename RootType, typename SchemaType>
typename RootType::Builder MessageBuilder::getRoot(SchemaType schema) {
  return getRootInternal().getAs<RootType>(schema);
}

template <typename RootType, typename SchemaType>
typename RootType::Builder MessageBuilder::initRoot(SchemaType schema) {
  return getRootInternal().initAs<RootType>(schema);
}

template <typename RootType>
typename RootType::Reader readMessageUnchecked(const word* data) {
  return AnyPointer::Reader(_::PointerReader::getRootUnchecked(data)).getAs<RootType>();
}

template <typename Reader>
void copyToUnchecked(Reader&& reader, kj::ArrayPtr<word> uncheckedBuffer) {
  FlatMessageBuilder builder(uncheckedBuffer);
  builder.setRoot(kj::fwd<Reader>(reader));
  builder.requireFilled();
}

template <typename RootType>
typename RootType::Reader readDataStruct(kj::ArrayPtr<const word> data) {
  return typename RootType::Reader(_::StructReader(data));
}

template <typename BuilderType>
typename kj::ArrayPtr<const word> writeDataStruct(BuilderType builder) {
  auto bytes = _::PointerHelpers<FromBuilder<BuilderType>>::getInternalBuilder(kj::mv(builder))
      .getDataSectionAsBlob();
  return kj::arrayPtr(reinterpret_cast<word*>(bytes.begin()),
                      reinterpret_cast<word*>(bytes.end()));
}

template <typename Type>
static typename Type::Reader defaultValue() {
  return typename Type::Reader(_::StructReader());
}

template <typename T>
kj::Array<word> canonicalize(T&& reader) {
    return _::PointerHelpers<FromReader<T>>::getInternalReader(reader).canonicalize();
}

}  // namespace capnp

#endif  // CAPNP_MESSAGE_H_
