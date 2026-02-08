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

#ifndef CAPNP_PRIVATE
#error "This header is only meant to be included by Cap'n Proto's own source code."
#endif

#include <kj/common.h>
#include <kj/mutex.h>
#include <kj/exception.h>
#include <kj/vector.h>
#include <kj/units.h>
#include "common.h"
#include "message.h"
#include "layout.h"
#include <kj/map.h>

#if !CAPNP_LITE
#include "capability.h"
#endif  // !CAPNP_LITE

CAPNP_BEGIN_HEADER

namespace capnp {

#if !CAPNP_LITE
class ClientHook;
#endif  // !CAPNP_LITE

namespace _ {  // private

class SegmentReader;
class SegmentBuilder;
class Arena;
class BuilderArena;
class ReadLimiter;

class Segment;
typedef kj::Id<uint32_t, Segment> SegmentId;

class ReadLimiter {
  // Used to keep track of how much data has been processed from a message, and cut off further
  // processing if and when a particular limit is reached.  This is primarily intended to guard
  // against maliciously-crafted messages which contain cycles or overlapping structures.  Cycles
  // and overlapping are not permitted by the Cap'n Proto format because in many cases they could
  // be used to craft a deceptively small message which could consume excessive server resources to
  // process, perhaps even sending it into an infinite loop.  Actually detecting overlaps would be
  // time-consuming, so instead we just keep track of how many words worth of data structures the
  // receiver has actually dereferenced and error out if this gets too high.
  //
  // This counting takes place as you call getters (for non-primitive values) on the message
  // readers.  If you call the same getter twice, the data it returns may be double-counted.  This
  // should not be a big deal in most cases -- just set the read limit high enough that it will
  // only trigger in unreasonable cases.
  //
  // This class is "safe" to use from multiple threads for its intended use case.  Threads may
  // overwrite each others' changes to the counter, but this is OK because it only means that the
  // limit is enforced a bit less strictly -- it will still kick in eventually.

public:
  inline explicit ReadLimiter();                     // No limit.
  inline explicit ReadLimiter(WordCount64 limit);    // Limit to the given number of words.

  inline void reset(WordCount64 limit);

  KJ_ALWAYS_INLINE(bool canRead(WordCount64 amount, Arena* arena));

  void unread(WordCount64 amount);
  // Adds back some words to the limit.  Useful when the caller knows they are double-reading
  // some data.

private:
  alignas(8) volatile uint64_t limit;
  // Current limit, decremented each time catRead() is called. We modify this variable using atomics
  // with "relaxed" thread safety to make TSAN happy (on ARM & x86 this is no different from a
  // regular read/write of the variable). See the class comment for why this is OK (previously we
  // used a regular volatile variable - this is just to make ASAN happy).
  //
  // alignas(8) is the default on 64-bit systems, but needed on 32-bit to avoid an expensive
  // unaligned atomic operation.

  KJ_DISALLOW_COPY_AND_MOVE(ReadLimiter);

  KJ_ALWAYS_INLINE(void setLimit(uint64_t newLimit)) {
#if defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(&limit, newLimit, __ATOMIC_RELAXED);
#else
    limit = newLimit;
#endif
  }

  KJ_ALWAYS_INLINE(uint64_t readLimit() const) {
#if defined(__GNUC__) || defined(__clang__)
    return __atomic_load_n(&limit, __ATOMIC_RELAXED);
#else
    return limit;
#endif
  }
};

#if !CAPNP_LITE
class BrokenCapFactory {
  // Callback for constructing broken caps.  We use this so that we can avoid arena.c++ having a
  // link-time dependency on capability code that lives in libcapnp-rpc.

public:
  virtual kj::Own<ClientHook> newBrokenCap(kj::StringPtr description) = 0;
  virtual kj::Own<ClientHook> newNullCap() = 0;
};
#endif  // !CAPNP_LITE

class SegmentReader {
public:
  inline SegmentReader(Arena* arena, SegmentId id, const word* ptr, SegmentWordCount size,
                       ReadLimiter* readLimiter);

  KJ_ALWAYS_INLINE(const word* checkOffset(const word* from, ptrdiff_t offset));
  // Adds the given offset to the given pointer, checks that it is still within the bounds of the
  // segment, then returns it. Note that the "end" pointer of the segment (which technically points
  // to the word after the last in the segment) is considered in-bounds for this purpose, so you
  // can't necessarily dereference it. You must call checkObject() next to check that the object
  // you want to read is entirely in-bounds.
  //
  // If `from + offset` is out-of-range, this returns a pointer to the end of the segment. Thus,
  // any non-zero-sized object will fail `checkObject()`. We do this instead of throwing to save
  // some code footprint.

  KJ_ALWAYS_INLINE(bool checkObject(const word* start, WordCountN<31> size));
  // Assuming that `start` is in-bounds for this segment (probably checked using `checkOffset()`),
  // check that `start + size` is also in-bounds, and hence the whole area in-between is valid.

  KJ_ALWAYS_INLINE(bool amplifiedRead(WordCount virtualAmount));
  // Indicates that the reader should pretend that `virtualAmount` additional data was read even
  // though no actual pointer was traversed. This is used e.g. when reading a struct list pointer
  // where the element sizes are zero -- the sender could set the list size arbitrarily high and
  // cause the receiver to iterate over this list even though the message itself is small, so we
  // need to defend against DoS attacks based on this.

  inline Arena* getArena();
  inline SegmentId getSegmentId();

  inline const word* getStartPtr();
  inline SegmentWordCount getOffsetTo(const word* ptr);
  inline SegmentWordCount getSize();

  inline kj::ArrayPtr<const word> getArray();

  inline void unread(WordCount64 amount);
  // Add back some words to the ReadLimiter.

private:
  Arena* arena;
  SegmentId id;
  kj::ArrayPtr<const word> ptr;  // size guaranteed to fit in SEGMENT_WORD_COUNT_BITS bits
  ReadLimiter* readLimiter;

  KJ_DISALLOW_COPY_AND_MOVE(SegmentReader);

  friend class SegmentBuilder;

  [[noreturn]] static void abortCheckObjectFault();
  // Called in debug mode in cases that would segfault in opt mode. (Should be impossible!)
};

class SegmentBuilder: public SegmentReader {
public:
  inline SegmentBuilder(BuilderArena* arena, SegmentId id, word* ptr, SegmentWordCount size,
                        ReadLimiter* readLimiter, SegmentWordCount wordsUsed = ZERO * WORDS);
  inline SegmentBuilder(BuilderArena* arena, SegmentId id, const word* ptr, SegmentWordCount size,
                        ReadLimiter* readLimiter);
  inline SegmentBuilder(BuilderArena* arena, SegmentId id, decltype(nullptr),
                        ReadLimiter* readLimiter);

  KJ_ALWAYS_INLINE(word* allocate(SegmentWordCount amount));

  KJ_ALWAYS_INLINE(void checkWritable());
  // Throw an exception if the segment is read-only (meaning it is a reference to external data).

  KJ_ALWAYS_INLINE(word* getPtrUnchecked(SegmentWordCount offset));
  // Get a writable pointer into the segment.  Throws an exception if the segment is read-only (i.e.
  // a reference to external immutable data).

  inline BuilderArena* getArena();

  inline kj::ArrayPtr<const word> currentlyAllocated();

  inline void reset();

  inline bool isWritable() { return !readOnly; }

  inline void tryTruncate(word* from, word* to);
  // If `from` points just past the current end of the segment, then move the end back to `to`.
  // Otherwise, do nothing.

  inline bool tryExtend(word* from, word* to);
  // If `from` points just past the current end of the segment, and `to` is within the segment
  // boundaries, then move the end up to `to` and return true. Otherwise, do nothing and return
  // false.

private:
  word* pos;
  // Pointer to a pointer to the current end point of the segment, i.e. the location where the
  // next object should be allocated.

  bool readOnly;

  [[noreturn]] void throwNotWritable();

  KJ_DISALLOW_COPY_AND_MOVE(SegmentBuilder);
};

class Arena {
public:
  virtual ~Arena() noexcept(false);

  virtual SegmentReader* tryGetSegment(SegmentId id) = 0;
  // Gets the segment with the given ID, or return nullptr if no such segment exists.

  virtual void reportReadLimitReached() = 0;
  // Called to report that the read limit has been reached.  See ReadLimiter, below.  This invokes
  // the VALIDATE_INPUT() macro which may throw an exception; if it returns normally, the caller
  // will need to continue with default values.
};

class ReaderArena final: public Arena {
public:
  explicit ReaderArena(MessageReader* message);
  ~ReaderArena() noexcept(false);
  KJ_DISALLOW_COPY_AND_MOVE(ReaderArena);

  size_t sizeInWords();

  // implements Arena ------------------------------------------------
  SegmentReader* tryGetSegment(SegmentId id) override;
  void reportReadLimitReached() override;

private:
  MessageReader* message;
  ReadLimiter readLimiter;

  // Optimize for single-segment messages so that small messages are handled quickly.
  SegmentReader segment0;

  typedef kj::HashMap<uint, kj::Own<SegmentReader>> SegmentMap;
  kj::MutexGuarded<kj::Maybe<SegmentMap>> moreSegments;
  // We need to mutex-guard the segment map because we lazily initialize segments when they are
  // first requested, but a Reader is allowed to be used concurrently in multiple threads.  Luckily
  // this only applies to large messages.
  //
  // TODO(perf):  Thread-local thing instead?  Some kind of lockless map?  Or do sharing of data
  //   in a different way, where you have to construct a new MessageReader in each thread (but
  //   possibly backed by the same data)?

  ReaderArena(MessageReader* message, kj::ArrayPtr<const word> firstSegment);
  ReaderArena(MessageReader* message, const word* firstSegment, SegmentWordCount firstSegmentSize);
};

class BuilderArena final: public Arena {
  // A BuilderArena that does not allow the injection of capabilities.

public:
  explicit BuilderArena(MessageBuilder* message);
  BuilderArena(MessageBuilder* message, kj::ArrayPtr<MessageBuilder::SegmentInit> segments);
  ~BuilderArena() noexcept(false);
  KJ_DISALLOW_COPY_AND_MOVE(BuilderArena);

  size_t sizeInWords();

  inline SegmentBuilder* getRootSegment() { return &segment0; }

  kj::ArrayPtr<const kj::ArrayPtr<const word>> getSegmentsForOutput();
  // Get an array of all the segments, suitable for writing out.  This only returns the allocated
  // portion of each segment, whereas tryGetSegment() returns something that includes
  // not-yet-allocated space.

  inline CapTableBuilder* getLocalCapTable() {
    // Return a CapTableBuilder that merely implements local loopback. That is, you can set
    // capabilities, then read the same capabilities back, but there is no intent ever to transmit
    // these capabilities. A MessageBuilder that isn't imbued with some other CapTable uses this
    // by default.
    //
    // TODO(cleanup): It's sort of a hack that this exists. In theory, perhaps, unimbued
    //   MessageBuilders should throw exceptions on any attempt to access capability fields, like
    //   unimbued MessageReaders do. However, lots of code exists which uses MallocMessageBuilder
    //   as a temporary holder for data to be copied in and out (without being serialized), and it
    //   is expected that such data can include capabilities, which is admittedly reasonable.
    //   Therefore, all MessageBuilders must have a cap table by default. Arguably we should
    //   deprecate this usage and instead define a new helper type for this exact purpose.

    return &localCapTable;
  }

  kj::Own<_::CapTableBuilder> releaseLocalCapTable() {
    return kj::heap<LocalCapTable>(kj::mv(localCapTable));
  }

  SegmentBuilder* getSegment(SegmentId id);
  // Get the segment with the given id.  Crashes or throws an exception if no such segment exists.

  struct AllocateResult {
    SegmentBuilder* segment;
    word* words;
  };

  AllocateResult allocate(SegmentWordCount amount);
  // Find a segment with at least the given amount of space available and allocate the space.
  // Note that allocating directly from a particular segment is much faster, but allocating from
  // the arena is guaranteed to succeed.  Therefore callers should try to allocate from a specific
  // segment first if there is one, then fall back to the arena.

  SegmentBuilder* addExternalSegment(kj::ArrayPtr<const word> content);
  // Add a new segment to the arena which points to some existing memory region.  The segment is
  // assumed to be completley full; the arena will never allocate from it.  In fact, the segment
  // is considered read-only.  Any attempt to get a Builder pointing into this segment will throw
  // an exception.  Readers are allowed, however.
  //
  // This can be used to inject some external data into a message without a copy, e.g. embedding a
  // large mmap'd file into a message as `Data` without forcing that data to actually be read in
  // from disk (until the message itself is written out).  `Orphanage` provides the public API for
  // this feature.

  // implements Arena ------------------------------------------------
  SegmentReader* tryGetSegment(SegmentId id) override;
  void reportReadLimitReached() override;

private:
  MessageBuilder* message;
  ReadLimiter dummyLimiter;

  class LocalCapTable final: public CapTableBuilder {
  public:
    kj::Maybe<kj::Own<ClientHook>> extractCap(uint index) override;
    uint injectCap(kj::Own<ClientHook>&& cap) override;
    void dropCap(uint index) override;

#if !CAPNP_LITE
  private:
    kj::Vector<kj::Maybe<kj::Own<ClientHook>>> capTable;
#endif // ! CAPNP_LITE
  };

  LocalCapTable localCapTable;

  SegmentBuilder segment0;
  kj::ArrayPtr<const word> segment0ForOutput;

  struct MultiSegmentState {
    kj::Vector<kj::Own<SegmentBuilder>> builders;
    kj::Vector<kj::ArrayPtr<const word>> forOutput;
  };
  kj::Maybe<kj::Own<MultiSegmentState>> moreSegments;

  SegmentBuilder* segmentWithSpace = nullptr;
  // When allocating, look for space in this segment first before resorting to allocating a new
  // segment.  This is not necessarily the last segment because addExternalSegment() may add a
  // segment that is already-full, in which case we don't update this pointer.

  template <typename T>  // Can be `word` or `const word`.
  SegmentBuilder* addSegmentInternal(kj::ArrayPtr<T> content);
};

// =======================================================================================

inline ReadLimiter::ReadLimiter()
    : limit(kj::maxValue) {}

inline ReadLimiter::ReadLimiter(WordCount64 limit): limit(unbound(limit / WORDS)) {}

inline void ReadLimiter::reset(WordCount64 limit) {
  setLimit(unbound(limit / WORDS));
}

inline bool ReadLimiter::canRead(WordCount64 amount, Arena* arena) {
  // Be careful not to store an underflowed value into `limit`, even if multiple threads are
  // decrementing it.
  uint64_t current = readLimit();
  if (KJ_UNLIKELY(unbound(amount / WORDS) > current)) {
    arena->reportReadLimitReached();
    return false;
  } else {
    setLimit(current - unbound(amount / WORDS));
    return true;
  }
}

// -------------------------------------------------------------------

inline SegmentReader::SegmentReader(Arena* arena, SegmentId id, const word* ptr,
                                    SegmentWordCount size, ReadLimiter* readLimiter)
    : arena(arena), id(id), ptr(kj::arrayPtr(ptr, unbound(size / WORDS))),
      readLimiter(readLimiter) {}

inline const word* SegmentReader::checkOffset(const word* from, ptrdiff_t offset) {
  ptrdiff_t min = ptr.begin() - from;
  ptrdiff_t max = ptr.end() - from;
  if (offset >= min && offset <= max) {
    return from + offset;
  } else {
    return ptr.end();
  }
}

inline bool SegmentReader::checkObject(const word* start, WordCountN<31> size) {
  auto startOffset = intervalLength(ptr.begin(), start, MAX_SEGMENT_WORDS);
#ifdef KJ_DEBUG
  if (startOffset > bounded(ptr.size()) * WORDS) {
    abortCheckObjectFault();
  }
#endif
  return startOffset + size <= bounded(ptr.size()) * WORDS &&
      readLimiter->canRead(size, arena);
}

inline bool SegmentReader::amplifiedRead(WordCount virtualAmount) {
  return readLimiter->canRead(virtualAmount, arena);
}

inline Arena* SegmentReader::getArena() { return arena; }
inline SegmentId SegmentReader::getSegmentId() { return id; }
inline const word* SegmentReader::getStartPtr() { return ptr.begin(); }
inline SegmentWordCount SegmentReader::getOffsetTo(const word* ptr) {
  KJ_IREQUIRE(this->ptr.begin() <= ptr && ptr <= this->ptr.end());
  return intervalLength(this->ptr.begin(), ptr, MAX_SEGMENT_WORDS);
}
inline SegmentWordCount SegmentReader::getSize() {
  return assumeBits<SEGMENT_WORD_COUNT_BITS>(ptr.size()) * WORDS;
}
inline kj::ArrayPtr<const word> SegmentReader::getArray() { return ptr; }
inline void SegmentReader::unread(WordCount64 amount) { readLimiter->unread(amount); }

// -------------------------------------------------------------------

inline SegmentBuilder::SegmentBuilder(
    BuilderArena* arena, SegmentId id, word* ptr, SegmentWordCount size,
    ReadLimiter* readLimiter, SegmentWordCount wordsUsed)
    : SegmentReader(arena, id, ptr, size, readLimiter),
      pos(ptr + wordsUsed), readOnly(false) {}
inline SegmentBuilder::SegmentBuilder(
    BuilderArena* arena, SegmentId id, const word* ptr, SegmentWordCount size,
    ReadLimiter* readLimiter)
    : SegmentReader(arena, id, ptr, size, readLimiter),
      // const_cast is safe here because the member won't ever be dereferenced because it appears
      // to point to the end of the segment anyway.
      pos(const_cast<word*>(ptr + size)), readOnly(true) {}
inline SegmentBuilder::SegmentBuilder(BuilderArena* arena, SegmentId id, decltype(nullptr),
                                      ReadLimiter* readLimiter)
    : SegmentReader(arena, id, nullptr, ZERO * WORDS, readLimiter),
      pos(nullptr), readOnly(false) {}

inline word* SegmentBuilder::allocate(SegmentWordCount amount) {
  if (intervalLength(pos, ptr.end(), MAX_SEGMENT_WORDS) < amount) {
    // Not enough space in the segment for this allocation.
    return nullptr;
  } else {
    // Success.
    word* result = pos;
    pos = pos + amount;
    return result;
  }
}

inline void SegmentBuilder::checkWritable() {
  if (KJ_UNLIKELY(readOnly)) throwNotWritable();
}

inline word* SegmentBuilder::getPtrUnchecked(SegmentWordCount offset) {
  return const_cast<word*>(ptr.begin() + offset);
}

inline BuilderArena* SegmentBuilder::getArena() {
  // Down-cast safe because SegmentBuilder's constructor always initializes its SegmentReader base
  // class with an Arena pointer that actually points to a BuilderArena.
  return static_cast<BuilderArena*>(arena);
}

inline kj::ArrayPtr<const word> SegmentBuilder::currentlyAllocated() {
  return kj::arrayPtr(ptr.begin(), pos - ptr.begin());
}

inline void SegmentBuilder::reset() {
  word* start = getPtrUnchecked(ZERO * WORDS);
  memset(start, 0, (pos - start) * sizeof(word));
  pos = start;
}

inline void SegmentBuilder::tryTruncate(word* from, word* to) {
  if (pos == from) pos = to;
}

inline bool SegmentBuilder::tryExtend(word* from, word* to) {
  // Careful about overflow.
  if (pos == from && to <= ptr.end() && to >= from) {
    pos = to;
    return true;
  } else {
    return false;
  }
}

}  // namespace _ (private)
}  // namespace capnp

CAPNP_END_HEADER
