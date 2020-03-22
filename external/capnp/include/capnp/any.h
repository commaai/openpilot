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

#ifndef CAPNP_ANY_H_
#define CAPNP_ANY_H_

#if defined(__GNUC__) && !defined(CAPNP_HEADER_WARNINGS)
#pragma GCC system_header
#endif

#include "layout.h"
#include "pointer-helpers.h"
#include "orphan.h"
#include "list.h"

namespace capnp {

class StructSchema;
class ListSchema;
class InterfaceSchema;
class Orphanage;
class ClientHook;
class PipelineHook;
struct PipelineOp;
struct AnyPointer;

struct AnyList {
  AnyList() = delete;

  class Reader;
  class Builder;
};

struct AnyStruct {
  AnyStruct() = delete;

  class Reader;
  class Builder;
  class Pipeline;
};

template<>
struct List<AnyStruct, Kind::OTHER> {
  List() = delete;

  class Reader;
  class Builder;
};

namespace _ {  // private
template <> struct Kind_<AnyPointer> { static constexpr Kind kind = Kind::OTHER; };
template <> struct Kind_<AnyStruct> { static constexpr Kind kind = Kind::OTHER; };
template <> struct Kind_<AnyList> { static constexpr Kind kind = Kind::OTHER; };
}  // namespace _ (private)

// =======================================================================================
// AnyPointer!

enum class Equality {
  NOT_EQUAL,
  EQUAL,
  UNKNOWN_CONTAINS_CAPS
};

kj::StringPtr KJ_STRINGIFY(Equality res);

struct AnyPointer {
  // Reader/Builder for the `AnyPointer` field type, i.e. a pointer that can point to an arbitrary
  // object.

  AnyPointer() = delete;

  class Reader {
  public:
    typedef AnyPointer Reads;

    Reader() = default;
    inline Reader(_::PointerReader reader): reader(reader) {}

    inline MessageSize targetSize() const;
    // Get the total size of the target object and all its children.

    inline PointerType getPointerType() const;

    inline bool isNull() const { return getPointerType() == PointerType::NULL_; }
    inline bool isStruct() const { return getPointerType() == PointerType::STRUCT; }
    inline bool isList() const { return getPointerType() == PointerType::LIST; }
    inline bool isCapability() const { return getPointerType() == PointerType::CAPABILITY; }

    Equality equals(AnyPointer::Reader right);
    bool operator==(AnyPointer::Reader right);
    inline bool operator!=(AnyPointer::Reader right) {
      return !(*this == right);
    }

    template <typename T>
    inline ReaderFor<T> getAs() const;
    // Valid for T = any generated struct type, interface type, List<U>, Text, or Data.

    template <typename T>
    inline ReaderFor<T> getAs(StructSchema schema) const;
    // Only valid for T = DynamicStruct.  Requires `#include <capnp/dynamic.h>`.

    template <typename T>
    inline ReaderFor<T> getAs(ListSchema schema) const;
    // Only valid for T = DynamicList.  Requires `#include <capnp/dynamic.h>`.

    template <typename T>
    inline ReaderFor<T> getAs(InterfaceSchema schema) const;
    // Only valid for T = DynamicCapability.  Requires `#include <capnp/dynamic.h>`.

#if !CAPNP_LITE
    kj::Own<ClientHook> getPipelinedCap(kj::ArrayPtr<const PipelineOp> ops) const;
    // Used by RPC system to implement pipelining.  Applications generally shouldn't use this
    // directly.
#endif  // !CAPNP_LITE

  private:
    _::PointerReader reader;
    friend struct AnyPointer;
    friend class Orphanage;
    friend class CapReaderContext;
    friend struct _::PointerHelpers<AnyPointer>;
  };

  class Builder {
  public:
    typedef AnyPointer Builds;

    Builder() = delete;
    inline Builder(decltype(nullptr)) {}
    inline Builder(_::PointerBuilder builder): builder(builder) {}

    inline MessageSize targetSize() const;
    // Get the total size of the target object and all its children.

    inline PointerType getPointerType();

    inline bool isNull() { return getPointerType() == PointerType::NULL_; }
    inline bool isStruct() { return getPointerType() == PointerType::STRUCT; }
    inline bool isList() { return getPointerType() == PointerType::LIST; }
    inline bool isCapability() { return getPointerType() == PointerType::CAPABILITY; }

    inline Equality equals(AnyPointer::Reader right) {
      return asReader().equals(right);
    }
    inline bool operator==(AnyPointer::Reader right) {
      return asReader() == right;
    }
    inline bool operator!=(AnyPointer::Reader right) {
      return !(*this == right);
    }

    inline void clear();
    // Set to null.

    template <typename T>
    inline BuilderFor<T> getAs();
    // Valid for T = any generated struct type, List<U>, Text, or Data.

    template <typename T>
    inline BuilderFor<T> getAs(StructSchema schema);
    // Only valid for T = DynamicStruct.  Requires `#include <capnp/dynamic.h>`.

    template <typename T>
    inline BuilderFor<T> getAs(ListSchema schema);
    // Only valid for T = DynamicList.  Requires `#include <capnp/dynamic.h>`.

    template <typename T>
    inline BuilderFor<T> getAs(InterfaceSchema schema);
    // Only valid for T = DynamicCapability.  Requires `#include <capnp/dynamic.h>`.

    template <typename T>
    inline BuilderFor<T> initAs();
    // Valid for T = any generated struct type.

    template <typename T>
    inline BuilderFor<T> initAs(uint elementCount);
    // Valid for T = List<U>, Text, or Data.

    template <typename T>
    inline BuilderFor<T> initAs(StructSchema schema);
    // Only valid for T = DynamicStruct.  Requires `#include <capnp/dynamic.h>`.

    template <typename T>
    inline BuilderFor<T> initAs(ListSchema schema, uint elementCount);
    // Only valid for T = DynamicList.  Requires `#include <capnp/dynamic.h>`.

    inline AnyList::Builder initAsAnyList(ElementSize elementSize, uint elementCount);
    // Note: Does not accept INLINE_COMPOSITE for elementSize.

    inline List<AnyStruct>::Builder initAsListOfAnyStruct(
        uint16_t dataWordCount, uint16_t pointerCount, uint elementCount);

    inline AnyStruct::Builder initAsAnyStruct(uint16_t dataWordCount, uint16_t pointerCount);

    template <typename T>
    inline void setAs(ReaderFor<T> value);
    // Valid for ReaderType = T::Reader for T = any generated struct type, List<U>, Text, Data,
    // DynamicStruct, or DynamicList (the dynamic types require `#include <capnp/dynamic.h>`).

    template <typename T>
    inline void setAs(std::initializer_list<ReaderFor<ListElementType<T>>> list);
    // Valid for T = List<?>.

    template <typename T>
    inline void setCanonicalAs(ReaderFor<T> value);

    inline void set(Reader value) { builder.copyFrom(value.reader); }
    // Set to a copy of another AnyPointer.

    inline void setCanonical(Reader value) { builder.copyFrom(value.reader, true); }

    template <typename T>
    inline void adopt(Orphan<T>&& orphan);
    // Valid for T = any generated struct type, List<U>, Text, Data, DynamicList, DynamicStruct,
    // or DynamicValue (the dynamic types require `#include <capnp/dynamic.h>`).

    template <typename T>
    inline Orphan<T> disownAs();
    // Valid for T = any generated struct type, List<U>, Text, Data.

    template <typename T>
    inline Orphan<T> disownAs(StructSchema schema);
    // Only valid for T = DynamicStruct.  Requires `#include <capnp/dynamic.h>`.

    template <typename T>
    inline Orphan<T> disownAs(ListSchema schema);
    // Only valid for T = DynamicList.  Requires `#include <capnp/dynamic.h>`.

    template <typename T>
    inline Orphan<T> disownAs(InterfaceSchema schema);
    // Only valid for T = DynamicCapability.  Requires `#include <capnp/dynamic.h>`.

    inline Orphan<AnyPointer> disown();
    // Disown without a type.

    inline Reader asReader() const { return Reader(builder.asReader()); }
    inline operator Reader() const { return Reader(builder.asReader()); }

  private:
    _::PointerBuilder builder;
    friend class Orphanage;
    friend class CapBuilderContext;
    friend struct _::PointerHelpers<AnyPointer>;
  };

#if !CAPNP_LITE
  class Pipeline {
  public:
    typedef AnyPointer Pipelines;

    inline Pipeline(decltype(nullptr)) {}
    inline explicit Pipeline(kj::Own<PipelineHook>&& hook): hook(kj::mv(hook)) {}

    Pipeline noop();
    // Just make a copy.

    Pipeline getPointerField(uint16_t pointerIndex);
    // Deprecated. In the future, we should use .asAnyStruct.getPointerField.

    inline AnyStruct::Pipeline asAnyStruct();

    kj::Own<ClientHook> asCap();
    // Expect that the result is a capability and construct a pipelined version of it now.

    inline kj::Own<PipelineHook> releasePipelineHook() { return kj::mv(hook); }
    // For use by RPC implementations.

    template <typename T, typename = kj::EnableIf<CAPNP_KIND(FromClient<T>) == Kind::INTERFACE>>
    inline operator T() { return T(asCap()); }

  private:
    kj::Own<PipelineHook> hook;
    kj::Array<PipelineOp> ops;

    inline Pipeline(kj::Own<PipelineHook>&& hook, kj::Array<PipelineOp>&& ops)
        : hook(kj::mv(hook)), ops(kj::mv(ops)) {}

    friend class LocalClient;
    friend class PipelineHook;
    friend class AnyStruct::Pipeline;
  };
#endif  // !CAPNP_LITE
};

template <>
class Orphan<AnyPointer> {
  // An orphaned object of unknown type.

public:
  Orphan() = default;
  KJ_DISALLOW_COPY(Orphan);
  Orphan(Orphan&&) = default;
  inline Orphan(_::OrphanBuilder&& builder)
      : builder(kj::mv(builder)) {}

  Orphan& operator=(Orphan&&) = default;

  template <typename T>
  inline Orphan(Orphan<T>&& other): builder(kj::mv(other.builder)) {}
  template <typename T>
  inline Orphan& operator=(Orphan<T>&& other) { builder = kj::mv(other.builder); return *this; }
  // Cast from typed orphan.

  // It's not possible to get an AnyPointer::{Reader,Builder} directly since there is no
  // underlying pointer (the pointer would normally live in the parent, but this object is
  // orphaned).  It is possible, however, to request typed readers/builders.

  template <typename T>
  inline BuilderFor<T> getAs();
  template <typename T>
  inline BuilderFor<T> getAs(StructSchema schema);
  template <typename T>
  inline BuilderFor<T> getAs(ListSchema schema);
  template <typename T>
  inline typename T::Client getAs(InterfaceSchema schema);
  template <typename T>
  inline ReaderFor<T> getAsReader() const;
  template <typename T>
  inline ReaderFor<T> getAsReader(StructSchema schema) const;
  template <typename T>
  inline ReaderFor<T> getAsReader(ListSchema schema) const;
  template <typename T>
  inline typename T::Client getAsReader(InterfaceSchema schema) const;

  template <typename T>
  inline Orphan<T> releaseAs();
  template <typename T>
  inline Orphan<T> releaseAs(StructSchema schema);
  template <typename T>
  inline Orphan<T> releaseAs(ListSchema schema);
  template <typename T>
  inline Orphan<T> releaseAs(InterfaceSchema schema);
  // Down-cast the orphan to a specific type.

  inline bool operator==(decltype(nullptr)) const { return builder == nullptr; }
  inline bool operator!=(decltype(nullptr)) const { return builder != nullptr; }

private:
  _::OrphanBuilder builder;

  template <typename, Kind>
  friend struct _::PointerHelpers;
  friend class Orphanage;
  template <typename U>
  friend class Orphan;
  friend class AnyPointer::Builder;
};

template <Kind k> struct AnyTypeFor_;
template <> struct AnyTypeFor_<Kind::STRUCT> { typedef AnyStruct Type; };
template <> struct AnyTypeFor_<Kind::LIST> { typedef AnyList Type; };

template <typename T>
using AnyTypeFor = typename AnyTypeFor_<CAPNP_KIND(T)>::Type;

template <typename T>
inline ReaderFor<AnyTypeFor<FromReader<T>>> toAny(T&& value) {
  return ReaderFor<AnyTypeFor<FromReader<T>>>(
      _::PointerHelpers<FromReader<T>>::getInternalReader(value));
}
template <typename T>
inline BuilderFor<AnyTypeFor<FromBuilder<T>>> toAny(T&& value) {
  return BuilderFor<AnyTypeFor<FromBuilder<T>>>(
      _::PointerHelpers<FromBuilder<T>>::getInternalBuilder(kj::mv(value)));
}

template <>
struct List<AnyPointer, Kind::OTHER> {
  // Note: This cannot be used for a list of structs, since such lists are not encoded as pointer
  //   lists! Use List<AnyStruct>.

  List() = delete;

  class Reader {
  public:
    typedef List<AnyPointer> Reads;

    inline Reader(): reader(ElementSize::POINTER) {}
    inline explicit Reader(_::ListReader reader): reader(reader) {}

    inline uint size() const { return unbound(reader.size() / ELEMENTS); }
    inline AnyPointer::Reader operator[](uint index) const {
      KJ_IREQUIRE(index < size());
      return AnyPointer::Reader(reader.getPointerElement(bounded(index) * ELEMENTS));
    }

    typedef _::IndexingIterator<const Reader, typename AnyPointer::Reader> Iterator;
    inline Iterator begin() const { return Iterator(this, 0); }
    inline Iterator end() const { return Iterator(this, size()); }

  private:
    _::ListReader reader;
    template <typename U, Kind K>
    friend struct _::PointerHelpers;
    template <typename U, Kind K>
    friend struct List;
    friend class Orphanage;
    template <typename U, Kind K>
    friend struct ToDynamic_;
  };

  class Builder {
  public:
    typedef List<AnyPointer> Builds;

    Builder() = delete;
    inline Builder(decltype(nullptr)): builder(ElementSize::POINTER) {}
    inline explicit Builder(_::ListBuilder builder): builder(builder) {}

    inline operator Reader() const { return Reader(builder.asReader()); }
    inline Reader asReader() const { return Reader(builder.asReader()); }

    inline uint size() const { return unbound(builder.size() / ELEMENTS); }
    inline AnyPointer::Builder operator[](uint index) {
      KJ_IREQUIRE(index < size());
      return AnyPointer::Builder(builder.getPointerElement(bounded(index) * ELEMENTS));
    }

    typedef _::IndexingIterator<Builder, typename AnyPointer::Builder> Iterator;
    inline Iterator begin() { return Iterator(this, 0); }
    inline Iterator end() { return Iterator(this, size()); }

  private:
    _::ListBuilder builder;
    template <typename, Kind>
    friend struct _::PointerHelpers;
    friend class Orphanage;
    template <typename, Kind>
    friend struct ToDynamic_;
  };
};

class AnyStruct::Reader {
public:
  typedef AnyStruct Reads;

  Reader() = default;
  inline Reader(_::StructReader reader): _reader(reader) {}

  template <typename T, typename = kj::EnableIf<CAPNP_KIND(FromReader<T>) == Kind::STRUCT>>
  inline Reader(T&& value)
      : _reader(_::PointerHelpers<FromReader<T>>::getInternalReader(kj::fwd<T>(value))) {}

  kj::ArrayPtr<const byte> getDataSection() {
    return _reader.getDataSectionAsBlob();
  }
  List<AnyPointer>::Reader getPointerSection() {
    return List<AnyPointer>::Reader(_reader.getPointerSectionAsList());
  }

  kj::Array<word> canonicalize() {
    return _reader.canonicalize();
  }

  Equality equals(AnyStruct::Reader right);
  bool operator==(AnyStruct::Reader right);
  inline bool operator!=(AnyStruct::Reader right) {
    return !(*this == right);
  }

  template <typename T>
  ReaderFor<T> as() const {
    // T must be a struct type.
    return typename T::Reader(_reader);
  }
private:
  _::StructReader _reader;

  template <typename, Kind>
  friend struct _::PointerHelpers;
  friend class Orphanage;
};

class AnyStruct::Builder {
public:
  typedef AnyStruct Builds;

  inline Builder(decltype(nullptr)) {}
  inline Builder(_::StructBuilder builder): _builder(builder) {}

#if !_MSC_VER  // TODO(msvc): MSVC ICEs on this. Try restoring when compiler improves.
  template <typename T, typename = kj::EnableIf<CAPNP_KIND(FromBuilder<T>) == Kind::STRUCT>>
  inline Builder(T&& value)
      : _builder(_::PointerHelpers<FromBuilder<T>>::getInternalBuilder(kj::fwd<T>(value))) {}
#endif

  inline kj::ArrayPtr<byte> getDataSection() {
    return _builder.getDataSectionAsBlob();
  }
  List<AnyPointer>::Builder getPointerSection() {
    return List<AnyPointer>::Builder(_builder.getPointerSectionAsList());
  }

  inline Equality equals(AnyStruct::Reader right) {
    return asReader().equals(right);
  }
  inline bool operator==(AnyStruct::Reader right) {
    return asReader() == right;
  }
  inline bool operator!=(AnyStruct::Reader right) {
    return !(*this == right);
  }

  inline operator Reader() const { return Reader(_builder.asReader()); }
  inline Reader asReader() const { return Reader(_builder.asReader()); }

  template <typename T>
  BuilderFor<T> as() {
    // T must be a struct type.
    return typename T::Builder(_builder);
  }
private:
  _::StructBuilder _builder;
  friend class Orphanage;
  friend class CapBuilderContext;
};

#if !CAPNP_LITE
class AnyStruct::Pipeline {
public:
  inline Pipeline(decltype(nullptr)): typeless(nullptr) {}
  inline explicit Pipeline(AnyPointer::Pipeline&& typeless)
      : typeless(kj::mv(typeless)) {}

  inline AnyPointer::Pipeline getPointerField(uint16_t pointerIndex) {
    // Return a new Promise representing a sub-object of the result.  `pointerIndex` is the index
    // of the sub-object within the pointer section of the result (the result must be a struct).
    //
    // TODO(perf):  On GCC 4.8 / Clang 3.3, use rvalue qualifiers to avoid the need for copies.
    //   Also make `ops` into a Vector to optimize this.
    return typeless.getPointerField(pointerIndex);
  }

private:
  AnyPointer::Pipeline typeless;
};
#endif  // !CAPNP_LITE

class List<AnyStruct, Kind::OTHER>::Reader {
public:
  typedef List<AnyStruct> Reads;

  inline Reader(): reader(ElementSize::INLINE_COMPOSITE) {}
  inline explicit Reader(_::ListReader reader): reader(reader) {}

  inline uint size() const { return unbound(reader.size() / ELEMENTS); }
  inline AnyStruct::Reader operator[](uint index) const {
    KJ_IREQUIRE(index < size());
    return AnyStruct::Reader(reader.getStructElement(bounded(index) * ELEMENTS));
  }

  typedef _::IndexingIterator<const Reader, typename AnyStruct::Reader> Iterator;
  inline Iterator begin() const { return Iterator(this, 0); }
  inline Iterator end() const { return Iterator(this, size()); }

private:
  _::ListReader reader;
  template <typename U, Kind K>
  friend struct _::PointerHelpers;
  template <typename U, Kind K>
  friend struct List;
  friend class Orphanage;
  template <typename U, Kind K>
  friend struct ToDynamic_;
};

class List<AnyStruct, Kind::OTHER>::Builder {
public:
  typedef List<AnyStruct> Builds;

  Builder() = delete;
  inline Builder(decltype(nullptr)): builder(ElementSize::INLINE_COMPOSITE) {}
  inline explicit Builder(_::ListBuilder builder): builder(builder) {}

  inline operator Reader() const { return Reader(builder.asReader()); }
  inline Reader asReader() const { return Reader(builder.asReader()); }

  inline uint size() const { return unbound(builder.size() / ELEMENTS); }
  inline AnyStruct::Builder operator[](uint index) {
    KJ_IREQUIRE(index < size());
    return AnyStruct::Builder(builder.getStructElement(bounded(index) * ELEMENTS));
  }

  typedef _::IndexingIterator<Builder, typename AnyStruct::Builder> Iterator;
  inline Iterator begin() { return Iterator(this, 0); }
  inline Iterator end() { return Iterator(this, size()); }

private:
  _::ListBuilder builder;
  template <typename U, Kind K>
  friend struct _::PointerHelpers;
  friend class Orphanage;
  template <typename U, Kind K>
  friend struct ToDynamic_;
};

class AnyList::Reader {
public:
  typedef AnyList Reads;

  inline Reader(): _reader(ElementSize::VOID) {}
  inline Reader(_::ListReader reader): _reader(reader) {}

#if !_MSC_VER  // TODO(msvc): MSVC ICEs on this. Try restoring when compiler improves.
  template <typename T, typename = kj::EnableIf<CAPNP_KIND(FromReader<T>) == Kind::LIST>>
  inline Reader(T&& value)
      : _reader(_::PointerHelpers<FromReader<T>>::getInternalReader(kj::fwd<T>(value))) {}
#endif

  inline ElementSize getElementSize() { return _reader.getElementSize(); }
  inline uint size() { return unbound(_reader.size() / ELEMENTS); }

  inline kj::ArrayPtr<const byte> getRawBytes() { return _reader.asRawBytes(); }

  Equality equals(AnyList::Reader right);
  bool operator==(AnyList::Reader right);
  inline bool operator!=(AnyList::Reader right) {
    return !(*this == right);
  }

  template <typename T> ReaderFor<T> as() {
    // T must be List<U>.
    return ReaderFor<T>(_reader);
  }
private:
  _::ListReader _reader;

  template <typename, Kind>
  friend struct _::PointerHelpers;
  friend class Orphanage;
};

class AnyList::Builder {
public:
  typedef AnyList Builds;

  inline Builder(decltype(nullptr)): _builder(ElementSize::VOID) {}
  inline Builder(_::ListBuilder builder): _builder(builder) {}

#if !_MSC_VER  // TODO(msvc): MSVC ICEs on this. Try restoring when compiler improves.
  template <typename T, typename = kj::EnableIf<CAPNP_KIND(FromBuilder<T>) == Kind::LIST>>
  inline Builder(T&& value)
      : _builder(_::PointerHelpers<FromBuilder<T>>::getInternalBuilder(kj::fwd<T>(value))) {}
#endif

  inline ElementSize getElementSize() { return _builder.getElementSize(); }
  inline uint size() { return unbound(_builder.size() / ELEMENTS); }

  Equality equals(AnyList::Reader right);
  inline bool operator==(AnyList::Reader right) {
    return asReader() == right;
  }
  inline bool operator!=(AnyList::Reader right) {
    return !(*this == right);
  }

  template <typename T> BuilderFor<T> as() {
    // T must be List<U>.
    return BuilderFor<T>(_builder);
  }

  inline operator Reader() const { return Reader(_builder.asReader()); }
  inline Reader asReader() const { return Reader(_builder.asReader()); }

private:
  _::ListBuilder _builder;

  friend class Orphanage;
};

// =======================================================================================
// Pipeline helpers
//
// These relate to capabilities, but we don't declare them in capability.h because generated code
// for structs needs to know about these, even in files that contain no interfaces.

#if !CAPNP_LITE

struct PipelineOp {
  // Corresponds to rpc.capnp's PromisedAnswer.Op.

  enum Type {
    NOOP,  // for convenience

    GET_POINTER_FIELD

    // There may be other types in the future...
  };

  Type type;
  union {
    uint16_t pointerIndex;  // for GET_POINTER_FIELD
  };
};

class PipelineHook {
  // Represents a currently-running call, and implements pipelined requests on its result.

public:
  virtual kj::Own<PipelineHook> addRef() = 0;
  // Increment this object's reference count.

  virtual kj::Own<ClientHook> getPipelinedCap(kj::ArrayPtr<const PipelineOp> ops) = 0;
  // Extract a promised Capability from the results.

  virtual kj::Own<ClientHook> getPipelinedCap(kj::Array<PipelineOp>&& ops);
  // Version of getPipelinedCap() passing the array by move.  May avoid a copy in some cases.
  // Default implementation just calls the other version.

  template <typename Pipeline, typename = FromPipeline<Pipeline>>
  static inline kj::Own<PipelineHook> from(Pipeline&& pipeline);

private:
  template <typename T> struct FromImpl;
};

#endif  // !CAPNP_LITE

// =======================================================================================
// Inline implementation details

inline MessageSize AnyPointer::Reader::targetSize() const {
  return reader.targetSize().asPublic();
}

inline PointerType AnyPointer::Reader::getPointerType() const {
  return reader.getPointerType();
}

template <typename T>
inline ReaderFor<T> AnyPointer::Reader::getAs() const {
  return _::PointerHelpers<T>::get(reader);
}

inline MessageSize AnyPointer::Builder::targetSize() const {
  return asReader().targetSize();
}

inline PointerType AnyPointer::Builder::getPointerType() {
  return builder.getPointerType();
}

inline void AnyPointer::Builder::clear() {
  return builder.clear();
}

template <typename T>
inline BuilderFor<T> AnyPointer::Builder::getAs() {
  return _::PointerHelpers<T>::get(builder);
}

template <typename T>
inline BuilderFor<T> AnyPointer::Builder::initAs() {
  return _::PointerHelpers<T>::init(builder);
}

template <typename T>
inline BuilderFor<T> AnyPointer::Builder::initAs(uint elementCount) {
  return _::PointerHelpers<T>::init(builder, elementCount);
}

inline AnyList::Builder AnyPointer::Builder::initAsAnyList(
    ElementSize elementSize, uint elementCount) {
  return AnyList::Builder(builder.initList(elementSize, bounded(elementCount) * ELEMENTS));
}

inline List<AnyStruct>::Builder AnyPointer::Builder::initAsListOfAnyStruct(
    uint16_t dataWordCount, uint16_t pointerCount, uint elementCount) {
  return List<AnyStruct>::Builder(builder.initStructList(bounded(elementCount) * ELEMENTS,
      _::StructSize(bounded(dataWordCount) * WORDS,
                    bounded(pointerCount) * POINTERS)));
}

inline AnyStruct::Builder AnyPointer::Builder::initAsAnyStruct(
    uint16_t dataWordCount, uint16_t pointerCount) {
  return AnyStruct::Builder(builder.initStruct(
      _::StructSize(bounded(dataWordCount) * WORDS,
                    bounded(pointerCount) * POINTERS)));
}

template <typename T>
inline void AnyPointer::Builder::setAs(ReaderFor<T> value) {
  return _::PointerHelpers<T>::set(builder, value);
}

template <typename T>
inline void AnyPointer::Builder::setCanonicalAs(ReaderFor<T> value) {
  return _::PointerHelpers<T>::setCanonical(builder, value);
}

template <typename T>
inline void AnyPointer::Builder::setAs(
    std::initializer_list<ReaderFor<ListElementType<T>>> list) {
  return _::PointerHelpers<T>::set(builder, list);
}

template <typename T>
inline void AnyPointer::Builder::adopt(Orphan<T>&& orphan) {
  _::PointerHelpers<T>::adopt(builder, kj::mv(orphan));
}

template <typename T>
inline Orphan<T> AnyPointer::Builder::disownAs() {
  return _::PointerHelpers<T>::disown(builder);
}

inline Orphan<AnyPointer> AnyPointer::Builder::disown() {
  return Orphan<AnyPointer>(builder.disown());
}

template <> struct ReaderFor_ <AnyPointer, Kind::OTHER> { typedef AnyPointer::Reader Type; };
template <> struct BuilderFor_<AnyPointer, Kind::OTHER> { typedef AnyPointer::Builder Type; };
template <> struct ReaderFor_ <AnyStruct, Kind::OTHER> { typedef AnyStruct::Reader Type; };
template <> struct BuilderFor_<AnyStruct, Kind::OTHER> { typedef AnyStruct::Builder Type; };

template <>
struct Orphanage::GetInnerReader<AnyPointer, Kind::OTHER> {
  static inline _::PointerReader apply(const AnyPointer::Reader& t) {
    return t.reader;
  }
};

template <>
struct Orphanage::GetInnerBuilder<AnyPointer, Kind::OTHER> {
  static inline _::PointerBuilder apply(AnyPointer::Builder& t) {
    return t.builder;
  }
};

template <>
struct Orphanage::GetInnerReader<AnyStruct, Kind::OTHER> {
  static inline _::StructReader apply(const AnyStruct::Reader& t) {
    return t._reader;
  }
};

template <>
struct Orphanage::GetInnerBuilder<AnyStruct, Kind::OTHER> {
  static inline _::StructBuilder apply(AnyStruct::Builder& t) {
    return t._builder;
  }
};

template <>
struct Orphanage::GetInnerReader<AnyList, Kind::OTHER> {
  static inline _::ListReader apply(const AnyList::Reader& t) {
    return t._reader;
  }
};

template <>
struct Orphanage::GetInnerBuilder<AnyList, Kind::OTHER> {
  static inline _::ListBuilder apply(AnyList::Builder& t) {
    return t._builder;
  }
};

template <typename T>
inline BuilderFor<T> Orphan<AnyPointer>::getAs() {
  return _::OrphanGetImpl<T>::apply(builder);
}
template <typename T>
inline ReaderFor<T> Orphan<AnyPointer>::getAsReader() const {
  return _::OrphanGetImpl<T>::applyReader(builder);
}
template <typename T>
inline Orphan<T> Orphan<AnyPointer>::releaseAs() {
  return Orphan<T>(kj::mv(builder));
}

// Using AnyPointer as the template type should work...

template <>
inline typename AnyPointer::Reader AnyPointer::Reader::getAs<AnyPointer>() const {
  return *this;
}
template <>
inline typename AnyPointer::Builder AnyPointer::Builder::getAs<AnyPointer>() {
  return *this;
}
template <>
inline typename AnyPointer::Builder AnyPointer::Builder::initAs<AnyPointer>() {
  clear();
  return *this;
}
template <>
inline void AnyPointer::Builder::setAs<AnyPointer>(AnyPointer::Reader value) {
  return builder.copyFrom(value.reader);
}
template <>
inline void AnyPointer::Builder::adopt<AnyPointer>(Orphan<AnyPointer>&& orphan) {
  builder.adopt(kj::mv(orphan.builder));
}
template <>
inline Orphan<AnyPointer> AnyPointer::Builder::disownAs<AnyPointer>() {
  return Orphan<AnyPointer>(builder.disown());
}
template <>
inline Orphan<AnyPointer> Orphan<AnyPointer>::releaseAs() {
  return kj::mv(*this);
}

namespace _ {  // private

// Specialize PointerHelpers for AnyPointer.

template <>
struct PointerHelpers<AnyPointer, Kind::OTHER> {
  static inline AnyPointer::Reader get(PointerReader reader,
                                       const void* defaultValue = nullptr,
                                       uint defaultBytes = 0) {
    return AnyPointer::Reader(reader);
  }
  static inline AnyPointer::Builder get(PointerBuilder builder,
                                        const void* defaultValue = nullptr,
                                        uint defaultBytes = 0) {
    return AnyPointer::Builder(builder);
  }
  static inline void set(PointerBuilder builder, AnyPointer::Reader value) {
    AnyPointer::Builder(builder).set(value);
  }
  static inline void adopt(PointerBuilder builder, Orphan<AnyPointer>&& value) {
    builder.adopt(kj::mv(value.builder));
  }
  static inline Orphan<AnyPointer> disown(PointerBuilder builder) {
    return Orphan<AnyPointer>(builder.disown());
  }
  static inline _::PointerReader getInternalReader(const AnyPointer::Reader& reader) {
    return reader.reader;
  }
  static inline _::PointerBuilder getInternalBuilder(AnyPointer::Builder&& builder) {
    return builder.builder;
  }
};

template <>
struct PointerHelpers<AnyStruct, Kind::OTHER> {
  static inline AnyStruct::Reader get(
      PointerReader reader, const word* defaultValue = nullptr) {
    return AnyStruct::Reader(reader.getStruct(defaultValue));
  }
  static inline AnyStruct::Builder get(
      PointerBuilder builder, const word* defaultValue = nullptr) {
    // TODO(someday): Allow specifying the size somehow?
    return AnyStruct::Builder(builder.getStruct(
        _::StructSize(ZERO * WORDS, ZERO * POINTERS), defaultValue));
  }
  static inline void set(PointerBuilder builder, AnyStruct::Reader value) {
    builder.setStruct(value._reader);
  }
  static inline AnyStruct::Builder init(
      PointerBuilder builder, uint16_t dataWordCount, uint16_t pointerCount) {
    return AnyStruct::Builder(builder.initStruct(
        StructSize(bounded(dataWordCount) * WORDS,
                   bounded(pointerCount) * POINTERS)));
  }

  static void adopt(PointerBuilder builder, Orphan<AnyStruct>&& value) {
    builder.adopt(kj::mv(value.builder));
  }
  static Orphan<AnyStruct> disown(PointerBuilder builder) {
    return Orphan<AnyStruct>(builder.disown());
  }
};

template <>
struct PointerHelpers<AnyList, Kind::OTHER> {
  static inline AnyList::Reader get(
      PointerReader reader, const word* defaultValue = nullptr) {
    return AnyList::Reader(reader.getListAnySize(defaultValue));
  }
  static inline AnyList::Builder get(
      PointerBuilder builder, const word* defaultValue = nullptr) {
    return AnyList::Builder(builder.getListAnySize(defaultValue));
  }
  static inline void set(PointerBuilder builder, AnyList::Reader value) {
    builder.setList(value._reader);
  }
  static inline AnyList::Builder init(
      PointerBuilder builder, ElementSize elementSize, uint elementCount) {
    return AnyList::Builder(builder.initList(
        elementSize, bounded(elementCount) * ELEMENTS));
  }
  static inline AnyList::Builder init(
      PointerBuilder builder, uint16_t dataWordCount, uint16_t pointerCount, uint elementCount) {
    return AnyList::Builder(builder.initStructList(
        bounded(elementCount) * ELEMENTS,
        StructSize(bounded(dataWordCount) * WORDS,
                   bounded(pointerCount) * POINTERS)));
  }

  static void adopt(PointerBuilder builder, Orphan<AnyList>&& value) {
    builder.adopt(kj::mv(value.builder));
  }
  static Orphan<AnyList> disown(PointerBuilder builder) {
    return Orphan<AnyList>(builder.disown());
  }
};

template <>
struct OrphanGetImpl<AnyStruct, Kind::OTHER> {
  static inline AnyStruct::Builder apply(_::OrphanBuilder& builder) {
    return AnyStruct::Builder(builder.asStruct(_::StructSize(ZERO * WORDS, ZERO * POINTERS)));
  }
  static inline AnyStruct::Reader applyReader(const _::OrphanBuilder& builder) {
    return AnyStruct::Reader(builder.asStructReader(_::StructSize(ZERO * WORDS, ZERO * POINTERS)));
  }
  static inline void truncateListOf(_::OrphanBuilder& builder, ElementCount size) {
    builder.truncate(size, _::StructSize(ZERO * WORDS, ZERO * POINTERS));
  }
};

template <>
struct OrphanGetImpl<AnyList, Kind::OTHER> {
  static inline AnyList::Builder apply(_::OrphanBuilder& builder) {
    return AnyList::Builder(builder.asListAnySize());
  }
  static inline AnyList::Reader applyReader(const _::OrphanBuilder& builder) {
    return AnyList::Reader(builder.asListReaderAnySize());
  }
  static inline void truncateListOf(_::OrphanBuilder& builder, ElementCount size) {
    builder.truncate(size, ElementSize::POINTER);
  }
};

}  // namespace _ (private)

#if !CAPNP_LITE

template <typename T>
struct PipelineHook::FromImpl {
  static inline kj::Own<PipelineHook> apply(typename T::Pipeline&& pipeline) {
    return from(kj::mv(pipeline._typeless));
  }
};

template <>
struct PipelineHook::FromImpl<AnyPointer> {
  static inline kj::Own<PipelineHook> apply(AnyPointer::Pipeline&& pipeline) {
    return kj::mv(pipeline.hook);
  }
};

template <typename Pipeline, typename T>
inline kj::Own<PipelineHook> PipelineHook::from(Pipeline&& pipeline) {
  return FromImpl<T>::apply(kj::fwd<Pipeline>(pipeline));
}

#endif  // !CAPNP_LITE

}  // namespace capnp

#endif  // CAPNP_ANY_H_
