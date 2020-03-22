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

#ifndef CAPNP_POINTER_HELPERS_H_
#define CAPNP_POINTER_HELPERS_H_

#if defined(__GNUC__) && !defined(CAPNP_HEADER_WARNINGS)
#pragma GCC system_header
#endif

#include "layout.h"
#include "list.h"

namespace capnp {
namespace _ {  // private

// PointerHelpers is a template class that assists in wrapping/unwrapping the low-level types in
// layout.h with the high-level public API and generated types.  This way, the code generator
// and other templates do not have to specialize on each kind of pointer.

template <typename T>
struct PointerHelpers<T, Kind::STRUCT> {
  static inline typename T::Reader get(PointerReader reader, const word* defaultValue = nullptr) {
    return typename T::Reader(reader.getStruct(defaultValue));
  }
  static inline typename T::Builder get(PointerBuilder builder,
                                        const word* defaultValue = nullptr) {
    return typename T::Builder(builder.getStruct(structSize<T>(), defaultValue));
  }
  static inline void set(PointerBuilder builder, typename T::Reader value) {
    builder.setStruct(value._reader);
  }
  static inline void setCanonical(PointerBuilder builder, typename T::Reader value) {
    builder.setStruct(value._reader, true);
  }
  static inline typename T::Builder init(PointerBuilder builder) {
    return typename T::Builder(builder.initStruct(structSize<T>()));
  }
  static inline void adopt(PointerBuilder builder, Orphan<T>&& value) {
    builder.adopt(kj::mv(value.builder));
  }
  static inline Orphan<T> disown(PointerBuilder builder) {
    return Orphan<T>(builder.disown());
  }
  static inline _::StructReader getInternalReader(const typename T::Reader& reader) {
    return reader._reader;
  }
  static inline _::StructBuilder getInternalBuilder(typename T::Builder&& builder) {
    return builder._builder;
  }
};

template <typename T>
struct PointerHelpers<List<T>, Kind::LIST> {
  static inline typename List<T>::Reader get(PointerReader reader,
                                             const word* defaultValue = nullptr) {
    return typename List<T>::Reader(List<T>::getFromPointer(reader, defaultValue));
  }
  static inline typename List<T>::Builder get(PointerBuilder builder,
                                              const word* defaultValue = nullptr) {
    return typename List<T>::Builder(List<T>::getFromPointer(builder, defaultValue));
  }
  static inline void set(PointerBuilder builder, typename List<T>::Reader value) {
    builder.setList(value.reader);
  }
  static inline void setCanonical(PointerBuilder builder, typename List<T>::Reader value) {
    builder.setList(value.reader, true);
  }
  static void set(PointerBuilder builder, kj::ArrayPtr<const ReaderFor<T>> value) {
    auto l = init(builder, value.size());
    uint i = 0;
    for (auto& element: value) {
      l.set(i++, element);
    }
  }
  static inline typename List<T>::Builder init(PointerBuilder builder, uint size) {
    return typename List<T>::Builder(List<T>::initPointer(builder, size));
  }
  static inline void adopt(PointerBuilder builder, Orphan<List<T>>&& value) {
    builder.adopt(kj::mv(value.builder));
  }
  static inline Orphan<List<T>> disown(PointerBuilder builder) {
    return Orphan<List<T>>(builder.disown());
  }
  static inline _::ListReader getInternalReader(const typename List<T>::Reader& reader) {
    return reader.reader;
  }
  static inline _::ListBuilder getInternalBuilder(typename List<T>::Builder&& builder) {
    return builder.builder;
  }
};

template <typename T>
struct PointerHelpers<T, Kind::BLOB> {
  static inline typename T::Reader get(PointerReader reader,
                                       const void* defaultValue = nullptr,
                                       uint defaultBytes = 0) {
    return reader.getBlob<T>(defaultValue, bounded(defaultBytes) * BYTES);
  }
  static inline typename T::Builder get(PointerBuilder builder,
                                        const void* defaultValue = nullptr,
                                        uint defaultBytes = 0) {
    return builder.getBlob<T>(defaultValue, bounded(defaultBytes) * BYTES);
  }
  static inline void set(PointerBuilder builder, typename T::Reader value) {
    builder.setBlob<T>(value);
  }
  static inline void setCanonical(PointerBuilder builder, typename T::Reader value) {
    builder.setBlob<T>(value);
  }
  static inline typename T::Builder init(PointerBuilder builder, uint size) {
    return builder.initBlob<T>(bounded(size) * BYTES);
  }
  static inline void adopt(PointerBuilder builder, Orphan<T>&& value) {
    builder.adopt(kj::mv(value.builder));
  }
  static inline Orphan<T> disown(PointerBuilder builder) {
    return Orphan<T>(builder.disown());
  }
};

struct UncheckedMessage {
  typedef const word* Reader;
};

template <> struct Kind_<UncheckedMessage> { static constexpr Kind kind = Kind::OTHER; };

template <>
struct PointerHelpers<UncheckedMessage> {
  // Reads an AnyPointer field as an unchecked message pointer.  Requires that the containing
  // message is itself unchecked.  This hack is currently private.  It is used to locate default
  // values within encoded schemas.

  static inline const word* get(PointerReader reader) {
    return reader.getUnchecked();
  }
};

}  // namespace _ (private)
}  // namespace capnp

#endif  // CAPNP_POINTER_HELPERS_H_
