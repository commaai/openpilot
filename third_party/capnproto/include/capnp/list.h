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

#include "layout.h"
#include "orphan.h"
#include <initializer_list>

CAPNP_BEGIN_HEADER

namespace capnp {
namespace _ {  // private

template <typename T>
class TemporaryPointer {
  // This class is a little hack which lets us define operator->() in cases where it needs to
  // return a pointer to a temporary value.  We instead construct a TemporaryPointer and return that
  // (by value).  The compiler then invokes operator->() on the TemporaryPointer, which itself is
  // able to return a real pointer to its member.

public:
  TemporaryPointer(T&& value): value(kj::mv(value)) {}
  TemporaryPointer(const T& value): value(value) {}

  inline T* operator->() { return &value; }
private:
  T value;
};

// By default this isn't compatible with STL algorithms. To add STL support either define
// KJ_STD_COMPAT at the top of your compilation unit or include capnp/compat/std-iterator.h.
template <typename Container, typename Element>
class IndexingIterator {
public:
  IndexingIterator() = default;

  inline Element operator*() const { return (*container)[index]; }
  inline TemporaryPointer<Element> operator->() const {
    return TemporaryPointer<Element>((*container)[index]);
  }
  inline Element operator[]( int off) const { return (*container)[index]; }
  inline Element operator[](uint off) const { return (*container)[index]; }

  inline IndexingIterator& operator++() { ++index; return *this; }
  inline IndexingIterator operator++(int) { IndexingIterator other = *this; ++index; return other; }
  inline IndexingIterator& operator--() { --index; return *this; }
  inline IndexingIterator operator--(int) { IndexingIterator other = *this; --index; return other; }

  inline IndexingIterator operator+(uint amount) const { return IndexingIterator(container, index + amount); }
  inline IndexingIterator operator-(uint amount) const { return IndexingIterator(container, index - amount); }
  inline IndexingIterator operator+( int amount) const { return IndexingIterator(container, index + amount); }
  inline IndexingIterator operator-( int amount) const { return IndexingIterator(container, index - amount); }

  inline int operator-(const IndexingIterator& other) const { return index - other.index; }

  inline IndexingIterator& operator+=(uint amount) { index += amount; return *this; }
  inline IndexingIterator& operator-=(uint amount) { index -= amount; return *this; }
  inline IndexingIterator& operator+=( int amount) { index += amount; return *this; }
  inline IndexingIterator& operator-=( int amount) { index -= amount; return *this; }

  // STL says comparing iterators of different containers is not allowed, so we only compare
  // indices here.
  inline bool operator==(const IndexingIterator& other) const { return index == other.index; }
  inline bool operator!=(const IndexingIterator& other) const { return index != other.index; }
  inline bool operator<=(const IndexingIterator& other) const { return index <= other.index; }
  inline bool operator>=(const IndexingIterator& other) const { return index >= other.index; }
  inline bool operator< (const IndexingIterator& other) const { return index <  other.index; }
  inline bool operator> (const IndexingIterator& other) const { return index >  other.index; }

private:
  Container* container;
  uint index;

  friend Container;
  inline IndexingIterator(Container* container, uint index)
      : container(container), index(index) {}
};

}  // namespace _ (private)

template <typename T>
struct List<T, Kind::PRIMITIVE> {
  // List of primitives.

  List() = delete;

  class Reader {
  public:
    typedef List<T> Reads;

    inline Reader(): reader(_::elementSizeForType<T>()) {}
    inline explicit Reader(_::ListReader reader): reader(reader) {}

    inline uint size() const { return unbound(reader.size() / ELEMENTS); }
    inline T operator[](uint index) const {
      KJ_IREQUIRE(index < size());
      return reader.template getDataElement<T>(bounded(index) * ELEMENTS);
    }

    typedef _::IndexingIterator<const Reader, T> Iterator;
    inline Iterator begin() const { return Iterator(this, 0); }
    inline Iterator end() const { return Iterator(this, size()); }

    inline MessageSize totalSize() const {
      return reader.totalSize().asPublic();
    }

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
    typedef List<T> Builds;

    inline Builder(): builder(_::elementSizeForType<T>()) {}
    inline Builder(decltype(nullptr)): Builder() {}
    inline explicit Builder(_::ListBuilder builder): builder(builder) {}

    inline operator Reader() const { return Reader(builder.asReader()); }
    inline Reader asReader() const { return Reader(builder.asReader()); }

    inline uint size() const { return unbound(builder.size() / ELEMENTS); }
    inline T operator[](uint index) {
      KJ_IREQUIRE(index < size());
      return builder.template getDataElement<T>(bounded(index) * ELEMENTS);
    }
    inline void set(uint index, T value) {
      // Alas, it is not possible to make operator[] return a reference to which you can assign,
      // since the encoded representation does not necessarily match the compiler's representation
      // of the type.  We can't even return a clever class that implements operator T() and
      // operator=() because it will lead to surprising behavior when using type inference (e.g.
      // calling a template function with inferred argument types, or using "auto" or "decltype").

      builder.template setDataElement<T>(bounded(index) * ELEMENTS, value);
    }

    typedef _::IndexingIterator<Builder, T> Iterator;
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

  class Pipeline {};

private:
  inline static _::ListBuilder initPointer(_::PointerBuilder builder, uint size) {
    return builder.initList(_::elementSizeForType<T>(), bounded(size) * ELEMENTS);
  }
  inline static _::ListBuilder getFromPointer(_::PointerBuilder builder, const word* defaultValue) {
    return builder.getList(_::elementSizeForType<T>(), defaultValue);
  }
  inline static _::ListReader getFromPointer(
      const _::PointerReader& reader, const word* defaultValue) {
    return reader.getList(_::elementSizeForType<T>(), defaultValue);
  }

  template <typename U, Kind k>
  friend struct List;
  template <typename U, Kind K>
  friend struct _::PointerHelpers;
};

template <typename T>
struct List<T, Kind::ENUM>: public List<T, Kind::PRIMITIVE> {};

template <typename T>
struct List<T, Kind::STRUCT> {
  // List of structs.

  List() = delete;

  class Reader {
  public:
    typedef List<T> Reads;

    inline Reader(): reader(ElementSize::INLINE_COMPOSITE) {}
    inline explicit Reader(_::ListReader reader): reader(reader) {}

    inline uint size() const { return unbound(reader.size() / ELEMENTS); }
    inline typename T::Reader operator[](uint index) const {
      KJ_IREQUIRE(index < size());
      return typename T::Reader(reader.getStructElement(bounded(index) * ELEMENTS));
    }

    typedef _::IndexingIterator<const Reader, typename T::Reader> Iterator;
    inline Iterator begin() const { return Iterator(this, 0); }
    inline Iterator end() const { return Iterator(this, size()); }

    inline MessageSize totalSize() const {
      return reader.totalSize().asPublic();
    }

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
    typedef List<T> Builds;

    inline Builder(): builder(ElementSize::INLINE_COMPOSITE) {}
    inline Builder(decltype(nullptr)): Builder() {}
    inline explicit Builder(_::ListBuilder builder): builder(builder) {}

    inline operator Reader() const { return Reader(builder.asReader()); }
    inline Reader asReader() const { return Reader(builder.asReader()); }

    inline uint size() const { return unbound(builder.size() / ELEMENTS); }
    inline typename T::Builder operator[](uint index) {
      KJ_IREQUIRE(index < size());
      return typename T::Builder(builder.getStructElement(bounded(index) * ELEMENTS));
    }

    inline void adoptWithCaveats(uint index, Orphan<T>&& orphan) {
      // Mostly behaves like you'd expect `adopt` to behave, but with two caveats originating from
      // the fact that structs in a struct list are allocated inline rather than by pointer:
      // * This actually performs a shallow copy, effectively adopting each of the orphan's
      //   children rather than adopting the orphan itself.  The orphan ends up being discarded,
      //   possibly wasting space in the message object.
      // * If the orphan is larger than the target struct -- say, because the orphan was built
      //   using a newer version of the schema that has additional fields -- it will be truncated,
      //   losing data.

      KJ_IREQUIRE(index < size());

      // We pass a zero-valued StructSize to asStruct() because we do not want the struct to be
      // expanded under any circumstances.  We're just going to throw it away anyway, and
      // transferContentFrom() already carefully compares the struct sizes before transferring.
      builder.getStructElement(bounded(index) * ELEMENTS).transferContentFrom(
          orphan.builder.asStruct(_::StructSize(ZERO * WORDS, ZERO * POINTERS)));
    }
    inline void setWithCaveats(uint index, const typename T::Reader& reader) {
      // Mostly behaves like you'd expect `set` to behave, but with a caveat originating from
      // the fact that structs in a struct list are allocated inline rather than by pointer:
      // If the source struct is larger than the target struct -- say, because the source was built
      // using a newer version of the schema that has additional fields -- it will be truncated,
      // losing data.
      //
      // Note: If you are trying to concatenate some lists, use Orphanage::newOrphanConcat() to
      //   do it without losing any data in case the source lists come from a newer version of the
      //   protocol. (Plus, it's easier to use anyhow.)

      KJ_IREQUIRE(index < size());
      builder.getStructElement(bounded(index) * ELEMENTS).copyContentFrom(reader._reader);
    }

    // There are no init(), set(), adopt(), or disown() methods for lists of structs because the
    // elements of the list are inlined and are initialized when the list is initialized.  This
    // means that init() would be redundant, and set() would risk data loss if the input struct
    // were from a newer version of the protocol.

    typedef _::IndexingIterator<Builder, typename T::Builder> Iterator;
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

  class Pipeline {};

private:
  inline static _::ListBuilder initPointer(_::PointerBuilder builder, uint size) {
    return builder.initStructList(bounded(size) * ELEMENTS, _::structSize<T>());
  }
  inline static _::ListBuilder getFromPointer(_::PointerBuilder builder, const word* defaultValue) {
    return builder.getStructList(_::structSize<T>(), defaultValue);
  }
  inline static _::ListReader getFromPointer(
      const _::PointerReader& reader, const word* defaultValue) {
    return reader.getList(ElementSize::INLINE_COMPOSITE, defaultValue);
  }

  template <typename U, Kind k>
  friend struct List;
  template <typename U, Kind K>
  friend struct _::PointerHelpers;
};

template <typename T>
struct List<List<T>, Kind::LIST> {
  // List of lists.

  List() = delete;

  class Reader {
  public:
    typedef List<List<T>> Reads;

    inline Reader(): reader(ElementSize::POINTER) {}
    inline explicit Reader(_::ListReader reader): reader(reader) {}

    inline uint size() const { return unbound(reader.size() / ELEMENTS); }
    inline typename List<T>::Reader operator[](uint index) const {
      KJ_IREQUIRE(index < size());
      return typename List<T>::Reader(_::PointerHelpers<List<T>>::get(
          reader.getPointerElement(bounded(index) * ELEMENTS)));
    }

    typedef _::IndexingIterator<const Reader, typename List<T>::Reader> Iterator;
    inline Iterator begin() const { return Iterator(this, 0); }
    inline Iterator end() const { return Iterator(this, size()); }

    inline MessageSize totalSize() const {
      return reader.totalSize().asPublic();
    }

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
    typedef List<List<T>> Builds;

    inline Builder(): builder(ElementSize::POINTER) {}
    inline Builder(decltype(nullptr)): Builder() {}
    inline explicit Builder(_::ListBuilder builder): builder(builder) {}

    inline operator Reader() const { return Reader(builder.asReader()); }
    inline Reader asReader() const { return Reader(builder.asReader()); }

    inline uint size() const { return unbound(builder.size() / ELEMENTS); }
    inline typename List<T>::Builder operator[](uint index) {
      KJ_IREQUIRE(index < size());
      return typename List<T>::Builder(_::PointerHelpers<List<T>>::get(
          builder.getPointerElement(bounded(index) * ELEMENTS)));
    }
    inline typename List<T>::Builder init(uint index, uint size) {
      KJ_IREQUIRE(index < this->size());
      return typename List<T>::Builder(_::PointerHelpers<List<T>>::init(
          builder.getPointerElement(bounded(index) * ELEMENTS), size));
    }
    inline void set(uint index, typename List<T>::Reader value) {
      KJ_IREQUIRE(index < size());
      builder.getPointerElement(bounded(index) * ELEMENTS).setList(value.reader);
    }
    void set(uint index, std::initializer_list<ReaderFor<T>> value) {
      KJ_IREQUIRE(index < size());
      auto l = init(index, value.size());
      uint i = 0;
      for (auto& element: value) {
        l.set(i++, element);
      }
    }
    inline void adopt(uint index, Orphan<List<T>>&& value) {
      KJ_IREQUIRE(index < size());
      builder.getPointerElement(bounded(index) * ELEMENTS).adopt(kj::mv(value.builder));
    }
    inline Orphan<List<T>> disown(uint index) {
      KJ_IREQUIRE(index < size());
      return Orphan<List<T>>(builder.getPointerElement(bounded(index) * ELEMENTS).disown());
    }

    typedef _::IndexingIterator<Builder, typename List<T>::Builder> Iterator;
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

  class Pipeline {};

private:
  inline static _::ListBuilder initPointer(_::PointerBuilder builder, uint size) {
    return builder.initList(ElementSize::POINTER, bounded(size) * ELEMENTS);
  }
  inline static _::ListBuilder getFromPointer(_::PointerBuilder builder, const word* defaultValue) {
    return builder.getList(ElementSize::POINTER, defaultValue);
  }
  inline static _::ListReader getFromPointer(
      const _::PointerReader& reader, const word* defaultValue) {
    return reader.getList(ElementSize::POINTER, defaultValue);
  }

  template <typename U, Kind k>
  friend struct List;
  template <typename U, Kind K>
  friend struct _::PointerHelpers;
};

template <typename T>
struct List<T, Kind::BLOB> {
  List() = delete;

  class Reader {
  public:
    typedef List<T> Reads;

    inline Reader(): reader(ElementSize::POINTER) {}
    inline explicit Reader(_::ListReader reader): reader(reader) {}

    inline uint size() const { return unbound(reader.size() / ELEMENTS); }
    inline typename T::Reader operator[](uint index) const {
      KJ_IREQUIRE(index < size());
      return reader.getPointerElement(bounded(index) * ELEMENTS)
          .template getBlob<T>(nullptr, ZERO * BYTES);
    }

    typedef _::IndexingIterator<const Reader, typename T::Reader> Iterator;
    inline Iterator begin() const { return Iterator(this, 0); }
    inline Iterator end() const { return Iterator(this, size()); }

    inline MessageSize totalSize() const {
      return reader.totalSize().asPublic();
    }

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
    typedef List<T> Builds;

    inline Builder(): builder(ElementSize::POINTER) {}
    inline Builder(decltype(nullptr)): Builder() {}
    inline explicit Builder(_::ListBuilder builder): builder(builder) {}

    inline operator Reader() const { return Reader(builder.asReader()); }
    inline Reader asReader() const { return Reader(builder.asReader()); }

    inline uint size() const { return unbound(builder.size() / ELEMENTS); }
    inline typename T::Builder operator[](uint index) {
      KJ_IREQUIRE(index < size());
      return builder.getPointerElement(bounded(index) * ELEMENTS)
          .template getBlob<T>(nullptr, ZERO * BYTES);
    }
    inline void set(uint index, typename T::Reader value) {
      KJ_IREQUIRE(index < size());
      builder.getPointerElement(bounded(index) * ELEMENTS).template setBlob<T>(value);
    }
    inline typename T::Builder init(uint index, uint size) {
      KJ_IREQUIRE(index < this->size());
      return builder.getPointerElement(bounded(index) * ELEMENTS)
          .template initBlob<T>(bounded(size) * BYTES);
    }
    inline void adopt(uint index, Orphan<T>&& value) {
      KJ_IREQUIRE(index < size());
      builder.getPointerElement(bounded(index) * ELEMENTS).adopt(kj::mv(value.builder));
    }
    inline Orphan<T> disown(uint index) {
      KJ_IREQUIRE(index < size());
      return Orphan<T>(builder.getPointerElement(bounded(index) * ELEMENTS).disown());
    }

    typedef _::IndexingIterator<Builder, typename T::Builder> Iterator;
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

  class Pipeline {};

private:
  inline static _::ListBuilder initPointer(_::PointerBuilder builder, uint size) {
    return builder.initList(ElementSize::POINTER, bounded(size) * ELEMENTS);
  }
  inline static _::ListBuilder getFromPointer(_::PointerBuilder builder, const word* defaultValue) {
    return builder.getList(ElementSize::POINTER, defaultValue);
  }
  inline static _::ListReader getFromPointer(
      const _::PointerReader& reader, const word* defaultValue) {
    return reader.getList(ElementSize::POINTER, defaultValue);
  }

  template <typename U, Kind k>
  friend struct List;
  template <typename U, Kind K>
  friend struct _::PointerHelpers;
};

}  // namespace capnp

#ifdef KJ_STD_COMPAT
#include "compat/std-iterator.h"
#endif  // KJ_STD_COMPAT

CAPNP_END_HEADER
