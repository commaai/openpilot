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

// This file is included from all generated headers.

#ifndef CAPNP_GENERATED_HEADER_SUPPORT_H_
#define CAPNP_GENERATED_HEADER_SUPPORT_H_

#if defined(__GNUC__) && !defined(CAPNP_HEADER_WARNINGS)
#pragma GCC system_header
#endif

#include "raw-schema.h"
#include "layout.h"
#include "list.h"
#include "orphan.h"
#include "pointer-helpers.h"
#include "any.h"
#include <kj/string.h>
#include <kj/string-tree.h>

namespace capnp {

class MessageBuilder;  // So that it can be declared a friend.

template <typename T, Kind k = CAPNP_KIND(T)>
struct ToDynamic_;   // Defined in dynamic.h, needs to be declared as everyone's friend.

struct DynamicStruct;  // So that it can be declared a friend.

struct Capability;  // To declare brandBindingFor<Capability>()

namespace _ {  // private

#if !CAPNP_LITE

template <typename T, typename CapnpPrivate = typename T::_capnpPrivate, bool = false>
inline const RawSchema& rawSchema() {
  return *CapnpPrivate::schema;
}
template <typename T, uint64_t id = schemas::EnumInfo<T>::typeId>
inline const RawSchema& rawSchema() {
  return *schemas::EnumInfo<T>::schema;
}

template <typename T, typename CapnpPrivate = typename T::_capnpPrivate>
inline const RawBrandedSchema& rawBrandedSchema() {
  return *CapnpPrivate::brand();
}
template <typename T, uint64_t id = schemas::EnumInfo<T>::typeId>
inline const RawBrandedSchema& rawBrandedSchema() {
  return schemas::EnumInfo<T>::schema->defaultBrand;
}

template <typename TypeTag, typename... Params>
struct ChooseBrand;
// If all of `Params` are `AnyPointer`, return the type's default brand. Otherwise, return a
// specific brand instance. TypeTag is the _capnpPrivate struct for the type in question.

template <typename TypeTag>
struct ChooseBrand<TypeTag> {
  // All params were AnyPointer. No specific brand needed.
  static constexpr _::RawBrandedSchema const* brand() { return &TypeTag::schema->defaultBrand; }
};

template <typename TypeTag, typename... Rest>
struct ChooseBrand<TypeTag, AnyPointer, Rest...>: public ChooseBrand<TypeTag, Rest...> {};
// The first parameter is AnyPointer, so recurse to check the rest.

template <typename TypeTag, typename First, typename... Rest>
struct ChooseBrand<TypeTag, First, Rest...> {
  // At least one parameter is not AnyPointer, so use the specificBrand constant.
  static constexpr _::RawBrandedSchema const* brand() { return &TypeTag::specificBrand; }
};

template <typename T, Kind k = kind<T>()>
struct BrandBindingFor_;

#define HANDLE_TYPE(Type, which) \
  template <> \
  struct BrandBindingFor_<Type, Kind::PRIMITIVE> { \
    static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) { \
      return { which, listDepth, nullptr }; \
    } \
  }
HANDLE_TYPE(Void, 0);
HANDLE_TYPE(bool, 1);
HANDLE_TYPE(int8_t, 2);
HANDLE_TYPE(int16_t, 3);
HANDLE_TYPE(int32_t, 4);
HANDLE_TYPE(int64_t, 5);
HANDLE_TYPE(uint8_t, 6);
HANDLE_TYPE(uint16_t, 7);
HANDLE_TYPE(uint32_t, 8);
HANDLE_TYPE(uint64_t, 9);
HANDLE_TYPE(float, 10);
HANDLE_TYPE(double, 11);
#undef HANDLE_TYPE

template <>
struct BrandBindingFor_<Text, Kind::BLOB> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 12, listDepth, nullptr };
  }
};

template <>
struct BrandBindingFor_<Data, Kind::BLOB> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 13, listDepth, nullptr };
  }
};

template <typename T>
struct BrandBindingFor_<List<T>, Kind::LIST> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return BrandBindingFor_<T>::get(listDepth + 1);
  }
};

template <typename T>
struct BrandBindingFor_<T, Kind::ENUM> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 15, listDepth, nullptr };
  }
};

template <typename T>
struct BrandBindingFor_<T, Kind::STRUCT> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 16, listDepth, T::_capnpPrivate::brand() };
  }
};

template <typename T>
struct BrandBindingFor_<T, Kind::INTERFACE> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 17, listDepth, T::_capnpPrivate::brand() };
  }
};

template <>
struct BrandBindingFor_<AnyPointer, Kind::OTHER> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 18, listDepth, 0, 0 };
  }
};

template <>
struct BrandBindingFor_<AnyStruct, Kind::OTHER> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 18, listDepth, 0, 1 };
  }
};

template <>
struct BrandBindingFor_<AnyList, Kind::OTHER> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 18, listDepth, 0, 2 };
  }
};

template <>
struct BrandBindingFor_<Capability, Kind::OTHER> {
  static constexpr RawBrandedSchema::Binding get(uint16_t listDepth) {
    return { 18, listDepth, 0, 3 };
  }
};

template <typename T>
constexpr RawBrandedSchema::Binding brandBindingFor() {
  return BrandBindingFor_<T>::get(0);
}

kj::StringTree structString(StructReader reader, const RawBrandedSchema& schema);
kj::String enumString(uint16_t value, const RawBrandedSchema& schema);
// Declared here so that we can declare inline stringify methods on generated types.
// Defined in stringify.c++, which depends on dynamic.c++, which is allowed not to be linked in.

template <typename T>
inline kj::StringTree structString(StructReader reader) {
  return structString(reader, rawBrandedSchema<T>());
}
template <typename T>
inline kj::String enumString(T value) {
  return enumString(static_cast<uint16_t>(value), rawBrandedSchema<T>());
}

#endif  // !CAPNP_LITE

// TODO(cleanup):  Unify ConstStruct and ConstList.
template <typename T>
class ConstStruct {
public:
  ConstStruct() = delete;
  KJ_DISALLOW_COPY(ConstStruct);
  inline explicit constexpr ConstStruct(const word* ptr): ptr(ptr) {}

  inline typename T::Reader get() const {
    return AnyPointer::Reader(PointerReader::getRootUnchecked(ptr)).getAs<T>();
  }

  inline operator typename T::Reader() const { return get(); }
  inline typename T::Reader operator*() const { return get(); }
  inline TemporaryPointer<typename T::Reader> operator->() const { return get(); }

private:
  const word* ptr;
};

template <typename T>
class ConstList {
public:
  ConstList() = delete;
  KJ_DISALLOW_COPY(ConstList);
  inline explicit constexpr ConstList(const word* ptr): ptr(ptr) {}

  inline typename List<T>::Reader get() const {
    return AnyPointer::Reader(PointerReader::getRootUnchecked(ptr)).getAs<List<T>>();
  }

  inline operator typename List<T>::Reader() const { return get(); }
  inline typename List<T>::Reader operator*() const { return get(); }
  inline TemporaryPointer<typename List<T>::Reader> operator->() const { return get(); }

private:
  const word* ptr;
};

template <size_t size>
class ConstText {
public:
  ConstText() = delete;
  KJ_DISALLOW_COPY(ConstText);
  inline explicit constexpr ConstText(const word* ptr): ptr(ptr) {}

  inline Text::Reader get() const {
    return Text::Reader(reinterpret_cast<const char*>(ptr), size);
  }

  inline operator Text::Reader() const { return get(); }
  inline Text::Reader operator*() const { return get(); }
  inline TemporaryPointer<Text::Reader> operator->() const { return get(); }

  inline kj::StringPtr toString() const {
    return get();
  }

private:
  const word* ptr;
};

template <size_t size>
inline kj::StringPtr KJ_STRINGIFY(const ConstText<size>& s) {
  return s.get();
}

template <size_t size>
class ConstData {
public:
  ConstData() = delete;
  KJ_DISALLOW_COPY(ConstData);
  inline explicit constexpr ConstData(const word* ptr): ptr(ptr) {}

  inline Data::Reader get() const {
    return Data::Reader(reinterpret_cast<const byte*>(ptr), size);
  }

  inline operator Data::Reader() const { return get(); }
  inline Data::Reader operator*() const { return get(); }
  inline TemporaryPointer<Data::Reader> operator->() const { return get(); }

private:
  const word* ptr;
};

template <size_t size>
inline auto KJ_STRINGIFY(const ConstData<size>& s) -> decltype(kj::toCharSequence(s.get())) {
  return kj::toCharSequence(s.get());
}

}  // namespace _ (private)

template <typename T, typename CapnpPrivate = typename T::_capnpPrivate>
inline constexpr uint64_t typeId() { return CapnpPrivate::typeId; }
template <typename T, uint64_t id = schemas::EnumInfo<T>::typeId>
inline constexpr uint64_t typeId() { return id; }
// typeId<MyType>() returns the type ID as defined in the schema.  Works with structs, enums, and
// interfaces.

template <typename T>
inline constexpr uint sizeInWords() {
  // Return the size, in words, of a Struct type, if allocated free-standing (not in a list).
  // May be useful for pre-computing space needed in order to precisely allocate messages.

  return unbound((upgradeBound<uint>(_::structSize<T>().data) +
      _::structSize<T>().pointers * WORDS_PER_POINTER) / WORDS);
}

}  // namespace capnp

#if _MSC_VER
// MSVC doesn't understand floating-point constexpr yet.
//
// TODO(msvc): Remove this hack when MSVC is fixed.
#define CAPNP_NON_INT_CONSTEXPR_DECL_INIT(value)
#define CAPNP_NON_INT_CONSTEXPR_DEF_INIT(value) = value
#else
#define CAPNP_NON_INT_CONSTEXPR_DECL_INIT(value) = value
#define CAPNP_NON_INT_CONSTEXPR_DEF_INIT(value)
#endif

#if _MSC_VER
// TODO(msvc): A little hack to allow MSVC to use C++14 return type deduction in cases where the
// explicit type exposes bugs in the compiler.
#define CAPNP_AUTO_IF_MSVC(...) auto
#else
#define CAPNP_AUTO_IF_MSVC(...) __VA_ARGS__
#endif

#if CAPNP_LITE

#define CAPNP_DECLARE_SCHEMA(id) \
    extern ::capnp::word const* const bp_##id

#define CAPNP_DECLARE_ENUM(type, id) \
    inline ::kj::String KJ_STRINGIFY(type##_##id value) { \
      return ::kj::str(static_cast<uint16_t>(value)); \
    } \
    template <> struct EnumInfo<type##_##id> { \
      struct IsEnum; \
      static constexpr uint64_t typeId = 0x##id; \
      static inline ::capnp::word const* encodedSchema() { return bp_##id; } \
    }

#if _MSC_VER
// TODO(msvc): MSVC dosen't expect constexprs to have definitions.
#define CAPNP_DEFINE_ENUM(type, id)
#else
#define CAPNP_DEFINE_ENUM(type, id) \
    constexpr uint64_t EnumInfo<type>::typeId
#endif

#define CAPNP_DECLARE_STRUCT_HEADER(id, dataWordSize_, pointerCount_) \
      struct IsStruct; \
      static constexpr uint64_t typeId = 0x##id; \
      static constexpr uint16_t dataWordSize = dataWordSize_; \
      static constexpr uint16_t pointerCount = pointerCount_; \
      static inline ::capnp::word const* encodedSchema() { return ::capnp::schemas::bp_##id; }

#else  // CAPNP_LITE

#define CAPNP_DECLARE_SCHEMA(id) \
    extern ::capnp::word const* const bp_##id; \
    extern const ::capnp::_::RawSchema s_##id

#define CAPNP_DECLARE_ENUM(type, id) \
    inline ::kj::String KJ_STRINGIFY(type##_##id value) { \
      return ::capnp::_::enumString(value); \
    } \
    template <> struct EnumInfo<type##_##id> { \
      struct IsEnum; \
      static constexpr uint64_t typeId = 0x##id; \
      static inline ::capnp::word const* encodedSchema() { return bp_##id; } \
      static constexpr ::capnp::_::RawSchema const* schema = &s_##id; \
    }
#define CAPNP_DEFINE_ENUM(type, id) \
    constexpr uint64_t EnumInfo<type>::typeId; \
    constexpr ::capnp::_::RawSchema const* EnumInfo<type>::schema

#define CAPNP_DECLARE_STRUCT_HEADER(id, dataWordSize_, pointerCount_) \
      struct IsStruct; \
      static constexpr uint64_t typeId = 0x##id; \
      static constexpr ::capnp::Kind kind = ::capnp::Kind::STRUCT; \
      static constexpr uint16_t dataWordSize = dataWordSize_; \
      static constexpr uint16_t pointerCount = pointerCount_; \
      static inline ::capnp::word const* encodedSchema() { return ::capnp::schemas::bp_##id; } \
      static constexpr ::capnp::_::RawSchema const* schema = &::capnp::schemas::s_##id;

#define CAPNP_DECLARE_INTERFACE_HEADER(id) \
      struct IsInterface; \
      static constexpr uint64_t typeId = 0x##id; \
      static constexpr ::capnp::Kind kind = ::capnp::Kind::INTERFACE; \
      static inline ::capnp::word const* encodedSchema() { return ::capnp::schemas::bp_##id; } \
      static constexpr ::capnp::_::RawSchema const* schema = &::capnp::schemas::s_##id;

#endif  // CAPNP_LITE, else

#endif  // CAPNP_GENERATED_HEADER_SUPPORT_H_
