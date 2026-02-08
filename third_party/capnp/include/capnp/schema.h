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

#if CAPNP_LITE
#error "Reflection APIs, including this header, are not available in lite mode."
#endif

#undef CONST
// For some ridiculous reason, Windows defines CONST to const. We have an enum value called CONST
// in schema.capnp.h, so if this is defined, compilation is gonna fail. So we undef it because
// that seems strictly better than failing entirely. But this could cause trouble for people later
// on if they, say, include windows.h, then include schema.h, then include another windows API
// header that uses CONST. I suppose they may have to re-#define CONST in between, or change the
// header ordering. Sorry.
//
// Please don't file a bug report telling us to change our enum naming style. You are at least
// seven years too late.

#include <capnp/schema.capnp.h>
#include <kj/hash.h>
#include <kj/windows-sanity.h>  // work-around macro conflict with `VOID`

CAPNP_BEGIN_HEADER

namespace capnp {

class Schema;
class StructSchema;
class EnumSchema;
class InterfaceSchema;
class ConstSchema;
class ListSchema;
class Type;

template <typename T, Kind k = kind<T>()> struct SchemaType_ { typedef Schema Type; };
template <typename T> struct SchemaType_<T, Kind::PRIMITIVE> { typedef schema::Type::Which Type; };
template <typename T> struct SchemaType_<T, Kind::BLOB> { typedef schema::Type::Which Type; };
template <typename T> struct SchemaType_<T, Kind::ENUM> { typedef EnumSchema Type; };
template <typename T> struct SchemaType_<T, Kind::STRUCT> { typedef StructSchema Type; };
template <typename T> struct SchemaType_<T, Kind::INTERFACE> { typedef InterfaceSchema Type; };
template <typename T> struct SchemaType_<T, Kind::LIST> { typedef ListSchema Type; };

template <typename T>
using SchemaType = typename SchemaType_<T>::Type;
// SchemaType<T> is the type of T's schema, e.g. StructSchema if T is a struct.

namespace _ {  // private
extern const RawSchema NULL_SCHEMA;
extern const RawSchema NULL_STRUCT_SCHEMA;
extern const RawSchema NULL_ENUM_SCHEMA;
extern const RawSchema NULL_INTERFACE_SCHEMA;
extern const RawSchema NULL_CONST_SCHEMA;
// The schema types default to these null (empty) schemas in case of error, especially when
// exceptions are disabled.
}  // namespace _ (private)

class Schema {
  // Convenience wrapper around capnp::schema::Node.

public:
  inline Schema(): raw(&_::NULL_SCHEMA.defaultBrand) {}

  template <typename T>
  static inline SchemaType<T> from() { return SchemaType<T>::template fromImpl<T>(); }
  // Get the Schema for a particular compiled-in type.

  schema::Node::Reader getProto() const;
  // Get the underlying Cap'n Proto representation of the schema node.  (Note that this accessor
  // has performance comparable to accessors of struct-typed fields on Reader classes.)

  kj::ArrayPtr<const word> asUncheckedMessage() const;
  // Get the encoded schema node content as a single message segment.  It is safe to read as an
  // unchecked message.

  Schema getDependency(uint64_t id) const CAPNP_DEPRECATED("Does not handle generics correctly.");
  // DEPRECATED: This method cannot correctly account for generic type parameter bindings that
  //   may apply to the dependency. Instead of using this method, use a method of the Schema API
  //   that corresponds to the exact kind of dependency. For example, to get a field type, use
  //   StructSchema::Field::getType().
  //
  // Gets the Schema for one of this Schema's dependencies.  For example, if this Schema is for a
  // struct, you could look up the schema for one of its fields' types.  Throws an exception if this
  // schema doesn't actually depend on the given id.
  //
  // Note that not all type IDs found in the schema node are considered "dependencies" -- only the
  // ones that are needed to implement the dynamic API are.  That includes:
  // - Field types.
  // - Group types.
  // - scopeId for group nodes, but NOT otherwise.
  // - Method parameter and return types.
  //
  // The following are NOT considered dependencies:
  // - Nested nodes.
  // - scopeId for a non-group node.
  // - Annotations.
  //
  // To obtain schemas for those, you would need a SchemaLoader.

  bool isBranded() const;
  // Returns true if this schema represents a non-default parameterization of this type.

  Schema getGeneric() const;
  // Get the version of this schema with any brands removed.

  class BrandArgumentList;
  BrandArgumentList getBrandArgumentsAtScope(uint64_t scopeId) const;
  // Gets the values bound to the brand parameters at the given scope.

  kj::Array<uint64_t> getGenericScopeIds() const;
  // Returns the type IDs of all parent scopes that have generic parameters, to which this type is
  // subject.

  StructSchema asStruct() const;
  EnumSchema asEnum() const;
  InterfaceSchema asInterface() const;
  ConstSchema asConst() const;
  // Cast the Schema to a specific type.  Throws an exception if the type doesn't match.  Use
  // getProto() to determine type, e.g. getProto().isStruct().

  inline bool operator==(const Schema& other) const { return raw == other.raw; }
  inline bool operator!=(const Schema& other) const { return raw != other.raw; }
  // Determine whether two Schemas are wrapping the exact same underlying data, by identity.  If
  // you want to check if two Schemas represent the same type (but possibly different versions of
  // it), compare their IDs instead.

  inline uint hashCode() const { return kj::hashCode(raw); }

  template <typename T>
  void requireUsableAs() const;
  // Throws an exception if a value with this Schema cannot safely be cast to a native value of
  // the given type.  This passes if either:
  // - *this == from<T>()
  // - This schema was loaded with SchemaLoader, the type ID matches typeId<T>(), and
  //   loadCompiledTypeAndDependencies<T>() was called on the SchemaLoader.

  kj::StringPtr getShortDisplayName() const;
  // Get the short version of the node's display name.

  const kj::StringPtr getUnqualifiedName() const;
  // Get the display name "nickname" of this node minus the prefix

private:
  const _::RawBrandedSchema* raw;

  inline explicit Schema(const _::RawBrandedSchema* raw): raw(raw) {
    KJ_IREQUIRE(raw->lazyInitializer == nullptr,
        "Must call ensureInitialized() on RawSchema before constructing Schema.");
  }

  template <typename T> static inline Schema fromImpl() {
    return Schema(&_::rawSchema<T>());
  }

  void requireUsableAs(const _::RawSchema* expected) const;

  uint32_t getSchemaOffset(const schema::Value::Reader& value) const;

  Type getBrandBinding(uint64_t scopeId, uint index) const;
  // Look up the binding for a brand parameter used by this Schema. Returns `AnyPointer` if the
  // parameter is not bound.
  //
  // TODO(someday): Public interface for iterating over all bindings?

  Schema getDependency(uint64_t id, uint location) const;
  // Look up schema for a particular dependency of this schema. `location` is the dependency
  // location number as defined in _::RawBrandedSchema.

  Type interpretType(schema::Type::Reader proto, uint location) const;
  // Interpret a schema::Type in the given location within the schema, compiling it into a
  // Type object.

  friend class StructSchema;
  friend class EnumSchema;
  friend class InterfaceSchema;
  friend class ConstSchema;
  friend class ListSchema;
  friend class SchemaLoader;
  friend class Type;
  friend kj::StringTree _::structString(
      _::StructReader reader, const _::RawBrandedSchema& schema);
  friend kj::String _::enumString(uint16_t value, const _::RawBrandedSchema& schema);
};

kj::StringPtr KJ_STRINGIFY(const Schema& schema);

class Schema::BrandArgumentList {
  // A list of generic parameter bindings for parameters of some particular type. Note that since
  // parameters on an outer type apply to all inner types as well, a deeply-nested type can have
  // multiple BrandArgumentLists that apply to it.
  //
  // A BrandArgumentList only represents the arguments that the client of the type specified. Since
  // new parameters can be added over time, this list may not cover all defined parameters for the
  // type. Missing parameters should be treated as AnyPointer. This class's implementation of
  // operator[] already does this for you; out-of-bounds access will safely return AnyPointer.

public:
  inline BrandArgumentList(): scopeId(0), size_(0), bindings(nullptr) {}

  inline uint size() const { return size_; }
  Type operator[](uint index) const;

  typedef _::IndexingIterator<const BrandArgumentList, Type> Iterator;
  inline Iterator begin() const { return Iterator(this, 0); }
  inline Iterator end() const { return Iterator(this, size()); }

private:
  uint64_t scopeId;
  uint size_;
  bool isUnbound;
  const _::RawBrandedSchema::Binding* bindings;

  inline BrandArgumentList(uint64_t scopeId, bool isUnbound)
      : scopeId(scopeId), size_(0), isUnbound(isUnbound), bindings(nullptr) {}
  inline BrandArgumentList(uint64_t scopeId, uint size,
                           const _::RawBrandedSchema::Binding* bindings)
      : scopeId(scopeId), size_(size), isUnbound(false), bindings(bindings) {}

  friend class Schema;
};

// -------------------------------------------------------------------

class StructSchema: public Schema {
public:
  inline StructSchema(): Schema(&_::NULL_STRUCT_SCHEMA.defaultBrand) {}

  class Field;
  class FieldList;
  class FieldSubset;

  FieldList getFields() const;
  // List top-level fields of this struct.  This list will contain top-level groups (including
  // named unions) but not the members of those groups.  The list does, however, contain the
  // members of the unnamed union, if there is one.

  FieldSubset getUnionFields() const;
  // If the field contains an unnamed union, get a list of fields in the union, ordered by
  // ordinal.  Since discriminant values are assigned sequentially by ordinal, you may index this
  // list by discriminant value.

  FieldSubset getNonUnionFields() const;
  // Get the fields of this struct which are not in an unnamed union, ordered by ordinal.

  kj::Maybe<Field> findFieldByName(kj::StringPtr name) const;
  // Find the field with the given name, or return null if there is no such field.  If the struct
  // contains an unnamed union, then this will find fields of that union in addition to fields
  // of the outer struct, since they exist in the same namespace.  It will not, however, find
  // members of groups (including named unions) -- you must first look up the group itself,
  // then dig into its type.

  Field getFieldByName(kj::StringPtr name) const;
  // Like findFieldByName() but throws an exception on failure.

  kj::Maybe<Field> getFieldByDiscriminant(uint16_t discriminant) const;
  // Finds the field whose `discriminantValue` is equal to the given value, or returns null if
  // there is no such field.  (If the schema does not represent a union or a struct containing
  // an unnamed union, then this always returns null.)

  bool isStreamResult() const;
  // Convenience method to check if this is the result type of a streaming RPC method.

  bool mayContainCapabilities() const { return raw->generic->mayContainCapabilities; }
  // Returns true if a struct of this type may transitively contain any capabilities. I.e., are
  // any of the fields an interface type, or a struct type that may in turn contain capabilities?
  //
  // This is meant for optimizations where various bookkeeping can possibly be skipped if it is
  // known in advance that there are no capabilities. Note that this may conservatively return true
  // spuriously, e.g. if it would be inconvenient to compute the correct answer. A false positive
  // should never cause incorrect behavior, just potentially hurt performance.
  //
  // It's important to keep in mind that even if a schema has no capability-typed fields today,
  // they could always be added in future versions of the schema. So, just because the schema
  // doesn't contain capabilities does NOT necessarily mean that an instance of the struct can't
  // contain capabilities. However, it is a pretty good hint that the application won't plan to
  // use such capabilities -- for example, if there are no caps in an RPC call's response type
  // according to the client's version of the schema, then the client clearly isn't going to try
  // to make any pipelined calls. The server could be operating with a new version of the schema
  // and could actually return capabilities, but for the client to make a pipelined call, the
  // client would have to know in advance that capabilities could be returned.

private:
  StructSchema(Schema base): Schema(base) {}
  template <typename T> static inline StructSchema fromImpl() {
    return StructSchema(Schema(&_::rawBrandedSchema<T>()));
  }
  friend class Schema;
  friend class Type;
};

class StructSchema::Field {
public:
  Field() = default;

  inline schema::Field::Reader getProto() const { return proto; }
  inline StructSchema getContainingStruct() const { return parent; }

  inline uint getIndex() const { return index; }
  // Get the index of this field within the containing struct or union.

  Type getType() const;
  // Get the type of this field. Note that this is preferred over getProto().getType() as this
  // method will apply generics.

  uint32_t getDefaultValueSchemaOffset() const;
  // For struct, list, and object fields, returns the offset, in words, within the first segment of
  // the struct's schema, where this field's default value pointer is located.  The schema is
  // always stored as a single-segment unchecked message, which in turn means that the default
  // value pointer itself can be treated as the root of an unchecked message -- if you know where
  // to find it, which is what this method helps you with.
  //
  // For blobs, returns the offset of the beginning of the blob's content within the first segment
  // of the struct's schema.
  //
  // This is primarily useful for code generators.  The C++ code generator, for example, embeds
  // the entire schema as a raw word array within the generated code.  Of course, to implement
  // field accessors, it needs access to those fields' default values.  Embedding separate copies
  // of those default values would be redundant since they are already included in the schema, but
  // seeking through the schema at runtime to find the default values would be ugly.  Instead,
  // the code generator can use getDefaultValueSchemaOffset() to find the offset of the default
  // value within the schema, and can simply apply that offset at runtime.
  //
  // If the above does not make sense, you probably don't need this method.

  inline bool operator==(const Field& other) const;
  inline bool operator!=(const Field& other) const { return !(*this == other); }
  inline uint hashCode() const;

private:
  StructSchema parent;
  uint index;
  schema::Field::Reader proto;

  inline Field(StructSchema parent, uint index, schema::Field::Reader proto)
      : parent(parent), index(index), proto(proto) {}

  friend class StructSchema;
};

kj::StringPtr KJ_STRINGIFY(const StructSchema::Field& field);

class StructSchema::FieldList {
public:
  FieldList() = default;  // empty list

  inline uint size() const { return list.size(); }
  inline Field operator[](uint index) const { return Field(parent, index, list[index]); }

  typedef _::IndexingIterator<const FieldList, Field> Iterator;
  inline Iterator begin() const { return Iterator(this, 0); }
  inline Iterator end() const { return Iterator(this, size()); }

private:
  StructSchema parent;
  List<schema::Field>::Reader list;

  inline FieldList(StructSchema parent, List<schema::Field>::Reader list)
      : parent(parent), list(list) {}

  friend class StructSchema;
};

class StructSchema::FieldSubset {
public:
  FieldSubset() = default;  // empty list

  inline uint size() const { return size_; }
  inline Field operator[](uint index) const {
    return Field(parent, indices[index], list[indices[index]]);
  }

  typedef _::IndexingIterator<const FieldSubset, Field> Iterator;
  inline Iterator begin() const { return Iterator(this, 0); }
  inline Iterator end() const { return Iterator(this, size()); }

private:
  StructSchema parent;
  List<schema::Field>::Reader list;
  const uint16_t* indices;
  uint size_;

  inline FieldSubset(StructSchema parent, List<schema::Field>::Reader list,
                     const uint16_t* indices, uint size)
      : parent(parent), list(list), indices(indices), size_(size) {}

  friend class StructSchema;
};

// -------------------------------------------------------------------

class EnumSchema: public Schema {
public:
  inline EnumSchema(): Schema(&_::NULL_ENUM_SCHEMA.defaultBrand) {}

  class Enumerant;
  class EnumerantList;

  EnumerantList getEnumerants() const;

  kj::Maybe<Enumerant> findEnumerantByName(kj::StringPtr name) const;

  Enumerant getEnumerantByName(kj::StringPtr name) const;
  // Like findEnumerantByName() but throws an exception on failure.

private:
  EnumSchema(Schema base): Schema(base) {}
  template <typename T> static inline EnumSchema fromImpl() {
    return EnumSchema(Schema(&_::rawBrandedSchema<T>()));
  }
  friend class Schema;
  friend class Type;
};

class EnumSchema::Enumerant {
public:
  Enumerant() = default;

  inline schema::Enumerant::Reader getProto() const { return proto; }
  inline EnumSchema getContainingEnum() const { return parent; }

  inline uint16_t getOrdinal() const { return ordinal; }
  inline uint getIndex() const { return ordinal; }

  inline bool operator==(const Enumerant& other) const;
  inline bool operator!=(const Enumerant& other) const { return !(*this == other); }
  inline uint hashCode() const;

private:
  EnumSchema parent;
  uint16_t ordinal;
  schema::Enumerant::Reader proto;

  inline Enumerant(EnumSchema parent, uint16_t ordinal, schema::Enumerant::Reader proto)
      : parent(parent), ordinal(ordinal), proto(proto) {}

  friend class EnumSchema;
};

class EnumSchema::EnumerantList {
public:
  EnumerantList() = default;  // empty list

  inline uint size() const { return list.size(); }
  inline Enumerant operator[](uint index) const { return Enumerant(parent, index, list[index]); }

  typedef _::IndexingIterator<const EnumerantList, Enumerant> Iterator;
  inline Iterator begin() const { return Iterator(this, 0); }
  inline Iterator end() const { return Iterator(this, size()); }

private:
  EnumSchema parent;
  List<schema::Enumerant>::Reader list;

  inline EnumerantList(EnumSchema parent, List<schema::Enumerant>::Reader list)
      : parent(parent), list(list) {}

  friend class EnumSchema;
};

// -------------------------------------------------------------------

class InterfaceSchema: public Schema {
public:
  inline InterfaceSchema(): Schema(&_::NULL_INTERFACE_SCHEMA.defaultBrand) {}

  class Method;
  class MethodList;

  MethodList getMethods() const;

  kj::Maybe<Method> findMethodByName(kj::StringPtr name) const;

  Method getMethodByName(kj::StringPtr name) const;
  // Like findMethodByName() but throws an exception on failure.

  class SuperclassList;

  SuperclassList getSuperclasses() const;
  // Get the immediate superclasses of this type, after applying generics.

  bool extends(InterfaceSchema other) const;
  // Returns true if `other` is a superclass of this interface (including if `other == *this`).

  kj::Maybe<InterfaceSchema> findSuperclass(uint64_t typeId) const;
  // Find the superclass of this interface with the given type ID.  Returns null if the interface
  // extends no such type.

private:
  InterfaceSchema(Schema base): Schema(base) {}
  template <typename T> static inline InterfaceSchema fromImpl() {
    return InterfaceSchema(Schema(&_::rawBrandedSchema<T>()));
  }
  friend class Schema;
  friend class Type;

  kj::Maybe<Method> findMethodByName(kj::StringPtr name, uint& counter) const;
  bool extends(InterfaceSchema other, uint& counter) const;
  kj::Maybe<InterfaceSchema> findSuperclass(uint64_t typeId, uint& counter) const;
  // We protect against malicious schemas with large or cyclic hierarchies by cutting off the
  // search when the counter reaches a threshold.
};

class InterfaceSchema::Method {
public:
  Method() = default;

  inline schema::Method::Reader getProto() const { return proto; }
  inline InterfaceSchema getContainingInterface() const { return parent; }

  inline uint16_t getOrdinal() const { return ordinal; }
  inline uint getIndex() const { return ordinal; }

  bool isStreaming() const { return getResultType().isStreamResult(); }
  // Check if this is a streaming method.

  StructSchema getParamType() const;
  StructSchema getResultType() const;
  // Get the parameter and result types, including substituting generic parameters.

  inline bool operator==(const Method& other) const;
  inline bool operator!=(const Method& other) const { return !(*this == other); }
  inline uint hashCode() const;

private:
  InterfaceSchema parent;
  uint16_t ordinal;
  schema::Method::Reader proto;

  inline Method(InterfaceSchema parent, uint16_t ordinal,
                schema::Method::Reader proto)
      : parent(parent), ordinal(ordinal), proto(proto) {}

  friend class InterfaceSchema;
};

class InterfaceSchema::MethodList {
public:
  MethodList() = default;  // empty list

  inline uint size() const { return list.size(); }
  inline Method operator[](uint index) const { return Method(parent, index, list[index]); }

  typedef _::IndexingIterator<const MethodList, Method> Iterator;
  inline Iterator begin() const { return Iterator(this, 0); }
  inline Iterator end() const { return Iterator(this, size()); }

private:
  InterfaceSchema parent;
  List<schema::Method>::Reader list;

  inline MethodList(InterfaceSchema parent, List<schema::Method>::Reader list)
      : parent(parent), list(list) {}

  friend class InterfaceSchema;
};

class InterfaceSchema::SuperclassList {
public:
  SuperclassList() = default;  // empty list

  inline uint size() const { return list.size(); }
  InterfaceSchema operator[](uint index) const;

  typedef _::IndexingIterator<const SuperclassList, InterfaceSchema> Iterator;
  inline Iterator begin() const { return Iterator(this, 0); }
  inline Iterator end() const { return Iterator(this, size()); }

private:
  InterfaceSchema parent;
  List<schema::Superclass>::Reader list;

  inline SuperclassList(InterfaceSchema parent, List<schema::Superclass>::Reader list)
      : parent(parent), list(list) {}

  friend class InterfaceSchema;
};

// -------------------------------------------------------------------

class ConstSchema: public Schema {
  // Represents a constant declaration.
  //
  // `ConstSchema` can be implicitly cast to DynamicValue to read its value.

public:
  inline ConstSchema(): Schema(&_::NULL_CONST_SCHEMA.defaultBrand) {}

  template <typename T>
  ReaderFor<T> as() const;
  // Read the constant's value.  This is a convenience method equivalent to casting the ConstSchema
  // to a DynamicValue and then calling its `as<T>()` method.  For dependency reasons, this method
  // is defined in <capnp/dynamic.h>, which you must #include explicitly.

  uint32_t getValueSchemaOffset() const;
  // Much like StructSchema::Field::getDefaultValueSchemaOffset(), if the constant has pointer
  // type, this gets the offset from the beginning of the constant's schema node to a pointer
  // representing the constant value.

  Type getType() const;

private:
  ConstSchema(Schema base): Schema(base) {}
  friend class Schema;
};

// -------------------------------------------------------------------

class Type {
public:
  struct BrandParameter {
    uint64_t scopeId;
    uint index;
  };
  struct ImplicitParameter {
    uint index;
  };

  inline Type();
  inline Type(schema::Type::Which primitive);
  inline Type(StructSchema schema);
  inline Type(EnumSchema schema);
  inline Type(InterfaceSchema schema);
  inline Type(ListSchema schema);
  inline Type(schema::Type::AnyPointer::Unconstrained::Which anyPointerKind);
  inline Type(BrandParameter param);
  inline Type(ImplicitParameter param);

  template <typename T>
  inline static Type from();
  template <typename T>
  inline static Type from(T&& value);

  inline schema::Type::Which which() const;

  StructSchema asStruct() const;
  EnumSchema asEnum() const;
  InterfaceSchema asInterface() const;
  ListSchema asList() const;
  // Each of these methods may only be called if which() returns the corresponding type.

  kj::Maybe<BrandParameter> getBrandParameter() const;
  // Only callable if which() returns ANY_POINTER. Returns null if the type is just a regular
  // AnyPointer and not a parameter.

  kj::Maybe<ImplicitParameter> getImplicitParameter() const;
  // Only callable if which() returns ANY_POINTER. Returns null if the type is just a regular
  // AnyPointer and not a parameter. "Implicit parameters" refer to type parameters on methods.

  inline schema::Type::AnyPointer::Unconstrained::Which whichAnyPointerKind() const;
  // Only callable if which() returns ANY_POINTER.

  inline bool isVoid() const;
  inline bool isBool() const;
  inline bool isInt8() const;
  inline bool isInt16() const;
  inline bool isInt32() const;
  inline bool isInt64() const;
  inline bool isUInt8() const;
  inline bool isUInt16() const;
  inline bool isUInt32() const;
  inline bool isUInt64() const;
  inline bool isFloat32() const;
  inline bool isFloat64() const;
  inline bool isText() const;
  inline bool isData() const;
  inline bool isList() const;
  inline bool isEnum() const;
  inline bool isStruct() const;
  inline bool isInterface() const;
  inline bool isAnyPointer() const;

  bool operator==(const Type& other) const;
  inline bool operator!=(const Type& other) const { return !(*this == other); }

  uint hashCode() const;

  inline Type wrapInList(uint depth = 1) const;
  // Return the Type formed by wrapping this type in List() `depth` times.

  inline Type(schema::Type::Which derived, const _::RawBrandedSchema* schema);
  // For internal use.

private:
  schema::Type::Which baseType;  // type not including applications of List()
  uint8_t listDepth;             // 0 for T, 1 for List(T), 2 for List(List(T)), ...

  bool isImplicitParam;
  // If true, this refers to an implicit method parameter. baseType must be ANY_POINTER, scopeId
  // must be zero, and paramIndex indicates the parameter index.

  union {
    uint16_t paramIndex;
    // If baseType is ANY_POINTER but this Type actually refers to a type parameter, this is the
    // index of the parameter among the parameters at its scope, and `scopeId` below is the type ID
    // of the scope where the parameter was defined.

    schema::Type::AnyPointer::Unconstrained::Which anyPointerKind;
    // If scopeId is zero and isImplicitParam is false.
  };

  union {
    const _::RawBrandedSchema* schema;  // if type is struct, enum, interface...
    uint64_t scopeId;  // if type is AnyPointer but it's actually a type parameter...
  };

  Type(schema::Type::Which baseType, uint8_t listDepth, const _::RawBrandedSchema* schema)
      : baseType(baseType), listDepth(listDepth), schema(schema) {
    KJ_IREQUIRE(baseType != schema::Type::ANY_POINTER);
  }

  void requireUsableAs(Type expected) const;

  template <typename T, Kind k>
  struct FromValueImpl;

  friend class ListSchema;  // only for requireUsableAs()
};

// -------------------------------------------------------------------

class ListSchema {
  // ListSchema is a little different because list types are not described by schema nodes.  So,
  // ListSchema doesn't subclass Schema.

public:
  ListSchema() = default;

  static ListSchema of(schema::Type::Which primitiveType);
  static ListSchema of(StructSchema elementType);
  static ListSchema of(EnumSchema elementType);
  static ListSchema of(InterfaceSchema elementType);
  static ListSchema of(ListSchema elementType);
  static ListSchema of(Type elementType);
  // Construct the schema for a list of the given type.

  static ListSchema of(schema::Type::Reader elementType, Schema context)
      CAPNP_DEPRECATED("Does not handle generics correctly.");
  // DEPRECATED: This method cannot correctly account for generic type parameter bindings that
  //   may apply to the input type. Instead of using this method, use a method of the Schema API
  //   that corresponds to the exact kind of dependency. For example, to get a field type, use
  //   StructSchema::Field::getType().
  //
  // Construct from an element type schema.  Requires a context which can handle getDependency()
  // requests for any type ID found in the schema.

  Type getElementType() const;

  inline schema::Type::Which whichElementType() const;
  // Get the element type's "which()".  ListSchema does not actually store a schema::Type::Reader
  // describing the element type, but if it did, this would be equivalent to calling
  // .getBody().which() on that type.

  StructSchema getStructElementType() const;
  EnumSchema getEnumElementType() const;
  InterfaceSchema getInterfaceElementType() const;
  ListSchema getListElementType() const;
  // Get the schema for complex element types.  Each of these throws an exception if the element
  // type is not of the requested kind.

  inline bool operator==(const ListSchema& other) const { return elementType == other.elementType; }
  inline bool operator!=(const ListSchema& other) const { return elementType != other.elementType; }

  template <typename T>
  void requireUsableAs() const;

private:
  Type elementType;

  inline explicit ListSchema(Type elementType): elementType(elementType) {}

  template <typename T>
  struct FromImpl;
  template <typename T> static inline ListSchema fromImpl() {
    return FromImpl<T>::get();
  }

  void requireUsableAs(ListSchema expected) const;

  friend class Schema;
};

// =======================================================================================
// inline implementation

template <> inline schema::Type::Which Schema::from<Void>() { return schema::Type::VOID; }
template <> inline schema::Type::Which Schema::from<bool>() { return schema::Type::BOOL; }
template <> inline schema::Type::Which Schema::from<int8_t>() { return schema::Type::INT8; }
template <> inline schema::Type::Which Schema::from<int16_t>() { return schema::Type::INT16; }
template <> inline schema::Type::Which Schema::from<int32_t>() { return schema::Type::INT32; }
template <> inline schema::Type::Which Schema::from<int64_t>() { return schema::Type::INT64; }
template <> inline schema::Type::Which Schema::from<uint8_t>() { return schema::Type::UINT8; }
template <> inline schema::Type::Which Schema::from<uint16_t>() { return schema::Type::UINT16; }
template <> inline schema::Type::Which Schema::from<uint32_t>() { return schema::Type::UINT32; }
template <> inline schema::Type::Which Schema::from<uint64_t>() { return schema::Type::UINT64; }
template <> inline schema::Type::Which Schema::from<float>() { return schema::Type::FLOAT32; }
template <> inline schema::Type::Which Schema::from<double>() { return schema::Type::FLOAT64; }
template <> inline schema::Type::Which Schema::from<Text>() { return schema::Type::TEXT; }
template <> inline schema::Type::Which Schema::from<Data>() { return schema::Type::DATA; }

inline Schema Schema::getDependency(uint64_t id) const {
  return getDependency(id, 0);
}

inline bool Schema::isBranded() const {
  return raw != &raw->generic->defaultBrand;
}

inline Schema Schema::getGeneric() const {
  return Schema(&raw->generic->defaultBrand);
}

template <typename T>
inline void Schema::requireUsableAs() const {
  requireUsableAs(&_::rawSchema<T>());
}

inline bool StructSchema::Field::operator==(const Field& other) const {
  return parent == other.parent && index == other.index;
}
inline bool EnumSchema::Enumerant::operator==(const Enumerant& other) const {
  return parent == other.parent && ordinal == other.ordinal;
}
inline bool InterfaceSchema::Method::operator==(const Method& other) const {
  return parent == other.parent && ordinal == other.ordinal;
}

inline uint StructSchema::Field::hashCode() const {
  return kj::hashCode(parent, index);
}
inline uint EnumSchema::Enumerant::hashCode() const {
  return kj::hashCode(parent, ordinal);
}
inline uint InterfaceSchema::Method::hashCode() const {
  return kj::hashCode(parent, ordinal);
}

inline ListSchema ListSchema::of(StructSchema elementType) {
  return ListSchema(Type(elementType));
}
inline ListSchema ListSchema::of(EnumSchema elementType) {
  return ListSchema(Type(elementType));
}
inline ListSchema ListSchema::of(InterfaceSchema elementType) {
  return ListSchema(Type(elementType));
}
inline ListSchema ListSchema::of(ListSchema elementType) {
  return ListSchema(Type(elementType));
}
inline ListSchema ListSchema::of(Type elementType) {
  return ListSchema(elementType);
}

inline Type ListSchema::getElementType() const {
  return elementType;
}

inline schema::Type::Which ListSchema::whichElementType() const {
  return elementType.which();
}

inline StructSchema ListSchema::getStructElementType() const {
  return elementType.asStruct();
}

inline EnumSchema ListSchema::getEnumElementType() const {
  return elementType.asEnum();
}

inline InterfaceSchema ListSchema::getInterfaceElementType() const {
  return elementType.asInterface();
}

inline ListSchema ListSchema::getListElementType() const {
  return elementType.asList();
}

template <typename T>
inline void ListSchema::requireUsableAs() const {
  static_assert(kind<T>() == Kind::LIST,
                "ListSchema::requireUsableAs<T>() requires T is a list type.");
  requireUsableAs(Schema::from<T>());
}

inline void ListSchema::requireUsableAs(ListSchema expected) const {
  elementType.requireUsableAs(expected.elementType);
}

template <typename T>
struct ListSchema::FromImpl<List<T>> {
  static inline ListSchema get() { return of(Schema::from<T>()); }
};

inline Type::Type(): baseType(schema::Type::VOID), listDepth(0), schema(nullptr) {}
inline Type::Type(schema::Type::Which primitive)
    : baseType(primitive), listDepth(0), isImplicitParam(false) {
  KJ_IREQUIRE(primitive != schema::Type::STRUCT &&
              primitive != schema::Type::ENUM &&
              primitive != schema::Type::INTERFACE &&
              primitive != schema::Type::LIST);
  if (primitive == schema::Type::ANY_POINTER) {
    scopeId = 0;
    anyPointerKind = schema::Type::AnyPointer::Unconstrained::ANY_KIND;
  } else {
    schema = nullptr;
  }
}
inline Type::Type(schema::Type::Which derived, const _::RawBrandedSchema* schema)
    : baseType(derived), listDepth(0), isImplicitParam(false), schema(schema) {
  KJ_IREQUIRE(derived == schema::Type::STRUCT ||
              derived == schema::Type::ENUM ||
              derived == schema::Type::INTERFACE);
}

inline Type::Type(StructSchema schema)
    : baseType(schema::Type::STRUCT), listDepth(0), schema(schema.raw) {}
inline Type::Type(EnumSchema schema)
    : baseType(schema::Type::ENUM), listDepth(0), schema(schema.raw) {}
inline Type::Type(InterfaceSchema schema)
    : baseType(schema::Type::INTERFACE), listDepth(0), schema(schema.raw) {}
inline Type::Type(ListSchema schema)
    : Type(schema.getElementType()) { ++listDepth; }
inline Type::Type(schema::Type::AnyPointer::Unconstrained::Which anyPointerKind)
    : baseType(schema::Type::ANY_POINTER), listDepth(0), isImplicitParam(false),
      anyPointerKind(anyPointerKind), scopeId(0) {}
inline Type::Type(BrandParameter param)
    : baseType(schema::Type::ANY_POINTER), listDepth(0), isImplicitParam(false),
      paramIndex(param.index), scopeId(param.scopeId) {}
inline Type::Type(ImplicitParameter param)
    : baseType(schema::Type::ANY_POINTER), listDepth(0), isImplicitParam(true),
      paramIndex(param.index), scopeId(0) {}

inline schema::Type::Which Type::which() const {
  return listDepth > 0 ? schema::Type::LIST : baseType;
}

inline schema::Type::AnyPointer::Unconstrained::Which Type::whichAnyPointerKind() const {
  KJ_IREQUIRE(baseType == schema::Type::ANY_POINTER);
  return !isImplicitParam && scopeId == 0 ? anyPointerKind
      : schema::Type::AnyPointer::Unconstrained::ANY_KIND;
}

template <typename T>
inline Type Type::from() { return Type(Schema::from<T>()); }

template <typename T, Kind k>
struct Type::FromValueImpl {
  template <typename U>
  static inline Type type(U&& value) {
    return Type::from<T>();
  }
};

template <typename T>
struct Type::FromValueImpl<T, Kind::OTHER> {
  template <typename U>
  static inline Type type(U&& value) {
    // All dynamic types have getSchema().
    return value.getSchema();
  }
};

template <typename T>
inline Type Type::from(T&& value) {
  typedef FromAny<kj::Decay<T>> Base;
  return Type::FromValueImpl<Base, kind<Base>()>::type(kj::fwd<T>(value));
}

inline bool Type::isVoid   () const { return baseType == schema::Type::VOID     && listDepth == 0; }
inline bool Type::isBool   () const { return baseType == schema::Type::BOOL     && listDepth == 0; }
inline bool Type::isInt8   () const { return baseType == schema::Type::INT8     && listDepth == 0; }
inline bool Type::isInt16  () const { return baseType == schema::Type::INT16    && listDepth == 0; }
inline bool Type::isInt32  () const { return baseType == schema::Type::INT32    && listDepth == 0; }
inline bool Type::isInt64  () const { return baseType == schema::Type::INT64    && listDepth == 0; }
inline bool Type::isUInt8  () const { return baseType == schema::Type::UINT8    && listDepth == 0; }
inline bool Type::isUInt16 () const { return baseType == schema::Type::UINT16   && listDepth == 0; }
inline bool Type::isUInt32 () const { return baseType == schema::Type::UINT32   && listDepth == 0; }
inline bool Type::isUInt64 () const { return baseType == schema::Type::UINT64   && listDepth == 0; }
inline bool Type::isFloat32() const { return baseType == schema::Type::FLOAT32  && listDepth == 0; }
inline bool Type::isFloat64() const { return baseType == schema::Type::FLOAT64  && listDepth == 0; }
inline bool Type::isText   () const { return baseType == schema::Type::TEXT     && listDepth == 0; }
inline bool Type::isData   () const { return baseType == schema::Type::DATA     && listDepth == 0; }
inline bool Type::isList   () const { return listDepth > 0; }
inline bool Type::isEnum   () const { return baseType == schema::Type::ENUM     && listDepth == 0; }
inline bool Type::isStruct () const { return baseType == schema::Type::STRUCT   && listDepth == 0; }
inline bool Type::isInterface() const {
  return baseType == schema::Type::INTERFACE && listDepth == 0;
}
inline bool Type::isAnyPointer() const {
  return baseType == schema::Type::ANY_POINTER && listDepth == 0;
}

inline Type Type::wrapInList(uint depth) const {
  Type result = *this;
  result.listDepth += depth;
  return result;
}

}  // namespace capnp

CAPNP_END_HEADER
