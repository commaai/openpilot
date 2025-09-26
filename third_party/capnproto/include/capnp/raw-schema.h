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

#include "common.h"  // for uint and friends

#if _MSC_VER && !defined(__clang__)
#include <atomic>
#endif

CAPNP_BEGIN_HEADER

namespace capnp {
namespace _ {  // private

struct RawSchema;

struct RawBrandedSchema {
  // Represents a combination of a schema and bindings for its generic parameters.
  //
  // Note that while we generate one `RawSchema` per type, we generate a `RawBrandedSchema` for
  // every _instance_ of a generic type -- or, at least, every instance that is actually used. For
  // generated-code types, we use template magic to initialize these.

  const RawSchema* generic;
  // Generic type which we're branding.

  struct Binding {
    uint8_t which;       // Numeric value of one of schema::Type::Which.

    bool isImplicitParameter;
    // For AnyPointer, true if it's an implicit method parameter.

    uint16_t listDepth;  // Number of times to wrap the base type in List().

    uint16_t paramIndex;
    // For AnyPointer. If it's a type parameter (scopeId is non-zero) or it's an implicit parameter
    // (isImplicitParameter is true), then this is the parameter index. Otherwise this is a numeric
    // value of one of schema::Type::AnyPointer::Unconstrained::Which.

    union {
      const RawBrandedSchema* schema;  // for struct, enum, interface
      uint64_t scopeId;                // for AnyPointer, if it's a type parameter
    };

    Binding() = default;
    inline constexpr Binding(uint8_t which, uint16_t listDepth, const RawBrandedSchema* schema)
        : which(which), isImplicitParameter(false), listDepth(listDepth), paramIndex(0),
          schema(schema) {}
    inline constexpr Binding(uint8_t which, uint16_t listDepth,
                             uint64_t scopeId, uint16_t paramIndex)
        : which(which), isImplicitParameter(false), listDepth(listDepth), paramIndex(paramIndex),
          scopeId(scopeId) {}
    inline constexpr Binding(uint8_t which, uint16_t listDepth, uint16_t implicitParamIndex)
        : which(which), isImplicitParameter(true), listDepth(listDepth),
          paramIndex(implicitParamIndex), scopeId(0) {}
  };

  struct Scope {
    uint64_t typeId;
    // Type ID whose parameters are being bound.

    const Binding* bindings;
    uint bindingCount;
    // Bindings for those parameters.

    bool isUnbound;
    // This scope is unbound, in the sense of SchemaLoader::getUnbound().
  };

  const Scope* scopes;
  // Array of enclosing scopes for which generic variables have been bound, sorted by type ID.

  struct Dependency {
    uint location;
    const RawBrandedSchema* schema;
  };

  const Dependency* dependencies;
  // Map of branded schemas for dependencies of this type, given our brand. Only dependencies that
  // are branded are included in this map; if a dependency is missing, use its `defaultBrand`.

  uint32_t scopeCount;
  uint32_t dependencyCount;

  enum class DepKind {
    // Component of a Dependency::location. Specifies what sort of dependency this is.

    INVALID,
    // Mostly defined to ensure that zero is not a valid location.

    FIELD,
    // Binding needed for a field's type. The index is the field index (NOT ordinal!).

    METHOD_PARAMS,
    // Bindings needed for a method's params type. The index is the method number.

    METHOD_RESULTS,
    // Bindings needed for a method's results type. The index is the method ordinal.

    SUPERCLASS,
    // Bindings needed for a superclass type. The index is the superclass's index in the
    // "extends" list.

    CONST_TYPE
    // Bindings needed for the type of a constant. The index is zero.
  };

  static inline uint makeDepLocation(DepKind kind, uint index) {
    // Make a number representing the location of a particular dependency within its parent
    // schema.

    return (static_cast<uint>(kind) << 24) | index;
  }

  class Initializer {
  public:
    virtual void init(const RawBrandedSchema* generic) const = 0;
  };

  const Initializer* lazyInitializer;
  // Lazy initializer, invoked by ensureInitialized().

  inline void ensureInitialized() const {
    // Lazy initialization support.  Invoke to ensure that initialization has taken place.  This
    // is required in particular when traversing the dependency list.  RawSchemas for compiled-in
    // types are always initialized; only dynamically-loaded schemas may be lazy.

#if __GNUC__ || defined(__clang__)
    const Initializer* i = __atomic_load_n(&lazyInitializer, __ATOMIC_ACQUIRE);
#elif _MSC_VER
    const Initializer* i = *static_cast<Initializer const* const volatile*>(&lazyInitializer);
    std::atomic_thread_fence(std::memory_order_acquire);
#else
#error "Platform not supported"
#endif
    if (i != nullptr) i->init(this);
  }

  inline bool isUnbound() const;
  // Checks if this schema is the result of calling SchemaLoader::getUnbound(), in which case
  // binding lookups need to be handled specially.
};

struct RawSchema {
  // The generated code defines a constant RawSchema for every compiled declaration.
  //
  // This is an internal structure which could change in the future.

  uint64_t id;

  const word* encodedNode;
  // Encoded SchemaNode, readable via readMessageUnchecked<schema::Node>(encodedNode).

  uint32_t encodedSize;
  // Size of encodedNode, in words.

  const RawSchema* const* dependencies;
  // Pointers to other types on which this one depends, sorted by ID.  The schemas in this table
  // may be uninitialized -- you must call ensureInitialized() on the one you wish to use before
  // using it.
  //
  // TODO(someday):  Make this a hashtable.

  const uint16_t* membersByName;
  // Indexes of members sorted by name.  Used to implement name lookup.
  // TODO(someday):  Make this a hashtable.

  uint32_t dependencyCount;
  uint32_t memberCount;
  // Sizes of above tables.

  const uint16_t* membersByDiscriminant;
  // List of all member indexes ordered by discriminant value.  Those which don't have a
  // discriminant value are listed at the end, in order by ordinal.

  const RawSchema* canCastTo;
  // Points to the RawSchema of a compiled-in type to which it is safe to cast any DynamicValue
  // with this schema.  This is null for all compiled-in types; it is only set by SchemaLoader on
  // dynamically-loaded types.

  class Initializer {
  public:
    virtual void init(const RawSchema* schema) const = 0;
  };

  const Initializer* lazyInitializer;
  // Lazy initializer, invoked by ensureInitialized().

  inline void ensureInitialized() const {
    // Lazy initialization support.  Invoke to ensure that initialization has taken place.  This
    // is required in particular when traversing the dependency list.  RawSchemas for compiled-in
    // types are always initialized; only dynamically-loaded schemas may be lazy.

#if __GNUC__ || defined(__clang__)
    const Initializer* i = __atomic_load_n(&lazyInitializer, __ATOMIC_ACQUIRE);
#elif _MSC_VER
    const Initializer* i = *static_cast<Initializer const* const volatile*>(&lazyInitializer);
    std::atomic_thread_fence(std::memory_order_acquire);
#else
#error "Platform not supported"
#endif
    if (i != nullptr) i->init(this);
  }

  RawBrandedSchema defaultBrand;
  // Specifies the brand to use for this schema if no generic parameters have been bound to
  // anything. Generally, in the default brand, all generic parameters are treated as if they were
  // bound to `AnyPointer`.

  bool mayContainCapabilities = true;
  // See StructSchema::mayContainCapabilities.
};

inline bool RawBrandedSchema::isUnbound() const {
  // The unbound schema is the only one that has no scopes but is not the default schema.
  return scopeCount == 0 && this != &generic->defaultBrand;
}

}  // namespace _ (private)
}  // namespace capnp

CAPNP_END_HEADER
