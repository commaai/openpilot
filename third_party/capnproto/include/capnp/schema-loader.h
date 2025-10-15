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

#include "schema.h"
#include <kj/memory.h>
#include <kj/mutex.h>

CAPNP_BEGIN_HEADER

namespace capnp {

class SchemaLoader {
  // Class which can be used to construct Schema objects from schema::Nodes as defined in
  // schema.capnp.
  //
  // It is a bad idea to use this class on untrusted input with exceptions disabled -- you may
  // be exposing yourself to denial-of-service attacks, as attackers can easily construct schemas
  // that are subtly inconsistent in a way that causes exceptions to be thrown either by
  // SchemaLoader or by the dynamic API when the schemas are subsequently used.  If you enable and
  // properly catch exceptions, you should be OK -- assuming no bugs in the Cap'n Proto
  // implementation, of course.

public:
  class LazyLoadCallback {
  public:
    virtual void load(const SchemaLoader& loader, uint64_t id) const = 0;
    // Request that the schema node with the given ID be loaded into the given SchemaLoader.  If
    // the callback is able to find a schema for this ID, it should invoke `loadOnce()` on
    // `loader` to load it.  If no such node exists, it should simply do nothing and return.
    //
    // The callback is allowed to load schema nodes other than the one requested, e.g. because it
    // expects they will be needed soon.
    //
    // If the `SchemaLoader` is used from multiple threads, the callback must be thread-safe.
    // In particular, it's possible for multiple threads to invoke `load()` with the same ID.
    // If the callback performs a large amount of work to look up IDs, it should be sure to
    // de-dup these requests.
  };

  SchemaLoader();

  SchemaLoader(const LazyLoadCallback& callback);
  // Construct a SchemaLoader which will invoke the given callback when a schema node is requested
  // that isn't already loaded.

  ~SchemaLoader() noexcept(false);
  KJ_DISALLOW_COPY_AND_MOVE(SchemaLoader);

  Schema get(uint64_t id, schema::Brand::Reader brand = schema::Brand::Reader(),
             Schema scope = Schema()) const;
  // Gets the schema for the given ID, throwing an exception if it isn't present.
  //
  // The returned schema may be invalidated if load() is called with a new schema for the same ID.
  // In general, you should not call load() while a schema from this loader is in-use.
  //
  // `brand` and `scope` are used to determine brand bindings where relevant. `brand` gives
  // parameter bindings for the target type's brand parameters that were specified at the reference
  // site. `scope` specifies the scope in which the type ID appeared -- if `brand` itself contains
  // parameter references or indicates that some parameters will be inherited, these will be
  // interpreted within / inherited from `scope`.

  kj::Maybe<Schema> tryGet(uint64_t id, schema::Brand::Reader bindings = schema::Brand::Reader(),
                           Schema scope = Schema()) const;
  // Like get() but doesn't throw.

  Schema getUnbound(uint64_t id) const;
  // Gets a special version of the schema in which all brand parameters are "unbound". This means
  // that if you look up a type via the Schema API, and it resolves to a brand parameter, the
  // returned Type's getBrandParameter() method will return info about that parameter. Otherwise,
  // normally, all brand parameters that aren't otherwise bound are assumed to simply be
  // "AnyPointer".

  Type getType(schema::Type::Reader type, Schema scope = Schema()) const;
  // Convenience method which interprets a schema::Type to produce a Type object. Implemented in
  // terms of get().

  Schema load(const schema::Node::Reader& reader);
  // Loads the given schema node.  Validates the node and throws an exception if invalid.  This
  // makes a copy of the schema, so the object passed in can be destroyed after this returns.
  //
  // If the node has any dependencies which are not already loaded, they will be initialized as
  // stubs -- empty schemas of whichever kind is expected.
  //
  // If another schema for the given reader has already been seen, the loader will inspect both
  // schemas to determine which one is newer, and use that that one.  If the two versions are
  // found to be incompatible, an exception is thrown.  If the two versions differ but are
  // compatible and the loader cannot determine which is newer (e.g., the only changes are renames),
  // the existing schema will be preferred.  Note that in any case, the loader will end up keeping
  // around copies of both schemas, so you shouldn't repeatedly reload schemas into the same loader.
  //
  // The following properties of the schema node are validated:
  // - Struct size and preferred list encoding are valid and consistent.
  // - Struct members are fields or unions.
  // - Union members are fields.
  // - Field offsets are in-bounds.
  // - Ordinals and codeOrders are sequential starting from zero.
  // - Values are of the right union case to match their types.
  //
  // You should assume anything not listed above is NOT validated.  In particular, things that are
  // not validated now, but could be in the future, include but are not limited to:
  // - Names.
  // - Annotation values.  (This is hard because the annotation declaration is not always
  //   available.)
  // - Content of default/constant values of pointer type.  (Validating these would require knowing
  //   their schema, but even if the schemas are available at validation time, they could be
  //   updated by a subsequent load(), invalidating existing values.  Instead, these values are
  //   validated at the time they are used, as usual for Cap'n Proto objects.)
  //
  // Also note that unknown types are not considered invalid.  Instead, the dynamic API returns
  // a DynamicValue with type UNKNOWN for these.

  Schema loadOnce(const schema::Node::Reader& reader) const;
  // Like `load()` but does nothing if a schema with the same ID is already loaded.  In contrast,
  // `load()` would attempt to compare the schemas and take the newer one.  `loadOnce()` is safe
  // to call even while concurrently using schemas from this loader.  It should be considered an
  // error to call `loadOnce()` with two non-identical schemas that share the same ID, although
  // this error may or may not actually be detected by the implementation.

  template <typename T>
  void loadCompiledTypeAndDependencies();
  // Load the schema for the given compiled-in type and all of its dependencies.
  //
  // If you want to be able to cast a DynamicValue built from this SchemaLoader to the compiled-in
  // type using as<T>(), you must call this method before constructing the DynamicValue.  Otherwise,
  // as<T>() will throw an exception complaining about type mismatch.

  kj::Array<Schema> getAllLoaded() const;
  // Get a complete list of all loaded schema nodes.  It is particularly useful to call this after
  // loadCompiledTypeAndDependencies<T>() in order to get a flat list of all of T's transitive
  // dependencies.

  void computeOptimizationHints();
  // Call after all interesting schemas have been loaded to compute optimization hints. In
  // particular, this initializes `hasNoCapabilities` for every struct type. Before this is called,
  // that value is initialized to false for all types (which ensures correct behavior but does not
  // allow the optimization).
  //
  // If any loaded struct types contain fields of types for which no schema has been loaded, they
  // will be presumed to possibly contain capabilities. `LazyLoadCallback` will NOT be invoked to
  // load any types that haven't been loaded yet.
  //
  // TODO(someday): Perhaps we could dynamically initialize the hints on-demand, but it would be
  //   much more work to implement.

private:
  class Validator;
  class CompatibilityChecker;
  class Impl;
  class InitializerImpl;
  class BrandedInitializerImpl;
  kj::MutexGuarded<kj::Own<Impl>> impl;

  void loadNative(const _::RawSchema* nativeSchema);
};

template <typename T>
inline void SchemaLoader::loadCompiledTypeAndDependencies() {
  loadNative(&_::rawSchema<T>());
}

}  // namespace capnp

CAPNP_END_HEADER
