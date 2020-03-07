// Copyright (c) 2015 Sandstorm Development Group, Inc. and contributors
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

#ifndef CAPNP_COMPAT_JSON_H_
#define CAPNP_COMPAT_JSON_H_

#include <capnp/schema.h>
#include <capnp/dynamic.h>
#include <capnp/compat/json.capnp.h>

namespace capnp {

class JsonCodec {
  // Flexible class for encoding Cap'n Proto types as JSON, and decoding JSON back to Cap'n Proto.
  //
  // Typical usage:
  //
  //     JsonCodec json;
  //
  //     // encode
  //     kj::String encoded = json.encode(someStructReader);
  //
  //     // decode
  //     json.decode(encoded, someStructBuilder);
  //
  // Advanced users can do fancy things like override the way certain types or fields are
  // represented in JSON by registering handlers. See the unit test for an example.
  //
  // Notes:
  // - When encoding, all primitive fields are always encoded, even if default-valued. Pointer
  //   fields are only encoded if they are non-null.
  // - 64-bit integers are encoded as strings, since JSON "numbers" are double-precision floating
  //   points which cannot store a 64-bit integer without losing data.
  // - NaNs and infinite floating point numbers are not allowed by the JSON spec, and so are encoded
  //   as null. This matches the behavior of `JSON.stringify` in at least Firefox and Chrome.
  // - Data is encoded as an array of numbers in the range [0,255]. You probably want to register
  //   a handler that does something better, like maybe base64 encoding, but there are a zillion
  //   different ways people do this.
  // - Encoding/decoding capabilities and AnyPointers requires registering a Handler, since there's
  //   no obvious default behavior.
  // - When decoding, unrecognized field names are ignored. Note: This means that JSON is NOT a
  //   good format for receiving input from a human. Consider `capnp eval` or the SchemaParser
  //   library for human input.

public:
  JsonCodec();
  ~JsonCodec() noexcept(false);

  // ---------------------------------------------------------------------------
  // standard API

  void setPrettyPrint(bool enabled);
  // Enable to insert newlines, indentation, and other extra spacing into the output. The default
  // is to use minimal whitespace.

  void setMaxNestingDepth(size_t maxNestingDepth);
  // Set maximum nesting depth when decoding JSON to prevent highly nested input from overflowing
  // the call stack. The default is 64.

  template <typename T>
  kj::String encode(T&& value);
  // Encode any Cap'n Proto value to JSON, including primitives and
  // Dynamic{Enum,Struct,List,Capability}, but not DynamicValue (see below).

  kj::String encode(DynamicValue::Reader value, Type type) const;
  // Encode a DynamicValue to JSON. `type` is needed because `DynamicValue` itself does
  // not distinguish between e.g. int32 and int64, which in JSON are handled differently. Most
  // of the time, though, you can use the single-argument templated version of `encode()` instead.

  void decode(kj::ArrayPtr<const char> input, DynamicStruct::Builder output) const;
  // Decode JSON text directly into a struct builder. This only works for structs since lists
  // need to be allocated with the correct size in advance.
  //
  // (Remember that any Cap'n Proto struct reader type can be implicitly cast to
  // DynamicStruct::Reader.)

  template <typename T>
  Orphan<T> decode(kj::ArrayPtr<const char> input, Orphanage orphanage) const;
  // Decode JSON text to any Cap'n Proto object (pointer value), allocated using the given
  // orphanage. T must be specified explicitly and cannot be dynamic, e.g.:
  //
  //     Orphan<MyType> orphan = json.decode<MyType>(text, orphanage);

  template <typename T>
  ReaderFor<T> decode(kj::ArrayPtr<const char> input) const;
  // Decode JSON text into a primitive or capability value. T must be specified explicitly and
  // cannot be dynamic, e.g.:
  //
  //     uint32_t n = json.decode<uint32_t>(text);

  Orphan<DynamicValue> decode(kj::ArrayPtr<const char> input, Type type, Orphanage orphanage) const;
  Orphan<DynamicList> decode(
      kj::ArrayPtr<const char> input, ListSchema type, Orphanage orphanage) const;
  Orphan<DynamicStruct> decode(
      kj::ArrayPtr<const char> input, StructSchema type, Orphanage orphanage) const;
  DynamicCapability::Client decode(kj::ArrayPtr<const char> input, InterfaceSchema type) const;
  DynamicEnum decode(kj::ArrayPtr<const char> input, EnumSchema type) const;
  // Decode to a dynamic value, specifying the type schema.

  // ---------------------------------------------------------------------------
  // layered API
  //
  // You can separate text <-> JsonValue from JsonValue <-> T. These are particularly useful
  // for calling from Handler implementations.

  kj::String encodeRaw(JsonValue::Reader value) const;
  void decodeRaw(kj::ArrayPtr<const char> input, JsonValue::Builder output) const;
  // Translate JsonValue <-> text.

  template <typename T>
  void encode(T&& value, JsonValue::Builder output);
  void encode(DynamicValue::Reader input, Type type, JsonValue::Builder output) const;
  void decode(JsonValue::Reader input, DynamicStruct::Builder output) const;
  template <typename T>
  Orphan<T> decode(JsonValue::Reader input, Orphanage orphanage) const;
  template <typename T>
  ReaderFor<T> decode(JsonValue::Reader input) const;

  Orphan<DynamicValue> decode(JsonValue::Reader input, Type type, Orphanage orphanage) const;
  Orphan<DynamicList> decode(JsonValue::Reader input, ListSchema type, Orphanage orphanage) const;
  Orphan<DynamicStruct> decode(
      JsonValue::Reader input, StructSchema type, Orphanage orphanage) const;
  DynamicCapability::Client decode(JsonValue::Reader input, InterfaceSchema type) const;
  DynamicEnum decode(JsonValue::Reader input, EnumSchema type) const;

  // ---------------------------------------------------------------------------
  // specializing particular types

  template <typename T, Style s = style<T>()>
  class Handler;
  // Implement this interface to specify a special encoding for a particular type or field.
  //
  // The templates are a bit ugly, but subclasses of this type essentially implement two methods,
  // one to encode values of this type and one to decode values of this type. `encode()` is simple:
  //
  //   void encode(const JsonCodec& codec, ReaderFor<T> input, JsonValue::Builder output) const;
  //
  // `decode()` is a bit trickier. When T is a struct (including DynamicStruct), it is:
  //
  //   void decode(const JsonCodec& codec, JsonValue::Reader input, BuilderFor<T> output) const;
  //
  // However, when T is a primitive, decode() is:
  //
  //   T decode(const JsonCodec& codec, JsonValue::Reader input) const;
  //
  // Or when T is any non-struct object (list, blob), decode() is:
  //
  //   Orphan<T> decode(const JsonCodec& codec, JsonValue::Reader input, Orphanage orphanage) const;
  //
  // Or when T is an interface:
  //
  //   T::Client decode(const JsonCodec& codec, JsonValue::Reader input) const;
  //
  // Additionally, when T is a struct you can *optionally* also implement the orphan-returning form
  // of decode(), but it will only be called when the struct would be allocated as an individual
  // object, not as part of a list. This allows you to return "nullptr" in these cases to say that
  // the pointer value should be null. This does not apply to list elements because struct list
  // elements cannot ever be null (since Cap'n Proto encodes struct lists as a flat list rather
  // than list-of-pointers).

  template <typename T>
  void addTypeHandler(Handler<T>& handler);
  void addTypeHandler(Type type, Handler<DynamicValue>& handler);
  void addTypeHandler(EnumSchema type, Handler<DynamicEnum>& handler);
  void addTypeHandler(StructSchema type, Handler<DynamicStruct>& handler);
  void addTypeHandler(ListSchema type, Handler<DynamicList>& handler);
  void addTypeHandler(InterfaceSchema type, Handler<DynamicCapability>& handler);
  // Arrange that whenever the type T appears in the message, your handler will be used to
  // encode/decode it.
  //
  // Note that if you register a handler for a capability type, it will also apply to subtypes.
  // Thus Handler<Capability> handles all capabilities.

  template <typename T>
  void addFieldHandler(StructSchema::Field field, Handler<T>& handler);
  // Matches only the specific field. T can be a dynamic type. T must match the field's type.

private:
  class HandlerBase;
  struct Impl;

  kj::Own<Impl> impl;

  void encodeField(StructSchema::Field field, DynamicValue::Reader input,
                   JsonValue::Builder output) const;
  void decodeArray(List<JsonValue>::Reader input, DynamicList::Builder output) const;
  void decodeObject(List<JsonValue::Field>::Reader input, DynamicStruct::Builder output) const;
  void addTypeHandlerImpl(Type type, HandlerBase& handler);
  void addFieldHandlerImpl(StructSchema::Field field, Type type, HandlerBase& handler);
};

// =======================================================================================
// inline implementation details

template <typename T>
kj::String JsonCodec::encode(T&& value) {
  typedef FromAny<kj::Decay<T>> Base;
  return encode(DynamicValue::Reader(ReaderFor<Base>(kj::fwd<T>(value))), Type::from<Base>());
}

template <typename T>
inline Orphan<T> JsonCodec::decode(kj::ArrayPtr<const char> input, Orphanage orphanage) const {
  return decode(input, Type::from<T>(), orphanage).template releaseAs<T>();
}

template <typename T>
inline ReaderFor<T> JsonCodec::decode(kj::ArrayPtr<const char> input) const {
  static_assert(style<T>() == Style::PRIMITIVE || style<T>() == Style::CAPABILITY,
                "must specify an orphanage to decode an object type");
  return decode(input, Type::from<T>(), Orphanage()).getReader().template as<T>();
}

inline Orphan<DynamicList> JsonCodec::decode(
    kj::ArrayPtr<const char> input, ListSchema type, Orphanage orphanage) const {
  return decode(input, Type(type), orphanage).releaseAs<DynamicList>();
}
inline Orphan<DynamicStruct> JsonCodec::decode(
    kj::ArrayPtr<const char> input, StructSchema type, Orphanage orphanage) const {
  return decode(input, Type(type), orphanage).releaseAs<DynamicStruct>();
}
inline DynamicCapability::Client JsonCodec::decode(
    kj::ArrayPtr<const char> input, InterfaceSchema type) const {
  return decode(input, Type(type), Orphanage()).getReader().as<DynamicCapability>();
}
inline DynamicEnum JsonCodec::decode(kj::ArrayPtr<const char> input, EnumSchema type) const {
  return decode(input, Type(type), Orphanage()).getReader().as<DynamicEnum>();
}

// -----------------------------------------------------------------------------

template <typename T>
void JsonCodec::encode(T&& value, JsonValue::Builder output) {
  typedef FromAny<kj::Decay<T>> Base;
  encode(DynamicValue::Reader(ReaderFor<Base>(kj::fwd<T>(value))), Type::from<Base>(), output);
}

template <typename T>
inline Orphan<T> JsonCodec::decode(JsonValue::Reader input, Orphanage orphanage) const {
  return decode(input, Type::from<T>(), orphanage).template releaseAs<T>();
}

template <typename T>
inline ReaderFor<T> JsonCodec::decode(JsonValue::Reader input) const {
  static_assert(style<T>() == Style::PRIMITIVE || style<T>() == Style::CAPABILITY,
                "must specify an orphanage to decode an object type");
  return decode(input, Type::from<T>(), Orphanage()).getReader().template as<T>();
}

inline Orphan<DynamicList> JsonCodec::decode(
    JsonValue::Reader input, ListSchema type, Orphanage orphanage) const {
  return decode(input, Type(type), orphanage).releaseAs<DynamicList>();
}
inline Orphan<DynamicStruct> JsonCodec::decode(
    JsonValue::Reader input, StructSchema type, Orphanage orphanage) const {
  return decode(input, Type(type), orphanage).releaseAs<DynamicStruct>();
}
inline DynamicCapability::Client JsonCodec::decode(
    JsonValue::Reader input, InterfaceSchema type) const {
  return decode(input, Type(type), Orphanage()).getReader().as<DynamicCapability>();
}
inline DynamicEnum JsonCodec::decode(JsonValue::Reader input, EnumSchema type) const {
  return decode(input, Type(type), Orphanage()).getReader().as<DynamicEnum>();
}

// -----------------------------------------------------------------------------

class JsonCodec::HandlerBase {
  // Internal helper; ignore.
public:
  virtual void encodeBase(const JsonCodec& codec, DynamicValue::Reader input,
                          JsonValue::Builder output) const = 0;
  virtual Orphan<DynamicValue> decodeBase(const JsonCodec& codec, JsonValue::Reader input,
                                          Type type, Orphanage orphanage) const;
  virtual void decodeStructBase(const JsonCodec& codec, JsonValue::Reader input,
                                DynamicStruct::Builder output) const;
};

template <typename T>
class JsonCodec::Handler<T, Style::POINTER>: private JsonCodec::HandlerBase {
public:
  virtual void encode(const JsonCodec& codec, ReaderFor<T> input,
                      JsonValue::Builder output) const = 0;
  virtual Orphan<T> decode(const JsonCodec& codec, JsonValue::Reader input,
                           Orphanage orphanage) const = 0;

private:
  void encodeBase(const JsonCodec& codec, DynamicValue::Reader input,
                  JsonValue::Builder output) const override final {
    encode(codec, input.as<T>(), output);
  }
  Orphan<DynamicValue> decodeBase(const JsonCodec& codec, JsonValue::Reader input,
                                  Type type, Orphanage orphanage) const override final {
    return decode(codec, input, orphanage);
  }
  friend class JsonCodec;
};

template <typename T>
class JsonCodec::Handler<T, Style::STRUCT>: private JsonCodec::HandlerBase {
public:
  virtual void encode(const JsonCodec& codec, ReaderFor<T> input,
                      JsonValue::Builder output) const = 0;
  virtual void decode(const JsonCodec& codec, JsonValue::Reader input,
                      BuilderFor<T> output) const = 0;
  virtual Orphan<T> decode(const JsonCodec& codec, JsonValue::Reader input,
                           Orphanage orphanage) const {
    // If subclass does not override, fall back to regular version.
    auto result = orphanage.newOrphan<T>();
    decode(codec, input, result.get());
    return result;
  }

private:
  void encodeBase(const JsonCodec& codec, DynamicValue::Reader input,
                  JsonValue::Builder output) const override final {
    encode(codec, input.as<T>(), output);
  }
  Orphan<DynamicValue> decodeBase(const JsonCodec& codec, JsonValue::Reader input,
                                  Type type, Orphanage orphanage) const override final {
    return decode(codec, input, orphanage);
  }
  void decodeStructBase(const JsonCodec& codec, JsonValue::Reader input,
                        DynamicStruct::Builder output) const override final {
    decode(codec, input, output.as<T>());
  }
  friend class JsonCodec;
};

template <>
class JsonCodec::Handler<DynamicStruct>: private JsonCodec::HandlerBase {
  // Almost identical to Style::STRUCT except that we pass the struct type to decode().

public:
  virtual void encode(const JsonCodec& codec, DynamicStruct::Reader input,
                      JsonValue::Builder output) const = 0;
  virtual void decode(const JsonCodec& codec, JsonValue::Reader input,
                      DynamicStruct::Builder output) const = 0;
  virtual Orphan<DynamicStruct> decode(const JsonCodec& codec, JsonValue::Reader input,
                                       StructSchema type, Orphanage orphanage) const {
    // If subclass does not override, fall back to regular version.
    auto result = orphanage.newOrphan(type);
    decode(codec, input, result.get());
    return result;
  }

private:
  void encodeBase(const JsonCodec& codec, DynamicValue::Reader input,
                  JsonValue::Builder output) const override final {
    encode(codec, input.as<DynamicStruct>(), output);
  }
  Orphan<DynamicValue> decodeBase(const JsonCodec& codec, JsonValue::Reader input,
                                  Type type, Orphanage orphanage) const override final {
    return decode(codec, input, type.asStruct(), orphanage);
  }
  void decodeStructBase(const JsonCodec& codec, JsonValue::Reader input,
                        DynamicStruct::Builder output) const override final {
    decode(codec, input, output.as<DynamicStruct>());
  }
  friend class JsonCodec;
};

template <typename T>
class JsonCodec::Handler<T, Style::PRIMITIVE>: private JsonCodec::HandlerBase {
public:
  virtual void encode(const JsonCodec& codec, T input, JsonValue::Builder output) const = 0;
  virtual T decode(const JsonCodec& codec, JsonValue::Reader input) const = 0;

private:
  void encodeBase(const JsonCodec& codec, DynamicValue::Reader input,
                  JsonValue::Builder output) const override final {
    encode(codec, input.as<T>(), output);
  }
  Orphan<DynamicValue> decodeBase(const JsonCodec& codec, JsonValue::Reader input,
                                  Type type, Orphanage orphanage) const override final {
    return decode(codec, input);
  }
  friend class JsonCodec;
};

template <typename T>
class JsonCodec::Handler<T, Style::CAPABILITY>: private JsonCodec::HandlerBase {
public:
  virtual void encode(const JsonCodec& codec, typename T::Client input,
                      JsonValue::Builder output) const = 0;
  virtual typename T::Client decode(const JsonCodec& codec, JsonValue::Reader input) const = 0;

private:
  void encodeBase(const JsonCodec& codec, DynamicValue::Reader input,
                  JsonValue::Builder output) const override final {
    encode(codec, input.as<T>(), output);
  }
  Orphan<DynamicValue> decodeBase(const JsonCodec& codec, JsonValue::Reader input,
                                  Type type, Orphanage orphanage) const override final {
    return orphanage.newOrphanCopy(decode(codec, input));
  }
  friend class JsonCodec;
};

template <typename T>
inline void JsonCodec::addTypeHandler(Handler<T>& handler) {
  addTypeHandlerImpl(Type::from<T>(), handler);
}
inline void JsonCodec::addTypeHandler(Type type, Handler<DynamicValue>& handler) {
  addTypeHandlerImpl(type, handler);
}
inline void JsonCodec::addTypeHandler(EnumSchema type, Handler<DynamicEnum>& handler) {
  addTypeHandlerImpl(type, handler);
}
inline void JsonCodec::addTypeHandler(StructSchema type, Handler<DynamicStruct>& handler) {
  addTypeHandlerImpl(type, handler);
}
inline void JsonCodec::addTypeHandler(ListSchema type, Handler<DynamicList>& handler) {
  addTypeHandlerImpl(type, handler);
}
inline void JsonCodec::addTypeHandler(InterfaceSchema type, Handler<DynamicCapability>& handler) {
  addTypeHandlerImpl(type, handler);
}

template <typename T>
inline void JsonCodec::addFieldHandler(StructSchema::Field field, Handler<T>& handler) {
  addFieldHandlerImpl(field, Type::from<T>(), handler);
}

template <> void JsonCodec::addTypeHandler(Handler<DynamicValue>& handler)
    KJ_UNAVAILABLE("JSON handlers for type sets (e.g. all structs, all lists) not implemented; "
                   "try specifying a specific type schema as the first parameter");
template <> void JsonCodec::addTypeHandler(Handler<DynamicEnum>& handler)
    KJ_UNAVAILABLE("JSON handlers for type sets (e.g. all structs, all lists) not implemented; "
                   "try specifying a specific type schema as the first parameter");
template <> void JsonCodec::addTypeHandler(Handler<DynamicStruct>& handler)
    KJ_UNAVAILABLE("JSON handlers for type sets (e.g. all structs, all lists) not implemented; "
                   "try specifying a specific type schema as the first parameter");
template <> void JsonCodec::addTypeHandler(Handler<DynamicList>& handler)
    KJ_UNAVAILABLE("JSON handlers for type sets (e.g. all structs, all lists) not implemented; "
                   "try specifying a specific type schema as the first parameter");
template <> void JsonCodec::addTypeHandler(Handler<DynamicCapability>& handler)
    KJ_UNAVAILABLE("JSON handlers for type sets (e.g. all structs, all lists) not implemented; "
                   "try specifying a specific type schema as the first parameter");
// TODO(someday): Implement support for registering handlers that cover thinsg like "all structs"
//   or "all lists". Currently you can only target a specific struct or list type.

} // namespace capnp

#endif // CAPNP_COMPAT_JSON_H_
