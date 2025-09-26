# Copyright (c) 2015 Sandstorm Development Group, Inc. and contributors
# Licensed under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

@0x8ef99297a43a5e34;

$import "/capnp/c++.capnp".namespace("capnp::json");

struct Value {
  union {
    null @0 :Void;
    boolean @1 :Bool;
    number @2 :Float64;
    string @3 :Text;
    array @4 :List(Value);
    object @5 :List(Field);
    # Standard JSON values.

    call @6 :Call;
    # Non-standard: A "function call", applying a named function (named by a single identifier)
    # to a parameter list. Examples:
    #
    #     BinData(0, "Zm9vCg==")
    #     ISODate("2015-04-15T08:44:50.218Z")
    #
    # Mongo DB users will recognize the above as exactly the syntax Mongo uses to represent BSON
    # "binary" and "date" types in text, since JSON has no analog of these. This is basically the
    # reason this extension exists. We do NOT recommend using `call` unless you specifically need
    # to be compatible with some silly format that uses this syntax.

    raw @7 :Text;
    # Used to indicate that the text should be written directly to the output without
    # modifications. Use this if you have an already serialized JSON value and don't want
    # to feel the cost of deserializing the value just to serialize it again.
    #
    # The parser will never produce a `raw` value -- this is only useful for serialization.
    #
    # WARNING: You MUST ensure that the value is valid stand-alone JSOn. It will not be verified.
    # Invalid JSON could mjake the whole message unparsable. Worse, a malicious raw value could
    # perform JSON injection attacks. Make sure that the value was produced by a trustworthy JSON
    # encoder.
  }

  struct Field {
    name @0 :Text;
    value @1 :Value;
  }

  struct Call {
    function @0 :Text;
    params @1 :List(Value);
  }
}

# ========================================================================================
# Annotations to control parsing. Typical usage:
#
#     using Json = import "/capnp/compat/json.capnp";
#
# And then later on:
#
#     myField @0 :Text $Json.name("my_field");

annotation name @0xfa5b1fd61c2e7c3d (field, enumerant, method, group, union) :Text;
# Define an alternative name to use when encoding the given item in JSON. This can be used, for
# example, to use snake_case names where needed, even though Cap'n Proto uses strictly camelCase.
#
# (However, because JSON is derived from JavaScript, you *should* use camelCase names when
# defining JSON-based APIs. But, when supporting a pre-existing API you may not have a choice.)

annotation flatten @0x82d3e852af0336bf (field, group, union) :FlattenOptions;
# Specifies that an aggregate field should be flattened into its parent.
#
# In order to flatten a member of a union, the union (or, for an anonymous union, the parent
# struct type) must have the $jsonDiscriminator annotation.
#
# TODO(someday): Maybe support "flattening" a List(Value.Field) as a way to support unknown JSON
#   fields?

struct FlattenOptions {
  prefix @0 :Text = "";
  # Optional: Adds the given prefix to flattened field names.
}

annotation discriminator @0xcfa794e8d19a0162 (struct, union) :DiscriminatorOptions;
# Specifies that a union's variant will be decided not by which fields are present, but instead
# by a special discriminator field. The value of the discriminator field is a string naming which
# variant is active. This allows the members of the union to have the $jsonFlatten annotation, or
# to all have the same name.

struct DiscriminatorOptions {
  name @0 :Text;
  # The name of the discriminator field. Defaults to matching the name of the union.

  valueName @1 :Text;
  # If non-null, specifies that the union's value shall have the given field name, rather than the
  # value's name. In this case the union's variant can only be determined by looking at the
  # discriminant field, not by inspecting which value field is present.
  #
  # It is an error to use `valueName` while also declaring some variants as $flatten.
}

annotation base64 @0xd7d879450a253e4b (field) :Void;
# Place on a field of type `Data` to indicate that its JSON representation is a Base64 string.

annotation hex @0xf061e22f0ae5c7b5 (field) :Void;
# Place on a field of type `Data` to indicate that its JSON representation is a hex string.

annotation notification @0xa0a054dea32fd98c (method) :Void;
# Indicates that this method is a JSON-RPC "notification", meaning it expects no response.
