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

$import "/capnp/c++.capnp".namespace("capnp");

struct JsonValue {
  union {
    null @0 :Void;
    boolean @1 :Bool;
    number @2 :Float64;
    string @3 :Text;
    array @4 :List(JsonValue);
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
  }

  struct Field {
    name @0 :Text;
    value @1 :JsonValue;
  }

  struct Call {
    function @0 :Text;
    params @1 :List(JsonValue);
  }
}
