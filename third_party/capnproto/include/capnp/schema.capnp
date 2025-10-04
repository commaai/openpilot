# Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
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

using Cxx = import "/capnp/c++.capnp";

@0xa93fc509624c72d9;
$Cxx.namespace("capnp::schema");

using Id = UInt64;
# The globally-unique ID of a file, type, or annotation.

struct Node {
  id @0 :Id;

  displayName @1 :Text;
  # Name to present to humans to identify this Node.  You should not attempt to parse this.  Its
  # format could change.  It is not guaranteed to be unique.
  #
  # (On Zooko's triangle, this is the node's nickname.)

  displayNamePrefixLength @2 :UInt32;
  # If you want a shorter version of `displayName` (just naming this node, without its surrounding
  # scope), chop off this many characters from the beginning of `displayName`.

  scopeId @3 :Id;
  # ID of the lexical parent node.  Typically, the scope node will have a NestedNode pointing back
  # at this node, but robust code should avoid relying on this (and, in fact, group nodes are not
  # listed in the outer struct's nestedNodes, since they are listed in the fields).  `scopeId` is
  # zero if the node has no parent, which is normally only the case with files, but should be
  # allowed for any kind of node (in order to make runtime type generation easier).

  parameters @32 :List(Parameter);
  # If this node is parameterized (generic), the list of parameters. Empty for non-generic types.

  isGeneric @33 :Bool;
  # True if this node is generic, meaning that it or one of its parent scopes has a non-empty
  # `parameters`.

  struct Parameter {
    # Information about one of the node's parameters.

    name @0 :Text;
  }

  nestedNodes @4 :List(NestedNode);
  # List of nodes nested within this node, along with the names under which they were declared.

  struct NestedNode {
    name @0 :Text;
    # Unqualified symbol name.  Unlike Node.displayName, this *can* be used programmatically.
    #
    # (On Zooko's triangle, this is the node's petname according to its parent scope.)

    id @1 :Id;
    # ID of the nested node.  Typically, the target node's scopeId points back to this node, but
    # robust code should avoid relying on this.
  }

  annotations @5 :List(Annotation);
  # Annotations applied to this node.

  union {
    # Info specific to each kind of node.

    file @6 :Void;

    struct :group {
      dataWordCount @7 :UInt16;
      # Size of the data section, in words.

      pointerCount @8 :UInt16;
      # Size of the pointer section, in pointers (which are one word each).

      preferredListEncoding @9 :ElementSize;
      # The preferred element size to use when encoding a list of this struct.  If this is anything
      # other than `inlineComposite` then the struct is one word or less in size and is a candidate
      # for list packing optimization.

      isGroup @10 :Bool;
      # If true, then this "struct" node is actually not an independent node, but merely represents
      # some named union or group within a particular parent struct.  This node's scopeId refers
      # to the parent struct, which may itself be a union/group in yet another struct.
      #
      # All group nodes share the same dataWordCount and pointerCount as the top-level
      # struct, and their fields live in the same ordinal and offset spaces as all other fields in
      # the struct.
      #
      # Note that a named union is considered a special kind of group -- in fact, a named union
      # is exactly equivalent to a group that contains nothing but an unnamed union.

      discriminantCount @11 :UInt16;
      # Number of fields in this struct which are members of an anonymous union, and thus may
      # overlap.  If this is non-zero, then a 16-bit discriminant is present indicating which
      # of the overlapping fields is active.  This can never be 1 -- if it is non-zero, it must be
      # two or more.
      #
      # Note that the fields of an unnamed union are considered fields of the scope containing the
      # union -- an unnamed union is not its own group.  So, a top-level struct may contain a
      # non-zero discriminant count.  Named unions, on the other hand, are equivalent to groups
      # containing unnamed unions.  So, a named union has its own independent schema node, with
      # `isGroup` = true.

      discriminantOffset @12 :UInt32;
      # If `discriminantCount` is non-zero, this is the offset of the union discriminant, in
      # multiples of 16 bits.

      fields @13 :List(Field);
      # Fields defined within this scope (either the struct's top-level fields, or the fields of
      # a particular group; see `isGroup`).
      #
      # The fields are sorted by ordinal number, but note that because groups share the same
      # ordinal space, the field's index in this list is not necessarily exactly its ordinal.
      # On the other hand, the field's position in this list does remain the same even as the
      # protocol evolves, since it is not possible to insert or remove an earlier ordinal.
      # Therefore, for most use cases, if you want to identify a field by number, it may make the
      # most sense to use the field's index in this list rather than its ordinal.
    }

    enum :group {
      enumerants@14 :List(Enumerant);
      # Enumerants ordered by numeric value (ordinal).
    }

    interface :group {
      methods @15 :List(Method);
      # Methods ordered by ordinal.

      superclasses @31 :List(Superclass);
      # Superclasses of this interface.
    }

    const :group {
      type @16 :Type;
      value @17 :Value;
    }

    annotation :group {
      type @18 :Type;

      targetsFile @19 :Bool;
      targetsConst @20 :Bool;
      targetsEnum @21 :Bool;
      targetsEnumerant @22 :Bool;
      targetsStruct @23 :Bool;
      targetsField @24 :Bool;
      targetsUnion @25 :Bool;
      targetsGroup @26 :Bool;
      targetsInterface @27 :Bool;
      targetsMethod @28 :Bool;
      targetsParam @29 :Bool;
      targetsAnnotation @30 :Bool;
    }
  }

  struct SourceInfo {
    # Additional information about a node which is not needed at runtime, but may be useful for
    # documentation or debugging purposes. This is kept in a separate struct to make sure it
    # doesn't accidentally get included in contexts where it is not needed. The
    # `CodeGeneratorRequest` includes this information in a separate array.

    id @0 :Id;
    # ID of the Node which this info describes.

    docComment @1 :Text;
    # The top-level doc comment for the Node.

    members @2 :List(Member);
    # Information about each member -- i.e. fields (for structs), enumerants (for enums), or
    # methods (for interfaces).
    #
    # This list is the same length and order as the corresponding list in the Node, i.e.
    # Node.struct.fields, Node.enum.enumerants, or Node.interface.methods.

    struct Member {
      docComment @0 :Text;
      # Doc comment on the member.
    }

    # TODO(someday): Record location of the declaration in the original source code.
  }
}

struct Field {
  # Schema for a field of a struct.

  name @0 :Text;

  codeOrder @1 :UInt16;
  # Indicates where this member appeared in the code, relative to other members.
  # Code ordering may have semantic relevance -- programmers tend to place related fields
  # together.  So, using code ordering makes sense in human-readable formats where ordering is
  # otherwise irrelevant, like JSON.  The values of codeOrder are tightly-packed, so the maximum
  # value is count(members) - 1.  Fields that are members of a union are only ordered relative to
  # the other members of that union, so the maximum value there is count(union.members).

  annotations @2 :List(Annotation);

  const noDiscriminant :UInt16 = 0xffff;

  discriminantValue @3 :UInt16 = Field.noDiscriminant;
  # If the field is in a union, this is the value which the union's discriminant should take when
  # the field is active.  If the field is not in a union, this is 0xffff.

  union {
    slot :group {
      # A regular, non-group, non-fixed-list field.

      offset @4 :UInt32;
      # Offset, in units of the field's size, from the beginning of the section in which the field
      # resides.  E.g. for a UInt32 field, multiply this by 4 to get the byte offset from the
      # beginning of the data section.

      type @5 :Type;
      defaultValue @6 :Value;

      hadExplicitDefault @10 :Bool;
      # Whether the default value was specified explicitly.  Non-explicit default values are always
      # zero or empty values.  Usually, whether the default value was explicit shouldn't matter.
      # The main use case for this flag is for structs representing method parameters:
      # explicitly-defaulted parameters may be allowed to be omitted when calling the method.
    }

    group :group {
      # A group.

      typeId @7 :Id;
      # The ID of the group's node.
    }
  }

  ordinal :union {
    implicit @8 :Void;
    explicit @9 :UInt16;
    # The original ordinal number given to the field.  You probably should NOT use this; if you need
    # a numeric identifier for a field, use its position within the field array for its scope.
    # The ordinal is given here mainly just so that the original schema text can be reproduced given
    # the compiled version -- i.e. so that `capnp compile -ocapnp` can do its job.
  }
}

struct Enumerant {
  # Schema for member of an enum.

  name @0 :Text;

  codeOrder @1 :UInt16;
  # Specifies order in which the enumerants were declared in the code.
  # Like Struct.Field.codeOrder.

  annotations @2 :List(Annotation);
}

struct Superclass {
  id @0 :Id;
  brand @1 :Brand;
}

struct Method {
  # Schema for method of an interface.

  name @0 :Text;

  codeOrder @1 :UInt16;
  # Specifies order in which the methods were declared in the code.
  # Like Struct.Field.codeOrder.

  implicitParameters @7 :List(Node.Parameter);
  # The parameters listed in [] (typically, type / generic parameters), whose bindings are intended
  # to be inferred rather than specified explicitly, although not all languages support this.

  paramStructType @2 :Id;
  # ID of the parameter struct type.  If a named parameter list was specified in the method
  # declaration (rather than a single struct parameter type) then a corresponding struct type is
  # auto-generated.  Such an auto-generated type will not be listed in the interface's
  # `nestedNodes` and its `scopeId` will be zero -- it is completely detached from the namespace.
  # (Awkwardly, it does of course inherit generic parameters from the method's scope, which makes
  # this a situation where you can't just climb the scope chain to find where a particular
  # generic parameter was introduced. Making the `scopeId` zero was a mistake.)

  paramBrand @5 :Brand;
  # Brand of param struct type.

  resultStructType @3 :Id;
  # ID of the return struct type; similar to `paramStructType`.

  resultBrand @6 :Brand;
  # Brand of result struct type.

  annotations @4 :List(Annotation);
}

struct Type {
  # Represents a type expression.

  union {
    # The ordinals intentionally match those of Value.

    void @0 :Void;
    bool @1 :Void;
    int8 @2 :Void;
    int16 @3 :Void;
    int32 @4 :Void;
    int64 @5 :Void;
    uint8 @6 :Void;
    uint16 @7 :Void;
    uint32 @8 :Void;
    uint64 @9 :Void;
    float32 @10 :Void;
    float64 @11 :Void;
    text @12 :Void;
    data @13 :Void;

    list :group {
      elementType @14 :Type;
    }

    enum :group {
      typeId @15 :Id;
      brand @21 :Brand;
    }
    struct :group {
      typeId @16 :Id;
      brand @22 :Brand;
    }
    interface :group {
      typeId @17 :Id;
      brand @23 :Brand;
    }

    anyPointer :union {
      unconstrained :union {
        # A regular AnyPointer.
        #
        # The name "unconstrained" means as opposed to constraining it to match a type parameter.
        # In retrospect this name is probably a poor choice given that it may still be constrained
        # to be a struct, list, or capability.

        anyKind @18 :Void;       # truly AnyPointer
        struct @25 :Void;        # AnyStruct
        list @26 :Void;          # AnyList
        capability @27 :Void;    # Capability
      }

      parameter :group {
        # This is actually a reference to a type parameter defined within this scope.

        scopeId @19 :Id;
        # ID of the generic type whose parameter we're referencing. This should be a parent of the
        # current scope.

        parameterIndex @20 :UInt16;
        # Index of the parameter within the generic type's parameter list.
      }

      implicitMethodParameter :group {
        # This is actually a reference to an implicit (generic) parameter of a method. The only
        # legal context for this type to appear is inside Method.paramBrand or Method.resultBrand.

        parameterIndex @24 :UInt16;
      }
    }
  }
}

struct Brand {
  # Specifies bindings for parameters of generics. Since these bindings turn a generic into a
  # non-generic, we call it the "brand".

  scopes @0 :List(Scope);
  # For each of the target type and each of its parent scopes, a parameterization may be included
  # in this list. If no parameterization is included for a particular relevant scope, then either
  # that scope has no parameters or all parameters should be considered to be `AnyPointer`.

  struct Scope {
    scopeId @0 :Id;
    # ID of the scope to which these params apply.

    union {
      bind @1 :List(Binding);
      # List of parameter bindings.

      inherit @2 :Void;
      # The place where the Brand appears is within this scope or a sub-scope, and bindings
      # for this scope are deferred to later Brand applications. This is equivalent to a
      # pass-through binding list, where each of this scope's parameters is bound to itself.
      # For example:
      #
      #   struct Outer(T) {
      #     struct Inner {
      #       value @0 :T;
      #     }
      #     innerInherit @0 :Inner;            # Outer Brand.Scope is `inherit`.
      #     innerBindSelf @1 :Outer(T).Inner;  # Outer Brand.Scope explicitly binds T to T.
      #   }
      #
      # The innerInherit and innerBindSelf fields have equivalent types, but different Brand
      # styles.
    }
  }

  struct Binding {
    union {
      unbound @0 :Void;
      type @1 :Type;

      # TODO(someday): Allow non-type parameters? Unsure if useful.
    }
  }
}

struct Value {
  # Represents a value, e.g. a field default value, constant value, or annotation value.

  union {
    # The ordinals intentionally match those of Type.

    void @0 :Void;
    bool @1 :Bool;
    int8 @2 :Int8;
    int16 @3 :Int16;
    int32 @4 :Int32;
    int64 @5 :Int64;
    uint8 @6 :UInt8;
    uint16 @7 :UInt16;
    uint32 @8 :UInt32;
    uint64 @9 :UInt64;
    float32 @10 :Float32;
    float64 @11 :Float64;
    text @12 :Text;
    data @13 :Data;

    list @14 :AnyPointer;

    enum @15 :UInt16;
    struct @16 :AnyPointer;

    interface @17 :Void;
    # The only interface value that can be represented statically is "null", whose methods always
    # throw exceptions.

    anyPointer @18 :AnyPointer;
  }
}

struct Annotation {
  # Describes an annotation applied to a declaration.  Note AnnotationNode describes the
  # annotation's declaration, while this describes a use of the annotation.

  id @0 :Id;
  # ID of the annotation node.

  brand @2 :Brand;
  # Brand of the annotation.
  #
  # Note that the annotation itself is not allowed to be parameterized, but its scope might be.

  value @1 :Value;
}

enum ElementSize {
  # Possible element sizes for encoded lists.  These correspond exactly to the possible values of
  # the 3-bit element size component of a list pointer.

  empty @0;    # aka "void", but that's a keyword.
  bit @1;
  byte @2;
  twoBytes @3;
  fourBytes @4;
  eightBytes @5;
  pointer @6;
  inlineComposite @7;
}

struct CapnpVersion {
  major @0 :UInt16;
  minor @1 :UInt8;
  micro @2 :UInt8;
}

struct CodeGeneratorRequest {
  capnpVersion @2 :CapnpVersion;
  # Version of the `capnp` executable. Generally, code generators should ignore this, but the code
  # generators that ship with `capnp` itself will print a warning if this mismatches since that
  # probably indicates something is misconfigured.
  #
  # The first version of 'capnp' to set this was 0.6.0. So, if it's missing, the compiler version
  # is older than that.

  nodes @0 :List(Node);
  # All nodes parsed by the compiler, including for the files on the command line and their
  # imports.

  sourceInfo @3 :List(Node.SourceInfo);
  # Information about the original source code for each node, where available. This array may be
  # omitted or may be missing some nodes if no info is available for them.

  requestedFiles @1 :List(RequestedFile);
  # Files which were listed on the command line.

  struct RequestedFile {
    id @0 :Id;
    # ID of the file.

    filename @1 :Text;
    # Name of the file as it appeared on the command-line (minus the src-prefix).  You may use
    # this to decide where to write the output.

    imports @2 :List(Import);
    # List of all imported paths seen in this file.

    struct Import {
      id @0 :Id;
      # ID of the imported file.

      name @1 :Text;
      # Name which *this* file used to refer to the foreign file.  This may be a relative name.
      # This information is provided because it might be useful for code generation, e.g. to
      # generate #include directives in C++.  We don't put this in Node.file because this
      # information is only meaningful at compile time anyway.
      #
      # (On Zooko's triangle, this is the import's petname according to the importing file.)
    }
  }
}
