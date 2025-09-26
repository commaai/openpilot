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

#include "schema-loader.h"
#include <kj/string.h>
#include <kj/filesystem.h>

CAPNP_BEGIN_HEADER

namespace capnp {

class ParsedSchema;
class SchemaFile;

class SchemaParser {
  // Parses `.capnp` files to produce `Schema` objects.
  //
  // This class is thread-safe, hence all its methods are const.

public:
  SchemaParser();
  ~SchemaParser() noexcept(false);

  ParsedSchema parseFromDirectory(
      const kj::ReadableDirectory& baseDir, kj::Path path,
      kj::ArrayPtr<const kj::ReadableDirectory* const> importPath) const;
  // Parse a file from the KJ filesystem API.  Throws an exception if the file doesn't exist.
  //
  // `baseDir` and `path` are used together to resolve relative imports. `path` is the source
  // file's path within `baseDir`. Relative imports will be interpreted relative to `path` and
  // will be opened using `baseDir`. Note that the KJ filesystem API prohibits "breaking out" of
  // a directory using "..", so relative imports will be restricted to children of `baseDir`.
  //
  // `importPath` is used for absolute imports (imports that start with a '/'). Each directory in
  // the array will be searched in order until a file is found.
  //
  // All `ReadableDirectory` objects must remain valid until the `SchemaParser` is destroyed. Also,
  // the `importPath` array must remain valid. `path` will be copied; it need not remain valid.
  //
  // This method is a shortcut, equivalent to:
  //     parser.parseFile(SchemaFile::newDiskFile(baseDir, path, importPath))`;
  //
  // This method throws an exception if any errors are encountered in the file or in anything the
  // file depends on.  Note that merely importing another file does not count as a dependency on
  // anything in the imported file -- only the imported types which are actually used are
  // "dependencies".
  //
  // Hint: Use kj::newDiskFilesystem() to initialize the KJ filesystem API. Usually you should do
  //   this at a high level in your program, e.g. the main() function, and then pass down the
  //   appropriate File/Directory objects to the components that need them. Example:
  //
  //     auto fs = kj::newDiskFilesystem();
  //     SchemaParser parser;
  //     auto schema = parser.parseFromDirectory(fs->getCurrent(),
  //         kj::Path::parse("foo/bar.capnp"), nullptr);
  //
  // Hint: To use in-memory data rather than real disk, you can use kj::newInMemoryDirectory(),
  //   write the files you want, then pass it to SchemaParser. Example:
  //
  //     auto dir = kj::newInMemoryDirectory(kj::nullClock());
  //     auto path = kj::Path::parse("foo/bar.capnp");
  //     dir->openFile(path, kj::WriteMode::CREATE | kj::WriteMode::CREATE_PARENT)
  //        ->writeAll("struct Foo {}");
  //     auto schema = parser.parseFromDirectory(*dir, path, nullptr);
  //
  // Hint: You can create an in-memory directory but then populate it with real files from disk,
  //   in order to control what is visible while also avoiding reading files yourself or making
  //   extra copies. Example:
  //
  //     auto fs = kj::newDiskFilesystem();
  //     auto dir = kj::newInMemoryDirectory(kj::nullClock());
  //     auto fakePath = kj::Path::parse("foo/bar.capnp");
  //     auto realPath = kj::Path::parse("path/to/some/file.capnp");
  //     dir->transfer(fakePath, kj::WriteMode::CREATE | kj::WriteMode::CREATE_PARENT,
  //                   fs->getCurrent(), realPath, kj::TransferMode::LINK);
  //     auto schema = parser.parseFromDirectory(*dir, fakePath, nullptr);
  //
  //   In this example, note that any imports in the file will fail, since the in-memory directory
  //   you created contains no files except the specific one you linked in.

  ParsedSchema parseDiskFile(kj::StringPtr displayName, kj::StringPtr diskPath,
                             kj::ArrayPtr<const kj::StringPtr> importPath) const
      CAPNP_DEPRECATED("Use parseFromDirectory() instead.");
  // Creates a private kj::Filesystem and uses it to parse files from the real disk.
  //
  // DO NOT USE in new code. Use parseFromDirectory() instead.
  //
  // This API has a serious problem: the file can import and embed files located anywhere on disk
  // using relative paths. Even if you specify no `importPath`, relative imports still work. By
  // using `parseFromDirectory()`, you can arrange so that imports are only allowed within a
  // particular directory, or even set up a dummy filesystem where other files are not visible.

  void setDiskFilesystem(kj::Filesystem& fs)
      CAPNP_DEPRECATED("Use parseFromDirectory() instead.");
  // Call before calling parseDiskFile() to choose an alternative disk filesystem implementation.
  // This exists mostly for testing purposes; new code should use parseFromDirectory() instead.
  //
  // If parseDiskFile() is called without having called setDiskFilesystem(), then
  // kj::newDiskFilesystem() will be used instead.

  ParsedSchema parseFile(kj::Own<SchemaFile>&& file) const;
  // Advanced interface for parsing a file that may or may not be located in any global namespace.
  // Most users will prefer `parseFromDirectory()`.
  //
  // If the file has already been parsed (that is, a SchemaFile that compares equal to this one
  // was parsed previously), the existing schema will be returned again.
  //
  // This method reports errors by calling SchemaFile::reportError() on the file where the error
  // is located.  If that call does not throw an exception, `parseFile()` may in fact return
  // normally.  In this case, the result is a best-effort attempt to compile the schema, but it
  // may be invalid or corrupt, and using it for anything may cause exceptions to be thrown.

  kj::Maybe<schema::Node::SourceInfo::Reader> getSourceInfo(Schema schema) const;
  // Look up source info (e.g. doc comments) for the given schema, which must have come from this
  // SchemaParser. Note that this will also work for implicit group and param types that don't have
  // a type name hence don't have a `ParsedSchema`.

  template <typename T>
  inline void loadCompiledTypeAndDependencies() {
    // See SchemaLoader::loadCompiledTypeAndDependencies().
    getLoader().loadCompiledTypeAndDependencies<T>();
  }

  kj::Array<Schema> getAllLoaded() const {
    // Gets an array of all schema nodes that have been parsed so far.
    return getLoader().getAllLoaded();
  }

  void setFileIdsRequired(bool value) { fileIdsRequired = value; }
  // By befault, capnp files must declare a file-level type ID (like `@0xbe702824338d3f7f;`).
  // Use `setFileIdsReqired(false)` to lift this requirement.
  //
  // If no ID is specified, a random one will be assigned. This will cause all types declared in
  // the file to have randomized IDs as well (unless they declare an ID explicitly), which means
  // that parsing the same file twice will appear to to produce a totally new, incompatible set of
  // types. In particular, this means that you will not be able to use any interface types in the
  // file for RPC, since the RPC protocol uses type IDs to identify methods.
  //
  // Setting this false is particularly useful when using Cap'n Proto as a config format. Typically
  // type IDs are irrelevant for config files, and the requirement to specify one is cumbersome.
  // For this reason, `capnp eval` does not require type ID to be present.

private:
  struct Impl;
  struct DiskFileCompat;
  class ModuleImpl;
  kj::Own<Impl> impl;
  mutable bool hadErrors = false;
  bool fileIdsRequired = true;

  ModuleImpl& getModuleImpl(kj::Own<SchemaFile>&& file) const;
  const SchemaLoader& getLoader() const;
  SchemaLoader& getLoader();

  friend class ParsedSchema;
};

class ParsedSchema: public Schema {
  // ParsedSchema is an extension of Schema which also has the ability to look up nested nodes
  // by name.  See `SchemaParser`.

  class ParsedSchemaList;
  friend class ParsedSchemaList;

public:
  inline ParsedSchema(): parser(nullptr) {}

  kj::Maybe<ParsedSchema> findNested(kj::StringPtr name) const;
  // Gets the nested node with the given name, or returns null if there is no such nested
  // declaration.

  ParsedSchema getNested(kj::StringPtr name) const;
  // Gets the nested node with the given name, or throws an exception if there is no such nested
  // declaration.

  ParsedSchemaList getAllNested() const;
  // Get all the nested nodes

  schema::Node::SourceInfo::Reader getSourceInfo() const;
  // Get the source info for this schema.

private:
  inline ParsedSchema(Schema inner, const SchemaParser& parser): Schema(inner), parser(&parser) {}

  const SchemaParser* parser;
  friend class SchemaParser;
};

class ParsedSchema::ParsedSchemaList {
public:
  ParsedSchemaList() = default;  // empty list

  inline uint size() const { return list.size(); }
  ParsedSchema operator[](uint index) const;

  typedef _::IndexingIterator<const ParsedSchemaList, ParsedSchema> Iterator;
  inline Iterator begin() const { return Iterator(this, 0); }
  inline Iterator end() const { return Iterator(this, size()); }

private:
  ParsedSchema parent;
  List<schema::Node::NestedNode>::Reader list;

  inline ParsedSchemaList(ParsedSchema parent, List<schema::Node::NestedNode>::Reader list)
      : parent(parent), list(list) {}

  friend class ParsedSchema;
};

// =======================================================================================
// Advanced API

class SchemaFile {
  // Abstract interface representing a schema file.  You can implement this yourself in order to
  // gain more control over how the compiler resolves imports and reads files.  For the
  // common case of files on disk or other global filesystem-like namespaces, use
  // `SchemaFile::newDiskFile()`.

public:
  // Note: Cap'n Proto 0.6.x and below had classes FileReader and DiskFileReader and a method
  //   newDiskFile() defined here. These were removed when SchemaParser was transitioned to use the
  //   KJ filesystem API. You should be able to get the same effect by subclassing
  //   kj::ReadableDirectory, or using kj::newInMemoryDirectory().

  static kj::Own<SchemaFile> newFromDirectory(
      const kj::ReadableDirectory& baseDir, kj::Path path,
      kj::ArrayPtr<const kj::ReadableDirectory* const> importPath,
      kj::Maybe<kj::String> displayNameOverride = nullptr);
  // Construct a SchemaFile representing a file in a kj::ReadableDirectory. This is used to
  // implement SchemaParser::parseFromDirectory(); see there for details.
  //
  // The SchemaFile compares equal to any other SchemaFile that has exactly the same `baseDir`
  // object (by identity) and `path` (by value).

  // -----------------------------------------------------------------
  // For more control, you can implement this interface.

  virtual kj::StringPtr getDisplayName() const = 0;
  // Get the file's name, as it should appear in the schema.

  virtual kj::Array<const char> readContent() const = 0;
  // Read the file's entire content and return it as a byte array.

  virtual kj::Maybe<kj::Own<SchemaFile>> import(kj::StringPtr path) const = 0;
  // Resolve an import, relative to this file.
  //
  // `path` is exactly what appears between quotes after the `import` keyword in the source code.
  // It is entirely up to the `SchemaFile` to decide how to map this to another file.  Typically,
  // a leading '/' means that the file is an "absolute" path and is searched for in some list of
  // schema file repositories.  On the other hand, a path that doesn't start with '/' is relative
  // to the importing file.

  virtual bool operator==(const SchemaFile& other) const = 0;
  virtual bool operator!=(const SchemaFile& other) const = 0;
  virtual size_t hashCode() const = 0;
  // Compare two SchemaFiles to see if they refer to the same underlying file.  This is an
  // optimization used to avoid the need to re-parse a file to check its ID.

  struct SourcePos {
    uint byte;
    uint line;
    uint column;
  };
  virtual void reportError(SourcePos start, SourcePos end, kj::StringPtr message) const = 0;
  // Report that the file contains an error at the given interval.

private:
  class DiskSchemaFile;
};

}  // namespace capnp

CAPNP_END_HEADER
