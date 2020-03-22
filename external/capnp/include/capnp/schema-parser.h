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

#ifndef CAPNP_SCHEMA_PARSER_H_
#define CAPNP_SCHEMA_PARSER_H_

#if defined(__GNUC__) && !defined(CAPNP_HEADER_WARNINGS)
#pragma GCC system_header
#endif

#include "schema-loader.h"
#include <kj/string.h>

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

  ParsedSchema parseDiskFile(kj::StringPtr displayName, kj::StringPtr diskPath,
                             kj::ArrayPtr<const kj::StringPtr> importPath) const;
  // Parse a file located on disk.  Throws an exception if the file dosen't exist.
  //
  // Parameters:
  // * `displayName`:  The name that will appear in the file's schema node.  (If the file has
  //   already been parsed, this will be ignored and the display name from the first time it was
  //   parsed will be kept.)
  // * `diskPath`:  The path to the file on disk.
  // * `importPath`:  Directories to search when resolving absolute imports within this file
  //   (imports that start with a `/`).  Must remain valid until the SchemaParser is destroyed.
  //   (If the file has already been parsed, this will be ignored and the import path from the
  //   first time it was parsed will be kept.)
  //
  // This method is a shortcut, equivalent to:
  //     parser.parseFile(SchemaFile::newDiskFile(displayName, diskPath, importPath))`;
  //
  // This method throws an exception if any errors are encountered in the file or in anything the
  // file depends on.  Note that merely importing another file does not count as a dependency on
  // anything in the imported file -- only the imported types which are actually used are
  // "dependencies".

  ParsedSchema parseFile(kj::Own<SchemaFile>&& file) const;
  // Advanced interface for parsing a file that may or may not be located in any global namespace.
  // Most users will prefer `parseDiskFile()`.
  //
  // If the file has already been parsed (that is, a SchemaFile that compares equal to this one
  // was parsed previously), the existing schema will be returned again.
  //
  // This method reports errors by calling SchemaFile::reportError() on the file where the error
  // is located.  If that call does not throw an exception, `parseFile()` may in fact return
  // normally.  In this case, the result is a best-effort attempt to compile the schema, but it
  // may be invalid or corrupt, and using it for anything may cause exceptions to be thrown.

  template <typename T>
  inline void loadCompiledTypeAndDependencies() {
    // See SchemaLoader::loadCompiledTypeAndDependencies().
    getLoader().loadCompiledTypeAndDependencies<T>();
  }

private:
  struct Impl;
  class ModuleImpl;
  kj::Own<Impl> impl;
  mutable bool hadErrors = false;

  ModuleImpl& getModuleImpl(kj::Own<SchemaFile>&& file) const;
  SchemaLoader& getLoader();

  friend class ParsedSchema;
};

class ParsedSchema: public Schema {
  // ParsedSchema is an extension of Schema which also has the ability to look up nested nodes
  // by name.  See `SchemaParser`.

public:
  inline ParsedSchema(): parser(nullptr) {}

  kj::Maybe<ParsedSchema> findNested(kj::StringPtr name) const;
  // Gets the nested node with the given name, or returns null if there is no such nested
  // declaration.

  ParsedSchema getNested(kj::StringPtr name) const;
  // Gets the nested node with the given name, or throws an exception if there is no such nested
  // declaration.

private:
  inline ParsedSchema(Schema inner, const SchemaParser& parser): Schema(inner), parser(&parser) {}

  const SchemaParser* parser;
  friend class SchemaParser;
};

// =======================================================================================
// Advanced API

class SchemaFile {
  // Abstract interface representing a schema file.  You can implement this yourself in order to
  // gain more control over how the compiler resolves imports and reads files.  For the
  // common case of files on disk or other global filesystem-like namespaces, use
  // `SchemaFile::newDiskFile()`.

public:
  class FileReader {
  public:
    virtual bool exists(kj::StringPtr path) const = 0;
    virtual kj::Array<const char> read(kj::StringPtr path) const = 0;
  };

  class DiskFileReader final: public FileReader {
    // Implementation of FileReader that uses the local disk.  Files are read using mmap() if
    // possible.

  public:
    static const DiskFileReader instance;

    bool exists(kj::StringPtr path) const override;
    kj::Array<const char> read(kj::StringPtr path) const override;
  };

  static kj::Own<SchemaFile> newDiskFile(
      kj::StringPtr displayName, kj::StringPtr diskPath,
      kj::ArrayPtr<const kj::StringPtr> importPath,
      const FileReader& fileReader = DiskFileReader::instance);
  // Construct a SchemaFile representing a file on disk (or located in the filesystem-like
  // namespace represented by `fileReader`).
  //
  // Parameters:
  // * `displayName`:  The name that will appear in the file's schema node.
  // * `diskPath`:  The path to the file on disk.
  // * `importPath`:  Directories to search when resolving absolute imports within this file
  //   (imports that start with a `/`).  The array content must remain valid as long as the
  //   SchemaFile exists (which is at least as long as the SchemaParser that parses it exists).
  // * `fileReader`:  Allows you to use a filesystem other than the actual local disk.  Although,
  //   if you find yourself using this, it may make more sense for you to implement SchemaFile
  //   yourself.
  //
  // The SchemaFile compares equal to any other SchemaFile that has exactly the same disk path,
  // after canonicalization.
  //
  // The SchemaFile will throw an exception if any errors are reported.

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

#endif  // CAPNP_SCHEMA_PARSER_H_
