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

#pragma once

#include "memory.h"
#include "io.h"
#include <inttypes.h>
#include "time.h"
#include "function.h"
#include "hash.h"

KJ_BEGIN_HEADER

namespace kj {

template <typename T>
class Vector;

class PathPtr;

class Path {
  // A Path identifies a file in a directory tree.
  //
  // In KJ, we avoid representing paths as plain strings because this can lead to path injection
  // bugs as well as numerous kinds of bugs relating to path parsing edge cases. The Path class's
  // interface is designed to "make it hard to screw up".
  //
  // A "Path" is in fact a list of strings, each string being one component of the path (as would
  // normally be separated by '/'s). Path components are not allowed to contain '/' nor '\0', nor
  // are they allowed to be the special names "", ".", nor "..".
  //
  // If you explicitly want to parse a path that contains '/'s, ".", and "..", you must use
  // parse() and/or eval(). However, users of this interface are encouraged to avoid parsing
  // paths at all, and instead express paths as string arrays.
  //
  // Note that when using the Path class, ".." is always canonicalized in path space without
  // consulting the actual filesystem. This means that "foo/some-symlink/../bar" is exactly
  // equivalent to "foo/bar". This differs from the kernel's behavior when resolving paths passed
  // to system calls: the kernel would have resolved "some-symlink" to its target physical path,
  // and then would have interpreted ".." relative to that. In practice, the kernel's behavior is
  // rarely what the user or programmer intended, hence canonicalizing in path space produces a
  // better result.
  //
  // Path objects are "immutable": functions that "modify" the path return a new path. However,
  // if the path being operated on is an rvalue, copying can be avoided. Hence it makes sense to
  // write code like:
  //
  //     Path p = ...;
  //     p = kj::mv(p).append("bar");  // in-place update, avoids string copying

public:
  Path(decltype(nullptr));  // empty path

  explicit Path(StringPtr name);
  explicit Path(String&& name);
  // Create a Path containing only one component. `name` is a single filename; it cannot contain
  // '/' nor '\0' nor can it be exactly "" nor "." nor "..".
  //
  // If you want to allow '/'s and such, you must call Path::parse(). We force you to do this to
  // prevent path injection bugs where you didn't consider what would happen if the path contained
  // a '/'.

  explicit Path(std::initializer_list<StringPtr> parts);
  explicit Path(ArrayPtr<const StringPtr> parts);
  explicit Path(Array<String> parts);
  // Construct a path from an array. Note that this means you can do:
  //
  //     Path{"foo", "bar", "baz"}   // equivalent to Path::parse("foo/bar/baz")

  KJ_DISALLOW_COPY(Path);
  Path(Path&&) = default;
  Path& operator=(Path&&) = default;

  Path clone() const;

  static Path parse(StringPtr path);
  // Parses a path in traditional format. Components are separated by '/'. Any use of "." or
  // ".." will be canonicalized (if they can't be canonicalized, e.g. because the path starts with
  // "..", an exception is thrown). Multiple consecutive '/'s will be collapsed. A leading '/'
  // is NOT accepted -- if that is a problem, you probably want `eval()`. Trailing '/'s are
  // ignored.

  Path append(Path&& suffix) const&;
  Path append(Path&& suffix) &&;
  Path append(PathPtr suffix) const&;
  Path append(PathPtr suffix) &&;
  Path append(StringPtr suffix) const&;
  Path append(StringPtr suffix) &&;
  Path append(String&& suffix) const&;
  Path append(String&& suffix) &&;
  // Create a new path by appending the given path to this path.
  //
  // `suffix` cannot contain '/' characters. Instead, you can append an array:
  //
  //     path.append({"foo", "bar"})
  //
  // Or, use Path::parse():
  //
  //     path.append(Path::parse("foo//baz/../bar"))

  Path eval(StringPtr pathText) const&;
  Path eval(StringPtr pathText) &&;
  // Evaluates a traditional path relative to this one. `pathText` is parsed like `parse()` would,
  // except that:
  // - It can contain leading ".." components that traverse up the tree.
  // - It can have a leading '/' which completely replaces the current path.
  //
  // THE NAME OF THIS METHOD WAS CHOSEN TO INSPIRE FEAR.
  //
  // Instead of using `path.eval(str)`, always consider whether you really want
  // `path.append(Path::parse(str))`. The former is much riskier than the latter in terms of path
  // injection vulnerabilities.

  PathPtr basename() const&;
  Path basename() &&;
  // Get the last component of the path. (Use `basename()[0]` to get just the string.)

  PathPtr parent() const&;
  Path parent() &&;
  // Get the parent path.

  String toString(bool absolute = false) const;
  // Converts the path to a traditional path string, appropriate to pass to a unix system call.
  // Never throws.

  const String& operator[](size_t i) const&;
  String operator[](size_t i) &&;
  size_t size() const;
  const String* begin() const;
  const String* end() const;
  PathPtr slice(size_t start, size_t end) const&;
  Path slice(size_t start, size_t end) &&;
  // A Path can be accessed as an array of strings.

  bool operator==(PathPtr other) const;
  bool operator!=(PathPtr other) const;
  bool operator< (PathPtr other) const;
  bool operator> (PathPtr other) const;
  bool operator<=(PathPtr other) const;
  bool operator>=(PathPtr other) const;
  // Compare path components lexically.

  bool operator==(const Path& other) const;
  bool operator!=(const Path& other) const;
  bool operator< (const Path& other) const;
  bool operator> (const Path& other) const;
  bool operator<=(const Path& other) const;
  bool operator>=(const Path& other) const;

  uint hashCode() const;
  // Can use in HashMap.

  bool startsWith(PathPtr prefix) const;
  bool endsWith(PathPtr suffix) const;
  // Compare prefix / suffix.

  Path evalWin32(StringPtr pathText) const&;
  Path evalWin32(StringPtr pathText) &&;
  // Evaluates a Win32-style path, as might be written by a user. Differences from `eval()`
  // include:
  //
  // - Backslashes can be used as path separators.
  // - Absolute paths begin with a drive letter followed by a colon. The drive letter, including
  //   the colon, will become the first component of the path, e.g. "c:\foo" becomes {"c:", "foo"}.
  // - A network path like "\\host\share\path" is parsed as {"host", "share", "path"}.

  Path evalNative(StringPtr pathText) const&;
  Path evalNative(StringPtr pathText) &&;
  // Alias for either eval() or evalWin32() depending on the target platform. Use this when you are
  // parsing a path provided by a user and you want the user to be able to use the "natural" format
  // for their platform.

  String toWin32String(bool absolute = false) const;
  // Converts the path to a Win32 path string, as you might display to a user.
  //
  // This is meant for display. For making Win32 system calls, consider `toWin32Api()` instead.
  //
  // If `absolute` is true, the path is expected to be an absolute path, meaning the first
  // component is a drive letter, namespace, or network host name. These are converted to their
  // regular Win32 format -- i.e. this method does the reverse of `evalWin32()`.
  //
  // This throws if the path would have unexpected special meaning or is otherwise invalid on
  // Windows, such as if it contains backslashes (within a path component), colons, or special
  // names like "con".

  String toNativeString(bool absolute = false) const;
  // Alias for either toString() or toWin32String() depending on the target platform. Use this when
  // you are formatting a path to display to a user and you want to present it in the "natural"
  // format for the user's platform.

  Array<wchar_t> forWin32Api(bool absolute) const;
  // Like toWin32String, but additionally:
  // - Converts the path to UTF-16, with a NUL terminator included.
  // - For absolute paths, adds the "\\?\" prefix which opts into permitting paths longer than
  //   MAX_PATH, and turns off relative path processing (which KJ paths already handle in userspace
  //   anyway).
  //
  // This method is good to use when making a Win32 API call, e.g.:
  //
  //     DeleteFileW(path.forWin32Api(true).begin());

  static Path parseWin32Api(ArrayPtr<const wchar_t> text);
  // Parses an absolute path as returned by a Win32 API call like GetFinalPathNameByHandle() or
  // GetCurrentDirectory(). A "\\?\" prefix is optional but understood if present.
  //
  // Since such Win32 API calls generally return a length, this function inputs an array slice.
  // The slice should not include any NUL terminator.

private:
  Array<String> parts;

  // TODO(perf): Consider unrolling one element from `parts`, so that a one-element path doesn't
  //   require allocation of an array.

  enum { ALREADY_CHECKED };
  Path(Array<String> parts, decltype(ALREADY_CHECKED));

  friend class PathPtr;

  static String stripNul(String input);
  static void validatePart(StringPtr part);
  static void evalPart(Vector<String>& parts, ArrayPtr<const char> part);
  static Path evalImpl(Vector<String>&& parts, StringPtr path);
  static Path evalWin32Impl(Vector<String>&& parts, StringPtr path, bool fromApi = false);
  static size_t countParts(StringPtr path);
  static size_t countPartsWin32(StringPtr path);
  static bool isWin32Drive(ArrayPtr<const char> part);
  static bool isNetbiosName(ArrayPtr<const char> part);
  static bool isWin32Special(StringPtr part);
};

class PathPtr {
  // Points to a Path or a slice of a Path, but doesn't own it.
  //
  // PathPtr is to Path as ArrayPtr is to Array and StringPtr is to String.

public:
  PathPtr(decltype(nullptr));
  PathPtr(const Path& path);

  Path clone();
  Path append(Path&& suffix) const;
  Path append(PathPtr suffix) const;
  Path append(StringPtr suffix) const;
  Path append(String&& suffix) const;
  Path eval(StringPtr pathText) const;
  PathPtr basename() const;
  PathPtr parent() const;
  String toString(bool absolute = false) const;
  const String& operator[](size_t i) const;
  size_t size() const;
  const String* begin() const;
  const String* end() const;
  PathPtr slice(size_t start, size_t end) const;
  bool operator==(PathPtr other) const;
  bool operator!=(PathPtr other) const;
  bool operator< (PathPtr other) const;
  bool operator> (PathPtr other) const;
  bool operator<=(PathPtr other) const;
  bool operator>=(PathPtr other) const;
  uint hashCode() const;
  bool startsWith(PathPtr prefix) const;
  bool endsWith(PathPtr suffix) const;
  Path evalWin32(StringPtr pathText) const;
  Path evalNative(StringPtr pathText) const;
  String toWin32String(bool absolute = false) const;
  String toNativeString(bool absolute = false) const;
  Array<wchar_t> forWin32Api(bool absolute) const;
  // Equivalent to the corresponding methods of `Path`.

private:
  ArrayPtr<const String> parts;

  explicit PathPtr(ArrayPtr<const String> parts);

  String toWin32StringImpl(bool absolute, bool forApi) const;

  friend class Path;
};

// =======================================================================================
// The filesystem API
//
// This API is strictly synchronous because, unfortunately, there's no such thing as asynchronous
// filesystem access in practice. The filesystem drivers on Linux are written to assume they can
// block. The AIO API is only actually asynchronous for reading/writing the raw file blocks, but if
// the filesystem needs to be involved (to allocate blocks, update metadata, etc.) that will block.
// It's best to imagine that the filesystem is just another tier of memory that happens to be
// slower than RAM (which is slower than L3 cache, which is slower than L2, which is slower than
// L1). You can't do asynchronous RAM access so why asynchronous filesystem? The only way to
// parallelize these is using threads.
//
// All KJ filesystem objects are thread-safe, and so all methods are marked "const" (even write
// methods). Of course, if you concurrently write the same bytes of a file from multiple threads,
// it's unspecified which write will "win".

class FsNode {
  // Base class for filesystem node types.

public:
  Own<const FsNode> clone() const;
  // Creates a new object of exactly the same type as this one, pointing at exactly the same
  // external object.
  //
  // Under the hood, this will call dup(), so the FD number will not be the same.

  virtual Maybe<int> getFd() const { return nullptr; }
  // Get the underlying Unix file descriptor, if any. Returns nullptr if this object actually isn't
  // wrapping a file descriptor.

  virtual Maybe<void*> getWin32Handle() const { return nullptr; }
  // Get the underlying Win32 HANDLE, if any. Returns nullptr if this object actually isn't
  // wrapping a handle.

  enum class Type {
    FILE,
    DIRECTORY,
    SYMLINK,
    BLOCK_DEVICE,
    CHARACTER_DEVICE,
    NAMED_PIPE,
    SOCKET,
    OTHER,
  };

  struct Metadata {
    Type type = Type::FILE;

    uint64_t size = 0;
    // Logical size of the file.

    uint64_t spaceUsed = 0;
    // Physical size of the file on disk. May be smaller for sparse files, or larger for
    // pre-allocated files.

    Date lastModified = UNIX_EPOCH;
    // Last modification time of the file.

    uint linkCount = 1;
    // Number of hard links pointing to this node.

    uint64_t hashCode = 0;
    // Hint which can be used to determine if two FsNode instances point to the same underlying
    // file object. If two FsNodes report different hashCodes, then they are not the same object.
    // If they report the same hashCode, then they may or may not be the same object.
    //
    // The Unix filesystem implementation builds the hashCode based on st_dev and st_ino of
    // `struct stat`. However, note that some filesystems -- especially FUSE-based -- may not fill
    // in st_ino.
    //
    // The Windows filesystem implementation builds the hashCode based on dwVolumeSerialNumber and
    // dwFileIndex{Low,High} of the BY_HANDLE_FILE_INFORMATION structure. However, these are again
    // not guaranteed to be unique on all filesystems. In particular the documentation says that
    // ReFS uses 128-bit identifiers which can't be represented here, and again virtual filesystems
    // may often not report real identifiers.
    //
    // Of course, the process of hashing values into a single hash code can also cause collisions
    // even if the filesystem reports reliable information.
    //
    // Additionally note that this value is not reliable when returned by `lstat()`. You should
    // actually open the object, then call `stat()` on the opened object.

    // Not currently included:
    // - Access control info: Differs wildly across platforms, and KJ prefers capabilities anyway.
    // - Other timestamps: Differs across platforms.
    // - Device number: If you care, you're probably doing platform-specific stuff anyway.

    Metadata() = default;
    Metadata(Type type, uint64_t size, uint64_t spaceUsed, Date lastModified, uint linkCount,
             uint64_t hashCode)
        : type(type), size(size), spaceUsed(spaceUsed), lastModified(lastModified),
          linkCount(linkCount), hashCode(hashCode) {}
    // TODO(cleanup): This constructor is redundant in C++14, but needed in C++11.
  };

  virtual Metadata stat() const = 0;

  virtual void sync() const = 0;
  virtual void datasync() const = 0;
  // Maps to fsync() and fdatasync() system calls.
  //
  // Also, when creating or overwriting a file, the first call to sync() atomically links the file
  // into the filesystem (*after* syncing the data), so than incomplete data is never visible to
  // other processes. (In practice this works by writing into a temporary file and then rename()ing
  // it.)

protected:
  virtual Own<const FsNode> cloneFsNode() const = 0;
  // Implements clone(). Required to return an object with exactly the same type as this one.
  // Hence, every subclass must implement this.
};

class ReadableFile: public FsNode {
public:
  Own<const ReadableFile> clone() const;

  String readAllText() const;
  // Read all text in the file and return as a big string.

  Array<byte> readAllBytes() const;
  // Read all bytes in the file and return as a big byte array.
  //
  // This differs from mmap() in that the read is performed all at once. Future changes to the file
  // do not affect the returned copy. Consider using mmap() instead, particularly for large files.

  virtual size_t read(uint64_t offset, ArrayPtr<byte> buffer) const = 0;
  // Fills `buffer` with data starting at `offset`. Returns the number of bytes actually read --
  // the only time this is less than `buffer.size()` is when EOF occurs mid-buffer.

  virtual Array<const byte> mmap(uint64_t offset, uint64_t size) const = 0;
  // Maps the file to memory read-only. The returned array always has exactly the requested size.
  // Depending on the capabilities of the OS and filesystem, the mapping may or may not reflect
  // changes that happen to the file after mmap() returns.
  //
  // Multiple calls to mmap() on the same file may or may not return the same mapping (it is
  // immutable, so there's no possibility of interference).
  //
  // If the file cannot be mmap()ed, an implementation may choose to allocate a buffer on the heap,
  // read into it, and return that. This should only happen if a real mmap() is impossible.
  //
  // The returned array is always exactly the size requested. However, accessing bytes beyond the
  // current end of the file may raise SIGBUS, or may simply return zero.

  virtual Array<byte> mmapPrivate(uint64_t offset, uint64_t size) const = 0;
  // Like mmap() but returns a view that the caller can modify. Modifications will not be written
  // to the underlying file. Every call to this method returns a unique mapping. Changes made to
  // the underlying file by other clients may or may not be reflected in the mapping -- in fact,
  // some changes may be reflected while others aren't, even within the same mapping.
  //
  // In practice this is often implemented using copy-on-write pages. When you first write to a
  // page, a copy is made. Hence, changes to the underlying file within that page stop being
  // reflected in the mapping.
};

class AppendableFile: public FsNode, public OutputStream {
public:
  Own<const AppendableFile> clone() const;

  // All methods are inherited.
};

class WritableFileMapping {
public:
  virtual ArrayPtr<byte> get() const = 0;
  // Gets the mapped bytes. The returned array can be modified, and those changes may be written to
  // the underlying file, but there is no guarantee that they are written unless you subsequently
  // call changed().

  virtual void changed(ArrayPtr<byte> slice) const = 0;
  // Notifies the implementation that the given bytes have changed. For some implementations this
  // may be a no-op while for others it may be necessary in order for the changes to be written
  // back at all.
  //
  // `slice` must be a slice of `bytes()`.

  virtual void sync(ArrayPtr<byte> slice) const = 0;
  // Implies `changed()`, and then waits until the range has actually been written to disk before
  // returning.
  //
  // `slice` must be a slice of `bytes()`.
  //
  // On Windows, this calls FlushViewOfFile(). The documentation for this function implies that in
  // some circumstances, to fully sync to physical disk, you may need to call FlushFileBuffers() on
  // the file HANDLE as well. The documentation is not very clear on when and why this is needed.
  // If you believe your program needs this, you can accomplish it by calling `.sync()` on the File
  // object after calling `.sync()` on the WritableFileMapping.
};

class File: public ReadableFile {
public:
  Own<const File> clone() const;

  void writeAll(ArrayPtr<const byte> bytes) const;
  void writeAll(StringPtr text) const;
  // Completely replace the file with the given bytes or text.

  virtual void write(uint64_t offset, ArrayPtr<const byte> data) const = 0;
  // Write the given data starting at the given offset in the file.

  virtual void zero(uint64_t offset, uint64_t size) const = 0;
  // Write zeros to the file, starting at `offset` and continuing for `size` bytes. If the platform
  // supports it, this will "punch a hole" in the file, such that blocks that are entirely zeros
  // do not take space on disk.

  virtual void truncate(uint64_t size) const = 0;
  // Set the file end pointer to `size`. If `size` is less than the current size, data past the end
  // is truncated. If `size` is larger than the current size, zeros are added to the end of the
  // file. If the platform supports it, blocks containing all-zeros will not be stored to disk.

  virtual Own<const WritableFileMapping> mmapWritable(uint64_t offset, uint64_t size) const = 0;
  // Like ReadableFile::mmap() but returns a mapping for which any changes will be immediately
  // visible in other mappings of the file on the same system and will eventually be written back
  // to the file.

  virtual size_t copy(uint64_t offset, const ReadableFile& from, uint64_t fromOffset,
                      uint64_t size) const;
  // Copies bytes from one file to another.
  //
  // Copies `size` bytes or to EOF, whichever comes first. Returns the number of bytes actually
  // copied. Hint: Pass kj::maxValue for `size` to always copy to EOF.
  //
  // The copy is not atomic. Concurrent writes may lead to garbage results.
  //
  // The default implementation performs a series of reads and writes. Subclasses can often provide
  // superior implementations that offload the work to the OS or even implement copy-on-write.
};

class ReadableDirectory: public FsNode {
  // Read-only subset of `Directory`.

public:
  Own<const ReadableDirectory> clone() const;

  virtual Array<String> listNames() const = 0;
  // List the contents of this directory. Does NOT include "." nor "..".

  struct Entry {
    FsNode::Type type;
    String name;

    inline bool operator< (const Entry& other) const { return name <  other.name; }
    inline bool operator> (const Entry& other) const { return name >  other.name; }
    inline bool operator<=(const Entry& other) const { return name <= other.name; }
    inline bool operator>=(const Entry& other) const { return name >= other.name; }
    // Convenience comparison operators to sort entries by name.
  };

  virtual Array<Entry> listEntries() const = 0;
  // List the contents of the directory including the type of each file. On some platforms and
  // filesystems, this is just as fast as listNames(), but on others it may require stat()ing each
  // file.

  virtual bool exists(PathPtr path) const = 0;
  // Does the specified path exist?
  //
  // If the path is a symlink, the symlink is followed and the return value indicates if the target
  // exists. If you want to know if the symlink exists, use lstat(). (This implies that listNames()
  // may return names for which exists() reports false.)

  FsNode::Metadata lstat(PathPtr path) const;
  virtual Maybe<FsNode::Metadata> tryLstat(PathPtr path) const = 0;
  // Gets metadata about the path. If the path is a symlink, it is not followed -- the metadata
  // describes the symlink itself. `tryLstat()` returns null if the path doesn't exist.

  Own<const ReadableFile> openFile(PathPtr path) const;
  virtual Maybe<Own<const ReadableFile>> tryOpenFile(PathPtr path) const = 0;
  // Open a file for reading.
  //
  // `tryOpenFile()` returns null if the path doesn't exist. Other errors still throw exceptions.

  Own<const ReadableDirectory> openSubdir(PathPtr path) const;
  virtual Maybe<Own<const ReadableDirectory>> tryOpenSubdir(PathPtr path) const = 0;
  // Opens a subdirectory.
  //
  // `tryOpenSubdir()` returns null if the path doesn't exist. Other errors still throw exceptions.

  String readlink(PathPtr path) const;
  virtual Maybe<String> tryReadlink(PathPtr path) const = 0;
  // If `path` is a symlink, reads and returns the link contents.
  //
  // Note that tryReadlink() differs subtly from tryOpen*(). For example, tryOpenFile() throws if
  // the path is not a file (e.g. if it's a directory); it only returns null if the path doesn't
  // exist at all. tryReadlink() returns null if either the path doesn't exist, or if it does exist
  // but isn't a symlink. This is because if it were to throw instead, then almost every real-world
  // use case of tryReadlink() would be forced to perform an lstat() first for the sole purpose of
  // checking if it is a link, wasting a syscall and a path traversal.
  //
  // See Directory::symlink() for warnings about symlinks.
};

enum class WriteMode {
  // Mode for opening a file (or directory) for write.
  //
  // (To open a file or directory read-only, do not specify a mode.)
  //
  // WriteMode is a bitfield. Hence, it overloads the bitwise logic operators. To check if a
  // particular bit is set in a bitfield, use kj::has(), like:
  //
  //     if (kj::has(mode, WriteMode::MUST_EXIST)) {
  //       requireExists(path);
  //     }
  //
  // (`if (mode & WriteMode::MUST_EXIST)` doesn't work because WriteMode is an enum class, which
  // cannot be converted to bool. Alas, C++ does not allow you to define a conversion operator
  // on an enum type, so we can't define a conversion to bool.)

  // -----------------------------------------
  // Core flags
  //
  // At least one of CREATE or MODIFY must be specified. Optionally, the two flags can be combined
  // with a bitwise-OR.

  CREATE = 1,
  // Create a new empty file.
  //
  // When not combined with MODIFY, if the file already exists (including as a broken symlink),
  // tryOpenFile() returns null (and openFile() throws).
  //
  // When combined with MODIFY, if the path already exists, it will be opened as if CREATE hadn't
  // been specified at all. If the path refers to a broken symlink, the file at the target of the
  // link will be created (if its parent directory exists).

  MODIFY = 2,
  // Modify an existing file.
  //
  // When not combined with CREATE, if the file doesn't exist (including if it is a broken symlink),
  // tryOpenFile() returns null (and openFile() throws).
  //
  // When combined with CREATE, if the path doesn't exist, it will be created as if MODIFY hadn't
  // been specified at all. If the path refers to a broken symlink, the file at the target of the
  // link will be created (if its parent directory exists).

  // -----------------------------------------
  // Additional flags
  //
  // Any number of these may be OR'd with the core flags.

  CREATE_PARENT = 4,
  // Indicates that if the target node's parent directory doesn't exist, it should be created
  // automatically, along with its parent, and so on. This creation is NOT atomic.
  //
  // This bit only makes sense with CREATE or REPLACE.

  EXECUTABLE = 8,
  // Mark this file executable, if this is a meaningful designation on the host platform.

  PRIVATE = 16,
  // Indicates that this file is sensitive and should have permissions masked so that it is only
  // accessible by the current user.
  //
  // When this is not used, the platform's default access control settings are used. On Unix,
  // that usually means the umask is applied. On Windows, it means permissions are inherited from
  // the parent.
};

inline constexpr WriteMode operator|(WriteMode a, WriteMode b) {
  return static_cast<WriteMode>(static_cast<uint>(a) | static_cast<uint>(b));
}
inline constexpr WriteMode operator&(WriteMode a, WriteMode b) {
  return static_cast<WriteMode>(static_cast<uint>(a) & static_cast<uint>(b));
}
inline constexpr WriteMode operator+(WriteMode a, WriteMode b) {
  return static_cast<WriteMode>(static_cast<uint>(a) | static_cast<uint>(b));
}
inline constexpr WriteMode operator-(WriteMode a, WriteMode b) {
  return static_cast<WriteMode>(static_cast<uint>(a) & ~static_cast<uint>(b));
}
template <typename T, typename = EnableIf<__is_enum(T)>>
bool has(T haystack, T needle) {
  return (static_cast<__underlying_type(T)>(haystack) &
          static_cast<__underlying_type(T)>(needle)) ==
          static_cast<__underlying_type(T)>(needle);
}

enum class TransferMode {
  // Specifies desired behavior for Directory::transfer().

  MOVE,
  // The node is moved to the new location, i.e. the old location is deleted. If possible, this
  // move is performed without copying, otherwise it is performed as a copy followed by a delete.

  LINK,
  // The new location becomes a synonym for the old location (a "hard link"). Filesystems have
  // varying support for this -- typically, it is not supported on directories.

  COPY
  // The new location becomes a copy of the old.
  //
  // Some filesystems may implement this in terms of copy-on-write.
  //
  // If the filesystem supports sparse files, COPY takes sparseness into account -- it will punch
  // holes in the target file where holes exist in the source file.
};

class Directory: public ReadableDirectory {
  // Refers to a specific directory on disk.
  //
  // A `Directory` object *only* provides access to children of the directory, not parents. That
  // is, you cannot open the file "..", nor jump to the root directory with "/".
  //
  // On OSs that support it, a `Directory` is backed by an open handle to the directory node. This
  // means:
  // - If the directory is renamed on-disk, the `Directory` object still points at it.
  // - Opening files in the directory only requires the OS to traverse the path from the directory
  //   to the file; it doesn't have to re-traverse all the way from the filesystem root.
  //
  // On Windows, a `Directory` object holds a lock on the underlying directory such that it cannot
  // be renamed nor deleted while the object exists. This is necessary because Windows does not
  // fully support traversing paths relative to file handles (it does for some operations but not
  // all), so the KJ filesystem implementation is forced to remember the full path and needs to
  // ensure that the path is not invalidated. If, in the future, Windows fully supports
  // handle-relative paths, KJ may stop locking directories in this way, so do not rely on this
  // behavior.

public:
  Own<const Directory> clone() const;

  template <typename T>
  class Replacer {
    // Implements an atomic replacement of a file or directory, allowing changes to be made to
    // storage in a way that avoids losing data in a power outage and prevents other processes
    // from observing content in an inconsistent state.
    //
    // `T` may be `File` or `Directory`. For readability, the text below describes replacing a
    // file, but the logic is the same for directories.
    //
    // When you call `Directory::replaceFile()`, a temporary file is created, but the specified
    // path is not yet touched. You may call `get()` to obtain the temporary file object, through
    // which you may initialize its content, knowing that no other process can see it yet. The file
    // is atomically moved to its final path when you call `commit()`. If you destroy the Replacer
    // without calling commit(), the temporary file is deleted.
    //
    // Note that most operating systems sadly do not support creating a truly unnamed temporary file
    // and then linking it in later. Moreover, the file cannot necessarily be created in the system
    // temporary directory because it might not be on the same filesystem as the target. Therefore,
    // the replacement file may initially be created in the same directory as its eventual target.
    // The implementation of Directory will choose a name that is unique and "hidden" according to
    // the conventions of the filesystem. Additionally, the implementation of Directory will avoid
    // returning these temporary files from its list*() methods, in order to avoid observable
    // inconsistencies across platforms.
  public:
    explicit Replacer(WriteMode mode);

    virtual const T& get() = 0;
    // Gets the File or Directory representing the replacement data. Fill in this object before
    // calling commit().

    void commit();
    virtual bool tryCommit() = 0;
    // Commit the replacement.
    //
    // `tryCommit()` may return false based on the CREATE/MODIFY bits passed as the WriteMode when
    // the replacement was initiated. (If CREATE but not MODIFY was used, tryCommit() returns
    // false to indicate that the target file already existed. If MODIFY but not CREATE was used,
    // tryCommit() returns false to indicate that the file didn't exist.)
    //
    // `commit()` is atomic, meaning that there is no point in time at which other processes
    // observing the file will see it in an intermediate state -- they will either see the old
    // content or the complete new content. This includes in the case of a power outage or machine
    // failure: on recovery, the file will either be in the old state or the new state, but not in
    // some intermediate state.
    //
    // It's important to note that a power failure *after commit() returns* can still revert the
    // file to its previous state. That is, `commit()` does NOT guarantee that, upon return, the
    // new content is durable. In order to guarantee this, you must call `sync()` on the immediate
    // parent directory of the replaced file.
    //
    // Note that, sadly, not all filesystems / platforms are capable of supporting all of the
    // guarantees documented above. In such cases, commit() will make a best-effort attempt to do
    // what it claims. Some examples of possible problems include:
    // - Any guarantees about durability through a power outage probably require a journaling
    //   filesystem.
    // - Many platforms do not support atomically replacing a non-empty directory. Linux does as
    //   of kernel 3.15 (via the renameat2() syscall using RENAME_EXCHANGE). Where not supported,
    //   the old directory will be moved away just before the replacement is moved into place.
    // - Many platforms do not support atomically requiring the existence or non-existence of a
    //   file before replacing it. In these cases, commit() may have to perform the check as a
    //   separate step, with a small window for a race condition.
    // - Many platforms do not support "unlinking" a non-empty directory, meaning that a replaced
    //   directory will need to be deconstructed by deleting all contents. If another process has
    //   the directory open when it is replaced, that process will observe the contents
    //   disappearing after the replacement (actually, a swap) has taken place. This differs from
    //   files, where a process that has opened a file before it is replaced will continue see the
    //   file's old content unchanged after the replacement.
    // - On Windows, there are multiple ways to replace one file with another in a single system
    //   call, but none are documented as being atomic. KJ always uses `MoveFileEx()` with
    //   MOVEFILE_REPLACE_EXISTING. While the alternative `ReplaceFile()` is attractive for many
    //   reasons, it has the critical problem that it cannot be used when the source file has open
    //   file handles, which is generally the case when using Replacer.

  protected:
    const WriteMode mode;
  };

  using ReadableDirectory::openFile;
  using ReadableDirectory::openSubdir;
  using ReadableDirectory::tryOpenFile;
  using ReadableDirectory::tryOpenSubdir;

  Own<const File> openFile(PathPtr path, WriteMode mode) const;
  virtual Maybe<Own<const File>> tryOpenFile(PathPtr path, WriteMode mode) const = 0;
  // Open a file for writing.
  //
  // `tryOpenFile()` returns null if the path is required to exist but doesn't (MODIFY or REPLACE)
  // or if the path is required not to exist but does (CREATE or RACE). These are the only cases
  // where it returns null -- all other types of errors (like "access denied") throw exceptions.

  virtual Own<Replacer<File>> replaceFile(PathPtr path, WriteMode mode) const = 0;
  // Construct a file which, when ready, will be atomically moved to `path`, replacing whatever
  // is there already. See `Replacer<T>` for detalis.
  //
  // The `CREATE` and `MODIFY` bits of `mode` are not enforced until commit time, hence
  // `replaceFile()` has no "try" variant.

  virtual Own<const File> createTemporary() const = 0;
  // Create a temporary file backed by this directory's filesystem, but which isn't linked into
  // the directory tree. The file is deleted from disk when all references to it have been dropped.

  Own<AppendableFile> appendFile(PathPtr path, WriteMode mode) const;
  virtual Maybe<Own<AppendableFile>> tryAppendFile(PathPtr path, WriteMode mode) const = 0;
  // Opens the file for appending only. Useful for log files.
  //
  // If the underlying filesystem supports it, writes to the file will always be appended even if
  // other writers are writing to the same file at the same time -- however, some implementations
  // may instead assume that no other process is changing the file size between writes.

  Own<const Directory> openSubdir(PathPtr path, WriteMode mode) const;
  virtual Maybe<Own<const Directory>> tryOpenSubdir(PathPtr path, WriteMode mode) const = 0;
  // Opens a subdirectory for writing.

  virtual Own<Replacer<Directory>> replaceSubdir(PathPtr path, WriteMode mode) const = 0;
  // Construct a directory which, when ready, will be atomically moved to `path`, replacing
  // whatever is there already. See `Replacer<T>` for detalis.
  //
  // The `CREATE` and `MODIFY` bits of `mode` are not enforced until commit time, hence
  // `replaceSubdir()` has no "try" variant.

  void symlink(PathPtr linkpath, StringPtr content, WriteMode mode) const;
  virtual bool trySymlink(PathPtr linkpath, StringPtr content, WriteMode mode) const = 0;
  // Create a symlink. `content` is the raw text which will be written into the symlink node.
  // How this text is interpreted is entirely dependent on the filesystem. Note in particular that:
  // - Windows will require a path that uses backslashes as the separator.
  // - InMemoryDirectory does not support symlinks containing "..".
  //
  // Unfortunately under many implementations symlink() can be used to break out of the directory
  // by writing an absolute path or utilizing "..". Do not call this method with a value for
  // `target` that you don't trust.
  //
  // `mode` must be CREATE or REPLACE, not MODIFY. CREATE_PARENT is honored but EXECUTABLE and
  // PRIVATE have no effect. `trySymlink()` returns false in CREATE mode when the target already
  // exists.

  void transfer(PathPtr toPath, WriteMode toMode,
                PathPtr fromPath, TransferMode mode) const;
  void transfer(PathPtr toPath, WriteMode toMode,
                const Directory& fromDirectory, PathPtr fromPath,
                TransferMode mode) const;
  virtual bool tryTransfer(PathPtr toPath, WriteMode toMode,
                           const Directory& fromDirectory, PathPtr fromPath,
                           TransferMode mode) const;
  virtual Maybe<bool> tryTransferTo(const Directory& toDirectory, PathPtr toPath, WriteMode toMode,
                                    PathPtr fromPath, TransferMode mode) const;
  // Move, link, or copy a file/directory tree from one location to another.
  //
  // Filesystems vary in what kinds of transfers are allowed, especially for TransferMode::LINK,
  // and whether TransferMode::MOVE is implemented as an actual move vs. copy+delete.
  //
  // tryTransfer() returns false if the source location didn't exist, or when `toMode` is CREATE
  // and the target already exists. The default implementation implements only TransferMode::COPY.
  //
  // tryTransferTo() exists to implement double-dispatch. It should be called as a fallback by
  // implementations of tryTransfer() in cases where the target directory would otherwise fail or
  // perform a pessimal transfer. The default implementation returns nullptr, which the caller
  // should interpret as: "I don't have any special optimizations; do the obvious thing."
  //
  // `toMode` controls how the target path is created. CREATE_PARENT is honored but EXECUTABLE and
  // PRIVATE have no effect.

  void remove(PathPtr path) const;
  virtual bool tryRemove(PathPtr path) const = 0;
  // Deletes/unlinks the given path. If the path names a directory, it is recursively deleted.
  //
  // tryRemove() returns false in the specific case that the path doesn't exist. remove() would
  // throw in this case. In all other error cases (like "access denied"), tryRemove() still throws;
  // it is only "does not exist" that produces a false return.
  //
  // WARNING: The Windows implementation of recursive deletion is currently not safe to call from a
  //   privileged process to delete directories writable by unprivileged users, due to a race
  //   condition in which the user could trick the algorithm into following a symlink and deleting
  //   everything at the destination. This race condition is not present in the Unix
  //   implementation. Fixing it for Windows would require rewriting a lot of code to use different
  //   APIs. If you're interested, see the TODO(security) in filesystem-disk-win32.c++.

  // TODO(someday):
  // - Support sockets? There's no openat()-like interface for sockets, so it's hard to support
  //   them currently. Also you'd probably want to use them with the async library.
  // - Support named pipes? Unclear if there's a use case that isn't better-served by sockets.
  //   Then again, they can be openat()ed.
  // - Support watching for changes (inotify). Probably also requires the async library. Also
  //   lacks openat()-like semantics.
  // - xattrs -- linux-specific
  // - chown/chmod/etc. -- unix-specific, ACLs, eww
  // - set timestamps -- only needed by archiving programs/
  // - advisory locks
  // - sendfile?
  // - fadvise and such

private:
  static void commitFailed(WriteMode mode);
};

class Filesystem {
public:
  virtual const Directory& getRoot() const = 0;
  // Get the filesystem's root directory, as of the time the Filesystem object was created.

  virtual const Directory& getCurrent() const = 0;
  // Get the filesystem's current directory, as of the time the Filesystem object was created.

  virtual PathPtr getCurrentPath() const = 0;
  // Get the path from the root to the current directory, as of the time the Filesystem object was
  // created. Note that because a `Directory` does not provide access to its parent, if you want to
  // follow `..` from the current directory, you must use `getCurrentPath().eval("..")` or
  // `getCurrentPath().parent()`.
  //
  // This function attempts to determine the path as it appeared in the user's shell before this
  // program was started. That means, if the user had `cd`ed into a symlink, the path through that
  // symlink is returned, *not* the canonical path.
  //
  // Because of this, there is an important difference between how the operating system interprets
  // "../foo" and what you get when you write `getCurrentPath().eval("../foo")`: The former
  // will interpret ".." relative to the directory's canonical path, whereas the latter will
  // interpret it relative to the path shown in the user's shell. In practice, the latter is
  // almost always what the user wants! But the former behavior is what almost all commands do
  // in practice, and it leads to confusion. KJ commands should implement the behavior the user
  // expects.
};

// =======================================================================================

Own<File> newInMemoryFile(const Clock& clock);
Own<Directory> newInMemoryDirectory(const Clock& clock);
// Construct file and directory objects which reside in-memory.
//
// InMemoryFile has the following special properties:
// - The backing store is not sparse and never gets smaller even if you truncate the file.
// - While a non-private memory mapping exists, the backing store cannot get larger. Any operation
//   which would expand it will throw.
//
// InMemoryDirectory has the following special properties:
// - Symlinks are processed using Path::parse(). This implies that a symlink cannot point to a
//   parent directory -- InMemoryDirectory does not know its parent.
// - link() can link directory nodes in addition to files.
// - link() and rename() accept any kind of Directory as `fromDirectory` -- it doesn't need to be
//   another InMemoryDirectory. However, for rename(), the from path must be a directory.

Own<AppendableFile> newFileAppender(Own<const File> inner);
// Creates an AppendableFile by wrapping a File. Note that this implementation assumes it is the
// only writer. A correct implementation should always append to the file even if other writes
// are happening simultaneously, as is achieved with the O_APPEND flag to open(2), but that
// behavior is not possible to emulate on top of `File`.

#if _WIN32
typedef AutoCloseHandle OsFileHandle;
#else
typedef AutoCloseFd OsFileHandle;
#endif

Own<ReadableFile> newDiskReadableFile(OsFileHandle fd);
Own<AppendableFile> newDiskAppendableFile(OsFileHandle fd);
Own<File> newDiskFile(OsFileHandle fd);
Own<ReadableDirectory> newDiskReadableDirectory(OsFileHandle fd);
Own<Directory> newDiskDirectory(OsFileHandle fd);
// Wrap a file descriptor (or Windows HANDLE) as various filesystem types.

Own<Filesystem> newDiskFilesystem();
// Get at implementation of `Filesystem` representing the real filesystem.
//
// DO NOT CALL THIS except at the top level of your program, e.g. in main(). Anywhere else, you
// should instead have your caller pass in a Filesystem object, or a specific Directory object,
// or whatever it is that your code needs. This ensures that your code supports dependency
// injection, which makes it more reusable and testable.
//
// newDiskFilesystem() reads the current working directory at the time it is called. The returned
// object is not affected by subsequent calls to chdir().

// =======================================================================================
// inline implementation details

inline Path::Path(decltype(nullptr)): parts(nullptr) {}
inline Path::Path(std::initializer_list<StringPtr> parts)
    : Path(arrayPtr(parts.begin(), parts.end())) {}
inline Path::Path(Array<String> parts, decltype(ALREADY_CHECKED))
    : parts(kj::mv(parts)) {}
inline Path Path::clone() const { return PathPtr(*this).clone(); }
inline Path Path::append(Path&& suffix) const& { return PathPtr(*this).append(kj::mv(suffix)); }
inline Path Path::append(PathPtr suffix) const& { return PathPtr(*this).append(suffix); }
inline Path Path::append(StringPtr suffix) const& { return append(Path(suffix)); }
inline Path Path::append(StringPtr suffix) && { return kj::mv(*this).append(Path(suffix)); }
inline Path Path::append(String&& suffix) const& { return append(Path(kj::mv(suffix))); }
inline Path Path::append(String&& suffix) && { return kj::mv(*this).append(Path(kj::mv(suffix))); }
inline Path Path::eval(StringPtr pathText) const& { return PathPtr(*this).eval(pathText); }
inline PathPtr Path::basename() const& { return PathPtr(*this).basename(); }
inline PathPtr Path::parent() const& { return PathPtr(*this).parent(); }
inline const String& Path::operator[](size_t i) const& { return parts[i]; }
inline String Path::operator[](size_t i) && { return kj::mv(parts[i]); }
inline size_t Path::size() const { return parts.size(); }
inline const String* Path::begin() const { return parts.begin(); }
inline const String* Path::end() const { return parts.end(); }
inline PathPtr Path::slice(size_t start, size_t end) const& {
  return PathPtr(*this).slice(start, end);
}
inline bool Path::operator==(PathPtr other) const { return PathPtr(*this) == other; }
inline bool Path::operator!=(PathPtr other) const { return PathPtr(*this) != other; }
inline bool Path::operator< (PathPtr other) const { return PathPtr(*this) <  other; }
inline bool Path::operator> (PathPtr other) const { return PathPtr(*this) >  other; }
inline bool Path::operator<=(PathPtr other) const { return PathPtr(*this) <= other; }
inline bool Path::operator>=(PathPtr other) const { return PathPtr(*this) >= other; }
inline bool Path::operator==(const Path& other) const { return PathPtr(*this) == PathPtr(other); }
inline bool Path::operator!=(const Path& other) const { return PathPtr(*this) != PathPtr(other); }
inline bool Path::operator< (const Path& other) const { return PathPtr(*this) <  PathPtr(other); }
inline bool Path::operator> (const Path& other) const { return PathPtr(*this) >  PathPtr(other); }
inline bool Path::operator<=(const Path& other) const { return PathPtr(*this) <= PathPtr(other); }
inline bool Path::operator>=(const Path& other) const { return PathPtr(*this) >= PathPtr(other); }
inline uint Path::hashCode() const { return kj::hashCode(parts); }
inline bool Path::startsWith(PathPtr prefix) const { return PathPtr(*this).startsWith(prefix); }
inline bool Path::endsWith  (PathPtr suffix) const { return PathPtr(*this).endsWith  (suffix); }
inline String Path::toString(bool absolute) const { return PathPtr(*this).toString(absolute); }
inline Path Path::evalWin32(StringPtr pathText) const& {
  return PathPtr(*this).evalWin32(pathText);
}
inline String Path::toWin32String(bool absolute) const {
  return PathPtr(*this).toWin32String(absolute);
}
inline Array<wchar_t> Path::forWin32Api(bool absolute) const {
  return PathPtr(*this).forWin32Api(absolute);
}

inline PathPtr::PathPtr(decltype(nullptr)): parts(nullptr) {}
inline PathPtr::PathPtr(const Path& path): parts(path.parts) {}
inline PathPtr::PathPtr(ArrayPtr<const String> parts): parts(parts) {}
inline Path PathPtr::append(StringPtr suffix) const { return append(Path(suffix)); }
inline Path PathPtr::append(String&& suffix) const { return append(Path(kj::mv(suffix))); }
inline const String& PathPtr::operator[](size_t i) const { return parts[i]; }
inline size_t PathPtr::size() const { return parts.size(); }
inline const String* PathPtr::begin() const { return parts.begin(); }
inline const String* PathPtr::end() const { return parts.end(); }
inline PathPtr PathPtr::slice(size_t start, size_t end) const {
  return PathPtr(parts.slice(start, end));
}
inline bool PathPtr::operator!=(PathPtr other) const { return !(*this == other); }
inline bool PathPtr::operator> (PathPtr other) const { return other < *this; }
inline bool PathPtr::operator<=(PathPtr other) const { return !(other < *this); }
inline bool PathPtr::operator>=(PathPtr other) const { return !(*this < other); }
inline uint PathPtr::hashCode() const { return kj::hashCode(parts); }
inline String PathPtr::toWin32String(bool absolute) const {
  return toWin32StringImpl(absolute, false);
}

#if _WIN32
inline Path Path::evalNative(StringPtr pathText) const& {
  return evalWin32(pathText);
}
inline Path Path::evalNative(StringPtr pathText) && {
  return kj::mv(*this).evalWin32(pathText);
}
inline String Path::toNativeString(bool absolute) const {
  return toWin32String(absolute);
}
inline Path PathPtr::evalNative(StringPtr pathText) const {
  return evalWin32(pathText);
}
inline String PathPtr::toNativeString(bool absolute) const {
  return toWin32String(absolute);
}
#else
inline Path Path::evalNative(StringPtr pathText) const& {
  return eval(pathText);
}
inline Path Path::evalNative(StringPtr pathText) && {
  return kj::mv(*this).eval(pathText);
}
inline String Path::toNativeString(bool absolute) const {
  return toString(absolute);
}
inline Path PathPtr::evalNative(StringPtr pathText) const {
  return eval(pathText);
}
inline String PathPtr::toNativeString(bool absolute) const {
  return toString(absolute);
}
#endif  // _WIN32, else

inline Own<const FsNode> FsNode::clone() const { return cloneFsNode(); }
inline Own<const ReadableFile> ReadableFile::clone() const {
  return cloneFsNode().downcast<const ReadableFile>();
}
inline Own<const AppendableFile> AppendableFile::clone() const {
  return cloneFsNode().downcast<const AppendableFile>();
}
inline Own<const File> File::clone() const { return cloneFsNode().downcast<const File>(); }
inline Own<const ReadableDirectory> ReadableDirectory::clone() const {
  return cloneFsNode().downcast<const ReadableDirectory>();
}
inline Own<const Directory> Directory::clone() const {
  return cloneFsNode().downcast<const Directory>();
}

inline void Directory::transfer(
    PathPtr toPath, WriteMode toMode, PathPtr fromPath, TransferMode mode) const {
  return transfer(toPath, toMode, *this, fromPath, mode);
}

template <typename T>
inline Directory::Replacer<T>::Replacer(WriteMode mode): mode(mode) {}

template <typename T>
void Directory::Replacer<T>::commit() {
  if (!tryCommit()) commitFailed(mode);
}

} // namespace kj

KJ_END_HEADER
