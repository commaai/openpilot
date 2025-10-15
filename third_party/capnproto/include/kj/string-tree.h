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

#include "string.h"

KJ_BEGIN_HEADER

namespace kj {

class StringTree {
  // A long string, represented internally as a tree of strings.  This data structure is like a
  // String, but optimized for concatenation and iteration at the expense of seek time.  The
  // structure is intended to be used for building large text blobs from many small pieces, where
  // repeatedly concatenating smaller strings into larger ones would waste copies.  This structure
  // is NOT intended for use cases requiring random access or computing substrings.  For those,
  // you should use a Rope, which is a much more complicated data structure.
  //
  // The proper way to construct a StringTree is via kj::strTree(...), which works just like
  // kj::str(...) but returns a StringTree rather than a String.
  //
  // KJ_STRINGIFY() functions that construct large strings from many smaller strings are encouraged
  // to return StringTree rather than a flat char container.

public:
  inline StringTree(): size_(0) {}
  inline StringTree(String&& text): size_(text.size()), text(kj::mv(text)) {}

  StringTree(Array<StringTree>&& pieces, StringPtr delim);
  // Build a StringTree by concatenating the given pieces, delimited by the given delimiter
  // (e.g. ", ").

  inline size_t size() const { return size_; }

  template <typename Func>
  void visit(Func&& func) const;

  String flatten() const;
  // Return the contents as a string.

  // TODO(someday):  flatten() when *this is an rvalue and when branches.size() == 0 could simply
  //   return `kj::mv(text)`.  Requires reference qualifiers (Clang 3.3 / GCC 4.8).

  char* flattenTo(char* __restrict__ target) const;
  char* flattenTo(char* __restrict__ target, char* limit) const;
  // Copy the contents to the given character array.  Does not add a NUL terminator. Returns a
  // pointer just past the end of what was filled.

private:
  size_t size_;
  String text;

  struct Branch;
  Array<Branch> branches;  // In order.

  inline void fill(char* pos, size_t branchIndex);
  template <typename First, typename... Rest>
  void fill(char* pos, size_t branchIndex, First&& first, Rest&&... rest);
  template <typename... Rest>
  void fill(char* pos, size_t branchIndex, StringTree&& first, Rest&&... rest);
  template <typename... Rest>
  void fill(char* pos, size_t branchIndex, Array<char>&& first, Rest&&... rest);
  template <typename... Rest>
  void fill(char* pos, size_t branchIndex, String&& first, Rest&&... rest);

  template <typename... Params>
  static StringTree concat(Params&&... params);
  static StringTree&& concat(StringTree&& param) { return kj::mv(param); }

  template <typename T>
  static inline size_t flatSize(const T& t) { return t.size(); }
  static inline size_t flatSize(String&& s) { return 0; }
  static inline size_t flatSize(StringTree&& s) { return 0; }

  template <typename T>
  static inline size_t branchCount(const T& t) { return 0; }
  static inline size_t branchCount(String&& s) { return 1; }
  static inline size_t branchCount(StringTree&& s) { return 1; }

  template <typename... Params>
  friend StringTree strTree(Params&&... params);
};

inline StringTree&& KJ_STRINGIFY(StringTree&& tree) { return kj::mv(tree); }
inline const StringTree& KJ_STRINGIFY(const StringTree& tree) { return tree; }

inline StringTree KJ_STRINGIFY(Array<StringTree>&& trees) { return StringTree(kj::mv(trees), ""); }

template <typename... Params>
StringTree strTree(Params&&... params);
// Build a StringTree by stringifying the given parameters and concatenating the results.
// If any of the parameters stringify to StringTree rvalues, they will be incorporated as
// branches to avoid a copy.

// =======================================================================================
// Inline implementation details

namespace _ {  // private

template <typename... Rest>
char* fill(char* __restrict__ target, const StringTree& first, Rest&&... rest) {
  // Make str() work with stringifiers that return StringTree by patching fill().

  first.flattenTo(target);
  return fill(target + first.size(), kj::fwd<Rest>(rest)...);
}

template <typename... Rest>
char* fillLimited(char* __restrict__ target, char* limit, const StringTree& first, Rest&&... rest) {
  // Make str() work with stringifiers that return StringTree by patching fill().

  target = first.flattenTo(target, limit);
  return fillLimited(target + first.size(), limit, kj::fwd<Rest>(rest)...);
}

template <typename T> constexpr bool isStringTree() { return false; }
template <> constexpr bool isStringTree<StringTree>() { return true; }

inline StringTree&& toStringTreeOrCharSequence(StringTree&& tree) { return kj::mv(tree); }
inline StringTree toStringTreeOrCharSequence(String&& str) { return StringTree(kj::mv(str)); }

template <typename T>
inline auto toStringTreeOrCharSequence(T&& value)
    -> decltype(toCharSequence(kj::fwd<T>(value))) {
  static_assert(!isStringTree<Decay<T>>(),
      "When passing a StringTree into kj::strTree(), either pass it by rvalue "
      "(use kj::mv(value)) or explicitly call value.flatten() to make a copy.");

  return toCharSequence(kj::fwd<T>(value));
}

}  // namespace _ (private)

struct StringTree::Branch {
  size_t index;
  // Index in `text` where this branch should be inserted.

  StringTree content;
};

template <typename Func>
void StringTree::visit(Func&& func) const {
  size_t pos = 0;
  for (auto& branch: branches) {
    if (branch.index > pos) {
      func(text.slice(pos, branch.index));
      pos = branch.index;
    }
    branch.content.visit(func);
  }
  if (text.size() > pos) {
    func(text.slice(pos, text.size()));
  }
}

inline void StringTree::fill(char* pos, size_t branchIndex) {
  KJ_IREQUIRE(pos == text.end() && branchIndex == branches.size(),
              kj::str(text.end() - pos, ' ', branches.size() - branchIndex).cStr());
}

template <typename First, typename... Rest>
void StringTree::fill(char* pos, size_t branchIndex, First&& first, Rest&&... rest) {
  pos = _::fill(pos, kj::fwd<First>(first));
  fill(pos, branchIndex, kj::fwd<Rest>(rest)...);
}

template <typename... Rest>
void StringTree::fill(char* pos, size_t branchIndex, StringTree&& first, Rest&&... rest) {
  branches[branchIndex].index = pos - text.begin();
  branches[branchIndex].content = kj::mv(first);
  fill(pos, branchIndex + 1, kj::fwd<Rest>(rest)...);
}

template <typename... Rest>
void StringTree::fill(char* pos, size_t branchIndex, String&& first, Rest&&... rest) {
  branches[branchIndex].index = pos - text.begin();
  branches[branchIndex].content = StringTree(kj::mv(first));
  fill(pos, branchIndex + 1, kj::fwd<Rest>(rest)...);
}

template <typename... Params>
StringTree StringTree::concat(Params&&... params) {
  StringTree result;
  result.size_ = _::sum({params.size()...});
  result.text = heapString(
      _::sum({StringTree::flatSize(kj::fwd<Params>(params))...}));
  result.branches = heapArray<StringTree::Branch>(
      _::sum({StringTree::branchCount(kj::fwd<Params>(params))...}));
  result.fill(result.text.begin(), 0, kj::fwd<Params>(params)...);
  return result;
}

template <typename... Params>
StringTree strTree(Params&&... params) {
  return StringTree::concat(_::toStringTreeOrCharSequence(kj::fwd<Params>(params))...);
}

}  // namespace kj

KJ_END_HEADER
