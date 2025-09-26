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

// Parser combinator framework!
//
// This file declares several functions which construct parsers, usually taking other parsers as
// input, thus making them parser combinators.
//
// A valid parser is any functor which takes a reference to an input cursor (defined below) as its
// input and returns a Maybe.  The parser returns null on parse failure, or returns the parsed
// result on success.
//
// An "input cursor" is any type which implements the same interface as IteratorInput, below.  Such
// a type acts as a pointer to the current input location.  When a parser returns successfully, it
// will have updated the input cursor to point to the position just past the end of what was parsed.
// On failure, the cursor position is unspecified.

#pragma once

#include "../common.h"
#include "../memory.h"
#include "../array.h"
#include "../tuple.h"
#include "../vector.h"

#if _MSC_VER && _MSC_VER < 1920 && !__clang__
#define KJ_MSVC_BROKEN_DECLTYPE 1
#endif

#if KJ_MSVC_BROKEN_DECLTYPE
#include <type_traits>  // result_of_t
#endif

KJ_BEGIN_HEADER

namespace kj {
namespace parse {

template <typename Element, typename Iterator>
class IteratorInput {
  // A parser input implementation based on an iterator range.

public:
  IteratorInput(Iterator begin, Iterator end)
      : parent(nullptr), pos(begin), end(end), best(begin) {}
  explicit IteratorInput(IteratorInput& parent)
      : parent(&parent), pos(parent.pos), end(parent.end), best(parent.pos) {}
  ~IteratorInput() {
    if (parent != nullptr) {
      parent->best = kj::max(kj::max(pos, best), parent->best);
    }
  }
  KJ_DISALLOW_COPY_AND_MOVE(IteratorInput);

  void advanceParent() {
    parent->pos = pos;
  }
  void forgetParent() {
    parent = nullptr;
  }

  bool atEnd() { return pos == end; }
  auto current() -> decltype(*instance<Iterator>()) {
    KJ_IREQUIRE(!atEnd());
    return *pos;
  }
  auto consume() -> decltype(*instance<Iterator>()) {
    KJ_IREQUIRE(!atEnd());
    return *pos++;
  }
  void next() {
    KJ_IREQUIRE(!atEnd());
    ++pos;
  }

  Iterator getBest() { return kj::max(pos, best); }

  Iterator getPosition() { return pos; }

private:
  IteratorInput* parent;
  Iterator pos;
  Iterator end;
  Iterator best;  // furthest we got with any sub-input
};

template <typename T> struct OutputType_;
template <typename T> struct OutputType_<Maybe<T>> { typedef T Type; };
template <typename Parser, typename Input>
using OutputType = typename OutputType_<
#if KJ_MSVC_BROKEN_DECLTYPE
    std::result_of_t<Parser(Input)>
    // The instance<T&>() based version below results in many compiler errors on MSVC2017.
#else
    decltype(instance<Parser&>()(instance<Input&>()))
#endif
    >::Type;
// Synonym for the output type of a parser, given the parser type and the input type.

// =======================================================================================

template <typename Input, typename Output>
class ParserRef {
  // Acts as a reference to some other parser, with simplified type.  The referenced parser
  // is polymorphic by virtual call rather than templates.  For grammars of non-trivial size,
  // it is important to inject refs into the grammar here and there to prevent the parser types
  // from becoming ridiculous.  Using too many of them can hurt performance, though.

public:
  ParserRef(): parser(nullptr), wrapper(nullptr) {}
  ParserRef(const ParserRef&) = default;
  ParserRef(ParserRef&&) = default;
  ParserRef& operator=(const ParserRef& other) = default;
  ParserRef& operator=(ParserRef&& other) = default;

  template <typename Other>
  constexpr ParserRef(Other&& other)
      : parser(&other), wrapper(&WrapperImplInstance<Decay<Other>>::instance) {
    static_assert(kj::isReference<Other>(), "ParserRef should not be assigned to a temporary.");
  }

  template <typename Other>
  inline ParserRef& operator=(Other&& other) {
    static_assert(kj::isReference<Other>(), "ParserRef should not be assigned to a temporary.");
    parser = &other;
    wrapper = &WrapperImplInstance<Decay<Other>>::instance;
    return *this;
  }

  KJ_ALWAYS_INLINE(Maybe<Output> operator()(Input& input) const) {
    // Always inline in the hopes that this allows branch prediction to kick in so the virtual call
    // doesn't hurt so much.
    return wrapper->parse(parser, input);
  }

private:
  struct Wrapper {
    virtual Maybe<Output> parse(const void* parser, Input& input) const = 0;
  };
  template <typename ParserImpl>
  struct WrapperImpl: public Wrapper {
    Maybe<Output> parse(const void* parser, Input& input) const override {
      return (*reinterpret_cast<const ParserImpl*>(parser))(input);
    }
  };
  template <typename ParserImpl>
  struct WrapperImplInstance {
#if _MSC_VER && !__clang__
    // TODO(msvc): MSVC currently fails to initialize vtable pointers for constexpr values so
    //   we have to make this just const instead.
    static const WrapperImpl<ParserImpl> instance;
#else
    static constexpr WrapperImpl<ParserImpl> instance = WrapperImpl<ParserImpl>();
#endif
  };

  const void* parser;
  const Wrapper* wrapper;
};

template <typename Input, typename Output>
template <typename ParserImpl>
#if _MSC_VER && !__clang__
const typename ParserRef<Input, Output>::template WrapperImpl<ParserImpl>
ParserRef<Input, Output>::WrapperImplInstance<ParserImpl>::instance = WrapperImpl<ParserImpl>();
#else
constexpr typename ParserRef<Input, Output>::template WrapperImpl<ParserImpl>
ParserRef<Input, Output>::WrapperImplInstance<ParserImpl>::instance;
#endif

template <typename Input, typename ParserImpl>
constexpr ParserRef<Input, OutputType<ParserImpl, Input>> ref(ParserImpl& impl) {
  // Constructs a ParserRef.  You must specify the input type explicitly, e.g.
  // `ref<MyInput>(myParser)`.

  return ParserRef<Input, OutputType<ParserImpl, Input>>(impl);
}

// -------------------------------------------------------------------
// any
// Output = one token

class Any_ {
public:
  template <typename Input>
  Maybe<Decay<decltype(instance<Input>().consume())>> operator()(Input& input) const {
    if (input.atEnd()) {
      return nullptr;
    } else {
      return input.consume();
    }
  }
};

constexpr Any_ any = Any_();
// A parser which matches any token and simply returns it.

// -------------------------------------------------------------------
// exactly()
// Output = Tuple<>

template <typename T>
class Exactly_ {
public:
  explicit constexpr Exactly_(T&& expected): expected(expected) {}

  template <typename Input>
  Maybe<Tuple<>> operator()(Input& input) const {
    if (input.atEnd() || input.current() != expected) {
      return nullptr;
    } else {
      input.next();
      return Tuple<>();
    }
  }

private:
  T expected;
};

template <typename T>
constexpr Exactly_<T> exactly(T&& expected) {
  // Constructs a parser which succeeds when the input is exactly the token specified.  The
  // result is always the empty tuple.

  return Exactly_<T>(kj::fwd<T>(expected));
}

// -------------------------------------------------------------------
// exactlyConst()
// Output = Tuple<>

template <typename T, T expected>
class ExactlyConst_ {
public:
  explicit constexpr ExactlyConst_() {}

  template <typename Input>
  Maybe<Tuple<>> operator()(Input& input) const {
    if (input.atEnd() || input.current() != expected) {
      return nullptr;
    } else {
      input.next();
      return Tuple<>();
    }
  }
};

template <typename T, T expected>
constexpr ExactlyConst_<T, expected> exactlyConst() {
  // Constructs a parser which succeeds when the input is exactly the token specified.  The
  // result is always the empty tuple.  This parser is templated on the token value which may cause
  // it to perform better -- or worse.  Be sure to measure.

  return ExactlyConst_<T, expected>();
}

// -------------------------------------------------------------------
// constResult()

template <typename SubParser, typename Result>
class ConstResult_ {
public:
  explicit constexpr ConstResult_(SubParser&& subParser, Result&& result)
      : subParser(kj::fwd<SubParser>(subParser)), result(kj::fwd<Result>(result)) {}

  template <typename Input>
  Maybe<Result> operator()(Input& input) const {
    if (subParser(input) == nullptr) {
      return nullptr;
    } else {
      return result;
    }
  }

private:
  SubParser subParser;
  Result result;
};

template <typename SubParser, typename Result>
constexpr ConstResult_<SubParser, Result> constResult(SubParser&& subParser, Result&& result) {
  // Constructs a parser which returns exactly `result` if `subParser` is successful.
  return ConstResult_<SubParser, Result>(kj::fwd<SubParser>(subParser), kj::fwd<Result>(result));
}

template <typename SubParser>
constexpr ConstResult_<SubParser, Tuple<>> discard(SubParser&& subParser) {
  // Constructs a parser which wraps `subParser` but discards the result.
  return constResult(kj::fwd<SubParser>(subParser), Tuple<>());
}

// -------------------------------------------------------------------
// sequence()
// Output = Flattened Tuple of outputs of sub-parsers.

template <typename... SubParsers> class Sequence_;

template <typename FirstSubParser, typename... SubParsers>
class Sequence_<FirstSubParser, SubParsers...> {
public:
  template <typename T, typename... U>
  explicit constexpr Sequence_(T&& firstSubParser, U&&... rest)
      : first(kj::fwd<T>(firstSubParser)), rest(kj::fwd<U>(rest)...) {}

  // TODO(msvc): The trailing return types on `operator()` and `parseNext()` expose at least two
  //   bugs in MSVC:
  //
  //     1. An ICE.
  //     2. 'error C2672: 'operator __surrogate_func': no matching overloaded function found)',
  //        which crops up in numerous places when trying to build the capnp command line tools.
  //
  //   The only workaround I found for both bugs is to omit the trailing return types and instead
  //   rely on C++14's return type deduction.

  template <typename Input>
  auto operator()(Input& input) const
#if !_MSC_VER || __clang__
      -> Maybe<decltype(tuple(
          instance<OutputType<FirstSubParser, Input>>(),
          instance<OutputType<SubParsers, Input>>()...))>
#endif
  {
    return parseNext(input);
  }

  template <typename Input, typename... InitialParams>
  auto parseNext(Input& input, InitialParams&&... initialParams) const
#if !_MSC_VER || __clang__
      -> Maybe<decltype(tuple(
          kj::fwd<InitialParams>(initialParams)...,
          instance<OutputType<FirstSubParser, Input>>(),
          instance<OutputType<SubParsers, Input>>()...))>
#endif
  {
    KJ_IF_MAYBE(firstResult, first(input)) {
      return rest.parseNext(input, kj::fwd<InitialParams>(initialParams)...,
                            kj::mv(*firstResult));
    } else {
      // TODO(msvc): MSVC depends on return type deduction to compile this function, so we need to
      //   help it deduce the right type on this code path.
      return Maybe<decltype(tuple(
          kj::fwd<InitialParams>(initialParams)...,
          instance<OutputType<FirstSubParser, Input>>(),
          instance<OutputType<SubParsers, Input>>()...))>{nullptr};
    }
  }

private:
  FirstSubParser first;
  Sequence_<SubParsers...> rest;
};

template <>
class Sequence_<> {
public:
  template <typename Input>
  Maybe<Tuple<>> operator()(Input& input) const {
    return parseNext(input);
  }

  template <typename Input, typename... Params>
  auto parseNext(Input& input, Params&&... params) const ->
      Maybe<decltype(tuple(kj::fwd<Params>(params)...))> {
    return tuple(kj::fwd<Params>(params)...);
  }
};

template <typename... SubParsers>
constexpr Sequence_<SubParsers...> sequence(SubParsers&&... subParsers) {
  // Constructs a parser that executes each of the parameter parsers in sequence and returns a
  // tuple of their results.

  return Sequence_<SubParsers...>(kj::fwd<SubParsers>(subParsers)...);
}

// -------------------------------------------------------------------
// many()
// Output = Array of output of sub-parser, or just a uint count if the sub-parser returns Tuple<>.

template <typename SubParser, bool atLeastOne>
class Many_ {
  template <typename Input, typename Output = OutputType<SubParser, Input>>
  struct Impl;
public:
  explicit constexpr Many_(SubParser&& subParser)
      : subParser(kj::fwd<SubParser>(subParser)) {}

  template <typename Input>
  auto operator()(Input& input) const
      -> decltype(Impl<Input>::apply(instance<const SubParser&>(), input));

private:
  SubParser subParser;
};

template <typename SubParser, bool atLeastOne>
template <typename Input, typename Output>
struct Many_<SubParser, atLeastOne>::Impl {
  static Maybe<Array<Output>> apply(const SubParser& subParser, Input& input) {
    typedef Vector<OutputType<SubParser, Input>> Results;
    Results results;

    while (!input.atEnd()) {
      Input subInput(input);

      KJ_IF_MAYBE(subResult, subParser(subInput)) {
        subInput.advanceParent();
        results.add(kj::mv(*subResult));
      } else {
        break;
      }
    }

    if (atLeastOne && results.empty()) {
      return nullptr;
    }

    return results.releaseAsArray();
  }
};

template <typename SubParser, bool atLeastOne>
template <typename Input>
struct Many_<SubParser, atLeastOne>::Impl<Input, Tuple<>> {
  // If the sub-parser output is Tuple<>, just return a count.

  static Maybe<uint> apply(const SubParser& subParser, Input& input) {
    uint count = 0;

    while (!input.atEnd()) {
      Input subInput(input);

      KJ_IF_MAYBE(subResult, subParser(subInput)) {
        subInput.advanceParent();
        ++count;
      } else {
        break;
      }
    }

    if (atLeastOne && count == 0) {
      return nullptr;
    }

    return count;
  }
};

template <typename SubParser, bool atLeastOne>
template <typename Input>
auto Many_<SubParser, atLeastOne>::operator()(Input& input) const
    -> decltype(Impl<Input>::apply(instance<const SubParser&>(), input)) {
  return Impl<Input, OutputType<SubParser, Input>>::apply(subParser, input);
}

template <typename SubParser>
constexpr Many_<SubParser, false> many(SubParser&& subParser) {
  // Constructs a parser that repeatedly executes the given parser until it fails, returning an
  // Array of the results (or a uint count if `subParser` returns an empty tuple).
  return Many_<SubParser, false>(kj::fwd<SubParser>(subParser));
}

template <typename SubParser>
constexpr Many_<SubParser, true> oneOrMore(SubParser&& subParser) {
  // Like `many()` but the parser must parse at least one item to be successful.
  return Many_<SubParser, true>(kj::fwd<SubParser>(subParser));
}

// -------------------------------------------------------------------
// times()
// Output = Array of output of sub-parser, or Tuple<> if sub-parser returns Tuple<>.

template <typename SubParser>
class Times_ {
  template <typename Input, typename Output = OutputType<SubParser, Input>>
  struct Impl;
public:
  explicit constexpr Times_(SubParser&& subParser, uint count)
      : subParser(kj::fwd<SubParser>(subParser)), count(count) {}

  template <typename Input>
  auto operator()(Input& input) const
      -> decltype(Impl<Input>::apply(instance<const SubParser&>(), instance<uint>(), input));

private:
  SubParser subParser;
  uint count;
};

template <typename SubParser>
template <typename Input, typename Output>
struct Times_<SubParser>::Impl {
  static Maybe<Array<Output>> apply(const SubParser& subParser, uint count, Input& input) {
    auto results = heapArrayBuilder<OutputType<SubParser, Input>>(count);

    while (results.size() < count) {
      if (input.atEnd()) {
        return nullptr;
      } else KJ_IF_MAYBE(subResult, subParser(input)) {
        results.add(kj::mv(*subResult));
      } else {
        return nullptr;
      }
    }

    return results.finish();
  }
};

template <typename SubParser>
template <typename Input>
struct Times_<SubParser>::Impl<Input, Tuple<>> {
  // If the sub-parser output is Tuple<>, just return a count.

  static Maybe<Tuple<>> apply(const SubParser& subParser, uint count, Input& input) {
    uint actualCount = 0;

    while (actualCount < count) {
      if (input.atEnd()) {
        return nullptr;
      } else KJ_IF_MAYBE(subResult, subParser(input)) {
        ++actualCount;
      } else {
        return nullptr;
      }
    }

    return tuple();
  }
};

template <typename SubParser>
template <typename Input>
auto Times_<SubParser>::operator()(Input& input) const
    -> decltype(Impl<Input>::apply(instance<const SubParser&>(), instance<uint>(), input)) {
  return Impl<Input, OutputType<SubParser, Input>>::apply(subParser, count, input);
}

template <typename SubParser>
constexpr Times_<SubParser> times(SubParser&& subParser, uint count) {
  // Constructs a parser that repeats the subParser exactly `count` times.
  return Times_<SubParser>(kj::fwd<SubParser>(subParser), count);
}

// -------------------------------------------------------------------
// optional()
// Output = Maybe<output of sub-parser>

template <typename SubParser>
class Optional_ {
public:
  explicit constexpr Optional_(SubParser&& subParser)
      : subParser(kj::fwd<SubParser>(subParser)) {}

  template <typename Input>
  Maybe<Maybe<OutputType<SubParser, Input>>> operator()(Input& input) const {
    typedef Maybe<OutputType<SubParser, Input>> Result;

    Input subInput(input);
    KJ_IF_MAYBE(subResult, subParser(subInput)) {
      subInput.advanceParent();
      return Result(kj::mv(*subResult));
    } else {
      return Result(nullptr);
    }
  }

private:
  SubParser subParser;
};

template <typename SubParser>
constexpr Optional_<SubParser> optional(SubParser&& subParser) {
  // Constructs a parser that accepts zero or one of the given sub-parser, returning a Maybe
  // of the sub-parser's result.
  return Optional_<SubParser>(kj::fwd<SubParser>(subParser));
}

// -------------------------------------------------------------------
// oneOf()
// All SubParsers must have same output type, which becomes the output type of the
// OneOfParser.

template <typename... SubParsers>
class OneOf_;

template <typename FirstSubParser, typename... SubParsers>
class OneOf_<FirstSubParser, SubParsers...> {
public:
  explicit constexpr OneOf_(FirstSubParser&& firstSubParser, SubParsers&&... rest)
      : first(kj::fwd<FirstSubParser>(firstSubParser)), rest(kj::fwd<SubParsers>(rest)...) {}

  template <typename Input>
  Maybe<OutputType<FirstSubParser, Input>> operator()(Input& input) const {
    {
      Input subInput(input);
      Maybe<OutputType<FirstSubParser, Input>> firstResult = first(subInput);

      if (firstResult != nullptr) {
        subInput.advanceParent();
        return kj::mv(firstResult);
      }
    }

    // Hoping for some tail recursion here...
    return rest(input);
  }

private:
  FirstSubParser first;
  OneOf_<SubParsers...> rest;
};

template <>
class OneOf_<> {
public:
  template <typename Input>
  decltype(nullptr) operator()(Input& input) const {
    return nullptr;
  }
};

template <typename... SubParsers>
constexpr OneOf_<SubParsers...> oneOf(SubParsers&&... parsers) {
  // Constructs a parser that accepts one of a set of options.  The parser behaves as the first
  // sub-parser in the list which returns successfully.  All of the sub-parsers must return the
  // same type.
  return OneOf_<SubParsers...>(kj::fwd<SubParsers>(parsers)...);
}

// -------------------------------------------------------------------
// transform()
// Output = Result of applying transform functor to input value.  If input is a tuple, it is
// unpacked to form the transformation parameters.

template <typename Position>
struct Span {
public:
  inline const Position& begin() const { return begin_; }
  inline const Position& end() const { return end_; }

  Span() = default;
  inline constexpr Span(Position&& begin, Position&& end): begin_(mv(begin)), end_(mv(end)) {}

private:
  Position begin_;
  Position end_;
};

template <typename Position>
constexpr Span<Decay<Position>> span(Position&& start, Position&& end) {
  return Span<Decay<Position>>(kj::fwd<Position>(start), kj::fwd<Position>(end));
}

template <typename SubParser, typename TransformFunc>
class Transform_ {
public:
  explicit constexpr Transform_(SubParser&& subParser, TransformFunc&& transform)
      : subParser(kj::fwd<SubParser>(subParser)), transform(kj::fwd<TransformFunc>(transform)) {}

  template <typename Input>
  Maybe<decltype(kj::apply(instance<TransformFunc&>(),
                           instance<OutputType<SubParser, Input>&&>()))>
      operator()(Input& input) const {
    KJ_IF_MAYBE(subResult, subParser(input)) {
      return kj::apply(transform, kj::mv(*subResult));
    } else {
      return nullptr;
    }
  }

private:
  SubParser subParser;
  TransformFunc transform;
};

template <typename SubParser, typename TransformFunc>
class TransformOrReject_ {
public:
  explicit constexpr TransformOrReject_(SubParser&& subParser, TransformFunc&& transform)
      : subParser(kj::fwd<SubParser>(subParser)), transform(kj::fwd<TransformFunc>(transform)) {}

  template <typename Input>
  decltype(kj::apply(instance<TransformFunc&>(), instance<OutputType<SubParser, Input>&&>()))
      operator()(Input& input) const {
    KJ_IF_MAYBE(subResult, subParser(input)) {
      return kj::apply(transform, kj::mv(*subResult));
    } else {
      return nullptr;
    }
  }

private:
  SubParser subParser;
  TransformFunc transform;
};

template <typename SubParser, typename TransformFunc>
class TransformWithLocation_ {
public:
  explicit constexpr TransformWithLocation_(SubParser&& subParser, TransformFunc&& transform)
      : subParser(kj::fwd<SubParser>(subParser)), transform(kj::fwd<TransformFunc>(transform)) {}

  template <typename Input>
  Maybe<decltype(kj::apply(instance<TransformFunc&>(),
                           instance<Span<Decay<decltype(instance<Input&>().getPosition())>>>(),
                           instance<OutputType<SubParser, Input>&&>()))>
      operator()(Input& input) const {
    auto start = input.getPosition();
    KJ_IF_MAYBE(subResult, subParser(input)) {
      return kj::apply(transform, Span<decltype(start)>(kj::mv(start), input.getPosition()),
                       kj::mv(*subResult));
    } else {
      return nullptr;
    }
  }

private:
  SubParser subParser;
  TransformFunc transform;
};

template <typename SubParser, typename TransformFunc>
constexpr Transform_<SubParser, TransformFunc> transform(
    SubParser&& subParser, TransformFunc&& functor) {
  // Constructs a parser which executes some other parser and then transforms the result by invoking
  // `functor` on it.  Typically `functor` is a lambda.  It is invoked using `kj::apply`,
  // meaning tuples will be unpacked as arguments.
  return Transform_<SubParser, TransformFunc>(
      kj::fwd<SubParser>(subParser), kj::fwd<TransformFunc>(functor));
}

template <typename SubParser, typename TransformFunc>
constexpr TransformOrReject_<SubParser, TransformFunc> transformOrReject(
    SubParser&& subParser, TransformFunc&& functor) {
  // Like `transform()` except that `functor` returns a `Maybe`.  If it returns null, parsing fails,
  // otherwise the parser's result is the content of the `Maybe`.
  return TransformOrReject_<SubParser, TransformFunc>(
      kj::fwd<SubParser>(subParser), kj::fwd<TransformFunc>(functor));
}

template <typename SubParser, typename TransformFunc>
constexpr TransformWithLocation_<SubParser, TransformFunc> transformWithLocation(
    SubParser&& subParser, TransformFunc&& functor) {
  // Like `transform` except that `functor` also takes a `Span` as its first parameter specifying
  // the location of the parsed content.  The span's position type is whatever the parser input's
  // getPosition() returns.
  return TransformWithLocation_<SubParser, TransformFunc>(
      kj::fwd<SubParser>(subParser), kj::fwd<TransformFunc>(functor));
}

// -------------------------------------------------------------------
// notLookingAt()
// Fails if the given parser succeeds at the current location.

template <typename SubParser>
class NotLookingAt_ {
public:
  explicit constexpr NotLookingAt_(SubParser&& subParser)
      : subParser(kj::fwd<SubParser>(subParser)) {}

  template <typename Input>
  Maybe<Tuple<>> operator()(Input& input) const {
    Input subInput(input);
    subInput.forgetParent();
    if (subParser(subInput) == nullptr) {
      return Tuple<>();
    } else {
      return nullptr;
    }
  }

private:
  SubParser subParser;
};

template <typename SubParser>
constexpr NotLookingAt_<SubParser> notLookingAt(SubParser&& subParser) {
  // Constructs a parser which fails at any position where the given parser succeeds.  Otherwise,
  // it succeeds without consuming any input and returns an empty tuple.
  return NotLookingAt_<SubParser>(kj::fwd<SubParser>(subParser));
}

// -------------------------------------------------------------------
// endOfInput()
// Output = Tuple<>, only succeeds if at end-of-input

class EndOfInput_ {
public:
  template <typename Input>
  Maybe<Tuple<>> operator()(Input& input) const {
    if (input.atEnd()) {
      return Tuple<>();
    } else {
      return nullptr;
    }
  }
};

constexpr EndOfInput_ endOfInput = EndOfInput_();
// A parser that succeeds only if it is called with no input.

}  // namespace parse
}  // namespace kj

KJ_END_HEADER
