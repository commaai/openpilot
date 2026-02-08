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

#include "common.h"

KJ_BEGIN_HEADER

namespace kj {

namespace _ {  // private

template <uint i, template<uint> class Fail, typename Key, typename... Variants>
struct TypeIndex_;
template <uint i, template<uint> class Fail, typename Key, typename First, typename... Rest>
struct TypeIndex_<i, Fail, Key, First, Rest...> {
  static constexpr uint value = TypeIndex_<i + 1, Fail, Key, Rest...>::value;
};
template <uint i, template<uint> class Fail, typename Key, typename... Rest>
struct TypeIndex_<i, Fail, Key, Key, Rest...> { static constexpr uint value = i; };
template <uint i, template<uint> class Fail, typename Key>
struct TypeIndex_<i, Fail, Key>: public Fail<i> {};

template <uint i>
struct OneOfFailError_ {
  static_assert(i == -1, "type does not match any in OneOf");
};
template <uint i>
struct OneOfFailZero_ {
  static constexpr int value = 0;
};

template <uint i>
struct SuccessIfNotZero {
  typedef int Success;
};
template <>
struct SuccessIfNotZero<0> {};

enum class Variants0 {};
enum class Variants1 { _variant0 };
enum class Variants2 { _variant0, _variant1 };
enum class Variants3 { _variant0, _variant1, _variant2 };
enum class Variants4 { _variant0, _variant1, _variant2, _variant3 };
enum class Variants5 { _variant0, _variant1, _variant2, _variant3, _variant4 };
enum class Variants6 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5 };
enum class Variants7 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6 };
enum class Variants8 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                       _variant7 };
enum class Variants9 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                       _variant7, _variant8 };
enum class Variants10 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9 };
enum class Variants11 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10 };
enum class Variants12 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11 };
enum class Variants13 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12 };
enum class Variants14 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13 };
enum class Variants15 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14 };
enum class Variants16 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15 };
enum class Variants17 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16 };
enum class Variants18 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17 };
enum class Variants19 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18 };
enum class Variants20 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19 };
enum class Variants21 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20 };
enum class Variants22 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21 };
enum class Variants23 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22 };
enum class Variants24 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23 };
enum class Variants25 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24 };
enum class Variants26 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25 };
enum class Variants27 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26 };
enum class Variants28 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27 };
enum class Variants29 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28 };
enum class Variants30 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29 };
enum class Variants31 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30 };
enum class Variants32 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31 };
enum class Variants33 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32 };
enum class Variants34 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33 };
enum class Variants35 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34 };
enum class Variants36 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35 };
enum class Variants37 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36 };
enum class Variants38 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37 };
enum class Variants39 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38 };
enum class Variants40 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39 };
enum class Variants41 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40 };
enum class Variants42 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41 };
enum class Variants43 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41, _variant42 };
enum class Variants44 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41, _variant42,
                        _variant43 };
enum class Variants45 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41, _variant42,
                        _variant43, _variant44 };
enum class Variants46 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41, _variant42,
                        _variant43, _variant44, _variant45 };
enum class Variants47 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41, _variant42,
                        _variant43, _variant44, _variant45, _variant46 };
enum class Variants48 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41, _variant42,
                        _variant43, _variant44, _variant45, _variant46, _variant47 };
enum class Variants49 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41, _variant42,
                        _variant43, _variant44, _variant45, _variant46, _variant47, _variant48 };
enum class Variants50 { _variant0, _variant1, _variant2, _variant3, _variant4, _variant5, _variant6,
                        _variant7, _variant8, _variant9, _variant10, _variant11, _variant12,
                        _variant13, _variant14, _variant15, _variant16, _variant17, _variant18,
                        _variant19, _variant20, _variant21, _variant22, _variant23, _variant24,
                        _variant25, _variant26, _variant27, _variant28, _variant29, _variant30,
                        _variant31, _variant32, _variant33, _variant34, _variant35, _variant36,
                        _variant37, _variant38, _variant39, _variant40, _variant41, _variant42,
                        _variant43, _variant44, _variant45, _variant46, _variant47, _variant48,
                        _variant49 };

template <uint i> struct Variants_;
template <> struct Variants_<0> { typedef Variants0 Type; };
template <> struct Variants_<1> { typedef Variants1 Type; };
template <> struct Variants_<2> { typedef Variants2 Type; };
template <> struct Variants_<3> { typedef Variants3 Type; };
template <> struct Variants_<4> { typedef Variants4 Type; };
template <> struct Variants_<5> { typedef Variants5 Type; };
template <> struct Variants_<6> { typedef Variants6 Type; };
template <> struct Variants_<7> { typedef Variants7 Type; };
template <> struct Variants_<8> { typedef Variants8 Type; };
template <> struct Variants_<9> { typedef Variants9 Type; };
template <> struct Variants_<10> { typedef Variants10 Type; };
template <> struct Variants_<11> { typedef Variants11 Type; };
template <> struct Variants_<12> { typedef Variants12 Type; };
template <> struct Variants_<13> { typedef Variants13 Type; };
template <> struct Variants_<14> { typedef Variants14 Type; };
template <> struct Variants_<15> { typedef Variants15 Type; };
template <> struct Variants_<16> { typedef Variants16 Type; };
template <> struct Variants_<17> { typedef Variants17 Type; };
template <> struct Variants_<18> { typedef Variants18 Type; };
template <> struct Variants_<19> { typedef Variants19 Type; };
template <> struct Variants_<20> { typedef Variants20 Type; };
template <> struct Variants_<21> { typedef Variants21 Type; };
template <> struct Variants_<22> { typedef Variants22 Type; };
template <> struct Variants_<23> { typedef Variants23 Type; };
template <> struct Variants_<24> { typedef Variants24 Type; };
template <> struct Variants_<25> { typedef Variants25 Type; };
template <> struct Variants_<26> { typedef Variants26 Type; };
template <> struct Variants_<27> { typedef Variants27 Type; };
template <> struct Variants_<28> { typedef Variants28 Type; };
template <> struct Variants_<29> { typedef Variants29 Type; };
template <> struct Variants_<30> { typedef Variants30 Type; };
template <> struct Variants_<31> { typedef Variants31 Type; };
template <> struct Variants_<32> { typedef Variants32 Type; };
template <> struct Variants_<33> { typedef Variants33 Type; };
template <> struct Variants_<34> { typedef Variants34 Type; };
template <> struct Variants_<35> { typedef Variants35 Type; };
template <> struct Variants_<36> { typedef Variants36 Type; };
template <> struct Variants_<37> { typedef Variants37 Type; };
template <> struct Variants_<38> { typedef Variants38 Type; };
template <> struct Variants_<39> { typedef Variants39 Type; };
template <> struct Variants_<40> { typedef Variants40 Type; };
template <> struct Variants_<41> { typedef Variants41 Type; };
template <> struct Variants_<42> { typedef Variants42 Type; };
template <> struct Variants_<43> { typedef Variants43 Type; };
template <> struct Variants_<44> { typedef Variants44 Type; };
template <> struct Variants_<45> { typedef Variants45 Type; };
template <> struct Variants_<46> { typedef Variants46 Type; };
template <> struct Variants_<47> { typedef Variants47 Type; };
template <> struct Variants_<48> { typedef Variants48 Type; };
template <> struct Variants_<49> { typedef Variants49 Type; };
template <> struct Variants_<50> { typedef Variants50 Type; };

template <uint i>
using Variants = typename Variants_<i>::Type;

}  // namespace _ (private)

template <typename... Variants>
class OneOf {
  template <typename Key>
  static inline constexpr uint typeIndex() {
    return _::TypeIndex_<1, _::OneOfFailError_, Key, Variants...>::value;
  }
  // Get the 1-based index of Key within the type list Types, or static_assert with a nice error.

  template <typename Key>
  static inline constexpr uint typeIndexOrZero() {
    return _::TypeIndex_<1, _::OneOfFailZero_, Key, Variants...>::value;
  }

  template <uint i, typename... OtherVariants>
  struct HasAll;
  // Has a member type called "Success" if and only if all of `OtherVariants` are types that
  // appear in `Variants`. Used with SFINAE to enable subset constructors.

public:
  inline OneOf(): tag(0) {}

  OneOf(const OneOf& other) { copyFrom(other); }
  OneOf(OneOf& other) { copyFrom(other); }
  OneOf(OneOf&& other) { moveFrom(other); }
  // Copy/move from same OneOf type.

  template <typename... OtherVariants, typename = typename HasAll<1, OtherVariants...>::Success>
  OneOf(const OneOf<OtherVariants...>& other) { copyFromSubset(other); }
  template <typename... OtherVariants, typename = typename HasAll<1, OtherVariants...>::Success>
  OneOf(OneOf<OtherVariants...>& other) { copyFromSubset(other); }
  template <typename... OtherVariants, typename = typename HasAll<1, OtherVariants...>::Success>
  OneOf(OneOf<OtherVariants...>&& other) { moveFromSubset(other); }
  // Copy/move from OneOf that contains a subset of the types we do.

  template <typename T, typename = typename HasAll<0, Decay<T>>::Success>
  OneOf(T&& other): tag(typeIndex<Decay<T>>()) {
    ctor(*reinterpret_cast<Decay<T>*>(space), kj::fwd<T>(other));
  }
  // Copy/move from a value that matches one of the individual types in the OneOf.

  ~OneOf() { destroy(); }

  OneOf& operator=(const OneOf& other) { if (tag != 0) destroy(); copyFrom(other); return *this; }
  OneOf& operator=(OneOf&& other) { if (tag != 0) destroy(); moveFrom(other); return *this; }

  inline bool operator==(decltype(nullptr)) const { return tag == 0; }
  inline bool operator!=(decltype(nullptr)) const { return tag != 0; }

  template <typename T>
  bool is() const {
    return tag == typeIndex<T>();
  }

  template <typename T>
  T& get() & {
    KJ_IREQUIRE(is<T>(), "Must check OneOf::is<T>() before calling get<T>().");
    return *reinterpret_cast<T*>(space);
  }
  template <typename T>
  T&& get() && {
    KJ_IREQUIRE(is<T>(), "Must check OneOf::is<T>() before calling get<T>().");
    return kj::mv(*reinterpret_cast<T*>(space));
  }
  template <typename T>
  const T& get() const& {
    KJ_IREQUIRE(is<T>(), "Must check OneOf::is<T>() before calling get<T>().");
    return *reinterpret_cast<const T*>(space);
  }
  template <typename T>
  const T&& get() const&& {
    KJ_IREQUIRE(is<T>(), "Must check OneOf::is<T>() before calling get<T>().");
    return kj::mv(*reinterpret_cast<const T*>(space));
  }

  template <typename T, typename... Params>
  T& init(Params&&... params) {
    if (tag != 0) destroy();
    ctor(*reinterpret_cast<T*>(space), kj::fwd<Params>(params)...);
    tag = typeIndex<T>();
    return *reinterpret_cast<T*>(space);
  }

  template <typename T>
  Maybe<T&> tryGet() {
    if (is<T>()) {
      return *reinterpret_cast<T*>(space);
    } else {
      return nullptr;
    }
  }
  template <typename T>
  Maybe<const T&> tryGet() const {
    if (is<T>()) {
      return *reinterpret_cast<const T*>(space);
    } else {
      return nullptr;
    }
  }

  template <uint i>
  KJ_NORETURN(void allHandled());
  // After a series of if/else blocks handling each variant of the OneOf, have the final else
  // block call allHandled<n>() where n is the number of variants. This will fail to compile
  // if new variants are added in the future.

  typedef _::Variants<sizeof...(Variants)> Tag;

  Tag which() const {
    KJ_IREQUIRE(tag != 0, "Can't KJ_SWITCH_ONEOF() on uninitialized value.");
    return static_cast<Tag>(tag - 1);
  }

  template <typename T>
  static constexpr Tag tagFor() {
    return static_cast<Tag>(typeIndex<T>() - 1);
  }

  OneOf* _switchSubject() & { return this; }
  const OneOf* _switchSubject() const& { return this; }
  _::NullableValue<OneOf> _switchSubject() && { return kj::mv(*this); }

private:
  uint tag;

  static inline constexpr size_t maxSize(size_t a) {
    return a;
  }
  template <typename... Rest>
  static inline constexpr size_t maxSize(size_t a, size_t b, Rest... rest) {
    return maxSize(kj::max(a, b), rest...);
  }
  // Returns the maximum of all the parameters.
  // TODO(someday):  Generalize the above template and make it common.  I tried, but C++ decided to
  //   be difficult so I cut my losses.

  static constexpr auto spaceSize = maxSize(sizeof(Variants)...);
  // TODO(msvc):  This constant could just as well go directly inside space's bracket's, where it's
  // used, but MSVC suffers a parse error on `...`.

  union {
    byte space[spaceSize];

    void* forceAligned;
    // TODO(someday):  Use C++11 alignas() once we require GCC 4.8 / Clang 3.3.
  };

  template <typename... T>
  inline void doAll(T... t) {}

  template <typename T>
  inline bool destroyVariant() {
    if (tag == typeIndex<T>()) {
      tag = 0;
      dtor(*reinterpret_cast<T*>(space));
    }
    return false;
  }
  void destroy() {
    doAll(destroyVariant<Variants>()...);
  }

  template <typename T>
  inline bool copyVariantFrom(const OneOf& other) {
    if (other.is<T>()) {
      ctor(*reinterpret_cast<T*>(space), other.get<T>());
    }
    return false;
  }
  void copyFrom(const OneOf& other) {
    // Initialize as a copy of `other`.  Expects that `this` starts out uninitialized, so the tag
    // is invalid.
    tag = other.tag;
    doAll(copyVariantFrom<Variants>(other)...);
  }

  template <typename T>
  inline bool copyVariantFrom(OneOf& other) {
    if (other.is<T>()) {
      ctor(*reinterpret_cast<T*>(space), other.get<T>());
    }
    return false;
  }
  void copyFrom(OneOf& other) {
    // Initialize as a copy of `other`.  Expects that `this` starts out uninitialized, so the tag
    // is invalid.
    tag = other.tag;
    doAll(copyVariantFrom<Variants>(other)...);
  }

  template <typename T>
  inline bool moveVariantFrom(OneOf& other) {
    if (other.is<T>()) {
      ctor(*reinterpret_cast<T*>(space), kj::mv(other.get<T>()));
    }
    return false;
  }
  void moveFrom(OneOf& other) {
    // Initialize as a copy of `other`.  Expects that `this` starts out uninitialized, so the tag
    // is invalid.
    tag = other.tag;
    doAll(moveVariantFrom<Variants>(other)...);
  }

  template <typename T, typename... OtherVariants>
  inline bool copySubsetVariantFrom(const OneOf<OtherVariants...>& other) {
    if (other.template is<T>()) {
      tag = typeIndex<Decay<T>>();
      ctor(*reinterpret_cast<T*>(space), other.template get<T>());
    }
    return false;
  }
  template <typename... OtherVariants>
  void copyFromSubset(const OneOf<OtherVariants...>& other) {
    doAll(copySubsetVariantFrom<OtherVariants>(other)...);
  }

  template <typename T, typename... OtherVariants>
  inline bool copySubsetVariantFrom(OneOf<OtherVariants...>& other) {
    if (other.template is<T>()) {
      tag = typeIndex<Decay<T>>();
      ctor(*reinterpret_cast<T*>(space), other.template get<T>());
    }
    return false;
  }
  template <typename... OtherVariants>
  void copyFromSubset(OneOf<OtherVariants...>& other) {
    doAll(copySubsetVariantFrom<OtherVariants>(other)...);
  }

  template <typename T, typename... OtherVariants>
  inline bool moveSubsetVariantFrom(OneOf<OtherVariants...>& other) {
    if (other.template is<T>()) {
      tag = typeIndex<Decay<T>>();
      ctor(*reinterpret_cast<T*>(space), kj::mv(other.template get<T>()));
    }
    return false;
  }
  template <typename... OtherVariants>
  void moveFromSubset(OneOf<OtherVariants...>& other) {
    doAll(moveSubsetVariantFrom<OtherVariants>(other)...);
  }
};

template <typename... Variants>
template <uint i, typename First, typename... Rest>
struct OneOf<Variants...>::HasAll<i, First, Rest...>
    : public HasAll<typeIndexOrZero<First>(), Rest...> {};
template <typename... Variants>
template <uint i>
struct OneOf<Variants...>::HasAll<i>: public _::SuccessIfNotZero<i> {};

template <typename... Variants>
template <uint i>
void OneOf<Variants...>::allHandled() {
  // After a series of if/else blocks handling each variant of the OneOf, have the final else
  // block call allHandled<n>() where n is the number of variants. This will fail to compile
  // if new variants are added in the future.

  static_assert(i == sizeof...(Variants), "new OneOf variants need to be handled here");
  KJ_UNREACHABLE;
}

#if __cplusplus > 201402L
#define KJ_SWITCH_ONEOF(value) \
  switch (auto _kj_switch_subject = (value)._switchSubject(); _kj_switch_subject->which())
#else
#define KJ_SWITCH_ONEOF(value) \
  /* Without C++17, we can only support one switch per containing block. Deal with it. */ \
  auto _kj_switch_subject = (value)._switchSubject(); \
  switch (_kj_switch_subject->which())
#endif
#if !_MSC_VER || defined(__clang__)
#define KJ_CASE_ONEOF(name, ...) \
    break; \
  case ::kj::Decay<decltype(*_kj_switch_subject)>::template tagFor<__VA_ARGS__>(): \
    for (auto& name = _kj_switch_subject->template get<__VA_ARGS__>(), *_kj_switch_done = &name; \
         _kj_switch_done; _kj_switch_done = nullptr)
#else
// TODO(msvc): The latest MSVC which ships with VS2019 now ICEs on the implementation above. It
//   appears we can hack around the problem by moving the `->template get<>()` syntax to an outer
//   `if`. (This unfortunately allows wonky syntax like `KJ_CASE_ONEOF(a, B) { } else { }`.)
//   https://developercommunity.visualstudio.com/content/problem/1143733/internal-compiler-error-on-v1670.html
#define KJ_CASE_ONEOF(name, ...) \
    break; \
  case ::kj::Decay<decltype(*_kj_switch_subject)>::template tagFor<__VA_ARGS__>(): \
    if (auto* _kj_switch_done = &_kj_switch_subject->template get<__VA_ARGS__>()) \
      for (auto& name = *_kj_switch_done; _kj_switch_done; _kj_switch_done = nullptr)
#endif
#define KJ_CASE_ONEOF_DEFAULT break; default:
// Allows switching over a OneOf.
//
// Example:
//
//     kj::OneOf<int, float, const char*> variant;
//     KJ_SWITCH_ONEOF(variant) {
//       KJ_CASE_ONEOF(i, int) {
//         doSomethingWithInt(i);
//       }
//       KJ_CASE_ONEOF(s, const char*) {
//         doSomethingWithString(s);
//       }
//       KJ_CASE_ONEOF_DEFAULT {
//         doSomethingElse();
//       }
//     }
//
// Notes:
// - If you don't handle all possible types and don't include a default branch, you'll get a
//   compiler warning, just like a regular switch() over an enum where one of the enum values is
//   missing.
// - There's no need for a `break` statement in a KJ_CASE_ONEOF; it is implied.
// - Under C++11 and C++14, only one KJ_SWITCH_ONEOF() can appear in a block. Wrap the switch in
//   a pair of braces if you need a second switch in the same block. If C++17 is enabled, this is
//   not an issue.
//
// Implementation notes:
// - The use of __VA_ARGS__ is to account for template types that have commas separating type
//   parameters, since macros don't recognize <> as grouping.
// - _kj_switch_done is really used as a boolean flag to prevent the for() loop from actually
//   looping, but it's defined as a pointer since that's all we can define in this context.

}  // namespace kj

KJ_END_HEADER
