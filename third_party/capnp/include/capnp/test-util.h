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

#include <capnp/test.capnp.h>
#include <iostream>
#include "blob.h"
#include <kj/compat/gtest.h>

#if !CAPNP_LITE
#include "dynamic.h"
#include <kj/io.h>
#endif  // !CAPNP_LITE

CAPNP_BEGIN_HEADER

// TODO(cleanup): Auto-generate stringification functions for union discriminants.
namespace capnproto_test {
namespace capnp {
namespace test {
inline kj::String KJ_STRINGIFY(TestUnion::Union0::Which which) {
  return kj::str(static_cast<uint16_t>(which));
}
inline kj::String KJ_STRINGIFY(TestUnion::Union1::Which which) {
  return kj::str(static_cast<uint16_t>(which));
}
inline kj::String KJ_STRINGIFY(TestUnion::Union2::Which which) {
  return kj::str(static_cast<uint16_t>(which));
}
inline kj::String KJ_STRINGIFY(TestUnion::Union3::Which which) {
  return kj::str(static_cast<uint16_t>(which));
}
inline kj::String KJ_STRINGIFY(TestUnnamedUnion::Which which) {
  return kj::str(static_cast<uint16_t>(which));
}
inline kj::String KJ_STRINGIFY(TestGroups::Groups::Which which) {
  return kj::str(static_cast<uint16_t>(which));
}
inline kj::String KJ_STRINGIFY(TestInterleavedGroups::Group1::Which which) {
  return kj::str(static_cast<uint16_t>(which));
}
}  // namespace test
}  // namespace capnp
}  // namespace capnproto_test

namespace capnp {
namespace _ {  // private

inline Data::Reader data(const char* str) {
  return Data::Reader(reinterpret_cast<const byte*>(str), strlen(str));
}

namespace test = capnproto_test::capnp::test;

// We don't use "using namespace" to pull these in because then things would still compile
// correctly if they were generated in the global namespace.
using ::capnproto_test::capnp::test::TestAllTypes;
using ::capnproto_test::capnp::test::TestDefaults;
using ::capnproto_test::capnp::test::TestEnum;
using ::capnproto_test::capnp::test::TestUnion;
using ::capnproto_test::capnp::test::TestUnionDefaults;
using ::capnproto_test::capnp::test::TestNestedTypes;
using ::capnproto_test::capnp::test::TestUsing;
using ::capnproto_test::capnp::test::TestListDefaults;

void initTestMessage(TestAllTypes::Builder builder);
void initTestMessage(TestDefaults::Builder builder);
void initTestMessage(TestListDefaults::Builder builder);

void checkTestMessage(TestAllTypes::Builder builder);
void checkTestMessage(TestDefaults::Builder builder);
void checkTestMessage(TestListDefaults::Builder builder);

void checkTestMessage(TestAllTypes::Reader reader);
void checkTestMessage(TestDefaults::Reader reader);
void checkTestMessage(TestListDefaults::Reader reader);

void checkTestMessageAllZero(TestAllTypes::Builder builder);
void checkTestMessageAllZero(TestAllTypes::Reader reader);

#if !CAPNP_LITE
void initDynamicTestMessage(DynamicStruct::Builder builder);
void initDynamicTestLists(DynamicStruct::Builder builder);
void checkDynamicTestMessage(DynamicStruct::Builder builder);
void checkDynamicTestLists(DynamicStruct::Builder builder);
void checkDynamicTestMessage(DynamicStruct::Reader reader);
void checkDynamicTestLists(DynamicStruct::Reader reader);
void checkDynamicTestMessageAllZero(DynamicStruct::Builder builder);
void checkDynamicTestMessageAllZero(DynamicStruct::Reader reader);
#endif  // !CAPNP_LITE

template <typename T>
inline void checkElement(T a, T b) {
  EXPECT_EQ(a, b);
}

template <>
inline void checkElement<float>(float a, float b) {
  EXPECT_FLOAT_EQ(a, b);
}

template <>
inline void checkElement<double>(double a, double b) {
  EXPECT_DOUBLE_EQ(a, b);
}

template <typename T, typename L = typename T::Reads>
void checkList(T reader, std::initializer_list<decltype(reader[0])> expected) {
  ASSERT_EQ(expected.size(), reader.size());
  for (uint i = 0; i < expected.size(); i++) {
    checkElement<decltype(reader[0])>(expected.begin()[i], reader[i]);
  }
}

template <typename T, typename L = typename T::Builds, bool = false>
void checkList(T reader, std::initializer_list<decltype(typename L::Reader()[0])> expected) {
  ASSERT_EQ(expected.size(), reader.size());
  for (uint i = 0; i < expected.size(); i++) {
    checkElement<decltype(typename L::Reader()[0])>(expected.begin()[i], reader[i]);
  }
}

inline void checkList(List<test::TestOldVersion>::Reader reader,
                      std::initializer_list<int64_t> expectedData,
                      std::initializer_list<Text::Reader> expectedPointers) {
  ASSERT_EQ(expectedData.size(), reader.size());
  for (uint i = 0; i < expectedData.size(); i++) {
    EXPECT_EQ(expectedData.begin()[i], reader[i].getOld1());
    EXPECT_EQ(expectedPointers.begin()[i], reader[i].getOld2());
  }
}

// Hack because as<>() is a template-parameter-dependent lookup everywhere below...
#define as template as

template <typename T> void expectPrimitiveEq(T a, T b) { EXPECT_EQ(a, b); }
inline void expectPrimitiveEq(float a, float b) { EXPECT_FLOAT_EQ(a, b); }
inline void expectPrimitiveEq(double a, double b) { EXPECT_DOUBLE_EQ(a, b); }
inline void expectPrimitiveEq(Text::Reader a, Text::Builder b) { EXPECT_EQ(a, b); }
inline void expectPrimitiveEq(Data::Reader a, Data::Builder b) { EXPECT_EQ(a, b); }

#if !CAPNP_LITE
template <typename Element, typename T>
void checkList(T reader, std::initializer_list<ReaderFor<Element>> expected) {
  auto list = reader.as<DynamicList>();
  ASSERT_EQ(expected.size(), list.size());
  for (uint i = 0; i < expected.size(); i++) {
    expectPrimitiveEq(expected.begin()[i], list[i].as<Element>());
  }

  auto typed = reader.as<List<Element>>();
  ASSERT_EQ(expected.size(), typed.size());
  for (uint i = 0; i < expected.size(); i++) {
    expectPrimitiveEq(expected.begin()[i], typed[i]);
  }
}
#endif  // !CAPNP_LITE

#undef as

// =======================================================================================
// Interface implementations.

#if !CAPNP_LITE

class TestInterfaceImpl final: public test::TestInterface::Server {
public:
  TestInterfaceImpl(int& callCount);

  kj::Promise<void> foo(FooContext context) override;

  kj::Promise<void> baz(BazContext context) override;

private:
  int& callCount;
};

class TestExtendsImpl final: public test::TestExtends2::Server {
public:
  TestExtendsImpl(int& callCount);

  kj::Promise<void> foo(FooContext context) override;

  kj::Promise<void> grault(GraultContext context) override;

private:
  int& callCount;
};

class TestPipelineImpl final: public test::TestPipeline::Server {
public:
  TestPipelineImpl(int& callCount);

  kj::Promise<void> getCap(GetCapContext context) override;
  kj::Promise<void> getAnyCap(GetAnyCapContext context) override;
  kj::Promise<void> getCapPipelineOnly(GetCapPipelineOnlyContext context) override;

private:
  int& callCount;
};

class TestCallOrderImpl final: public test::TestCallOrder::Server {
public:
  kj::Promise<void> getCallSequence(GetCallSequenceContext context) override;

  uint getCount() { return count; }

private:
  uint count = 0;
};

class TestTailCallerImpl final: public test::TestTailCaller::Server {
public:
  TestTailCallerImpl(int& callCount);

  kj::Promise<void> foo(FooContext context) override;

private:
  int& callCount;
};

class TestTailCalleeImpl final: public test::TestTailCallee::Server {
public:
  TestTailCalleeImpl(int& callCount);

  kj::Promise<void> foo(FooContext context) override;

private:
  int& callCount;
};

class TestMoreStuffImpl final: public test::TestMoreStuff::Server {
public:
  TestMoreStuffImpl(int& callCount, int& handleCount);

  kj::Promise<void> getCallSequence(GetCallSequenceContext context) override;

  kj::Promise<void> callFoo(CallFooContext context) override;

  kj::Promise<void> callFooWhenResolved(CallFooWhenResolvedContext context) override;

  kj::Promise<void> neverReturn(NeverReturnContext context) override;

  kj::Promise<void> hold(HoldContext context) override;

  kj::Promise<void> callHeld(CallHeldContext context) override;

  kj::Promise<void> getHeld(GetHeldContext context) override;

  kj::Promise<void> echo(EchoContext context) override;

  kj::Promise<void> expectCancel(ExpectCancelContext context) override;

  kj::Promise<void> getHandle(GetHandleContext context) override;

  kj::Promise<void> getNull(GetNullContext context) override;

  kj::Promise<void> getEnormousString(GetEnormousStringContext context) override;

  kj::Promise<void> writeToFd(WriteToFdContext context) override;

  kj::Promise<void> throwException(ThrowExceptionContext context) override;

  kj::Promise<void> throwRemoteException(ThrowRemoteExceptionContext context) override;

private:
  int& callCount;
  int& handleCount;
  test::TestInterface::Client clientToHold = nullptr;

  kj::Promise<void> loop(uint depth, test::TestInterface::Client cap, ExpectCancelContext context);
};

class TestCapDestructor final: public test::TestInterface::Server {
  // Implementation of TestInterface that notifies when it is destroyed.

public:
  TestCapDestructor(kj::Own<kj::PromiseFulfiller<void>>&& fulfiller)
      : fulfiller(kj::mv(fulfiller)), impl(dummy) {}

  ~TestCapDestructor() {
    fulfiller->fulfill();
  }

  kj::Promise<void> foo(FooContext context) {
    return impl.foo(context);
  }

private:
  kj::Own<kj::PromiseFulfiller<void>> fulfiller;
  int dummy = 0;
  TestInterfaceImpl impl;
};

class TestFdCap final: public test::TestInterface::Server {
  // Implementation of TestInterface that wraps a file descriptor.

public:
  TestFdCap(kj::AutoCloseFd fd): fd(kj::mv(fd)) {}

  kj::Maybe<int> getFd() override { return fd.get(); }

private:
  kj::AutoCloseFd fd;
};

class TestStreamingImpl final: public test::TestStreaming::Server {
public:
  uint iSum = 0;
  uint jSum = 0;
  kj::Maybe<kj::Own<kj::PromiseFulfiller<void>>> fulfiller;
  bool jShouldThrow = false;

  kj::Promise<void> doStreamI(DoStreamIContext context) override {
    iSum += context.getParams().getI();
    auto paf = kj::newPromiseAndFulfiller<void>();
    fulfiller = kj::mv(paf.fulfiller);
    return kj::mv(paf.promise);
  }

  kj::Promise<void> doStreamJ(DoStreamJContext context) override {
    jSum += context.getParams().getJ();

    if (jShouldThrow) {
      KJ_FAIL_ASSERT("throw requested") { break; }
      return kj::READY_NOW;
    }

    auto paf = kj::newPromiseAndFulfiller<void>();
    fulfiller = kj::mv(paf.fulfiller);
    return kj::mv(paf.promise);
  }

  kj::Promise<void> finishStream(FinishStreamContext context) override {
    auto results = context.getResults();
    results.setTotalI(iSum);
    results.setTotalJ(jSum);
    return kj::READY_NOW;
  }
};

#endif  // !CAPNP_LITE

}  // namespace _ (private)
}  // namespace capnp

CAPNP_END_HEADER
