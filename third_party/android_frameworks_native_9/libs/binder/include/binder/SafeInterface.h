/*
 * Copyright 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <binder/IInterface.h>
#include <binder/Parcel.h>
#include <cutils/compiler.h>

// Set to 1 to enable CallStacks when logging errors
#define SI_DUMP_CALLSTACKS 0
#if SI_DUMP_CALLSTACKS
#include <utils/CallStack.h>
#endif

#include <utils/NativeHandle.h>

#include <functional>
#include <type_traits>

namespace android {
namespace SafeInterface {

// ParcelHandler is responsible for writing/reading various types to/from a Parcel in a generic way
class ParcelHandler {
public:
    explicit ParcelHandler(const char* logTag) : mLogTag(logTag) {}

    // Specializations for types with dedicated handling in Parcel
    status_t read(const Parcel& parcel, bool* b) const {
        return callParcel("readBool", [&]() { return parcel.readBool(b); });
    }
    status_t write(Parcel* parcel, bool b) const {
        return callParcel("writeBool", [&]() { return parcel->writeBool(b); });
    }
    template <typename E>
    typename std::enable_if<std::is_enum<E>::value, status_t>::type read(const Parcel& parcel,
                                                                         E* e) const {
        typename std::underlying_type<E>::type u{};
        status_t result = read(parcel, &u);
        *e = static_cast<E>(u);
        return result;
    }
    template <typename E>
    typename std::enable_if<std::is_enum<E>::value, status_t>::type write(Parcel* parcel,
                                                                          E e) const {
        return write(parcel, static_cast<typename std::underlying_type<E>::type>(e));
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<Flattenable<T>, T>::value, status_t>::type read(
            const Parcel& parcel, T* t) const {
        return callParcel("read(Flattenable)", [&]() { return parcel.read(*t); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<Flattenable<T>, T>::value, status_t>::type write(
            Parcel* parcel, const T& t) const {
        return callParcel("write(Flattenable)", [&]() { return parcel->write(t); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<Flattenable<T>, T>::value, status_t>::type read(
            const Parcel& parcel, sp<T>* t) const {
        *t = new T{};
        return callParcel("read(sp<Flattenable>)", [&]() { return parcel.read(*(t->get())); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<Flattenable<T>, T>::value, status_t>::type write(
            Parcel* parcel, const sp<T>& t) const {
        return callParcel("write(sp<Flattenable>)", [&]() { return parcel->write(*(t.get())); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<LightFlattenable<T>, T>::value, status_t>::type read(
            const Parcel& parcel, T* t) const {
        return callParcel("read(LightFlattenable)", [&]() { return parcel.read(*t); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<LightFlattenable<T>, T>::value, status_t>::type write(
            Parcel* parcel, const T& t) const {
        return callParcel("write(LightFlattenable)", [&]() { return parcel->write(t); });
    }
    template <typename NH>
    typename std::enable_if<std::is_same<NH, sp<NativeHandle>>::value, status_t>::type read(
            const Parcel& parcel, NH* nh) {
        *nh = NativeHandle::create(parcel.readNativeHandle(), true);
        return NO_ERROR;
    }
    template <typename NH>
    typename std::enable_if<std::is_same<NH, sp<NativeHandle>>::value, status_t>::type write(
            Parcel* parcel, const NH& nh) {
        return callParcel("write(sp<NativeHandle>)",
                          [&]() { return parcel->writeNativeHandle(nh->handle()); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<Parcelable, T>::value, status_t>::type read(
            const Parcel& parcel, T* t) const {
        return callParcel("readParcelable", [&]() { return parcel.readParcelable(t); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<Parcelable, T>::value, status_t>::type write(
            Parcel* parcel, const T& t) const {
        return callParcel("writeParcelable", [&]() { return parcel->writeParcelable(t); });
    }
    status_t read(const Parcel& parcel, String8* str) const {
        return callParcel("readString8", [&]() { return parcel.readString8(str); });
    }
    status_t write(Parcel* parcel, const String8& str) const {
        return callParcel("writeString8", [&]() { return parcel->writeString8(str); });
    }
    template <typename T>
    typename std::enable_if<std::is_same<IBinder, T>::value, status_t>::type read(
            const Parcel& parcel, sp<T>* pointer) const {
        return callParcel("readNullableStrongBinder",
                          [&]() { return parcel.readNullableStrongBinder(pointer); });
    }
    template <typename T>
    typename std::enable_if<std::is_same<IBinder, T>::value, status_t>::type write(
            Parcel* parcel, const sp<T>& pointer) const {
        return callParcel("writeStrongBinder",
                          [&]() { return parcel->writeStrongBinder(pointer); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<IInterface, T>::value, status_t>::type read(
            const Parcel& parcel, sp<T>* pointer) const {
        return callParcel("readNullableStrongBinder[IInterface]",
                          [&]() { return parcel.readNullableStrongBinder(pointer); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<IInterface, T>::value, status_t>::type write(
            Parcel* parcel, const sp<T>& interface) const {
        return write(parcel, IInterface::asBinder(interface));
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<Parcelable, T>::value, status_t>::type read(
            const Parcel& parcel, std::vector<T>* v) const {
        return callParcel("readParcelableVector", [&]() { return parcel.readParcelableVector(v); });
    }
    template <typename T>
    typename std::enable_if<std::is_base_of<Parcelable, T>::value, status_t>::type write(
            Parcel* parcel, const std::vector<T>& v) const {
        return callParcel("writeParcelableVector",
                          [&]() { return parcel->writeParcelableVector(v); });
    }

    // Templates to handle integral types. We use a struct template to require that the called
    // function exactly matches the signedness and size of the argument (e.g., the argument isn't
    // silently widened).
    template <bool isSigned, size_t size, typename I>
    struct HandleInt;
    template <typename I>
    struct HandleInt<true, 4, I> {
        static status_t read(const ParcelHandler& handler, const Parcel& parcel, I* i) {
            return handler.callParcel("readInt32", [&]() { return parcel.readInt32(i); });
        }
        static status_t write(const ParcelHandler& handler, Parcel* parcel, I i) {
            return handler.callParcel("writeInt32", [&]() { return parcel->writeInt32(i); });
        }
    };
    template <typename I>
    struct HandleInt<false, 4, I> {
        static status_t read(const ParcelHandler& handler, const Parcel& parcel, I* i) {
            return handler.callParcel("readUint32", [&]() { return parcel.readUint32(i); });
        }
        static status_t write(const ParcelHandler& handler, Parcel* parcel, I i) {
            return handler.callParcel("writeUint32", [&]() { return parcel->writeUint32(i); });
        }
    };
    template <typename I>
    struct HandleInt<true, 8, I> {
        static status_t read(const ParcelHandler& handler, const Parcel& parcel, I* i) {
            return handler.callParcel("readInt64", [&]() { return parcel.readInt64(i); });
        }
        static status_t write(const ParcelHandler& handler, Parcel* parcel, I i) {
            return handler.callParcel("writeInt64", [&]() { return parcel->writeInt64(i); });
        }
    };
    template <typename I>
    struct HandleInt<false, 8, I> {
        static status_t read(const ParcelHandler& handler, const Parcel& parcel, I* i) {
            return handler.callParcel("readUint64", [&]() { return parcel.readUint64(i); });
        }
        static status_t write(const ParcelHandler& handler, Parcel* parcel, I i) {
            return handler.callParcel("writeUint64", [&]() { return parcel->writeUint64(i); });
        }
    };
    template <typename I>
    typename std::enable_if<std::is_integral<I>::value, status_t>::type read(const Parcel& parcel,
                                                                             I* i) const {
        return HandleInt<std::is_signed<I>::value, sizeof(I), I>::read(*this, parcel, i);
    }
    template <typename I>
    typename std::enable_if<std::is_integral<I>::value, status_t>::type write(Parcel* parcel,
                                                                              I i) const {
        return HandleInt<std::is_signed<I>::value, sizeof(I), I>::write(*this, parcel, i);
    }

private:
    const char* const mLogTag;

    // Helper to encapsulate error handling while calling the various Parcel methods
    template <typename Function>
    status_t callParcel(const char* name, Function f) const {
        status_t error = f();
        if (CC_UNLIKELY(error != NO_ERROR)) {
            ALOG(LOG_ERROR, mLogTag, "Failed to %s, (%d: %s)", name, error, strerror(-error));
#if SI_DUMP_CALLSTACKS
            CallStack callStack(mLogTag);
#endif
        }
        return error;
    }
};

// Utility struct template which allows us to retrieve the types of the parameters of a member
// function pointer
template <typename T>
struct ParamExtractor;
template <typename Class, typename Return, typename... Params>
struct ParamExtractor<Return (Class::*)(Params...)> {
    using ParamTuple = std::tuple<Params...>;
};
template <typename Class, typename Return, typename... Params>
struct ParamExtractor<Return (Class::*)(Params...) const> {
    using ParamTuple = std::tuple<Params...>;
};

} // namespace SafeInterface

template <typename Interface>
class SafeBpInterface : public BpInterface<Interface> {
protected:
    SafeBpInterface(const sp<IBinder>& impl, const char* logTag)
          : BpInterface<Interface>(impl), mLogTag(logTag) {}
    ~SafeBpInterface() override = default;

    // callRemote is used to invoke a synchronous procedure call over Binder
    template <typename Method, typename TagType, typename... Args>
    status_t callRemote(TagType tag, Args&&... args) const {
        static_assert(sizeof(TagType) <= sizeof(uint32_t), "Tag must fit inside uint32_t");

        // Verify that the arguments are compatible with the parameters
        using ParamTuple = typename SafeInterface::ParamExtractor<Method>::ParamTuple;
        static_assert(ArgsMatchParams<std::tuple<Args...>, ParamTuple>::value,
                      "Invalid argument type");

        // Write the input arguments to the data Parcel
        Parcel data;
        data.writeInterfaceToken(this->getInterfaceDescriptor());

        status_t error = writeInputs(&data, std::forward<Args>(args)...);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            // A message will have been logged by writeInputs
            return error;
        }

        // Send the data Parcel to the remote and retrieve the reply parcel
        Parcel reply;
        error = this->remote()->transact(static_cast<uint32_t>(tag), data, &reply);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            ALOG(LOG_ERROR, mLogTag, "Failed to transact (%d)", error);
#if SI_DUMP_CALLSTACKS
            CallStack callStack(mLogTag);
#endif
            return error;
        }

        // Read the outputs from the reply Parcel into the output arguments
        error = readOutputs(reply, std::forward<Args>(args)...);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            // A message will have been logged by readOutputs
            return error;
        }

        // Retrieve the result code from the reply Parcel
        status_t result = NO_ERROR;
        error = reply.readInt32(&result);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            ALOG(LOG_ERROR, mLogTag, "Failed to obtain result");
#if SI_DUMP_CALLSTACKS
            CallStack callStack(mLogTag);
#endif
            return error;
        }
        return result;
    }

    // callRemoteAsync is used to invoke an asynchronous procedure call over Binder
    template <typename Method, typename TagType, typename... Args>
    void callRemoteAsync(TagType tag, Args&&... args) const {
        static_assert(sizeof(TagType) <= sizeof(uint32_t), "Tag must fit inside uint32_t");

        // Verify that the arguments are compatible with the parameters
        using ParamTuple = typename SafeInterface::ParamExtractor<Method>::ParamTuple;
        static_assert(ArgsMatchParams<std::tuple<Args...>, ParamTuple>::value,
                      "Invalid argument type");

        // Write the input arguments to the data Parcel
        Parcel data;
        data.writeInterfaceToken(this->getInterfaceDescriptor());
        status_t error = writeInputs(&data, std::forward<Args>(args)...);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            // A message will have been logged by writeInputs
            return;
        }

        // There will be no data in the reply Parcel since the call is one-way
        Parcel reply;
        error = this->remote()->transact(static_cast<uint32_t>(tag), data, &reply,
                                         IBinder::FLAG_ONEWAY);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            ALOG(LOG_ERROR, mLogTag, "Failed to transact (%d)", error);
#if SI_DUMP_CALLSTACKS
            CallStack callStack(mLogTag);
#endif
        }
    }

private:
    const char* const mLogTag;

    // This struct provides information on whether the decayed types of the elements at Index in the
    // tuple types T and U (that is, the types after stripping cv-qualifiers, removing references,
    // and a few other less common operations) are the same
    template <size_t Index, typename T, typename U>
    struct DecayedElementsMatch {
    private:
        using FirstT = typename std::tuple_element<Index, T>::type;
        using DecayedT = typename std::decay<FirstT>::type;
        using FirstU = typename std::tuple_element<Index, U>::type;
        using DecayedU = typename std::decay<FirstU>::type;

    public:
        static constexpr bool value = std::is_same<DecayedT, DecayedU>::value;
    };

    // When comparing whether the argument types match the parameter types, we first decay them (see
    // DecayedElementsMatch) to avoid falsely flagging, say, T&& against T even though they are
    // equivalent enough for our purposes
    template <typename T, typename U>
    struct ArgsMatchParams {};
    template <typename... Args, typename... Params>
    struct ArgsMatchParams<std::tuple<Args...>, std::tuple<Params...>> {
        static_assert(sizeof...(Args) <= sizeof...(Params), "Too many arguments");
        static_assert(sizeof...(Args) >= sizeof...(Params), "Not enough arguments");

    private:
        template <size_t Index>
        static constexpr typename std::enable_if<(Index < sizeof...(Args)), bool>::type
        elementsMatch() {
            if (!DecayedElementsMatch<Index, std::tuple<Args...>, std::tuple<Params...>>::value) {
                return false;
            }
            return elementsMatch<Index + 1>();
        }
        template <size_t Index>
        static constexpr typename std::enable_if<(Index >= sizeof...(Args)), bool>::type
        elementsMatch() {
            return true;
        }

    public:
        static constexpr bool value = elementsMatch<0>();
    };

    // Since we assume that pointer arguments are outputs, we can use this template struct to
    // determine whether or not a given argument is fundamentally a pointer type and thus an output
    template <typename T>
    struct IsPointerIfDecayed {
    private:
        using Decayed = typename std::decay<T>::type;

    public:
        static constexpr bool value = std::is_pointer<Decayed>::value;
    };

    template <typename T>
    typename std::enable_if<!IsPointerIfDecayed<T>::value, status_t>::type writeIfInput(
            Parcel* data, T&& t) const {
        return SafeInterface::ParcelHandler{mLogTag}.write(data, std::forward<T>(t));
    }
    template <typename T>
    typename std::enable_if<IsPointerIfDecayed<T>::value, status_t>::type writeIfInput(
            Parcel* /*data*/, T&& /*t*/) const {
        return NO_ERROR;
    }

    // This method iterates through all of the arguments, writing them to the data Parcel if they
    // are an input (i.e., if they are not a pointer type)
    template <typename T, typename... Remaining>
    status_t writeInputs(Parcel* data, T&& t, Remaining&&... remaining) const {
        status_t error = writeIfInput(data, std::forward<T>(t));
        if (CC_UNLIKELY(error != NO_ERROR)) {
            // A message will have been logged by writeIfInput
            return error;
        }
        return writeInputs(data, std::forward<Remaining>(remaining)...);
    }
    static status_t writeInputs(Parcel* /*data*/) { return NO_ERROR; }

    template <typename T>
    typename std::enable_if<IsPointerIfDecayed<T>::value, status_t>::type readIfOutput(
            const Parcel& reply, T&& t) const {
        return SafeInterface::ParcelHandler{mLogTag}.read(reply, std::forward<T>(t));
    }
    template <typename T>
    static typename std::enable_if<!IsPointerIfDecayed<T>::value, status_t>::type readIfOutput(
            const Parcel& /*reply*/, T&& /*t*/) {
        return NO_ERROR;
    }

    // Similar to writeInputs except that it reads output arguments from the reply Parcel
    template <typename T, typename... Remaining>
    status_t readOutputs(const Parcel& reply, T&& t, Remaining&&... remaining) const {
        status_t error = readIfOutput(reply, std::forward<T>(t));
        if (CC_UNLIKELY(error != NO_ERROR)) {
            // A message will have been logged by readIfOutput
            return error;
        }
        return readOutputs(reply, std::forward<Remaining>(remaining)...);
    }
    static status_t readOutputs(const Parcel& /*data*/) { return NO_ERROR; }
};

template <typename Interface>
class SafeBnInterface : public BnInterface<Interface> {
public:
    explicit SafeBnInterface(const char* logTag) : mLogTag(logTag) {}

protected:
    template <typename Method>
    status_t callLocal(const Parcel& data, Parcel* reply, Method method) {
        CHECK_INTERFACE(this, data, reply);

        // Since we need to both pass inputs into the call as well as retrieve outputs, we create a
        // "raw" tuple, where the inputs are interleaved with actual, non-pointer versions of the
        // outputs. When we ultimately call into the method, we will pass the addresses of the
        // output arguments instead of their tuple members directly, but the storage will live in
        // the tuple.
        using ParamTuple = typename SafeInterface::ParamExtractor<Method>::ParamTuple;
        typename RawConverter<std::tuple<>, ParamTuple>::type rawArgs{};

        // Read the inputs from the data Parcel into the argument tuple
        status_t error = InputReader<ParamTuple>{mLogTag}.readInputs(data, &rawArgs);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            // A message will have been logged by read
            return error;
        }

        // Call the local method
        status_t result = MethodCaller<ParamTuple>::call(this, method, &rawArgs);

        // Extract the outputs from the argument tuple and write them into the reply Parcel
        error = OutputWriter<ParamTuple>{mLogTag}.writeOutputs(reply, &rawArgs);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            // A message will have been logged by write
            return error;
        }

        // Return the result code in the reply Parcel
        error = reply->writeInt32(result);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            ALOG(LOG_ERROR, mLogTag, "Failed to write result");
#if SI_DUMP_CALLSTACKS
            CallStack callStack(mLogTag);
#endif
            return error;
        }
        return NO_ERROR;
    }

    template <typename Method>
    status_t callLocalAsync(const Parcel& data, Parcel* /*reply*/, Method method) {
        // reply is not actually used by CHECK_INTERFACE
        CHECK_INTERFACE(this, data, reply);

        // Since we need to both pass inputs into the call as well as retrieve outputs, we create a
        // "raw" tuple, where the inputs are interleaved with actual, non-pointer versions of the
        // outputs. When we ultimately call into the method, we will pass the addresses of the
        // output arguments instead of their tuple members directly, but the storage will live in
        // the tuple.
        using ParamTuple = typename SafeInterface::ParamExtractor<Method>::ParamTuple;
        typename RawConverter<std::tuple<>, ParamTuple>::type rawArgs{};

        // Read the inputs from the data Parcel into the argument tuple
        status_t error = InputReader<ParamTuple>{mLogTag}.readInputs(data, &rawArgs);
        if (CC_UNLIKELY(error != NO_ERROR)) {
            // A message will have been logged by read
            return error;
        }

        // Call the local method
        MethodCaller<ParamTuple>::callVoid(this, method, &rawArgs);

        // After calling, there is nothing more to do since asynchronous calls do not return a value
        // to the caller
        return NO_ERROR;
    }

private:
    const char* const mLogTag;

    // RemoveFirst strips the first element from a tuple.
    // For example, given T = std::tuple<A, B, C>, RemoveFirst<T>::type = std::tuple<B, C>
    template <typename T, typename... Args>
    struct RemoveFirst;
    template <typename T, typename... Args>
    struct RemoveFirst<std::tuple<T, Args...>> {
        using type = std::tuple<Args...>;
    };

    // RawConverter strips a tuple down to its fundamental types, discarding both pointers and
    // references. This allows us to allocate storage for both input (non-pointer) arguments and
    // output (pointer) arguments in one tuple.
    // For example, given T = std::tuple<const A&, B*>, RawConverter<T>::type = std::tuple<A, B>
    template <typename Unconverted, typename... Converted>
    struct RawConverter;
    template <typename Unconverted, typename... Converted>
    struct RawConverter<std::tuple<Converted...>, Unconverted> {
    private:
        using ElementType = typename std::tuple_element<0, Unconverted>::type;
        using Decayed = typename std::decay<ElementType>::type;
        using WithoutPointer = typename std::remove_pointer<Decayed>::type;

    public:
        using type = typename RawConverter<std::tuple<Converted..., WithoutPointer>,
                                           typename RemoveFirst<Unconverted>::type>::type;
    };
    template <typename... Converted>
    struct RawConverter<std::tuple<Converted...>, std::tuple<>> {
        using type = std::tuple<Converted...>;
    };

    // This provides a simple way to determine whether the indexed element of Args... is a pointer
    template <size_t I, typename... Args>
    struct ElementIsPointer {
    private:
        using ElementType = typename std::tuple_element<I, std::tuple<Args...>>::type;

    public:
        static constexpr bool value = std::is_pointer<ElementType>::value;
    };

    // This class iterates over the parameter types, and if a given parameter is an input
    // (i.e., is not a pointer), reads the corresponding argument tuple element from the data Parcel
    template <typename... Params>
    class InputReader;
    template <typename... Params>
    class InputReader<std::tuple<Params...>> {
    public:
        explicit InputReader(const char* logTag) : mLogTag(logTag) {}

        // Note that in this case (as opposed to in SafeBpInterface), we iterate using an explicit
        // index (starting with 0 here) instead of using recursion and stripping the first element.
        // This is because in SafeBpInterface we aren't actually operating on a real tuple, but are
        // instead just using a tuple as a convenient container for variadic types, whereas here we
        // can't modify the argument tuple without causing unnecessary copies or moves of the data
        // contained therein.
        template <typename RawTuple>
        status_t readInputs(const Parcel& data, RawTuple* args) {
            return dispatchArg<0>(data, args);
        }

    private:
        const char* const mLogTag;

        template <std::size_t I, typename RawTuple>
        typename std::enable_if<!ElementIsPointer<I, Params...>::value, status_t>::type readIfInput(
                const Parcel& data, RawTuple* args) {
            return SafeInterface::ParcelHandler{mLogTag}.read(data, &std::get<I>(*args));
        }
        template <std::size_t I, typename RawTuple>
        typename std::enable_if<ElementIsPointer<I, Params...>::value, status_t>::type readIfInput(
                const Parcel& /*data*/, RawTuple* /*args*/) {
            return NO_ERROR;
        }

        // Recursively iterate through the arguments
        template <std::size_t I, typename RawTuple>
        typename std::enable_if<(I < sizeof...(Params)), status_t>::type dispatchArg(
                const Parcel& data, RawTuple* args) {
            status_t error = readIfInput<I>(data, args);
            if (CC_UNLIKELY(error != NO_ERROR)) {
                // A message will have been logged in read
                return error;
            }
            return dispatchArg<I + 1>(data, args);
        }
        template <std::size_t I, typename RawTuple>
        typename std::enable_if<(I >= sizeof...(Params)), status_t>::type dispatchArg(
                const Parcel& /*data*/, RawTuple* /*args*/) {
            return NO_ERROR;
        }
    };

    // getForCall uses the types of the parameters to determine whether a given element of the
    // argument tuple is an input, which should be passed directly into the call, or an output, for
    // which its address should be passed into the call
    template <size_t I, typename RawTuple, typename... Params>
    static typename std::enable_if<
            ElementIsPointer<I, Params...>::value,
            typename std::tuple_element<I, std::tuple<Params...>>::type>::type
    getForCall(RawTuple* args) {
        return &std::get<I>(*args);
    }
    template <size_t I, typename RawTuple, typename... Params>
    static typename std::enable_if<
            !ElementIsPointer<I, Params...>::value,
            typename std::tuple_element<I, std::tuple<Params...>>::type>::type&
    getForCall(RawTuple* args) {
        return std::get<I>(*args);
    }

    // This template class uses std::index_sequence and parameter pack expansion to call the given
    // method using the elements of the argument tuple (after those arguments are passed through
    // getForCall to get addresses instead of values for output arguments)
    template <typename... Params>
    struct MethodCaller;
    template <typename... Params>
    struct MethodCaller<std::tuple<Params...>> {
    public:
        // The calls through these to the helper methods are necessary to generate the
        // std::index_sequences used to unpack the argument tuple into the method call
        template <typename Class, typename MemberFunction, typename RawTuple>
        static status_t call(Class* instance, MemberFunction function, RawTuple* args) {
            return callHelper(instance, function, args, std::index_sequence_for<Params...>{});
        }
        template <typename Class, typename MemberFunction, typename RawTuple>
        static void callVoid(Class* instance, MemberFunction function, RawTuple* args) {
            callVoidHelper(instance, function, args, std::index_sequence_for<Params...>{});
        }

    private:
        template <typename Class, typename MemberFunction, typename RawTuple, std::size_t... I>
        static status_t callHelper(Class* instance, MemberFunction function, RawTuple* args,
                                   std::index_sequence<I...> /*unused*/) {
            return (instance->*function)(getForCall<I, RawTuple, Params...>(args)...);
        }
        template <typename Class, typename MemberFunction, typename RawTuple, std::size_t... I>
        static void callVoidHelper(Class* instance, MemberFunction function, RawTuple* args,
                                   std::index_sequence<I...> /*unused*/) {
            (instance->*function)(getForCall<I, RawTuple, Params...>(args)...);
        }
    };

    // This class iterates over the parameter types, and if a given parameter is an output
    // (i.e., is a pointer), writes the corresponding argument tuple element into the reply Parcel
    template <typename... Params>
    struct OutputWriter;
    template <typename... Params>
    struct OutputWriter<std::tuple<Params...>> {
    public:
        explicit OutputWriter(const char* logTag) : mLogTag(logTag) {}

        // See the note on InputReader::readInputs for why this differs from the arguably simpler
        // RemoveFirst approach in SafeBpInterface
        template <typename RawTuple>
        status_t writeOutputs(Parcel* reply, RawTuple* args) {
            return dispatchArg<0>(reply, args);
        }

    private:
        const char* const mLogTag;

        template <std::size_t I, typename RawTuple>
        typename std::enable_if<ElementIsPointer<I, Params...>::value, status_t>::type
        writeIfOutput(Parcel* reply, RawTuple* args) {
            return SafeInterface::ParcelHandler{mLogTag}.write(reply, std::get<I>(*args));
        }
        template <std::size_t I, typename RawTuple>
        typename std::enable_if<!ElementIsPointer<I, Params...>::value, status_t>::type
        writeIfOutput(Parcel* /*reply*/, RawTuple* /*args*/) {
            return NO_ERROR;
        }

        // Recursively iterate through the arguments
        template <std::size_t I, typename RawTuple>
        typename std::enable_if<(I < sizeof...(Params)), status_t>::type dispatchArg(
                Parcel* reply, RawTuple* args) {
            status_t error = writeIfOutput<I>(reply, args);
            if (CC_UNLIKELY(error != NO_ERROR)) {
                // A message will have been logged in read
                return error;
            }
            return dispatchArg<I + 1>(reply, args);
        }
        template <std::size_t I, typename RawTuple>
        typename std::enable_if<(I >= sizeof...(Params)), status_t>::type dispatchArg(
                Parcel* /*reply*/, RawTuple* /*args*/) {
            return NO_ERROR;
        }
    };
};

} // namespace android
