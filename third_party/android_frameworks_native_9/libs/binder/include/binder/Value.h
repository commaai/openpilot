/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef ANDROID_VALUE_H
#define ANDROID_VALUE_H

#include <stdint.h>
#include <map>
#include <set>
#include <vector>
#include <string>

#include <binder/Parcelable.h>
#include <binder/PersistableBundle.h>
#include <binder/Map.h>
#include <utils/String8.h>
#include <utils/String16.h>
#include <utils/StrongPointer.h>

namespace android {

class Parcel;

namespace binder {

/**
 * A limited C++ generic type. The purpose of this class is to allow C++
 * programs to make use of (or implement) Binder interfaces which make use
 * the Java "Object" generic type (either via the use of the Map type or
 * some other mechanism).
 *
 * This class only supports a limited set of types, but additional types
 * may be easily added to this class in the future as needed---without
 * breaking binary compatability.
 *
 * This class was written in such a way as to help avoid type errors by
 * giving each type their own explicity-named accessor methods (rather than
 * overloaded methods).
 *
 * When reading or writing this class to a Parcel, use the `writeValue()`
 * and `readValue()` methods.
 */
class Value {
public:
    Value();
    virtual ~Value();

    Value& swap(Value &);

    bool empty() const;

    void clear();

#ifdef LIBBINDER_VALUE_SUPPORTS_TYPE_INFO
    const std::type_info& type() const;
#endif

    int32_t parcelType() const;

    bool operator==(const Value& rhs) const;
    bool operator!=(const Value& rhs) const { return !this->operator==(rhs); }

    Value(const Value& value);
    Value(const bool& value);
    Value(const int8_t& value);
    Value(const int32_t& value);
    Value(const int64_t& value);
    Value(const double& value);
    Value(const String16& value);
    Value(const std::vector<bool>& value);
    Value(const std::vector<uint8_t>& value);
    Value(const std::vector<int32_t>& value);
    Value(const std::vector<int64_t>& value);
    Value(const std::vector<double>& value);
    Value(const std::vector<String16>& value);
    Value(const os::PersistableBundle& value);
    Value(const binder::Map& value);

    Value& operator=(const Value& rhs);
    Value& operator=(const int8_t& rhs);
    Value& operator=(const bool& rhs);
    Value& operator=(const int32_t& rhs);
    Value& operator=(const int64_t& rhs);
    Value& operator=(const double& rhs);
    Value& operator=(const String16& rhs);
    Value& operator=(const std::vector<bool>& rhs);
    Value& operator=(const std::vector<uint8_t>& rhs);
    Value& operator=(const std::vector<int32_t>& rhs);
    Value& operator=(const std::vector<int64_t>& rhs);
    Value& operator=(const std::vector<double>& rhs);
    Value& operator=(const std::vector<String16>& rhs);
    Value& operator=(const os::PersistableBundle& rhs);
    Value& operator=(const binder::Map& rhs);

    void putBoolean(const bool& value);
    void putByte(const int8_t& value);
    void putInt(const int32_t& value);
    void putLong(const int64_t& value);
    void putDouble(const double& value);
    void putString(const String16& value);
    void putBooleanVector(const std::vector<bool>& value);
    void putByteVector(const std::vector<uint8_t>& value);
    void putIntVector(const std::vector<int32_t>& value);
    void putLongVector(const std::vector<int64_t>& value);
    void putDoubleVector(const std::vector<double>& value);
    void putStringVector(const std::vector<String16>& value);
    void putPersistableBundle(const os::PersistableBundle& value);
    void putMap(const binder::Map& value);

    bool getBoolean(bool* out) const;
    bool getByte(int8_t* out) const;
    bool getInt(int32_t* out) const;
    bool getLong(int64_t* out) const;
    bool getDouble(double* out) const;
    bool getString(String16* out) const;
    bool getBooleanVector(std::vector<bool>* out) const;
    bool getByteVector(std::vector<uint8_t>* out) const;
    bool getIntVector(std::vector<int32_t>* out) const;
    bool getLongVector(std::vector<int64_t>* out) const;
    bool getDoubleVector(std::vector<double>* out) const;
    bool getStringVector(std::vector<String16>* out) const;
    bool getPersistableBundle(os::PersistableBundle* out) const;
    bool getMap(binder::Map* out) const;

    bool isBoolean() const;
    bool isByte() const;
    bool isInt() const;
    bool isLong() const;
    bool isDouble() const;
    bool isString() const;
    bool isBooleanVector() const;
    bool isByteVector() const;
    bool isIntVector() const;
    bool isLongVector() const;
    bool isDoubleVector() const;
    bool isStringVector() const;
    bool isPersistableBundle() const;
    bool isMap() const;

    // String Convenience Adapters
    // ---------------------------

    Value(const String8& value):               Value(String16(value)) { }
    Value(const ::std::string& value):         Value(String8(value.c_str())) { }
    void putString(const String8& value)       { return putString(String16(value)); }
    void putString(const ::std::string& value) { return putString(String8(value.c_str())); }
    Value& operator=(const String8& rhs)       { return *this = String16(rhs); }
    Value& operator=(const ::std::string& rhs) { return *this = String8(rhs.c_str()); }
    bool getString(String8* out) const;
    bool getString(::std::string* out) const;

private:

    // This allows ::android::Parcel to call the two methods below.
    friend class ::android::Parcel;

    // This is called by ::android::Parcel::writeValue()
    status_t writeToParcel(Parcel* parcel) const;

    // This is called by ::android::Parcel::readValue()
    status_t readFromParcel(const Parcel* parcel);

    template<typename T> class Content;
    class ContentBase;

    ContentBase* mContent;
};

}  // namespace binder

}  // namespace android

#endif  // ANDROID_VALUE_H
