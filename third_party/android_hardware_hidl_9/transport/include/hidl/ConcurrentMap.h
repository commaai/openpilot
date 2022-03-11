/*
 * Copyright (C) 2016 The Android Open Source Project
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
#ifndef ANDROID_HIDL_CONCURRENT_MAP_H
#define ANDROID_HIDL_CONCURRENT_MAP_H

#include <mutex>
#include <map>

namespace android {
namespace hardware {

template<typename K, typename V>
class ConcurrentMap {
private:
    using size_type = typename std::map<K, V>::size_type;
    using iterator = typename std::map<K, V>::iterator;
    using const_iterator = typename std::map<K, V>::const_iterator;

public:
    void set(K &&k, V &&v) {
        std::unique_lock<std::mutex> _lock(mMutex);
        mMap[std::forward<K>(k)] = std::forward<V>(v);
    }

    // get with the given default value.
    const V &get(const K &k, const V &def) const {
        std::unique_lock<std::mutex> _lock(mMutex);
        const_iterator iter = mMap.find(k);
        if (iter == mMap.end()) {
            return def;
        }
        return iter->second;
    }

    size_type erase(const K &k) {
        std::unique_lock<std::mutex> _lock(mMutex);
        return mMap.erase(k);
    }

    size_type eraseIfEqual(const K& k, const V& v) {
        std::unique_lock<std::mutex> _lock(mMutex);
        const_iterator iter = mMap.find(k);
        if (iter == mMap.end()) {
            return 0;
        }
        if (iter->second == v) {
            mMap.erase(iter);
            return 1;
        } else {
            return 0;
        }
    }

    std::unique_lock<std::mutex> lock() { return std::unique_lock<std::mutex>(mMutex); }

    void setLocked(K&& k, V&& v) { mMap[std::forward<K>(k)] = std::forward<V>(v); }
    void setLocked(K&& k, const V& v) { mMap[std::forward<K>(k)] = v; }

    const V& getLocked(const K& k, const V& def) const {
        const_iterator iter = mMap.find(k);
        if (iter == mMap.end()) {
            return def;
        }
        return iter->second;
    }

    size_type eraseLocked(const K& k) { return mMap.erase(k); }

    // the concurrent map must be locked in order to iterate over it
    iterator begin() { return mMap.begin(); }
    iterator end() { return mMap.end(); }
    const_iterator begin() const { return mMap.begin(); }
    const_iterator end() const { return mMap.end(); }

   private:
    mutable std::mutex mMutex;
    std::map<K, V> mMap;
};

}  // namespace hardware
}  // namespace android


#endif  // ANDROID_HIDL_CONCURRENT_MAP_H
