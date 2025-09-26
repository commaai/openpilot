// Copyright (c) 2018 Kenton Varda and contributors
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

#include "table.h"
#include "hash.h"

KJ_BEGIN_HEADER

namespace kj {

template <typename Key, typename Value>
class HashMap {
  // A key/value mapping backed by hashing.
  //
  // `Key` must be hashable (via a `.hashCode()` method or `KJ_HASHCODE()`; see `hash.h`) and must
  // implement `operator==()`. Additionally, when performing lookups, you can use key types other
  // than `Key` as long as the other type is also hashable (producing the same hash codes) and
  // there is an `operator==` implementation with `Key` on the left and that other type on the
  // right. For example, if the key type is `String`, you can pass `StringPtr` to `find()`.

public:
  void reserve(size_t size);
  // Pre-allocates space for a map of the given size.

  size_t size() const;
  size_t capacity() const;
  void clear();

  struct Entry {
    Key key;
    Value value;
  };

  Entry* begin();
  Entry* end();
  const Entry* begin() const;
  const Entry* end() const;
  // Deterministic iteration. If you only ever insert(), iteration order will be insertion order.
  // If you erase(), the erased element is swapped with the last element in the ordering.

  Entry& insert(Key key, Value value);
  // Inserts a new entry. Throws if the key already exists.

  template <typename Collection>
  void insertAll(Collection&& collection);
  // Given an iterable collection of `Entry`s, inserts all of them into this map. If the
  // input is an rvalue, the entries will be moved rather than copied.

  template <typename UpdateFunc>
  Entry& upsert(Key key, Value value, UpdateFunc&& update);
  Entry& upsert(Key key, Value value);
  // Tries to insert a new entry. However, if a duplicate already exists (according to some index),
  // then update(Value& existingValue, Value&& newValue) is called to modify the existing value.
  // If no function is provided, the default is to simply replace the value (but not the key).

  template <typename KeyLike>
  kj::Maybe<Value&> find(KeyLike&& key);
  template <typename KeyLike>
  kj::Maybe<const Value&> find(KeyLike&& key) const;
  // Search for a matching key. The input does not have to be of type `Key`; it merely has to
  // be something that the Hasher accepts.
  //
  // Note that the default hasher for String accepts StringPtr.

  template <typename KeyLike, typename Func>
  Value& findOrCreate(KeyLike&& key, Func&& createEntry);
  // Like find() but if the key isn't present then call createEntry() to create the corresponding
  // entry and insert it. createEntry() must return type `Entry`.

  template <typename KeyLike>
  kj::Maybe<Entry&> findEntry(KeyLike&& key);
  template <typename KeyLike>
  kj::Maybe<const Entry&> findEntry(KeyLike&& key) const;
  template <typename KeyLike, typename Func>
  Entry& findOrCreateEntry(KeyLike&& key, Func&& createEntry);
  // Sometimes you need to see the whole matching Entry, not just the Value.

  template <typename KeyLike>
  bool erase(KeyLike&& key);
  // Erase the entry with the matching key.
  //
  // WARNING: This invalidates all pointers and iterators into the map. Use eraseAll() if you need
  //   to iterate and erase multiple entries.

  void erase(Entry& entry);
  // Erase an entry by reference.

  Entry release(Entry& row);
  // Erase an entry and return its content by move.

  template <typename Predicate,
      typename = decltype(instance<Predicate>()(instance<Key&>(), instance<Value&>()))>
  size_t eraseAll(Predicate&& predicate);
  // Erase all values for which predicate(key, value) returns true. This scans over the entire map.

private:
  class Callbacks {
  public:
    inline const Key& keyForRow(const Entry& entry) const { return entry.key; }
    inline Key& keyForRow(Entry& entry) const { return entry.key; }

    template <typename KeyLike>
    inline bool matches(Entry& e, KeyLike&& key) const {
      return e.key == key;
    }
    template <typename KeyLike>
    inline bool matches(const Entry& e, KeyLike&& key) const {
      return e.key == key;
    }
    template <typename KeyLike>
    inline auto hashCode(KeyLike&& key) const {
      return kj::hashCode(key);
    }
  };

  kj::Table<Entry, HashIndex<Callbacks>> table;
};

template <typename Key, typename Value>
class TreeMap {
  // A key/value mapping backed by a B-tree.
  //
  // `Key` must support `operator<` and `operator==` against other Keys, and against any type
  // which you might want to pass to find() (with `Key` always on the left of the comparison).

public:
  void reserve(size_t size);
  // Pre-allocates space for a map of the given size.

  size_t size() const;
  size_t capacity() const;
  void clear();

  struct Entry {
    Key key;
    Value value;
  };

  auto begin();
  auto end();
  auto begin() const;
  auto end() const;
  // Iteration is in sorted order by key.

  Entry& insert(Key key, Value value);
  // Inserts a new entry. Throws if the key already exists.

  template <typename Collection>
  void insertAll(Collection&& collection);
  // Given an iterable collection of `Entry`s, inserts all of them into this map. If the
  // input is an rvalue, the entries will be moved rather than copied.

  template <typename UpdateFunc>
  Entry& upsert(Key key, Value value, UpdateFunc&& update);
  Entry& upsert(Key key, Value value);
  // Tries to insert a new entry. However, if a duplicate already exists (according to some index),
  // then update(Value& existingValue, Value&& newValue) is called to modify the existing value.
  // If no function is provided, the default is to simply replace the value (but not the key).

  template <typename KeyLike>
  kj::Maybe<Value&> find(KeyLike&& key);
  template <typename KeyLike>
  kj::Maybe<const Value&> find(KeyLike&& key) const;
  // Search for a matching key. The input does not have to be of type `Key`; it merely has to
  // be something that can be compared against `Key`.

  template <typename KeyLike, typename Func>
  Value& findOrCreate(KeyLike&& key, Func&& createEntry);
  // Like find() but if the key isn't present then call createEntry() to create the corresponding
  // entry and insert it. createEntry() must return type `Entry`.

  template <typename KeyLike>
  kj::Maybe<Entry&> findEntry(KeyLike&& key);
  template <typename KeyLike>
  kj::Maybe<const Entry&> findEntry(KeyLike&& key) const;
  template <typename KeyLike, typename Func>
  Entry& findOrCreateEntry(KeyLike&& key, Func&& createEntry);
  // Sometimes you need to see the whole matching Entry, not just the Value.

  template <typename K1, typename K2>
  auto range(K1&& k1, K2&& k2);
  template <typename K1, typename K2>
  auto range(K1&& k1, K2&& k2) const;
  // Returns an iterable range of entries with keys between k1 (inclusive) and k2 (exclusive).

  template <typename KeyLike>
  bool erase(KeyLike&& key);
  // Erase the entry with the matching key.
  //
  // WARNING: This invalidates all pointers and iterators into the map. Use eraseAll() if you need
  //   to iterate and erase multiple entries.

  void erase(Entry& entry);
  // Erase an entry by reference.

  Entry release(Entry& row);
  // Erase an entry and return its content by move.

  template <typename Predicate,
      typename = decltype(instance<Predicate>()(instance<Key&>(), instance<Value&>()))>
  size_t eraseAll(Predicate&& predicate);
  // Erase all values for which predicate(key, value) returns true. This scans over the entire map.

  template <typename K1, typename K2>
  size_t eraseRange(K1&& k1, K2&& k2);
  // Erases all entries with keys between k1 (inclusive) and k2 (exclusive).

private:
  class Callbacks {
  public:
    inline const Key& keyForRow(const Entry& entry) const { return entry.key; }
    inline Key& keyForRow(Entry& entry) const { return entry.key; }

    template <typename KeyLike>
    inline bool matches(Entry& e, KeyLike&& key) const {
      return e.key == key;
    }
    template <typename KeyLike>
    inline bool matches(const Entry& e, KeyLike&& key) const {
      return e.key == key;
    }
    template <typename KeyLike>
    inline bool isBefore(Entry& e, KeyLike&& key) const {
      return e.key < key;
    }
    template <typename KeyLike>
    inline bool isBefore(const Entry& e, KeyLike&& key) const {
      return e.key < key;
    }
  };

  kj::Table<Entry, TreeIndex<Callbacks>> table;
};

namespace _ {  // private

class HashSetCallbacks {
public:
  template <typename Row>
  inline Row& keyForRow(Row& row) const { return row; }

  template <typename T, typename U>
  inline bool matches(T& a, U& b) const { return a == b; }
  template <typename KeyLike>
  inline auto hashCode(KeyLike&& key) const {
    return kj::hashCode(key);
  }
};

class TreeSetCallbacks {
public:
  template <typename Row>
  inline Row& keyForRow(Row& row) const { return row; }

  template <typename T, typename U>
  inline bool matches(T& a, U& b) const { return a == b; }
  template <typename T, typename U>
  inline bool isBefore(T& a, U& b) const { return a < b; }
};

}  // namespace _ (private)

template <typename Element>
class HashSet: public Table<Element, HashIndex<_::HashSetCallbacks>> {
  // A simple hashtable-based set, using kj::hashCode() and operator==().

public:
  // Everything is inherited.

  template <typename... Params>
  inline bool contains(Params&&... params) const {
    return this->find(kj::fwd<Params>(params)...) != nullptr;
  }
};

template <typename Element>
class TreeSet: public Table<Element, TreeIndex<_::TreeSetCallbacks>> {
  // A simple b-tree-based set, using operator<() and operator==().

public:
  // Everything is inherited.
};

// =======================================================================================
// inline implementation details

template <typename Key, typename Value>
void HashMap<Key, Value>::reserve(size_t size) {
  table.reserve(size);
}

template <typename Key, typename Value>
size_t HashMap<Key, Value>::size() const {
  return table.size();
}
template <typename Key, typename Value>
size_t HashMap<Key, Value>::capacity() const {
  return table.capacity();
}
template <typename Key, typename Value>
void HashMap<Key, Value>::clear() {
  return table.clear();
}

template <typename Key, typename Value>
typename HashMap<Key, Value>::Entry* HashMap<Key, Value>::begin() {
  return table.begin();
}
template <typename Key, typename Value>
typename HashMap<Key, Value>::Entry* HashMap<Key, Value>::end() {
  return table.end();
}
template <typename Key, typename Value>
const typename HashMap<Key, Value>::Entry* HashMap<Key, Value>::begin() const {
  return table.begin();
}
template <typename Key, typename Value>
const typename HashMap<Key, Value>::Entry* HashMap<Key, Value>::end() const {
  return table.end();
}

template <typename Key, typename Value>
typename HashMap<Key, Value>::Entry& HashMap<Key, Value>::insert(Key key, Value value) {
  return table.insert(Entry { kj::mv(key), kj::mv(value) });
}

template <typename Key, typename Value>
template <typename Collection>
void HashMap<Key, Value>::insertAll(Collection&& collection) {
  return table.insertAll(kj::fwd<Collection>(collection));
}

template <typename Key, typename Value>
template <typename UpdateFunc>
typename HashMap<Key, Value>::Entry& HashMap<Key, Value>::upsert(
    Key key, Value value, UpdateFunc&& update) {
  return table.upsert(Entry { kj::mv(key), kj::mv(value) },
      [&](Entry& existingEntry, Entry&& newEntry) {
    update(existingEntry.value, kj::mv(newEntry.value));
  });
}

template <typename Key, typename Value>
typename HashMap<Key, Value>::Entry& HashMap<Key, Value>::upsert(
    Key key, Value value) {
  return table.upsert(Entry { kj::mv(key), kj::mv(value) },
      [&](Entry& existingEntry, Entry&& newEntry) {
    existingEntry.value = kj::mv(newEntry.value);
  });
}

template <typename Key, typename Value>
template <typename KeyLike>
kj::Maybe<Value&> HashMap<Key, Value>::find(KeyLike&& key) {
  return table.find(key).map([](Entry& e) -> Value& { return e.value; });
}
template <typename Key, typename Value>
template <typename KeyLike>
kj::Maybe<const Value&> HashMap<Key, Value>::find(KeyLike&& key) const {
  return table.find(key).map([](const Entry& e) -> const Value& { return e.value; });
}

template <typename Key, typename Value>
template <typename KeyLike, typename Func>
Value& HashMap<Key, Value>::findOrCreate(KeyLike&& key, Func&& createEntry) {
  return table.findOrCreate(key, kj::fwd<Func>(createEntry)).value;
}

template <typename Key, typename Value>
template <typename KeyLike>
kj::Maybe<typename HashMap<Key, Value>::Entry&>
HashMap<Key, Value>::findEntry(KeyLike&& key) {
  return table.find(kj::fwd<KeyLike>(key));
}
template <typename Key, typename Value>
template <typename KeyLike>
kj::Maybe<const typename HashMap<Key, Value>::Entry&>
HashMap<Key, Value>::findEntry(KeyLike&& key) const {
  return table.find(kj::fwd<KeyLike>(key));
}
template <typename Key, typename Value>
template <typename KeyLike, typename Func>
typename HashMap<Key, Value>::Entry&
HashMap<Key, Value>::findOrCreateEntry(KeyLike&& key, Func&& createEntry) {
  return table.findOrCreate(kj::fwd<KeyLike>(key), kj::fwd<Func>(createEntry));
}

template <typename Key, typename Value>
template <typename KeyLike>
bool HashMap<Key, Value>::erase(KeyLike&& key) {
  return table.eraseMatch(key);
}

template <typename Key, typename Value>
void HashMap<Key, Value>::erase(Entry& entry) {
  table.erase(entry);
}

template <typename Key, typename Value>
typename HashMap<Key, Value>::Entry HashMap<Key, Value>::release(Entry& entry) {
  return table.release(entry);
}

template <typename Key, typename Value>
template <typename Predicate, typename>
size_t HashMap<Key, Value>::eraseAll(Predicate&& predicate) {
  return table.eraseAll([&](Entry& entry) {
    return predicate(entry.key, entry.value);
  });
}

// -----------------------------------------------------------------------------

template <typename Key, typename Value>
void TreeMap<Key, Value>::reserve(size_t size) {
  table.reserve(size);
}

template <typename Key, typename Value>
size_t TreeMap<Key, Value>::size() const {
  return table.size();
}
template <typename Key, typename Value>
size_t TreeMap<Key, Value>::capacity() const {
  return table.capacity();
}
template <typename Key, typename Value>
void TreeMap<Key, Value>::clear() {
  return table.clear();
}

template <typename Key, typename Value>
auto TreeMap<Key, Value>::begin() {
  return table.ordered().begin();
}
template <typename Key, typename Value>
auto TreeMap<Key, Value>::end() {
  return table.ordered().end();
}
template <typename Key, typename Value>
auto TreeMap<Key, Value>::begin() const {
  return table.ordered().begin();
}
template <typename Key, typename Value>
auto TreeMap<Key, Value>::end() const {
  return table.ordered().end();
}

template <typename Key, typename Value>
typename TreeMap<Key, Value>::Entry& TreeMap<Key, Value>::insert(Key key, Value value) {
  return table.insert(Entry { kj::mv(key), kj::mv(value) });
}

template <typename Key, typename Value>
template <typename Collection>
void TreeMap<Key, Value>::insertAll(Collection&& collection) {
  return table.insertAll(kj::fwd<Collection>(collection));
}

template <typename Key, typename Value>
template <typename UpdateFunc>
typename TreeMap<Key, Value>::Entry& TreeMap<Key, Value>::upsert(
    Key key, Value value, UpdateFunc&& update) {
  return table.upsert(Entry { kj::mv(key), kj::mv(value) },
      [&](Entry& existingEntry, Entry&& newEntry) {
    update(existingEntry.value, kj::mv(newEntry.value));
  });
}

template <typename Key, typename Value>
typename TreeMap<Key, Value>::Entry& TreeMap<Key, Value>::upsert(
    Key key, Value value) {
  return table.upsert(Entry { kj::mv(key), kj::mv(value) },
      [&](Entry& existingEntry, Entry&& newEntry) {
    existingEntry.value = kj::mv(newEntry.value);
  });
}

template <typename Key, typename Value>
template <typename KeyLike>
kj::Maybe<Value&> TreeMap<Key, Value>::find(KeyLike&& key) {
  return table.find(key).map([](Entry& e) -> Value& { return e.value; });
}
template <typename Key, typename Value>
template <typename KeyLike>
kj::Maybe<const Value&> TreeMap<Key, Value>::find(KeyLike&& key) const {
  return table.find(key).map([](const Entry& e) -> const Value& { return e.value; });
}

template <typename Key, typename Value>
template <typename KeyLike, typename Func>
Value& TreeMap<Key, Value>::findOrCreate(KeyLike&& key, Func&& createEntry) {
  return table.findOrCreate(key, kj::fwd<Func>(createEntry)).value;
}

template <typename Key, typename Value>
template <typename KeyLike>
kj::Maybe<typename TreeMap<Key, Value>::Entry&>
TreeMap<Key, Value>::findEntry(KeyLike&& key) {
  return table.find(kj::fwd<KeyLike>(key));
}
template <typename Key, typename Value>
template <typename KeyLike>
kj::Maybe<const typename TreeMap<Key, Value>::Entry&>
TreeMap<Key, Value>::findEntry(KeyLike&& key) const {
  return table.find(kj::fwd<KeyLike>(key));
}
template <typename Key, typename Value>
template <typename KeyLike, typename Func>
typename TreeMap<Key, Value>::Entry&
TreeMap<Key, Value>::findOrCreateEntry(KeyLike&& key, Func&& createEntry) {
  return table.findOrCreate(kj::fwd<KeyLike>(key), kj::fwd<Func>(createEntry));
}

template <typename Key, typename Value>
template <typename K1, typename K2>
auto TreeMap<Key, Value>::range(K1&& k1, K2&& k2) {
  return table.range(kj::fwd<K1>(k1), kj::fwd<K2>(k2));
}
template <typename Key, typename Value>
template <typename K1, typename K2>
auto TreeMap<Key, Value>::range(K1&& k1, K2&& k2) const {
  return table.range(kj::fwd<K1>(k1), kj::fwd<K2>(k2));
}

template <typename Key, typename Value>
template <typename KeyLike>
bool TreeMap<Key, Value>::erase(KeyLike&& key) {
  return table.eraseMatch(key);
}

template <typename Key, typename Value>
void TreeMap<Key, Value>::erase(Entry& entry) {
  table.erase(entry);
}

template <typename Key, typename Value>
typename TreeMap<Key, Value>::Entry TreeMap<Key, Value>::release(Entry& entry) {
  return table.release(entry);
}

template <typename Key, typename Value>
template <typename Predicate, typename>
size_t TreeMap<Key, Value>::eraseAll(Predicate&& predicate) {
  return table.eraseAll([&](Entry& entry) {
    return predicate(entry.key, entry.value);
  });
}

template <typename Key, typename Value>
template <typename K1, typename K2>
size_t TreeMap<Key, Value>::eraseRange(K1&& k1, K2&& k2) {
  return table.eraseRange(kj::fwd<K1>(k1), kj::fwd<K2>(k2));
}

} // namespace kj

KJ_END_HEADER
