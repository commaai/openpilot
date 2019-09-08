#ifndef VALUE_DETAIL_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define VALUE_DETAIL_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/dll.h"
#include "yaml-cpp/node/ptr.h"
#include "yaml-cpp/node/detail/node_iterator.h"
#include <cstddef>
#include <iterator>

namespace YAML {
namespace detail {
struct iterator_value;

template <typename V>
class iterator_base : public std::iterator<std::forward_iterator_tag, V,
                                           std::ptrdiff_t, V*, V> {

 private:
  template <typename>
  friend class iterator_base;
  struct enabler {};
  typedef node_iterator base_type;

  struct proxy {
    explicit proxy(const V& x) : m_ref(x) {}
    V* operator->() { return std::addressof(m_ref); }
    operator V*() { return std::addressof(m_ref); }

    V m_ref;
  };

 public:
  typedef typename iterator_base::value_type value_type;

 public:
  iterator_base() : m_iterator(), m_pMemory() {}
  explicit iterator_base(base_type rhs, shared_memory_holder pMemory)
      : m_iterator(rhs), m_pMemory(pMemory) {}

  template <class W>
  iterator_base(const iterator_base<W>& rhs,
                typename std::enable_if<std::is_convertible<W*, V*>::value,
                                        enabler>::type = enabler())
      : m_iterator(rhs.m_iterator), m_pMemory(rhs.m_pMemory) {}

  iterator_base<V>& operator++() {
    ++m_iterator;
    return *this;
  }

  iterator_base<V> operator++(int) {
    iterator_base<V> iterator_pre(*this);
    ++(*this);
    return iterator_pre;
  }

  template <typename W>
  bool operator==(const iterator_base<W>& rhs) const {
    return m_iterator == rhs.m_iterator;
  }

  template <typename W>
  bool operator!=(const iterator_base<W>& rhs) const {
    return m_iterator != rhs.m_iterator;
  }

  value_type operator*() const {
    const typename base_type::value_type& v = *m_iterator;
    if (v.pNode)
      return value_type(Node(*v, m_pMemory));
    if (v.first && v.second)
      return value_type(Node(*v.first, m_pMemory), Node(*v.second, m_pMemory));
    return value_type();
  }

  proxy operator->() const { return proxy(**this); }

 private:
  base_type m_iterator;
  shared_memory_holder m_pMemory;
};
}
}

#endif  // VALUE_DETAIL_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
