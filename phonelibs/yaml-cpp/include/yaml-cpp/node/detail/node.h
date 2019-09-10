#ifndef NODE_DETAIL_NODE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define NODE_DETAIL_NODE_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/emitterstyle.h"
#include "yaml-cpp/dll.h"
#include "yaml-cpp/node/type.h"
#include "yaml-cpp/node/ptr.h"
#include "yaml-cpp/node/detail/node_ref.h"
#include <set>

namespace YAML {
namespace detail {
class node {
 public:
  node() : m_pRef(new node_ref) {}
  node(const node&) = delete;
  node& operator=(const node&) = delete;

  bool is(const node& rhs) const { return m_pRef == rhs.m_pRef; }
  const node_ref* ref() const { return m_pRef.get(); }

  bool is_defined() const { return m_pRef->is_defined(); }
  const Mark& mark() const { return m_pRef->mark(); }
  NodeType::value type() const { return m_pRef->type(); }

  const std::string& scalar() const { return m_pRef->scalar(); }
  const std::string& tag() const { return m_pRef->tag(); }
  EmitterStyle::value style() const { return m_pRef->style(); }

  template <typename T>
  bool equals(const T& rhs, shared_memory_holder pMemory);
  bool equals(const char* rhs, shared_memory_holder pMemory);

  void mark_defined() {
    if (is_defined())
      return;

    m_pRef->mark_defined();
    for (nodes::iterator it = m_dependencies.begin();
         it != m_dependencies.end(); ++it)
      (*it)->mark_defined();
    m_dependencies.clear();
  }

  void add_dependency(node& rhs) {
    if (is_defined())
      rhs.mark_defined();
    else
      m_dependencies.insert(&rhs);
  }

  void set_ref(const node& rhs) {
    if (rhs.is_defined())
      mark_defined();
    m_pRef = rhs.m_pRef;
  }
  void set_data(const node& rhs) {
    if (rhs.is_defined())
      mark_defined();
    m_pRef->set_data(*rhs.m_pRef);
  }

  void set_mark(const Mark& mark) { m_pRef->set_mark(mark); }

  void set_type(NodeType::value type) {
    if (type != NodeType::Undefined)
      mark_defined();
    m_pRef->set_type(type);
  }
  void set_null() {
    mark_defined();
    m_pRef->set_null();
  }
  void set_scalar(const std::string& scalar) {
    mark_defined();
    m_pRef->set_scalar(scalar);
  }
  void set_tag(const std::string& tag) {
    mark_defined();
    m_pRef->set_tag(tag);
  }

  // style
  void set_style(EmitterStyle::value style) {
    mark_defined();
    m_pRef->set_style(style);
  }

  // size/iterator
  std::size_t size() const { return m_pRef->size(); }

  const_node_iterator begin() const {
    return static_cast<const node_ref&>(*m_pRef).begin();
  }
  node_iterator begin() { return m_pRef->begin(); }

  const_node_iterator end() const {
    return static_cast<const node_ref&>(*m_pRef).end();
  }
  node_iterator end() { return m_pRef->end(); }

  // sequence
  void push_back(node& node, shared_memory_holder pMemory) {
    m_pRef->push_back(node, pMemory);
    node.add_dependency(*this);
  }
  void insert(node& key, node& value, shared_memory_holder pMemory) {
    m_pRef->insert(key, value, pMemory);
    key.add_dependency(*this);
    value.add_dependency(*this);
  }

  // indexing
  template <typename Key>
  node* get(const Key& key, shared_memory_holder pMemory) const {
    // NOTE: this returns a non-const node so that the top-level Node can wrap
    // it, and returns a pointer so that it can be NULL (if there is no such
    // key).
    return static_cast<const node_ref&>(*m_pRef).get(key, pMemory);
  }
  template <typename Key>
  node& get(const Key& key, shared_memory_holder pMemory) {
    node& value = m_pRef->get(key, pMemory);
    value.add_dependency(*this);
    return value;
  }
  template <typename Key>
  bool remove(const Key& key, shared_memory_holder pMemory) {
    return m_pRef->remove(key, pMemory);
  }

  node* get(node& key, shared_memory_holder pMemory) const {
    // NOTE: this returns a non-const node so that the top-level Node can wrap
    // it, and returns a pointer so that it can be NULL (if there is no such
    // key).
    return static_cast<const node_ref&>(*m_pRef).get(key, pMemory);
  }
  node& get(node& key, shared_memory_holder pMemory) {
    node& value = m_pRef->get(key, pMemory);
    key.add_dependency(*this);
    value.add_dependency(*this);
    return value;
  }
  bool remove(node& key, shared_memory_holder pMemory) {
    return m_pRef->remove(key, pMemory);
  }

  // map
  template <typename Key, typename Value>
  void force_insert(const Key& key, const Value& value,
                    shared_memory_holder pMemory) {
    m_pRef->force_insert(key, value, pMemory);
  }

 private:
  shared_node_ref m_pRef;
  typedef std::set<node*> nodes;
  nodes m_dependencies;
};
}
}

#endif  // NODE_DETAIL_NODE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
