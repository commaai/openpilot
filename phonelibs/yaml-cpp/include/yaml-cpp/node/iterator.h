#ifndef VALUE_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define VALUE_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/dll.h"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/node/detail/iterator_fwd.h"
#include "yaml-cpp/node/detail/iterator.h"
#include <list>
#include <utility>
#include <vector>

namespace YAML {
namespace detail {
struct iterator_value : public Node, std::pair<Node, Node> {
  iterator_value() {}
  explicit iterator_value(const Node& rhs)
      : Node(rhs),
        std::pair<Node, Node>(Node(Node::ZombieNode), Node(Node::ZombieNode)) {}
  explicit iterator_value(const Node& key, const Node& value)
      : Node(Node::ZombieNode), std::pair<Node, Node>(key, value) {}
};
}
}

#endif  // VALUE_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
