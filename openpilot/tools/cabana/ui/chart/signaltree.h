#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tools/cabana/utils/strings.h"

namespace chart {

class SignalTree {
public:
  struct Node {
    std::string name, key, path;
    size_t parent = 0;
    int depth = -1;
    std::vector<size_t> children;
    size_t matches = 0;
    bool signal_matches = false;
    bool custom = false;
  };

  void rebuild(const std::vector<std::string> &paths, const std::unordered_set<std::string> &custom_paths = {}) {
    nodes = {Node{}};
    std::unordered_map<std::string, size_t> indices;
    for (const auto &path : paths) {
      size_t parent = 0;
      for (size_t start = path.find_first_not_of('/'); start != std::string::npos;) {
        const size_t end = path.find('/', start);
        const std::string key = path.substr(0, end);
        auto [it, inserted] = indices.emplace(key, nodes.size());
        const size_t index = it->second;
        if (inserted) {
          Node node;
          node.name = path.substr(start, end == std::string::npos ? end : end - start);
          node.key = key;
          node.parent = parent;
          node.depth = nodes[parent].depth + 1;
          nodes[parent].children.push_back(index);
          nodes.push_back(std::move(node));
        }
        nodes[index].custom |= custom_paths.count(path) != 0;
        parent = index;
        start = end == std::string::npos ? end : path.find_first_not_of('/', end);
      }
      if (parent) nodes[parent].path = path;
    }
    for (auto &node : nodes) {
      std::sort(node.children.begin(), node.children.end(), [&](size_t a, size_t b) {
        if (nodes[a].custom != nodes[b].custom) return nodes[a].custom;
        const auto &left = nodes[a].name, &right = nodes[b].name;
        if (nodes[a].custom) return left < right;
        const bool left_index = isIndex(left), right_index = isIndex(right);
        if (left_index != right_index) return left_index;
        if (left_index && left.size() != right.size()) return left.size() < right.size();
        return left < right;
      });
    }
  }

  void filter(const std::string &query) {
    for (auto &node : nodes) {
      node.signal_matches = !node.path.empty() && utils::containsCI(node.path, query);
      node.matches = node.signal_matches;
    }
    // Parents are inserted before their descendants, regardless of sibling order.
    for (size_t i = nodes.size(); i-- > 1;) nodes[nodes[i].parent].matches += nodes[i].matches;
  }

  std::vector<size_t> visible(const std::unordered_set<std::string> &expanded) const {
    std::vector<size_t> rows;
    std::vector<size_t> pending(nodes[0].children.rbegin(), nodes[0].children.rend());
    while (!pending.empty()) {
      const size_t i = pending.back();
      pending.pop_back();
      const auto &node = nodes[i];
      if (!node.matches) continue;
      rows.push_back(i);
      if (expanded.count(node.key)) pending.insert(pending.end(), node.children.rbegin(), node.children.rend());
    }
    return rows;
  }

  static bool isIndex(const std::string &name) {
    return !name.empty() && name.find_first_not_of("0123456789") == std::string::npos;
  }

  std::vector<Node> nodes{Node{}};
};

}  // namespace chart
