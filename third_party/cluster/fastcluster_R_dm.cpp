//
// Excerpt from fastcluster_R.cpp
//
// Copyright: Daniel Müllner, 2011 <http://danifold.net>
//

struct pos_node {
  t_index pos;
  int node;
};

void order_nodes(const int N, const int * const merge, const t_index * const node_size, int * const order) {
  /* Parameters:
     N         : number of data points
     merge     : (N-1)×2 array which specifies the node indices which are
                 merged in each step of the clustering procedure.
                 Negative entries -1...-N point to singleton nodes, while
                 positive entries 1...(N-1) point to nodes which are themselves
                 parents of other nodes.
     node_size : array of node sizes - makes it easier
     order     : output array of size N

     Runtime: Θ(N)
  */
  std::stack<std::pair<int, int>> queue;
  queue.push({N-2, 0});

  int parent;
  int child;
  t_index pos;

  while (!queue.empty()) {
      std::tie(parent, pos) = queue.top();
      queue.pop();

      // First child
      child = merge[parent];
      if (child < 0) {
          order[pos] = -child;
          ++pos;
      } else {
          queue.push({child-1, pos});
          pos += node_size[child-1];
      }

      // Second child
      child = merge[parent+N-1];
      if (child < 0) {
          order[pos] = -child;
      } else {
          queue.push({child-1, pos});
      }
    }
}

#define size_(r_) ( ((r_<N) ? 1 : node_size[r_-N]) )

template <const bool sorted>
void generate_R_dendrogram(int * const merge, double * const height, int * const order, cluster_result & Z2, const int N) {
  // The array "nodes" is a union-find data structure for the cluster
  // identites (only needed for unsorted cluster_result input).
  union_find nodes(sorted ? 0 : N);
  if (!sorted) {
    std::stable_sort(Z2[0], Z2[N-1]);
  }

  t_index node1, node2;
  auto_array_ptr<t_index> node_size(N-1);

  for (t_index i=0; i<N-1; ++i) {
    // Get two data points whose clusters are merged in step i.
    // Find the cluster identifiers for these points.
    if (sorted) {
      node1 = Z2[i]->node1;
      node2 = Z2[i]->node2;
    }
    else {
      node1 = nodes.Find(Z2[i]->node1);
      node2 = nodes.Find(Z2[i]->node2);
      // Merge the nodes in the union-find data structure by making them
      // children of a new node.
      nodes.Union(node1, node2);
    }
    // Sort the nodes in the output array.
    if (node1>node2) {
      t_index tmp = node1;
      node1 = node2;
      node2 = tmp;
    }
    /* Conversion between labeling conventions.
       Input:  singleton nodes 0,...,N-1
               compound nodes  N,...,2N-2
       Output: singleton nodes -1,...,-N
               compound nodes  1,...,N
    */
    merge[i]     = (node1<N) ? -static_cast<int>(node1)-1
                              : static_cast<int>(node1)-N+1;
    merge[i+N-1] = (node2<N) ? -static_cast<int>(node2)-1
                              : static_cast<int>(node2)-N+1;
    height[i] = Z2[i]->dist;
    node_size[i] = size_(node1) + size_(node2);
  }

  order_nodes(N, merge, node_size, order);
}
