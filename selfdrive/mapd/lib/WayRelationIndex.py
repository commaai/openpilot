

class WayRelationIndex():
  """
  A class containing an index of WayRelations by node ids of internal nodes and edge nodes.
  """
  def __init__(self, way_relations):
    self._edge_nodes_index_dict = {}
    self._full_nodes_index_dict = {}

    for wr in way_relations:
      self.add(wr)

  def add(self, way_relation):
    for node in way_relation.way.nodes:
      node_id = node.id
      self._full_nodes_index_dict[node_id] = self._full_nodes_index_dict.get(node_id, []) + [way_relation]
      if node_id in way_relation.edge_nodes_ids:
        self._edge_nodes_index_dict[node_id] = self._edge_nodes_index_dict.get(node_id, []) + [way_relation]

  def remove(self, way_relation):
    for node in way_relation.way.nodes:
      node_id = node.id
      self._full_nodes_index_dict[node_id] = [wr for wr in self._full_nodes_index_dict.get(node_id, [])
                                              if wr is not way_relation]
      if node_id in way_relation.edge_nodes_ids:
        self._edge_nodes_index_dict[node_id] = [wr for wr in self._edge_nodes_index_dict.get(node_id, [])
                                                if wr is not way_relation]

  def way_relations_with_edge_node_id(self, node_id):
    return self._edge_nodes_index_dict.get(node_id, [])

  def way_relations_with_node_id(self, node_id):
    return self._full_nodes_index_dict.get(node_id, [])
