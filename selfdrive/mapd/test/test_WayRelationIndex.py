import unittest
from selfdrive.mapd.lib.WayRelationIndex import WayRelationIndex
from selfdrive.mapd.test.mock_data import mockWayCollection01


class TestWayRelationIndex(unittest.TestCase):
  def test_init_and_add(self):
    wrs = mockWayCollection01.way_relations
    wr_index = WayRelationIndex(wrs)

    # expected init logic, including add logic.
    edge_nodes_index_dict = {}
    full_nodes_index_dict = {}
    for wr in wrs:
      for node in wr.way.nodes:
        node_id = node.id
        full_nodes_index_dict[node_id] = full_nodes_index_dict.get(node_id, []) + [wr]
        if node_id in wr.edge_nodes_ids:
          edge_nodes_index_dict[node_id] = edge_nodes_index_dict.get(node_id, []) + [wr]

    # assert logic delivers same result
    self.assertDictEqual(edge_nodes_index_dict, wr_index._edge_nodes_index_dict)
    self.assertDictEqual(full_nodes_index_dict, wr_index._full_nodes_index_dict)
    self.assertEqual(len(wr_index._edge_nodes_index_dict), 586)
    self.assertEqual(len(wr_index._full_nodes_index_dict), 2342)

  def test_remove(self):
    wrs = mockWayCollection01.way_relations
    wr_index = WayRelationIndex(wrs)

    wr_to_remove = wrs[0]
    affected_full_node_ids = [nodesData.id for nodesData in wr_to_remove.way.nodes]
    affected_edge_node_ids = wr_to_remove.edge_nodes_ids

    initial_full_lists = [wr_index._full_nodes_index_dict[ndid] for ndid in affected_full_node_ids]
    initial_edge_lists = [wr_index._edge_nodes_index_dict[ndid] for ndid in affected_edge_node_ids]

    expected_final_full_lists = [[wr for wr in li if wr is not wr_to_remove] for li in initial_full_lists]
    expected_final_edge_lists = [[wr for wr in li if wr is not wr_to_remove] for li in initial_edge_lists]

    wr_index.remove(wr_to_remove)

    final_full_lists = [wr_index._full_nodes_index_dict[ndid] for ndid in affected_full_node_ids]
    final_edge_lists = [wr_index._edge_nodes_index_dict[ndid] for ndid in affected_edge_node_ids]

    for idx, li in enumerate(final_full_lists):
      self.assertListEqual(li, expected_final_full_lists[idx])

    for idx, li in enumerate(final_edge_lists):
      self.assertListEqual(li, expected_final_edge_lists[idx])

  def test_way_relations_with_edge_node_id(self):
    wr_index = WayRelationIndex([])
    ref_dict = {
      0: ["fake_wr1", "fake_wr2"],
      1: ["fake_wr3"],
      3: ["fake_wr4", "fake_wr5", "fake_wr6"],
    }
    wr_index._edge_nodes_index_dict = ref_dict

    for key, li in ref_dict.items():
      self.assertListEqual(li, wr_index.way_relations_with_edge_node_id(key))

  def test_way_relations_with_node_id(self):
    wr_index = WayRelationIndex([])
    ref_dict = {
      0: ["fake_wr1", "fake_wr2"],
      1: ["fake_wr3"],
      3: ["fake_wr4", "fake_wr5", "fake_wr6"],
    }
    wr_index._full_nodes_index_dict = ref_dict

    for key, li in ref_dict.items():
      self.assertListEqual(li, wr_index.way_relations_with_node_id(key))
