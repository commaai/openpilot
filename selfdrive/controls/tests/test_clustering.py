import time
import unittest
import numpy as np
from fastcluster import linkage_vector
from scipy.cluster import _hierarchy
from scipy.spatial.distance import pdist

from selfdrive.controls.lib.cluster.fastcluster_py import hclust, ffi
from selfdrive.controls.lib.cluster.fastcluster_py import cluster_points_centroid


def fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
  # supersimplified function to get fast clustering. Got it from scipy
  Z = np.asarray(Z, order='c')
  n = Z.shape[0] + 1
  T = np.zeros((n,), dtype='i')
  _hierarchy.cluster_dist(Z, T, float(t), int(n))
  return T


TRACK_PTS = np.array([[59.26000137, -9.35999966, -5.42500019],
                      [91.61999817, -0.31999999, -2.75],
                      [31.38000031, 0.40000001, -0.2],
                      [89.57999725, -8.07999992, -18.04999924],
                      [53.42000122, 0.63999999, -0.175],
                      [31.38000031, 0.47999999, -0.2],
                      [36.33999939, 0.16, -0.2],
                      [53.33999939, 0.95999998, -0.175],
                      [59.26000137, -9.76000023, -5.44999981],
                      [33.93999977, 0.40000001, -0.22499999],
                      [106.74000092, -5.76000023, -18.04999924]])

CORRECT_LINK = np.array([[2., 5., 0.07999998, 2.],
                         [4., 7., 0.32984889, 2.],
                         [0., 8., 0.40078104, 2.],
                         [6., 9., 2.41209933, 2.],
                         [11., 14., 3.76342275, 4.],
                         [12., 13., 13.02297651, 4.],
                         [1., 3., 17.27626057, 2.],
                         [10., 17., 17.92918845, 3.],
                         [15., 16., 23.68525366, 8.],
                         [18., 19., 52.52351319, 11.]])

CORRECT_LABELS = np.array([7, 1, 4, 2, 6, 4, 5, 6, 7, 5, 3], dtype=np.int32)


def plot_cluster(pts, idx_old, idx_new):
    import matplotlib.pyplot as plt
    m = 'Set1'

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(pts[:, 0], pts[:, 1], c=idx_old, cmap=m)
    plt.title("Old")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(pts[:, 0], pts[:, 1], c=idx_new, cmap=m)
    plt.title("New")
    plt.colorbar()

    plt.show()


def same_clusters(correct, other):
  correct = np.asarray(correct)
  other = np.asarray(other)
  if len(correct) != len(other):
    return False

  for i in range(len(correct)):
    c = np.where(correct == correct[i])
    o = np.where(other == other[i])
    if not np.array_equal(c, o):
      return False
  return True


class TestClustering(unittest.TestCase):
  def test_scipy_clustering(self):
    old_link = linkage_vector(TRACK_PTS, method='centroid')
    old_cluster_idxs = fcluster(old_link, 2.5, criterion='distance')

    np.testing.assert_allclose(old_link, CORRECT_LINK)
    np.testing.assert_allclose(old_cluster_idxs, CORRECT_LABELS)

  def test_pdist(self):
    pts = np.ascontiguousarray(TRACK_PTS, dtype=np.float64)
    pts_ptr = ffi.cast("double *", pts.ctypes.data)

    n, m = pts.shape
    out = np.zeros((n * (n - 1) // 2, ), dtype=np.float64)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    hclust.hclust_pdist(n, m, pts_ptr, out_ptr)

    np.testing.assert_allclose(out, np.power(pdist(TRACK_PTS), 2))

  def test_cpp_clustering(self):
    pts = np.ascontiguousarray(TRACK_PTS, dtype=np.float64)
    pts_ptr = ffi.cast("double *", pts.ctypes.data)
    n, m = pts.shape

    labels = np.zeros((n, ), dtype=np.int32)
    labels_ptr = ffi.cast("int *", labels.ctypes.data)
    hclust.cluster_points_centroid(n, m, pts_ptr, 2.5**2, labels_ptr)
    self.assertTrue(same_clusters(CORRECT_LABELS, labels))

  def test_cpp_wrapper_clustering(self):
    labels = cluster_points_centroid(TRACK_PTS, 2.5)
    self.assertTrue(same_clusters(CORRECT_LABELS, labels))

  def test_random_cluster(self):
    np.random.seed(1337)
    N = 1000

    t_old = 0.
    t_new = 0.

    for _ in range(N):
      n = int(np.random.uniform(2, 32))
      x = np.random.uniform(-10, 50, (n, 1))
      y = np.random.uniform(-5, 5, (n, 1))
      vrel = np.random.uniform(-5, 5, (n, 1))
      pts = np.hstack([x, y, vrel])

      t = time.time()
      old_link = linkage_vector(pts, method='centroid')
      old_cluster_idx = fcluster(old_link, 2.5, criterion='distance')
      t_old += time.time() - t

      t = time.time()
      cluster_idx = cluster_points_centroid(pts, 2.5)
      t_new += time.time() - t

      self.assertTrue(same_clusters(old_cluster_idx, cluster_idx))


if __name__ == "__main__":
  unittest.main()
