import os
import numpy as np

from cffi import FFI
from common.ffi_wrapper import suffix

cluster_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
cluster_fn = os.path.join(cluster_dir, f"libfastcluster{suffix()}")

ffi = FFI()
ffi.cdef("""
int hclust_fast(int n, double* distmat, int method, int* merge, double* height);
void cutree_cdist(int n, const int* merge, double* height, double cdist, int* labels);
void hclust_pdist(int n, int m, double* pts, double* out);
void cluster_points_centroid(int n, int m, double* pts, double dist, int* idx);
""")

hclust = ffi.dlopen(cluster_fn)


def cluster_points_centroid(pts, dist):
  pts = np.ascontiguousarray(pts, dtype=np.float64)
  pts_ptr = ffi.cast("double *", pts.ctypes.data)
  n, m = pts.shape

  labels_ptr = ffi.new("int[]", n)
  hclust.cluster_points_centroid(n, m, pts_ptr, dist**2, labels_ptr)
  return list(labels_ptr)
