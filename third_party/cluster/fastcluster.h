//
// C++ standalone verion of fastcluster by Daniel Muellner
//
// Copyright: Daniel Muellner, 2011
//            Christoph Dalitz, 2018
// License:   BSD style license
//            (see the file LICENSE for details)
//

#ifndef fastclustercpp_H
#define fastclustercpp_H

//
// Assigns cluster labels (0, ..., nclust-1) to the n points such
// that the cluster result is split into nclust clusters.
//
// Input arguments:
//   n      = number of observables
//   merge  = clustering result in R format
//   nclust = number of clusters
// Output arguments:
//   labels = allocated integer array of size n for result
//
void cutree_k(int n, const int* merge, int nclust, int* labels);

//
// Assigns cluster labels (0, ..., nclust-1) to the n points such
// that the hierarchical clsutering is stopped at cluster distance cdist
//
// Input arguments:
//   n      = number of observables
//   merge  = clustering result in R format
//   height = cluster distance at each merge step
//   cdist  = cutoff cluster distance
// Output arguments:
//   labels = allocated integer array of size n for result
//
void cutree_cdist(int n, const int* merge, double* height, double cdist, int* labels);

//
// Hierarchical clustering with one of Daniel Muellner's fast algorithms
//
// Input arguments:
//   n       = number of observables
//   distmat = condensed distance matrix, i.e. an n*(n-1)/2 array representing
//             the upper triangle (without diagonal elements) of the distance
//             matrix, e.g. for n=4:
//               d00 d01 d02 d03
//               d10 d11 d12 d13   ->  d01 d02 d03 d12 d13 d23
//               d20 d21 d22 d23
//               d30 d31 d32 d33
//   method  = cluster metric (see enum method_code)
// Output arguments:
//   merge   = allocated (n-1)x2 matrix (2*(n-1) array) for storing result.
//             Result follows R hclust convention:
//              - observabe indices start with one
//              - merge[i][] contains the merged nodes in step i
//              - merge[i][j] is negative when the node is an atom
//   height  = allocated (n-1) array with distances at each merge step
// Return code:
//   0 = ok
//   1 = invalid method
//
int hclust_fast(int n, double* distmat, int method, int* merge, double* height);
enum hclust_fast_methods {
  HCLUST_METHOD_SINGLE = 0,
  HCLUST_METHOD_COMPLETE = 1,
  HCLUST_METHOD_AVERAGE = 2,
  HCLUST_METHOD_MEDIAN = 3,
  HCLUST_METHOD_CENTROID = 5,
};

void hclust_pdist(int n, int m, double* pts, double* out);
void cluster_points_centroid(int n, int m, double* pts, double dist, int* idx);


#endif
