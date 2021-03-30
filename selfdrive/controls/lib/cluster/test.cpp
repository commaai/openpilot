#include <cassert>

extern "C" {
#include "fastcluster.h"
}


int main(int argc, const char* argv[]){
  const int n = 11;
  const int m = 3;
  double* pts = new double[n*m]{59.26000137, -9.35999966, -5.42500019,
                                91.61999817, -0.31999999, -2.75,
                                31.38000031, 0.40000001, -0.2,
                                89.57999725, -8.07999992, -18.04999924,
                                53.42000122, 0.63999999, -0.175,
                                31.38000031, 0.47999999, -0.2,
                                36.33999939, 0.16, -0.2,
                                53.33999939, 0.95999998, -0.175,
                                59.26000137, -9.76000023, -5.44999981,
                                33.93999977, 0.40000001, -0.22499999,
                                106.74000092, -5.76000023, -18.04999924};

  int * idx = new int[n];
  int * correct_idx = new int[n]{0, 1, 2, 3, 4, 2, 5, 4, 0, 5, 6};

  cluster_points_centroid(n, m, pts, 2.5 * 2.5, idx);

  for (int i = 0; i < n; i++){
    assert(idx[i] == correct_idx[i]);
  }

  delete[] idx;
  delete[] correct_idx;
  delete[] pts;
}
