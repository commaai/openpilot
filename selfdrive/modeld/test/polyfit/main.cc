#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "data.h"

Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> vander;

void poly_init() {
  // Build Vandermonde matrix
  for(int i = 0; i < MODEL_PATH_DISTANCE; i++) {
    for(int j = 0; j < POLYFIT_DEGREE; j++) {
      vander(i, j) = pow(i, POLYFIT_DEGREE-j-1);
    }
  }
}

void poly_fit(float *in_pts, float *in_stds, float *out) {
  // References to inputs
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > pts(in_pts, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > std(in_stds, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, POLYFIT_DEGREE, 1> > p(out, POLYFIT_DEGREE);

  // Build Least Squares equations
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> lhs = vander.array().colwise() / std.array();
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> rhs = pts.array() / std.array();

  Eigen::Matrix<float, POLYFIT_DEGREE, 1> scale = 1. / (lhs.array()*lhs.array()).sqrt().colwise().sum();
  lhs = lhs * scale.asDiagonal();

  // Solve inplace
  Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXf> > qr(lhs);
  p = qr.solve(rhs);

  p = p.transpose() * scale.asDiagonal();
}

int main(void) {
  poly_init();


  float poly[4];
  poly_fit(pts, stds, poly);

  std::cout << "[";
  std::cout << poly[0] << ",";
  std::cout << poly[1] << ",";
  std::cout << poly[2] << ",";
  std::cout << poly[3];
  std::cout << "]" << std::endl;
}
