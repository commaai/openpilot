#include <Eigen/QR>
#include <Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, KDIM*2, 3, Eigen::RowMajor> R3M;
typedef Eigen::Matrix<double, KDIM*2, 1> R1M;
typedef Eigen::Matrix<double, 3, 1> O1M;
typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> M3D;

extern "C" {
void gauss_newton(double *in_x, double *in_poses, double *in_img_positions) {

  double res[KDIM*2] = {0};
  double jac[KDIM*6] = {0};

  O1M x(in_x);
  O1M delta;
  int counter = 0;
  while ((delta.squaredNorm() > 0.0001 and counter < 30) or counter == 0){
    res_fun(in_x, in_poses, in_img_positions, res);
    jac_fun(in_x, in_poses, in_img_positions, jac);
    R1M E(res); R3M J(jac);
    delta = (J.transpose()*J).inverse() * J.transpose() * E;
    x = x - delta;
    memcpy(in_x, x.data(), 3 * sizeof(double));
    counter = counter + 1;
  }
}


void compute_pos(double *to_c, double *poses, double *img_positions, double *param, double *pos) {
    param[0] = img_positions[KDIM*2-2];
    param[1] = img_positions[KDIM*2-1];
    param[2] = 0.1;
    gauss_newton(param, poses, img_positions);

    Eigen::Quaterniond q;
    q.w() = poses[KDIM*7-4];
    q.x() = poses[KDIM*7-3];
    q.y() = poses[KDIM*7-2];
    q.z() = poses[KDIM*7-1];
    M3D RC(to_c);
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    Eigen::Matrix3d rot = R * RC.transpose();

    pos[0] = param[0]/param[2];
    pos[1] = param[1]/param[2];
    pos[2] = 1.0/param[2];
    O1M ecef_offset(poses + KDIM*7-7);
    O1M ecef_output(pos);
    ecef_output = rot*ecef_output + ecef_offset;
    memcpy(pos, ecef_output.data(), 3 * sizeof(double));
}
}
