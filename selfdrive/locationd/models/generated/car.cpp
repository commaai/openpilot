#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

double mass;

void set_mass(double x){ mass = x;}

double rotational_inertia;

void set_rotational_inertia(double x){ rotational_inertia = x;}

double center_to_front;

void set_center_to_front(double x){ center_to_front = x;}

double center_to_rear;

void set_center_to_rear(double x){ center_to_rear = x;}

double stiffness_front;

void set_stiffness_front(double x){ stiffness_front = x;}

double stiffness_rear;

void set_stiffness_rear(double x){ stiffness_rear = x;}
const static double MAHA_THRESH_25 = 3.8414588206941227;
const static double MAHA_THRESH_24 = 5.991464547107981;
const static double MAHA_THRESH_30 = 3.8414588206941227;
const static double MAHA_THRESH_26 = 3.8414588206941227;
const static double MAHA_THRESH_27 = 3.8414588206941227;
const static double MAHA_THRESH_29 = 3.8414588206941227;
const static double MAHA_THRESH_28 = 3.8414588206941227;
const static double MAHA_THRESH_31 = 3.8414588206941227;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_3990407830206607826) {
   out_3990407830206607826[0] = delta_x[0] + nom_x[0];
   out_3990407830206607826[1] = delta_x[1] + nom_x[1];
   out_3990407830206607826[2] = delta_x[2] + nom_x[2];
   out_3990407830206607826[3] = delta_x[3] + nom_x[3];
   out_3990407830206607826[4] = delta_x[4] + nom_x[4];
   out_3990407830206607826[5] = delta_x[5] + nom_x[5];
   out_3990407830206607826[6] = delta_x[6] + nom_x[6];
   out_3990407830206607826[7] = delta_x[7] + nom_x[7];
   out_3990407830206607826[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_596620799835131520) {
   out_596620799835131520[0] = -nom_x[0] + true_x[0];
   out_596620799835131520[1] = -nom_x[1] + true_x[1];
   out_596620799835131520[2] = -nom_x[2] + true_x[2];
   out_596620799835131520[3] = -nom_x[3] + true_x[3];
   out_596620799835131520[4] = -nom_x[4] + true_x[4];
   out_596620799835131520[5] = -nom_x[5] + true_x[5];
   out_596620799835131520[6] = -nom_x[6] + true_x[6];
   out_596620799835131520[7] = -nom_x[7] + true_x[7];
   out_596620799835131520[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_4704152346071038473) {
   out_4704152346071038473[0] = 1.0;
   out_4704152346071038473[1] = 0.0;
   out_4704152346071038473[2] = 0.0;
   out_4704152346071038473[3] = 0.0;
   out_4704152346071038473[4] = 0.0;
   out_4704152346071038473[5] = 0.0;
   out_4704152346071038473[6] = 0.0;
   out_4704152346071038473[7] = 0.0;
   out_4704152346071038473[8] = 0.0;
   out_4704152346071038473[9] = 0.0;
   out_4704152346071038473[10] = 1.0;
   out_4704152346071038473[11] = 0.0;
   out_4704152346071038473[12] = 0.0;
   out_4704152346071038473[13] = 0.0;
   out_4704152346071038473[14] = 0.0;
   out_4704152346071038473[15] = 0.0;
   out_4704152346071038473[16] = 0.0;
   out_4704152346071038473[17] = 0.0;
   out_4704152346071038473[18] = 0.0;
   out_4704152346071038473[19] = 0.0;
   out_4704152346071038473[20] = 1.0;
   out_4704152346071038473[21] = 0.0;
   out_4704152346071038473[22] = 0.0;
   out_4704152346071038473[23] = 0.0;
   out_4704152346071038473[24] = 0.0;
   out_4704152346071038473[25] = 0.0;
   out_4704152346071038473[26] = 0.0;
   out_4704152346071038473[27] = 0.0;
   out_4704152346071038473[28] = 0.0;
   out_4704152346071038473[29] = 0.0;
   out_4704152346071038473[30] = 1.0;
   out_4704152346071038473[31] = 0.0;
   out_4704152346071038473[32] = 0.0;
   out_4704152346071038473[33] = 0.0;
   out_4704152346071038473[34] = 0.0;
   out_4704152346071038473[35] = 0.0;
   out_4704152346071038473[36] = 0.0;
   out_4704152346071038473[37] = 0.0;
   out_4704152346071038473[38] = 0.0;
   out_4704152346071038473[39] = 0.0;
   out_4704152346071038473[40] = 1.0;
   out_4704152346071038473[41] = 0.0;
   out_4704152346071038473[42] = 0.0;
   out_4704152346071038473[43] = 0.0;
   out_4704152346071038473[44] = 0.0;
   out_4704152346071038473[45] = 0.0;
   out_4704152346071038473[46] = 0.0;
   out_4704152346071038473[47] = 0.0;
   out_4704152346071038473[48] = 0.0;
   out_4704152346071038473[49] = 0.0;
   out_4704152346071038473[50] = 1.0;
   out_4704152346071038473[51] = 0.0;
   out_4704152346071038473[52] = 0.0;
   out_4704152346071038473[53] = 0.0;
   out_4704152346071038473[54] = 0.0;
   out_4704152346071038473[55] = 0.0;
   out_4704152346071038473[56] = 0.0;
   out_4704152346071038473[57] = 0.0;
   out_4704152346071038473[58] = 0.0;
   out_4704152346071038473[59] = 0.0;
   out_4704152346071038473[60] = 1.0;
   out_4704152346071038473[61] = 0.0;
   out_4704152346071038473[62] = 0.0;
   out_4704152346071038473[63] = 0.0;
   out_4704152346071038473[64] = 0.0;
   out_4704152346071038473[65] = 0.0;
   out_4704152346071038473[66] = 0.0;
   out_4704152346071038473[67] = 0.0;
   out_4704152346071038473[68] = 0.0;
   out_4704152346071038473[69] = 0.0;
   out_4704152346071038473[70] = 1.0;
   out_4704152346071038473[71] = 0.0;
   out_4704152346071038473[72] = 0.0;
   out_4704152346071038473[73] = 0.0;
   out_4704152346071038473[74] = 0.0;
   out_4704152346071038473[75] = 0.0;
   out_4704152346071038473[76] = 0.0;
   out_4704152346071038473[77] = 0.0;
   out_4704152346071038473[78] = 0.0;
   out_4704152346071038473[79] = 0.0;
   out_4704152346071038473[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_2233432862498219035) {
   out_2233432862498219035[0] = state[0];
   out_2233432862498219035[1] = state[1];
   out_2233432862498219035[2] = state[2];
   out_2233432862498219035[3] = state[3];
   out_2233432862498219035[4] = state[4];
   out_2233432862498219035[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_2233432862498219035[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_2233432862498219035[7] = state[7];
   out_2233432862498219035[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8517999407047013030) {
   out_8517999407047013030[0] = 1;
   out_8517999407047013030[1] = 0;
   out_8517999407047013030[2] = 0;
   out_8517999407047013030[3] = 0;
   out_8517999407047013030[4] = 0;
   out_8517999407047013030[5] = 0;
   out_8517999407047013030[6] = 0;
   out_8517999407047013030[7] = 0;
   out_8517999407047013030[8] = 0;
   out_8517999407047013030[9] = 0;
   out_8517999407047013030[10] = 1;
   out_8517999407047013030[11] = 0;
   out_8517999407047013030[12] = 0;
   out_8517999407047013030[13] = 0;
   out_8517999407047013030[14] = 0;
   out_8517999407047013030[15] = 0;
   out_8517999407047013030[16] = 0;
   out_8517999407047013030[17] = 0;
   out_8517999407047013030[18] = 0;
   out_8517999407047013030[19] = 0;
   out_8517999407047013030[20] = 1;
   out_8517999407047013030[21] = 0;
   out_8517999407047013030[22] = 0;
   out_8517999407047013030[23] = 0;
   out_8517999407047013030[24] = 0;
   out_8517999407047013030[25] = 0;
   out_8517999407047013030[26] = 0;
   out_8517999407047013030[27] = 0;
   out_8517999407047013030[28] = 0;
   out_8517999407047013030[29] = 0;
   out_8517999407047013030[30] = 1;
   out_8517999407047013030[31] = 0;
   out_8517999407047013030[32] = 0;
   out_8517999407047013030[33] = 0;
   out_8517999407047013030[34] = 0;
   out_8517999407047013030[35] = 0;
   out_8517999407047013030[36] = 0;
   out_8517999407047013030[37] = 0;
   out_8517999407047013030[38] = 0;
   out_8517999407047013030[39] = 0;
   out_8517999407047013030[40] = 1;
   out_8517999407047013030[41] = 0;
   out_8517999407047013030[42] = 0;
   out_8517999407047013030[43] = 0;
   out_8517999407047013030[44] = 0;
   out_8517999407047013030[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8517999407047013030[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8517999407047013030[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8517999407047013030[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8517999407047013030[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8517999407047013030[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8517999407047013030[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8517999407047013030[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8517999407047013030[53] = -9.8100000000000005*dt;
   out_8517999407047013030[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8517999407047013030[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8517999407047013030[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8517999407047013030[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8517999407047013030[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8517999407047013030[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8517999407047013030[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8517999407047013030[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8517999407047013030[62] = 0;
   out_8517999407047013030[63] = 0;
   out_8517999407047013030[64] = 0;
   out_8517999407047013030[65] = 0;
   out_8517999407047013030[66] = 0;
   out_8517999407047013030[67] = 0;
   out_8517999407047013030[68] = 0;
   out_8517999407047013030[69] = 0;
   out_8517999407047013030[70] = 1;
   out_8517999407047013030[71] = 0;
   out_8517999407047013030[72] = 0;
   out_8517999407047013030[73] = 0;
   out_8517999407047013030[74] = 0;
   out_8517999407047013030[75] = 0;
   out_8517999407047013030[76] = 0;
   out_8517999407047013030[77] = 0;
   out_8517999407047013030[78] = 0;
   out_8517999407047013030[79] = 0;
   out_8517999407047013030[80] = 1;
}
void h_25(double *state, double *unused, double *out_3158706812719367378) {
   out_3158706812719367378[0] = state[6];
}
void H_25(double *state, double *unused, double *out_2847211384764115064) {
   out_2847211384764115064[0] = 0;
   out_2847211384764115064[1] = 0;
   out_2847211384764115064[2] = 0;
   out_2847211384764115064[3] = 0;
   out_2847211384764115064[4] = 0;
   out_2847211384764115064[5] = 0;
   out_2847211384764115064[6] = 1;
   out_2847211384764115064[7] = 0;
   out_2847211384764115064[8] = 0;
}
void h_24(double *state, double *unused, double *out_2356632080120596950) {
   out_2356632080120596950[0] = state[4];
   out_2356632080120596950[1] = state[5];
}
void H_24(double *state, double *unused, double *out_3353821295087046362) {
   out_3353821295087046362[0] = 0;
   out_3353821295087046362[1] = 0;
   out_3353821295087046362[2] = 0;
   out_3353821295087046362[3] = 0;
   out_3353821295087046362[4] = 1;
   out_3353821295087046362[5] = 0;
   out_3353821295087046362[6] = 0;
   out_3353821295087046362[7] = 0;
   out_3353821295087046362[8] = 0;
   out_3353821295087046362[9] = 0;
   out_3353821295087046362[10] = 0;
   out_3353821295087046362[11] = 0;
   out_3353821295087046362[12] = 0;
   out_3353821295087046362[13] = 0;
   out_3353821295087046362[14] = 1;
   out_3353821295087046362[15] = 0;
   out_3353821295087046362[16] = 0;
   out_3353821295087046362[17] = 0;
}
void h_30(double *state, double *unused, double *out_1340624676461203017) {
   out_1340624676461203017[0] = state[4];
}
void H_30(double *state, double *unused, double *out_2717872437620874994) {
   out_2717872437620874994[0] = 0;
   out_2717872437620874994[1] = 0;
   out_2717872437620874994[2] = 0;
   out_2717872437620874994[3] = 0;
   out_2717872437620874994[4] = 1;
   out_2717872437620874994[5] = 0;
   out_2717872437620874994[6] = 0;
   out_2717872437620874994[7] = 0;
   out_2717872437620874994[8] = 0;
}
void h_26(double *state, double *unused, double *out_8343387380254268796) {
   out_8343387380254268796[0] = state[7];
}
void H_26(double *state, double *unused, double *out_894291934109941160) {
   out_894291934109941160[0] = 0;
   out_894291934109941160[1] = 0;
   out_894291934109941160[2] = 0;
   out_894291934109941160[3] = 0;
   out_894291934109941160[4] = 0;
   out_894291934109941160[5] = 0;
   out_894291934109941160[6] = 0;
   out_894291934109941160[7] = 1;
   out_894291934109941160[8] = 0;
}
void h_27(double *state, double *unused, double *out_121920761275034693) {
   out_121920761275034693[0] = state[3];
}
void H_27(double *state, double *unused, double *out_543109125820450083) {
   out_543109125820450083[0] = 0;
   out_543109125820450083[1] = 0;
   out_543109125820450083[2] = 0;
   out_543109125820450083[3] = 1;
   out_543109125820450083[4] = 0;
   out_543109125820450083[5] = 0;
   out_543109125820450083[6] = 0;
   out_543109125820450083[7] = 0;
   out_543109125820450083[8] = 0;
}
void h_29(double *state, double *unused, double *out_397114823559540582) {
   out_397114823559540582[0] = state[1];
}
void H_29(double *state, double *unused, double *out_3228103781935267178) {
   out_3228103781935267178[0] = 0;
   out_3228103781935267178[1] = 1;
   out_3228103781935267178[2] = 0;
   out_3228103781935267178[3] = 0;
   out_3228103781935267178[4] = 0;
   out_3228103781935267178[5] = 0;
   out_3228103781935267178[6] = 0;
   out_3228103781935267178[7] = 0;
   out_3228103781935267178[8] = 0;
}
void h_28(double *state, double *unused, double *out_5123594028935880338) {
   out_5123594028935880338[0] = state[0];
}
void H_28(double *state, double *unused, double *out_6252652618118631524) {
   out_6252652618118631524[0] = 1;
   out_6252652618118631524[1] = 0;
   out_6252652618118631524[2] = 0;
   out_6252652618118631524[3] = 0;
   out_6252652618118631524[4] = 0;
   out_6252652618118631524[5] = 0;
   out_6252652618118631524[6] = 0;
   out_6252652618118631524[7] = 0;
   out_6252652618118631524[8] = 0;
}
void h_31(double *state, double *unused, double *out_8517202034639833302) {
   out_8517202034639833302[0] = state[8];
}
void H_31(double *state, double *unused, double *out_2877857346641075492) {
   out_2877857346641075492[0] = 0;
   out_2877857346641075492[1] = 0;
   out_2877857346641075492[2] = 0;
   out_2877857346641075492[3] = 0;
   out_2877857346641075492[4] = 0;
   out_2877857346641075492[5] = 0;
   out_2877857346641075492[6] = 0;
   out_2877857346641075492[7] = 0;
   out_2877857346641075492[8] = 1;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
}
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<2, 3, 0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
}
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
}
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
}
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
}
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
}
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_31, H_31, NULL, in_z, in_R, in_ea, MAHA_THRESH_31);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_3990407830206607826) {
  err_fun(nom_x, delta_x, out_3990407830206607826);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_596620799835131520) {
  inv_err_fun(nom_x, true_x, out_596620799835131520);
}
void car_H_mod_fun(double *state, double *out_4704152346071038473) {
  H_mod_fun(state, out_4704152346071038473);
}
void car_f_fun(double *state, double dt, double *out_2233432862498219035) {
  f_fun(state,  dt, out_2233432862498219035);
}
void car_F_fun(double *state, double dt, double *out_8517999407047013030) {
  F_fun(state,  dt, out_8517999407047013030);
}
void car_h_25(double *state, double *unused, double *out_3158706812719367378) {
  h_25(state, unused, out_3158706812719367378);
}
void car_H_25(double *state, double *unused, double *out_2847211384764115064) {
  H_25(state, unused, out_2847211384764115064);
}
void car_h_24(double *state, double *unused, double *out_2356632080120596950) {
  h_24(state, unused, out_2356632080120596950);
}
void car_H_24(double *state, double *unused, double *out_3353821295087046362) {
  H_24(state, unused, out_3353821295087046362);
}
void car_h_30(double *state, double *unused, double *out_1340624676461203017) {
  h_30(state, unused, out_1340624676461203017);
}
void car_H_30(double *state, double *unused, double *out_2717872437620874994) {
  H_30(state, unused, out_2717872437620874994);
}
void car_h_26(double *state, double *unused, double *out_8343387380254268796) {
  h_26(state, unused, out_8343387380254268796);
}
void car_H_26(double *state, double *unused, double *out_894291934109941160) {
  H_26(state, unused, out_894291934109941160);
}
void car_h_27(double *state, double *unused, double *out_121920761275034693) {
  h_27(state, unused, out_121920761275034693);
}
void car_H_27(double *state, double *unused, double *out_543109125820450083) {
  H_27(state, unused, out_543109125820450083);
}
void car_h_29(double *state, double *unused, double *out_397114823559540582) {
  h_29(state, unused, out_397114823559540582);
}
void car_H_29(double *state, double *unused, double *out_3228103781935267178) {
  H_29(state, unused, out_3228103781935267178);
}
void car_h_28(double *state, double *unused, double *out_5123594028935880338) {
  h_28(state, unused, out_5123594028935880338);
}
void car_H_28(double *state, double *unused, double *out_6252652618118631524) {
  H_28(state, unused, out_6252652618118631524);
}
void car_h_31(double *state, double *unused, double *out_8517202034639833302) {
  h_31(state, unused, out_8517202034639833302);
}
void car_H_31(double *state, double *unused, double *out_2877857346641075492) {
  H_31(state, unused, out_2877857346641075492);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
void car_set_mass(double x) {
  set_mass(x);
}
void car_set_rotational_inertia(double x) {
  set_rotational_inertia(x);
}
void car_set_center_to_front(double x) {
  set_center_to_front(x);
}
void car_set_center_to_rear(double x) {
  set_center_to_rear(x);
}
void car_set_stiffness_front(double x) {
  set_stiffness_front(x);
}
void car_set_stiffness_rear(double x) {
  set_stiffness_rear(x);
}
}

const EKF car = {
  .name = "car",
  .kinds = { 25, 24, 30, 26, 27, 29, 28, 31 },
  .feature_kinds = {  },
  .f_fun = car_f_fun,
  .F_fun = car_F_fun,
  .err_fun = car_err_fun,
  .inv_err_fun = car_inv_err_fun,
  .H_mod_fun = car_H_mod_fun,
  .predict = car_predict,
  .hs = {
    { 25, car_h_25 },
    { 24, car_h_24 },
    { 30, car_h_30 },
    { 26, car_h_26 },
    { 27, car_h_27 },
    { 29, car_h_29 },
    { 28, car_h_28 },
    { 31, car_h_31 },
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
    { 31, car_H_31 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
    { 31, car_update_31 },
  },
  .Hes = {
  },
  .sets = {
    { "mass", car_set_mass },
    { "rotational_inertia", car_set_rotational_inertia },
    { "center_to_front", car_set_center_to_front },
    { "center_to_rear", car_set_center_to_rear },
    { "stiffness_front", car_set_stiffness_front },
    { "stiffness_rear", car_set_stiffness_rear },
  },
  .extra_routines = {
  },
};

ekf_lib_init(car)
