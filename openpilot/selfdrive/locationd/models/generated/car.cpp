#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

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
void err_fun(double *nom_x, double *delta_x, double *out_7661067751022070458) {
   out_7661067751022070458[0] = delta_x[0] + nom_x[0];
   out_7661067751022070458[1] = delta_x[1] + nom_x[1];
   out_7661067751022070458[2] = delta_x[2] + nom_x[2];
   out_7661067751022070458[3] = delta_x[3] + nom_x[3];
   out_7661067751022070458[4] = delta_x[4] + nom_x[4];
   out_7661067751022070458[5] = delta_x[5] + nom_x[5];
   out_7661067751022070458[6] = delta_x[6] + nom_x[6];
   out_7661067751022070458[7] = delta_x[7] + nom_x[7];
   out_7661067751022070458[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5136938064863750921) {
   out_5136938064863750921[0] = -nom_x[0] + true_x[0];
   out_5136938064863750921[1] = -nom_x[1] + true_x[1];
   out_5136938064863750921[2] = -nom_x[2] + true_x[2];
   out_5136938064863750921[3] = -nom_x[3] + true_x[3];
   out_5136938064863750921[4] = -nom_x[4] + true_x[4];
   out_5136938064863750921[5] = -nom_x[5] + true_x[5];
   out_5136938064863750921[6] = -nom_x[6] + true_x[6];
   out_5136938064863750921[7] = -nom_x[7] + true_x[7];
   out_5136938064863750921[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_3671347944419902585) {
   out_3671347944419902585[0] = 1.0;
   out_3671347944419902585[1] = 0.0;
   out_3671347944419902585[2] = 0.0;
   out_3671347944419902585[3] = 0.0;
   out_3671347944419902585[4] = 0.0;
   out_3671347944419902585[5] = 0.0;
   out_3671347944419902585[6] = 0.0;
   out_3671347944419902585[7] = 0.0;
   out_3671347944419902585[8] = 0.0;
   out_3671347944419902585[9] = 0.0;
   out_3671347944419902585[10] = 1.0;
   out_3671347944419902585[11] = 0.0;
   out_3671347944419902585[12] = 0.0;
   out_3671347944419902585[13] = 0.0;
   out_3671347944419902585[14] = 0.0;
   out_3671347944419902585[15] = 0.0;
   out_3671347944419902585[16] = 0.0;
   out_3671347944419902585[17] = 0.0;
   out_3671347944419902585[18] = 0.0;
   out_3671347944419902585[19] = 0.0;
   out_3671347944419902585[20] = 1.0;
   out_3671347944419902585[21] = 0.0;
   out_3671347944419902585[22] = 0.0;
   out_3671347944419902585[23] = 0.0;
   out_3671347944419902585[24] = 0.0;
   out_3671347944419902585[25] = 0.0;
   out_3671347944419902585[26] = 0.0;
   out_3671347944419902585[27] = 0.0;
   out_3671347944419902585[28] = 0.0;
   out_3671347944419902585[29] = 0.0;
   out_3671347944419902585[30] = 1.0;
   out_3671347944419902585[31] = 0.0;
   out_3671347944419902585[32] = 0.0;
   out_3671347944419902585[33] = 0.0;
   out_3671347944419902585[34] = 0.0;
   out_3671347944419902585[35] = 0.0;
   out_3671347944419902585[36] = 0.0;
   out_3671347944419902585[37] = 0.0;
   out_3671347944419902585[38] = 0.0;
   out_3671347944419902585[39] = 0.0;
   out_3671347944419902585[40] = 1.0;
   out_3671347944419902585[41] = 0.0;
   out_3671347944419902585[42] = 0.0;
   out_3671347944419902585[43] = 0.0;
   out_3671347944419902585[44] = 0.0;
   out_3671347944419902585[45] = 0.0;
   out_3671347944419902585[46] = 0.0;
   out_3671347944419902585[47] = 0.0;
   out_3671347944419902585[48] = 0.0;
   out_3671347944419902585[49] = 0.0;
   out_3671347944419902585[50] = 1.0;
   out_3671347944419902585[51] = 0.0;
   out_3671347944419902585[52] = 0.0;
   out_3671347944419902585[53] = 0.0;
   out_3671347944419902585[54] = 0.0;
   out_3671347944419902585[55] = 0.0;
   out_3671347944419902585[56] = 0.0;
   out_3671347944419902585[57] = 0.0;
   out_3671347944419902585[58] = 0.0;
   out_3671347944419902585[59] = 0.0;
   out_3671347944419902585[60] = 1.0;
   out_3671347944419902585[61] = 0.0;
   out_3671347944419902585[62] = 0.0;
   out_3671347944419902585[63] = 0.0;
   out_3671347944419902585[64] = 0.0;
   out_3671347944419902585[65] = 0.0;
   out_3671347944419902585[66] = 0.0;
   out_3671347944419902585[67] = 0.0;
   out_3671347944419902585[68] = 0.0;
   out_3671347944419902585[69] = 0.0;
   out_3671347944419902585[70] = 1.0;
   out_3671347944419902585[71] = 0.0;
   out_3671347944419902585[72] = 0.0;
   out_3671347944419902585[73] = 0.0;
   out_3671347944419902585[74] = 0.0;
   out_3671347944419902585[75] = 0.0;
   out_3671347944419902585[76] = 0.0;
   out_3671347944419902585[77] = 0.0;
   out_3671347944419902585[78] = 0.0;
   out_3671347944419902585[79] = 0.0;
   out_3671347944419902585[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7888740610946713678) {
   out_7888740610946713678[0] = state[0];
   out_7888740610946713678[1] = state[1];
   out_7888740610946713678[2] = state[2];
   out_7888740610946713678[3] = state[3];
   out_7888740610946713678[4] = state[4];
   out_7888740610946713678[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_7888740610946713678[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7888740610946713678[7] = state[7];
   out_7888740610946713678[8] = state[8];
}
void F_fun(double *state, double dt, double *out_3995884459612552931) {
   out_3995884459612552931[0] = 1;
   out_3995884459612552931[1] = 0;
   out_3995884459612552931[2] = 0;
   out_3995884459612552931[3] = 0;
   out_3995884459612552931[4] = 0;
   out_3995884459612552931[5] = 0;
   out_3995884459612552931[6] = 0;
   out_3995884459612552931[7] = 0;
   out_3995884459612552931[8] = 0;
   out_3995884459612552931[9] = 0;
   out_3995884459612552931[10] = 1;
   out_3995884459612552931[11] = 0;
   out_3995884459612552931[12] = 0;
   out_3995884459612552931[13] = 0;
   out_3995884459612552931[14] = 0;
   out_3995884459612552931[15] = 0;
   out_3995884459612552931[16] = 0;
   out_3995884459612552931[17] = 0;
   out_3995884459612552931[18] = 0;
   out_3995884459612552931[19] = 0;
   out_3995884459612552931[20] = 1;
   out_3995884459612552931[21] = 0;
   out_3995884459612552931[22] = 0;
   out_3995884459612552931[23] = 0;
   out_3995884459612552931[24] = 0;
   out_3995884459612552931[25] = 0;
   out_3995884459612552931[26] = 0;
   out_3995884459612552931[27] = 0;
   out_3995884459612552931[28] = 0;
   out_3995884459612552931[29] = 0;
   out_3995884459612552931[30] = 1;
   out_3995884459612552931[31] = 0;
   out_3995884459612552931[32] = 0;
   out_3995884459612552931[33] = 0;
   out_3995884459612552931[34] = 0;
   out_3995884459612552931[35] = 0;
   out_3995884459612552931[36] = 0;
   out_3995884459612552931[37] = 0;
   out_3995884459612552931[38] = 0;
   out_3995884459612552931[39] = 0;
   out_3995884459612552931[40] = 1;
   out_3995884459612552931[41] = 0;
   out_3995884459612552931[42] = 0;
   out_3995884459612552931[43] = 0;
   out_3995884459612552931[44] = 0;
   out_3995884459612552931[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_3995884459612552931[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_3995884459612552931[47] = -dt*stiffness_front*state[0]/state[1];
   out_3995884459612552931[48] = -dt*stiffness_front*state[0]/state[1];
   out_3995884459612552931[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_3995884459612552931[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_3995884459612552931[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_3995884459612552931[52] = dt*stiffness_front*state[0]/state[1];
   out_3995884459612552931[53] = -9.8100000000000005*dt;
   out_3995884459612552931[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_3995884459612552931[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_3995884459612552931[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3995884459612552931[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3995884459612552931[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_3995884459612552931[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_3995884459612552931[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_3995884459612552931[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3995884459612552931[62] = 0;
   out_3995884459612552931[63] = 0;
   out_3995884459612552931[64] = 0;
   out_3995884459612552931[65] = 0;
   out_3995884459612552931[66] = 0;
   out_3995884459612552931[67] = 0;
   out_3995884459612552931[68] = 0;
   out_3995884459612552931[69] = 0;
   out_3995884459612552931[70] = 1;
   out_3995884459612552931[71] = 0;
   out_3995884459612552931[72] = 0;
   out_3995884459612552931[73] = 0;
   out_3995884459612552931[74] = 0;
   out_3995884459612552931[75] = 0;
   out_3995884459612552931[76] = 0;
   out_3995884459612552931[77] = 0;
   out_3995884459612552931[78] = 0;
   out_3995884459612552931[79] = 0;
   out_3995884459612552931[80] = 1;
}
void h_25(double *state, double *unused, double *out_3968497935547833524) {
   out_3968497935547833524[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1232343880764762255) {
   out_1232343880764762255[0] = 0;
   out_1232343880764762255[1] = 0;
   out_1232343880764762255[2] = 0;
   out_1232343880764762255[3] = 0;
   out_1232343880764762255[4] = 0;
   out_1232343880764762255[5] = 0;
   out_1232343880764762255[6] = 1;
   out_1232343880764762255[7] = 0;
   out_1232343880764762255[8] = 0;
}
void h_24(double *state, double *unused, double *out_3248956284435332617) {
   out_3248956284435332617[0] = state[4];
   out_3248956284435332617[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6210330712223954065) {
   out_6210330712223954065[0] = 0;
   out_6210330712223954065[1] = 0;
   out_6210330712223954065[2] = 0;
   out_6210330712223954065[3] = 0;
   out_6210330712223954065[4] = 1;
   out_6210330712223954065[5] = 0;
   out_6210330712223954065[6] = 0;
   out_6210330712223954065[7] = 0;
   out_6210330712223954065[8] = 0;
   out_6210330712223954065[9] = 0;
   out_6210330712223954065[10] = 0;
   out_6210330712223954065[11] = 0;
   out_6210330712223954065[12] = 0;
   out_6210330712223954065[13] = 0;
   out_6210330712223954065[14] = 1;
   out_6210330712223954065[15] = 0;
   out_6210330712223954065[16] = 0;
   out_6210330712223954065[17] = 0;
}
void h_30(double *state, double *unused, double *out_5370462393698290492) {
   out_5370462393698290492[0] = state[4];
}
void H_30(double *state, double *unused, double *out_3295352449362845943) {
   out_3295352449362845943[0] = 0;
   out_3295352449362845943[1] = 0;
   out_3295352449362845943[2] = 0;
   out_3295352449362845943[3] = 0;
   out_3295352449362845943[4] = 1;
   out_3295352449362845943[5] = 0;
   out_3295352449362845943[6] = 0;
   out_3295352449362845943[7] = 0;
   out_3295352449362845943[8] = 0;
}
void h_26(double *state, double *unused, double *out_1313132500629109286) {
   out_1313132500629109286[0] = state[7];
}
void H_26(double *state, double *unused, double *out_2509159438109293969) {
   out_2509159438109293969[0] = 0;
   out_2509159438109293969[1] = 0;
   out_2509159438109293969[2] = 0;
   out_2509159438109293969[3] = 0;
   out_2509159438109293969[4] = 0;
   out_2509159438109293969[5] = 0;
   out_2509159438109293969[6] = 0;
   out_2509159438109293969[7] = 1;
   out_2509159438109293969[8] = 0;
}
void h_27(double *state, double *unused, double *out_1033175332113012121) {
   out_1033175332113012121[0] = state[3];
}
void H_27(double *state, double *unused, double *out_5470115761163270854) {
   out_5470115761163270854[0] = 0;
   out_5470115761163270854[1] = 0;
   out_5470115761163270854[2] = 0;
   out_5470115761163270854[3] = 1;
   out_5470115761163270854[4] = 0;
   out_5470115761163270854[5] = 0;
   out_5470115761163270854[6] = 0;
   out_5470115761163270854[7] = 0;
   out_5470115761163270854[8] = 0;
}
void h_29(double *state, double *unused, double *out_875988663761882976) {
   out_875988663761882976[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2785121105048453759) {
   out_2785121105048453759[0] = 0;
   out_2785121105048453759[1] = 1;
   out_2785121105048453759[2] = 0;
   out_2785121105048453759[3] = 0;
   out_2785121105048453759[4] = 0;
   out_2785121105048453759[5] = 0;
   out_2785121105048453759[6] = 0;
   out_2785121105048453759[7] = 0;
   out_2785121105048453759[8] = 0;
}
void h_28(double *state, double *unused, double *out_5458536247411243654) {
   out_5458536247411243654[0] = state[0];
}
void H_28(double *state, double *unused, double *out_821490833483127508) {
   out_821490833483127508[0] = 1;
   out_821490833483127508[1] = 0;
   out_821490833483127508[2] = 0;
   out_821490833483127508[3] = 0;
   out_821490833483127508[4] = 0;
   out_821490833483127508[5] = 0;
   out_821490833483127508[6] = 0;
   out_821490833483127508[7] = 0;
   out_821490833483127508[8] = 0;
}
void h_31(double *state, double *unused, double *out_3492060828701054801) {
   out_3492060828701054801[0] = state[8];
}
void H_31(double *state, double *unused, double *out_3135367540342645445) {
   out_3135367540342645445[0] = 0;
   out_3135367540342645445[1] = 0;
   out_3135367540342645445[2] = 0;
   out_3135367540342645445[3] = 0;
   out_3135367540342645445[4] = 0;
   out_3135367540342645445[5] = 0;
   out_3135367540342645445[6] = 0;
   out_3135367540342645445[7] = 0;
   out_3135367540342645445[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_7661067751022070458) {
  err_fun(nom_x, delta_x, out_7661067751022070458);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5136938064863750921) {
  inv_err_fun(nom_x, true_x, out_5136938064863750921);
}
void car_H_mod_fun(double *state, double *out_3671347944419902585) {
  H_mod_fun(state, out_3671347944419902585);
}
void car_f_fun(double *state, double dt, double *out_7888740610946713678) {
  f_fun(state,  dt, out_7888740610946713678);
}
void car_F_fun(double *state, double dt, double *out_3995884459612552931) {
  F_fun(state,  dt, out_3995884459612552931);
}
void car_h_25(double *state, double *unused, double *out_3968497935547833524) {
  h_25(state, unused, out_3968497935547833524);
}
void car_H_25(double *state, double *unused, double *out_1232343880764762255) {
  H_25(state, unused, out_1232343880764762255);
}
void car_h_24(double *state, double *unused, double *out_3248956284435332617) {
  h_24(state, unused, out_3248956284435332617);
}
void car_H_24(double *state, double *unused, double *out_6210330712223954065) {
  H_24(state, unused, out_6210330712223954065);
}
void car_h_30(double *state, double *unused, double *out_5370462393698290492) {
  h_30(state, unused, out_5370462393698290492);
}
void car_H_30(double *state, double *unused, double *out_3295352449362845943) {
  H_30(state, unused, out_3295352449362845943);
}
void car_h_26(double *state, double *unused, double *out_1313132500629109286) {
  h_26(state, unused, out_1313132500629109286);
}
void car_H_26(double *state, double *unused, double *out_2509159438109293969) {
  H_26(state, unused, out_2509159438109293969);
}
void car_h_27(double *state, double *unused, double *out_1033175332113012121) {
  h_27(state, unused, out_1033175332113012121);
}
void car_H_27(double *state, double *unused, double *out_5470115761163270854) {
  H_27(state, unused, out_5470115761163270854);
}
void car_h_29(double *state, double *unused, double *out_875988663761882976) {
  h_29(state, unused, out_875988663761882976);
}
void car_H_29(double *state, double *unused, double *out_2785121105048453759) {
  H_29(state, unused, out_2785121105048453759);
}
void car_h_28(double *state, double *unused, double *out_5458536247411243654) {
  h_28(state, unused, out_5458536247411243654);
}
void car_H_28(double *state, double *unused, double *out_821490833483127508) {
  H_28(state, unused, out_821490833483127508);
}
void car_h_31(double *state, double *unused, double *out_3492060828701054801) {
  h_31(state, unused, out_3492060828701054801);
}
void car_H_31(double *state, double *unused, double *out_3135367540342645445) {
  H_31(state, unused, out_3135367540342645445);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
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
