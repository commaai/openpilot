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
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_1778852089894645104) {
   out_1778852089894645104[0] = delta_x[0] + nom_x[0];
   out_1778852089894645104[1] = delta_x[1] + nom_x[1];
   out_1778852089894645104[2] = delta_x[2] + nom_x[2];
   out_1778852089894645104[3] = delta_x[3] + nom_x[3];
   out_1778852089894645104[4] = delta_x[4] + nom_x[4];
   out_1778852089894645104[5] = delta_x[5] + nom_x[5];
   out_1778852089894645104[6] = delta_x[6] + nom_x[6];
   out_1778852089894645104[7] = delta_x[7] + nom_x[7];
   out_1778852089894645104[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2123954078847546568) {
   out_2123954078847546568[0] = -nom_x[0] + true_x[0];
   out_2123954078847546568[1] = -nom_x[1] + true_x[1];
   out_2123954078847546568[2] = -nom_x[2] + true_x[2];
   out_2123954078847546568[3] = -nom_x[3] + true_x[3];
   out_2123954078847546568[4] = -nom_x[4] + true_x[4];
   out_2123954078847546568[5] = -nom_x[5] + true_x[5];
   out_2123954078847546568[6] = -nom_x[6] + true_x[6];
   out_2123954078847546568[7] = -nom_x[7] + true_x[7];
   out_2123954078847546568[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_1868636510444762077) {
   out_1868636510444762077[0] = 1.0;
   out_1868636510444762077[1] = 0;
   out_1868636510444762077[2] = 0;
   out_1868636510444762077[3] = 0;
   out_1868636510444762077[4] = 0;
   out_1868636510444762077[5] = 0;
   out_1868636510444762077[6] = 0;
   out_1868636510444762077[7] = 0;
   out_1868636510444762077[8] = 0;
   out_1868636510444762077[9] = 0;
   out_1868636510444762077[10] = 1.0;
   out_1868636510444762077[11] = 0;
   out_1868636510444762077[12] = 0;
   out_1868636510444762077[13] = 0;
   out_1868636510444762077[14] = 0;
   out_1868636510444762077[15] = 0;
   out_1868636510444762077[16] = 0;
   out_1868636510444762077[17] = 0;
   out_1868636510444762077[18] = 0;
   out_1868636510444762077[19] = 0;
   out_1868636510444762077[20] = 1.0;
   out_1868636510444762077[21] = 0;
   out_1868636510444762077[22] = 0;
   out_1868636510444762077[23] = 0;
   out_1868636510444762077[24] = 0;
   out_1868636510444762077[25] = 0;
   out_1868636510444762077[26] = 0;
   out_1868636510444762077[27] = 0;
   out_1868636510444762077[28] = 0;
   out_1868636510444762077[29] = 0;
   out_1868636510444762077[30] = 1.0;
   out_1868636510444762077[31] = 0;
   out_1868636510444762077[32] = 0;
   out_1868636510444762077[33] = 0;
   out_1868636510444762077[34] = 0;
   out_1868636510444762077[35] = 0;
   out_1868636510444762077[36] = 0;
   out_1868636510444762077[37] = 0;
   out_1868636510444762077[38] = 0;
   out_1868636510444762077[39] = 0;
   out_1868636510444762077[40] = 1.0;
   out_1868636510444762077[41] = 0;
   out_1868636510444762077[42] = 0;
   out_1868636510444762077[43] = 0;
   out_1868636510444762077[44] = 0;
   out_1868636510444762077[45] = 0;
   out_1868636510444762077[46] = 0;
   out_1868636510444762077[47] = 0;
   out_1868636510444762077[48] = 0;
   out_1868636510444762077[49] = 0;
   out_1868636510444762077[50] = 1.0;
   out_1868636510444762077[51] = 0;
   out_1868636510444762077[52] = 0;
   out_1868636510444762077[53] = 0;
   out_1868636510444762077[54] = 0;
   out_1868636510444762077[55] = 0;
   out_1868636510444762077[56] = 0;
   out_1868636510444762077[57] = 0;
   out_1868636510444762077[58] = 0;
   out_1868636510444762077[59] = 0;
   out_1868636510444762077[60] = 1.0;
   out_1868636510444762077[61] = 0;
   out_1868636510444762077[62] = 0;
   out_1868636510444762077[63] = 0;
   out_1868636510444762077[64] = 0;
   out_1868636510444762077[65] = 0;
   out_1868636510444762077[66] = 0;
   out_1868636510444762077[67] = 0;
   out_1868636510444762077[68] = 0;
   out_1868636510444762077[69] = 0;
   out_1868636510444762077[70] = 1.0;
   out_1868636510444762077[71] = 0;
   out_1868636510444762077[72] = 0;
   out_1868636510444762077[73] = 0;
   out_1868636510444762077[74] = 0;
   out_1868636510444762077[75] = 0;
   out_1868636510444762077[76] = 0;
   out_1868636510444762077[77] = 0;
   out_1868636510444762077[78] = 0;
   out_1868636510444762077[79] = 0;
   out_1868636510444762077[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_5217702743341681626) {
   out_5217702743341681626[0] = state[0];
   out_5217702743341681626[1] = state[1];
   out_5217702743341681626[2] = state[2];
   out_5217702743341681626[3] = state[3];
   out_5217702743341681626[4] = state[4];
   out_5217702743341681626[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_5217702743341681626[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_5217702743341681626[7] = state[7];
   out_5217702743341681626[8] = state[8];
}
void F_fun(double *state, double dt, double *out_2769937760945802053) {
   out_2769937760945802053[0] = 1;
   out_2769937760945802053[1] = 0;
   out_2769937760945802053[2] = 0;
   out_2769937760945802053[3] = 0;
   out_2769937760945802053[4] = 0;
   out_2769937760945802053[5] = 0;
   out_2769937760945802053[6] = 0;
   out_2769937760945802053[7] = 0;
   out_2769937760945802053[8] = 0;
   out_2769937760945802053[9] = 0;
   out_2769937760945802053[10] = 1;
   out_2769937760945802053[11] = 0;
   out_2769937760945802053[12] = 0;
   out_2769937760945802053[13] = 0;
   out_2769937760945802053[14] = 0;
   out_2769937760945802053[15] = 0;
   out_2769937760945802053[16] = 0;
   out_2769937760945802053[17] = 0;
   out_2769937760945802053[18] = 0;
   out_2769937760945802053[19] = 0;
   out_2769937760945802053[20] = 1;
   out_2769937760945802053[21] = 0;
   out_2769937760945802053[22] = 0;
   out_2769937760945802053[23] = 0;
   out_2769937760945802053[24] = 0;
   out_2769937760945802053[25] = 0;
   out_2769937760945802053[26] = 0;
   out_2769937760945802053[27] = 0;
   out_2769937760945802053[28] = 0;
   out_2769937760945802053[29] = 0;
   out_2769937760945802053[30] = 1;
   out_2769937760945802053[31] = 0;
   out_2769937760945802053[32] = 0;
   out_2769937760945802053[33] = 0;
   out_2769937760945802053[34] = 0;
   out_2769937760945802053[35] = 0;
   out_2769937760945802053[36] = 0;
   out_2769937760945802053[37] = 0;
   out_2769937760945802053[38] = 0;
   out_2769937760945802053[39] = 0;
   out_2769937760945802053[40] = 1;
   out_2769937760945802053[41] = 0;
   out_2769937760945802053[42] = 0;
   out_2769937760945802053[43] = 0;
   out_2769937760945802053[44] = 0;
   out_2769937760945802053[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_2769937760945802053[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_2769937760945802053[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_2769937760945802053[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_2769937760945802053[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_2769937760945802053[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_2769937760945802053[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_2769937760945802053[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_2769937760945802053[53] = -9.8000000000000007*dt;
   out_2769937760945802053[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_2769937760945802053[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_2769937760945802053[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2769937760945802053[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2769937760945802053[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_2769937760945802053[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_2769937760945802053[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_2769937760945802053[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2769937760945802053[62] = 0;
   out_2769937760945802053[63] = 0;
   out_2769937760945802053[64] = 0;
   out_2769937760945802053[65] = 0;
   out_2769937760945802053[66] = 0;
   out_2769937760945802053[67] = 0;
   out_2769937760945802053[68] = 0;
   out_2769937760945802053[69] = 0;
   out_2769937760945802053[70] = 1;
   out_2769937760945802053[71] = 0;
   out_2769937760945802053[72] = 0;
   out_2769937760945802053[73] = 0;
   out_2769937760945802053[74] = 0;
   out_2769937760945802053[75] = 0;
   out_2769937760945802053[76] = 0;
   out_2769937760945802053[77] = 0;
   out_2769937760945802053[78] = 0;
   out_2769937760945802053[79] = 0;
   out_2769937760945802053[80] = 1;
}
void h_25(double *state, double *unused, double *out_8414728814641436601) {
   out_8414728814641436601[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7023695843139760480) {
   out_7023695843139760480[0] = 0;
   out_7023695843139760480[1] = 0;
   out_7023695843139760480[2] = 0;
   out_7023695843139760480[3] = 0;
   out_7023695843139760480[4] = 0;
   out_7023695843139760480[5] = 0;
   out_7023695843139760480[6] = 1;
   out_7023695843139760480[7] = 0;
   out_7023695843139760480[8] = 0;
}
void h_24(double *state, double *unused, double *out_4219916024026276113) {
   out_4219916024026276113[0] = state[4];
   out_4219916024026276113[1] = state[5];
}
void H_24(double *state, double *unused, double *out_822663163288599054) {
   out_822663163288599054[0] = 0;
   out_822663163288599054[1] = 0;
   out_822663163288599054[2] = 0;
   out_822663163288599054[3] = 0;
   out_822663163288599054[4] = 1;
   out_822663163288599054[5] = 0;
   out_822663163288599054[6] = 0;
   out_822663163288599054[7] = 0;
   out_822663163288599054[8] = 0;
   out_822663163288599054[9] = 0;
   out_822663163288599054[10] = 0;
   out_822663163288599054[11] = 0;
   out_822663163288599054[12] = 0;
   out_822663163288599054[13] = 0;
   out_822663163288599054[14] = 1;
   out_822663163288599054[15] = 0;
   out_822663163288599054[16] = 0;
   out_822663163288599054[17] = 0;
}
void h_30(double *state, double *unused, double *out_2828799302082129045) {
   out_2828799302082129045[0] = state[4];
}
void H_30(double *state, double *unused, double *out_2495999513012152282) {
   out_2495999513012152282[0] = 0;
   out_2495999513012152282[1] = 0;
   out_2495999513012152282[2] = 0;
   out_2495999513012152282[3] = 0;
   out_2495999513012152282[4] = 1;
   out_2495999513012152282[5] = 0;
   out_2495999513012152282[6] = 0;
   out_2495999513012152282[7] = 0;
   out_2495999513012152282[8] = 0;
}
void h_26(double *state, double *unused, double *out_8603047268603794642) {
   out_8603047268603794642[0] = state[7];
}
void H_26(double *state, double *unused, double *out_3282192524265704256) {
   out_3282192524265704256[0] = 0;
   out_3282192524265704256[1] = 0;
   out_3282192524265704256[2] = 0;
   out_3282192524265704256[3] = 0;
   out_3282192524265704256[4] = 0;
   out_3282192524265704256[5] = 0;
   out_3282192524265704256[6] = 0;
   out_3282192524265704256[7] = 1;
   out_3282192524265704256[8] = 0;
}
void h_27(double *state, double *unused, double *out_3413055546980590956) {
   out_3413055546980590956[0] = state[3];
}
void H_27(double *state, double *unused, double *out_4719593584196095499) {
   out_4719593584196095499[0] = 0;
   out_4719593584196095499[1] = 0;
   out_4719593584196095499[2] = 0;
   out_4719593584196095499[3] = 1;
   out_4719593584196095499[4] = 0;
   out_4719593584196095499[5] = 0;
   out_4719593584196095499[6] = 0;
   out_4719593584196095499[7] = 0;
   out_4719593584196095499[8] = 0;
}
void h_29(double *state, double *unused, double *out_224393981273431394) {
   out_224393981273431394[0] = state[1];
}
void H_29(double *state, double *unused, double *out_3006230857326544466) {
   out_3006230857326544466[0] = 0;
   out_3006230857326544466[1] = 1;
   out_3006230857326544466[2] = 0;
   out_3006230857326544466[3] = 0;
   out_3006230857326544466[4] = 0;
   out_3006230857326544466[5] = 0;
   out_3006230857326544466[6] = 0;
   out_3006230857326544466[7] = 0;
   out_3006230857326544466[8] = 0;
}
void h_28(double *state, double *unused, double *out_6449841598424923641) {
   out_6449841598424923641[0] = state[0];
}
void H_28(double *state, double *unused, double *out_2076168159742986108) {
   out_2076168159742986108[0] = 1;
   out_2076168159742986108[1] = 0;
   out_2076168159742986108[2] = 0;
   out_2076168159742986108[3] = 0;
   out_2076168159742986108[4] = 0;
   out_2076168159742986108[5] = 0;
   out_2076168159742986108[6] = 0;
   out_2076168159742986108[7] = 0;
   out_2076168159742986108[8] = 0;
}
void h_31(double *state, double *unused, double *out_8689922876925942490) {
   out_8689922876925942490[0] = state[8];
}
void H_31(double *state, double *unused, double *out_2655984422032352780) {
   out_2655984422032352780[0] = 0;
   out_2655984422032352780[1] = 0;
   out_2655984422032352780[2] = 0;
   out_2655984422032352780[3] = 0;
   out_2655984422032352780[4] = 0;
   out_2655984422032352780[5] = 0;
   out_2655984422032352780[6] = 0;
   out_2655984422032352780[7] = 0;
   out_2655984422032352780[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_1778852089894645104) {
  err_fun(nom_x, delta_x, out_1778852089894645104);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2123954078847546568) {
  inv_err_fun(nom_x, true_x, out_2123954078847546568);
}
void car_H_mod_fun(double *state, double *out_1868636510444762077) {
  H_mod_fun(state, out_1868636510444762077);
}
void car_f_fun(double *state, double dt, double *out_5217702743341681626) {
  f_fun(state,  dt, out_5217702743341681626);
}
void car_F_fun(double *state, double dt, double *out_2769937760945802053) {
  F_fun(state,  dt, out_2769937760945802053);
}
void car_h_25(double *state, double *unused, double *out_8414728814641436601) {
  h_25(state, unused, out_8414728814641436601);
}
void car_H_25(double *state, double *unused, double *out_7023695843139760480) {
  H_25(state, unused, out_7023695843139760480);
}
void car_h_24(double *state, double *unused, double *out_4219916024026276113) {
  h_24(state, unused, out_4219916024026276113);
}
void car_H_24(double *state, double *unused, double *out_822663163288599054) {
  H_24(state, unused, out_822663163288599054);
}
void car_h_30(double *state, double *unused, double *out_2828799302082129045) {
  h_30(state, unused, out_2828799302082129045);
}
void car_H_30(double *state, double *unused, double *out_2495999513012152282) {
  H_30(state, unused, out_2495999513012152282);
}
void car_h_26(double *state, double *unused, double *out_8603047268603794642) {
  h_26(state, unused, out_8603047268603794642);
}
void car_H_26(double *state, double *unused, double *out_3282192524265704256) {
  H_26(state, unused, out_3282192524265704256);
}
void car_h_27(double *state, double *unused, double *out_3413055546980590956) {
  h_27(state, unused, out_3413055546980590956);
}
void car_H_27(double *state, double *unused, double *out_4719593584196095499) {
  H_27(state, unused, out_4719593584196095499);
}
void car_h_29(double *state, double *unused, double *out_224393981273431394) {
  h_29(state, unused, out_224393981273431394);
}
void car_H_29(double *state, double *unused, double *out_3006230857326544466) {
  H_29(state, unused, out_3006230857326544466);
}
void car_h_28(double *state, double *unused, double *out_6449841598424923641) {
  h_28(state, unused, out_6449841598424923641);
}
void car_H_28(double *state, double *unused, double *out_2076168159742986108) {
  H_28(state, unused, out_2076168159742986108);
}
void car_h_31(double *state, double *unused, double *out_8689922876925942490) {
  h_31(state, unused, out_8689922876925942490);
}
void car_H_31(double *state, double *unused, double *out_2655984422032352780) {
  H_31(state, unused, out_2655984422032352780);
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
