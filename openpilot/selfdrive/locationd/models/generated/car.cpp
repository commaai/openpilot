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
void err_fun(double *nom_x, double *delta_x, double *out_7626122079372452194) {
   out_7626122079372452194[0] = delta_x[0] + nom_x[0];
   out_7626122079372452194[1] = delta_x[1] + nom_x[1];
   out_7626122079372452194[2] = delta_x[2] + nom_x[2];
   out_7626122079372452194[3] = delta_x[3] + nom_x[3];
   out_7626122079372452194[4] = delta_x[4] + nom_x[4];
   out_7626122079372452194[5] = delta_x[5] + nom_x[5];
   out_7626122079372452194[6] = delta_x[6] + nom_x[6];
   out_7626122079372452194[7] = delta_x[7] + nom_x[7];
   out_7626122079372452194[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6779862585021228643) {
   out_6779862585021228643[0] = -nom_x[0] + true_x[0];
   out_6779862585021228643[1] = -nom_x[1] + true_x[1];
   out_6779862585021228643[2] = -nom_x[2] + true_x[2];
   out_6779862585021228643[3] = -nom_x[3] + true_x[3];
   out_6779862585021228643[4] = -nom_x[4] + true_x[4];
   out_6779862585021228643[5] = -nom_x[5] + true_x[5];
   out_6779862585021228643[6] = -nom_x[6] + true_x[6];
   out_6779862585021228643[7] = -nom_x[7] + true_x[7];
   out_6779862585021228643[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_3551509842771640505) {
   out_3551509842771640505[0] = 1.0;
   out_3551509842771640505[1] = 0.0;
   out_3551509842771640505[2] = 0.0;
   out_3551509842771640505[3] = 0.0;
   out_3551509842771640505[4] = 0.0;
   out_3551509842771640505[5] = 0.0;
   out_3551509842771640505[6] = 0.0;
   out_3551509842771640505[7] = 0.0;
   out_3551509842771640505[8] = 0.0;
   out_3551509842771640505[9] = 0.0;
   out_3551509842771640505[10] = 1.0;
   out_3551509842771640505[11] = 0.0;
   out_3551509842771640505[12] = 0.0;
   out_3551509842771640505[13] = 0.0;
   out_3551509842771640505[14] = 0.0;
   out_3551509842771640505[15] = 0.0;
   out_3551509842771640505[16] = 0.0;
   out_3551509842771640505[17] = 0.0;
   out_3551509842771640505[18] = 0.0;
   out_3551509842771640505[19] = 0.0;
   out_3551509842771640505[20] = 1.0;
   out_3551509842771640505[21] = 0.0;
   out_3551509842771640505[22] = 0.0;
   out_3551509842771640505[23] = 0.0;
   out_3551509842771640505[24] = 0.0;
   out_3551509842771640505[25] = 0.0;
   out_3551509842771640505[26] = 0.0;
   out_3551509842771640505[27] = 0.0;
   out_3551509842771640505[28] = 0.0;
   out_3551509842771640505[29] = 0.0;
   out_3551509842771640505[30] = 1.0;
   out_3551509842771640505[31] = 0.0;
   out_3551509842771640505[32] = 0.0;
   out_3551509842771640505[33] = 0.0;
   out_3551509842771640505[34] = 0.0;
   out_3551509842771640505[35] = 0.0;
   out_3551509842771640505[36] = 0.0;
   out_3551509842771640505[37] = 0.0;
   out_3551509842771640505[38] = 0.0;
   out_3551509842771640505[39] = 0.0;
   out_3551509842771640505[40] = 1.0;
   out_3551509842771640505[41] = 0.0;
   out_3551509842771640505[42] = 0.0;
   out_3551509842771640505[43] = 0.0;
   out_3551509842771640505[44] = 0.0;
   out_3551509842771640505[45] = 0.0;
   out_3551509842771640505[46] = 0.0;
   out_3551509842771640505[47] = 0.0;
   out_3551509842771640505[48] = 0.0;
   out_3551509842771640505[49] = 0.0;
   out_3551509842771640505[50] = 1.0;
   out_3551509842771640505[51] = 0.0;
   out_3551509842771640505[52] = 0.0;
   out_3551509842771640505[53] = 0.0;
   out_3551509842771640505[54] = 0.0;
   out_3551509842771640505[55] = 0.0;
   out_3551509842771640505[56] = 0.0;
   out_3551509842771640505[57] = 0.0;
   out_3551509842771640505[58] = 0.0;
   out_3551509842771640505[59] = 0.0;
   out_3551509842771640505[60] = 1.0;
   out_3551509842771640505[61] = 0.0;
   out_3551509842771640505[62] = 0.0;
   out_3551509842771640505[63] = 0.0;
   out_3551509842771640505[64] = 0.0;
   out_3551509842771640505[65] = 0.0;
   out_3551509842771640505[66] = 0.0;
   out_3551509842771640505[67] = 0.0;
   out_3551509842771640505[68] = 0.0;
   out_3551509842771640505[69] = 0.0;
   out_3551509842771640505[70] = 1.0;
   out_3551509842771640505[71] = 0.0;
   out_3551509842771640505[72] = 0.0;
   out_3551509842771640505[73] = 0.0;
   out_3551509842771640505[74] = 0.0;
   out_3551509842771640505[75] = 0.0;
   out_3551509842771640505[76] = 0.0;
   out_3551509842771640505[77] = 0.0;
   out_3551509842771640505[78] = 0.0;
   out_3551509842771640505[79] = 0.0;
   out_3551509842771640505[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_5520057530326467442) {
   out_5520057530326467442[0] = state[0];
   out_5520057530326467442[1] = state[1];
   out_5520057530326467442[2] = state[2];
   out_5520057530326467442[3] = state[3];
   out_5520057530326467442[4] = state[4];
   out_5520057530326467442[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_5520057530326467442[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_5520057530326467442[7] = state[7];
   out_5520057530326467442[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8593367633030745139) {
   out_8593367633030745139[0] = 1;
   out_8593367633030745139[1] = 0;
   out_8593367633030745139[2] = 0;
   out_8593367633030745139[3] = 0;
   out_8593367633030745139[4] = 0;
   out_8593367633030745139[5] = 0;
   out_8593367633030745139[6] = 0;
   out_8593367633030745139[7] = 0;
   out_8593367633030745139[8] = 0;
   out_8593367633030745139[9] = 0;
   out_8593367633030745139[10] = 1;
   out_8593367633030745139[11] = 0;
   out_8593367633030745139[12] = 0;
   out_8593367633030745139[13] = 0;
   out_8593367633030745139[14] = 0;
   out_8593367633030745139[15] = 0;
   out_8593367633030745139[16] = 0;
   out_8593367633030745139[17] = 0;
   out_8593367633030745139[18] = 0;
   out_8593367633030745139[19] = 0;
   out_8593367633030745139[20] = 1;
   out_8593367633030745139[21] = 0;
   out_8593367633030745139[22] = 0;
   out_8593367633030745139[23] = 0;
   out_8593367633030745139[24] = 0;
   out_8593367633030745139[25] = 0;
   out_8593367633030745139[26] = 0;
   out_8593367633030745139[27] = 0;
   out_8593367633030745139[28] = 0;
   out_8593367633030745139[29] = 0;
   out_8593367633030745139[30] = 1;
   out_8593367633030745139[31] = 0;
   out_8593367633030745139[32] = 0;
   out_8593367633030745139[33] = 0;
   out_8593367633030745139[34] = 0;
   out_8593367633030745139[35] = 0;
   out_8593367633030745139[36] = 0;
   out_8593367633030745139[37] = 0;
   out_8593367633030745139[38] = 0;
   out_8593367633030745139[39] = 0;
   out_8593367633030745139[40] = 1;
   out_8593367633030745139[41] = 0;
   out_8593367633030745139[42] = 0;
   out_8593367633030745139[43] = 0;
   out_8593367633030745139[44] = 0;
   out_8593367633030745139[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8593367633030745139[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8593367633030745139[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8593367633030745139[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8593367633030745139[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8593367633030745139[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8593367633030745139[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8593367633030745139[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8593367633030745139[53] = -9.8100000000000005*dt;
   out_8593367633030745139[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8593367633030745139[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8593367633030745139[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8593367633030745139[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8593367633030745139[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8593367633030745139[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8593367633030745139[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8593367633030745139[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8593367633030745139[62] = 0;
   out_8593367633030745139[63] = 0;
   out_8593367633030745139[64] = 0;
   out_8593367633030745139[65] = 0;
   out_8593367633030745139[66] = 0;
   out_8593367633030745139[67] = 0;
   out_8593367633030745139[68] = 0;
   out_8593367633030745139[69] = 0;
   out_8593367633030745139[70] = 1;
   out_8593367633030745139[71] = 0;
   out_8593367633030745139[72] = 0;
   out_8593367633030745139[73] = 0;
   out_8593367633030745139[74] = 0;
   out_8593367633030745139[75] = 0;
   out_8593367633030745139[76] = 0;
   out_8593367633030745139[77] = 0;
   out_8593367633030745139[78] = 0;
   out_8593367633030745139[79] = 0;
   out_8593367633030745139[80] = 1;
}
void h_25(double *state, double *unused, double *out_2326995945769605353) {
   out_2326995945769605353[0] = state[6];
}
void H_25(double *state, double *unused, double *out_8455201667956305345) {
   out_8455201667956305345[0] = 0;
   out_8455201667956305345[1] = 0;
   out_8455201667956305345[2] = 0;
   out_8455201667956305345[3] = 0;
   out_8455201667956305345[4] = 0;
   out_8455201667956305345[5] = 0;
   out_8455201667956305345[6] = 1;
   out_8455201667956305345[7] = 0;
   out_8455201667956305345[8] = 0;
}
void h_24(double *state, double *unused, double *out_5998448526015495371) {
   out_5998448526015495371[0] = state[4];
   out_5998448526015495371[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8058556363602445850) {
   out_8058556363602445850[0] = 0;
   out_8058556363602445850[1] = 0;
   out_8058556363602445850[2] = 0;
   out_8058556363602445850[3] = 0;
   out_8058556363602445850[4] = 1;
   out_8058556363602445850[5] = 0;
   out_8058556363602445850[6] = 0;
   out_8058556363602445850[7] = 0;
   out_8058556363602445850[8] = 0;
   out_8058556363602445850[9] = 0;
   out_8058556363602445850[10] = 0;
   out_8058556363602445850[11] = 0;
   out_8058556363602445850[12] = 0;
   out_8058556363602445850[13] = 0;
   out_8058556363602445850[14] = 1;
   out_8058556363602445850[15] = 0;
   out_8058556363602445850[16] = 0;
   out_8058556363602445850[17] = 0;
}
void h_30(double *state, double *unused, double *out_2484182614120734498) {
   out_2484182614120734498[0] = state[4];
}
void H_30(double *state, double *unused, double *out_7473209447245997644) {
   out_7473209447245997644[0] = 0;
   out_7473209447245997644[1] = 0;
   out_7473209447245997644[2] = 0;
   out_7473209447245997644[3] = 0;
   out_7473209447245997644[4] = 1;
   out_7473209447245997644[5] = 0;
   out_7473209447245997644[6] = 0;
   out_7473209447245997644[7] = 0;
   out_7473209447245997644[8] = 0;
}
void h_26(double *state, double *unused, double *out_7485855881781580143) {
   out_7485855881781580143[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4713698349082249121) {
   out_4713698349082249121[0] = 0;
   out_4713698349082249121[1] = 0;
   out_4713698349082249121[2] = 0;
   out_4713698349082249121[3] = 0;
   out_4713698349082249121[4] = 0;
   out_4713698349082249121[5] = 0;
   out_4713698349082249121[6] = 0;
   out_4713698349082249121[7] = 1;
   out_4713698349082249121[8] = 0;
}
void h_27(double *state, double *unused, double *out_8908838641685724752) {
   out_8908838641685724752[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1752742026028272236) {
   out_1752742026028272236[0] = 0;
   out_1752742026028272236[1] = 0;
   out_1752742026028272236[2] = 0;
   out_1752742026028272236[3] = 1;
   out_1752742026028272236[4] = 0;
   out_1752742026028272236[5] = 0;
   out_1752742026028272236[6] = 0;
   out_1752742026028272236[7] = 0;
   out_1752742026028272236[8] = 0;
}
void h_29(double *state, double *unused, double *out_9184032703970230641) {
   out_9184032703970230641[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4437736682143089331) {
   out_4437736682143089331[0] = 0;
   out_4437736682143089331[1] = 1;
   out_4437736682143089331[2] = 0;
   out_4437736682143089331[3] = 0;
   out_4437736682143089331[4] = 0;
   out_4437736682143089331[5] = 0;
   out_4437736682143089331[6] = 0;
   out_4437736682143089331[7] = 0;
   out_4437736682143089331[8] = 0;
}
void h_28(double *state, double *unused, double *out_4536232164362981219) {
   out_4536232164362981219[0] = state[0];
}
void H_28(double *state, double *unused, double *out_6401366953708415582) {
   out_6401366953708415582[0] = 1;
   out_6401366953708415582[1] = 0;
   out_6401366953708415582[2] = 0;
   out_6401366953708415582[3] = 0;
   out_6401366953708415582[4] = 0;
   out_6401366953708415582[5] = 0;
   out_6401366953708415582[6] = 0;
   out_6401366953708415582[7] = 0;
   out_6401366953708415582[8] = 0;
}
void h_31(double *state, double *unused, double *out_4443839280580745583) {
   out_4443839280580745583[0] = state[8];
}
void H_31(double *state, double *unused, double *out_4087490246848897645) {
   out_4087490246848897645[0] = 0;
   out_4087490246848897645[1] = 0;
   out_4087490246848897645[2] = 0;
   out_4087490246848897645[3] = 0;
   out_4087490246848897645[4] = 0;
   out_4087490246848897645[5] = 0;
   out_4087490246848897645[6] = 0;
   out_4087490246848897645[7] = 0;
   out_4087490246848897645[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_7626122079372452194) {
  err_fun(nom_x, delta_x, out_7626122079372452194);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6779862585021228643) {
  inv_err_fun(nom_x, true_x, out_6779862585021228643);
}
void car_H_mod_fun(double *state, double *out_3551509842771640505) {
  H_mod_fun(state, out_3551509842771640505);
}
void car_f_fun(double *state, double dt, double *out_5520057530326467442) {
  f_fun(state,  dt, out_5520057530326467442);
}
void car_F_fun(double *state, double dt, double *out_8593367633030745139) {
  F_fun(state,  dt, out_8593367633030745139);
}
void car_h_25(double *state, double *unused, double *out_2326995945769605353) {
  h_25(state, unused, out_2326995945769605353);
}
void car_H_25(double *state, double *unused, double *out_8455201667956305345) {
  H_25(state, unused, out_8455201667956305345);
}
void car_h_24(double *state, double *unused, double *out_5998448526015495371) {
  h_24(state, unused, out_5998448526015495371);
}
void car_H_24(double *state, double *unused, double *out_8058556363602445850) {
  H_24(state, unused, out_8058556363602445850);
}
void car_h_30(double *state, double *unused, double *out_2484182614120734498) {
  h_30(state, unused, out_2484182614120734498);
}
void car_H_30(double *state, double *unused, double *out_7473209447245997644) {
  H_30(state, unused, out_7473209447245997644);
}
void car_h_26(double *state, double *unused, double *out_7485855881781580143) {
  h_26(state, unused, out_7485855881781580143);
}
void car_H_26(double *state, double *unused, double *out_4713698349082249121) {
  H_26(state, unused, out_4713698349082249121);
}
void car_h_27(double *state, double *unused, double *out_8908838641685724752) {
  h_27(state, unused, out_8908838641685724752);
}
void car_H_27(double *state, double *unused, double *out_1752742026028272236) {
  H_27(state, unused, out_1752742026028272236);
}
void car_h_29(double *state, double *unused, double *out_9184032703970230641) {
  h_29(state, unused, out_9184032703970230641);
}
void car_H_29(double *state, double *unused, double *out_4437736682143089331) {
  H_29(state, unused, out_4437736682143089331);
}
void car_h_28(double *state, double *unused, double *out_4536232164362981219) {
  h_28(state, unused, out_4536232164362981219);
}
void car_H_28(double *state, double *unused, double *out_6401366953708415582) {
  H_28(state, unused, out_6401366953708415582);
}
void car_h_31(double *state, double *unused, double *out_4443839280580745583) {
  h_31(state, unused, out_4443839280580745583);
}
void car_H_31(double *state, double *unused, double *out_4087490246848897645) {
  H_31(state, unused, out_4087490246848897645);
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
