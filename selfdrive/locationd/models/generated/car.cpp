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
void err_fun(double *nom_x, double *delta_x, double *out_2846118138154075095) {
   out_2846118138154075095[0] = delta_x[0] + nom_x[0];
   out_2846118138154075095[1] = delta_x[1] + nom_x[1];
   out_2846118138154075095[2] = delta_x[2] + nom_x[2];
   out_2846118138154075095[3] = delta_x[3] + nom_x[3];
   out_2846118138154075095[4] = delta_x[4] + nom_x[4];
   out_2846118138154075095[5] = delta_x[5] + nom_x[5];
   out_2846118138154075095[6] = delta_x[6] + nom_x[6];
   out_2846118138154075095[7] = delta_x[7] + nom_x[7];
   out_2846118138154075095[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_4044333420406863401) {
   out_4044333420406863401[0] = -nom_x[0] + true_x[0];
   out_4044333420406863401[1] = -nom_x[1] + true_x[1];
   out_4044333420406863401[2] = -nom_x[2] + true_x[2];
   out_4044333420406863401[3] = -nom_x[3] + true_x[3];
   out_4044333420406863401[4] = -nom_x[4] + true_x[4];
   out_4044333420406863401[5] = -nom_x[5] + true_x[5];
   out_4044333420406863401[6] = -nom_x[6] + true_x[6];
   out_4044333420406863401[7] = -nom_x[7] + true_x[7];
   out_4044333420406863401[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_698882326502663349) {
   out_698882326502663349[0] = 1.0;
   out_698882326502663349[1] = 0.0;
   out_698882326502663349[2] = 0.0;
   out_698882326502663349[3] = 0.0;
   out_698882326502663349[4] = 0.0;
   out_698882326502663349[5] = 0.0;
   out_698882326502663349[6] = 0.0;
   out_698882326502663349[7] = 0.0;
   out_698882326502663349[8] = 0.0;
   out_698882326502663349[9] = 0.0;
   out_698882326502663349[10] = 1.0;
   out_698882326502663349[11] = 0.0;
   out_698882326502663349[12] = 0.0;
   out_698882326502663349[13] = 0.0;
   out_698882326502663349[14] = 0.0;
   out_698882326502663349[15] = 0.0;
   out_698882326502663349[16] = 0.0;
   out_698882326502663349[17] = 0.0;
   out_698882326502663349[18] = 0.0;
   out_698882326502663349[19] = 0.0;
   out_698882326502663349[20] = 1.0;
   out_698882326502663349[21] = 0.0;
   out_698882326502663349[22] = 0.0;
   out_698882326502663349[23] = 0.0;
   out_698882326502663349[24] = 0.0;
   out_698882326502663349[25] = 0.0;
   out_698882326502663349[26] = 0.0;
   out_698882326502663349[27] = 0.0;
   out_698882326502663349[28] = 0.0;
   out_698882326502663349[29] = 0.0;
   out_698882326502663349[30] = 1.0;
   out_698882326502663349[31] = 0.0;
   out_698882326502663349[32] = 0.0;
   out_698882326502663349[33] = 0.0;
   out_698882326502663349[34] = 0.0;
   out_698882326502663349[35] = 0.0;
   out_698882326502663349[36] = 0.0;
   out_698882326502663349[37] = 0.0;
   out_698882326502663349[38] = 0.0;
   out_698882326502663349[39] = 0.0;
   out_698882326502663349[40] = 1.0;
   out_698882326502663349[41] = 0.0;
   out_698882326502663349[42] = 0.0;
   out_698882326502663349[43] = 0.0;
   out_698882326502663349[44] = 0.0;
   out_698882326502663349[45] = 0.0;
   out_698882326502663349[46] = 0.0;
   out_698882326502663349[47] = 0.0;
   out_698882326502663349[48] = 0.0;
   out_698882326502663349[49] = 0.0;
   out_698882326502663349[50] = 1.0;
   out_698882326502663349[51] = 0.0;
   out_698882326502663349[52] = 0.0;
   out_698882326502663349[53] = 0.0;
   out_698882326502663349[54] = 0.0;
   out_698882326502663349[55] = 0.0;
   out_698882326502663349[56] = 0.0;
   out_698882326502663349[57] = 0.0;
   out_698882326502663349[58] = 0.0;
   out_698882326502663349[59] = 0.0;
   out_698882326502663349[60] = 1.0;
   out_698882326502663349[61] = 0.0;
   out_698882326502663349[62] = 0.0;
   out_698882326502663349[63] = 0.0;
   out_698882326502663349[64] = 0.0;
   out_698882326502663349[65] = 0.0;
   out_698882326502663349[66] = 0.0;
   out_698882326502663349[67] = 0.0;
   out_698882326502663349[68] = 0.0;
   out_698882326502663349[69] = 0.0;
   out_698882326502663349[70] = 1.0;
   out_698882326502663349[71] = 0.0;
   out_698882326502663349[72] = 0.0;
   out_698882326502663349[73] = 0.0;
   out_698882326502663349[74] = 0.0;
   out_698882326502663349[75] = 0.0;
   out_698882326502663349[76] = 0.0;
   out_698882326502663349[77] = 0.0;
   out_698882326502663349[78] = 0.0;
   out_698882326502663349[79] = 0.0;
   out_698882326502663349[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7869785476666687441) {
   out_7869785476666687441[0] = state[0];
   out_7869785476666687441[1] = state[1];
   out_7869785476666687441[2] = state[2];
   out_7869785476666687441[3] = state[3];
   out_7869785476666687441[4] = state[4];
   out_7869785476666687441[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_7869785476666687441[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7869785476666687441[7] = state[7];
   out_7869785476666687441[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8888598582008610072) {
   out_8888598582008610072[0] = 1;
   out_8888598582008610072[1] = 0;
   out_8888598582008610072[2] = 0;
   out_8888598582008610072[3] = 0;
   out_8888598582008610072[4] = 0;
   out_8888598582008610072[5] = 0;
   out_8888598582008610072[6] = 0;
   out_8888598582008610072[7] = 0;
   out_8888598582008610072[8] = 0;
   out_8888598582008610072[9] = 0;
   out_8888598582008610072[10] = 1;
   out_8888598582008610072[11] = 0;
   out_8888598582008610072[12] = 0;
   out_8888598582008610072[13] = 0;
   out_8888598582008610072[14] = 0;
   out_8888598582008610072[15] = 0;
   out_8888598582008610072[16] = 0;
   out_8888598582008610072[17] = 0;
   out_8888598582008610072[18] = 0;
   out_8888598582008610072[19] = 0;
   out_8888598582008610072[20] = 1;
   out_8888598582008610072[21] = 0;
   out_8888598582008610072[22] = 0;
   out_8888598582008610072[23] = 0;
   out_8888598582008610072[24] = 0;
   out_8888598582008610072[25] = 0;
   out_8888598582008610072[26] = 0;
   out_8888598582008610072[27] = 0;
   out_8888598582008610072[28] = 0;
   out_8888598582008610072[29] = 0;
   out_8888598582008610072[30] = 1;
   out_8888598582008610072[31] = 0;
   out_8888598582008610072[32] = 0;
   out_8888598582008610072[33] = 0;
   out_8888598582008610072[34] = 0;
   out_8888598582008610072[35] = 0;
   out_8888598582008610072[36] = 0;
   out_8888598582008610072[37] = 0;
   out_8888598582008610072[38] = 0;
   out_8888598582008610072[39] = 0;
   out_8888598582008610072[40] = 1;
   out_8888598582008610072[41] = 0;
   out_8888598582008610072[42] = 0;
   out_8888598582008610072[43] = 0;
   out_8888598582008610072[44] = 0;
   out_8888598582008610072[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8888598582008610072[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8888598582008610072[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8888598582008610072[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8888598582008610072[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8888598582008610072[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8888598582008610072[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8888598582008610072[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8888598582008610072[53] = -9.8100000000000005*dt;
   out_8888598582008610072[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8888598582008610072[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8888598582008610072[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8888598582008610072[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8888598582008610072[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8888598582008610072[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8888598582008610072[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8888598582008610072[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8888598582008610072[62] = 0;
   out_8888598582008610072[63] = 0;
   out_8888598582008610072[64] = 0;
   out_8888598582008610072[65] = 0;
   out_8888598582008610072[66] = 0;
   out_8888598582008610072[67] = 0;
   out_8888598582008610072[68] = 0;
   out_8888598582008610072[69] = 0;
   out_8888598582008610072[70] = 1;
   out_8888598582008610072[71] = 0;
   out_8888598582008610072[72] = 0;
   out_8888598582008610072[73] = 0;
   out_8888598582008610072[74] = 0;
   out_8888598582008610072[75] = 0;
   out_8888598582008610072[76] = 0;
   out_8888598582008610072[77] = 0;
   out_8888598582008610072[78] = 0;
   out_8888598582008610072[79] = 0;
   out_8888598582008610072[80] = 1;
}
void h_25(double *state, double *unused, double *out_4497380850200226036) {
   out_4497380850200226036[0] = state[6];
}
void H_25(double *state, double *unused, double *out_2286274964032937428) {
   out_2286274964032937428[0] = 0;
   out_2286274964032937428[1] = 0;
   out_2286274964032937428[2] = 0;
   out_2286274964032937428[3] = 0;
   out_2286274964032937428[4] = 0;
   out_2286274964032937428[5] = 0;
   out_2286274964032937428[6] = 1;
   out_2286274964032937428[7] = 0;
   out_2286274964032937428[8] = 0;
}
void h_24(double *state, double *unused, double *out_1105660614370515943) {
   out_1105660614370515943[0] = state[4];
   out_1105660614370515943[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2987219318880075166) {
   out_2987219318880075166[0] = 0;
   out_2987219318880075166[1] = 0;
   out_2987219318880075166[2] = 0;
   out_2987219318880075166[3] = 0;
   out_2987219318880075166[4] = 1;
   out_2987219318880075166[5] = 0;
   out_2987219318880075166[6] = 0;
   out_2987219318880075166[7] = 0;
   out_2987219318880075166[8] = 0;
   out_2987219318880075166[9] = 0;
   out_2987219318880075166[10] = 0;
   out_2987219318880075166[11] = 0;
   out_2987219318880075166[12] = 0;
   out_2987219318880075166[13] = 0;
   out_2987219318880075166[14] = 1;
   out_2987219318880075166[15] = 0;
   out_2987219318880075166[16] = 0;
   out_2987219318880075166[17] = 0;
}
void h_30(double *state, double *unused, double *out_4340194181849096891) {
   out_4340194181849096891[0] = state[4];
}
void H_30(double *state, double *unused, double *out_232057994474311199) {
   out_232057994474311199[0] = 0;
   out_232057994474311199[1] = 0;
   out_232057994474311199[2] = 0;
   out_232057994474311199[3] = 0;
   out_232057994474311199[4] = 1;
   out_232057994474311199[5] = 0;
   out_232057994474311199[6] = 0;
   out_232057994474311199[7] = 0;
   out_232057994474311199[8] = 0;
}
void h_26(double *state, double *unused, double *out_7376980233293429576) {
   out_7376980233293429576[0] = state[7];
}
void H_26(double *state, double *unused, double *out_6027778282906993652) {
   out_6027778282906993652[0] = 0;
   out_6027778282906993652[1] = 0;
   out_6027778282906993652[2] = 0;
   out_6027778282906993652[3] = 0;
   out_6027778282906993652[4] = 0;
   out_6027778282906993652[5] = 0;
   out_6027778282906993652[6] = 0;
   out_6027778282906993652[7] = 1;
   out_6027778282906993652[8] = 0;
}
void h_27(double *state, double *unused, double *out_7157120249877593801) {
   out_7157120249877593801[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2455652065658254416) {
   out_2455652065658254416[0] = 0;
   out_2455652065658254416[1] = 0;
   out_2455652065658254416[2] = 0;
   out_2455652065658254416[3] = 1;
   out_2455652065658254416[4] = 0;
   out_2455652065658254416[5] = 0;
   out_2455652065658254416[6] = 0;
   out_2455652065658254416[7] = 0;
   out_2455652065658254416[8] = 0;
}
void h_29(double *state, double *unused, double *out_3396684328947434456) {
   out_3396684328947434456[0] = state[1];
}
void H_29(double *state, double *unused, double *out_742289338788703383) {
   out_742289338788703383[0] = 0;
   out_742289338788703383[1] = 1;
   out_742289338788703383[2] = 0;
   out_742289338788703383[3] = 0;
   out_742289338788703383[4] = 0;
   out_742289338788703383[5] = 0;
   out_742289338788703383[6] = 0;
   out_742289338788703383[7] = 0;
   out_742289338788703383[8] = 0;
}
void h_28(double *state, double *unused, double *out_517084945854230916) {
   out_517084945854230916[0] = state[0];
}
void H_28(double *state, double *unused, double *out_4340109678280827191) {
   out_4340109678280827191[0] = 1;
   out_4340109678280827191[1] = 0;
   out_4340109678280827191[2] = 0;
   out_4340109678280827191[3] = 0;
   out_4340109678280827191[4] = 0;
   out_4340109678280827191[5] = 0;
   out_4340109678280827191[6] = 0;
   out_4340109678280827191[7] = 0;
   out_4340109678280827191[8] = 0;
}
void h_31(double *state, double *unused, double *out_5309485746790783696) {
   out_5309485746790783696[0] = state[8];
}
void H_31(double *state, double *unused, double *out_6653986385140345128) {
   out_6653986385140345128[0] = 0;
   out_6653986385140345128[1] = 0;
   out_6653986385140345128[2] = 0;
   out_6653986385140345128[3] = 0;
   out_6653986385140345128[4] = 0;
   out_6653986385140345128[5] = 0;
   out_6653986385140345128[6] = 0;
   out_6653986385140345128[7] = 0;
   out_6653986385140345128[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_2846118138154075095) {
  err_fun(nom_x, delta_x, out_2846118138154075095);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4044333420406863401) {
  inv_err_fun(nom_x, true_x, out_4044333420406863401);
}
void car_H_mod_fun(double *state, double *out_698882326502663349) {
  H_mod_fun(state, out_698882326502663349);
}
void car_f_fun(double *state, double dt, double *out_7869785476666687441) {
  f_fun(state,  dt, out_7869785476666687441);
}
void car_F_fun(double *state, double dt, double *out_8888598582008610072) {
  F_fun(state,  dt, out_8888598582008610072);
}
void car_h_25(double *state, double *unused, double *out_4497380850200226036) {
  h_25(state, unused, out_4497380850200226036);
}
void car_H_25(double *state, double *unused, double *out_2286274964032937428) {
  H_25(state, unused, out_2286274964032937428);
}
void car_h_24(double *state, double *unused, double *out_1105660614370515943) {
  h_24(state, unused, out_1105660614370515943);
}
void car_H_24(double *state, double *unused, double *out_2987219318880075166) {
  H_24(state, unused, out_2987219318880075166);
}
void car_h_30(double *state, double *unused, double *out_4340194181849096891) {
  h_30(state, unused, out_4340194181849096891);
}
void car_H_30(double *state, double *unused, double *out_232057994474311199) {
  H_30(state, unused, out_232057994474311199);
}
void car_h_26(double *state, double *unused, double *out_7376980233293429576) {
  h_26(state, unused, out_7376980233293429576);
}
void car_H_26(double *state, double *unused, double *out_6027778282906993652) {
  H_26(state, unused, out_6027778282906993652);
}
void car_h_27(double *state, double *unused, double *out_7157120249877593801) {
  h_27(state, unused, out_7157120249877593801);
}
void car_H_27(double *state, double *unused, double *out_2455652065658254416) {
  H_27(state, unused, out_2455652065658254416);
}
void car_h_29(double *state, double *unused, double *out_3396684328947434456) {
  h_29(state, unused, out_3396684328947434456);
}
void car_H_29(double *state, double *unused, double *out_742289338788703383) {
  H_29(state, unused, out_742289338788703383);
}
void car_h_28(double *state, double *unused, double *out_517084945854230916) {
  h_28(state, unused, out_517084945854230916);
}
void car_H_28(double *state, double *unused, double *out_4340109678280827191) {
  H_28(state, unused, out_4340109678280827191);
}
void car_h_31(double *state, double *unused, double *out_5309485746790783696) {
  h_31(state, unused, out_5309485746790783696);
}
void car_H_31(double *state, double *unused, double *out_6653986385140345128) {
  H_31(state, unused, out_6653986385140345128);
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
