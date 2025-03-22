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
 *                      Code generated with SymPy 1.13.2                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7345504276725310336) {
   out_7345504276725310336[0] = delta_x[0] + nom_x[0];
   out_7345504276725310336[1] = delta_x[1] + nom_x[1];
   out_7345504276725310336[2] = delta_x[2] + nom_x[2];
   out_7345504276725310336[3] = delta_x[3] + nom_x[3];
   out_7345504276725310336[4] = delta_x[4] + nom_x[4];
   out_7345504276725310336[5] = delta_x[5] + nom_x[5];
   out_7345504276725310336[6] = delta_x[6] + nom_x[6];
   out_7345504276725310336[7] = delta_x[7] + nom_x[7];
   out_7345504276725310336[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1483419534327179826) {
   out_1483419534327179826[0] = -nom_x[0] + true_x[0];
   out_1483419534327179826[1] = -nom_x[1] + true_x[1];
   out_1483419534327179826[2] = -nom_x[2] + true_x[2];
   out_1483419534327179826[3] = -nom_x[3] + true_x[3];
   out_1483419534327179826[4] = -nom_x[4] + true_x[4];
   out_1483419534327179826[5] = -nom_x[5] + true_x[5];
   out_1483419534327179826[6] = -nom_x[6] + true_x[6];
   out_1483419534327179826[7] = -nom_x[7] + true_x[7];
   out_1483419534327179826[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_5980828091480380917) {
   out_5980828091480380917[0] = 1.0;
   out_5980828091480380917[1] = 0.0;
   out_5980828091480380917[2] = 0.0;
   out_5980828091480380917[3] = 0.0;
   out_5980828091480380917[4] = 0.0;
   out_5980828091480380917[5] = 0.0;
   out_5980828091480380917[6] = 0.0;
   out_5980828091480380917[7] = 0.0;
   out_5980828091480380917[8] = 0.0;
   out_5980828091480380917[9] = 0.0;
   out_5980828091480380917[10] = 1.0;
   out_5980828091480380917[11] = 0.0;
   out_5980828091480380917[12] = 0.0;
   out_5980828091480380917[13] = 0.0;
   out_5980828091480380917[14] = 0.0;
   out_5980828091480380917[15] = 0.0;
   out_5980828091480380917[16] = 0.0;
   out_5980828091480380917[17] = 0.0;
   out_5980828091480380917[18] = 0.0;
   out_5980828091480380917[19] = 0.0;
   out_5980828091480380917[20] = 1.0;
   out_5980828091480380917[21] = 0.0;
   out_5980828091480380917[22] = 0.0;
   out_5980828091480380917[23] = 0.0;
   out_5980828091480380917[24] = 0.0;
   out_5980828091480380917[25] = 0.0;
   out_5980828091480380917[26] = 0.0;
   out_5980828091480380917[27] = 0.0;
   out_5980828091480380917[28] = 0.0;
   out_5980828091480380917[29] = 0.0;
   out_5980828091480380917[30] = 1.0;
   out_5980828091480380917[31] = 0.0;
   out_5980828091480380917[32] = 0.0;
   out_5980828091480380917[33] = 0.0;
   out_5980828091480380917[34] = 0.0;
   out_5980828091480380917[35] = 0.0;
   out_5980828091480380917[36] = 0.0;
   out_5980828091480380917[37] = 0.0;
   out_5980828091480380917[38] = 0.0;
   out_5980828091480380917[39] = 0.0;
   out_5980828091480380917[40] = 1.0;
   out_5980828091480380917[41] = 0.0;
   out_5980828091480380917[42] = 0.0;
   out_5980828091480380917[43] = 0.0;
   out_5980828091480380917[44] = 0.0;
   out_5980828091480380917[45] = 0.0;
   out_5980828091480380917[46] = 0.0;
   out_5980828091480380917[47] = 0.0;
   out_5980828091480380917[48] = 0.0;
   out_5980828091480380917[49] = 0.0;
   out_5980828091480380917[50] = 1.0;
   out_5980828091480380917[51] = 0.0;
   out_5980828091480380917[52] = 0.0;
   out_5980828091480380917[53] = 0.0;
   out_5980828091480380917[54] = 0.0;
   out_5980828091480380917[55] = 0.0;
   out_5980828091480380917[56] = 0.0;
   out_5980828091480380917[57] = 0.0;
   out_5980828091480380917[58] = 0.0;
   out_5980828091480380917[59] = 0.0;
   out_5980828091480380917[60] = 1.0;
   out_5980828091480380917[61] = 0.0;
   out_5980828091480380917[62] = 0.0;
   out_5980828091480380917[63] = 0.0;
   out_5980828091480380917[64] = 0.0;
   out_5980828091480380917[65] = 0.0;
   out_5980828091480380917[66] = 0.0;
   out_5980828091480380917[67] = 0.0;
   out_5980828091480380917[68] = 0.0;
   out_5980828091480380917[69] = 0.0;
   out_5980828091480380917[70] = 1.0;
   out_5980828091480380917[71] = 0.0;
   out_5980828091480380917[72] = 0.0;
   out_5980828091480380917[73] = 0.0;
   out_5980828091480380917[74] = 0.0;
   out_5980828091480380917[75] = 0.0;
   out_5980828091480380917[76] = 0.0;
   out_5980828091480380917[77] = 0.0;
   out_5980828091480380917[78] = 0.0;
   out_5980828091480380917[79] = 0.0;
   out_5980828091480380917[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_6771744444959754452) {
   out_6771744444959754452[0] = state[0];
   out_6771744444959754452[1] = state[1];
   out_6771744444959754452[2] = state[2];
   out_6771744444959754452[3] = state[3];
   out_6771744444959754452[4] = state[4];
   out_6771744444959754452[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_6771744444959754452[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_6771744444959754452[7] = state[7];
   out_6771744444959754452[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8079006821062657786) {
   out_8079006821062657786[0] = 1;
   out_8079006821062657786[1] = 0;
   out_8079006821062657786[2] = 0;
   out_8079006821062657786[3] = 0;
   out_8079006821062657786[4] = 0;
   out_8079006821062657786[5] = 0;
   out_8079006821062657786[6] = 0;
   out_8079006821062657786[7] = 0;
   out_8079006821062657786[8] = 0;
   out_8079006821062657786[9] = 0;
   out_8079006821062657786[10] = 1;
   out_8079006821062657786[11] = 0;
   out_8079006821062657786[12] = 0;
   out_8079006821062657786[13] = 0;
   out_8079006821062657786[14] = 0;
   out_8079006821062657786[15] = 0;
   out_8079006821062657786[16] = 0;
   out_8079006821062657786[17] = 0;
   out_8079006821062657786[18] = 0;
   out_8079006821062657786[19] = 0;
   out_8079006821062657786[20] = 1;
   out_8079006821062657786[21] = 0;
   out_8079006821062657786[22] = 0;
   out_8079006821062657786[23] = 0;
   out_8079006821062657786[24] = 0;
   out_8079006821062657786[25] = 0;
   out_8079006821062657786[26] = 0;
   out_8079006821062657786[27] = 0;
   out_8079006821062657786[28] = 0;
   out_8079006821062657786[29] = 0;
   out_8079006821062657786[30] = 1;
   out_8079006821062657786[31] = 0;
   out_8079006821062657786[32] = 0;
   out_8079006821062657786[33] = 0;
   out_8079006821062657786[34] = 0;
   out_8079006821062657786[35] = 0;
   out_8079006821062657786[36] = 0;
   out_8079006821062657786[37] = 0;
   out_8079006821062657786[38] = 0;
   out_8079006821062657786[39] = 0;
   out_8079006821062657786[40] = 1;
   out_8079006821062657786[41] = 0;
   out_8079006821062657786[42] = 0;
   out_8079006821062657786[43] = 0;
   out_8079006821062657786[44] = 0;
   out_8079006821062657786[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8079006821062657786[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8079006821062657786[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8079006821062657786[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8079006821062657786[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8079006821062657786[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8079006821062657786[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8079006821062657786[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8079006821062657786[53] = -9.8000000000000007*dt;
   out_8079006821062657786[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8079006821062657786[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8079006821062657786[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8079006821062657786[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8079006821062657786[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8079006821062657786[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8079006821062657786[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8079006821062657786[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8079006821062657786[62] = 0;
   out_8079006821062657786[63] = 0;
   out_8079006821062657786[64] = 0;
   out_8079006821062657786[65] = 0;
   out_8079006821062657786[66] = 0;
   out_8079006821062657786[67] = 0;
   out_8079006821062657786[68] = 0;
   out_8079006821062657786[69] = 0;
   out_8079006821062657786[70] = 1;
   out_8079006821062657786[71] = 0;
   out_8079006821062657786[72] = 0;
   out_8079006821062657786[73] = 0;
   out_8079006821062657786[74] = 0;
   out_8079006821062657786[75] = 0;
   out_8079006821062657786[76] = 0;
   out_8079006821062657786[77] = 0;
   out_8079006821062657786[78] = 0;
   out_8079006821062657786[79] = 0;
   out_8079006821062657786[80] = 1;
}
void h_25(double *state, double *unused, double *out_6013514099318955817) {
   out_6013514099318955817[0] = state[6];
}
void H_25(double *state, double *unused, double *out_8892173204326502465) {
   out_8892173204326502465[0] = 0;
   out_8892173204326502465[1] = 0;
   out_8892173204326502465[2] = 0;
   out_8892173204326502465[3] = 0;
   out_8892173204326502465[4] = 0;
   out_8892173204326502465[5] = 0;
   out_8892173204326502465[6] = 1;
   out_8892173204326502465[7] = 0;
   out_8892173204326502465[8] = 0;
}
void h_24(double *state, double *unused, double *out_6766398854122891435) {
   out_6766398854122891435[0] = state[4];
   out_6766398854122891435[1] = state[5];
}
void H_24(double *state, double *unused, double *out_7204356768457398274) {
   out_7204356768457398274[0] = 0;
   out_7204356768457398274[1] = 0;
   out_7204356768457398274[2] = 0;
   out_7204356768457398274[3] = 0;
   out_7204356768457398274[4] = 1;
   out_7204356768457398274[5] = 0;
   out_7204356768457398274[6] = 0;
   out_7204356768457398274[7] = 0;
   out_7204356768457398274[8] = 0;
   out_7204356768457398274[9] = 0;
   out_7204356768457398274[10] = 0;
   out_7204356768457398274[11] = 0;
   out_7204356768457398274[12] = 0;
   out_7204356768457398274[13] = 0;
   out_7204356768457398274[14] = 1;
   out_7204356768457398274[15] = 0;
   out_7204356768457398274[16] = 0;
   out_7204356768457398274[17] = 0;
}
void h_30(double *state, double *unused, double *out_5662394748040244863) {
   out_5662394748040244863[0] = state[4];
}
void H_30(double *state, double *unused, double *out_5026874539255440953) {
   out_5026874539255440953[0] = 0;
   out_5026874539255440953[1] = 0;
   out_5026874539255440953[2] = 0;
   out_5026874539255440953[3] = 0;
   out_5026874539255440953[4] = 1;
   out_5026874539255440953[5] = 0;
   out_5026874539255440953[6] = 0;
   out_5026874539255440953[7] = 0;
   out_5026874539255440953[8] = 0;
}
void h_26(double *state, double *unused, double *out_253236005693617473) {
   out_253236005693617473[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5813067550508992927) {
   out_5813067550508992927[0] = 0;
   out_5813067550508992927[1] = 0;
   out_5813067550508992927[2] = 0;
   out_5813067550508992927[3] = 0;
   out_5813067550508992927[4] = 0;
   out_5813067550508992927[5] = 0;
   out_5813067550508992927[6] = 0;
   out_5813067550508992927[7] = 1;
   out_5813067550508992927[8] = 0;
}
void h_27(double *state, double *unused, double *out_6177405252431548545) {
   out_6177405252431548545[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2852111227455016042) {
   out_2852111227455016042[0] = 0;
   out_2852111227455016042[1] = 0;
   out_2852111227455016042[2] = 0;
   out_2852111227455016042[3] = 1;
   out_2852111227455016042[4] = 0;
   out_2852111227455016042[5] = 0;
   out_2852111227455016042[6] = 0;
   out_2852111227455016042[7] = 0;
   out_2852111227455016042[8] = 0;
}
void h_29(double *state, double *unused, double *out_8631889293023980721) {
   out_8631889293023980721[0] = state[1];
}
void H_29(double *state, double *unused, double *out_5537105883569833137) {
   out_5537105883569833137[0] = 0;
   out_5537105883569833137[1] = 1;
   out_5537105883569833137[2] = 0;
   out_5537105883569833137[3] = 0;
   out_5537105883569833137[4] = 0;
   out_5537105883569833137[5] = 0;
   out_5537105883569833137[6] = 0;
   out_5537105883569833137[7] = 0;
   out_5537105883569833137[8] = 0;
}
void h_28(double *state, double *unused, double *out_7267665553617157426) {
   out_7267665553617157426[0] = state[0];
}
void H_28(double *state, double *unused, double *out_454706866500302563) {
   out_454706866500302563[0] = 1;
   out_454706866500302563[1] = 0;
   out_454706866500302563[2] = 0;
   out_454706866500302563[3] = 0;
   out_454706866500302563[4] = 0;
   out_454706866500302563[5] = 0;
   out_454706866500302563[6] = 0;
   out_454706866500302563[7] = 0;
   out_454706866500302563[8] = 0;
}
void h_31(double *state, double *unused, double *out_2626363377399586067) {
   out_2626363377399586067[0] = state[8];
}
void H_31(double *state, double *unused, double *out_5186859448275641451) {
   out_5186859448275641451[0] = 0;
   out_5186859448275641451[1] = 0;
   out_5186859448275641451[2] = 0;
   out_5186859448275641451[3] = 0;
   out_5186859448275641451[4] = 0;
   out_5186859448275641451[5] = 0;
   out_5186859448275641451[6] = 0;
   out_5186859448275641451[7] = 0;
   out_5186859448275641451[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_7345504276725310336) {
  err_fun(nom_x, delta_x, out_7345504276725310336);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_1483419534327179826) {
  inv_err_fun(nom_x, true_x, out_1483419534327179826);
}
void car_H_mod_fun(double *state, double *out_5980828091480380917) {
  H_mod_fun(state, out_5980828091480380917);
}
void car_f_fun(double *state, double dt, double *out_6771744444959754452) {
  f_fun(state,  dt, out_6771744444959754452);
}
void car_F_fun(double *state, double dt, double *out_8079006821062657786) {
  F_fun(state,  dt, out_8079006821062657786);
}
void car_h_25(double *state, double *unused, double *out_6013514099318955817) {
  h_25(state, unused, out_6013514099318955817);
}
void car_H_25(double *state, double *unused, double *out_8892173204326502465) {
  H_25(state, unused, out_8892173204326502465);
}
void car_h_24(double *state, double *unused, double *out_6766398854122891435) {
  h_24(state, unused, out_6766398854122891435);
}
void car_H_24(double *state, double *unused, double *out_7204356768457398274) {
  H_24(state, unused, out_7204356768457398274);
}
void car_h_30(double *state, double *unused, double *out_5662394748040244863) {
  h_30(state, unused, out_5662394748040244863);
}
void car_H_30(double *state, double *unused, double *out_5026874539255440953) {
  H_30(state, unused, out_5026874539255440953);
}
void car_h_26(double *state, double *unused, double *out_253236005693617473) {
  h_26(state, unused, out_253236005693617473);
}
void car_H_26(double *state, double *unused, double *out_5813067550508992927) {
  H_26(state, unused, out_5813067550508992927);
}
void car_h_27(double *state, double *unused, double *out_6177405252431548545) {
  h_27(state, unused, out_6177405252431548545);
}
void car_H_27(double *state, double *unused, double *out_2852111227455016042) {
  H_27(state, unused, out_2852111227455016042);
}
void car_h_29(double *state, double *unused, double *out_8631889293023980721) {
  h_29(state, unused, out_8631889293023980721);
}
void car_H_29(double *state, double *unused, double *out_5537105883569833137) {
  H_29(state, unused, out_5537105883569833137);
}
void car_h_28(double *state, double *unused, double *out_7267665553617157426) {
  h_28(state, unused, out_7267665553617157426);
}
void car_H_28(double *state, double *unused, double *out_454706866500302563) {
  H_28(state, unused, out_454706866500302563);
}
void car_h_31(double *state, double *unused, double *out_2626363377399586067) {
  h_31(state, unused, out_2626363377399586067);
}
void car_H_31(double *state, double *unused, double *out_5186859448275641451) {
  H_31(state, unused, out_5186859448275641451);
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
