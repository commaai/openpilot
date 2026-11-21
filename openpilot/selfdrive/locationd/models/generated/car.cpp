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
void err_fun(double *nom_x, double *delta_x, double *out_2069516698642958211) {
   out_2069516698642958211[0] = delta_x[0] + nom_x[0];
   out_2069516698642958211[1] = delta_x[1] + nom_x[1];
   out_2069516698642958211[2] = delta_x[2] + nom_x[2];
   out_2069516698642958211[3] = delta_x[3] + nom_x[3];
   out_2069516698642958211[4] = delta_x[4] + nom_x[4];
   out_2069516698642958211[5] = delta_x[5] + nom_x[5];
   out_2069516698642958211[6] = delta_x[6] + nom_x[6];
   out_2069516698642958211[7] = delta_x[7] + nom_x[7];
   out_2069516698642958211[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8998358058995224576) {
   out_8998358058995224576[0] = -nom_x[0] + true_x[0];
   out_8998358058995224576[1] = -nom_x[1] + true_x[1];
   out_8998358058995224576[2] = -nom_x[2] + true_x[2];
   out_8998358058995224576[3] = -nom_x[3] + true_x[3];
   out_8998358058995224576[4] = -nom_x[4] + true_x[4];
   out_8998358058995224576[5] = -nom_x[5] + true_x[5];
   out_8998358058995224576[6] = -nom_x[6] + true_x[6];
   out_8998358058995224576[7] = -nom_x[7] + true_x[7];
   out_8998358058995224576[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_7695547058271246574) {
   out_7695547058271246574[0] = 1.0;
   out_7695547058271246574[1] = 0.0;
   out_7695547058271246574[2] = 0.0;
   out_7695547058271246574[3] = 0.0;
   out_7695547058271246574[4] = 0.0;
   out_7695547058271246574[5] = 0.0;
   out_7695547058271246574[6] = 0.0;
   out_7695547058271246574[7] = 0.0;
   out_7695547058271246574[8] = 0.0;
   out_7695547058271246574[9] = 0.0;
   out_7695547058271246574[10] = 1.0;
   out_7695547058271246574[11] = 0.0;
   out_7695547058271246574[12] = 0.0;
   out_7695547058271246574[13] = 0.0;
   out_7695547058271246574[14] = 0.0;
   out_7695547058271246574[15] = 0.0;
   out_7695547058271246574[16] = 0.0;
   out_7695547058271246574[17] = 0.0;
   out_7695547058271246574[18] = 0.0;
   out_7695547058271246574[19] = 0.0;
   out_7695547058271246574[20] = 1.0;
   out_7695547058271246574[21] = 0.0;
   out_7695547058271246574[22] = 0.0;
   out_7695547058271246574[23] = 0.0;
   out_7695547058271246574[24] = 0.0;
   out_7695547058271246574[25] = 0.0;
   out_7695547058271246574[26] = 0.0;
   out_7695547058271246574[27] = 0.0;
   out_7695547058271246574[28] = 0.0;
   out_7695547058271246574[29] = 0.0;
   out_7695547058271246574[30] = 1.0;
   out_7695547058271246574[31] = 0.0;
   out_7695547058271246574[32] = 0.0;
   out_7695547058271246574[33] = 0.0;
   out_7695547058271246574[34] = 0.0;
   out_7695547058271246574[35] = 0.0;
   out_7695547058271246574[36] = 0.0;
   out_7695547058271246574[37] = 0.0;
   out_7695547058271246574[38] = 0.0;
   out_7695547058271246574[39] = 0.0;
   out_7695547058271246574[40] = 1.0;
   out_7695547058271246574[41] = 0.0;
   out_7695547058271246574[42] = 0.0;
   out_7695547058271246574[43] = 0.0;
   out_7695547058271246574[44] = 0.0;
   out_7695547058271246574[45] = 0.0;
   out_7695547058271246574[46] = 0.0;
   out_7695547058271246574[47] = 0.0;
   out_7695547058271246574[48] = 0.0;
   out_7695547058271246574[49] = 0.0;
   out_7695547058271246574[50] = 1.0;
   out_7695547058271246574[51] = 0.0;
   out_7695547058271246574[52] = 0.0;
   out_7695547058271246574[53] = 0.0;
   out_7695547058271246574[54] = 0.0;
   out_7695547058271246574[55] = 0.0;
   out_7695547058271246574[56] = 0.0;
   out_7695547058271246574[57] = 0.0;
   out_7695547058271246574[58] = 0.0;
   out_7695547058271246574[59] = 0.0;
   out_7695547058271246574[60] = 1.0;
   out_7695547058271246574[61] = 0.0;
   out_7695547058271246574[62] = 0.0;
   out_7695547058271246574[63] = 0.0;
   out_7695547058271246574[64] = 0.0;
   out_7695547058271246574[65] = 0.0;
   out_7695547058271246574[66] = 0.0;
   out_7695547058271246574[67] = 0.0;
   out_7695547058271246574[68] = 0.0;
   out_7695547058271246574[69] = 0.0;
   out_7695547058271246574[70] = 1.0;
   out_7695547058271246574[71] = 0.0;
   out_7695547058271246574[72] = 0.0;
   out_7695547058271246574[73] = 0.0;
   out_7695547058271246574[74] = 0.0;
   out_7695547058271246574[75] = 0.0;
   out_7695547058271246574[76] = 0.0;
   out_7695547058271246574[77] = 0.0;
   out_7695547058271246574[78] = 0.0;
   out_7695547058271246574[79] = 0.0;
   out_7695547058271246574[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_2845810596498337567) {
   out_2845810596498337567[0] = state[0];
   out_2845810596498337567[1] = state[1];
   out_2845810596498337567[2] = state[2];
   out_2845810596498337567[3] = state[3];
   out_2845810596498337567[4] = state[4];
   out_2845810596498337567[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_2845810596498337567[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_2845810596498337567[7] = state[7];
   out_2845810596498337567[8] = state[8];
}
void F_fun(double *state, double dt, double *out_6746448386310637026) {
   out_6746448386310637026[0] = 1;
   out_6746448386310637026[1] = 0;
   out_6746448386310637026[2] = 0;
   out_6746448386310637026[3] = 0;
   out_6746448386310637026[4] = 0;
   out_6746448386310637026[5] = 0;
   out_6746448386310637026[6] = 0;
   out_6746448386310637026[7] = 0;
   out_6746448386310637026[8] = 0;
   out_6746448386310637026[9] = 0;
   out_6746448386310637026[10] = 1;
   out_6746448386310637026[11] = 0;
   out_6746448386310637026[12] = 0;
   out_6746448386310637026[13] = 0;
   out_6746448386310637026[14] = 0;
   out_6746448386310637026[15] = 0;
   out_6746448386310637026[16] = 0;
   out_6746448386310637026[17] = 0;
   out_6746448386310637026[18] = 0;
   out_6746448386310637026[19] = 0;
   out_6746448386310637026[20] = 1;
   out_6746448386310637026[21] = 0;
   out_6746448386310637026[22] = 0;
   out_6746448386310637026[23] = 0;
   out_6746448386310637026[24] = 0;
   out_6746448386310637026[25] = 0;
   out_6746448386310637026[26] = 0;
   out_6746448386310637026[27] = 0;
   out_6746448386310637026[28] = 0;
   out_6746448386310637026[29] = 0;
   out_6746448386310637026[30] = 1;
   out_6746448386310637026[31] = 0;
   out_6746448386310637026[32] = 0;
   out_6746448386310637026[33] = 0;
   out_6746448386310637026[34] = 0;
   out_6746448386310637026[35] = 0;
   out_6746448386310637026[36] = 0;
   out_6746448386310637026[37] = 0;
   out_6746448386310637026[38] = 0;
   out_6746448386310637026[39] = 0;
   out_6746448386310637026[40] = 1;
   out_6746448386310637026[41] = 0;
   out_6746448386310637026[42] = 0;
   out_6746448386310637026[43] = 0;
   out_6746448386310637026[44] = 0;
   out_6746448386310637026[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_6746448386310637026[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_6746448386310637026[47] = -dt*stiffness_front*state[0]/state[1];
   out_6746448386310637026[48] = -dt*stiffness_front*state[0]/state[1];
   out_6746448386310637026[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_6746448386310637026[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_6746448386310637026[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_6746448386310637026[52] = dt*stiffness_front*state[0]/state[1];
   out_6746448386310637026[53] = -9.8100000000000005*dt;
   out_6746448386310637026[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_6746448386310637026[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_6746448386310637026[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6746448386310637026[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6746448386310637026[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_6746448386310637026[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_6746448386310637026[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_6746448386310637026[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6746448386310637026[62] = 0;
   out_6746448386310637026[63] = 0;
   out_6746448386310637026[64] = 0;
   out_6746448386310637026[65] = 0;
   out_6746448386310637026[66] = 0;
   out_6746448386310637026[67] = 0;
   out_6746448386310637026[68] = 0;
   out_6746448386310637026[69] = 0;
   out_6746448386310637026[70] = 1;
   out_6746448386310637026[71] = 0;
   out_6746448386310637026[72] = 0;
   out_6746448386310637026[73] = 0;
   out_6746448386310637026[74] = 0;
   out_6746448386310637026[75] = 0;
   out_6746448386310637026[76] = 0;
   out_6746448386310637026[77] = 0;
   out_6746448386310637026[78] = 0;
   out_6746448386310637026[79] = 0;
   out_6746448386310637026[80] = 1;
}
void h_25(double *state, double *unused, double *out_707314913295055078) {
   out_707314913295055078[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1198524098381216623) {
   out_1198524098381216623[0] = 0;
   out_1198524098381216623[1] = 0;
   out_1198524098381216623[2] = 0;
   out_1198524098381216623[3] = 0;
   out_1198524098381216623[4] = 0;
   out_1198524098381216623[5] = 0;
   out_1198524098381216623[6] = 1;
   out_1198524098381216623[7] = 0;
   out_1198524098381216623[8] = 0;
}
void h_24(double *state, double *unused, double *out_6878495411227386244) {
   out_6878495411227386244[0] = state[4];
   out_6878495411227386244[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8616915732058143516) {
   out_8616915732058143516[0] = 0;
   out_8616915732058143516[1] = 0;
   out_8616915732058143516[2] = 0;
   out_8616915732058143516[3] = 0;
   out_8616915732058143516[4] = 1;
   out_8616915732058143516[5] = 0;
   out_8616915732058143516[6] = 0;
   out_8616915732058143516[7] = 0;
   out_8616915732058143516[8] = 0;
   out_8616915732058143516[9] = 0;
   out_8616915732058143516[10] = 0;
   out_8616915732058143516[11] = 0;
   out_8616915732058143516[12] = 0;
   out_8616915732058143516[13] = 0;
   out_8616915732058143516[14] = 1;
   out_8616915732058143516[15] = 0;
   out_8616915732058143516[16] = 0;
   out_8616915732058143516[17] = 0;
}
void h_30(double *state, double *unused, double *out_2240411077060429883) {
   out_2240411077060429883[0] = state[4];
}
void H_30(double *state, double *unused, double *out_1069185151237976553) {
   out_1069185151237976553[0] = 0;
   out_1069185151237976553[1] = 0;
   out_1069185151237976553[2] = 0;
   out_1069185151237976553[3] = 0;
   out_1069185151237976553[4] = 1;
   out_1069185151237976553[5] = 0;
   out_1069185151237976553[6] = 0;
   out_1069185151237976553[7] = 0;
   out_1069185151237976553[8] = 0;
}
void h_26(double *state, double *unused, double *out_5133980333049960676) {
   out_5133980333049960676[0] = state[7];
}
void H_26(double *state, double *unused, double *out_2542979220492839601) {
   out_2542979220492839601[0] = 0;
   out_2542979220492839601[1] = 0;
   out_2542979220492839601[2] = 0;
   out_2542979220492839601[3] = 0;
   out_2542979220492839601[4] = 0;
   out_2542979220492839601[5] = 0;
   out_2542979220492839601[6] = 0;
   out_2542979220492839601[7] = 1;
   out_2542979220492839601[8] = 0;
}
void h_27(double *state, double *unused, double *out_7320798468675099859) {
   out_7320798468675099859[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1105578160562448358) {
   out_1105578160562448358[0] = 0;
   out_1105578160562448358[1] = 0;
   out_1105578160562448358[2] = 0;
   out_1105578160562448358[3] = 1;
   out_1105578160562448358[4] = 0;
   out_1105578160562448358[5] = 0;
   out_1105578160562448358[6] = 0;
   out_1105578160562448358[7] = 0;
   out_1105578160562448358[8] = 0;
}
void h_29(double *state, double *unused, double *out_3244672954280402572) {
   out_3244672954280402572[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2818940887431999391) {
   out_2818940887431999391[0] = 0;
   out_2818940887431999391[1] = 1;
   out_2818940887431999391[2] = 0;
   out_2818940887431999391[3] = 0;
   out_2818940887431999391[4] = 0;
   out_2818940887431999391[5] = 0;
   out_2818940887431999391[6] = 0;
   out_2818940887431999391[7] = 0;
   out_2818940887431999391[8] = 0;
}
void h_28(double *state, double *unused, double *out_6124272337373606112) {
   out_6124272337373606112[0] = state[0];
}
void H_28(double *state, double *unused, double *out_7901339904501529965) {
   out_7901339904501529965[0] = 1;
   out_7901339904501529965[1] = 0;
   out_7901339904501529965[2] = 0;
   out_7901339904501529965[3] = 0;
   out_7901339904501529965[4] = 0;
   out_7901339904501529965[5] = 0;
   out_7901339904501529965[6] = 0;
   out_7901339904501529965[7] = 0;
   out_7901339904501529965[8] = 0;
}
void h_31(double *state, double *unused, double *out_8364353615874624961) {
   out_8364353615874624961[0] = state[8];
}
void H_31(double *state, double *unused, double *out_1229170060258177051) {
   out_1229170060258177051[0] = 0;
   out_1229170060258177051[1] = 0;
   out_1229170060258177051[2] = 0;
   out_1229170060258177051[3] = 0;
   out_1229170060258177051[4] = 0;
   out_1229170060258177051[5] = 0;
   out_1229170060258177051[6] = 0;
   out_1229170060258177051[7] = 0;
   out_1229170060258177051[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_2069516698642958211) {
  err_fun(nom_x, delta_x, out_2069516698642958211);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8998358058995224576) {
  inv_err_fun(nom_x, true_x, out_8998358058995224576);
}
void car_H_mod_fun(double *state, double *out_7695547058271246574) {
  H_mod_fun(state, out_7695547058271246574);
}
void car_f_fun(double *state, double dt, double *out_2845810596498337567) {
  f_fun(state,  dt, out_2845810596498337567);
}
void car_F_fun(double *state, double dt, double *out_6746448386310637026) {
  F_fun(state,  dt, out_6746448386310637026);
}
void car_h_25(double *state, double *unused, double *out_707314913295055078) {
  h_25(state, unused, out_707314913295055078);
}
void car_H_25(double *state, double *unused, double *out_1198524098381216623) {
  H_25(state, unused, out_1198524098381216623);
}
void car_h_24(double *state, double *unused, double *out_6878495411227386244) {
  h_24(state, unused, out_6878495411227386244);
}
void car_H_24(double *state, double *unused, double *out_8616915732058143516) {
  H_24(state, unused, out_8616915732058143516);
}
void car_h_30(double *state, double *unused, double *out_2240411077060429883) {
  h_30(state, unused, out_2240411077060429883);
}
void car_H_30(double *state, double *unused, double *out_1069185151237976553) {
  H_30(state, unused, out_1069185151237976553);
}
void car_h_26(double *state, double *unused, double *out_5133980333049960676) {
  h_26(state, unused, out_5133980333049960676);
}
void car_H_26(double *state, double *unused, double *out_2542979220492839601) {
  H_26(state, unused, out_2542979220492839601);
}
void car_h_27(double *state, double *unused, double *out_7320798468675099859) {
  h_27(state, unused, out_7320798468675099859);
}
void car_H_27(double *state, double *unused, double *out_1105578160562448358) {
  H_27(state, unused, out_1105578160562448358);
}
void car_h_29(double *state, double *unused, double *out_3244672954280402572) {
  h_29(state, unused, out_3244672954280402572);
}
void car_H_29(double *state, double *unused, double *out_2818940887431999391) {
  H_29(state, unused, out_2818940887431999391);
}
void car_h_28(double *state, double *unused, double *out_6124272337373606112) {
  h_28(state, unused, out_6124272337373606112);
}
void car_H_28(double *state, double *unused, double *out_7901339904501529965) {
  H_28(state, unused, out_7901339904501529965);
}
void car_h_31(double *state, double *unused, double *out_8364353615874624961) {
  h_31(state, unused, out_8364353615874624961);
}
void car_H_31(double *state, double *unused, double *out_1229170060258177051) {
  H_31(state, unused, out_1229170060258177051);
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
