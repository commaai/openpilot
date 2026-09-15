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
void err_fun(double *nom_x, double *delta_x, double *out_3156418432254414727) {
   out_3156418432254414727[0] = delta_x[0] + nom_x[0];
   out_3156418432254414727[1] = delta_x[1] + nom_x[1];
   out_3156418432254414727[2] = delta_x[2] + nom_x[2];
   out_3156418432254414727[3] = delta_x[3] + nom_x[3];
   out_3156418432254414727[4] = delta_x[4] + nom_x[4];
   out_3156418432254414727[5] = delta_x[5] + nom_x[5];
   out_3156418432254414727[6] = delta_x[6] + nom_x[6];
   out_3156418432254414727[7] = delta_x[7] + nom_x[7];
   out_3156418432254414727[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_666590418838797292) {
   out_666590418838797292[0] = -nom_x[0] + true_x[0];
   out_666590418838797292[1] = -nom_x[1] + true_x[1];
   out_666590418838797292[2] = -nom_x[2] + true_x[2];
   out_666590418838797292[3] = -nom_x[3] + true_x[3];
   out_666590418838797292[4] = -nom_x[4] + true_x[4];
   out_666590418838797292[5] = -nom_x[5] + true_x[5];
   out_666590418838797292[6] = -nom_x[6] + true_x[6];
   out_666590418838797292[7] = -nom_x[7] + true_x[7];
   out_666590418838797292[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_6212745509768565208) {
   out_6212745509768565208[0] = 1.0;
   out_6212745509768565208[1] = 0.0;
   out_6212745509768565208[2] = 0.0;
   out_6212745509768565208[3] = 0.0;
   out_6212745509768565208[4] = 0.0;
   out_6212745509768565208[5] = 0.0;
   out_6212745509768565208[6] = 0.0;
   out_6212745509768565208[7] = 0.0;
   out_6212745509768565208[8] = 0.0;
   out_6212745509768565208[9] = 0.0;
   out_6212745509768565208[10] = 1.0;
   out_6212745509768565208[11] = 0.0;
   out_6212745509768565208[12] = 0.0;
   out_6212745509768565208[13] = 0.0;
   out_6212745509768565208[14] = 0.0;
   out_6212745509768565208[15] = 0.0;
   out_6212745509768565208[16] = 0.0;
   out_6212745509768565208[17] = 0.0;
   out_6212745509768565208[18] = 0.0;
   out_6212745509768565208[19] = 0.0;
   out_6212745509768565208[20] = 1.0;
   out_6212745509768565208[21] = 0.0;
   out_6212745509768565208[22] = 0.0;
   out_6212745509768565208[23] = 0.0;
   out_6212745509768565208[24] = 0.0;
   out_6212745509768565208[25] = 0.0;
   out_6212745509768565208[26] = 0.0;
   out_6212745509768565208[27] = 0.0;
   out_6212745509768565208[28] = 0.0;
   out_6212745509768565208[29] = 0.0;
   out_6212745509768565208[30] = 1.0;
   out_6212745509768565208[31] = 0.0;
   out_6212745509768565208[32] = 0.0;
   out_6212745509768565208[33] = 0.0;
   out_6212745509768565208[34] = 0.0;
   out_6212745509768565208[35] = 0.0;
   out_6212745509768565208[36] = 0.0;
   out_6212745509768565208[37] = 0.0;
   out_6212745509768565208[38] = 0.0;
   out_6212745509768565208[39] = 0.0;
   out_6212745509768565208[40] = 1.0;
   out_6212745509768565208[41] = 0.0;
   out_6212745509768565208[42] = 0.0;
   out_6212745509768565208[43] = 0.0;
   out_6212745509768565208[44] = 0.0;
   out_6212745509768565208[45] = 0.0;
   out_6212745509768565208[46] = 0.0;
   out_6212745509768565208[47] = 0.0;
   out_6212745509768565208[48] = 0.0;
   out_6212745509768565208[49] = 0.0;
   out_6212745509768565208[50] = 1.0;
   out_6212745509768565208[51] = 0.0;
   out_6212745509768565208[52] = 0.0;
   out_6212745509768565208[53] = 0.0;
   out_6212745509768565208[54] = 0.0;
   out_6212745509768565208[55] = 0.0;
   out_6212745509768565208[56] = 0.0;
   out_6212745509768565208[57] = 0.0;
   out_6212745509768565208[58] = 0.0;
   out_6212745509768565208[59] = 0.0;
   out_6212745509768565208[60] = 1.0;
   out_6212745509768565208[61] = 0.0;
   out_6212745509768565208[62] = 0.0;
   out_6212745509768565208[63] = 0.0;
   out_6212745509768565208[64] = 0.0;
   out_6212745509768565208[65] = 0.0;
   out_6212745509768565208[66] = 0.0;
   out_6212745509768565208[67] = 0.0;
   out_6212745509768565208[68] = 0.0;
   out_6212745509768565208[69] = 0.0;
   out_6212745509768565208[70] = 1.0;
   out_6212745509768565208[71] = 0.0;
   out_6212745509768565208[72] = 0.0;
   out_6212745509768565208[73] = 0.0;
   out_6212745509768565208[74] = 0.0;
   out_6212745509768565208[75] = 0.0;
   out_6212745509768565208[76] = 0.0;
   out_6212745509768565208[77] = 0.0;
   out_6212745509768565208[78] = 0.0;
   out_6212745509768565208[79] = 0.0;
   out_6212745509768565208[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_2453583229613865851) {
   out_2453583229613865851[0] = state[0];
   out_2453583229613865851[1] = state[1];
   out_2453583229613865851[2] = state[2];
   out_2453583229613865851[3] = state[3];
   out_2453583229613865851[4] = state[4];
   out_2453583229613865851[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_2453583229613865851[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_2453583229613865851[7] = state[7];
   out_2453583229613865851[8] = state[8];
}
void F_fun(double *state, double dt, double *out_4617154081104001635) {
   out_4617154081104001635[0] = 1;
   out_4617154081104001635[1] = 0;
   out_4617154081104001635[2] = 0;
   out_4617154081104001635[3] = 0;
   out_4617154081104001635[4] = 0;
   out_4617154081104001635[5] = 0;
   out_4617154081104001635[6] = 0;
   out_4617154081104001635[7] = 0;
   out_4617154081104001635[8] = 0;
   out_4617154081104001635[9] = 0;
   out_4617154081104001635[10] = 1;
   out_4617154081104001635[11] = 0;
   out_4617154081104001635[12] = 0;
   out_4617154081104001635[13] = 0;
   out_4617154081104001635[14] = 0;
   out_4617154081104001635[15] = 0;
   out_4617154081104001635[16] = 0;
   out_4617154081104001635[17] = 0;
   out_4617154081104001635[18] = 0;
   out_4617154081104001635[19] = 0;
   out_4617154081104001635[20] = 1;
   out_4617154081104001635[21] = 0;
   out_4617154081104001635[22] = 0;
   out_4617154081104001635[23] = 0;
   out_4617154081104001635[24] = 0;
   out_4617154081104001635[25] = 0;
   out_4617154081104001635[26] = 0;
   out_4617154081104001635[27] = 0;
   out_4617154081104001635[28] = 0;
   out_4617154081104001635[29] = 0;
   out_4617154081104001635[30] = 1;
   out_4617154081104001635[31] = 0;
   out_4617154081104001635[32] = 0;
   out_4617154081104001635[33] = 0;
   out_4617154081104001635[34] = 0;
   out_4617154081104001635[35] = 0;
   out_4617154081104001635[36] = 0;
   out_4617154081104001635[37] = 0;
   out_4617154081104001635[38] = 0;
   out_4617154081104001635[39] = 0;
   out_4617154081104001635[40] = 1;
   out_4617154081104001635[41] = 0;
   out_4617154081104001635[42] = 0;
   out_4617154081104001635[43] = 0;
   out_4617154081104001635[44] = 0;
   out_4617154081104001635[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_4617154081104001635[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_4617154081104001635[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4617154081104001635[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4617154081104001635[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_4617154081104001635[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_4617154081104001635[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_4617154081104001635[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_4617154081104001635[53] = -9.8100000000000005*dt;
   out_4617154081104001635[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4617154081104001635[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4617154081104001635[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4617154081104001635[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4617154081104001635[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4617154081104001635[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4617154081104001635[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4617154081104001635[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4617154081104001635[62] = 0;
   out_4617154081104001635[63] = 0;
   out_4617154081104001635[64] = 0;
   out_4617154081104001635[65] = 0;
   out_4617154081104001635[66] = 0;
   out_4617154081104001635[67] = 0;
   out_4617154081104001635[68] = 0;
   out_4617154081104001635[69] = 0;
   out_4617154081104001635[70] = 1;
   out_4617154081104001635[71] = 0;
   out_4617154081104001635[72] = 0;
   out_4617154081104001635[73] = 0;
   out_4617154081104001635[74] = 0;
   out_4617154081104001635[75] = 0;
   out_4617154081104001635[76] = 0;
   out_4617154081104001635[77] = 0;
   out_4617154081104001635[78] = 0;
   out_4617154081104001635[79] = 0;
   out_4617154081104001635[80] = 1;
}
void h_25(double *state, double *unused, double *out_8034832062215994505) {
   out_8034832062215994505[0] = state[6];
}
void H_25(double *state, double *unused, double *out_2980442716143672029) {
   out_2980442716143672029[0] = 0;
   out_2980442716143672029[1] = 0;
   out_2980442716143672029[2] = 0;
   out_2980442716143672029[3] = 0;
   out_2980442716143672029[4] = 0;
   out_2980442716143672029[5] = 0;
   out_2980442716143672029[6] = 1;
   out_2980442716143672029[7] = 0;
   out_2980442716143672029[8] = 0;
}
void h_24(double *state, double *unused, double *out_2871818575794667755) {
   out_2871818575794667755[0] = state[4];
   out_2871818575794667755[1] = state[5];
}
void H_24(double *state, double *unused, double *out_7726952043110181063) {
   out_7726952043110181063[0] = 0;
   out_7726952043110181063[1] = 0;
   out_7726952043110181063[2] = 0;
   out_7726952043110181063[3] = 0;
   out_7726952043110181063[4] = 1;
   out_7726952043110181063[5] = 0;
   out_7726952043110181063[6] = 0;
   out_7726952043110181063[7] = 0;
   out_7726952043110181063[8] = 0;
   out_7726952043110181063[9] = 0;
   out_7726952043110181063[10] = 0;
   out_7726952043110181063[11] = 0;
   out_7726952043110181063[12] = 0;
   out_7726952043110181063[13] = 0;
   out_7726952043110181063[14] = 1;
   out_7726952043110181063[15] = 0;
   out_7726952043110181063[16] = 0;
   out_7726952043110181063[17] = 0;
}
void h_30(double *state, double *unused, double *out_6313608713214185247) {
   out_6313608713214185247[0] = state[4];
}
void H_30(double *state, double *unused, double *out_3936247625347944726) {
   out_3936247625347944726[0] = 0;
   out_3936247625347944726[1] = 0;
   out_3936247625347944726[2] = 0;
   out_3936247625347944726[3] = 0;
   out_3936247625347944726[4] = 1;
   out_3936247625347944726[5] = 0;
   out_3936247625347944726[6] = 0;
   out_3936247625347944726[7] = 0;
   out_3936247625347944726[8] = 0;
}
void h_26(double *state, double *unused, double *out_9096349309051033684) {
   out_9096349309051033684[0] = state[7];
}
void H_26(double *state, double *unused, double *out_6721946035017728253) {
   out_6721946035017728253[0] = 0;
   out_6721946035017728253[1] = 0;
   out_6721946035017728253[2] = 0;
   out_6721946035017728253[3] = 0;
   out_6721946035017728253[4] = 0;
   out_6721946035017728253[5] = 0;
   out_6721946035017728253[6] = 0;
   out_6721946035017728253[7] = 1;
   out_6721946035017728253[8] = 0;
}
void h_27(double *state, double *unused, double *out_2167868242960558368) {
   out_2167868242960558368[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1761484313547519815) {
   out_1761484313547519815[0] = 0;
   out_1761484313547519815[1] = 0;
   out_1761484313547519815[2] = 0;
   out_1761484313547519815[3] = 1;
   out_1761484313547519815[4] = 0;
   out_1761484313547519815[5] = 0;
   out_1761484313547519815[6] = 0;
   out_1761484313547519815[7] = 0;
   out_1761484313547519815[8] = 0;
}
void h_29(double *state, double *unused, double *out_828524681873352792) {
   out_828524681873352792[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4446478969662336910) {
   out_4446478969662336910[0] = 0;
   out_4446478969662336910[1] = 1;
   out_4446478969662336910[2] = 0;
   out_4446478969662336910[3] = 0;
   out_4446478969662336910[4] = 0;
   out_4446478969662336910[5] = 0;
   out_4446478969662336910[6] = 0;
   out_4446478969662336910[7] = 0;
   out_4446478969662336910[8] = 0;
}
void h_28(double *state, double *unused, double *out_6431616107147561991) {
   out_6431616107147561991[0] = state[0];
}
void H_28(double *state, double *unused, double *out_5034277430391561792) {
   out_5034277430391561792[0] = 1;
   out_5034277430391561792[1] = 0;
   out_5034277430391561792[2] = 0;
   out_5034277430391561792[3] = 0;
   out_5034277430391561792[4] = 0;
   out_5034277430391561792[5] = 0;
   out_5034277430391561792[6] = 0;
   out_5034277430391561792[7] = 0;
   out_5034277430391561792[8] = 0;
}
void h_31(double *state, double *unused, double *out_4291155979720869597) {
   out_4291155979720869597[0] = state[8];
}
void H_31(double *state, double *unused, double *out_2949796754266711601) {
   out_2949796754266711601[0] = 0;
   out_2949796754266711601[1] = 0;
   out_2949796754266711601[2] = 0;
   out_2949796754266711601[3] = 0;
   out_2949796754266711601[4] = 0;
   out_2949796754266711601[5] = 0;
   out_2949796754266711601[6] = 0;
   out_2949796754266711601[7] = 0;
   out_2949796754266711601[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_3156418432254414727) {
  err_fun(nom_x, delta_x, out_3156418432254414727);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_666590418838797292) {
  inv_err_fun(nom_x, true_x, out_666590418838797292);
}
void car_H_mod_fun(double *state, double *out_6212745509768565208) {
  H_mod_fun(state, out_6212745509768565208);
}
void car_f_fun(double *state, double dt, double *out_2453583229613865851) {
  f_fun(state,  dt, out_2453583229613865851);
}
void car_F_fun(double *state, double dt, double *out_4617154081104001635) {
  F_fun(state,  dt, out_4617154081104001635);
}
void car_h_25(double *state, double *unused, double *out_8034832062215994505) {
  h_25(state, unused, out_8034832062215994505);
}
void car_H_25(double *state, double *unused, double *out_2980442716143672029) {
  H_25(state, unused, out_2980442716143672029);
}
void car_h_24(double *state, double *unused, double *out_2871818575794667755) {
  h_24(state, unused, out_2871818575794667755);
}
void car_H_24(double *state, double *unused, double *out_7726952043110181063) {
  H_24(state, unused, out_7726952043110181063);
}
void car_h_30(double *state, double *unused, double *out_6313608713214185247) {
  h_30(state, unused, out_6313608713214185247);
}
void car_H_30(double *state, double *unused, double *out_3936247625347944726) {
  H_30(state, unused, out_3936247625347944726);
}
void car_h_26(double *state, double *unused, double *out_9096349309051033684) {
  h_26(state, unused, out_9096349309051033684);
}
void car_H_26(double *state, double *unused, double *out_6721946035017728253) {
  H_26(state, unused, out_6721946035017728253);
}
void car_h_27(double *state, double *unused, double *out_2167868242960558368) {
  h_27(state, unused, out_2167868242960558368);
}
void car_H_27(double *state, double *unused, double *out_1761484313547519815) {
  H_27(state, unused, out_1761484313547519815);
}
void car_h_29(double *state, double *unused, double *out_828524681873352792) {
  h_29(state, unused, out_828524681873352792);
}
void car_H_29(double *state, double *unused, double *out_4446478969662336910) {
  H_29(state, unused, out_4446478969662336910);
}
void car_h_28(double *state, double *unused, double *out_6431616107147561991) {
  h_28(state, unused, out_6431616107147561991);
}
void car_H_28(double *state, double *unused, double *out_5034277430391561792) {
  H_28(state, unused, out_5034277430391561792);
}
void car_h_31(double *state, double *unused, double *out_4291155979720869597) {
  h_31(state, unused, out_4291155979720869597);
}
void car_H_31(double *state, double *unused, double *out_2949796754266711601) {
  H_31(state, unused, out_2949796754266711601);
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
