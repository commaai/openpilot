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
void err_fun(double *nom_x, double *delta_x, double *out_5914414296846804453) {
   out_5914414296846804453[0] = delta_x[0] + nom_x[0];
   out_5914414296846804453[1] = delta_x[1] + nom_x[1];
   out_5914414296846804453[2] = delta_x[2] + nom_x[2];
   out_5914414296846804453[3] = delta_x[3] + nom_x[3];
   out_5914414296846804453[4] = delta_x[4] + nom_x[4];
   out_5914414296846804453[5] = delta_x[5] + nom_x[5];
   out_5914414296846804453[6] = delta_x[6] + nom_x[6];
   out_5914414296846804453[7] = delta_x[7] + nom_x[7];
   out_5914414296846804453[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5011141162173190918) {
   out_5011141162173190918[0] = -nom_x[0] + true_x[0];
   out_5011141162173190918[1] = -nom_x[1] + true_x[1];
   out_5011141162173190918[2] = -nom_x[2] + true_x[2];
   out_5011141162173190918[3] = -nom_x[3] + true_x[3];
   out_5011141162173190918[4] = -nom_x[4] + true_x[4];
   out_5011141162173190918[5] = -nom_x[5] + true_x[5];
   out_5011141162173190918[6] = -nom_x[6] + true_x[6];
   out_5011141162173190918[7] = -nom_x[7] + true_x[7];
   out_5011141162173190918[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_4291403483710066446) {
   out_4291403483710066446[0] = 1.0;
   out_4291403483710066446[1] = 0.0;
   out_4291403483710066446[2] = 0.0;
   out_4291403483710066446[3] = 0.0;
   out_4291403483710066446[4] = 0.0;
   out_4291403483710066446[5] = 0.0;
   out_4291403483710066446[6] = 0.0;
   out_4291403483710066446[7] = 0.0;
   out_4291403483710066446[8] = 0.0;
   out_4291403483710066446[9] = 0.0;
   out_4291403483710066446[10] = 1.0;
   out_4291403483710066446[11] = 0.0;
   out_4291403483710066446[12] = 0.0;
   out_4291403483710066446[13] = 0.0;
   out_4291403483710066446[14] = 0.0;
   out_4291403483710066446[15] = 0.0;
   out_4291403483710066446[16] = 0.0;
   out_4291403483710066446[17] = 0.0;
   out_4291403483710066446[18] = 0.0;
   out_4291403483710066446[19] = 0.0;
   out_4291403483710066446[20] = 1.0;
   out_4291403483710066446[21] = 0.0;
   out_4291403483710066446[22] = 0.0;
   out_4291403483710066446[23] = 0.0;
   out_4291403483710066446[24] = 0.0;
   out_4291403483710066446[25] = 0.0;
   out_4291403483710066446[26] = 0.0;
   out_4291403483710066446[27] = 0.0;
   out_4291403483710066446[28] = 0.0;
   out_4291403483710066446[29] = 0.0;
   out_4291403483710066446[30] = 1.0;
   out_4291403483710066446[31] = 0.0;
   out_4291403483710066446[32] = 0.0;
   out_4291403483710066446[33] = 0.0;
   out_4291403483710066446[34] = 0.0;
   out_4291403483710066446[35] = 0.0;
   out_4291403483710066446[36] = 0.0;
   out_4291403483710066446[37] = 0.0;
   out_4291403483710066446[38] = 0.0;
   out_4291403483710066446[39] = 0.0;
   out_4291403483710066446[40] = 1.0;
   out_4291403483710066446[41] = 0.0;
   out_4291403483710066446[42] = 0.0;
   out_4291403483710066446[43] = 0.0;
   out_4291403483710066446[44] = 0.0;
   out_4291403483710066446[45] = 0.0;
   out_4291403483710066446[46] = 0.0;
   out_4291403483710066446[47] = 0.0;
   out_4291403483710066446[48] = 0.0;
   out_4291403483710066446[49] = 0.0;
   out_4291403483710066446[50] = 1.0;
   out_4291403483710066446[51] = 0.0;
   out_4291403483710066446[52] = 0.0;
   out_4291403483710066446[53] = 0.0;
   out_4291403483710066446[54] = 0.0;
   out_4291403483710066446[55] = 0.0;
   out_4291403483710066446[56] = 0.0;
   out_4291403483710066446[57] = 0.0;
   out_4291403483710066446[58] = 0.0;
   out_4291403483710066446[59] = 0.0;
   out_4291403483710066446[60] = 1.0;
   out_4291403483710066446[61] = 0.0;
   out_4291403483710066446[62] = 0.0;
   out_4291403483710066446[63] = 0.0;
   out_4291403483710066446[64] = 0.0;
   out_4291403483710066446[65] = 0.0;
   out_4291403483710066446[66] = 0.0;
   out_4291403483710066446[67] = 0.0;
   out_4291403483710066446[68] = 0.0;
   out_4291403483710066446[69] = 0.0;
   out_4291403483710066446[70] = 1.0;
   out_4291403483710066446[71] = 0.0;
   out_4291403483710066446[72] = 0.0;
   out_4291403483710066446[73] = 0.0;
   out_4291403483710066446[74] = 0.0;
   out_4291403483710066446[75] = 0.0;
   out_4291403483710066446[76] = 0.0;
   out_4291403483710066446[77] = 0.0;
   out_4291403483710066446[78] = 0.0;
   out_4291403483710066446[79] = 0.0;
   out_4291403483710066446[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_6295743357823790773) {
   out_6295743357823790773[0] = state[0];
   out_6295743357823790773[1] = state[1];
   out_6295743357823790773[2] = state[2];
   out_6295743357823790773[3] = state[3];
   out_6295743357823790773[4] = state[4];
   out_6295743357823790773[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_6295743357823790773[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_6295743357823790773[7] = state[7];
   out_6295743357823790773[8] = state[8];
}
void F_fun(double *state, double dt, double *out_428584837545743588) {
   out_428584837545743588[0] = 1;
   out_428584837545743588[1] = 0;
   out_428584837545743588[2] = 0;
   out_428584837545743588[3] = 0;
   out_428584837545743588[4] = 0;
   out_428584837545743588[5] = 0;
   out_428584837545743588[6] = 0;
   out_428584837545743588[7] = 0;
   out_428584837545743588[8] = 0;
   out_428584837545743588[9] = 0;
   out_428584837545743588[10] = 1;
   out_428584837545743588[11] = 0;
   out_428584837545743588[12] = 0;
   out_428584837545743588[13] = 0;
   out_428584837545743588[14] = 0;
   out_428584837545743588[15] = 0;
   out_428584837545743588[16] = 0;
   out_428584837545743588[17] = 0;
   out_428584837545743588[18] = 0;
   out_428584837545743588[19] = 0;
   out_428584837545743588[20] = 1;
   out_428584837545743588[21] = 0;
   out_428584837545743588[22] = 0;
   out_428584837545743588[23] = 0;
   out_428584837545743588[24] = 0;
   out_428584837545743588[25] = 0;
   out_428584837545743588[26] = 0;
   out_428584837545743588[27] = 0;
   out_428584837545743588[28] = 0;
   out_428584837545743588[29] = 0;
   out_428584837545743588[30] = 1;
   out_428584837545743588[31] = 0;
   out_428584837545743588[32] = 0;
   out_428584837545743588[33] = 0;
   out_428584837545743588[34] = 0;
   out_428584837545743588[35] = 0;
   out_428584837545743588[36] = 0;
   out_428584837545743588[37] = 0;
   out_428584837545743588[38] = 0;
   out_428584837545743588[39] = 0;
   out_428584837545743588[40] = 1;
   out_428584837545743588[41] = 0;
   out_428584837545743588[42] = 0;
   out_428584837545743588[43] = 0;
   out_428584837545743588[44] = 0;
   out_428584837545743588[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_428584837545743588[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_428584837545743588[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_428584837545743588[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_428584837545743588[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_428584837545743588[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_428584837545743588[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_428584837545743588[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_428584837545743588[53] = -9.8000000000000007*dt;
   out_428584837545743588[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_428584837545743588[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_428584837545743588[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_428584837545743588[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_428584837545743588[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_428584837545743588[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_428584837545743588[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_428584837545743588[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_428584837545743588[62] = 0;
   out_428584837545743588[63] = 0;
   out_428584837545743588[64] = 0;
   out_428584837545743588[65] = 0;
   out_428584837545743588[66] = 0;
   out_428584837545743588[67] = 0;
   out_428584837545743588[68] = 0;
   out_428584837545743588[69] = 0;
   out_428584837545743588[70] = 1;
   out_428584837545743588[71] = 0;
   out_428584837545743588[72] = 0;
   out_428584837545743588[73] = 0;
   out_428584837545743588[74] = 0;
   out_428584837545743588[75] = 0;
   out_428584837545743588[76] = 0;
   out_428584837545743588[77] = 0;
   out_428584837545743588[78] = 0;
   out_428584837545743588[79] = 0;
   out_428584837545743588[80] = 1;
}
void h_25(double *state, double *unused, double *out_2854088158423216169) {
   out_2854088158423216169[0] = state[6];
}
void H_25(double *state, double *unused, double *out_5240524405738548655) {
   out_5240524405738548655[0] = 0;
   out_5240524405738548655[1] = 0;
   out_5240524405738548655[2] = 0;
   out_5240524405738548655[3] = 0;
   out_5240524405738548655[4] = 0;
   out_5240524405738548655[5] = 0;
   out_5240524405738548655[6] = 1;
   out_5240524405738548655[7] = 0;
   out_5240524405738548655[8] = 0;
}
void h_24(double *state, double *unused, double *out_359654079315781340) {
   out_359654079315781340[0] = state[4];
   out_359654079315781340[1] = state[5];
}
void H_24(double *state, double *unused, double *out_371709540710841803) {
   out_371709540710841803[0] = 0;
   out_371709540710841803[1] = 0;
   out_371709540710841803[2] = 0;
   out_371709540710841803[3] = 0;
   out_371709540710841803[4] = 1;
   out_371709540710841803[5] = 0;
   out_371709540710841803[6] = 0;
   out_371709540710841803[7] = 0;
   out_371709540710841803[8] = 0;
   out_371709540710841803[9] = 0;
   out_371709540710841803[10] = 0;
   out_371709540710841803[11] = 0;
   out_371709540710841803[12] = 0;
   out_371709540710841803[13] = 0;
   out_371709540710841803[14] = 1;
   out_371709540710841803[15] = 0;
   out_371709540710841803[16] = 0;
   out_371709540710841803[17] = 0;
}
void h_30(double *state, double *unused, double *out_3011274826774345314) {
   out_3011274826774345314[0] = state[4];
}
void H_30(double *state, double *unused, double *out_712828075610940457) {
   out_712828075610940457[0] = 0;
   out_712828075610940457[1] = 0;
   out_712828075610940457[2] = 0;
   out_712828075610940457[3] = 0;
   out_712828075610940457[4] = 1;
   out_712828075610940457[5] = 0;
   out_712828075610940457[6] = 0;
   out_712828075610940457[7] = 0;
   out_712828075610940457[8] = 0;
}
void h_26(double *state, double *unused, double *out_7436635742072576847) {
   out_7436635742072576847[0] = state[7];
}
void H_26(double *state, double *unused, double *out_1499021086864492431) {
   out_1499021086864492431[0] = 0;
   out_1499021086864492431[1] = 0;
   out_1499021086864492431[2] = 0;
   out_1499021086864492431[3] = 0;
   out_1499021086864492431[4] = 0;
   out_1499021086864492431[5] = 0;
   out_1499021086864492431[6] = 0;
   out_1499021086864492431[7] = 1;
   out_1499021086864492431[8] = 0;
}
void h_27(double *state, double *unused, double *out_1005994483947410829) {
   out_1005994483947410829[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2936422146794883674) {
   out_2936422146794883674[0] = 0;
   out_2936422146794883674[1] = 0;
   out_2936422146794883674[2] = 0;
   out_2936422146794883674[3] = 1;
   out_2936422146794883674[4] = 0;
   out_2936422146794883674[5] = 0;
   out_2936422146794883674[6] = 0;
   out_2936422146794883674[7] = 0;
   out_2936422146794883674[8] = 0;
}
void h_29(double *state, double *unused, double *out_1990398440886500331) {
   out_1990398440886500331[0] = state[1];
}
void H_29(double *state, double *unused, double *out_1223059419925332641) {
   out_1223059419925332641[0] = 0;
   out_1223059419925332641[1] = 1;
   out_1223059419925332641[2] = 0;
   out_1223059419925332641[3] = 0;
   out_1223059419925332641[4] = 0;
   out_1223059419925332641[5] = 0;
   out_1223059419925332641[6] = 0;
   out_1223059419925332641[7] = 0;
   out_1223059419925332641[8] = 0;
}
void h_28(double *state, double *unused, double *out_7070785832501170307) {
   out_7070785832501170307[0] = state[0];
}
void H_28(double *state, double *unused, double *out_3186689691490658892) {
   out_3186689691490658892[0] = 1;
   out_3186689691490658892[1] = 0;
   out_3186689691490658892[2] = 0;
   out_3186689691490658892[3] = 0;
   out_3186689691490658892[4] = 0;
   out_3186689691490658892[5] = 0;
   out_3186689691490658892[6] = 0;
   out_3186689691490658892[7] = 0;
   out_3186689691490658892[8] = 0;
}
void h_31(double *state, double *unused, double *out_4606470605825672156) {
   out_4606470605825672156[0] = state[8];
}
void H_31(double *state, double *unused, double *out_872812984631140955) {
   out_872812984631140955[0] = 0;
   out_872812984631140955[1] = 0;
   out_872812984631140955[2] = 0;
   out_872812984631140955[3] = 0;
   out_872812984631140955[4] = 0;
   out_872812984631140955[5] = 0;
   out_872812984631140955[6] = 0;
   out_872812984631140955[7] = 0;
   out_872812984631140955[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_5914414296846804453) {
  err_fun(nom_x, delta_x, out_5914414296846804453);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5011141162173190918) {
  inv_err_fun(nom_x, true_x, out_5011141162173190918);
}
void car_H_mod_fun(double *state, double *out_4291403483710066446) {
  H_mod_fun(state, out_4291403483710066446);
}
void car_f_fun(double *state, double dt, double *out_6295743357823790773) {
  f_fun(state,  dt, out_6295743357823790773);
}
void car_F_fun(double *state, double dt, double *out_428584837545743588) {
  F_fun(state,  dt, out_428584837545743588);
}
void car_h_25(double *state, double *unused, double *out_2854088158423216169) {
  h_25(state, unused, out_2854088158423216169);
}
void car_H_25(double *state, double *unused, double *out_5240524405738548655) {
  H_25(state, unused, out_5240524405738548655);
}
void car_h_24(double *state, double *unused, double *out_359654079315781340) {
  h_24(state, unused, out_359654079315781340);
}
void car_H_24(double *state, double *unused, double *out_371709540710841803) {
  H_24(state, unused, out_371709540710841803);
}
void car_h_30(double *state, double *unused, double *out_3011274826774345314) {
  h_30(state, unused, out_3011274826774345314);
}
void car_H_30(double *state, double *unused, double *out_712828075610940457) {
  H_30(state, unused, out_712828075610940457);
}
void car_h_26(double *state, double *unused, double *out_7436635742072576847) {
  h_26(state, unused, out_7436635742072576847);
}
void car_H_26(double *state, double *unused, double *out_1499021086864492431) {
  H_26(state, unused, out_1499021086864492431);
}
void car_h_27(double *state, double *unused, double *out_1005994483947410829) {
  h_27(state, unused, out_1005994483947410829);
}
void car_H_27(double *state, double *unused, double *out_2936422146794883674) {
  H_27(state, unused, out_2936422146794883674);
}
void car_h_29(double *state, double *unused, double *out_1990398440886500331) {
  h_29(state, unused, out_1990398440886500331);
}
void car_H_29(double *state, double *unused, double *out_1223059419925332641) {
  H_29(state, unused, out_1223059419925332641);
}
void car_h_28(double *state, double *unused, double *out_7070785832501170307) {
  h_28(state, unused, out_7070785832501170307);
}
void car_H_28(double *state, double *unused, double *out_3186689691490658892) {
  H_28(state, unused, out_3186689691490658892);
}
void car_h_31(double *state, double *unused, double *out_4606470605825672156) {
  h_31(state, unused, out_4606470605825672156);
}
void car_H_31(double *state, double *unused, double *out_872812984631140955) {
  H_31(state, unused, out_872812984631140955);
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
