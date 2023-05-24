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
 *                      Code generated with SymPy 1.11.1                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_1299726308890531037) {
   out_1299726308890531037[0] = delta_x[0] + nom_x[0];
   out_1299726308890531037[1] = delta_x[1] + nom_x[1];
   out_1299726308890531037[2] = delta_x[2] + nom_x[2];
   out_1299726308890531037[3] = delta_x[3] + nom_x[3];
   out_1299726308890531037[4] = delta_x[4] + nom_x[4];
   out_1299726308890531037[5] = delta_x[5] + nom_x[5];
   out_1299726308890531037[6] = delta_x[6] + nom_x[6];
   out_1299726308890531037[7] = delta_x[7] + nom_x[7];
   out_1299726308890531037[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2964139867777846776) {
   out_2964139867777846776[0] = -nom_x[0] + true_x[0];
   out_2964139867777846776[1] = -nom_x[1] + true_x[1];
   out_2964139867777846776[2] = -nom_x[2] + true_x[2];
   out_2964139867777846776[3] = -nom_x[3] + true_x[3];
   out_2964139867777846776[4] = -nom_x[4] + true_x[4];
   out_2964139867777846776[5] = -nom_x[5] + true_x[5];
   out_2964139867777846776[6] = -nom_x[6] + true_x[6];
   out_2964139867777846776[7] = -nom_x[7] + true_x[7];
   out_2964139867777846776[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_8544953491798085886) {
   out_8544953491798085886[0] = 1.0;
   out_8544953491798085886[1] = 0;
   out_8544953491798085886[2] = 0;
   out_8544953491798085886[3] = 0;
   out_8544953491798085886[4] = 0;
   out_8544953491798085886[5] = 0;
   out_8544953491798085886[6] = 0;
   out_8544953491798085886[7] = 0;
   out_8544953491798085886[8] = 0;
   out_8544953491798085886[9] = 0;
   out_8544953491798085886[10] = 1.0;
   out_8544953491798085886[11] = 0;
   out_8544953491798085886[12] = 0;
   out_8544953491798085886[13] = 0;
   out_8544953491798085886[14] = 0;
   out_8544953491798085886[15] = 0;
   out_8544953491798085886[16] = 0;
   out_8544953491798085886[17] = 0;
   out_8544953491798085886[18] = 0;
   out_8544953491798085886[19] = 0;
   out_8544953491798085886[20] = 1.0;
   out_8544953491798085886[21] = 0;
   out_8544953491798085886[22] = 0;
   out_8544953491798085886[23] = 0;
   out_8544953491798085886[24] = 0;
   out_8544953491798085886[25] = 0;
   out_8544953491798085886[26] = 0;
   out_8544953491798085886[27] = 0;
   out_8544953491798085886[28] = 0;
   out_8544953491798085886[29] = 0;
   out_8544953491798085886[30] = 1.0;
   out_8544953491798085886[31] = 0;
   out_8544953491798085886[32] = 0;
   out_8544953491798085886[33] = 0;
   out_8544953491798085886[34] = 0;
   out_8544953491798085886[35] = 0;
   out_8544953491798085886[36] = 0;
   out_8544953491798085886[37] = 0;
   out_8544953491798085886[38] = 0;
   out_8544953491798085886[39] = 0;
   out_8544953491798085886[40] = 1.0;
   out_8544953491798085886[41] = 0;
   out_8544953491798085886[42] = 0;
   out_8544953491798085886[43] = 0;
   out_8544953491798085886[44] = 0;
   out_8544953491798085886[45] = 0;
   out_8544953491798085886[46] = 0;
   out_8544953491798085886[47] = 0;
   out_8544953491798085886[48] = 0;
   out_8544953491798085886[49] = 0;
   out_8544953491798085886[50] = 1.0;
   out_8544953491798085886[51] = 0;
   out_8544953491798085886[52] = 0;
   out_8544953491798085886[53] = 0;
   out_8544953491798085886[54] = 0;
   out_8544953491798085886[55] = 0;
   out_8544953491798085886[56] = 0;
   out_8544953491798085886[57] = 0;
   out_8544953491798085886[58] = 0;
   out_8544953491798085886[59] = 0;
   out_8544953491798085886[60] = 1.0;
   out_8544953491798085886[61] = 0;
   out_8544953491798085886[62] = 0;
   out_8544953491798085886[63] = 0;
   out_8544953491798085886[64] = 0;
   out_8544953491798085886[65] = 0;
   out_8544953491798085886[66] = 0;
   out_8544953491798085886[67] = 0;
   out_8544953491798085886[68] = 0;
   out_8544953491798085886[69] = 0;
   out_8544953491798085886[70] = 1.0;
   out_8544953491798085886[71] = 0;
   out_8544953491798085886[72] = 0;
   out_8544953491798085886[73] = 0;
   out_8544953491798085886[74] = 0;
   out_8544953491798085886[75] = 0;
   out_8544953491798085886[76] = 0;
   out_8544953491798085886[77] = 0;
   out_8544953491798085886[78] = 0;
   out_8544953491798085886[79] = 0;
   out_8544953491798085886[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_5256147688816920328) {
   out_5256147688816920328[0] = state[0];
   out_5256147688816920328[1] = state[1];
   out_5256147688816920328[2] = state[2];
   out_5256147688816920328[3] = state[3];
   out_5256147688816920328[4] = state[4];
   out_5256147688816920328[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_5256147688816920328[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_5256147688816920328[7] = state[7];
   out_5256147688816920328[8] = state[8];
}
void F_fun(double *state, double dt, double *out_6834940091299106646) {
   out_6834940091299106646[0] = 1;
   out_6834940091299106646[1] = 0;
   out_6834940091299106646[2] = 0;
   out_6834940091299106646[3] = 0;
   out_6834940091299106646[4] = 0;
   out_6834940091299106646[5] = 0;
   out_6834940091299106646[6] = 0;
   out_6834940091299106646[7] = 0;
   out_6834940091299106646[8] = 0;
   out_6834940091299106646[9] = 0;
   out_6834940091299106646[10] = 1;
   out_6834940091299106646[11] = 0;
   out_6834940091299106646[12] = 0;
   out_6834940091299106646[13] = 0;
   out_6834940091299106646[14] = 0;
   out_6834940091299106646[15] = 0;
   out_6834940091299106646[16] = 0;
   out_6834940091299106646[17] = 0;
   out_6834940091299106646[18] = 0;
   out_6834940091299106646[19] = 0;
   out_6834940091299106646[20] = 1;
   out_6834940091299106646[21] = 0;
   out_6834940091299106646[22] = 0;
   out_6834940091299106646[23] = 0;
   out_6834940091299106646[24] = 0;
   out_6834940091299106646[25] = 0;
   out_6834940091299106646[26] = 0;
   out_6834940091299106646[27] = 0;
   out_6834940091299106646[28] = 0;
   out_6834940091299106646[29] = 0;
   out_6834940091299106646[30] = 1;
   out_6834940091299106646[31] = 0;
   out_6834940091299106646[32] = 0;
   out_6834940091299106646[33] = 0;
   out_6834940091299106646[34] = 0;
   out_6834940091299106646[35] = 0;
   out_6834940091299106646[36] = 0;
   out_6834940091299106646[37] = 0;
   out_6834940091299106646[38] = 0;
   out_6834940091299106646[39] = 0;
   out_6834940091299106646[40] = 1;
   out_6834940091299106646[41] = 0;
   out_6834940091299106646[42] = 0;
   out_6834940091299106646[43] = 0;
   out_6834940091299106646[44] = 0;
   out_6834940091299106646[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_6834940091299106646[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_6834940091299106646[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6834940091299106646[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6834940091299106646[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_6834940091299106646[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_6834940091299106646[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_6834940091299106646[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_6834940091299106646[53] = -9.8000000000000007*dt;
   out_6834940091299106646[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_6834940091299106646[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_6834940091299106646[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6834940091299106646[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6834940091299106646[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_6834940091299106646[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_6834940091299106646[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_6834940091299106646[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6834940091299106646[62] = 0;
   out_6834940091299106646[63] = 0;
   out_6834940091299106646[64] = 0;
   out_6834940091299106646[65] = 0;
   out_6834940091299106646[66] = 0;
   out_6834940091299106646[67] = 0;
   out_6834940091299106646[68] = 0;
   out_6834940091299106646[69] = 0;
   out_6834940091299106646[70] = 1;
   out_6834940091299106646[71] = 0;
   out_6834940091299106646[72] = 0;
   out_6834940091299106646[73] = 0;
   out_6834940091299106646[74] = 0;
   out_6834940091299106646[75] = 0;
   out_6834940091299106646[76] = 0;
   out_6834940091299106646[77] = 0;
   out_6834940091299106646[78] = 0;
   out_6834940091299106646[79] = 0;
   out_6834940091299106646[80] = 1;
}
void h_25(double *state, double *unused, double *out_6166873754981284210) {
   out_6166873754981284210[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7142658040254357010) {
   out_7142658040254357010[0] = 0;
   out_7142658040254357010[1] = 0;
   out_7142658040254357010[2] = 0;
   out_7142658040254357010[3] = 0;
   out_7142658040254357010[4] = 0;
   out_7142658040254357010[5] = 0;
   out_7142658040254357010[6] = 1;
   out_7142658040254357010[7] = 0;
   out_7142658040254357010[8] = 0;
}
void h_24(double *state, double *unused, double *out_2192149979727043903) {
   out_2192149979727043903[0] = state[4];
   out_2192149979727043903[1] = state[5];
}
void H_24(double *state, double *unused, double *out_1331272409231580045) {
   out_1331272409231580045[0] = 0;
   out_1331272409231580045[1] = 0;
   out_1331272409231580045[2] = 0;
   out_1331272409231580045[3] = 0;
   out_1331272409231580045[4] = 1;
   out_1331272409231580045[5] = 0;
   out_1331272409231580045[6] = 0;
   out_1331272409231580045[7] = 0;
   out_1331272409231580045[8] = 0;
   out_1331272409231580045[9] = 0;
   out_1331272409231580045[10] = 0;
   out_1331272409231580045[11] = 0;
   out_1331272409231580045[12] = 0;
   out_1331272409231580045[13] = 0;
   out_1331272409231580045[14] = 1;
   out_1331272409231580045[15] = 0;
   out_1331272409231580045[16] = 0;
   out_1331272409231580045[17] = 0;
}
void h_30(double *state, double *unused, double *out_4319993932698147994) {
   out_4319993932698147994[0] = state[4];
}
void H_30(double *state, double *unused, double *out_7271996987397597080) {
   out_7271996987397597080[0] = 0;
   out_7271996987397597080[1] = 0;
   out_7271996987397597080[2] = 0;
   out_7271996987397597080[3] = 0;
   out_7271996987397597080[4] = 1;
   out_7271996987397597080[5] = 0;
   out_7271996987397597080[6] = 0;
   out_7271996987397597080[7] = 0;
   out_7271996987397597080[8] = 0;
}
void h_26(double *state, double *unused, double *out_7644508679924030782) {
   out_7644508679924030782[0] = state[7];
}
void H_26(double *state, double *unused, double *out_7562582714581138382) {
   out_7562582714581138382[0] = 0;
   out_7562582714581138382[1] = 0;
   out_7562582714581138382[2] = 0;
   out_7562582714581138382[3] = 0;
   out_7562582714581138382[4] = 0;
   out_7562582714581138382[5] = 0;
   out_7562582714581138382[6] = 0;
   out_7562582714581138382[7] = 1;
   out_7562582714581138382[8] = 0;
}
void h_27(double *state, double *unused, double *out_3874213499084040753) {
   out_3874213499084040753[0] = state[3];
}
void H_27(double *state, double *unused, double *out_8999983774511529625) {
   out_8999983774511529625[0] = 0;
   out_8999983774511529625[1] = 0;
   out_8999983774511529625[2] = 0;
   out_8999983774511529625[3] = 1;
   out_8999983774511529625[4] = 0;
   out_8999983774511529625[5] = 0;
   out_8999983774511529625[6] = 0;
   out_8999983774511529625[7] = 0;
   out_8999983774511529625[8] = 0;
}
void h_29(double *state, double *unused, double *out_5263503785599810429) {
   out_5263503785599810429[0] = state[1];
}
void H_29(double *state, double *unused, double *out_7286621047641978592) {
   out_7286621047641978592[0] = 0;
   out_7286621047641978592[1] = 1;
   out_7286621047641978592[2] = 0;
   out_7286621047641978592[3] = 0;
   out_7286621047641978592[4] = 0;
   out_7286621047641978592[5] = 0;
   out_7286621047641978592[6] = 0;
   out_7286621047641978592[7] = 0;
   out_7286621047641978592[8] = 0;
}
void h_28(double *state, double *unused, double *out_407350342159619755) {
   out_407350342159619755[0] = state[0];
}
void H_28(double *state, double *unused, double *out_9196492754502246773) {
   out_9196492754502246773[0] = 1;
   out_9196492754502246773[1] = 0;
   out_9196492754502246773[2] = 0;
   out_9196492754502246773[3] = 0;
   out_9196492754502246773[4] = 0;
   out_9196492754502246773[5] = 0;
   out_9196492754502246773[6] = 0;
   out_9196492754502246773[7] = 0;
   out_9196492754502246773[8] = 0;
}
void h_31(double *state, double *unused, double *out_2647431620660638604) {
   out_2647431620660638604[0] = state[8];
}
void H_31(double *state, double *unused, double *out_7112012078377396582) {
   out_7112012078377396582[0] = 0;
   out_7112012078377396582[1] = 0;
   out_7112012078377396582[2] = 0;
   out_7112012078377396582[3] = 0;
   out_7112012078377396582[4] = 0;
   out_7112012078377396582[5] = 0;
   out_7112012078377396582[6] = 0;
   out_7112012078377396582[7] = 0;
   out_7112012078377396582[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_1299726308890531037) {
  err_fun(nom_x, delta_x, out_1299726308890531037);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2964139867777846776) {
  inv_err_fun(nom_x, true_x, out_2964139867777846776);
}
void car_H_mod_fun(double *state, double *out_8544953491798085886) {
  H_mod_fun(state, out_8544953491798085886);
}
void car_f_fun(double *state, double dt, double *out_5256147688816920328) {
  f_fun(state,  dt, out_5256147688816920328);
}
void car_F_fun(double *state, double dt, double *out_6834940091299106646) {
  F_fun(state,  dt, out_6834940091299106646);
}
void car_h_25(double *state, double *unused, double *out_6166873754981284210) {
  h_25(state, unused, out_6166873754981284210);
}
void car_H_25(double *state, double *unused, double *out_7142658040254357010) {
  H_25(state, unused, out_7142658040254357010);
}
void car_h_24(double *state, double *unused, double *out_2192149979727043903) {
  h_24(state, unused, out_2192149979727043903);
}
void car_H_24(double *state, double *unused, double *out_1331272409231580045) {
  H_24(state, unused, out_1331272409231580045);
}
void car_h_30(double *state, double *unused, double *out_4319993932698147994) {
  h_30(state, unused, out_4319993932698147994);
}
void car_H_30(double *state, double *unused, double *out_7271996987397597080) {
  H_30(state, unused, out_7271996987397597080);
}
void car_h_26(double *state, double *unused, double *out_7644508679924030782) {
  h_26(state, unused, out_7644508679924030782);
}
void car_H_26(double *state, double *unused, double *out_7562582714581138382) {
  H_26(state, unused, out_7562582714581138382);
}
void car_h_27(double *state, double *unused, double *out_3874213499084040753) {
  h_27(state, unused, out_3874213499084040753);
}
void car_H_27(double *state, double *unused, double *out_8999983774511529625) {
  H_27(state, unused, out_8999983774511529625);
}
void car_h_29(double *state, double *unused, double *out_5263503785599810429) {
  h_29(state, unused, out_5263503785599810429);
}
void car_H_29(double *state, double *unused, double *out_7286621047641978592) {
  H_29(state, unused, out_7286621047641978592);
}
void car_h_28(double *state, double *unused, double *out_407350342159619755) {
  h_28(state, unused, out_407350342159619755);
}
void car_H_28(double *state, double *unused, double *out_9196492754502246773) {
  H_28(state, unused, out_9196492754502246773);
}
void car_h_31(double *state, double *unused, double *out_2647431620660638604) {
  h_31(state, unused, out_2647431620660638604);
}
void car_H_31(double *state, double *unused, double *out_7112012078377396582) {
  H_31(state, unused, out_7112012078377396582);
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

ekf_init(car);
