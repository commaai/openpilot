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
void err_fun(double *nom_x, double *delta_x, double *out_7126119498051021768) {
   out_7126119498051021768[0] = delta_x[0] + nom_x[0];
   out_7126119498051021768[1] = delta_x[1] + nom_x[1];
   out_7126119498051021768[2] = delta_x[2] + nom_x[2];
   out_7126119498051021768[3] = delta_x[3] + nom_x[3];
   out_7126119498051021768[4] = delta_x[4] + nom_x[4];
   out_7126119498051021768[5] = delta_x[5] + nom_x[5];
   out_7126119498051021768[6] = delta_x[6] + nom_x[6];
   out_7126119498051021768[7] = delta_x[7] + nom_x[7];
   out_7126119498051021768[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2773982190367778391) {
   out_2773982190367778391[0] = -nom_x[0] + true_x[0];
   out_2773982190367778391[1] = -nom_x[1] + true_x[1];
   out_2773982190367778391[2] = -nom_x[2] + true_x[2];
   out_2773982190367778391[3] = -nom_x[3] + true_x[3];
   out_2773982190367778391[4] = -nom_x[4] + true_x[4];
   out_2773982190367778391[5] = -nom_x[5] + true_x[5];
   out_2773982190367778391[6] = -nom_x[6] + true_x[6];
   out_2773982190367778391[7] = -nom_x[7] + true_x[7];
   out_2773982190367778391[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_8542714350044848421) {
   out_8542714350044848421[0] = 1.0;
   out_8542714350044848421[1] = 0.0;
   out_8542714350044848421[2] = 0.0;
   out_8542714350044848421[3] = 0.0;
   out_8542714350044848421[4] = 0.0;
   out_8542714350044848421[5] = 0.0;
   out_8542714350044848421[6] = 0.0;
   out_8542714350044848421[7] = 0.0;
   out_8542714350044848421[8] = 0.0;
   out_8542714350044848421[9] = 0.0;
   out_8542714350044848421[10] = 1.0;
   out_8542714350044848421[11] = 0.0;
   out_8542714350044848421[12] = 0.0;
   out_8542714350044848421[13] = 0.0;
   out_8542714350044848421[14] = 0.0;
   out_8542714350044848421[15] = 0.0;
   out_8542714350044848421[16] = 0.0;
   out_8542714350044848421[17] = 0.0;
   out_8542714350044848421[18] = 0.0;
   out_8542714350044848421[19] = 0.0;
   out_8542714350044848421[20] = 1.0;
   out_8542714350044848421[21] = 0.0;
   out_8542714350044848421[22] = 0.0;
   out_8542714350044848421[23] = 0.0;
   out_8542714350044848421[24] = 0.0;
   out_8542714350044848421[25] = 0.0;
   out_8542714350044848421[26] = 0.0;
   out_8542714350044848421[27] = 0.0;
   out_8542714350044848421[28] = 0.0;
   out_8542714350044848421[29] = 0.0;
   out_8542714350044848421[30] = 1.0;
   out_8542714350044848421[31] = 0.0;
   out_8542714350044848421[32] = 0.0;
   out_8542714350044848421[33] = 0.0;
   out_8542714350044848421[34] = 0.0;
   out_8542714350044848421[35] = 0.0;
   out_8542714350044848421[36] = 0.0;
   out_8542714350044848421[37] = 0.0;
   out_8542714350044848421[38] = 0.0;
   out_8542714350044848421[39] = 0.0;
   out_8542714350044848421[40] = 1.0;
   out_8542714350044848421[41] = 0.0;
   out_8542714350044848421[42] = 0.0;
   out_8542714350044848421[43] = 0.0;
   out_8542714350044848421[44] = 0.0;
   out_8542714350044848421[45] = 0.0;
   out_8542714350044848421[46] = 0.0;
   out_8542714350044848421[47] = 0.0;
   out_8542714350044848421[48] = 0.0;
   out_8542714350044848421[49] = 0.0;
   out_8542714350044848421[50] = 1.0;
   out_8542714350044848421[51] = 0.0;
   out_8542714350044848421[52] = 0.0;
   out_8542714350044848421[53] = 0.0;
   out_8542714350044848421[54] = 0.0;
   out_8542714350044848421[55] = 0.0;
   out_8542714350044848421[56] = 0.0;
   out_8542714350044848421[57] = 0.0;
   out_8542714350044848421[58] = 0.0;
   out_8542714350044848421[59] = 0.0;
   out_8542714350044848421[60] = 1.0;
   out_8542714350044848421[61] = 0.0;
   out_8542714350044848421[62] = 0.0;
   out_8542714350044848421[63] = 0.0;
   out_8542714350044848421[64] = 0.0;
   out_8542714350044848421[65] = 0.0;
   out_8542714350044848421[66] = 0.0;
   out_8542714350044848421[67] = 0.0;
   out_8542714350044848421[68] = 0.0;
   out_8542714350044848421[69] = 0.0;
   out_8542714350044848421[70] = 1.0;
   out_8542714350044848421[71] = 0.0;
   out_8542714350044848421[72] = 0.0;
   out_8542714350044848421[73] = 0.0;
   out_8542714350044848421[74] = 0.0;
   out_8542714350044848421[75] = 0.0;
   out_8542714350044848421[76] = 0.0;
   out_8542714350044848421[77] = 0.0;
   out_8542714350044848421[78] = 0.0;
   out_8542714350044848421[79] = 0.0;
   out_8542714350044848421[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7713290282379442360) {
   out_7713290282379442360[0] = state[0];
   out_7713290282379442360[1] = state[1];
   out_7713290282379442360[2] = state[2];
   out_7713290282379442360[3] = state[3];
   out_7713290282379442360[4] = state[4];
   out_7713290282379442360[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_7713290282379442360[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7713290282379442360[7] = state[7];
   out_7713290282379442360[8] = state[8];
}
void F_fun(double *state, double dt, double *out_2189258202331065399) {
   out_2189258202331065399[0] = 1;
   out_2189258202331065399[1] = 0;
   out_2189258202331065399[2] = 0;
   out_2189258202331065399[3] = 0;
   out_2189258202331065399[4] = 0;
   out_2189258202331065399[5] = 0;
   out_2189258202331065399[6] = 0;
   out_2189258202331065399[7] = 0;
   out_2189258202331065399[8] = 0;
   out_2189258202331065399[9] = 0;
   out_2189258202331065399[10] = 1;
   out_2189258202331065399[11] = 0;
   out_2189258202331065399[12] = 0;
   out_2189258202331065399[13] = 0;
   out_2189258202331065399[14] = 0;
   out_2189258202331065399[15] = 0;
   out_2189258202331065399[16] = 0;
   out_2189258202331065399[17] = 0;
   out_2189258202331065399[18] = 0;
   out_2189258202331065399[19] = 0;
   out_2189258202331065399[20] = 1;
   out_2189258202331065399[21] = 0;
   out_2189258202331065399[22] = 0;
   out_2189258202331065399[23] = 0;
   out_2189258202331065399[24] = 0;
   out_2189258202331065399[25] = 0;
   out_2189258202331065399[26] = 0;
   out_2189258202331065399[27] = 0;
   out_2189258202331065399[28] = 0;
   out_2189258202331065399[29] = 0;
   out_2189258202331065399[30] = 1;
   out_2189258202331065399[31] = 0;
   out_2189258202331065399[32] = 0;
   out_2189258202331065399[33] = 0;
   out_2189258202331065399[34] = 0;
   out_2189258202331065399[35] = 0;
   out_2189258202331065399[36] = 0;
   out_2189258202331065399[37] = 0;
   out_2189258202331065399[38] = 0;
   out_2189258202331065399[39] = 0;
   out_2189258202331065399[40] = 1;
   out_2189258202331065399[41] = 0;
   out_2189258202331065399[42] = 0;
   out_2189258202331065399[43] = 0;
   out_2189258202331065399[44] = 0;
   out_2189258202331065399[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_2189258202331065399[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_2189258202331065399[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_2189258202331065399[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_2189258202331065399[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_2189258202331065399[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_2189258202331065399[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_2189258202331065399[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_2189258202331065399[53] = -9.8100000000000005*dt;
   out_2189258202331065399[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_2189258202331065399[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_2189258202331065399[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2189258202331065399[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2189258202331065399[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_2189258202331065399[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_2189258202331065399[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_2189258202331065399[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2189258202331065399[62] = 0;
   out_2189258202331065399[63] = 0;
   out_2189258202331065399[64] = 0;
   out_2189258202331065399[65] = 0;
   out_2189258202331065399[66] = 0;
   out_2189258202331065399[67] = 0;
   out_2189258202331065399[68] = 0;
   out_2189258202331065399[69] = 0;
   out_2189258202331065399[70] = 1;
   out_2189258202331065399[71] = 0;
   out_2189258202331065399[72] = 0;
   out_2189258202331065399[73] = 0;
   out_2189258202331065399[74] = 0;
   out_2189258202331065399[75] = 0;
   out_2189258202331065399[76] = 0;
   out_2189258202331065399[77] = 0;
   out_2189258202331065399[78] = 0;
   out_2189258202331065399[79] = 0;
   out_2189258202331065399[80] = 1;
}
void h_25(double *state, double *unused, double *out_2256658881339624696) {
   out_2256658881339624696[0] = state[6];
}
void H_25(double *state, double *unused, double *out_5019733700055356744) {
   out_5019733700055356744[0] = 0;
   out_5019733700055356744[1] = 0;
   out_5019733700055356744[2] = 0;
   out_5019733700055356744[3] = 0;
   out_5019733700055356744[4] = 0;
   out_5019733700055356744[5] = 0;
   out_5019733700055356744[6] = 1;
   out_5019733700055356744[7] = 0;
   out_5019733700055356744[8] = 0;
}
void h_24(double *state, double *unused, double *out_1108309987830345920) {
   out_1108309987830345920[0] = state[4];
   out_1108309987830345920[1] = state[5];
}
void H_24(double *state, double *unused, double *out_1461808101253033608) {
   out_1461808101253033608[0] = 0;
   out_1461808101253033608[1] = 0;
   out_1461808101253033608[2] = 0;
   out_1461808101253033608[3] = 0;
   out_1461808101253033608[4] = 1;
   out_1461808101253033608[5] = 0;
   out_1461808101253033608[6] = 0;
   out_1461808101253033608[7] = 0;
   out_1461808101253033608[8] = 0;
   out_1461808101253033608[9] = 0;
   out_1461808101253033608[10] = 0;
   out_1461808101253033608[11] = 0;
   out_1461808101253033608[12] = 0;
   out_1461808101253033608[13] = 0;
   out_1461808101253033608[14] = 1;
   out_1461808101253033608[15] = 0;
   out_1461808101253033608[16] = 0;
   out_1461808101253033608[17] = 0;
}
void h_30(double *state, double *unused, double *out_1981464819055118807) {
   out_1981464819055118807[0] = state[4];
}
void H_30(double *state, double *unused, double *out_1896956641436260011) {
   out_1896956641436260011[0] = 0;
   out_1896956641436260011[1] = 0;
   out_1896956641436260011[2] = 0;
   out_1896956641436260011[3] = 0;
   out_1896956641436260011[4] = 1;
   out_1896956641436260011[5] = 0;
   out_1896956641436260011[6] = 0;
   out_1896956641436260011[7] = 0;
   out_1896956641436260011[8] = 0;
}
void h_26(double *state, double *unused, double *out_3461392923629465767) {
   out_3461392923629465767[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8761237018929412968) {
   out_8761237018929412968[0] = 0;
   out_8761237018929412968[1] = 0;
   out_8761237018929412968[2] = 0;
   out_8761237018929412968[3] = 0;
   out_8761237018929412968[4] = 0;
   out_8761237018929412968[5] = 0;
   out_8761237018929412968[6] = 0;
   out_8761237018929412968[7] = 1;
   out_8761237018929412968[8] = 0;
}
void h_27(double *state, double *unused, double *out_2969248334495700251) {
   out_2969248334495700251[0] = state[3];
}
void H_27(double *state, double *unused, double *out_277806670364164900) {
   out_277806670364164900[0] = 0;
   out_277806670364164900[1] = 0;
   out_277806670364164900[2] = 0;
   out_277806670364164900[3] = 1;
   out_277806670364164900[4] = 0;
   out_277806670364164900[5] = 0;
   out_277806670364164900[6] = 0;
   out_277806670364164900[7] = 0;
   out_277806670364164900[8] = 0;
}
void h_29(double *state, double *unused, double *out_3020208448605726838) {
   out_3020208448605726838[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2407187985750652195) {
   out_2407187985750652195[0] = 0;
   out_2407187985750652195[1] = 1;
   out_2407187985750652195[2] = 0;
   out_2407187985750652195[3] = 0;
   out_2407187985750652195[4] = 0;
   out_2407187985750652195[5] = 0;
   out_2407187985750652195[6] = 0;
   out_2407187985750652195[7] = 0;
   out_2407187985750652195[8] = 0;
}
void h_28(double *state, double *unused, double *out_364843013687002600) {
   out_364843013687002600[0] = state[0];
}
void H_28(double *state, double *unused, double *out_2675211031318878379) {
   out_2675211031318878379[0] = 1;
   out_2675211031318878379[1] = 0;
   out_2675211031318878379[2] = 0;
   out_2675211031318878379[3] = 0;
   out_2675211031318878379[4] = 0;
   out_2675211031318878379[5] = 0;
   out_2675211031318878379[6] = 0;
   out_2675211031318878379[7] = 0;
   out_2675211031318878379[8] = 0;
}
void h_31(double *state, double *unused, double *out_5894108409593647046) {
   out_5894108409593647046[0] = state[8];
}
void H_31(double *state, double *unused, double *out_4989087738178396316) {
   out_4989087738178396316[0] = 0;
   out_4989087738178396316[1] = 0;
   out_4989087738178396316[2] = 0;
   out_4989087738178396316[3] = 0;
   out_4989087738178396316[4] = 0;
   out_4989087738178396316[5] = 0;
   out_4989087738178396316[6] = 0;
   out_4989087738178396316[7] = 0;
   out_4989087738178396316[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_7126119498051021768) {
  err_fun(nom_x, delta_x, out_7126119498051021768);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2773982190367778391) {
  inv_err_fun(nom_x, true_x, out_2773982190367778391);
}
void car_H_mod_fun(double *state, double *out_8542714350044848421) {
  H_mod_fun(state, out_8542714350044848421);
}
void car_f_fun(double *state, double dt, double *out_7713290282379442360) {
  f_fun(state,  dt, out_7713290282379442360);
}
void car_F_fun(double *state, double dt, double *out_2189258202331065399) {
  F_fun(state,  dt, out_2189258202331065399);
}
void car_h_25(double *state, double *unused, double *out_2256658881339624696) {
  h_25(state, unused, out_2256658881339624696);
}
void car_H_25(double *state, double *unused, double *out_5019733700055356744) {
  H_25(state, unused, out_5019733700055356744);
}
void car_h_24(double *state, double *unused, double *out_1108309987830345920) {
  h_24(state, unused, out_1108309987830345920);
}
void car_H_24(double *state, double *unused, double *out_1461808101253033608) {
  H_24(state, unused, out_1461808101253033608);
}
void car_h_30(double *state, double *unused, double *out_1981464819055118807) {
  h_30(state, unused, out_1981464819055118807);
}
void car_H_30(double *state, double *unused, double *out_1896956641436260011) {
  H_30(state, unused, out_1896956641436260011);
}
void car_h_26(double *state, double *unused, double *out_3461392923629465767) {
  h_26(state, unused, out_3461392923629465767);
}
void car_H_26(double *state, double *unused, double *out_8761237018929412968) {
  H_26(state, unused, out_8761237018929412968);
}
void car_h_27(double *state, double *unused, double *out_2969248334495700251) {
  h_27(state, unused, out_2969248334495700251);
}
void car_H_27(double *state, double *unused, double *out_277806670364164900) {
  H_27(state, unused, out_277806670364164900);
}
void car_h_29(double *state, double *unused, double *out_3020208448605726838) {
  h_29(state, unused, out_3020208448605726838);
}
void car_H_29(double *state, double *unused, double *out_2407187985750652195) {
  H_29(state, unused, out_2407187985750652195);
}
void car_h_28(double *state, double *unused, double *out_364843013687002600) {
  h_28(state, unused, out_364843013687002600);
}
void car_H_28(double *state, double *unused, double *out_2675211031318878379) {
  H_28(state, unused, out_2675211031318878379);
}
void car_h_31(double *state, double *unused, double *out_5894108409593647046) {
  h_31(state, unused, out_5894108409593647046);
}
void car_H_31(double *state, double *unused, double *out_4989087738178396316) {
  H_31(state, unused, out_4989087738178396316);
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
