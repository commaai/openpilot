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
void err_fun(double *nom_x, double *delta_x, double *out_8735429740637832661) {
   out_8735429740637832661[0] = delta_x[0] + nom_x[0];
   out_8735429740637832661[1] = delta_x[1] + nom_x[1];
   out_8735429740637832661[2] = delta_x[2] + nom_x[2];
   out_8735429740637832661[3] = delta_x[3] + nom_x[3];
   out_8735429740637832661[4] = delta_x[4] + nom_x[4];
   out_8735429740637832661[5] = delta_x[5] + nom_x[5];
   out_8735429740637832661[6] = delta_x[6] + nom_x[6];
   out_8735429740637832661[7] = delta_x[7] + nom_x[7];
   out_8735429740637832661[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7497417333367825202) {
   out_7497417333367825202[0] = -nom_x[0] + true_x[0];
   out_7497417333367825202[1] = -nom_x[1] + true_x[1];
   out_7497417333367825202[2] = -nom_x[2] + true_x[2];
   out_7497417333367825202[3] = -nom_x[3] + true_x[3];
   out_7497417333367825202[4] = -nom_x[4] + true_x[4];
   out_7497417333367825202[5] = -nom_x[5] + true_x[5];
   out_7497417333367825202[6] = -nom_x[6] + true_x[6];
   out_7497417333367825202[7] = -nom_x[7] + true_x[7];
   out_7497417333367825202[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_3386214859166967837) {
   out_3386214859166967837[0] = 1.0;
   out_3386214859166967837[1] = 0.0;
   out_3386214859166967837[2] = 0.0;
   out_3386214859166967837[3] = 0.0;
   out_3386214859166967837[4] = 0.0;
   out_3386214859166967837[5] = 0.0;
   out_3386214859166967837[6] = 0.0;
   out_3386214859166967837[7] = 0.0;
   out_3386214859166967837[8] = 0.0;
   out_3386214859166967837[9] = 0.0;
   out_3386214859166967837[10] = 1.0;
   out_3386214859166967837[11] = 0.0;
   out_3386214859166967837[12] = 0.0;
   out_3386214859166967837[13] = 0.0;
   out_3386214859166967837[14] = 0.0;
   out_3386214859166967837[15] = 0.0;
   out_3386214859166967837[16] = 0.0;
   out_3386214859166967837[17] = 0.0;
   out_3386214859166967837[18] = 0.0;
   out_3386214859166967837[19] = 0.0;
   out_3386214859166967837[20] = 1.0;
   out_3386214859166967837[21] = 0.0;
   out_3386214859166967837[22] = 0.0;
   out_3386214859166967837[23] = 0.0;
   out_3386214859166967837[24] = 0.0;
   out_3386214859166967837[25] = 0.0;
   out_3386214859166967837[26] = 0.0;
   out_3386214859166967837[27] = 0.0;
   out_3386214859166967837[28] = 0.0;
   out_3386214859166967837[29] = 0.0;
   out_3386214859166967837[30] = 1.0;
   out_3386214859166967837[31] = 0.0;
   out_3386214859166967837[32] = 0.0;
   out_3386214859166967837[33] = 0.0;
   out_3386214859166967837[34] = 0.0;
   out_3386214859166967837[35] = 0.0;
   out_3386214859166967837[36] = 0.0;
   out_3386214859166967837[37] = 0.0;
   out_3386214859166967837[38] = 0.0;
   out_3386214859166967837[39] = 0.0;
   out_3386214859166967837[40] = 1.0;
   out_3386214859166967837[41] = 0.0;
   out_3386214859166967837[42] = 0.0;
   out_3386214859166967837[43] = 0.0;
   out_3386214859166967837[44] = 0.0;
   out_3386214859166967837[45] = 0.0;
   out_3386214859166967837[46] = 0.0;
   out_3386214859166967837[47] = 0.0;
   out_3386214859166967837[48] = 0.0;
   out_3386214859166967837[49] = 0.0;
   out_3386214859166967837[50] = 1.0;
   out_3386214859166967837[51] = 0.0;
   out_3386214859166967837[52] = 0.0;
   out_3386214859166967837[53] = 0.0;
   out_3386214859166967837[54] = 0.0;
   out_3386214859166967837[55] = 0.0;
   out_3386214859166967837[56] = 0.0;
   out_3386214859166967837[57] = 0.0;
   out_3386214859166967837[58] = 0.0;
   out_3386214859166967837[59] = 0.0;
   out_3386214859166967837[60] = 1.0;
   out_3386214859166967837[61] = 0.0;
   out_3386214859166967837[62] = 0.0;
   out_3386214859166967837[63] = 0.0;
   out_3386214859166967837[64] = 0.0;
   out_3386214859166967837[65] = 0.0;
   out_3386214859166967837[66] = 0.0;
   out_3386214859166967837[67] = 0.0;
   out_3386214859166967837[68] = 0.0;
   out_3386214859166967837[69] = 0.0;
   out_3386214859166967837[70] = 1.0;
   out_3386214859166967837[71] = 0.0;
   out_3386214859166967837[72] = 0.0;
   out_3386214859166967837[73] = 0.0;
   out_3386214859166967837[74] = 0.0;
   out_3386214859166967837[75] = 0.0;
   out_3386214859166967837[76] = 0.0;
   out_3386214859166967837[77] = 0.0;
   out_3386214859166967837[78] = 0.0;
   out_3386214859166967837[79] = 0.0;
   out_3386214859166967837[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_4508993452591091797) {
   out_4508993452591091797[0] = state[0];
   out_4508993452591091797[1] = state[1];
   out_4508993452591091797[2] = state[2];
   out_4508993452591091797[3] = state[3];
   out_4508993452591091797[4] = state[4];
   out_4508993452591091797[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_4508993452591091797[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_4508993452591091797[7] = state[7];
   out_4508993452591091797[8] = state[8];
}
void F_fun(double *state, double dt, double *out_1336565245694139133) {
   out_1336565245694139133[0] = 1;
   out_1336565245694139133[1] = 0;
   out_1336565245694139133[2] = 0;
   out_1336565245694139133[3] = 0;
   out_1336565245694139133[4] = 0;
   out_1336565245694139133[5] = 0;
   out_1336565245694139133[6] = 0;
   out_1336565245694139133[7] = 0;
   out_1336565245694139133[8] = 0;
   out_1336565245694139133[9] = 0;
   out_1336565245694139133[10] = 1;
   out_1336565245694139133[11] = 0;
   out_1336565245694139133[12] = 0;
   out_1336565245694139133[13] = 0;
   out_1336565245694139133[14] = 0;
   out_1336565245694139133[15] = 0;
   out_1336565245694139133[16] = 0;
   out_1336565245694139133[17] = 0;
   out_1336565245694139133[18] = 0;
   out_1336565245694139133[19] = 0;
   out_1336565245694139133[20] = 1;
   out_1336565245694139133[21] = 0;
   out_1336565245694139133[22] = 0;
   out_1336565245694139133[23] = 0;
   out_1336565245694139133[24] = 0;
   out_1336565245694139133[25] = 0;
   out_1336565245694139133[26] = 0;
   out_1336565245694139133[27] = 0;
   out_1336565245694139133[28] = 0;
   out_1336565245694139133[29] = 0;
   out_1336565245694139133[30] = 1;
   out_1336565245694139133[31] = 0;
   out_1336565245694139133[32] = 0;
   out_1336565245694139133[33] = 0;
   out_1336565245694139133[34] = 0;
   out_1336565245694139133[35] = 0;
   out_1336565245694139133[36] = 0;
   out_1336565245694139133[37] = 0;
   out_1336565245694139133[38] = 0;
   out_1336565245694139133[39] = 0;
   out_1336565245694139133[40] = 1;
   out_1336565245694139133[41] = 0;
   out_1336565245694139133[42] = 0;
   out_1336565245694139133[43] = 0;
   out_1336565245694139133[44] = 0;
   out_1336565245694139133[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_1336565245694139133[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_1336565245694139133[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_1336565245694139133[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_1336565245694139133[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_1336565245694139133[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_1336565245694139133[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_1336565245694139133[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_1336565245694139133[53] = -9.8100000000000005*dt;
   out_1336565245694139133[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_1336565245694139133[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_1336565245694139133[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1336565245694139133[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1336565245694139133[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_1336565245694139133[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_1336565245694139133[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_1336565245694139133[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1336565245694139133[62] = 0;
   out_1336565245694139133[63] = 0;
   out_1336565245694139133[64] = 0;
   out_1336565245694139133[65] = 0;
   out_1336565245694139133[66] = 0;
   out_1336565245694139133[67] = 0;
   out_1336565245694139133[68] = 0;
   out_1336565245694139133[69] = 0;
   out_1336565245694139133[70] = 1;
   out_1336565245694139133[71] = 0;
   out_1336565245694139133[72] = 0;
   out_1336565245694139133[73] = 0;
   out_1336565245694139133[74] = 0;
   out_1336565245694139133[75] = 0;
   out_1336565245694139133[76] = 0;
   out_1336565245694139133[77] = 0;
   out_1336565245694139133[78] = 0;
   out_1336565245694139133[79] = 0;
   out_1336565245694139133[80] = 1;
}
void h_25(double *state, double *unused, double *out_2296868611573953905) {
   out_2296868611573953905[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1408615983760901272) {
   out_1408615983760901272[0] = 0;
   out_1408615983760901272[1] = 0;
   out_1408615983760901272[2] = 0;
   out_1408615983760901272[3] = 0;
   out_1408615983760901272[4] = 0;
   out_1408615983760901272[5] = 0;
   out_1408615983760901272[6] = 1;
   out_1408615983760901272[7] = 0;
   out_1408615983760901272[8] = 0;
}
void h_24(double *state, double *unused, double *out_3819261768936520190) {
   out_3819261768936520190[0] = state[4];
   out_3819261768936520190[1] = state[5];
}
void H_24(double *state, double *unused, double *out_3581265582766400838) {
   out_3581265582766400838[0] = 0;
   out_3581265582766400838[1] = 0;
   out_3581265582766400838[2] = 0;
   out_3581265582766400838[3] = 0;
   out_3581265582766400838[4] = 1;
   out_3581265582766400838[5] = 0;
   out_3581265582766400838[6] = 0;
   out_3581265582766400838[7] = 0;
   out_3581265582766400838[8] = 0;
   out_3581265582766400838[9] = 0;
   out_3581265582766400838[10] = 0;
   out_3581265582766400838[11] = 0;
   out_3581265582766400838[12] = 0;
   out_3581265582766400838[13] = 0;
   out_3581265582766400838[14] = 1;
   out_3581265582766400838[15] = 0;
   out_3581265582766400838[16] = 0;
   out_3581265582766400838[17] = 0;
}
void h_30(double *state, double *unused, double *out_9067703837924304841) {
   out_9067703837924304841[0] = state[4];
}
void H_30(double *state, double *unused, double *out_1109716974746347355) {
   out_1109716974746347355[0] = 0;
   out_1109716974746347355[1] = 0;
   out_1109716974746347355[2] = 0;
   out_1109716974746347355[3] = 0;
   out_1109716974746347355[4] = 1;
   out_1109716974746347355[5] = 0;
   out_1109716974746347355[6] = 0;
   out_1109716974746347355[7] = 0;
   out_1109716974746347355[8] = 0;
}
void h_26(double *state, double *unused, double *out_4377366968124401130) {
   out_4377366968124401130[0] = state[7];
}
void H_26(double *state, double *unused, double *out_1895909985999899329) {
   out_1895909985999899329[0] = 0;
   out_1895909985999899329[1] = 0;
   out_1895909985999899329[2] = 0;
   out_1895909985999899329[3] = 0;
   out_1895909985999899329[4] = 0;
   out_1895909985999899329[5] = 0;
   out_1895909985999899329[6] = 0;
   out_1895909985999899329[7] = 1;
   out_1895909985999899329[8] = 0;
}
void h_27(double *state, double *unused, double *out_2704804656086891740) {
   out_2704804656086891740[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1065046337054077556) {
   out_1065046337054077556[0] = 0;
   out_1065046337054077556[1] = 0;
   out_1065046337054077556[2] = 0;
   out_1065046337054077556[3] = 1;
   out_1065046337054077556[4] = 0;
   out_1065046337054077556[5] = 0;
   out_1065046337054077556[6] = 0;
   out_1065046337054077556[7] = 0;
   out_1065046337054077556[8] = 0;
}
void h_29(double *state, double *unused, double *out_6566488132515393392) {
   out_6566488132515393392[0] = state[1];
}
void H_29(double *state, double *unused, double *out_1619948319060739539) {
   out_1619948319060739539[0] = 0;
   out_1619948319060739539[1] = 1;
   out_1619948319060739539[2] = 0;
   out_1619948319060739539[3] = 0;
   out_1619948319060739539[4] = 0;
   out_1619948319060739539[5] = 0;
   out_1619948319060739539[6] = 0;
   out_1619948319060739539[7] = 0;
   out_1619948319060739539[8] = 0;
}
void h_28(double *state, double *unused, double *out_7771222174805234463) {
   out_7771222174805234463[0] = state[0];
}
void H_28(double *state, double *unused, double *out_3462450698008791035) {
   out_3462450698008791035[0] = 1;
   out_3462450698008791035[1] = 0;
   out_3462450698008791035[2] = 0;
   out_3462450698008791035[3] = 0;
   out_3462450698008791035[4] = 0;
   out_3462450698008791035[5] = 0;
   out_3462450698008791035[6] = 0;
   out_3462450698008791035[7] = 0;
   out_3462450698008791035[8] = 0;
}
void h_31(double *state, double *unused, double *out_5244594601929438866) {
   out_5244594601929438866[0] = state[8];
}
void H_31(double *state, double *unused, double *out_1269701883766547853) {
   out_1269701883766547853[0] = 0;
   out_1269701883766547853[1] = 0;
   out_1269701883766547853[2] = 0;
   out_1269701883766547853[3] = 0;
   out_1269701883766547853[4] = 0;
   out_1269701883766547853[5] = 0;
   out_1269701883766547853[6] = 0;
   out_1269701883766547853[7] = 0;
   out_1269701883766547853[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_8735429740637832661) {
  err_fun(nom_x, delta_x, out_8735429740637832661);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7497417333367825202) {
  inv_err_fun(nom_x, true_x, out_7497417333367825202);
}
void car_H_mod_fun(double *state, double *out_3386214859166967837) {
  H_mod_fun(state, out_3386214859166967837);
}
void car_f_fun(double *state, double dt, double *out_4508993452591091797) {
  f_fun(state,  dt, out_4508993452591091797);
}
void car_F_fun(double *state, double dt, double *out_1336565245694139133) {
  F_fun(state,  dt, out_1336565245694139133);
}
void car_h_25(double *state, double *unused, double *out_2296868611573953905) {
  h_25(state, unused, out_2296868611573953905);
}
void car_H_25(double *state, double *unused, double *out_1408615983760901272) {
  H_25(state, unused, out_1408615983760901272);
}
void car_h_24(double *state, double *unused, double *out_3819261768936520190) {
  h_24(state, unused, out_3819261768936520190);
}
void car_H_24(double *state, double *unused, double *out_3581265582766400838) {
  H_24(state, unused, out_3581265582766400838);
}
void car_h_30(double *state, double *unused, double *out_9067703837924304841) {
  h_30(state, unused, out_9067703837924304841);
}
void car_H_30(double *state, double *unused, double *out_1109716974746347355) {
  H_30(state, unused, out_1109716974746347355);
}
void car_h_26(double *state, double *unused, double *out_4377366968124401130) {
  h_26(state, unused, out_4377366968124401130);
}
void car_H_26(double *state, double *unused, double *out_1895909985999899329) {
  H_26(state, unused, out_1895909985999899329);
}
void car_h_27(double *state, double *unused, double *out_2704804656086891740) {
  h_27(state, unused, out_2704804656086891740);
}
void car_H_27(double *state, double *unused, double *out_1065046337054077556) {
  H_27(state, unused, out_1065046337054077556);
}
void car_h_29(double *state, double *unused, double *out_6566488132515393392) {
  h_29(state, unused, out_6566488132515393392);
}
void car_H_29(double *state, double *unused, double *out_1619948319060739539) {
  H_29(state, unused, out_1619948319060739539);
}
void car_h_28(double *state, double *unused, double *out_7771222174805234463) {
  h_28(state, unused, out_7771222174805234463);
}
void car_H_28(double *state, double *unused, double *out_3462450698008791035) {
  H_28(state, unused, out_3462450698008791035);
}
void car_h_31(double *state, double *unused, double *out_5244594601929438866) {
  h_31(state, unused, out_5244594601929438866);
}
void car_H_31(double *state, double *unused, double *out_1269701883766547853) {
  H_31(state, unused, out_1269701883766547853);
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
