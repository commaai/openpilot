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
void err_fun(double *nom_x, double *delta_x, double *out_6823495906263670498) {
   out_6823495906263670498[0] = delta_x[0] + nom_x[0];
   out_6823495906263670498[1] = delta_x[1] + nom_x[1];
   out_6823495906263670498[2] = delta_x[2] + nom_x[2];
   out_6823495906263670498[3] = delta_x[3] + nom_x[3];
   out_6823495906263670498[4] = delta_x[4] + nom_x[4];
   out_6823495906263670498[5] = delta_x[5] + nom_x[5];
   out_6823495906263670498[6] = delta_x[6] + nom_x[6];
   out_6823495906263670498[7] = delta_x[7] + nom_x[7];
   out_6823495906263670498[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2832333554644134208) {
   out_2832333554644134208[0] = -nom_x[0] + true_x[0];
   out_2832333554644134208[1] = -nom_x[1] + true_x[1];
   out_2832333554644134208[2] = -nom_x[2] + true_x[2];
   out_2832333554644134208[3] = -nom_x[3] + true_x[3];
   out_2832333554644134208[4] = -nom_x[4] + true_x[4];
   out_2832333554644134208[5] = -nom_x[5] + true_x[5];
   out_2832333554644134208[6] = -nom_x[6] + true_x[6];
   out_2832333554644134208[7] = -nom_x[7] + true_x[7];
   out_2832333554644134208[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_8294511297811177194) {
   out_8294511297811177194[0] = 1.0;
   out_8294511297811177194[1] = 0.0;
   out_8294511297811177194[2] = 0.0;
   out_8294511297811177194[3] = 0.0;
   out_8294511297811177194[4] = 0.0;
   out_8294511297811177194[5] = 0.0;
   out_8294511297811177194[6] = 0.0;
   out_8294511297811177194[7] = 0.0;
   out_8294511297811177194[8] = 0.0;
   out_8294511297811177194[9] = 0.0;
   out_8294511297811177194[10] = 1.0;
   out_8294511297811177194[11] = 0.0;
   out_8294511297811177194[12] = 0.0;
   out_8294511297811177194[13] = 0.0;
   out_8294511297811177194[14] = 0.0;
   out_8294511297811177194[15] = 0.0;
   out_8294511297811177194[16] = 0.0;
   out_8294511297811177194[17] = 0.0;
   out_8294511297811177194[18] = 0.0;
   out_8294511297811177194[19] = 0.0;
   out_8294511297811177194[20] = 1.0;
   out_8294511297811177194[21] = 0.0;
   out_8294511297811177194[22] = 0.0;
   out_8294511297811177194[23] = 0.0;
   out_8294511297811177194[24] = 0.0;
   out_8294511297811177194[25] = 0.0;
   out_8294511297811177194[26] = 0.0;
   out_8294511297811177194[27] = 0.0;
   out_8294511297811177194[28] = 0.0;
   out_8294511297811177194[29] = 0.0;
   out_8294511297811177194[30] = 1.0;
   out_8294511297811177194[31] = 0.0;
   out_8294511297811177194[32] = 0.0;
   out_8294511297811177194[33] = 0.0;
   out_8294511297811177194[34] = 0.0;
   out_8294511297811177194[35] = 0.0;
   out_8294511297811177194[36] = 0.0;
   out_8294511297811177194[37] = 0.0;
   out_8294511297811177194[38] = 0.0;
   out_8294511297811177194[39] = 0.0;
   out_8294511297811177194[40] = 1.0;
   out_8294511297811177194[41] = 0.0;
   out_8294511297811177194[42] = 0.0;
   out_8294511297811177194[43] = 0.0;
   out_8294511297811177194[44] = 0.0;
   out_8294511297811177194[45] = 0.0;
   out_8294511297811177194[46] = 0.0;
   out_8294511297811177194[47] = 0.0;
   out_8294511297811177194[48] = 0.0;
   out_8294511297811177194[49] = 0.0;
   out_8294511297811177194[50] = 1.0;
   out_8294511297811177194[51] = 0.0;
   out_8294511297811177194[52] = 0.0;
   out_8294511297811177194[53] = 0.0;
   out_8294511297811177194[54] = 0.0;
   out_8294511297811177194[55] = 0.0;
   out_8294511297811177194[56] = 0.0;
   out_8294511297811177194[57] = 0.0;
   out_8294511297811177194[58] = 0.0;
   out_8294511297811177194[59] = 0.0;
   out_8294511297811177194[60] = 1.0;
   out_8294511297811177194[61] = 0.0;
   out_8294511297811177194[62] = 0.0;
   out_8294511297811177194[63] = 0.0;
   out_8294511297811177194[64] = 0.0;
   out_8294511297811177194[65] = 0.0;
   out_8294511297811177194[66] = 0.0;
   out_8294511297811177194[67] = 0.0;
   out_8294511297811177194[68] = 0.0;
   out_8294511297811177194[69] = 0.0;
   out_8294511297811177194[70] = 1.0;
   out_8294511297811177194[71] = 0.0;
   out_8294511297811177194[72] = 0.0;
   out_8294511297811177194[73] = 0.0;
   out_8294511297811177194[74] = 0.0;
   out_8294511297811177194[75] = 0.0;
   out_8294511297811177194[76] = 0.0;
   out_8294511297811177194[77] = 0.0;
   out_8294511297811177194[78] = 0.0;
   out_8294511297811177194[79] = 0.0;
   out_8294511297811177194[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_156104586604879915) {
   out_156104586604879915[0] = state[0];
   out_156104586604879915[1] = state[1];
   out_156104586604879915[2] = state[2];
   out_156104586604879915[3] = state[3];
   out_156104586604879915[4] = state[4];
   out_156104586604879915[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_156104586604879915[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_156104586604879915[7] = state[7];
   out_156104586604879915[8] = state[8];
}
void F_fun(double *state, double dt, double *out_221487960878628462) {
   out_221487960878628462[0] = 1;
   out_221487960878628462[1] = 0;
   out_221487960878628462[2] = 0;
   out_221487960878628462[3] = 0;
   out_221487960878628462[4] = 0;
   out_221487960878628462[5] = 0;
   out_221487960878628462[6] = 0;
   out_221487960878628462[7] = 0;
   out_221487960878628462[8] = 0;
   out_221487960878628462[9] = 0;
   out_221487960878628462[10] = 1;
   out_221487960878628462[11] = 0;
   out_221487960878628462[12] = 0;
   out_221487960878628462[13] = 0;
   out_221487960878628462[14] = 0;
   out_221487960878628462[15] = 0;
   out_221487960878628462[16] = 0;
   out_221487960878628462[17] = 0;
   out_221487960878628462[18] = 0;
   out_221487960878628462[19] = 0;
   out_221487960878628462[20] = 1;
   out_221487960878628462[21] = 0;
   out_221487960878628462[22] = 0;
   out_221487960878628462[23] = 0;
   out_221487960878628462[24] = 0;
   out_221487960878628462[25] = 0;
   out_221487960878628462[26] = 0;
   out_221487960878628462[27] = 0;
   out_221487960878628462[28] = 0;
   out_221487960878628462[29] = 0;
   out_221487960878628462[30] = 1;
   out_221487960878628462[31] = 0;
   out_221487960878628462[32] = 0;
   out_221487960878628462[33] = 0;
   out_221487960878628462[34] = 0;
   out_221487960878628462[35] = 0;
   out_221487960878628462[36] = 0;
   out_221487960878628462[37] = 0;
   out_221487960878628462[38] = 0;
   out_221487960878628462[39] = 0;
   out_221487960878628462[40] = 1;
   out_221487960878628462[41] = 0;
   out_221487960878628462[42] = 0;
   out_221487960878628462[43] = 0;
   out_221487960878628462[44] = 0;
   out_221487960878628462[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_221487960878628462[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_221487960878628462[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_221487960878628462[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_221487960878628462[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_221487960878628462[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_221487960878628462[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_221487960878628462[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_221487960878628462[53] = -9.8100000000000005*dt;
   out_221487960878628462[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_221487960878628462[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_221487960878628462[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_221487960878628462[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_221487960878628462[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_221487960878628462[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_221487960878628462[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_221487960878628462[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_221487960878628462[62] = 0;
   out_221487960878628462[63] = 0;
   out_221487960878628462[64] = 0;
   out_221487960878628462[65] = 0;
   out_221487960878628462[66] = 0;
   out_221487960878628462[67] = 0;
   out_221487960878628462[68] = 0;
   out_221487960878628462[69] = 0;
   out_221487960878628462[70] = 1;
   out_221487960878628462[71] = 0;
   out_221487960878628462[72] = 0;
   out_221487960878628462[73] = 0;
   out_221487960878628462[74] = 0;
   out_221487960878628462[75] = 0;
   out_221487960878628462[76] = 0;
   out_221487960878628462[77] = 0;
   out_221487960878628462[78] = 0;
   out_221487960878628462[79] = 0;
   out_221487960878628462[80] = 1;
}
void h_25(double *state, double *unused, double *out_319137421637609022) {
   out_319137421637609022[0] = state[6];
}
void H_25(double *state, double *unused, double *out_910996624291208289) {
   out_910996624291208289[0] = 0;
   out_910996624291208289[1] = 0;
   out_910996624291208289[2] = 0;
   out_910996624291208289[3] = 0;
   out_910996624291208289[4] = 0;
   out_910996624291208289[5] = 0;
   out_910996624291208289[6] = 1;
   out_910996624291208289[7] = 0;
   out_910996624291208289[8] = 0;
}
void h_24(double *state, double *unused, double *out_2686288490551722631) {
   out_2686288490551722631[0] = state[4];
   out_2686288490551722631[1] = state[5];
}
void H_24(double *state, double *unused, double *out_4912708702921716922) {
   out_4912708702921716922[0] = 0;
   out_4912708702921716922[1] = 0;
   out_4912708702921716922[2] = 0;
   out_4912708702921716922[3] = 0;
   out_4912708702921716922[4] = 1;
   out_4912708702921716922[5] = 0;
   out_4912708702921716922[6] = 0;
   out_4912708702921716922[7] = 0;
   out_4912708702921716922[8] = 0;
   out_4912708702921716922[9] = 0;
   out_4912708702921716922[10] = 0;
   out_4912708702921716922[11] = 0;
   out_4912708702921716922[12] = 0;
   out_4912708702921716922[13] = 0;
   out_4912708702921716922[14] = 1;
   out_4912708702921716922[15] = 0;
   out_4912708702921716922[16] = 0;
   out_4912708702921716922[17] = 0;
}
void h_30(double *state, double *unused, double *out_7573802073246914337) {
   out_7573802073246914337[0] = state[4];
}
void H_30(double *state, double *unused, double *out_7827686965782825044) {
   out_7827686965782825044[0] = 0;
   out_7827686965782825044[1] = 0;
   out_7827686965782825044[2] = 0;
   out_7827686965782825044[3] = 0;
   out_7827686965782825044[4] = 1;
   out_7827686965782825044[5] = 0;
   out_7827686965782825044[6] = 0;
   out_7827686965782825044[7] = 0;
   out_7827686965782825044[8] = 0;
}
void h_26(double *state, double *unused, double *out_2284779318657795871) {
   out_2284779318657795871[0] = state[7];
}
void H_26(double *state, double *unused, double *out_2830506694582847935) {
   out_2830506694582847935[0] = 0;
   out_2830506694582847935[1] = 0;
   out_2830506694582847935[2] = 0;
   out_2830506694582847935[3] = 0;
   out_2830506694582847935[4] = 0;
   out_2830506694582847935[5] = 0;
   out_2830506694582847935[6] = 0;
   out_2830506694582847935[7] = 1;
   out_2830506694582847935[8] = 0;
}
void h_27(double *state, double *unused, double *out_5320810689298454667) {
   out_5320810689298454667[0] = state[3];
}
void H_27(double *state, double *unused, double *out_5652923653982400133) {
   out_5652923653982400133[0] = 0;
   out_5652923653982400133[1] = 0;
   out_5652923653982400133[2] = 0;
   out_5652923653982400133[3] = 1;
   out_5652923653982400133[4] = 0;
   out_5652923653982400133[5] = 0;
   out_5652923653982400133[6] = 0;
   out_5652923653982400133[7] = 0;
   out_5652923653982400133[8] = 0;
}
void h_29(double *state, double *unused, double *out_3870179296669571500) {
   out_3870179296669571500[0] = state[1];
}
void H_29(double *state, double *unused, double *out_8337918310097217228) {
   out_8337918310097217228[0] = 0;
   out_8337918310097217228[1] = 1;
   out_8337918310097217228[2] = 0;
   out_8337918310097217228[3] = 0;
   out_8337918310097217228[4] = 0;
   out_8337918310097217228[5] = 0;
   out_8337918310097217228[6] = 0;
   out_8337918310097217228[7] = 0;
   out_8337918310097217228[8] = 0;
}
void h_28(double *state, double *unused, double *out_3661158482708386098) {
   out_3661158482708386098[0] = state[0];
}
void H_28(double *state, double *unused, double *out_1142838089956681474) {
   out_1142838089956681474[0] = 1;
   out_1142838089956681474[1] = 0;
   out_1142838089956681474[2] = 0;
   out_1142838089956681474[3] = 0;
   out_1142838089956681474[4] = 0;
   out_1142838089956681474[5] = 0;
   out_1142838089956681474[6] = 0;
   out_1142838089956681474[7] = 0;
   out_1142838089956681474[8] = 0;
}
void h_31(double *state, double *unused, double *out_7089972647987959958) {
   out_7089972647987959958[0] = state[8];
}
void H_31(double *state, double *unused, double *out_941642586168168717) {
   out_941642586168168717[0] = 0;
   out_941642586168168717[1] = 0;
   out_941642586168168717[2] = 0;
   out_941642586168168717[3] = 0;
   out_941642586168168717[4] = 0;
   out_941642586168168717[5] = 0;
   out_941642586168168717[6] = 0;
   out_941642586168168717[7] = 0;
   out_941642586168168717[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6823495906263670498) {
  err_fun(nom_x, delta_x, out_6823495906263670498);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2832333554644134208) {
  inv_err_fun(nom_x, true_x, out_2832333554644134208);
}
void car_H_mod_fun(double *state, double *out_8294511297811177194) {
  H_mod_fun(state, out_8294511297811177194);
}
void car_f_fun(double *state, double dt, double *out_156104586604879915) {
  f_fun(state,  dt, out_156104586604879915);
}
void car_F_fun(double *state, double dt, double *out_221487960878628462) {
  F_fun(state,  dt, out_221487960878628462);
}
void car_h_25(double *state, double *unused, double *out_319137421637609022) {
  h_25(state, unused, out_319137421637609022);
}
void car_H_25(double *state, double *unused, double *out_910996624291208289) {
  H_25(state, unused, out_910996624291208289);
}
void car_h_24(double *state, double *unused, double *out_2686288490551722631) {
  h_24(state, unused, out_2686288490551722631);
}
void car_H_24(double *state, double *unused, double *out_4912708702921716922) {
  H_24(state, unused, out_4912708702921716922);
}
void car_h_30(double *state, double *unused, double *out_7573802073246914337) {
  h_30(state, unused, out_7573802073246914337);
}
void car_H_30(double *state, double *unused, double *out_7827686965782825044) {
  H_30(state, unused, out_7827686965782825044);
}
void car_h_26(double *state, double *unused, double *out_2284779318657795871) {
  h_26(state, unused, out_2284779318657795871);
}
void car_H_26(double *state, double *unused, double *out_2830506694582847935) {
  H_26(state, unused, out_2830506694582847935);
}
void car_h_27(double *state, double *unused, double *out_5320810689298454667) {
  h_27(state, unused, out_5320810689298454667);
}
void car_H_27(double *state, double *unused, double *out_5652923653982400133) {
  H_27(state, unused, out_5652923653982400133);
}
void car_h_29(double *state, double *unused, double *out_3870179296669571500) {
  h_29(state, unused, out_3870179296669571500);
}
void car_H_29(double *state, double *unused, double *out_8337918310097217228) {
  H_29(state, unused, out_8337918310097217228);
}
void car_h_28(double *state, double *unused, double *out_3661158482708386098) {
  h_28(state, unused, out_3661158482708386098);
}
void car_H_28(double *state, double *unused, double *out_1142838089956681474) {
  H_28(state, unused, out_1142838089956681474);
}
void car_h_31(double *state, double *unused, double *out_7089972647987959958) {
  h_31(state, unused, out_7089972647987959958);
}
void car_H_31(double *state, double *unused, double *out_941642586168168717) {
  H_31(state, unused, out_941642586168168717);
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
