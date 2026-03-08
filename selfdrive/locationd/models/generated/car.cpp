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
void err_fun(double *nom_x, double *delta_x, double *out_6612010329800967159) {
   out_6612010329800967159[0] = delta_x[0] + nom_x[0];
   out_6612010329800967159[1] = delta_x[1] + nom_x[1];
   out_6612010329800967159[2] = delta_x[2] + nom_x[2];
   out_6612010329800967159[3] = delta_x[3] + nom_x[3];
   out_6612010329800967159[4] = delta_x[4] + nom_x[4];
   out_6612010329800967159[5] = delta_x[5] + nom_x[5];
   out_6612010329800967159[6] = delta_x[6] + nom_x[6];
   out_6612010329800967159[7] = delta_x[7] + nom_x[7];
   out_6612010329800967159[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8935913149389536496) {
   out_8935913149389536496[0] = -nom_x[0] + true_x[0];
   out_8935913149389536496[1] = -nom_x[1] + true_x[1];
   out_8935913149389536496[2] = -nom_x[2] + true_x[2];
   out_8935913149389536496[3] = -nom_x[3] + true_x[3];
   out_8935913149389536496[4] = -nom_x[4] + true_x[4];
   out_8935913149389536496[5] = -nom_x[5] + true_x[5];
   out_8935913149389536496[6] = -nom_x[6] + true_x[6];
   out_8935913149389536496[7] = -nom_x[7] + true_x[7];
   out_8935913149389536496[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_1202674038686624606) {
   out_1202674038686624606[0] = 1.0;
   out_1202674038686624606[1] = 0.0;
   out_1202674038686624606[2] = 0.0;
   out_1202674038686624606[3] = 0.0;
   out_1202674038686624606[4] = 0.0;
   out_1202674038686624606[5] = 0.0;
   out_1202674038686624606[6] = 0.0;
   out_1202674038686624606[7] = 0.0;
   out_1202674038686624606[8] = 0.0;
   out_1202674038686624606[9] = 0.0;
   out_1202674038686624606[10] = 1.0;
   out_1202674038686624606[11] = 0.0;
   out_1202674038686624606[12] = 0.0;
   out_1202674038686624606[13] = 0.0;
   out_1202674038686624606[14] = 0.0;
   out_1202674038686624606[15] = 0.0;
   out_1202674038686624606[16] = 0.0;
   out_1202674038686624606[17] = 0.0;
   out_1202674038686624606[18] = 0.0;
   out_1202674038686624606[19] = 0.0;
   out_1202674038686624606[20] = 1.0;
   out_1202674038686624606[21] = 0.0;
   out_1202674038686624606[22] = 0.0;
   out_1202674038686624606[23] = 0.0;
   out_1202674038686624606[24] = 0.0;
   out_1202674038686624606[25] = 0.0;
   out_1202674038686624606[26] = 0.0;
   out_1202674038686624606[27] = 0.0;
   out_1202674038686624606[28] = 0.0;
   out_1202674038686624606[29] = 0.0;
   out_1202674038686624606[30] = 1.0;
   out_1202674038686624606[31] = 0.0;
   out_1202674038686624606[32] = 0.0;
   out_1202674038686624606[33] = 0.0;
   out_1202674038686624606[34] = 0.0;
   out_1202674038686624606[35] = 0.0;
   out_1202674038686624606[36] = 0.0;
   out_1202674038686624606[37] = 0.0;
   out_1202674038686624606[38] = 0.0;
   out_1202674038686624606[39] = 0.0;
   out_1202674038686624606[40] = 1.0;
   out_1202674038686624606[41] = 0.0;
   out_1202674038686624606[42] = 0.0;
   out_1202674038686624606[43] = 0.0;
   out_1202674038686624606[44] = 0.0;
   out_1202674038686624606[45] = 0.0;
   out_1202674038686624606[46] = 0.0;
   out_1202674038686624606[47] = 0.0;
   out_1202674038686624606[48] = 0.0;
   out_1202674038686624606[49] = 0.0;
   out_1202674038686624606[50] = 1.0;
   out_1202674038686624606[51] = 0.0;
   out_1202674038686624606[52] = 0.0;
   out_1202674038686624606[53] = 0.0;
   out_1202674038686624606[54] = 0.0;
   out_1202674038686624606[55] = 0.0;
   out_1202674038686624606[56] = 0.0;
   out_1202674038686624606[57] = 0.0;
   out_1202674038686624606[58] = 0.0;
   out_1202674038686624606[59] = 0.0;
   out_1202674038686624606[60] = 1.0;
   out_1202674038686624606[61] = 0.0;
   out_1202674038686624606[62] = 0.0;
   out_1202674038686624606[63] = 0.0;
   out_1202674038686624606[64] = 0.0;
   out_1202674038686624606[65] = 0.0;
   out_1202674038686624606[66] = 0.0;
   out_1202674038686624606[67] = 0.0;
   out_1202674038686624606[68] = 0.0;
   out_1202674038686624606[69] = 0.0;
   out_1202674038686624606[70] = 1.0;
   out_1202674038686624606[71] = 0.0;
   out_1202674038686624606[72] = 0.0;
   out_1202674038686624606[73] = 0.0;
   out_1202674038686624606[74] = 0.0;
   out_1202674038686624606[75] = 0.0;
   out_1202674038686624606[76] = 0.0;
   out_1202674038686624606[77] = 0.0;
   out_1202674038686624606[78] = 0.0;
   out_1202674038686624606[79] = 0.0;
   out_1202674038686624606[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_2072276713079294601) {
   out_2072276713079294601[0] = state[0];
   out_2072276713079294601[1] = state[1];
   out_2072276713079294601[2] = state[2];
   out_2072276713079294601[3] = state[3];
   out_2072276713079294601[4] = state[4];
   out_2072276713079294601[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_2072276713079294601[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_2072276713079294601[7] = state[7];
   out_2072276713079294601[8] = state[8];
}
void F_fun(double *state, double dt, double *out_6787727984604679414) {
   out_6787727984604679414[0] = 1;
   out_6787727984604679414[1] = 0;
   out_6787727984604679414[2] = 0;
   out_6787727984604679414[3] = 0;
   out_6787727984604679414[4] = 0;
   out_6787727984604679414[5] = 0;
   out_6787727984604679414[6] = 0;
   out_6787727984604679414[7] = 0;
   out_6787727984604679414[8] = 0;
   out_6787727984604679414[9] = 0;
   out_6787727984604679414[10] = 1;
   out_6787727984604679414[11] = 0;
   out_6787727984604679414[12] = 0;
   out_6787727984604679414[13] = 0;
   out_6787727984604679414[14] = 0;
   out_6787727984604679414[15] = 0;
   out_6787727984604679414[16] = 0;
   out_6787727984604679414[17] = 0;
   out_6787727984604679414[18] = 0;
   out_6787727984604679414[19] = 0;
   out_6787727984604679414[20] = 1;
   out_6787727984604679414[21] = 0;
   out_6787727984604679414[22] = 0;
   out_6787727984604679414[23] = 0;
   out_6787727984604679414[24] = 0;
   out_6787727984604679414[25] = 0;
   out_6787727984604679414[26] = 0;
   out_6787727984604679414[27] = 0;
   out_6787727984604679414[28] = 0;
   out_6787727984604679414[29] = 0;
   out_6787727984604679414[30] = 1;
   out_6787727984604679414[31] = 0;
   out_6787727984604679414[32] = 0;
   out_6787727984604679414[33] = 0;
   out_6787727984604679414[34] = 0;
   out_6787727984604679414[35] = 0;
   out_6787727984604679414[36] = 0;
   out_6787727984604679414[37] = 0;
   out_6787727984604679414[38] = 0;
   out_6787727984604679414[39] = 0;
   out_6787727984604679414[40] = 1;
   out_6787727984604679414[41] = 0;
   out_6787727984604679414[42] = 0;
   out_6787727984604679414[43] = 0;
   out_6787727984604679414[44] = 0;
   out_6787727984604679414[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_6787727984604679414[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_6787727984604679414[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6787727984604679414[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6787727984604679414[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_6787727984604679414[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_6787727984604679414[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_6787727984604679414[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_6787727984604679414[53] = -9.8100000000000005*dt;
   out_6787727984604679414[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_6787727984604679414[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_6787727984604679414[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6787727984604679414[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6787727984604679414[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_6787727984604679414[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_6787727984604679414[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_6787727984604679414[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6787727984604679414[62] = 0;
   out_6787727984604679414[63] = 0;
   out_6787727984604679414[64] = 0;
   out_6787727984604679414[65] = 0;
   out_6787727984604679414[66] = 0;
   out_6787727984604679414[67] = 0;
   out_6787727984604679414[68] = 0;
   out_6787727984604679414[69] = 0;
   out_6787727984604679414[70] = 1;
   out_6787727984604679414[71] = 0;
   out_6787727984604679414[72] = 0;
   out_6787727984604679414[73] = 0;
   out_6787727984604679414[74] = 0;
   out_6787727984604679414[75] = 0;
   out_6787727984604679414[76] = 0;
   out_6787727984604679414[77] = 0;
   out_6787727984604679414[78] = 0;
   out_6787727984604679414[79] = 0;
   out_6787727984604679414[80] = 1;
}
void h_25(double *state, double *unused, double *out_2354092351086383892) {
   out_2354092351086383892[0] = state[6];
}
void H_25(double *state, double *unused, double *out_9124012071660484411) {
   out_9124012071660484411[0] = 0;
   out_9124012071660484411[1] = 0;
   out_9124012071660484411[2] = 0;
   out_9124012071660484411[3] = 0;
   out_9124012071660484411[4] = 0;
   out_9124012071660484411[5] = 0;
   out_9124012071660484411[6] = 1;
   out_9124012071660484411[7] = 0;
   out_9124012071660484411[8] = 0;
}
void h_24(double *state, double *unused, double *out_5283557851803415774) {
   out_5283557851803415774[0] = state[4];
   out_5283557851803415774[1] = state[5];
}
void H_24(double *state, double *unused, double *out_5764806403246744069) {
   out_5764806403246744069[0] = 0;
   out_5764806403246744069[1] = 0;
   out_5764806403246744069[2] = 0;
   out_5764806403246744069[3] = 0;
   out_5764806403246744069[4] = 1;
   out_5764806403246744069[5] = 0;
   out_5764806403246744069[6] = 0;
   out_5764806403246744069[7] = 0;
   out_5764806403246744069[8] = 0;
   out_5764806403246744069[9] = 0;
   out_5764806403246744069[10] = 0;
   out_5764806403246744069[11] = 0;
   out_5764806403246744069[12] = 0;
   out_5764806403246744069[13] = 0;
   out_5764806403246744069[14] = 1;
   out_5764806403246744069[15] = 0;
   out_5764806403246744069[16] = 0;
   out_5764806403246744069[17] = 0;
}
void h_30(double *state, double *unused, double *out_1507591125342117760) {
   out_1507591125342117760[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4596315741532876213) {
   out_4596315741532876213[0] = 0;
   out_4596315741532876213[1] = 0;
   out_4596315741532876213[2] = 0;
   out_4596315741532876213[3] = 0;
   out_4596315741532876213[4] = 1;
   out_4596315741532876213[5] = 0;
   out_4596315741532876213[6] = 0;
   out_4596315741532876213[7] = 0;
   out_4596315741532876213[8] = 0;
}
void h_26(double *state, double *unused, double *out_2451100978243780195) {
   out_2451100978243780195[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5382508752786428187) {
   out_5382508752786428187[0] = 0;
   out_5382508752786428187[1] = 0;
   out_5382508752786428187[2] = 0;
   out_5382508752786428187[3] = 0;
   out_5382508752786428187[4] = 0;
   out_5382508752786428187[5] = 0;
   out_5382508752786428187[6] = 0;
   out_5382508752786428187[7] = 1;
   out_5382508752786428187[8] = 0;
}
void h_27(double *state, double *unused, double *out_1372008257751085780) {
   out_1372008257751085780[0] = state[3];
}
void H_27(double *state, double *unused, double *out_6819909812716819430) {
   out_6819909812716819430[0] = 0;
   out_6819909812716819430[1] = 0;
   out_6819909812716819430[2] = 0;
   out_6819909812716819430[3] = 1;
   out_6819909812716819430[4] = 0;
   out_6819909812716819430[5] = 0;
   out_6819909812716819430[6] = 0;
   out_6819909812716819430[7] = 0;
   out_6819909812716819430[8] = 0;
}
void h_29(double *state, double *unused, double *out_6509264393002963405) {
   out_6509264393002963405[0] = state[1];
}
void H_29(double *state, double *unused, double *out_5106547085847268397) {
   out_5106547085847268397[0] = 0;
   out_5106547085847268397[1] = 1;
   out_5106547085847268397[2] = 0;
   out_5106547085847268397[3] = 0;
   out_5106547085847268397[4] = 0;
   out_5106547085847268397[5] = 0;
   out_5106547085847268397[6] = 0;
   out_5106547085847268397[7] = 0;
   out_5106547085847268397[8] = 0;
}
void h_28(double *state, double *unused, double *out_768692373074608263) {
   out_768692373074608263[0] = state[0];
}
void H_28(double *state, double *unused, double *out_7070177357412594648) {
   out_7070177357412594648[0] = 1;
   out_7070177357412594648[1] = 0;
   out_7070177357412594648[2] = 0;
   out_7070177357412594648[3] = 0;
   out_7070177357412594648[4] = 0;
   out_7070177357412594648[5] = 0;
   out_7070177357412594648[6] = 0;
   out_7070177357412594648[7] = 0;
   out_7070177357412594648[8] = 0;
}
void h_31(double *state, double *unused, double *out_1389583731408741016) {
   out_1389583731408741016[0] = state[8];
}
void H_31(double *state, double *unused, double *out_4756300650553076711) {
   out_4756300650553076711[0] = 0;
   out_4756300650553076711[1] = 0;
   out_4756300650553076711[2] = 0;
   out_4756300650553076711[3] = 0;
   out_4756300650553076711[4] = 0;
   out_4756300650553076711[5] = 0;
   out_4756300650553076711[6] = 0;
   out_4756300650553076711[7] = 0;
   out_4756300650553076711[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6612010329800967159) {
  err_fun(nom_x, delta_x, out_6612010329800967159);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8935913149389536496) {
  inv_err_fun(nom_x, true_x, out_8935913149389536496);
}
void car_H_mod_fun(double *state, double *out_1202674038686624606) {
  H_mod_fun(state, out_1202674038686624606);
}
void car_f_fun(double *state, double dt, double *out_2072276713079294601) {
  f_fun(state,  dt, out_2072276713079294601);
}
void car_F_fun(double *state, double dt, double *out_6787727984604679414) {
  F_fun(state,  dt, out_6787727984604679414);
}
void car_h_25(double *state, double *unused, double *out_2354092351086383892) {
  h_25(state, unused, out_2354092351086383892);
}
void car_H_25(double *state, double *unused, double *out_9124012071660484411) {
  H_25(state, unused, out_9124012071660484411);
}
void car_h_24(double *state, double *unused, double *out_5283557851803415774) {
  h_24(state, unused, out_5283557851803415774);
}
void car_H_24(double *state, double *unused, double *out_5764806403246744069) {
  H_24(state, unused, out_5764806403246744069);
}
void car_h_30(double *state, double *unused, double *out_1507591125342117760) {
  h_30(state, unused, out_1507591125342117760);
}
void car_H_30(double *state, double *unused, double *out_4596315741532876213) {
  H_30(state, unused, out_4596315741532876213);
}
void car_h_26(double *state, double *unused, double *out_2451100978243780195) {
  h_26(state, unused, out_2451100978243780195);
}
void car_H_26(double *state, double *unused, double *out_5382508752786428187) {
  H_26(state, unused, out_5382508752786428187);
}
void car_h_27(double *state, double *unused, double *out_1372008257751085780) {
  h_27(state, unused, out_1372008257751085780);
}
void car_H_27(double *state, double *unused, double *out_6819909812716819430) {
  H_27(state, unused, out_6819909812716819430);
}
void car_h_29(double *state, double *unused, double *out_6509264393002963405) {
  h_29(state, unused, out_6509264393002963405);
}
void car_H_29(double *state, double *unused, double *out_5106547085847268397) {
  H_29(state, unused, out_5106547085847268397);
}
void car_h_28(double *state, double *unused, double *out_768692373074608263) {
  h_28(state, unused, out_768692373074608263);
}
void car_H_28(double *state, double *unused, double *out_7070177357412594648) {
  H_28(state, unused, out_7070177357412594648);
}
void car_h_31(double *state, double *unused, double *out_1389583731408741016) {
  h_31(state, unused, out_1389583731408741016);
}
void car_H_31(double *state, double *unused, double *out_4756300650553076711) {
  H_31(state, unused, out_4756300650553076711);
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
