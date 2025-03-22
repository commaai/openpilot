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
void err_fun(double *nom_x, double *delta_x, double *out_3675669119552892075) {
   out_3675669119552892075[0] = delta_x[0] + nom_x[0];
   out_3675669119552892075[1] = delta_x[1] + nom_x[1];
   out_3675669119552892075[2] = delta_x[2] + nom_x[2];
   out_3675669119552892075[3] = delta_x[3] + nom_x[3];
   out_3675669119552892075[4] = delta_x[4] + nom_x[4];
   out_3675669119552892075[5] = delta_x[5] + nom_x[5];
   out_3675669119552892075[6] = delta_x[6] + nom_x[6];
   out_3675669119552892075[7] = delta_x[7] + nom_x[7];
   out_3675669119552892075[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1525509250774896015) {
   out_1525509250774896015[0] = -nom_x[0] + true_x[0];
   out_1525509250774896015[1] = -nom_x[1] + true_x[1];
   out_1525509250774896015[2] = -nom_x[2] + true_x[2];
   out_1525509250774896015[3] = -nom_x[3] + true_x[3];
   out_1525509250774896015[4] = -nom_x[4] + true_x[4];
   out_1525509250774896015[5] = -nom_x[5] + true_x[5];
   out_1525509250774896015[6] = -nom_x[6] + true_x[6];
   out_1525509250774896015[7] = -nom_x[7] + true_x[7];
   out_1525509250774896015[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_8908179234073782012) {
   out_8908179234073782012[0] = 1.0;
   out_8908179234073782012[1] = 0.0;
   out_8908179234073782012[2] = 0.0;
   out_8908179234073782012[3] = 0.0;
   out_8908179234073782012[4] = 0.0;
   out_8908179234073782012[5] = 0.0;
   out_8908179234073782012[6] = 0.0;
   out_8908179234073782012[7] = 0.0;
   out_8908179234073782012[8] = 0.0;
   out_8908179234073782012[9] = 0.0;
   out_8908179234073782012[10] = 1.0;
   out_8908179234073782012[11] = 0.0;
   out_8908179234073782012[12] = 0.0;
   out_8908179234073782012[13] = 0.0;
   out_8908179234073782012[14] = 0.0;
   out_8908179234073782012[15] = 0.0;
   out_8908179234073782012[16] = 0.0;
   out_8908179234073782012[17] = 0.0;
   out_8908179234073782012[18] = 0.0;
   out_8908179234073782012[19] = 0.0;
   out_8908179234073782012[20] = 1.0;
   out_8908179234073782012[21] = 0.0;
   out_8908179234073782012[22] = 0.0;
   out_8908179234073782012[23] = 0.0;
   out_8908179234073782012[24] = 0.0;
   out_8908179234073782012[25] = 0.0;
   out_8908179234073782012[26] = 0.0;
   out_8908179234073782012[27] = 0.0;
   out_8908179234073782012[28] = 0.0;
   out_8908179234073782012[29] = 0.0;
   out_8908179234073782012[30] = 1.0;
   out_8908179234073782012[31] = 0.0;
   out_8908179234073782012[32] = 0.0;
   out_8908179234073782012[33] = 0.0;
   out_8908179234073782012[34] = 0.0;
   out_8908179234073782012[35] = 0.0;
   out_8908179234073782012[36] = 0.0;
   out_8908179234073782012[37] = 0.0;
   out_8908179234073782012[38] = 0.0;
   out_8908179234073782012[39] = 0.0;
   out_8908179234073782012[40] = 1.0;
   out_8908179234073782012[41] = 0.0;
   out_8908179234073782012[42] = 0.0;
   out_8908179234073782012[43] = 0.0;
   out_8908179234073782012[44] = 0.0;
   out_8908179234073782012[45] = 0.0;
   out_8908179234073782012[46] = 0.0;
   out_8908179234073782012[47] = 0.0;
   out_8908179234073782012[48] = 0.0;
   out_8908179234073782012[49] = 0.0;
   out_8908179234073782012[50] = 1.0;
   out_8908179234073782012[51] = 0.0;
   out_8908179234073782012[52] = 0.0;
   out_8908179234073782012[53] = 0.0;
   out_8908179234073782012[54] = 0.0;
   out_8908179234073782012[55] = 0.0;
   out_8908179234073782012[56] = 0.0;
   out_8908179234073782012[57] = 0.0;
   out_8908179234073782012[58] = 0.0;
   out_8908179234073782012[59] = 0.0;
   out_8908179234073782012[60] = 1.0;
   out_8908179234073782012[61] = 0.0;
   out_8908179234073782012[62] = 0.0;
   out_8908179234073782012[63] = 0.0;
   out_8908179234073782012[64] = 0.0;
   out_8908179234073782012[65] = 0.0;
   out_8908179234073782012[66] = 0.0;
   out_8908179234073782012[67] = 0.0;
   out_8908179234073782012[68] = 0.0;
   out_8908179234073782012[69] = 0.0;
   out_8908179234073782012[70] = 1.0;
   out_8908179234073782012[71] = 0.0;
   out_8908179234073782012[72] = 0.0;
   out_8908179234073782012[73] = 0.0;
   out_8908179234073782012[74] = 0.0;
   out_8908179234073782012[75] = 0.0;
   out_8908179234073782012[76] = 0.0;
   out_8908179234073782012[77] = 0.0;
   out_8908179234073782012[78] = 0.0;
   out_8908179234073782012[79] = 0.0;
   out_8908179234073782012[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_2817676241617045765) {
   out_2817676241617045765[0] = state[0];
   out_2817676241617045765[1] = state[1];
   out_2817676241617045765[2] = state[2];
   out_2817676241617045765[3] = state[3];
   out_2817676241617045765[4] = state[4];
   out_2817676241617045765[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_2817676241617045765[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_2817676241617045765[7] = state[7];
   out_2817676241617045765[8] = state[8];
}
void F_fun(double *state, double dt, double *out_587119351607645481) {
   out_587119351607645481[0] = 1;
   out_587119351607645481[1] = 0;
   out_587119351607645481[2] = 0;
   out_587119351607645481[3] = 0;
   out_587119351607645481[4] = 0;
   out_587119351607645481[5] = 0;
   out_587119351607645481[6] = 0;
   out_587119351607645481[7] = 0;
   out_587119351607645481[8] = 0;
   out_587119351607645481[9] = 0;
   out_587119351607645481[10] = 1;
   out_587119351607645481[11] = 0;
   out_587119351607645481[12] = 0;
   out_587119351607645481[13] = 0;
   out_587119351607645481[14] = 0;
   out_587119351607645481[15] = 0;
   out_587119351607645481[16] = 0;
   out_587119351607645481[17] = 0;
   out_587119351607645481[18] = 0;
   out_587119351607645481[19] = 0;
   out_587119351607645481[20] = 1;
   out_587119351607645481[21] = 0;
   out_587119351607645481[22] = 0;
   out_587119351607645481[23] = 0;
   out_587119351607645481[24] = 0;
   out_587119351607645481[25] = 0;
   out_587119351607645481[26] = 0;
   out_587119351607645481[27] = 0;
   out_587119351607645481[28] = 0;
   out_587119351607645481[29] = 0;
   out_587119351607645481[30] = 1;
   out_587119351607645481[31] = 0;
   out_587119351607645481[32] = 0;
   out_587119351607645481[33] = 0;
   out_587119351607645481[34] = 0;
   out_587119351607645481[35] = 0;
   out_587119351607645481[36] = 0;
   out_587119351607645481[37] = 0;
   out_587119351607645481[38] = 0;
   out_587119351607645481[39] = 0;
   out_587119351607645481[40] = 1;
   out_587119351607645481[41] = 0;
   out_587119351607645481[42] = 0;
   out_587119351607645481[43] = 0;
   out_587119351607645481[44] = 0;
   out_587119351607645481[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_587119351607645481[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_587119351607645481[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_587119351607645481[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_587119351607645481[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_587119351607645481[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_587119351607645481[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_587119351607645481[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_587119351607645481[53] = -9.8000000000000007*dt;
   out_587119351607645481[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_587119351607645481[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_587119351607645481[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_587119351607645481[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_587119351607645481[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_587119351607645481[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_587119351607645481[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_587119351607645481[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_587119351607645481[62] = 0;
   out_587119351607645481[63] = 0;
   out_587119351607645481[64] = 0;
   out_587119351607645481[65] = 0;
   out_587119351607645481[66] = 0;
   out_587119351607645481[67] = 0;
   out_587119351607645481[68] = 0;
   out_587119351607645481[69] = 0;
   out_587119351607645481[70] = 1;
   out_587119351607645481[71] = 0;
   out_587119351607645481[72] = 0;
   out_587119351607645481[73] = 0;
   out_587119351607645481[74] = 0;
   out_587119351607645481[75] = 0;
   out_587119351607645481[76] = 0;
   out_587119351607645481[77] = 0;
   out_587119351607645481[78] = 0;
   out_587119351607645481[79] = 0;
   out_587119351607645481[80] = 1;
}
void h_25(double *state, double *unused, double *out_2948496957738460434) {
   out_2948496957738460434[0] = state[6];
}
void H_25(double *state, double *unused, double *out_345376613723532367) {
   out_345376613723532367[0] = 0;
   out_345376613723532367[1] = 0;
   out_345376613723532367[2] = 0;
   out_345376613723532367[3] = 0;
   out_345376613723532367[4] = 0;
   out_345376613723532367[5] = 0;
   out_345376613723532367[6] = 1;
   out_345376613723532367[7] = 0;
   out_345376613723532367[8] = 0;
}
void h_24(double *state, double *unused, double *out_1253312341000543437) {
   out_1253312341000543437[0] = state[4];
   out_1253312341000543437[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8873302273916824024) {
   out_8873302273916824024[0] = 0;
   out_8873302273916824024[1] = 0;
   out_8873302273916824024[2] = 0;
   out_8873302273916824024[3] = 0;
   out_8873302273916824024[4] = 1;
   out_8873302273916824024[5] = 0;
   out_8873302273916824024[6] = 0;
   out_8873302273916824024[7] = 0;
   out_8873302273916824024[8] = 0;
   out_8873302273916824024[9] = 0;
   out_8873302273916824024[10] = 0;
   out_8873302273916824024[11] = 0;
   out_8873302273916824024[12] = 0;
   out_8873302273916824024[13] = 0;
   out_8873302273916824024[14] = 1;
   out_8873302273916824024[15] = 0;
   out_8873302273916824024[16] = 0;
   out_8873302273916824024[17] = 0;
}
void h_30(double *state, double *unused, double *out_4399128350367343601) {
   out_4399128350367343601[0] = state[4];
}
void H_30(double *state, double *unused, double *out_216037666580292297) {
   out_216037666580292297[0] = 0;
   out_216037666580292297[1] = 0;
   out_216037666580292297[2] = 0;
   out_216037666580292297[3] = 0;
   out_216037666580292297[4] = 1;
   out_216037666580292297[5] = 0;
   out_216037666580292297[6] = 0;
   out_216037666580292297[7] = 0;
   out_216037666580292297[8] = 0;
}
void h_26(double *state, double *unused, double *out_1362342298923010916) {
   out_1362342298923010916[0] = state[7];
}
void H_26(double *state, double *unused, double *out_3396126705150523857) {
   out_3396126705150523857[0] = 0;
   out_3396126705150523857[1] = 0;
   out_3396126705150523857[2] = 0;
   out_3396126705150523857[3] = 0;
   out_3396126705150523857[4] = 0;
   out_3396126705150523857[5] = 0;
   out_3396126705150523857[6] = 0;
   out_3396126705150523857[7] = 1;
   out_3396126705150523857[8] = 0;
}
void h_27(double *state, double *unused, double *out_9204321679697507688) {
   out_9204321679697507688[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1958725645220132614) {
   out_1958725645220132614[0] = 0;
   out_1958725645220132614[1] = 0;
   out_1958725645220132614[2] = 0;
   out_1958725645220132614[3] = 1;
   out_1958725645220132614[4] = 0;
   out_1958725645220132614[5] = 0;
   out_1958725645220132614[6] = 0;
   out_1958725645220132614[7] = 0;
   out_1958725645220132614[8] = 0;
}
void h_29(double *state, double *unused, double *out_602544917293502044) {
   out_602544917293502044[0] = state[1];
}
void H_29(double *state, double *unused, double *out_3672088372089683647) {
   out_3672088372089683647[0] = 0;
   out_3672088372089683647[1] = 1;
   out_3672088372089683647[2] = 0;
   out_3672088372089683647[3] = 0;
   out_3672088372089683647[4] = 0;
   out_3672088372089683647[5] = 0;
   out_3672088372089683647[6] = 0;
   out_3672088372089683647[7] = 0;
   out_3672088372089683647[8] = 0;
}
void h_28(double *state, double *unused, double *out_4281120956433966857) {
   out_4281120956433966857[0] = state[0];
}
void H_28(double *state, double *unused, double *out_1708458100524357396) {
   out_1708458100524357396[0] = 1;
   out_1708458100524357396[1] = 0;
   out_1708458100524357396[2] = 0;
   out_1708458100524357396[3] = 0;
   out_1708458100524357396[4] = 0;
   out_1708458100524357396[5] = 0;
   out_1708458100524357396[6] = 0;
   out_1708458100524357396[7] = 0;
   out_1708458100524357396[8] = 0;
}
void h_31(double *state, double *unused, double *out_5944889882572371594) {
   out_5944889882572371594[0] = state[8];
}
void H_31(double *state, double *unused, double *out_376022575600492795) {
   out_376022575600492795[0] = 0;
   out_376022575600492795[1] = 0;
   out_376022575600492795[2] = 0;
   out_376022575600492795[3] = 0;
   out_376022575600492795[4] = 0;
   out_376022575600492795[5] = 0;
   out_376022575600492795[6] = 0;
   out_376022575600492795[7] = 0;
   out_376022575600492795[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_3675669119552892075) {
  err_fun(nom_x, delta_x, out_3675669119552892075);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_1525509250774896015) {
  inv_err_fun(nom_x, true_x, out_1525509250774896015);
}
void car_H_mod_fun(double *state, double *out_8908179234073782012) {
  H_mod_fun(state, out_8908179234073782012);
}
void car_f_fun(double *state, double dt, double *out_2817676241617045765) {
  f_fun(state,  dt, out_2817676241617045765);
}
void car_F_fun(double *state, double dt, double *out_587119351607645481) {
  F_fun(state,  dt, out_587119351607645481);
}
void car_h_25(double *state, double *unused, double *out_2948496957738460434) {
  h_25(state, unused, out_2948496957738460434);
}
void car_H_25(double *state, double *unused, double *out_345376613723532367) {
  H_25(state, unused, out_345376613723532367);
}
void car_h_24(double *state, double *unused, double *out_1253312341000543437) {
  h_24(state, unused, out_1253312341000543437);
}
void car_H_24(double *state, double *unused, double *out_8873302273916824024) {
  H_24(state, unused, out_8873302273916824024);
}
void car_h_30(double *state, double *unused, double *out_4399128350367343601) {
  h_30(state, unused, out_4399128350367343601);
}
void car_H_30(double *state, double *unused, double *out_216037666580292297) {
  H_30(state, unused, out_216037666580292297);
}
void car_h_26(double *state, double *unused, double *out_1362342298923010916) {
  h_26(state, unused, out_1362342298923010916);
}
void car_H_26(double *state, double *unused, double *out_3396126705150523857) {
  H_26(state, unused, out_3396126705150523857);
}
void car_h_27(double *state, double *unused, double *out_9204321679697507688) {
  h_27(state, unused, out_9204321679697507688);
}
void car_H_27(double *state, double *unused, double *out_1958725645220132614) {
  H_27(state, unused, out_1958725645220132614);
}
void car_h_29(double *state, double *unused, double *out_602544917293502044) {
  h_29(state, unused, out_602544917293502044);
}
void car_H_29(double *state, double *unused, double *out_3672088372089683647) {
  H_29(state, unused, out_3672088372089683647);
}
void car_h_28(double *state, double *unused, double *out_4281120956433966857) {
  h_28(state, unused, out_4281120956433966857);
}
void car_H_28(double *state, double *unused, double *out_1708458100524357396) {
  H_28(state, unused, out_1708458100524357396);
}
void car_h_31(double *state, double *unused, double *out_5944889882572371594) {
  h_31(state, unused, out_5944889882572371594);
}
void car_H_31(double *state, double *unused, double *out_376022575600492795) {
  H_31(state, unused, out_376022575600492795);
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
