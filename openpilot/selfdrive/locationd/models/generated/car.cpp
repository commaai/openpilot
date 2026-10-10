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
void err_fun(double *nom_x, double *delta_x, double *out_7002678419470636064) {
   out_7002678419470636064[0] = delta_x[0] + nom_x[0];
   out_7002678419470636064[1] = delta_x[1] + nom_x[1];
   out_7002678419470636064[2] = delta_x[2] + nom_x[2];
   out_7002678419470636064[3] = delta_x[3] + nom_x[3];
   out_7002678419470636064[4] = delta_x[4] + nom_x[4];
   out_7002678419470636064[5] = delta_x[5] + nom_x[5];
   out_7002678419470636064[6] = delta_x[6] + nom_x[6];
   out_7002678419470636064[7] = delta_x[7] + nom_x[7];
   out_7002678419470636064[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_4320393458615466509) {
   out_4320393458615466509[0] = -nom_x[0] + true_x[0];
   out_4320393458615466509[1] = -nom_x[1] + true_x[1];
   out_4320393458615466509[2] = -nom_x[2] + true_x[2];
   out_4320393458615466509[3] = -nom_x[3] + true_x[3];
   out_4320393458615466509[4] = -nom_x[4] + true_x[4];
   out_4320393458615466509[5] = -nom_x[5] + true_x[5];
   out_4320393458615466509[6] = -nom_x[6] + true_x[6];
   out_4320393458615466509[7] = -nom_x[7] + true_x[7];
   out_4320393458615466509[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_7742013015815190770) {
   out_7742013015815190770[0] = 1.0;
   out_7742013015815190770[1] = 0.0;
   out_7742013015815190770[2] = 0.0;
   out_7742013015815190770[3] = 0.0;
   out_7742013015815190770[4] = 0.0;
   out_7742013015815190770[5] = 0.0;
   out_7742013015815190770[6] = 0.0;
   out_7742013015815190770[7] = 0.0;
   out_7742013015815190770[8] = 0.0;
   out_7742013015815190770[9] = 0.0;
   out_7742013015815190770[10] = 1.0;
   out_7742013015815190770[11] = 0.0;
   out_7742013015815190770[12] = 0.0;
   out_7742013015815190770[13] = 0.0;
   out_7742013015815190770[14] = 0.0;
   out_7742013015815190770[15] = 0.0;
   out_7742013015815190770[16] = 0.0;
   out_7742013015815190770[17] = 0.0;
   out_7742013015815190770[18] = 0.0;
   out_7742013015815190770[19] = 0.0;
   out_7742013015815190770[20] = 1.0;
   out_7742013015815190770[21] = 0.0;
   out_7742013015815190770[22] = 0.0;
   out_7742013015815190770[23] = 0.0;
   out_7742013015815190770[24] = 0.0;
   out_7742013015815190770[25] = 0.0;
   out_7742013015815190770[26] = 0.0;
   out_7742013015815190770[27] = 0.0;
   out_7742013015815190770[28] = 0.0;
   out_7742013015815190770[29] = 0.0;
   out_7742013015815190770[30] = 1.0;
   out_7742013015815190770[31] = 0.0;
   out_7742013015815190770[32] = 0.0;
   out_7742013015815190770[33] = 0.0;
   out_7742013015815190770[34] = 0.0;
   out_7742013015815190770[35] = 0.0;
   out_7742013015815190770[36] = 0.0;
   out_7742013015815190770[37] = 0.0;
   out_7742013015815190770[38] = 0.0;
   out_7742013015815190770[39] = 0.0;
   out_7742013015815190770[40] = 1.0;
   out_7742013015815190770[41] = 0.0;
   out_7742013015815190770[42] = 0.0;
   out_7742013015815190770[43] = 0.0;
   out_7742013015815190770[44] = 0.0;
   out_7742013015815190770[45] = 0.0;
   out_7742013015815190770[46] = 0.0;
   out_7742013015815190770[47] = 0.0;
   out_7742013015815190770[48] = 0.0;
   out_7742013015815190770[49] = 0.0;
   out_7742013015815190770[50] = 1.0;
   out_7742013015815190770[51] = 0.0;
   out_7742013015815190770[52] = 0.0;
   out_7742013015815190770[53] = 0.0;
   out_7742013015815190770[54] = 0.0;
   out_7742013015815190770[55] = 0.0;
   out_7742013015815190770[56] = 0.0;
   out_7742013015815190770[57] = 0.0;
   out_7742013015815190770[58] = 0.0;
   out_7742013015815190770[59] = 0.0;
   out_7742013015815190770[60] = 1.0;
   out_7742013015815190770[61] = 0.0;
   out_7742013015815190770[62] = 0.0;
   out_7742013015815190770[63] = 0.0;
   out_7742013015815190770[64] = 0.0;
   out_7742013015815190770[65] = 0.0;
   out_7742013015815190770[66] = 0.0;
   out_7742013015815190770[67] = 0.0;
   out_7742013015815190770[68] = 0.0;
   out_7742013015815190770[69] = 0.0;
   out_7742013015815190770[70] = 1.0;
   out_7742013015815190770[71] = 0.0;
   out_7742013015815190770[72] = 0.0;
   out_7742013015815190770[73] = 0.0;
   out_7742013015815190770[74] = 0.0;
   out_7742013015815190770[75] = 0.0;
   out_7742013015815190770[76] = 0.0;
   out_7742013015815190770[77] = 0.0;
   out_7742013015815190770[78] = 0.0;
   out_7742013015815190770[79] = 0.0;
   out_7742013015815190770[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7561579577695376867) {
   out_7561579577695376867[0] = state[0];
   out_7561579577695376867[1] = state[1];
   out_7561579577695376867[2] = state[2];
   out_7561579577695376867[3] = state[3];
   out_7561579577695376867[4] = state[4];
   out_7561579577695376867[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_7561579577695376867[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7561579577695376867[7] = state[7];
   out_7561579577695376867[8] = state[8];
}
void F_fun(double *state, double dt, double *out_3970912247205260696) {
   out_3970912247205260696[0] = 1;
   out_3970912247205260696[1] = 0;
   out_3970912247205260696[2] = 0;
   out_3970912247205260696[3] = 0;
   out_3970912247205260696[4] = 0;
   out_3970912247205260696[5] = 0;
   out_3970912247205260696[6] = 0;
   out_3970912247205260696[7] = 0;
   out_3970912247205260696[8] = 0;
   out_3970912247205260696[9] = 0;
   out_3970912247205260696[10] = 1;
   out_3970912247205260696[11] = 0;
   out_3970912247205260696[12] = 0;
   out_3970912247205260696[13] = 0;
   out_3970912247205260696[14] = 0;
   out_3970912247205260696[15] = 0;
   out_3970912247205260696[16] = 0;
   out_3970912247205260696[17] = 0;
   out_3970912247205260696[18] = 0;
   out_3970912247205260696[19] = 0;
   out_3970912247205260696[20] = 1;
   out_3970912247205260696[21] = 0;
   out_3970912247205260696[22] = 0;
   out_3970912247205260696[23] = 0;
   out_3970912247205260696[24] = 0;
   out_3970912247205260696[25] = 0;
   out_3970912247205260696[26] = 0;
   out_3970912247205260696[27] = 0;
   out_3970912247205260696[28] = 0;
   out_3970912247205260696[29] = 0;
   out_3970912247205260696[30] = 1;
   out_3970912247205260696[31] = 0;
   out_3970912247205260696[32] = 0;
   out_3970912247205260696[33] = 0;
   out_3970912247205260696[34] = 0;
   out_3970912247205260696[35] = 0;
   out_3970912247205260696[36] = 0;
   out_3970912247205260696[37] = 0;
   out_3970912247205260696[38] = 0;
   out_3970912247205260696[39] = 0;
   out_3970912247205260696[40] = 1;
   out_3970912247205260696[41] = 0;
   out_3970912247205260696[42] = 0;
   out_3970912247205260696[43] = 0;
   out_3970912247205260696[44] = 0;
   out_3970912247205260696[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_3970912247205260696[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_3970912247205260696[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_3970912247205260696[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_3970912247205260696[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_3970912247205260696[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_3970912247205260696[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_3970912247205260696[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_3970912247205260696[53] = -9.8100000000000005*dt;
   out_3970912247205260696[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_3970912247205260696[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_3970912247205260696[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3970912247205260696[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3970912247205260696[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_3970912247205260696[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_3970912247205260696[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_3970912247205260696[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3970912247205260696[62] = 0;
   out_3970912247205260696[63] = 0;
   out_3970912247205260696[64] = 0;
   out_3970912247205260696[65] = 0;
   out_3970912247205260696[66] = 0;
   out_3970912247205260696[67] = 0;
   out_3970912247205260696[68] = 0;
   out_3970912247205260696[69] = 0;
   out_3970912247205260696[70] = 1;
   out_3970912247205260696[71] = 0;
   out_3970912247205260696[72] = 0;
   out_3970912247205260696[73] = 0;
   out_3970912247205260696[74] = 0;
   out_3970912247205260696[75] = 0;
   out_3970912247205260696[76] = 0;
   out_3970912247205260696[77] = 0;
   out_3970912247205260696[78] = 0;
   out_3970912247205260696[79] = 0;
   out_3970912247205260696[80] = 1;
}
void h_25(double *state, double *unused, double *out_416451695291826654) {
   out_416451695291826654[0] = state[6];
}
void H_25(double *state, double *unused, double *out_4218964377169825622) {
   out_4218964377169825622[0] = 0;
   out_4218964377169825622[1] = 0;
   out_4218964377169825622[2] = 0;
   out_4218964377169825622[3] = 0;
   out_4218964377169825622[4] = 0;
   out_4218964377169825622[5] = 0;
   out_4218964377169825622[6] = 1;
   out_4218964377169825622[7] = 0;
   out_4218964377169825622[8] = 0;
}
void h_24(double *state, double *unused, double *out_3590841845749764283) {
   out_3590841845749764283[0] = state[4];
   out_3590841845749764283[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2046314778164326056) {
   out_2046314778164326056[0] = 0;
   out_2046314778164326056[1] = 0;
   out_2046314778164326056[2] = 0;
   out_2046314778164326056[3] = 0;
   out_2046314778164326056[4] = 1;
   out_2046314778164326056[5] = 0;
   out_2046314778164326056[6] = 0;
   out_2046314778164326056[7] = 0;
   out_2046314778164326056[8] = 0;
   out_2046314778164326056[9] = 0;
   out_2046314778164326056[10] = 0;
   out_2046314778164326056[11] = 0;
   out_2046314778164326056[12] = 0;
   out_2046314778164326056[13] = 0;
   out_2046314778164326056[14] = 1;
   out_2046314778164326056[15] = 0;
   out_2046314778164326056[16] = 0;
   out_2046314778164326056[17] = 0;
}
void h_30(double *state, double *unused, double *out_6354383531058524282) {
   out_6354383531058524282[0] = state[4];
}
void H_30(double *state, double *unused, double *out_7311089355048109239) {
   out_7311089355048109239[0] = 0;
   out_7311089355048109239[1] = 0;
   out_7311089355048109239[2] = 0;
   out_7311089355048109239[3] = 0;
   out_7311089355048109239[4] = 1;
   out_7311089355048109239[5] = 0;
   out_7311089355048109239[6] = 0;
   out_7311089355048109239[7] = 0;
   out_7311089355048109239[8] = 0;
}
void h_26(double *state, double *unused, double *out_3941861940183054722) {
   out_3941861940183054722[0] = state[7];
}
void H_26(double *state, double *unused, double *out_7523490346930626223) {
   out_7523490346930626223[0] = 0;
   out_7523490346930626223[1] = 0;
   out_7523490346930626223[2] = 0;
   out_7523490346930626223[3] = 0;
   out_7523490346930626223[4] = 0;
   out_7523490346930626223[5] = 0;
   out_7523490346930626223[6] = 0;
   out_7523490346930626223[7] = 1;
   out_7523490346930626223[8] = 0;
}
void h_27(double *state, double *unused, double *out_5418124962952672299) {
   out_5418124962952672299[0] = state[3];
}
void H_27(double *state, double *unused, double *out_8960891406861017466) {
   out_8960891406861017466[0] = 0;
   out_8960891406861017466[1] = 0;
   out_8960891406861017466[2] = 0;
   out_8960891406861017466[3] = 1;
   out_8960891406861017466[4] = 0;
   out_8960891406861017466[5] = 0;
   out_8960891406861017466[6] = 0;
   out_8960891406861017466[7] = 0;
   out_8960891406861017466[8] = 0;
}
void h_29(double *state, double *unused, double *out_107932473598350814) {
   out_107932473598350814[0] = state[1];
}
void H_29(double *state, double *unused, double *out_7247528679991466433) {
   out_7247528679991466433[0] = 0;
   out_7247528679991466433[1] = 1;
   out_7247528679991466433[2] = 0;
   out_7247528679991466433[3] = 0;
   out_7247528679991466433[4] = 0;
   out_7247528679991466433[5] = 0;
   out_7247528679991466433[6] = 0;
   out_7247528679991466433[7] = 0;
   out_7247528679991466433[8] = 0;
}
void h_28(double *state, double *unused, double *out_4972454918016319162) {
   out_4972454918016319162[0] = state[0];
}
void H_28(double *state, double *unused, double *out_2165129662921935859) {
   out_2165129662921935859[0] = 1;
   out_2165129662921935859[1] = 0;
   out_2165129662921935859[2] = 0;
   out_2165129662921935859[3] = 0;
   out_2165129662921935859[4] = 0;
   out_2165129662921935859[5] = 0;
   out_2165129662921935859[6] = 0;
   out_2165129662921935859[7] = 0;
   out_2165129662921935859[8] = 0;
}
void h_31(double *state, double *unused, double *out_1867083087920709821) {
   out_1867083087920709821[0] = state[8];
}
void H_31(double *state, double *unused, double *out_7151104446027908741) {
   out_7151104446027908741[0] = 0;
   out_7151104446027908741[1] = 0;
   out_7151104446027908741[2] = 0;
   out_7151104446027908741[3] = 0;
   out_7151104446027908741[4] = 0;
   out_7151104446027908741[5] = 0;
   out_7151104446027908741[6] = 0;
   out_7151104446027908741[7] = 0;
   out_7151104446027908741[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_7002678419470636064) {
  err_fun(nom_x, delta_x, out_7002678419470636064);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4320393458615466509) {
  inv_err_fun(nom_x, true_x, out_4320393458615466509);
}
void car_H_mod_fun(double *state, double *out_7742013015815190770) {
  H_mod_fun(state, out_7742013015815190770);
}
void car_f_fun(double *state, double dt, double *out_7561579577695376867) {
  f_fun(state,  dt, out_7561579577695376867);
}
void car_F_fun(double *state, double dt, double *out_3970912247205260696) {
  F_fun(state,  dt, out_3970912247205260696);
}
void car_h_25(double *state, double *unused, double *out_416451695291826654) {
  h_25(state, unused, out_416451695291826654);
}
void car_H_25(double *state, double *unused, double *out_4218964377169825622) {
  H_25(state, unused, out_4218964377169825622);
}
void car_h_24(double *state, double *unused, double *out_3590841845749764283) {
  h_24(state, unused, out_3590841845749764283);
}
void car_H_24(double *state, double *unused, double *out_2046314778164326056) {
  H_24(state, unused, out_2046314778164326056);
}
void car_h_30(double *state, double *unused, double *out_6354383531058524282) {
  h_30(state, unused, out_6354383531058524282);
}
void car_H_30(double *state, double *unused, double *out_7311089355048109239) {
  H_30(state, unused, out_7311089355048109239);
}
void car_h_26(double *state, double *unused, double *out_3941861940183054722) {
  h_26(state, unused, out_3941861940183054722);
}
void car_H_26(double *state, double *unused, double *out_7523490346930626223) {
  H_26(state, unused, out_7523490346930626223);
}
void car_h_27(double *state, double *unused, double *out_5418124962952672299) {
  h_27(state, unused, out_5418124962952672299);
}
void car_H_27(double *state, double *unused, double *out_8960891406861017466) {
  H_27(state, unused, out_8960891406861017466);
}
void car_h_29(double *state, double *unused, double *out_107932473598350814) {
  h_29(state, unused, out_107932473598350814);
}
void car_H_29(double *state, double *unused, double *out_7247528679991466433) {
  H_29(state, unused, out_7247528679991466433);
}
void car_h_28(double *state, double *unused, double *out_4972454918016319162) {
  h_28(state, unused, out_4972454918016319162);
}
void car_H_28(double *state, double *unused, double *out_2165129662921935859) {
  H_28(state, unused, out_2165129662921935859);
}
void car_h_31(double *state, double *unused, double *out_1867083087920709821) {
  h_31(state, unused, out_1867083087920709821);
}
void car_H_31(double *state, double *unused, double *out_7151104446027908741) {
  H_31(state, unused, out_7151104446027908741);
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
