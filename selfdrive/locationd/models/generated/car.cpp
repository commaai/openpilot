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
void err_fun(double *nom_x, double *delta_x, double *out_5154743392088968873) {
   out_5154743392088968873[0] = delta_x[0] + nom_x[0];
   out_5154743392088968873[1] = delta_x[1] + nom_x[1];
   out_5154743392088968873[2] = delta_x[2] + nom_x[2];
   out_5154743392088968873[3] = delta_x[3] + nom_x[3];
   out_5154743392088968873[4] = delta_x[4] + nom_x[4];
   out_5154743392088968873[5] = delta_x[5] + nom_x[5];
   out_5154743392088968873[6] = delta_x[6] + nom_x[6];
   out_5154743392088968873[7] = delta_x[7] + nom_x[7];
   out_5154743392088968873[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6484687437990115350) {
   out_6484687437990115350[0] = -nom_x[0] + true_x[0];
   out_6484687437990115350[1] = -nom_x[1] + true_x[1];
   out_6484687437990115350[2] = -nom_x[2] + true_x[2];
   out_6484687437990115350[3] = -nom_x[3] + true_x[3];
   out_6484687437990115350[4] = -nom_x[4] + true_x[4];
   out_6484687437990115350[5] = -nom_x[5] + true_x[5];
   out_6484687437990115350[6] = -nom_x[6] + true_x[6];
   out_6484687437990115350[7] = -nom_x[7] + true_x[7];
   out_6484687437990115350[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_2724143367716012553) {
   out_2724143367716012553[0] = 1.0;
   out_2724143367716012553[1] = 0.0;
   out_2724143367716012553[2] = 0.0;
   out_2724143367716012553[3] = 0.0;
   out_2724143367716012553[4] = 0.0;
   out_2724143367716012553[5] = 0.0;
   out_2724143367716012553[6] = 0.0;
   out_2724143367716012553[7] = 0.0;
   out_2724143367716012553[8] = 0.0;
   out_2724143367716012553[9] = 0.0;
   out_2724143367716012553[10] = 1.0;
   out_2724143367716012553[11] = 0.0;
   out_2724143367716012553[12] = 0.0;
   out_2724143367716012553[13] = 0.0;
   out_2724143367716012553[14] = 0.0;
   out_2724143367716012553[15] = 0.0;
   out_2724143367716012553[16] = 0.0;
   out_2724143367716012553[17] = 0.0;
   out_2724143367716012553[18] = 0.0;
   out_2724143367716012553[19] = 0.0;
   out_2724143367716012553[20] = 1.0;
   out_2724143367716012553[21] = 0.0;
   out_2724143367716012553[22] = 0.0;
   out_2724143367716012553[23] = 0.0;
   out_2724143367716012553[24] = 0.0;
   out_2724143367716012553[25] = 0.0;
   out_2724143367716012553[26] = 0.0;
   out_2724143367716012553[27] = 0.0;
   out_2724143367716012553[28] = 0.0;
   out_2724143367716012553[29] = 0.0;
   out_2724143367716012553[30] = 1.0;
   out_2724143367716012553[31] = 0.0;
   out_2724143367716012553[32] = 0.0;
   out_2724143367716012553[33] = 0.0;
   out_2724143367716012553[34] = 0.0;
   out_2724143367716012553[35] = 0.0;
   out_2724143367716012553[36] = 0.0;
   out_2724143367716012553[37] = 0.0;
   out_2724143367716012553[38] = 0.0;
   out_2724143367716012553[39] = 0.0;
   out_2724143367716012553[40] = 1.0;
   out_2724143367716012553[41] = 0.0;
   out_2724143367716012553[42] = 0.0;
   out_2724143367716012553[43] = 0.0;
   out_2724143367716012553[44] = 0.0;
   out_2724143367716012553[45] = 0.0;
   out_2724143367716012553[46] = 0.0;
   out_2724143367716012553[47] = 0.0;
   out_2724143367716012553[48] = 0.0;
   out_2724143367716012553[49] = 0.0;
   out_2724143367716012553[50] = 1.0;
   out_2724143367716012553[51] = 0.0;
   out_2724143367716012553[52] = 0.0;
   out_2724143367716012553[53] = 0.0;
   out_2724143367716012553[54] = 0.0;
   out_2724143367716012553[55] = 0.0;
   out_2724143367716012553[56] = 0.0;
   out_2724143367716012553[57] = 0.0;
   out_2724143367716012553[58] = 0.0;
   out_2724143367716012553[59] = 0.0;
   out_2724143367716012553[60] = 1.0;
   out_2724143367716012553[61] = 0.0;
   out_2724143367716012553[62] = 0.0;
   out_2724143367716012553[63] = 0.0;
   out_2724143367716012553[64] = 0.0;
   out_2724143367716012553[65] = 0.0;
   out_2724143367716012553[66] = 0.0;
   out_2724143367716012553[67] = 0.0;
   out_2724143367716012553[68] = 0.0;
   out_2724143367716012553[69] = 0.0;
   out_2724143367716012553[70] = 1.0;
   out_2724143367716012553[71] = 0.0;
   out_2724143367716012553[72] = 0.0;
   out_2724143367716012553[73] = 0.0;
   out_2724143367716012553[74] = 0.0;
   out_2724143367716012553[75] = 0.0;
   out_2724143367716012553[76] = 0.0;
   out_2724143367716012553[77] = 0.0;
   out_2724143367716012553[78] = 0.0;
   out_2724143367716012553[79] = 0.0;
   out_2724143367716012553[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_4240112907484928210) {
   out_4240112907484928210[0] = state[0];
   out_4240112907484928210[1] = state[1];
   out_4240112907484928210[2] = state[2];
   out_4240112907484928210[3] = state[3];
   out_4240112907484928210[4] = state[4];
   out_4240112907484928210[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_4240112907484928210[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_4240112907484928210[7] = state[7];
   out_4240112907484928210[8] = state[8];
}
void F_fun(double *state, double dt, double *out_7340475373084997924) {
   out_7340475373084997924[0] = 1;
   out_7340475373084997924[1] = 0;
   out_7340475373084997924[2] = 0;
   out_7340475373084997924[3] = 0;
   out_7340475373084997924[4] = 0;
   out_7340475373084997924[5] = 0;
   out_7340475373084997924[6] = 0;
   out_7340475373084997924[7] = 0;
   out_7340475373084997924[8] = 0;
   out_7340475373084997924[9] = 0;
   out_7340475373084997924[10] = 1;
   out_7340475373084997924[11] = 0;
   out_7340475373084997924[12] = 0;
   out_7340475373084997924[13] = 0;
   out_7340475373084997924[14] = 0;
   out_7340475373084997924[15] = 0;
   out_7340475373084997924[16] = 0;
   out_7340475373084997924[17] = 0;
   out_7340475373084997924[18] = 0;
   out_7340475373084997924[19] = 0;
   out_7340475373084997924[20] = 1;
   out_7340475373084997924[21] = 0;
   out_7340475373084997924[22] = 0;
   out_7340475373084997924[23] = 0;
   out_7340475373084997924[24] = 0;
   out_7340475373084997924[25] = 0;
   out_7340475373084997924[26] = 0;
   out_7340475373084997924[27] = 0;
   out_7340475373084997924[28] = 0;
   out_7340475373084997924[29] = 0;
   out_7340475373084997924[30] = 1;
   out_7340475373084997924[31] = 0;
   out_7340475373084997924[32] = 0;
   out_7340475373084997924[33] = 0;
   out_7340475373084997924[34] = 0;
   out_7340475373084997924[35] = 0;
   out_7340475373084997924[36] = 0;
   out_7340475373084997924[37] = 0;
   out_7340475373084997924[38] = 0;
   out_7340475373084997924[39] = 0;
   out_7340475373084997924[40] = 1;
   out_7340475373084997924[41] = 0;
   out_7340475373084997924[42] = 0;
   out_7340475373084997924[43] = 0;
   out_7340475373084997924[44] = 0;
   out_7340475373084997924[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_7340475373084997924[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_7340475373084997924[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_7340475373084997924[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_7340475373084997924[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_7340475373084997924[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_7340475373084997924[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_7340475373084997924[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_7340475373084997924[53] = -9.8000000000000007*dt;
   out_7340475373084997924[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_7340475373084997924[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_7340475373084997924[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7340475373084997924[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7340475373084997924[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_7340475373084997924[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_7340475373084997924[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_7340475373084997924[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7340475373084997924[62] = 0;
   out_7340475373084997924[63] = 0;
   out_7340475373084997924[64] = 0;
   out_7340475373084997924[65] = 0;
   out_7340475373084997924[66] = 0;
   out_7340475373084997924[67] = 0;
   out_7340475373084997924[68] = 0;
   out_7340475373084997924[69] = 0;
   out_7340475373084997924[70] = 1;
   out_7340475373084997924[71] = 0;
   out_7340475373084997924[72] = 0;
   out_7340475373084997924[73] = 0;
   out_7340475373084997924[74] = 0;
   out_7340475373084997924[75] = 0;
   out_7340475373084997924[76] = 0;
   out_7340475373084997924[77] = 0;
   out_7340475373084997924[78] = 0;
   out_7340475373084997924[79] = 0;
   out_7340475373084997924[80] = 1;
}
void h_25(double *state, double *unused, double *out_9000275207519044529) {
   out_9000275207519044529[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7801262673019679258) {
   out_7801262673019679258[0] = 0;
   out_7801262673019679258[1] = 0;
   out_7801262673019679258[2] = 0;
   out_7801262673019679258[3] = 0;
   out_7801262673019679258[4] = 0;
   out_7801262673019679258[5] = 0;
   out_7801262673019679258[6] = 1;
   out_7801262673019679258[7] = 0;
   out_7801262673019679258[8] = 0;
}
void h_24(double *state, double *unused, double *out_1494680429397070930) {
   out_1494680429397070930[0] = state[4];
   out_1494680429397070930[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6388234424981270421) {
   out_6388234424981270421[0] = 0;
   out_6388234424981270421[1] = 0;
   out_6388234424981270421[2] = 0;
   out_6388234424981270421[3] = 0;
   out_6388234424981270421[4] = 1;
   out_6388234424981270421[5] = 0;
   out_6388234424981270421[6] = 0;
   out_6388234424981270421[7] = 0;
   out_6388234424981270421[8] = 0;
   out_6388234424981270421[9] = 0;
   out_6388234424981270421[10] = 0;
   out_6388234424981270421[11] = 0;
   out_6388234424981270421[12] = 0;
   out_6388234424981270421[13] = 0;
   out_6388234424981270421[14] = 1;
   out_6388234424981270421[15] = 0;
   out_6388234424981270421[16] = 0;
   out_6388234424981270421[17] = 0;
}
void h_30(double *state, double *unused, double *out_360397730800502645) {
   out_360397730800502645[0] = state[4];
}
void H_30(double *state, double *unused, double *out_6117785070562264160) {
   out_6117785070562264160[0] = 0;
   out_6117785070562264160[1] = 0;
   out_6117785070562264160[2] = 0;
   out_6117785070562264160[3] = 0;
   out_6117785070562264160[4] = 1;
   out_6117785070562264160[5] = 0;
   out_6117785070562264160[6] = 0;
   out_6117785070562264160[7] = 0;
   out_6117785070562264160[8] = 0;
}
void h_26(double *state, double *unused, double *out_5845481762141335100) {
   out_5845481762141335100[0] = state[7];
}
void H_26(double *state, double *unused, double *out_6903978081815816134) {
   out_6903978081815816134[0] = 0;
   out_6903978081815816134[1] = 0;
   out_6903978081815816134[2] = 0;
   out_6903978081815816134[3] = 0;
   out_6903978081815816134[4] = 0;
   out_6903978081815816134[5] = 0;
   out_6903978081815816134[6] = 0;
   out_6903978081815816134[7] = 1;
   out_6903978081815816134[8] = 0;
}
void h_27(double *state, double *unused, double *out_5963489156074711844) {
   out_5963489156074711844[0] = state[3];
}
void H_27(double *state, double *unused, double *out_3943021758761839249) {
   out_3943021758761839249[0] = 0;
   out_3943021758761839249[1] = 0;
   out_3943021758761839249[2] = 0;
   out_3943021758761839249[3] = 1;
   out_3943021758761839249[4] = 0;
   out_3943021758761839249[5] = 0;
   out_3943021758761839249[6] = 0;
   out_3943021758761839249[7] = 0;
   out_3943021758761839249[8] = 0;
}
void h_29(double *state, double *unused, double *out_5505287793916150182) {
   out_5505287793916150182[0] = state[1];
}
void H_29(double *state, double *unused, double *out_6628016414876656344) {
   out_6628016414876656344[0] = 0;
   out_6628016414876656344[1] = 1;
   out_6628016414876656344[2] = 0;
   out_6628016414876656344[3] = 0;
   out_6628016414876656344[4] = 0;
   out_6628016414876656344[5] = 0;
   out_6628016414876656344[6] = 0;
   out_6628016414876656344[7] = 0;
   out_6628016414876656344[8] = 0;
}
void h_28(double *state, double *unused, double *out_7481581649973994127) {
   out_7481581649973994127[0] = state[0];
}
void H_28(double *state, double *unused, double *out_8591646686441982595) {
   out_8591646686441982595[0] = 1;
   out_8591646686441982595[1] = 0;
   out_8591646686441982595[2] = 0;
   out_8591646686441982595[3] = 0;
   out_8591646686441982595[4] = 0;
   out_8591646686441982595[5] = 0;
   out_8591646686441982595[6] = 0;
   out_8591646686441982595[7] = 0;
   out_8591646686441982595[8] = 0;
}
void h_31(double *state, double *unused, double *out_8725081145234538640) {
   out_8725081145234538640[0] = state[8];
}
void H_31(double *state, double *unused, double *out_6277769979582464658) {
   out_6277769979582464658[0] = 0;
   out_6277769979582464658[1] = 0;
   out_6277769979582464658[2] = 0;
   out_6277769979582464658[3] = 0;
   out_6277769979582464658[4] = 0;
   out_6277769979582464658[5] = 0;
   out_6277769979582464658[6] = 0;
   out_6277769979582464658[7] = 0;
   out_6277769979582464658[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_5154743392088968873) {
  err_fun(nom_x, delta_x, out_5154743392088968873);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6484687437990115350) {
  inv_err_fun(nom_x, true_x, out_6484687437990115350);
}
void car_H_mod_fun(double *state, double *out_2724143367716012553) {
  H_mod_fun(state, out_2724143367716012553);
}
void car_f_fun(double *state, double dt, double *out_4240112907484928210) {
  f_fun(state,  dt, out_4240112907484928210);
}
void car_F_fun(double *state, double dt, double *out_7340475373084997924) {
  F_fun(state,  dt, out_7340475373084997924);
}
void car_h_25(double *state, double *unused, double *out_9000275207519044529) {
  h_25(state, unused, out_9000275207519044529);
}
void car_H_25(double *state, double *unused, double *out_7801262673019679258) {
  H_25(state, unused, out_7801262673019679258);
}
void car_h_24(double *state, double *unused, double *out_1494680429397070930) {
  h_24(state, unused, out_1494680429397070930);
}
void car_H_24(double *state, double *unused, double *out_6388234424981270421) {
  H_24(state, unused, out_6388234424981270421);
}
void car_h_30(double *state, double *unused, double *out_360397730800502645) {
  h_30(state, unused, out_360397730800502645);
}
void car_H_30(double *state, double *unused, double *out_6117785070562264160) {
  H_30(state, unused, out_6117785070562264160);
}
void car_h_26(double *state, double *unused, double *out_5845481762141335100) {
  h_26(state, unused, out_5845481762141335100);
}
void car_H_26(double *state, double *unused, double *out_6903978081815816134) {
  H_26(state, unused, out_6903978081815816134);
}
void car_h_27(double *state, double *unused, double *out_5963489156074711844) {
  h_27(state, unused, out_5963489156074711844);
}
void car_H_27(double *state, double *unused, double *out_3943021758761839249) {
  H_27(state, unused, out_3943021758761839249);
}
void car_h_29(double *state, double *unused, double *out_5505287793916150182) {
  h_29(state, unused, out_5505287793916150182);
}
void car_H_29(double *state, double *unused, double *out_6628016414876656344) {
  H_29(state, unused, out_6628016414876656344);
}
void car_h_28(double *state, double *unused, double *out_7481581649973994127) {
  h_28(state, unused, out_7481581649973994127);
}
void car_H_28(double *state, double *unused, double *out_8591646686441982595) {
  H_28(state, unused, out_8591646686441982595);
}
void car_h_31(double *state, double *unused, double *out_8725081145234538640) {
  h_31(state, unused, out_8725081145234538640);
}
void car_H_31(double *state, double *unused, double *out_6277769979582464658) {
  H_31(state, unused, out_6277769979582464658);
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
