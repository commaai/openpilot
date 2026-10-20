#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

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
void err_fun(double *nom_x, double *delta_x, double *out_6382149531773064090) {
   out_6382149531773064090[0] = delta_x[0] + nom_x[0];
   out_6382149531773064090[1] = delta_x[1] + nom_x[1];
   out_6382149531773064090[2] = delta_x[2] + nom_x[2];
   out_6382149531773064090[3] = delta_x[3] + nom_x[3];
   out_6382149531773064090[4] = delta_x[4] + nom_x[4];
   out_6382149531773064090[5] = delta_x[5] + nom_x[5];
   out_6382149531773064090[6] = delta_x[6] + nom_x[6];
   out_6382149531773064090[7] = delta_x[7] + nom_x[7];
   out_6382149531773064090[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_4625467165850258549) {
   out_4625467165850258549[0] = -nom_x[0] + true_x[0];
   out_4625467165850258549[1] = -nom_x[1] + true_x[1];
   out_4625467165850258549[2] = -nom_x[2] + true_x[2];
   out_4625467165850258549[3] = -nom_x[3] + true_x[3];
   out_4625467165850258549[4] = -nom_x[4] + true_x[4];
   out_4625467165850258549[5] = -nom_x[5] + true_x[5];
   out_4625467165850258549[6] = -nom_x[6] + true_x[6];
   out_4625467165850258549[7] = -nom_x[7] + true_x[7];
   out_4625467165850258549[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_1214021453202461484) {
   out_1214021453202461484[0] = 1.0;
   out_1214021453202461484[1] = 0.0;
   out_1214021453202461484[2] = 0.0;
   out_1214021453202461484[3] = 0.0;
   out_1214021453202461484[4] = 0.0;
   out_1214021453202461484[5] = 0.0;
   out_1214021453202461484[6] = 0.0;
   out_1214021453202461484[7] = 0.0;
   out_1214021453202461484[8] = 0.0;
   out_1214021453202461484[9] = 0.0;
   out_1214021453202461484[10] = 1.0;
   out_1214021453202461484[11] = 0.0;
   out_1214021453202461484[12] = 0.0;
   out_1214021453202461484[13] = 0.0;
   out_1214021453202461484[14] = 0.0;
   out_1214021453202461484[15] = 0.0;
   out_1214021453202461484[16] = 0.0;
   out_1214021453202461484[17] = 0.0;
   out_1214021453202461484[18] = 0.0;
   out_1214021453202461484[19] = 0.0;
   out_1214021453202461484[20] = 1.0;
   out_1214021453202461484[21] = 0.0;
   out_1214021453202461484[22] = 0.0;
   out_1214021453202461484[23] = 0.0;
   out_1214021453202461484[24] = 0.0;
   out_1214021453202461484[25] = 0.0;
   out_1214021453202461484[26] = 0.0;
   out_1214021453202461484[27] = 0.0;
   out_1214021453202461484[28] = 0.0;
   out_1214021453202461484[29] = 0.0;
   out_1214021453202461484[30] = 1.0;
   out_1214021453202461484[31] = 0.0;
   out_1214021453202461484[32] = 0.0;
   out_1214021453202461484[33] = 0.0;
   out_1214021453202461484[34] = 0.0;
   out_1214021453202461484[35] = 0.0;
   out_1214021453202461484[36] = 0.0;
   out_1214021453202461484[37] = 0.0;
   out_1214021453202461484[38] = 0.0;
   out_1214021453202461484[39] = 0.0;
   out_1214021453202461484[40] = 1.0;
   out_1214021453202461484[41] = 0.0;
   out_1214021453202461484[42] = 0.0;
   out_1214021453202461484[43] = 0.0;
   out_1214021453202461484[44] = 0.0;
   out_1214021453202461484[45] = 0.0;
   out_1214021453202461484[46] = 0.0;
   out_1214021453202461484[47] = 0.0;
   out_1214021453202461484[48] = 0.0;
   out_1214021453202461484[49] = 0.0;
   out_1214021453202461484[50] = 1.0;
   out_1214021453202461484[51] = 0.0;
   out_1214021453202461484[52] = 0.0;
   out_1214021453202461484[53] = 0.0;
   out_1214021453202461484[54] = 0.0;
   out_1214021453202461484[55] = 0.0;
   out_1214021453202461484[56] = 0.0;
   out_1214021453202461484[57] = 0.0;
   out_1214021453202461484[58] = 0.0;
   out_1214021453202461484[59] = 0.0;
   out_1214021453202461484[60] = 1.0;
   out_1214021453202461484[61] = 0.0;
   out_1214021453202461484[62] = 0.0;
   out_1214021453202461484[63] = 0.0;
   out_1214021453202461484[64] = 0.0;
   out_1214021453202461484[65] = 0.0;
   out_1214021453202461484[66] = 0.0;
   out_1214021453202461484[67] = 0.0;
   out_1214021453202461484[68] = 0.0;
   out_1214021453202461484[69] = 0.0;
   out_1214021453202461484[70] = 1.0;
   out_1214021453202461484[71] = 0.0;
   out_1214021453202461484[72] = 0.0;
   out_1214021453202461484[73] = 0.0;
   out_1214021453202461484[74] = 0.0;
   out_1214021453202461484[75] = 0.0;
   out_1214021453202461484[76] = 0.0;
   out_1214021453202461484[77] = 0.0;
   out_1214021453202461484[78] = 0.0;
   out_1214021453202461484[79] = 0.0;
   out_1214021453202461484[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_5102529219309380439) {
   out_5102529219309380439[0] = state[0];
   out_5102529219309380439[1] = state[1];
   out_5102529219309380439[2] = state[2];
   out_5102529219309380439[3] = state[3];
   out_5102529219309380439[4] = state[4];
   out_5102529219309380439[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_5102529219309380439[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_5102529219309380439[7] = state[7];
   out_5102529219309380439[8] = state[8];
}
void F_fun(double *state, double dt, double *out_1049561808137205989) {
   out_1049561808137205989[0] = 1;
   out_1049561808137205989[1] = 0;
   out_1049561808137205989[2] = 0;
   out_1049561808137205989[3] = 0;
   out_1049561808137205989[4] = 0;
   out_1049561808137205989[5] = 0;
   out_1049561808137205989[6] = 0;
   out_1049561808137205989[7] = 0;
   out_1049561808137205989[8] = 0;
   out_1049561808137205989[9] = 0;
   out_1049561808137205989[10] = 1;
   out_1049561808137205989[11] = 0;
   out_1049561808137205989[12] = 0;
   out_1049561808137205989[13] = 0;
   out_1049561808137205989[14] = 0;
   out_1049561808137205989[15] = 0;
   out_1049561808137205989[16] = 0;
   out_1049561808137205989[17] = 0;
   out_1049561808137205989[18] = 0;
   out_1049561808137205989[19] = 0;
   out_1049561808137205989[20] = 1;
   out_1049561808137205989[21] = 0;
   out_1049561808137205989[22] = 0;
   out_1049561808137205989[23] = 0;
   out_1049561808137205989[24] = 0;
   out_1049561808137205989[25] = 0;
   out_1049561808137205989[26] = 0;
   out_1049561808137205989[27] = 0;
   out_1049561808137205989[28] = 0;
   out_1049561808137205989[29] = 0;
   out_1049561808137205989[30] = 1;
   out_1049561808137205989[31] = 0;
   out_1049561808137205989[32] = 0;
   out_1049561808137205989[33] = 0;
   out_1049561808137205989[34] = 0;
   out_1049561808137205989[35] = 0;
   out_1049561808137205989[36] = 0;
   out_1049561808137205989[37] = 0;
   out_1049561808137205989[38] = 0;
   out_1049561808137205989[39] = 0;
   out_1049561808137205989[40] = 1;
   out_1049561808137205989[41] = 0;
   out_1049561808137205989[42] = 0;
   out_1049561808137205989[43] = 0;
   out_1049561808137205989[44] = 0;
   out_1049561808137205989[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_1049561808137205989[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_1049561808137205989[47] = -dt*stiffness_front*state[0]/state[1];
   out_1049561808137205989[48] = -dt*stiffness_front*state[0]/state[1];
   out_1049561808137205989[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_1049561808137205989[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_1049561808137205989[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_1049561808137205989[52] = dt*stiffness_front*state[0]/state[1];
   out_1049561808137205989[53] = -9.8100000000000005*dt;
   out_1049561808137205989[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_1049561808137205989[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_1049561808137205989[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1049561808137205989[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1049561808137205989[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_1049561808137205989[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_1049561808137205989[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_1049561808137205989[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1049561808137205989[62] = 0;
   out_1049561808137205989[63] = 0;
   out_1049561808137205989[64] = 0;
   out_1049561808137205989[65] = 0;
   out_1049561808137205989[66] = 0;
   out_1049561808137205989[67] = 0;
   out_1049561808137205989[68] = 0;
   out_1049561808137205989[69] = 0;
   out_1049561808137205989[70] = 1;
   out_1049561808137205989[71] = 0;
   out_1049561808137205989[72] = 0;
   out_1049561808137205989[73] = 0;
   out_1049561808137205989[74] = 0;
   out_1049561808137205989[75] = 0;
   out_1049561808137205989[76] = 0;
   out_1049561808137205989[77] = 0;
   out_1049561808137205989[78] = 0;
   out_1049561808137205989[79] = 0;
   out_1049561808137205989[80] = 1;
}
void h_25(double *state, double *unused, double *out_4688551692223537173) {
   out_4688551692223537173[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1719355895402758196) {
   out_1719355895402758196[0] = 0;
   out_1719355895402758196[1] = 0;
   out_1719355895402758196[2] = 0;
   out_1719355895402758196[3] = 0;
   out_1719355895402758196[4] = 0;
   out_1719355895402758196[5] = 0;
   out_1719355895402758196[6] = 1;
   out_1719355895402758196[7] = 0;
   out_1719355895402758196[8] = 0;
}
void h_24(double *state, double *unused, double *out_8426987597188806191) {
   out_8426987597188806191[0] = state[4];
   out_8426987597188806191[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8294927701994276297) {
   out_8294927701994276297[0] = 0;
   out_8294927701994276297[1] = 0;
   out_8294927701994276297[2] = 0;
   out_8294927701994276297[3] = 0;
   out_8294927701994276297[4] = 1;
   out_8294927701994276297[5] = 0;
   out_8294927701994276297[6] = 0;
   out_8294927701994276297[7] = 0;
   out_8294927701994276297[8] = 0;
   out_8294927701994276297[9] = 0;
   out_8294927701994276297[10] = 0;
   out_8294927701994276297[11] = 0;
   out_8294927701994276297[12] = 0;
   out_8294927701994276297[13] = 0;
   out_8294927701994276297[14] = 1;
   out_8294927701994276297[15] = 0;
   out_8294927701994276297[16] = 0;
   out_8294927701994276297[17] = 0;
}
void h_30(double *state, double *unused, double *out_5865252886601491084) {
   out_5865252886601491084[0] = state[4];
}
void H_30(double *state, double *unused, double *out_8636046236894374951) {
   out_8636046236894374951[0] = 0;
   out_8636046236894374951[1] = 0;
   out_8636046236894374951[2] = 0;
   out_8636046236894374951[3] = 0;
   out_8636046236894374951[4] = 1;
   out_8636046236894374951[5] = 0;
   out_8636046236894374951[6] = 0;
   out_8636046236894374951[7] = 0;
   out_8636046236894374951[8] = 0;
}
void h_26(double *state, double *unused, double *out_3599522015101219767) {
   out_3599522015101219767[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5023881865163558797) {
   out_5023881865163558797[0] = 0;
   out_5023881865163558797[1] = 0;
   out_5023881865163558797[2] = 0;
   out_5023881865163558797[3] = 0;
   out_5023881865163558797[4] = 0;
   out_5023881865163558797[5] = 0;
   out_5023881865163558797[6] = 0;
   out_5023881865163558797[7] = 1;
   out_5023881865163558797[8] = 0;
}
void h_27(double *state, double *unused, double *out_7725337743667869858) {
   out_7725337743667869858[0] = state[3];
}
void H_27(double *state, double *unused, double *out_6461282925093950040) {
   out_6461282925093950040[0] = 0;
   out_6461282925093950040[1] = 0;
   out_6461282925093950040[2] = 0;
   out_6461282925093950040[3] = 1;
   out_6461282925093950040[4] = 0;
   out_6461282925093950040[5] = 0;
   out_6461282925093950040[6] = 0;
   out_6461282925093950040[7] = 0;
   out_6461282925093950040[8] = 0;
}
void h_29(double *state, double *unused, double *out_4932146013796726190) {
   out_4932146013796726190[0] = state[1];
}
void H_29(double *state, double *unused, double *out_9146277581208767135) {
   out_9146277581208767135[0] = 0;
   out_9146277581208767135[1] = 1;
   out_9146277581208767135[2] = 0;
   out_9146277581208767135[3] = 0;
   out_9146277581208767135[4] = 0;
   out_9146277581208767135[5] = 0;
   out_9146277581208767135[6] = 0;
   out_9146277581208767135[7] = 0;
   out_9146277581208767135[8] = 0;
}
void h_28(double *state, double *unused, double *out_2499430527832544911) {
   out_2499430527832544911[0] = state[0];
}
void H_28(double *state, double *unused, double *out_4063878564139236561) {
   out_4063878564139236561[0] = 1;
   out_4063878564139236561[1] = 0;
   out_4063878564139236561[2] = 0;
   out_4063878564139236561[3] = 0;
   out_4063878564139236561[4] = 0;
   out_4063878564139236561[5] = 0;
   out_4063878564139236561[6] = 0;
   out_4063878564139236561[7] = 0;
   out_4063878564139236561[8] = 0;
}
void h_31(double *state, double *unused, double *out_2082283534126813763) {
   out_2082283534126813763[0] = state[8];
}
void H_31(double *state, double *unused, double *out_8796031145914575449) {
   out_8796031145914575449[0] = 0;
   out_8796031145914575449[1] = 0;
   out_8796031145914575449[2] = 0;
   out_8796031145914575449[3] = 0;
   out_8796031145914575449[4] = 0;
   out_8796031145914575449[5] = 0;
   out_8796031145914575449[6] = 0;
   out_8796031145914575449[7] = 0;
   out_8796031145914575449[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6382149531773064090) {
  err_fun(nom_x, delta_x, out_6382149531773064090);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4625467165850258549) {
  inv_err_fun(nom_x, true_x, out_4625467165850258549);
}
void car_H_mod_fun(double *state, double *out_1214021453202461484) {
  H_mod_fun(state, out_1214021453202461484);
}
void car_f_fun(double *state, double dt, double *out_5102529219309380439) {
  f_fun(state,  dt, out_5102529219309380439);
}
void car_F_fun(double *state, double dt, double *out_1049561808137205989) {
  F_fun(state,  dt, out_1049561808137205989);
}
void car_h_25(double *state, double *unused, double *out_4688551692223537173) {
  h_25(state, unused, out_4688551692223537173);
}
void car_H_25(double *state, double *unused, double *out_1719355895402758196) {
  H_25(state, unused, out_1719355895402758196);
}
void car_h_24(double *state, double *unused, double *out_8426987597188806191) {
  h_24(state, unused, out_8426987597188806191);
}
void car_H_24(double *state, double *unused, double *out_8294927701994276297) {
  H_24(state, unused, out_8294927701994276297);
}
void car_h_30(double *state, double *unused, double *out_5865252886601491084) {
  h_30(state, unused, out_5865252886601491084);
}
void car_H_30(double *state, double *unused, double *out_8636046236894374951) {
  H_30(state, unused, out_8636046236894374951);
}
void car_h_26(double *state, double *unused, double *out_3599522015101219767) {
  h_26(state, unused, out_3599522015101219767);
}
void car_H_26(double *state, double *unused, double *out_5023881865163558797) {
  H_26(state, unused, out_5023881865163558797);
}
void car_h_27(double *state, double *unused, double *out_7725337743667869858) {
  h_27(state, unused, out_7725337743667869858);
}
void car_H_27(double *state, double *unused, double *out_6461282925093950040) {
  H_27(state, unused, out_6461282925093950040);
}
void car_h_29(double *state, double *unused, double *out_4932146013796726190) {
  h_29(state, unused, out_4932146013796726190);
}
void car_H_29(double *state, double *unused, double *out_9146277581208767135) {
  H_29(state, unused, out_9146277581208767135);
}
void car_h_28(double *state, double *unused, double *out_2499430527832544911) {
  h_28(state, unused, out_2499430527832544911);
}
void car_H_28(double *state, double *unused, double *out_4063878564139236561) {
  H_28(state, unused, out_4063878564139236561);
}
void car_h_31(double *state, double *unused, double *out_2082283534126813763) {
  h_31(state, unused, out_2082283534126813763);
}
void car_H_31(double *state, double *unused, double *out_8796031145914575449) {
  H_31(state, unused, out_8796031145914575449);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
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
