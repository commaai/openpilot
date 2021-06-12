#include "car.h"

namespace {
#define DIM 8
#define EDIM 8
#define MEDIM 8
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
const static double MAHA_THRESH_28 = 5.991464547107981;

/******************************************************************************
 *                      Code generated with sympy 1.7.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_3157583653505174283) {
   out_3157583653505174283[0] = delta_x[0] + nom_x[0];
   out_3157583653505174283[1] = delta_x[1] + nom_x[1];
   out_3157583653505174283[2] = delta_x[2] + nom_x[2];
   out_3157583653505174283[3] = delta_x[3] + nom_x[3];
   out_3157583653505174283[4] = delta_x[4] + nom_x[4];
   out_3157583653505174283[5] = delta_x[5] + nom_x[5];
   out_3157583653505174283[6] = delta_x[6] + nom_x[6];
   out_3157583653505174283[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5329952054111744571) {
   out_5329952054111744571[0] = -nom_x[0] + true_x[0];
   out_5329952054111744571[1] = -nom_x[1] + true_x[1];
   out_5329952054111744571[2] = -nom_x[2] + true_x[2];
   out_5329952054111744571[3] = -nom_x[3] + true_x[3];
   out_5329952054111744571[4] = -nom_x[4] + true_x[4];
   out_5329952054111744571[5] = -nom_x[5] + true_x[5];
   out_5329952054111744571[6] = -nom_x[6] + true_x[6];
   out_5329952054111744571[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_8803128394076082262) {
   out_8803128394076082262[0] = 1.0;
   out_8803128394076082262[1] = 0.0;
   out_8803128394076082262[2] = 0.0;
   out_8803128394076082262[3] = 0.0;
   out_8803128394076082262[4] = 0.0;
   out_8803128394076082262[5] = 0.0;
   out_8803128394076082262[6] = 0.0;
   out_8803128394076082262[7] = 0.0;
   out_8803128394076082262[8] = 0.0;
   out_8803128394076082262[9] = 1.0;
   out_8803128394076082262[10] = 0.0;
   out_8803128394076082262[11] = 0.0;
   out_8803128394076082262[12] = 0.0;
   out_8803128394076082262[13] = 0.0;
   out_8803128394076082262[14] = 0.0;
   out_8803128394076082262[15] = 0.0;
   out_8803128394076082262[16] = 0.0;
   out_8803128394076082262[17] = 0.0;
   out_8803128394076082262[18] = 1.0;
   out_8803128394076082262[19] = 0.0;
   out_8803128394076082262[20] = 0.0;
   out_8803128394076082262[21] = 0.0;
   out_8803128394076082262[22] = 0.0;
   out_8803128394076082262[23] = 0.0;
   out_8803128394076082262[24] = 0.0;
   out_8803128394076082262[25] = 0.0;
   out_8803128394076082262[26] = 0.0;
   out_8803128394076082262[27] = 1.0;
   out_8803128394076082262[28] = 0.0;
   out_8803128394076082262[29] = 0.0;
   out_8803128394076082262[30] = 0.0;
   out_8803128394076082262[31] = 0.0;
   out_8803128394076082262[32] = 0.0;
   out_8803128394076082262[33] = 0.0;
   out_8803128394076082262[34] = 0.0;
   out_8803128394076082262[35] = 0.0;
   out_8803128394076082262[36] = 1.0;
   out_8803128394076082262[37] = 0.0;
   out_8803128394076082262[38] = 0.0;
   out_8803128394076082262[39] = 0.0;
   out_8803128394076082262[40] = 0.0;
   out_8803128394076082262[41] = 0.0;
   out_8803128394076082262[42] = 0.0;
   out_8803128394076082262[43] = 0.0;
   out_8803128394076082262[44] = 0.0;
   out_8803128394076082262[45] = 1.0;
   out_8803128394076082262[46] = 0.0;
   out_8803128394076082262[47] = 0.0;
   out_8803128394076082262[48] = 0.0;
   out_8803128394076082262[49] = 0.0;
   out_8803128394076082262[50] = 0.0;
   out_8803128394076082262[51] = 0.0;
   out_8803128394076082262[52] = 0.0;
   out_8803128394076082262[53] = 0.0;
   out_8803128394076082262[54] = 1.0;
   out_8803128394076082262[55] = 0.0;
   out_8803128394076082262[56] = 0.0;
   out_8803128394076082262[57] = 0.0;
   out_8803128394076082262[58] = 0.0;
   out_8803128394076082262[59] = 0.0;
   out_8803128394076082262[60] = 0.0;
   out_8803128394076082262[61] = 0.0;
   out_8803128394076082262[62] = 0.0;
   out_8803128394076082262[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_5487969453112289401) {
   out_5487969453112289401[0] = state[0];
   out_5487969453112289401[1] = state[1];
   out_5487969453112289401[2] = state[2];
   out_5487969453112289401[3] = state[3];
   out_5487969453112289401[4] = state[4];
   out_5487969453112289401[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_5487969453112289401[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_5487969453112289401[7] = state[7];
}
void F_fun(double *state, double dt, double *out_249163952273201207) {
   out_249163952273201207[0] = 1;
   out_249163952273201207[1] = 0;
   out_249163952273201207[2] = 0;
   out_249163952273201207[3] = 0;
   out_249163952273201207[4] = 0;
   out_249163952273201207[5] = 0;
   out_249163952273201207[6] = 0;
   out_249163952273201207[7] = 0;
   out_249163952273201207[8] = 0;
   out_249163952273201207[9] = 1;
   out_249163952273201207[10] = 0;
   out_249163952273201207[11] = 0;
   out_249163952273201207[12] = 0;
   out_249163952273201207[13] = 0;
   out_249163952273201207[14] = 0;
   out_249163952273201207[15] = 0;
   out_249163952273201207[16] = 0;
   out_249163952273201207[17] = 0;
   out_249163952273201207[18] = 1;
   out_249163952273201207[19] = 0;
   out_249163952273201207[20] = 0;
   out_249163952273201207[21] = 0;
   out_249163952273201207[22] = 0;
   out_249163952273201207[23] = 0;
   out_249163952273201207[24] = 0;
   out_249163952273201207[25] = 0;
   out_249163952273201207[26] = 0;
   out_249163952273201207[27] = 1;
   out_249163952273201207[28] = 0;
   out_249163952273201207[29] = 0;
   out_249163952273201207[30] = 0;
   out_249163952273201207[31] = 0;
   out_249163952273201207[32] = 0;
   out_249163952273201207[33] = 0;
   out_249163952273201207[34] = 0;
   out_249163952273201207[35] = 0;
   out_249163952273201207[36] = 1;
   out_249163952273201207[37] = 0;
   out_249163952273201207[38] = 0;
   out_249163952273201207[39] = 0;
   out_249163952273201207[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_249163952273201207[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_249163952273201207[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_249163952273201207[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_249163952273201207[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_249163952273201207[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_249163952273201207[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_249163952273201207[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_249163952273201207[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_249163952273201207[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_249163952273201207[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_249163952273201207[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_249163952273201207[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_249163952273201207[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_249163952273201207[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_249163952273201207[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_249163952273201207[56] = 0;
   out_249163952273201207[57] = 0;
   out_249163952273201207[58] = 0;
   out_249163952273201207[59] = 0;
   out_249163952273201207[60] = 0;
   out_249163952273201207[61] = 0;
   out_249163952273201207[62] = 0;
   out_249163952273201207[63] = 1;
}
void h_25(double *state, double *unused, double *out_1166715342447388350) {
   out_1166715342447388350[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7931236138849448777) {
   out_7931236138849448777[0] = 0;
   out_7931236138849448777[1] = 0;
   out_7931236138849448777[2] = 0;
   out_7931236138849448777[3] = 0;
   out_7931236138849448777[4] = 0;
   out_7931236138849448777[5] = 0;
   out_7931236138849448777[6] = 1;
   out_7931236138849448777[7] = 0;
}
void h_24(double *state, double *unused, double *out_3877885127749003657) {
   out_3877885127749003657[0] = state[4];
   out_3877885127749003657[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6235472662875167742) {
   out_6235472662875167742[0] = 0;
   out_6235472662875167742[1] = 0;
   out_6235472662875167742[2] = 0;
   out_6235472662875167742[3] = 0;
   out_6235472662875167742[4] = 1;
   out_6235472662875167742[5] = 0;
   out_6235472662875167742[6] = 0;
   out_6235472662875167742[7] = 0;
   out_6235472662875167742[8] = 0;
   out_6235472662875167742[9] = 0;
   out_6235472662875167742[10] = 0;
   out_6235472662875167742[11] = 0;
   out_6235472662875167742[12] = 0;
   out_6235472662875167742[13] = 1;
   out_6235472662875167742[14] = 0;
   out_6235472662875167742[15] = 0;
}
void h_30(double *state, double *unused, double *out_891521280162882461) {
   out_891521280162882461[0] = state[4];
}
void H_30(double *state, double *unused, double *out_1682662772099734503) {
   out_1682662772099734503[0] = 0;
   out_1682662772099734503[1] = 0;
   out_1682662772099734503[2] = 0;
   out_1682662772099734503[3] = 0;
   out_1682662772099734503[4] = 1;
   out_1682662772099734503[5] = 0;
   out_1682662772099734503[6] = 0;
   out_1682662772099734503[7] = 0;
}
void h_26(double *state, double *unused, double *out_6111241423630183318) {
   out_6111241423630183318[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4810914779920893566) {
   out_4810914779920893566[0] = 0;
   out_4810914779920893566[1] = 0;
   out_4810914779920893566[2] = 0;
   out_4810914779920893566[3] = 0;
   out_4810914779920893566[4] = 0;
   out_4810914779920893566[5] = 0;
   out_4810914779920893566[6] = 0;
   out_4810914779920893566[7] = 1;
}
void h_27(double *state, double *unused, double *out_4711570145111326738) {
   out_4711570145111326738[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2970244759936359815) {
   out_2970244759936359815[0] = 0;
   out_2970244759936359815[1] = 0;
   out_2970244759936359815[2] = 0;
   out_2970244759936359815[3] = 1;
   out_2970244759936359815[4] = 0;
   out_2970244759936359815[5] = 0;
   out_2970244759936359815[6] = 0;
   out_2970244759936359815[7] = 0;
}
void h_29(double *state, double *unused, double *out_3928307331607215146) {
   out_3928307331607215146[0] = state[1];
}
void H_29(double *state, double *unused, double *out_1539567747406290369) {
   out_1539567747406290369[0] = 0;
   out_1539567747406290369[1] = 1;
   out_1539567747406290369[2] = 0;
   out_1539567747406290369[3] = 0;
   out_1539567747406290369[4] = 0;
   out_1539567747406290369[5] = 0;
   out_1539567747406290369[6] = 0;
   out_1539567747406290369[7] = 0;
}
void h_28(double *state, double *unused, double *out_6192292826639898941) {
   out_6192292826639898941[0] = state[5];
   out_6192292826639898941[1] = state[6];
}
void H_28(double *state, double *unused, double *out_782359815361327124) {
   out_782359815361327124[0] = 0;
   out_782359815361327124[1] = 0;
   out_782359815361327124[2] = 0;
   out_782359815361327124[3] = 0;
   out_782359815361327124[4] = 0;
   out_782359815361327124[5] = 1;
   out_782359815361327124[6] = 0;
   out_782359815361327124[7] = 0;
   out_782359815361327124[8] = 0;
   out_782359815361327124[9] = 0;
   out_782359815361327124[10] = 0;
   out_782359815361327124[11] = 0;
   out_782359815361327124[12] = 0;
   out_782359815361327124[13] = 0;
   out_782359815361327124[14] = 1;
   out_782359815361327124[15] = 0;
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
  update<2, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_3157583653505174283) {
  err_fun(nom_x, delta_x, out_3157583653505174283);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5329952054111744571) {
  inv_err_fun(nom_x, true_x, out_5329952054111744571);
}
void car_H_mod_fun(double *state, double *out_8803128394076082262) {
  H_mod_fun(state, out_8803128394076082262);
}
void car_f_fun(double *state, double dt, double *out_5487969453112289401) {
  f_fun(state,  dt, out_5487969453112289401);
}
void car_F_fun(double *state, double dt, double *out_249163952273201207) {
  F_fun(state,  dt, out_249163952273201207);
}
void car_h_25(double *state, double *unused, double *out_1166715342447388350) {
  h_25(state, unused, out_1166715342447388350);
}
void car_H_25(double *state, double *unused, double *out_7931236138849448777) {
  H_25(state, unused, out_7931236138849448777);
}
void car_h_24(double *state, double *unused, double *out_3877885127749003657) {
  h_24(state, unused, out_3877885127749003657);
}
void car_H_24(double *state, double *unused, double *out_6235472662875167742) {
  H_24(state, unused, out_6235472662875167742);
}
void car_h_30(double *state, double *unused, double *out_891521280162882461) {
  h_30(state, unused, out_891521280162882461);
}
void car_H_30(double *state, double *unused, double *out_1682662772099734503) {
  H_30(state, unused, out_1682662772099734503);
}
void car_h_26(double *state, double *unused, double *out_6111241423630183318) {
  h_26(state, unused, out_6111241423630183318);
}
void car_H_26(double *state, double *unused, double *out_4810914779920893566) {
  H_26(state, unused, out_4810914779920893566);
}
void car_h_27(double *state, double *unused, double *out_4711570145111326738) {
  h_27(state, unused, out_4711570145111326738);
}
void car_H_27(double *state, double *unused, double *out_2970244759936359815) {
  H_27(state, unused, out_2970244759936359815);
}
void car_h_29(double *state, double *unused, double *out_3928307331607215146) {
  h_29(state, unused, out_3928307331607215146);
}
void car_H_29(double *state, double *unused, double *out_1539567747406290369) {
  H_29(state, unused, out_1539567747406290369);
}
void car_h_28(double *state, double *unused, double *out_6192292826639898941) {
  h_28(state, unused, out_6192292826639898941);
}
void car_H_28(double *state, double *unused, double *out_782359815361327124) {
  H_28(state, unused, out_782359815361327124);
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
  .kinds = { 25, 24, 30, 26, 27, 29, 28 },
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
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
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
