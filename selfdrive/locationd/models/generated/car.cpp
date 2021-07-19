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
void err_fun(double *nom_x, double *delta_x, double *out_2195023721757572954) {
   out_2195023721757572954[0] = delta_x[0] + nom_x[0];
   out_2195023721757572954[1] = delta_x[1] + nom_x[1];
   out_2195023721757572954[2] = delta_x[2] + nom_x[2];
   out_2195023721757572954[3] = delta_x[3] + nom_x[3];
   out_2195023721757572954[4] = delta_x[4] + nom_x[4];
   out_2195023721757572954[5] = delta_x[5] + nom_x[5];
   out_2195023721757572954[6] = delta_x[6] + nom_x[6];
   out_2195023721757572954[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6339939555658911788) {
   out_6339939555658911788[0] = -nom_x[0] + true_x[0];
   out_6339939555658911788[1] = -nom_x[1] + true_x[1];
   out_6339939555658911788[2] = -nom_x[2] + true_x[2];
   out_6339939555658911788[3] = -nom_x[3] + true_x[3];
   out_6339939555658911788[4] = -nom_x[4] + true_x[4];
   out_6339939555658911788[5] = -nom_x[5] + true_x[5];
   out_6339939555658911788[6] = -nom_x[6] + true_x[6];
   out_6339939555658911788[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_8639855113291280281) {
   out_8639855113291280281[0] = 1.0;
   out_8639855113291280281[1] = 0.0;
   out_8639855113291280281[2] = 0.0;
   out_8639855113291280281[3] = 0.0;
   out_8639855113291280281[4] = 0.0;
   out_8639855113291280281[5] = 0.0;
   out_8639855113291280281[6] = 0.0;
   out_8639855113291280281[7] = 0.0;
   out_8639855113291280281[8] = 0.0;
   out_8639855113291280281[9] = 1.0;
   out_8639855113291280281[10] = 0.0;
   out_8639855113291280281[11] = 0.0;
   out_8639855113291280281[12] = 0.0;
   out_8639855113291280281[13] = 0.0;
   out_8639855113291280281[14] = 0.0;
   out_8639855113291280281[15] = 0.0;
   out_8639855113291280281[16] = 0.0;
   out_8639855113291280281[17] = 0.0;
   out_8639855113291280281[18] = 1.0;
   out_8639855113291280281[19] = 0.0;
   out_8639855113291280281[20] = 0.0;
   out_8639855113291280281[21] = 0.0;
   out_8639855113291280281[22] = 0.0;
   out_8639855113291280281[23] = 0.0;
   out_8639855113291280281[24] = 0.0;
   out_8639855113291280281[25] = 0.0;
   out_8639855113291280281[26] = 0.0;
   out_8639855113291280281[27] = 1.0;
   out_8639855113291280281[28] = 0.0;
   out_8639855113291280281[29] = 0.0;
   out_8639855113291280281[30] = 0.0;
   out_8639855113291280281[31] = 0.0;
   out_8639855113291280281[32] = 0.0;
   out_8639855113291280281[33] = 0.0;
   out_8639855113291280281[34] = 0.0;
   out_8639855113291280281[35] = 0.0;
   out_8639855113291280281[36] = 1.0;
   out_8639855113291280281[37] = 0.0;
   out_8639855113291280281[38] = 0.0;
   out_8639855113291280281[39] = 0.0;
   out_8639855113291280281[40] = 0.0;
   out_8639855113291280281[41] = 0.0;
   out_8639855113291280281[42] = 0.0;
   out_8639855113291280281[43] = 0.0;
   out_8639855113291280281[44] = 0.0;
   out_8639855113291280281[45] = 1.0;
   out_8639855113291280281[46] = 0.0;
   out_8639855113291280281[47] = 0.0;
   out_8639855113291280281[48] = 0.0;
   out_8639855113291280281[49] = 0.0;
   out_8639855113291280281[50] = 0.0;
   out_8639855113291280281[51] = 0.0;
   out_8639855113291280281[52] = 0.0;
   out_8639855113291280281[53] = 0.0;
   out_8639855113291280281[54] = 1.0;
   out_8639855113291280281[55] = 0.0;
   out_8639855113291280281[56] = 0.0;
   out_8639855113291280281[57] = 0.0;
   out_8639855113291280281[58] = 0.0;
   out_8639855113291280281[59] = 0.0;
   out_8639855113291280281[60] = 0.0;
   out_8639855113291280281[61] = 0.0;
   out_8639855113291280281[62] = 0.0;
   out_8639855113291280281[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_8753740640370834672) {
   out_8753740640370834672[0] = state[0];
   out_8753740640370834672[1] = state[1];
   out_8753740640370834672[2] = state[2];
   out_8753740640370834672[3] = state[3];
   out_8753740640370834672[4] = state[4];
   out_8753740640370834672[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_8753740640370834672[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_8753740640370834672[7] = state[7];
}
void F_fun(double *state, double dt, double *out_4380248598819438044) {
   out_4380248598819438044[0] = 1;
   out_4380248598819438044[1] = 0;
   out_4380248598819438044[2] = 0;
   out_4380248598819438044[3] = 0;
   out_4380248598819438044[4] = 0;
   out_4380248598819438044[5] = 0;
   out_4380248598819438044[6] = 0;
   out_4380248598819438044[7] = 0;
   out_4380248598819438044[8] = 0;
   out_4380248598819438044[9] = 1;
   out_4380248598819438044[10] = 0;
   out_4380248598819438044[11] = 0;
   out_4380248598819438044[12] = 0;
   out_4380248598819438044[13] = 0;
   out_4380248598819438044[14] = 0;
   out_4380248598819438044[15] = 0;
   out_4380248598819438044[16] = 0;
   out_4380248598819438044[17] = 0;
   out_4380248598819438044[18] = 1;
   out_4380248598819438044[19] = 0;
   out_4380248598819438044[20] = 0;
   out_4380248598819438044[21] = 0;
   out_4380248598819438044[22] = 0;
   out_4380248598819438044[23] = 0;
   out_4380248598819438044[24] = 0;
   out_4380248598819438044[25] = 0;
   out_4380248598819438044[26] = 0;
   out_4380248598819438044[27] = 1;
   out_4380248598819438044[28] = 0;
   out_4380248598819438044[29] = 0;
   out_4380248598819438044[30] = 0;
   out_4380248598819438044[31] = 0;
   out_4380248598819438044[32] = 0;
   out_4380248598819438044[33] = 0;
   out_4380248598819438044[34] = 0;
   out_4380248598819438044[35] = 0;
   out_4380248598819438044[36] = 1;
   out_4380248598819438044[37] = 0;
   out_4380248598819438044[38] = 0;
   out_4380248598819438044[39] = 0;
   out_4380248598819438044[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_4380248598819438044[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_4380248598819438044[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4380248598819438044[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4380248598819438044[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_4380248598819438044[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_4380248598819438044[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_4380248598819438044[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_4380248598819438044[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4380248598819438044[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4380248598819438044[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4380248598819438044[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4380248598819438044[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4380248598819438044[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4380248598819438044[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4380248598819438044[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4380248598819438044[56] = 0;
   out_4380248598819438044[57] = 0;
   out_4380248598819438044[58] = 0;
   out_4380248598819438044[59] = 0;
   out_4380248598819438044[60] = 0;
   out_4380248598819438044[61] = 0;
   out_4380248598819438044[62] = 0;
   out_4380248598819438044[63] = 1;
}
void h_25(double *state, double *unused, double *out_5041160612363327269) {
   out_5041160612363327269[0] = state[6];
}
void H_25(double *state, double *unused, double *out_2934563808916292903) {
   out_2934563808916292903[0] = 0;
   out_2934563808916292903[1] = 0;
   out_2934563808916292903[2] = 0;
   out_2934563808916292903[3] = 0;
   out_2934563808916292903[4] = 0;
   out_2934563808916292903[5] = 0;
   out_2934563808916292903[6] = 1;
   out_2934563808916292903[7] = 0;
}
void h_24(double *state, double *unused, double *out_6427212538985091517) {
   out_6427212538985091517[0] = state[4];
   out_6427212538985091517[1] = state[5];
}
void H_24(double *state, double *unused, double *out_7214599080901228000) {
   out_7214599080901228000[0] = 0;
   out_7214599080901228000[1] = 0;
   out_7214599080901228000[2] = 0;
   out_7214599080901228000[3] = 0;
   out_7214599080901228000[4] = 1;
   out_7214599080901228000[5] = 0;
   out_7214599080901228000[6] = 0;
   out_7214599080901228000[7] = 0;
   out_7214599080901228000[8] = 0;
   out_7214599080901228000[9] = 0;
   out_7214599080901228000[10] = 0;
   out_7214599080901228000[11] = 0;
   out_7214599080901228000[12] = 0;
   out_7214599080901228000[13] = 1;
   out_7214599080901228000[14] = 0;
   out_7214599080901228000[15] = 0;
}
void h_30(double *state, double *unused, double *out_1729674613987023667) {
   out_1729674613987023667[0] = state[4];
}
void H_30(double *state, double *unused, double *out_6679335102032890377) {
   out_6679335102032890377[0] = 0;
   out_6679335102032890377[1] = 0;
   out_6679335102032890377[2] = 0;
   out_6679335102032890377[3] = 0;
   out_6679335102032890377[4] = 1;
   out_6679335102032890377[5] = 0;
   out_6679335102032890377[6] = 0;
   out_6679335102032890377[7] = 0;
}
void h_26(double *state, double *unused, double *out_2332990498663501184) {
   out_2332990498663501184[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8639156963855502176) {
   out_8639156963855502176[0] = 0;
   out_8639156963855502176[1] = 0;
   out_8639156963855502176[2] = 0;
   out_8639156963855502176[3] = 0;
   out_8639156963855502176[4] = 0;
   out_8639156963855502176[5] = 0;
   out_8639156963855502176[6] = 0;
   out_8639156963855502176[7] = 1;
}
void h_27(double *state, double *unused, double *out_4097650759461664834) {
   out_4097650759461664834[0] = state[3];
}
void H_27(double *state, double *unused, double *out_7966917089869515689) {
   out_7966917089869515689[0] = 0;
   out_7966917089869515689[1] = 0;
   out_7966917089869515689[2] = 0;
   out_7966917089869515689[3] = 1;
   out_7966917089869515689[4] = 0;
   out_7966917089869515689[5] = 0;
   out_7966917089869515689[6] = 0;
   out_7966917089869515689[7] = 0;
}
void h_29(double *state, double *unused, double *out_7735100287715687184) {
   out_7735100287715687184[0] = state[1];
}
void H_29(double *state, double *unused, double *out_6104776488177354202) {
   out_6104776488177354202[0] = 0;
   out_6104776488177354202[1] = 1;
   out_6104776488177354202[2] = 0;
   out_6104776488177354202[3] = 0;
   out_6104776488177354202[4] = 0;
   out_6104776488177354202[5] = 0;
   out_6104776488177354202[6] = 0;
   out_6104776488177354202[7] = 0;
}
void h_28(double *state, double *unused, double *out_3201468729684098971) {
   out_3201468729684098971[0] = state[5];
   out_3201468729684098971[1] = state[6];
}
void H_28(double *state, double *unused, double *out_4214312514571828750) {
   out_4214312514571828750[0] = 0;
   out_4214312514571828750[1] = 0;
   out_4214312514571828750[2] = 0;
   out_4214312514571828750[3] = 0;
   out_4214312514571828750[4] = 0;
   out_4214312514571828750[5] = 1;
   out_4214312514571828750[6] = 0;
   out_4214312514571828750[7] = 0;
   out_4214312514571828750[8] = 0;
   out_4214312514571828750[9] = 0;
   out_4214312514571828750[10] = 0;
   out_4214312514571828750[11] = 0;
   out_4214312514571828750[12] = 0;
   out_4214312514571828750[13] = 0;
   out_4214312514571828750[14] = 1;
   out_4214312514571828750[15] = 0;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_2195023721757572954) {
  err_fun(nom_x, delta_x, out_2195023721757572954);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6339939555658911788) {
  inv_err_fun(nom_x, true_x, out_6339939555658911788);
}
void car_H_mod_fun(double *state, double *out_8639855113291280281) {
  H_mod_fun(state, out_8639855113291280281);
}
void car_f_fun(double *state, double dt, double *out_8753740640370834672) {
  f_fun(state,  dt, out_8753740640370834672);
}
void car_F_fun(double *state, double dt, double *out_4380248598819438044) {
  F_fun(state,  dt, out_4380248598819438044);
}
void car_h_25(double *state, double *unused, double *out_5041160612363327269) {
  h_25(state, unused, out_5041160612363327269);
}
void car_H_25(double *state, double *unused, double *out_2934563808916292903) {
  H_25(state, unused, out_2934563808916292903);
}
void car_h_24(double *state, double *unused, double *out_6427212538985091517) {
  h_24(state, unused, out_6427212538985091517);
}
void car_H_24(double *state, double *unused, double *out_7214599080901228000) {
  H_24(state, unused, out_7214599080901228000);
}
void car_h_30(double *state, double *unused, double *out_1729674613987023667) {
  h_30(state, unused, out_1729674613987023667);
}
void car_H_30(double *state, double *unused, double *out_6679335102032890377) {
  H_30(state, unused, out_6679335102032890377);
}
void car_h_26(double *state, double *unused, double *out_2332990498663501184) {
  h_26(state, unused, out_2332990498663501184);
}
void car_H_26(double *state, double *unused, double *out_8639156963855502176) {
  H_26(state, unused, out_8639156963855502176);
}
void car_h_27(double *state, double *unused, double *out_4097650759461664834) {
  h_27(state, unused, out_4097650759461664834);
}
void car_H_27(double *state, double *unused, double *out_7966917089869515689) {
  H_27(state, unused, out_7966917089869515689);
}
void car_h_29(double *state, double *unused, double *out_7735100287715687184) {
  h_29(state, unused, out_7735100287715687184);
}
void car_H_29(double *state, double *unused, double *out_6104776488177354202) {
  H_29(state, unused, out_6104776488177354202);
}
void car_h_28(double *state, double *unused, double *out_3201468729684098971) {
  h_28(state, unused, out_3201468729684098971);
}
void car_H_28(double *state, double *unused, double *out_4214312514571828750) {
  H_28(state, unused, out_4214312514571828750);
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
