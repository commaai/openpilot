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
 *                       Code generated with sympy 1.8                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_4697919531062724430) {
   out_4697919531062724430[0] = delta_x[0] + nom_x[0];
   out_4697919531062724430[1] = delta_x[1] + nom_x[1];
   out_4697919531062724430[2] = delta_x[2] + nom_x[2];
   out_4697919531062724430[3] = delta_x[3] + nom_x[3];
   out_4697919531062724430[4] = delta_x[4] + nom_x[4];
   out_4697919531062724430[5] = delta_x[5] + nom_x[5];
   out_4697919531062724430[6] = delta_x[6] + nom_x[6];
   out_4697919531062724430[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8372100327763513160) {
   out_8372100327763513160[0] = -nom_x[0] + true_x[0];
   out_8372100327763513160[1] = -nom_x[1] + true_x[1];
   out_8372100327763513160[2] = -nom_x[2] + true_x[2];
   out_8372100327763513160[3] = -nom_x[3] + true_x[3];
   out_8372100327763513160[4] = -nom_x[4] + true_x[4];
   out_8372100327763513160[5] = -nom_x[5] + true_x[5];
   out_8372100327763513160[6] = -nom_x[6] + true_x[6];
   out_8372100327763513160[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_5880162746762264552) {
   out_5880162746762264552[0] = 1.0;
   out_5880162746762264552[1] = 0.0;
   out_5880162746762264552[2] = 0.0;
   out_5880162746762264552[3] = 0.0;
   out_5880162746762264552[4] = 0.0;
   out_5880162746762264552[5] = 0.0;
   out_5880162746762264552[6] = 0.0;
   out_5880162746762264552[7] = 0.0;
   out_5880162746762264552[8] = 0.0;
   out_5880162746762264552[9] = 1.0;
   out_5880162746762264552[10] = 0.0;
   out_5880162746762264552[11] = 0.0;
   out_5880162746762264552[12] = 0.0;
   out_5880162746762264552[13] = 0.0;
   out_5880162746762264552[14] = 0.0;
   out_5880162746762264552[15] = 0.0;
   out_5880162746762264552[16] = 0.0;
   out_5880162746762264552[17] = 0.0;
   out_5880162746762264552[18] = 1.0;
   out_5880162746762264552[19] = 0.0;
   out_5880162746762264552[20] = 0.0;
   out_5880162746762264552[21] = 0.0;
   out_5880162746762264552[22] = 0.0;
   out_5880162746762264552[23] = 0.0;
   out_5880162746762264552[24] = 0.0;
   out_5880162746762264552[25] = 0.0;
   out_5880162746762264552[26] = 0.0;
   out_5880162746762264552[27] = 1.0;
   out_5880162746762264552[28] = 0.0;
   out_5880162746762264552[29] = 0.0;
   out_5880162746762264552[30] = 0.0;
   out_5880162746762264552[31] = 0.0;
   out_5880162746762264552[32] = 0.0;
   out_5880162746762264552[33] = 0.0;
   out_5880162746762264552[34] = 0.0;
   out_5880162746762264552[35] = 0.0;
   out_5880162746762264552[36] = 1.0;
   out_5880162746762264552[37] = 0.0;
   out_5880162746762264552[38] = 0.0;
   out_5880162746762264552[39] = 0.0;
   out_5880162746762264552[40] = 0.0;
   out_5880162746762264552[41] = 0.0;
   out_5880162746762264552[42] = 0.0;
   out_5880162746762264552[43] = 0.0;
   out_5880162746762264552[44] = 0.0;
   out_5880162746762264552[45] = 1.0;
   out_5880162746762264552[46] = 0.0;
   out_5880162746762264552[47] = 0.0;
   out_5880162746762264552[48] = 0.0;
   out_5880162746762264552[49] = 0.0;
   out_5880162746762264552[50] = 0.0;
   out_5880162746762264552[51] = 0.0;
   out_5880162746762264552[52] = 0.0;
   out_5880162746762264552[53] = 0.0;
   out_5880162746762264552[54] = 1.0;
   out_5880162746762264552[55] = 0.0;
   out_5880162746762264552[56] = 0.0;
   out_5880162746762264552[57] = 0.0;
   out_5880162746762264552[58] = 0.0;
   out_5880162746762264552[59] = 0.0;
   out_5880162746762264552[60] = 0.0;
   out_5880162746762264552[61] = 0.0;
   out_5880162746762264552[62] = 0.0;
   out_5880162746762264552[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_7591948858249745570) {
   out_7591948858249745570[0] = state[0];
   out_7591948858249745570[1] = state[1];
   out_7591948858249745570[2] = state[2];
   out_7591948858249745570[3] = state[3];
   out_7591948858249745570[4] = state[4];
   out_7591948858249745570[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_7591948858249745570[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7591948858249745570[7] = state[7];
}
void F_fun(double *state, double dt, double *out_748199326983412132) {
   out_748199326983412132[0] = 1;
   out_748199326983412132[1] = 0;
   out_748199326983412132[2] = 0;
   out_748199326983412132[3] = 0;
   out_748199326983412132[4] = 0;
   out_748199326983412132[5] = 0;
   out_748199326983412132[6] = 0;
   out_748199326983412132[7] = 0;
   out_748199326983412132[8] = 0;
   out_748199326983412132[9] = 1;
   out_748199326983412132[10] = 0;
   out_748199326983412132[11] = 0;
   out_748199326983412132[12] = 0;
   out_748199326983412132[13] = 0;
   out_748199326983412132[14] = 0;
   out_748199326983412132[15] = 0;
   out_748199326983412132[16] = 0;
   out_748199326983412132[17] = 0;
   out_748199326983412132[18] = 1;
   out_748199326983412132[19] = 0;
   out_748199326983412132[20] = 0;
   out_748199326983412132[21] = 0;
   out_748199326983412132[22] = 0;
   out_748199326983412132[23] = 0;
   out_748199326983412132[24] = 0;
   out_748199326983412132[25] = 0;
   out_748199326983412132[26] = 0;
   out_748199326983412132[27] = 1;
   out_748199326983412132[28] = 0;
   out_748199326983412132[29] = 0;
   out_748199326983412132[30] = 0;
   out_748199326983412132[31] = 0;
   out_748199326983412132[32] = 0;
   out_748199326983412132[33] = 0;
   out_748199326983412132[34] = 0;
   out_748199326983412132[35] = 0;
   out_748199326983412132[36] = 1;
   out_748199326983412132[37] = 0;
   out_748199326983412132[38] = 0;
   out_748199326983412132[39] = 0;
   out_748199326983412132[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_748199326983412132[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_748199326983412132[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_748199326983412132[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_748199326983412132[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_748199326983412132[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_748199326983412132[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_748199326983412132[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_748199326983412132[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_748199326983412132[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_748199326983412132[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_748199326983412132[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_748199326983412132[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_748199326983412132[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_748199326983412132[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_748199326983412132[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_748199326983412132[56] = 0;
   out_748199326983412132[57] = 0;
   out_748199326983412132[58] = 0;
   out_748199326983412132[59] = 0;
   out_748199326983412132[60] = 0;
   out_748199326983412132[61] = 0;
   out_748199326983412132[62] = 0;
   out_748199326983412132[63] = 1;
}
void h_25(double *state, double *unused, double *out_7489972780543671829) {
   out_7489972780543671829[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7196505734501455454) {
   out_7196505734501455454[0] = 0;
   out_7196505734501455454[1] = 0;
   out_7196505734501455454[2] = 0;
   out_7196505734501455454[3] = 0;
   out_7196505734501455454[4] = 0;
   out_7196505734501455454[5] = 0;
   out_7196505734501455454[6] = 1;
   out_7196505734501455454[7] = 0;
}
void h_24(double *state, double *unused, double *out_1608199115319231918) {
   out_1608199115319231918[0] = state[4];
   out_1608199115319231918[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2753669124750351732) {
   out_2753669124750351732[0] = 0;
   out_2753669124750351732[1] = 0;
   out_2753669124750351732[2] = 0;
   out_2753669124750351732[3] = 0;
   out_2753669124750351732[4] = 1;
   out_2753669124750351732[5] = 0;
   out_2753669124750351732[6] = 0;
   out_2753669124750351732[7] = 0;
   out_2753669124750351732[8] = 0;
   out_2753669124750351732[9] = 0;
   out_2753669124750351732[10] = 0;
   out_2753669124750351732[11] = 0;
   out_2753669124750351732[12] = 0;
   out_2753669124750351732[13] = 1;
   out_2753669124750351732[14] = 0;
   out_2753669124750351732[15] = 0;
}
void h_30(double *state, double *unused, double *out_3391669482264299965) {
   out_3391669482264299965[0] = state[4];
}
void H_30(double *state, double *unused, double *out_1636339428258912882) {
   out_1636339428258912882[0] = 0;
   out_1636339428258912882[1] = 0;
   out_1636339428258912882[2] = 0;
   out_1636339428258912882[3] = 0;
   out_1636339428258912882[4] = 1;
   out_1636339428258912882[5] = 0;
   out_1636339428258912882[6] = 0;
   out_1636339428258912882[7] = 0;
}
void h_26(double *state, double *unused, double *out_6151577963835715700) {
   out_6151577963835715700[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8537941868197103006) {
   out_8537941868197103006[0] = 0;
   out_8537941868197103006[1] = 0;
   out_8537941868197103006[2] = 0;
   out_8537941868197103006[3] = 0;
   out_8537941868197103006[4] = 0;
   out_8537941868197103006[5] = 0;
   out_8537941868197103006[6] = 0;
   out_8537941868197103006[7] = 1;
}
void h_27(double *state, double *unused, double *out_8472056873878969941) {
   out_8472056873878969941[0] = state[3];
}
void H_27(double *state, double *unused, double *out_348757440422287570) {
   out_348757440422287570[0] = 0;
   out_348757440422287570[1] = 0;
   out_348757440422287570[2] = 0;
   out_348757440422287570[3] = 1;
   out_348757440422287570[4] = 0;
   out_348757440422287570[5] = 0;
   out_348757440422287570[6] = 0;
   out_348757440422287570[7] = 0;
}
void h_29(double *state, double *unused, double *out_8393342749925145610) {
   out_8393342749925145610[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2187459340869919071) {
   out_2187459340869919071[0] = 0;
   out_2187459340869919071[1] = 1;
   out_2187459340869919071[2] = 0;
   out_2187459340869919071[3] = 0;
   out_2187459340869919071[4] = 0;
   out_2187459340869919071[5] = 0;
   out_2187459340869919071[6] = 0;
   out_2187459340869919071[7] = 0;
}
void h_28(double *state, double *unused, double *out_3943743383085014638) {
   out_3943743383085014638[0] = state[5];
   out_3943743383085014638[1] = state[6];
}
void H_28(double *state, double *unused, double *out_8675242470722705018) {
   out_8675242470722705018[0] = 0;
   out_8675242470722705018[1] = 0;
   out_8675242470722705018[2] = 0;
   out_8675242470722705018[3] = 0;
   out_8675242470722705018[4] = 0;
   out_8675242470722705018[5] = 1;
   out_8675242470722705018[6] = 0;
   out_8675242470722705018[7] = 0;
   out_8675242470722705018[8] = 0;
   out_8675242470722705018[9] = 0;
   out_8675242470722705018[10] = 0;
   out_8675242470722705018[11] = 0;
   out_8675242470722705018[12] = 0;
   out_8675242470722705018[13] = 0;
   out_8675242470722705018[14] = 1;
   out_8675242470722705018[15] = 0;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_4697919531062724430) {
  err_fun(nom_x, delta_x, out_4697919531062724430);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8372100327763513160) {
  inv_err_fun(nom_x, true_x, out_8372100327763513160);
}
void car_H_mod_fun(double *state, double *out_5880162746762264552) {
  H_mod_fun(state, out_5880162746762264552);
}
void car_f_fun(double *state, double dt, double *out_7591948858249745570) {
  f_fun(state,  dt, out_7591948858249745570);
}
void car_F_fun(double *state, double dt, double *out_748199326983412132) {
  F_fun(state,  dt, out_748199326983412132);
}
void car_h_25(double *state, double *unused, double *out_7489972780543671829) {
  h_25(state, unused, out_7489972780543671829);
}
void car_H_25(double *state, double *unused, double *out_7196505734501455454) {
  H_25(state, unused, out_7196505734501455454);
}
void car_h_24(double *state, double *unused, double *out_1608199115319231918) {
  h_24(state, unused, out_1608199115319231918);
}
void car_H_24(double *state, double *unused, double *out_2753669124750351732) {
  H_24(state, unused, out_2753669124750351732);
}
void car_h_30(double *state, double *unused, double *out_3391669482264299965) {
  h_30(state, unused, out_3391669482264299965);
}
void car_H_30(double *state, double *unused, double *out_1636339428258912882) {
  H_30(state, unused, out_1636339428258912882);
}
void car_h_26(double *state, double *unused, double *out_6151577963835715700) {
  h_26(state, unused, out_6151577963835715700);
}
void car_H_26(double *state, double *unused, double *out_8537941868197103006) {
  H_26(state, unused, out_8537941868197103006);
}
void car_h_27(double *state, double *unused, double *out_8472056873878969941) {
  h_27(state, unused, out_8472056873878969941);
}
void car_H_27(double *state, double *unused, double *out_348757440422287570) {
  H_27(state, unused, out_348757440422287570);
}
void car_h_29(double *state, double *unused, double *out_8393342749925145610) {
  h_29(state, unused, out_8393342749925145610);
}
void car_H_29(double *state, double *unused, double *out_2187459340869919071) {
  H_29(state, unused, out_2187459340869919071);
}
void car_h_28(double *state, double *unused, double *out_3943743383085014638) {
  h_28(state, unused, out_3943743383085014638);
}
void car_H_28(double *state, double *unused, double *out_8675242470722705018) {
  H_28(state, unused, out_8675242470722705018);
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
