
extern "C"{

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

}
extern "C" {
#include <math.h>
/******************************************************************************
 *                      Code generated with sympy 1.6.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_5976358665325250733) {
   out_5976358665325250733[0] = delta_x[0] + nom_x[0];
   out_5976358665325250733[1] = delta_x[1] + nom_x[1];
   out_5976358665325250733[2] = delta_x[2] + nom_x[2];
   out_5976358665325250733[3] = delta_x[3] + nom_x[3];
   out_5976358665325250733[4] = delta_x[4] + nom_x[4];
   out_5976358665325250733[5] = delta_x[5] + nom_x[5];
   out_5976358665325250733[6] = delta_x[6] + nom_x[6];
   out_5976358665325250733[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_545149083074471775) {
   out_545149083074471775[0] = -nom_x[0] + true_x[0];
   out_545149083074471775[1] = -nom_x[1] + true_x[1];
   out_545149083074471775[2] = -nom_x[2] + true_x[2];
   out_545149083074471775[3] = -nom_x[3] + true_x[3];
   out_545149083074471775[4] = -nom_x[4] + true_x[4];
   out_545149083074471775[5] = -nom_x[5] + true_x[5];
   out_545149083074471775[6] = -nom_x[6] + true_x[6];
   out_545149083074471775[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_6429485867732329304) {
   out_6429485867732329304[0] = 1.0;
   out_6429485867732329304[1] = 0.0;
   out_6429485867732329304[2] = 0.0;
   out_6429485867732329304[3] = 0.0;
   out_6429485867732329304[4] = 0.0;
   out_6429485867732329304[5] = 0.0;
   out_6429485867732329304[6] = 0.0;
   out_6429485867732329304[7] = 0.0;
   out_6429485867732329304[8] = 0.0;
   out_6429485867732329304[9] = 1.0;
   out_6429485867732329304[10] = 0.0;
   out_6429485867732329304[11] = 0.0;
   out_6429485867732329304[12] = 0.0;
   out_6429485867732329304[13] = 0.0;
   out_6429485867732329304[14] = 0.0;
   out_6429485867732329304[15] = 0.0;
   out_6429485867732329304[16] = 0.0;
   out_6429485867732329304[17] = 0.0;
   out_6429485867732329304[18] = 1.0;
   out_6429485867732329304[19] = 0.0;
   out_6429485867732329304[20] = 0.0;
   out_6429485867732329304[21] = 0.0;
   out_6429485867732329304[22] = 0.0;
   out_6429485867732329304[23] = 0.0;
   out_6429485867732329304[24] = 0.0;
   out_6429485867732329304[25] = 0.0;
   out_6429485867732329304[26] = 0.0;
   out_6429485867732329304[27] = 1.0;
   out_6429485867732329304[28] = 0.0;
   out_6429485867732329304[29] = 0.0;
   out_6429485867732329304[30] = 0.0;
   out_6429485867732329304[31] = 0.0;
   out_6429485867732329304[32] = 0.0;
   out_6429485867732329304[33] = 0.0;
   out_6429485867732329304[34] = 0.0;
   out_6429485867732329304[35] = 0.0;
   out_6429485867732329304[36] = 1.0;
   out_6429485867732329304[37] = 0.0;
   out_6429485867732329304[38] = 0.0;
   out_6429485867732329304[39] = 0.0;
   out_6429485867732329304[40] = 0.0;
   out_6429485867732329304[41] = 0.0;
   out_6429485867732329304[42] = 0.0;
   out_6429485867732329304[43] = 0.0;
   out_6429485867732329304[44] = 0.0;
   out_6429485867732329304[45] = 1.0;
   out_6429485867732329304[46] = 0.0;
   out_6429485867732329304[47] = 0.0;
   out_6429485867732329304[48] = 0.0;
   out_6429485867732329304[49] = 0.0;
   out_6429485867732329304[50] = 0.0;
   out_6429485867732329304[51] = 0.0;
   out_6429485867732329304[52] = 0.0;
   out_6429485867732329304[53] = 0.0;
   out_6429485867732329304[54] = 1.0;
   out_6429485867732329304[55] = 0.0;
   out_6429485867732329304[56] = 0.0;
   out_6429485867732329304[57] = 0.0;
   out_6429485867732329304[58] = 0.0;
   out_6429485867732329304[59] = 0.0;
   out_6429485867732329304[60] = 0.0;
   out_6429485867732329304[61] = 0.0;
   out_6429485867732329304[62] = 0.0;
   out_6429485867732329304[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_1814214989434170929) {
   out_1814214989434170929[0] = state[0];
   out_1814214989434170929[1] = state[1];
   out_1814214989434170929[2] = state[2];
   out_1814214989434170929[3] = state[3];
   out_1814214989434170929[4] = state[4];
   out_1814214989434170929[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_1814214989434170929[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_1814214989434170929[7] = state[7];
}
void F_fun(double *state, double dt, double *out_24478973576452139) {
   out_24478973576452139[0] = 1;
   out_24478973576452139[1] = 0;
   out_24478973576452139[2] = 0;
   out_24478973576452139[3] = 0;
   out_24478973576452139[4] = 0;
   out_24478973576452139[5] = 0;
   out_24478973576452139[6] = 0;
   out_24478973576452139[7] = 0;
   out_24478973576452139[8] = 0;
   out_24478973576452139[9] = 1;
   out_24478973576452139[10] = 0;
   out_24478973576452139[11] = 0;
   out_24478973576452139[12] = 0;
   out_24478973576452139[13] = 0;
   out_24478973576452139[14] = 0;
   out_24478973576452139[15] = 0;
   out_24478973576452139[16] = 0;
   out_24478973576452139[17] = 0;
   out_24478973576452139[18] = 1;
   out_24478973576452139[19] = 0;
   out_24478973576452139[20] = 0;
   out_24478973576452139[21] = 0;
   out_24478973576452139[22] = 0;
   out_24478973576452139[23] = 0;
   out_24478973576452139[24] = 0;
   out_24478973576452139[25] = 0;
   out_24478973576452139[26] = 0;
   out_24478973576452139[27] = 1;
   out_24478973576452139[28] = 0;
   out_24478973576452139[29] = 0;
   out_24478973576452139[30] = 0;
   out_24478973576452139[31] = 0;
   out_24478973576452139[32] = 0;
   out_24478973576452139[33] = 0;
   out_24478973576452139[34] = 0;
   out_24478973576452139[35] = 0;
   out_24478973576452139[36] = 1;
   out_24478973576452139[37] = 0;
   out_24478973576452139[38] = 0;
   out_24478973576452139[39] = 0;
   out_24478973576452139[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_24478973576452139[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_24478973576452139[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_24478973576452139[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_24478973576452139[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_24478973576452139[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_24478973576452139[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_24478973576452139[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_24478973576452139[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_24478973576452139[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_24478973576452139[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_24478973576452139[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_24478973576452139[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_24478973576452139[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_24478973576452139[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_24478973576452139[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_24478973576452139[56] = 0;
   out_24478973576452139[57] = 0;
   out_24478973576452139[58] = 0;
   out_24478973576452139[59] = 0;
   out_24478973576452139[60] = 0;
   out_24478973576452139[61] = 0;
   out_24478973576452139[62] = 0;
   out_24478973576452139[63] = 1;
}
void h_25(double *state, double *unused, double *out_6753750671493724343) {
   out_6753750671493724343[0] = state[6];
}
void H_25(double *state, double *unused, double *out_505484718201953462) {
   out_505484718201953462[0] = 0;
   out_505484718201953462[1] = 0;
   out_505484718201953462[2] = 0;
   out_505484718201953462[3] = 0;
   out_505484718201953462[4] = 0;
   out_505484718201953462[5] = 0;
   out_505484718201953462[6] = 1;
   out_505484718201953462[7] = 0;
}
void h_24(double *state, double *unused, double *out_7677385143792155145) {
   out_7677385143792155145[0] = state[4];
   out_7677385143792155145[1] = state[5];
}
void H_24(double *state, double *unused, double *out_1767873782397693594) {
   out_1767873782397693594[0] = 0;
   out_1767873782397693594[1] = 0;
   out_1767873782397693594[2] = 0;
   out_1767873782397693594[3] = 0;
   out_1767873782397693594[4] = 1;
   out_1767873782397693594[5] = 0;
   out_1767873782397693594[6] = 0;
   out_1767873782397693594[7] = 0;
   out_1767873782397693594[8] = 0;
   out_1767873782397693594[9] = 0;
   out_1767873782397693594[10] = 0;
   out_1767873782397693594[11] = 0;
   out_1767873782397693594[12] = 0;
   out_1767873782397693594[13] = 1;
   out_1767873782397693594[14] = 0;
   out_1767873782397693594[15] = 0;
}
void h_30(double *state, double *unused, double *out_6793389117300471979) {
   out_6793389117300471979[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4939972497977953670) {
   out_4939972497977953670[0] = 0;
   out_4939972497977953670[1] = 0;
   out_4939972497977953670[2] = 0;
   out_4939972497977953670[3] = 0;
   out_4939972497977953670[4] = 1;
   out_4939972497977953670[5] = 0;
   out_4939972497977953670[6] = 0;
   out_4939972497977953670[7] = 0;
}
void h_26(double *state, double *unused, double *out_6190073232623994462) {
   out_6190073232623994462[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5234308798478062218) {
   out_5234308798478062218[0] = 0;
   out_5234308798478062218[1] = 0;
   out_5234308798478062218[2] = 0;
   out_5234308798478062218[3] = 0;
   out_5234308798478062218[4] = 0;
   out_5234308798478062218[5] = 0;
   out_5234308798478062218[6] = 0;
   out_5234308798478062218[7] = 1;
}
void h_27(double *state, double *unused, double *out_6050263531134870438) {
   out_6050263531134870438[0] = state[3];
}
void H_27(double *state, double *unused, double *out_8050747893125696486) {
   out_8050747893125696486[0] = 0;
   out_8050747893125696486[1] = 0;
   out_8050747893125696486[2] = 0;
   out_8050747893125696486[3] = 1;
   out_8050747893125696486[4] = 0;
   out_8050747893125696486[5] = 0;
   out_8050747893125696486[6] = 0;
   out_8050747893125696486[7] = 0;
}
void h_29(double *state, double *unused, double *out_2829636658126479976) {
   out_2829636658126479976[0] = state[1];
}
void H_29(double *state, double *unused, double *out_5514531111833489845) {
   out_5514531111833489845[0] = 0;
   out_5514531111833489845[1] = 1;
   out_5514531111833489845[2] = 0;
   out_5514531111833489845[3] = 0;
   out_5514531111833489845[4] = 0;
   out_5514531111833489845[5] = 0;
   out_5514531111833489845[6] = 0;
   out_5514531111833489845[7] = 0;
}
void h_28(double *state, double *unused, double *out_5669571770271976550) {
   out_5669571770271976550[0] = state[5];
   out_5669571770271976550[1] = state[6];
}
void H_28(double *state, double *unused, double *out_8785706260634188460) {
   out_8785706260634188460[0] = 0;
   out_8785706260634188460[1] = 0;
   out_8785706260634188460[2] = 0;
   out_8785706260634188460[3] = 0;
   out_8785706260634188460[4] = 0;
   out_8785706260634188460[5] = 1;
   out_8785706260634188460[6] = 0;
   out_8785706260634188460[7] = 0;
   out_8785706260634188460[8] = 0;
   out_8785706260634188460[9] = 0;
   out_8785706260634188460[10] = 0;
   out_8785706260634188460[11] = 0;
   out_8785706260634188460[12] = 0;
   out_8785706260634188460[13] = 0;
   out_8785706260634188460[14] = 1;
   out_8785706260634188460[15] = 0;
}
}

extern "C"{
#define DIM 8
#define EDIM 8
#define MEDIM 8
typedef void (*Hfun)(double *, double *, double *);

void predict(double *x, double *P, double *Q, double dt);
const static double MAHA_THRESH_25 = 3.841459;
void update_25(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_24 = 5.991465;
void update_24(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_30 = 3.841459;
void update_30(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_26 = 3.841459;
void update_26(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_27 = 3.841459;
void update_27(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_29 = 3.841459;
void update_29(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_28 = 5.991465;
void update_28(double *, double *, double *, double *, double *);
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



extern "C"{

      void update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
      }
    
      void update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<2,3,0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
      }
    
      void update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
      }
    
      void update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
      }
    
      void update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
      }
    
      void update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
      }
    
      void update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<2,3,0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
      }
    
}
