
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
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7280050432103966623) {
   out_7280050432103966623[0] = delta_x[0] + nom_x[0];
   out_7280050432103966623[1] = delta_x[1] + nom_x[1];
   out_7280050432103966623[2] = delta_x[2] + nom_x[2];
   out_7280050432103966623[3] = delta_x[3] + nom_x[3];
   out_7280050432103966623[4] = delta_x[4] + nom_x[4];
   out_7280050432103966623[5] = delta_x[5] + nom_x[5];
   out_7280050432103966623[6] = delta_x[6] + nom_x[6];
   out_7280050432103966623[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3047360746371905783) {
   out_3047360746371905783[0] = -nom_x[0] + true_x[0];
   out_3047360746371905783[1] = -nom_x[1] + true_x[1];
   out_3047360746371905783[2] = -nom_x[2] + true_x[2];
   out_3047360746371905783[3] = -nom_x[3] + true_x[3];
   out_3047360746371905783[4] = -nom_x[4] + true_x[4];
   out_3047360746371905783[5] = -nom_x[5] + true_x[5];
   out_3047360746371905783[6] = -nom_x[6] + true_x[6];
   out_3047360746371905783[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_6308920866961272604) {
   out_6308920866961272604[0] = 1.0;
   out_6308920866961272604[1] = 0.0;
   out_6308920866961272604[2] = 0.0;
   out_6308920866961272604[3] = 0.0;
   out_6308920866961272604[4] = 0.0;
   out_6308920866961272604[5] = 0.0;
   out_6308920866961272604[6] = 0.0;
   out_6308920866961272604[7] = 0.0;
   out_6308920866961272604[8] = 0.0;
   out_6308920866961272604[9] = 1.0;
   out_6308920866961272604[10] = 0.0;
   out_6308920866961272604[11] = 0.0;
   out_6308920866961272604[12] = 0.0;
   out_6308920866961272604[13] = 0.0;
   out_6308920866961272604[14] = 0.0;
   out_6308920866961272604[15] = 0.0;
   out_6308920866961272604[16] = 0.0;
   out_6308920866961272604[17] = 0.0;
   out_6308920866961272604[18] = 1.0;
   out_6308920866961272604[19] = 0.0;
   out_6308920866961272604[20] = 0.0;
   out_6308920866961272604[21] = 0.0;
   out_6308920866961272604[22] = 0.0;
   out_6308920866961272604[23] = 0.0;
   out_6308920866961272604[24] = 0.0;
   out_6308920866961272604[25] = 0.0;
   out_6308920866961272604[26] = 0.0;
   out_6308920866961272604[27] = 1.0;
   out_6308920866961272604[28] = 0.0;
   out_6308920866961272604[29] = 0.0;
   out_6308920866961272604[30] = 0.0;
   out_6308920866961272604[31] = 0.0;
   out_6308920866961272604[32] = 0.0;
   out_6308920866961272604[33] = 0.0;
   out_6308920866961272604[34] = 0.0;
   out_6308920866961272604[35] = 0.0;
   out_6308920866961272604[36] = 1.0;
   out_6308920866961272604[37] = 0.0;
   out_6308920866961272604[38] = 0.0;
   out_6308920866961272604[39] = 0.0;
   out_6308920866961272604[40] = 0.0;
   out_6308920866961272604[41] = 0.0;
   out_6308920866961272604[42] = 0.0;
   out_6308920866961272604[43] = 0.0;
   out_6308920866961272604[44] = 0.0;
   out_6308920866961272604[45] = 1.0;
   out_6308920866961272604[46] = 0.0;
   out_6308920866961272604[47] = 0.0;
   out_6308920866961272604[48] = 0.0;
   out_6308920866961272604[49] = 0.0;
   out_6308920866961272604[50] = 0.0;
   out_6308920866961272604[51] = 0.0;
   out_6308920866961272604[52] = 0.0;
   out_6308920866961272604[53] = 0.0;
   out_6308920866961272604[54] = 1.0;
   out_6308920866961272604[55] = 0.0;
   out_6308920866961272604[56] = 0.0;
   out_6308920866961272604[57] = 0.0;
   out_6308920866961272604[58] = 0.0;
   out_6308920866961272604[59] = 0.0;
   out_6308920866961272604[60] = 0.0;
   out_6308920866961272604[61] = 0.0;
   out_6308920866961272604[62] = 0.0;
   out_6308920866961272604[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_3308595574725787248) {
   out_3308595574725787248[0] = state[0];
   out_3308595574725787248[1] = state[1];
   out_3308595574725787248[2] = state[2];
   out_3308595574725787248[3] = state[3];
   out_3308595574725787248[4] = state[4];
   out_3308595574725787248[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_3308595574725787248[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_3308595574725787248[7] = state[7];
}
void F_fun(double *state, double dt, double *out_4160370806524182836) {
   out_4160370806524182836[0] = 1;
   out_4160370806524182836[1] = 0;
   out_4160370806524182836[2] = 0;
   out_4160370806524182836[3] = 0;
   out_4160370806524182836[4] = 0;
   out_4160370806524182836[5] = 0;
   out_4160370806524182836[6] = 0;
   out_4160370806524182836[7] = 0;
   out_4160370806524182836[8] = 0;
   out_4160370806524182836[9] = 1;
   out_4160370806524182836[10] = 0;
   out_4160370806524182836[11] = 0;
   out_4160370806524182836[12] = 0;
   out_4160370806524182836[13] = 0;
   out_4160370806524182836[14] = 0;
   out_4160370806524182836[15] = 0;
   out_4160370806524182836[16] = 0;
   out_4160370806524182836[17] = 0;
   out_4160370806524182836[18] = 1;
   out_4160370806524182836[19] = 0;
   out_4160370806524182836[20] = 0;
   out_4160370806524182836[21] = 0;
   out_4160370806524182836[22] = 0;
   out_4160370806524182836[23] = 0;
   out_4160370806524182836[24] = 0;
   out_4160370806524182836[25] = 0;
   out_4160370806524182836[26] = 0;
   out_4160370806524182836[27] = 1;
   out_4160370806524182836[28] = 0;
   out_4160370806524182836[29] = 0;
   out_4160370806524182836[30] = 0;
   out_4160370806524182836[31] = 0;
   out_4160370806524182836[32] = 0;
   out_4160370806524182836[33] = 0;
   out_4160370806524182836[34] = 0;
   out_4160370806524182836[35] = 0;
   out_4160370806524182836[36] = 1;
   out_4160370806524182836[37] = 0;
   out_4160370806524182836[38] = 0;
   out_4160370806524182836[39] = 0;
   out_4160370806524182836[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_4160370806524182836[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_4160370806524182836[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4160370806524182836[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4160370806524182836[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_4160370806524182836[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_4160370806524182836[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_4160370806524182836[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_4160370806524182836[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4160370806524182836[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4160370806524182836[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4160370806524182836[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4160370806524182836[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4160370806524182836[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4160370806524182836[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4160370806524182836[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4160370806524182836[56] = 0;
   out_4160370806524182836[57] = 0;
   out_4160370806524182836[58] = 0;
   out_4160370806524182836[59] = 0;
   out_4160370806524182836[60] = 0;
   out_4160370806524182836[61] = 0;
   out_4160370806524182836[62] = 0;
   out_4160370806524182836[63] = 1;
}
void h_25(double *state, double *unused, double *out_8731921353915695165) {
   out_8731921353915695165[0] = state[6];
}
void H_25(double *state, double *unused, double *out_3358246505016874748) {
   out_3358246505016874748[0] = 0;
   out_3358246505016874748[1] = 0;
   out_3358246505016874748[2] = 0;
   out_3358246505016874748[3] = 0;
   out_3358246505016874748[4] = 0;
   out_3358246505016874748[5] = 0;
   out_3358246505016874748[6] = 1;
   out_3358246505016874748[7] = 0;
}
void h_24(double *state, double *unused, double *out_291084043109330954) {
   out_291084043109330954[0] = state[4];
   out_291084043109330954[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8184636798247949712) {
   out_8184636798247949712[0] = 0;
   out_8184636798247949712[1] = 0;
   out_8184636798247949712[2] = 0;
   out_8184636798247949712[3] = 0;
   out_8184636798247949712[4] = 1;
   out_8184636798247949712[5] = 0;
   out_8184636798247949712[6] = 0;
   out_8184636798247949712[7] = 0;
   out_8184636798247949712[8] = 0;
   out_8184636798247949712[9] = 0;
   out_8184636798247949712[10] = 0;
   out_8184636798247949712[11] = 0;
   out_8184636798247949712[12] = 0;
   out_8184636798247949712[13] = 1;
   out_8184636798247949712[14] = 0;
   out_8184636798247949712[15] = 0;
}
void h_30(double *state, double *unused, double *out_303723421636649087) {
   out_303723421636649087[0] = state[4];
}
void H_30(double *state, double *unused, double *out_6693907007073320002) {
   out_6693907007073320002[0] = 0;
   out_6693907007073320002[1] = 0;
   out_6693907007073320002[2] = 0;
   out_6693907007073320002[3] = 0;
   out_6693907007073320002[4] = 1;
   out_6693907007073320002[5] = 0;
   out_6693907007073320002[6] = 0;
   out_6693907007073320002[7] = 0;
}
void h_26(double *state, double *unused, double *out_6620125193237786428) {
   out_6620125193237786428[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5876036152884916164) {
   out_5876036152884916164[0] = 0;
   out_5876036152884916164[1] = 0;
   out_5876036152884916164[2] = 0;
   out_5876036152884916164[3] = 0;
   out_5876036152884916164[4] = 0;
   out_5876036152884916164[5] = 0;
   out_5876036152884916164[6] = 0;
   out_5876036152884916164[7] = 1;
}
void h_27(double *state, double *unused, double *out_1541777291655224576) {
   out_1541777291655224576[0] = state[3];
}
void H_27(double *state, double *unused, double *out_7363913394360118048) {
   out_7363913394360118048[0] = 0;
   out_7363913394360118048[1] = 0;
   out_7363913394360118048[2] = 0;
   out_7363913394360118048[3] = 1;
   out_7363913394360118048[4] = 0;
   out_7363913394360118048[5] = 0;
   out_7363913394360118048[6] = 0;
   out_7363913394360118048[7] = 0;
}
void h_29(double *state, double *unused, double *out_1735585773123112638) {
   out_1735585773123112638[0] = state[1];
}
void H_29(double *state, double *unused, double *out_6594505521491184084) {
   out_6594505521491184084[0] = 0;
   out_6594505521491184084[1] = 1;
   out_6594505521491184084[2] = 0;
   out_6594505521491184084[3] = 0;
   out_6594505521491184084[4] = 0;
   out_6594505521491184084[5] = 0;
   out_6594505521491184084[6] = 0;
   out_6594505521491184084[7] = 0;
}
void h_28(double *state, double *unused, double *out_5837464036911225666) {
   out_5837464036911225666[0] = state[5];
   out_5837464036911225666[1] = state[6];
}
void H_28(double *state, double *unused, double *out_1866410131138996922) {
   out_1866410131138996922[0] = 0;
   out_1866410131138996922[1] = 0;
   out_1866410131138996922[2] = 0;
   out_1866410131138996922[3] = 0;
   out_1866410131138996922[4] = 0;
   out_1866410131138996922[5] = 1;
   out_1866410131138996922[6] = 0;
   out_1866410131138996922[7] = 0;
   out_1866410131138996922[8] = 0;
   out_1866410131138996922[9] = 0;
   out_1866410131138996922[10] = 0;
   out_1866410131138996922[11] = 0;
   out_1866410131138996922[12] = 0;
   out_1866410131138996922[13] = 0;
   out_1866410131138996922[14] = 1;
   out_1866410131138996922[15] = 0;
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
