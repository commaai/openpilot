
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
void err_fun(double *nom_x, double *delta_x, double *out_4186435242622652043) {
   out_4186435242622652043[0] = delta_x[0] + nom_x[0];
   out_4186435242622652043[1] = delta_x[1] + nom_x[1];
   out_4186435242622652043[2] = delta_x[2] + nom_x[2];
   out_4186435242622652043[3] = delta_x[3] + nom_x[3];
   out_4186435242622652043[4] = delta_x[4] + nom_x[4];
   out_4186435242622652043[5] = delta_x[5] + nom_x[5];
   out_4186435242622652043[6] = delta_x[6] + nom_x[6];
   out_4186435242622652043[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1772598286649612301) {
   out_1772598286649612301[0] = -nom_x[0] + true_x[0];
   out_1772598286649612301[1] = -nom_x[1] + true_x[1];
   out_1772598286649612301[2] = -nom_x[2] + true_x[2];
   out_1772598286649612301[3] = -nom_x[3] + true_x[3];
   out_1772598286649612301[4] = -nom_x[4] + true_x[4];
   out_1772598286649612301[5] = -nom_x[5] + true_x[5];
   out_1772598286649612301[6] = -nom_x[6] + true_x[6];
   out_1772598286649612301[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_7982322623617124930) {
   out_7982322623617124930[0] = 1.0;
   out_7982322623617124930[1] = 0.0;
   out_7982322623617124930[2] = 0.0;
   out_7982322623617124930[3] = 0.0;
   out_7982322623617124930[4] = 0.0;
   out_7982322623617124930[5] = 0.0;
   out_7982322623617124930[6] = 0.0;
   out_7982322623617124930[7] = 0.0;
   out_7982322623617124930[8] = 0.0;
   out_7982322623617124930[9] = 1.0;
   out_7982322623617124930[10] = 0.0;
   out_7982322623617124930[11] = 0.0;
   out_7982322623617124930[12] = 0.0;
   out_7982322623617124930[13] = 0.0;
   out_7982322623617124930[14] = 0.0;
   out_7982322623617124930[15] = 0.0;
   out_7982322623617124930[16] = 0.0;
   out_7982322623617124930[17] = 0.0;
   out_7982322623617124930[18] = 1.0;
   out_7982322623617124930[19] = 0.0;
   out_7982322623617124930[20] = 0.0;
   out_7982322623617124930[21] = 0.0;
   out_7982322623617124930[22] = 0.0;
   out_7982322623617124930[23] = 0.0;
   out_7982322623617124930[24] = 0.0;
   out_7982322623617124930[25] = 0.0;
   out_7982322623617124930[26] = 0.0;
   out_7982322623617124930[27] = 1.0;
   out_7982322623617124930[28] = 0.0;
   out_7982322623617124930[29] = 0.0;
   out_7982322623617124930[30] = 0.0;
   out_7982322623617124930[31] = 0.0;
   out_7982322623617124930[32] = 0.0;
   out_7982322623617124930[33] = 0.0;
   out_7982322623617124930[34] = 0.0;
   out_7982322623617124930[35] = 0.0;
   out_7982322623617124930[36] = 1.0;
   out_7982322623617124930[37] = 0.0;
   out_7982322623617124930[38] = 0.0;
   out_7982322623617124930[39] = 0.0;
   out_7982322623617124930[40] = 0.0;
   out_7982322623617124930[41] = 0.0;
   out_7982322623617124930[42] = 0.0;
   out_7982322623617124930[43] = 0.0;
   out_7982322623617124930[44] = 0.0;
   out_7982322623617124930[45] = 1.0;
   out_7982322623617124930[46] = 0.0;
   out_7982322623617124930[47] = 0.0;
   out_7982322623617124930[48] = 0.0;
   out_7982322623617124930[49] = 0.0;
   out_7982322623617124930[50] = 0.0;
   out_7982322623617124930[51] = 0.0;
   out_7982322623617124930[52] = 0.0;
   out_7982322623617124930[53] = 0.0;
   out_7982322623617124930[54] = 1.0;
   out_7982322623617124930[55] = 0.0;
   out_7982322623617124930[56] = 0.0;
   out_7982322623617124930[57] = 0.0;
   out_7982322623617124930[58] = 0.0;
   out_7982322623617124930[59] = 0.0;
   out_7982322623617124930[60] = 0.0;
   out_7982322623617124930[61] = 0.0;
   out_7982322623617124930[62] = 0.0;
   out_7982322623617124930[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_8672621219632901476) {
   out_8672621219632901476[0] = state[0];
   out_8672621219632901476[1] = state[1];
   out_8672621219632901476[2] = state[2];
   out_8672621219632901476[3] = state[3];
   out_8672621219632901476[4] = state[4];
   out_8672621219632901476[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_8672621219632901476[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_8672621219632901476[7] = state[7];
}
void F_fun(double *state, double dt, double *out_6946257997102710784) {
   out_6946257997102710784[0] = 1;
   out_6946257997102710784[1] = 0;
   out_6946257997102710784[2] = 0;
   out_6946257997102710784[3] = 0;
   out_6946257997102710784[4] = 0;
   out_6946257997102710784[5] = 0;
   out_6946257997102710784[6] = 0;
   out_6946257997102710784[7] = 0;
   out_6946257997102710784[8] = 0;
   out_6946257997102710784[9] = 1;
   out_6946257997102710784[10] = 0;
   out_6946257997102710784[11] = 0;
   out_6946257997102710784[12] = 0;
   out_6946257997102710784[13] = 0;
   out_6946257997102710784[14] = 0;
   out_6946257997102710784[15] = 0;
   out_6946257997102710784[16] = 0;
   out_6946257997102710784[17] = 0;
   out_6946257997102710784[18] = 1;
   out_6946257997102710784[19] = 0;
   out_6946257997102710784[20] = 0;
   out_6946257997102710784[21] = 0;
   out_6946257997102710784[22] = 0;
   out_6946257997102710784[23] = 0;
   out_6946257997102710784[24] = 0;
   out_6946257997102710784[25] = 0;
   out_6946257997102710784[26] = 0;
   out_6946257997102710784[27] = 1;
   out_6946257997102710784[28] = 0;
   out_6946257997102710784[29] = 0;
   out_6946257997102710784[30] = 0;
   out_6946257997102710784[31] = 0;
   out_6946257997102710784[32] = 0;
   out_6946257997102710784[33] = 0;
   out_6946257997102710784[34] = 0;
   out_6946257997102710784[35] = 0;
   out_6946257997102710784[36] = 1;
   out_6946257997102710784[37] = 0;
   out_6946257997102710784[38] = 0;
   out_6946257997102710784[39] = 0;
   out_6946257997102710784[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_6946257997102710784[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_6946257997102710784[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6946257997102710784[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6946257997102710784[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_6946257997102710784[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_6946257997102710784[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_6946257997102710784[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_6946257997102710784[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_6946257997102710784[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_6946257997102710784[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6946257997102710784[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6946257997102710784[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_6946257997102710784[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_6946257997102710784[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_6946257997102710784[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6946257997102710784[56] = 0;
   out_6946257997102710784[57] = 0;
   out_6946257997102710784[58] = 0;
   out_6946257997102710784[59] = 0;
   out_6946257997102710784[60] = 0;
   out_6946257997102710784[61] = 0;
   out_6946257997102710784[62] = 0;
   out_6946257997102710784[63] = 1;
}
void h_25(double *state, double *unused, double *out_4034598713074493671) {
   out_4034598713074493671[0] = state[6];
}
void H_25(double *state, double *unused, double *out_392631468657234558) {
   out_392631468657234558[0] = 0;
   out_392631468657234558[1] = 0;
   out_392631468657234558[2] = 0;
   out_392631468657234558[3] = 0;
   out_392631468657234558[4] = 0;
   out_392631468657234558[5] = 0;
   out_392631468657234558[6] = 1;
   out_392631468657234558[7] = 0;
}
void h_24(double *state, double *unused, double *out_85866795352703332) {
   out_85866795352703332[0] = state[4];
   out_85866795352703332[1] = state[5];
}
void H_24(double *state, double *unused, double *out_280208781190912538) {
   out_280208781190912538[0] = 0;
   out_280208781190912538[1] = 0;
   out_280208781190912538[2] = 0;
   out_280208781190912538[3] = 0;
   out_280208781190912538[4] = 1;
   out_280208781190912538[5] = 0;
   out_280208781190912538[6] = 0;
   out_280208781190912538[7] = 0;
   out_280208781190912538[8] = 0;
   out_280208781190912538[9] = 0;
   out_280208781190912538[10] = 0;
   out_280208781190912538[11] = 0;
   out_280208781190912538[12] = 0;
   out_280208781190912538[13] = 1;
   out_280208781190912538[14] = 0;
   out_280208781190912538[15] = 0;
}
void h_30(double *state, double *unused, double *out_336119608556182619) {
   out_336119608556182619[0] = state[4];
}
void H_30(double *state, double *unused, double *out_3691700952709061188) {
   out_3691700952709061188[0] = 0;
   out_3691700952709061188[1] = 0;
   out_3691700952709061188[2] = 0;
   out_3691700952709061188[3] = 0;
   out_3691700952709061188[4] = 1;
   out_3691700952709061188[5] = 0;
   out_3691700952709061188[6] = 0;
   out_3691700952709061188[7] = 0;
}
void h_26(double *state, double *unused, double *out_2927538577387815664) {
   out_2927538577387815664[0] = state[7];
}
void H_26(double *state, double *unused, double *out_6767758879922125158) {
   out_6767758879922125158[0] = 0;
   out_6767758879922125158[1] = 0;
   out_6767758879922125158[2] = 0;
   out_6767758879922125158[3] = 0;
   out_6767758879922125158[4] = 0;
   out_6767758879922125158[5] = 0;
   out_6767758879922125158[6] = 0;
   out_6767758879922125158[7] = 1;
}
void h_27(double *state, double *unused, double *out_3160421697585685380) {
   out_3160421697585685380[0] = state[3];
}
void H_27(double *state, double *unused, double *out_3779149164388235766) {
   out_3779149164388235766[0] = 0;
   out_3779149164388235766[1] = 0;
   out_3779149164388235766[2] = 0;
   out_3779149164388235766[3] = 1;
   out_3779149164388235766[4] = 0;
   out_3779149164388235766[5] = 0;
   out_3779149164388235766[6] = 0;
   out_3779149164388235766[7] = 0;
}
void h_29(double *state, double *unused, double *out_5643797306792045766) {
   out_5643797306792045766[0] = state[1];
}
void H_29(double *state, double *unused, double *out_1912218037297707854) {
   out_1912218037297707854[0] = 0;
   out_1912218037297707854[1] = 1;
   out_1912218037297707854[2] = 0;
   out_1912218037297707854[3] = 0;
   out_1912218037297707854[4] = 0;
   out_1912218037297707854[5] = 0;
   out_1912218037297707854[6] = 0;
   out_1912218037297707854[7] = 0;
}
void h_28(double *state, double *unused, double *out_6319867348620254732) {
   out_6319867348620254732[0] = state[5];
   out_6319867348620254732[1] = state[6];
}
void H_28(double *state, double *unused, double *out_6072820350638133456) {
   out_6072820350638133456[0] = 0;
   out_6072820350638133456[1] = 0;
   out_6072820350638133456[2] = 0;
   out_6072820350638133456[3] = 0;
   out_6072820350638133456[4] = 0;
   out_6072820350638133456[5] = 1;
   out_6072820350638133456[6] = 0;
   out_6072820350638133456[7] = 0;
   out_6072820350638133456[8] = 0;
   out_6072820350638133456[9] = 0;
   out_6072820350638133456[10] = 0;
   out_6072820350638133456[11] = 0;
   out_6072820350638133456[12] = 0;
   out_6072820350638133456[13] = 0;
   out_6072820350638133456[14] = 1;
   out_6072820350638133456[15] = 0;
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
