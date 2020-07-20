/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_4186435242622652043);
void inv_err_fun(double *nom_x, double *true_x, double *out_1772598286649612301);
void H_mod_fun(double *state, double *out_7982322623617124930);
void f_fun(double *state, double dt, double *out_8672621219632901476);
void F_fun(double *state, double dt, double *out_6946257997102710784);
void h_25(double *state, double *unused, double *out_4034598713074493671);
void H_25(double *state, double *unused, double *out_392631468657234558);
void h_24(double *state, double *unused, double *out_85866795352703332);
void H_24(double *state, double *unused, double *out_280208781190912538);
void h_30(double *state, double *unused, double *out_336119608556182619);
void H_30(double *state, double *unused, double *out_3691700952709061188);
void h_26(double *state, double *unused, double *out_2927538577387815664);
void H_26(double *state, double *unused, double *out_6767758879922125158);
void h_27(double *state, double *unused, double *out_3160421697585685380);
void H_27(double *state, double *unused, double *out_3779149164388235766);
void h_29(double *state, double *unused, double *out_5643797306792045766);
void H_29(double *state, double *unused, double *out_1912218037297707854);
void h_28(double *state, double *unused, double *out_6319867348620254732);
void H_28(double *state, double *unused, double *out_6072820350638133456);
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
void set_mass(double x);

void set_rotational_inertia(double x);

void set_center_to_front(double x);

void set_center_to_rear(double x);

void set_stiffness_front(double x);

void set_stiffness_rear(double x);
