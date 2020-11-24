/******************************************************************************
 *                      Code generated with sympy 1.6.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_5976358665325250733);
void inv_err_fun(double *nom_x, double *true_x, double *out_545149083074471775);
void H_mod_fun(double *state, double *out_6429485867732329304);
void f_fun(double *state, double dt, double *out_1814214989434170929);
void F_fun(double *state, double dt, double *out_24478973576452139);
void h_25(double *state, double *unused, double *out_6753750671493724343);
void H_25(double *state, double *unused, double *out_505484718201953462);
void h_24(double *state, double *unused, double *out_7677385143792155145);
void H_24(double *state, double *unused, double *out_1767873782397693594);
void h_30(double *state, double *unused, double *out_6793389117300471979);
void H_30(double *state, double *unused, double *out_4939972497977953670);
void h_26(double *state, double *unused, double *out_6190073232623994462);
void H_26(double *state, double *unused, double *out_5234308798478062218);
void h_27(double *state, double *unused, double *out_6050263531134870438);
void H_27(double *state, double *unused, double *out_8050747893125696486);
void h_29(double *state, double *unused, double *out_2829636658126479976);
void H_29(double *state, double *unused, double *out_5514531111833489845);
void h_28(double *state, double *unused, double *out_5669571770271976550);
void H_28(double *state, double *unused, double *out_8785706260634188460);
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
