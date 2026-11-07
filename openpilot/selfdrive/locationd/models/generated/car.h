#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_3664446416432096692);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2220382408259350967);
void car_H_mod_fun(double *state, double *out_5101481675894590140);
void car_f_fun(double *state, double dt, double *out_1374429393672774254);
void car_F_fun(double *state, double dt, double *out_2233291611378471363);
void car_h_25(double *state, double *unused, double *out_3490619282190537336);
void car_H_25(double *state, double *unused, double *out_1578501025905098463);
void car_h_24(double *state, double *unused, double *out_1487557793108610185);
void car_H_24(double *state, double *unused, double *out_3751150624910598029);
void car_h_30(double *state, double *unused, double *out_8164170727459411353);
void car_H_30(double *state, double *unused, double *out_5338189315586518292);
void car_h_26(double *state, double *unused, double *out_3162497459798565708);
void car_H_26(double *state, double *unused, double *out_5320004344779154687);
void car_h_27(double *state, double *unused, double *out_886213961381839685);
void car_H_27(double *state, double *unused, double *out_3163426003786093381);
void car_h_29(double *state, double *unused, double *out_729027293030710540);
void car_H_29(double *state, double *unused, double *out_1450063276916542348);
void car_h_28(double *state, double *unused, double *out_6369929755711612024);
void car_H_28(double *state, double *unused, double *out_3632335740152988226);
void car_h_31(double *state, double *unused, double *out_2664827913650122793);
void car_H_31(double *state, double *unused, double *out_1547855064028138035);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}