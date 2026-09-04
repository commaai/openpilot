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
void car_err_fun(double *nom_x, double *delta_x, double *out_3156418432254414727);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_666590418838797292);
void car_H_mod_fun(double *state, double *out_6212745509768565208);
void car_f_fun(double *state, double dt, double *out_2453583229613865851);
void car_F_fun(double *state, double dt, double *out_4617154081104001635);
void car_h_25(double *state, double *unused, double *out_8034832062215994505);
void car_H_25(double *state, double *unused, double *out_2980442716143672029);
void car_h_24(double *state, double *unused, double *out_2871818575794667755);
void car_H_24(double *state, double *unused, double *out_7726952043110181063);
void car_h_30(double *state, double *unused, double *out_6313608713214185247);
void car_H_30(double *state, double *unused, double *out_3936247625347944726);
void car_h_26(double *state, double *unused, double *out_9096349309051033684);
void car_H_26(double *state, double *unused, double *out_6721946035017728253);
void car_h_27(double *state, double *unused, double *out_2167868242960558368);
void car_H_27(double *state, double *unused, double *out_1761484313547519815);
void car_h_29(double *state, double *unused, double *out_828524681873352792);
void car_H_29(double *state, double *unused, double *out_4446478969662336910);
void car_h_28(double *state, double *unused, double *out_6431616107147561991);
void car_H_28(double *state, double *unused, double *out_5034277430391561792);
void car_h_31(double *state, double *unused, double *out_4291155979720869597);
void car_H_31(double *state, double *unused, double *out_2949796754266711601);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}