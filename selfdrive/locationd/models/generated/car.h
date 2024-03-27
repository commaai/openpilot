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
void car_err_fun(double *nom_x, double *delta_x, double *out_1345357640820912998);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_565955163399411295);
void car_H_mod_fun(double *state, double *out_84820155383353969);
void car_f_fun(double *state, double dt, double *out_1203605472893796358);
void car_F_fun(double *state, double dt, double *out_7074967459219019176);
void car_h_25(double *state, double *unused, double *out_2135589314781233871);
void car_H_25(double *state, double *unused, double *out_5897105509821386881);
void car_h_24(double *state, double *unused, double *out_746443049065001390);
void car_H_24(double *state, double *unused, double *out_2488139849162399552);
void car_h_30(double *state, double *unused, double *out_8723417241257529034);
void car_H_30(double *state, double *unused, double *out_3378772551314138254);
void car_h_26(double *state, double *unused, double *out_3324285315038173436);
void car_H_26(double *state, double *unused, double *out_8808135245014108511);
void car_h_27(double *state, double *unused, double *out_6533541396062862883);
void car_H_27(double *state, double *unused, double *out_1155178480130195037);
void car_h_29(double *state, double *unused, double *out_4721653564791176937);
void car_H_29(double *state, double *unused, double *out_2868541206999746070);
void car_h_28(double *state, double *unused, double *out_5590031543161200448);
void car_H_28(double *state, double *unused, double *out_7950940224069276644);
void car_h_31(double *state, double *unused, double *out_2292775983132363016);
void car_H_31(double *state, double *unused, double *out_5866459547944426453);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}