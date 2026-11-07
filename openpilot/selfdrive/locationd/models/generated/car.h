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
void car_err_fun(double *nom_x, double *delta_x, double *out_6912122597548111650);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5839740732604401630);
void car_H_mod_fun(double *state, double *out_5459200475037534350);
void car_f_fun(double *state, double dt, double *out_7919248626531568362);
void car_F_fun(double *state, double dt, double *out_8728212283248373813);
void car_h_25(double *state, double *unused, double *out_7012538561194516599);
void car_H_25(double *state, double *unused, double *out_4510079553009052141);
void car_h_24(double *state, double *unused, double *out_3347341305780913336);
void car_H_24(double *state, double *unused, double *out_523191577009633836);
void car_h_30(double *state, double *unused, double *out_4663370286164684081);
void car_H_30(double *state, double *unused, double *out_9037775883136660339);
void car_h_26(double *state, double *unused, double *out_6353818120900365041);
void car_H_26(double *state, double *unused, double *out_8251582871883108365);
void car_h_27(double *state, double *unused, double *out_2010865293533670954);
void car_H_27(double *state, double *unused, double *out_7234204878772466366);
void car_h_29(double *state, double *unused, double *out_3303742755155314253);
void car_H_29(double *state, double *unused, double *out_8527544538822268155);
void car_h_28(double *state, double *unused, double *out_2954375146435333389);
void car_H_28(double *state, double *unused, double *out_4836800517817752887);
void car_h_31(double *state, double *unused, double *out_5561907168565633432);
void car_H_31(double *state, double *unused, double *out_8877790974116459841);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}