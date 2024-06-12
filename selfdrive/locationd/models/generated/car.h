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
void car_err_fun(double *nom_x, double *delta_x, double *out_2637591915669889113);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7341338816872437038);
void car_H_mod_fun(double *state, double *out_3673445024695914894);
void car_f_fun(double *state, double dt, double *out_8303322641241644895);
void car_F_fun(double *state, double dt, double *out_8675149220450121470);
void car_h_25(double *state, double *unused, double *out_6066814403066876528);
void car_H_25(double *state, double *unused, double *out_132270908145005372);
void car_h_24(double *state, double *unused, double *out_186230791854934322);
void car_H_24(double *state, double *unused, double *out_2040378690860494194);
void car_h_30(double *state, double *unused, double *out_8871672977068494370);
void car_H_30(double *state, double *unused, double *out_4395425421982602826);
void car_h_26(double *state, double *unused, double *out_7271548445356717599);
void car_H_26(double *state, double *unused, double *out_3609232410729050852);
void car_h_27(double *state, double *unused, double *out_1065141135406030883);
void car_H_27(double *state, double *unused, double *out_2171831350798659609);
void car_h_29(double *state, double *unused, double *out_2796542341022470769);
void car_H_29(double *state, double *unused, double *out_3885194077668210642);
void car_h_28(double *state, double *unused, double *out_3187969700777346877);
void car_H_28(double *state, double *unused, double *out_1921563806102884391);
void car_h_31(double *state, double *unused, double *out_5792101744166379925);
void car_H_31(double *state, double *unused, double *out_162916870021965800);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}