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
void car_err_fun(double *nom_x, double *delta_x, double *out_7626122079372452194);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6779862585021228643);
void car_H_mod_fun(double *state, double *out_3551509842771640505);
void car_f_fun(double *state, double dt, double *out_5520057530326467442);
void car_F_fun(double *state, double dt, double *out_8593367633030745139);
void car_h_25(double *state, double *unused, double *out_2326995945769605353);
void car_H_25(double *state, double *unused, double *out_8455201667956305345);
void car_h_24(double *state, double *unused, double *out_5998448526015495371);
void car_H_24(double *state, double *unused, double *out_8058556363602445850);
void car_h_30(double *state, double *unused, double *out_2484182614120734498);
void car_H_30(double *state, double *unused, double *out_7473209447245997644);
void car_h_26(double *state, double *unused, double *out_7485855881781580143);
void car_H_26(double *state, double *unused, double *out_4713698349082249121);
void car_h_27(double *state, double *unused, double *out_8908838641685724752);
void car_H_27(double *state, double *unused, double *out_1752742026028272236);
void car_h_29(double *state, double *unused, double *out_9184032703970230641);
void car_H_29(double *state, double *unused, double *out_4437736682143089331);
void car_h_28(double *state, double *unused, double *out_4536232164362981219);
void car_H_28(double *state, double *unused, double *out_6401366953708415582);
void car_h_31(double *state, double *unused, double *out_4443839280580745583);
void car_H_31(double *state, double *unused, double *out_4087490246848897645);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}