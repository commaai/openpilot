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
void car_err_fun(double *nom_x, double *delta_x, double *out_2846118138154075095);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4044333420406863401);
void car_H_mod_fun(double *state, double *out_698882326502663349);
void car_f_fun(double *state, double dt, double *out_7869785476666687441);
void car_F_fun(double *state, double dt, double *out_8888598582008610072);
void car_h_25(double *state, double *unused, double *out_4497380850200226036);
void car_H_25(double *state, double *unused, double *out_2286274964032937428);
void car_h_24(double *state, double *unused, double *out_1105660614370515943);
void car_H_24(double *state, double *unused, double *out_2987219318880075166);
void car_h_30(double *state, double *unused, double *out_4340194181849096891);
void car_H_30(double *state, double *unused, double *out_232057994474311199);
void car_h_26(double *state, double *unused, double *out_7376980233293429576);
void car_H_26(double *state, double *unused, double *out_6027778282906993652);
void car_h_27(double *state, double *unused, double *out_7157120249877593801);
void car_H_27(double *state, double *unused, double *out_2455652065658254416);
void car_h_29(double *state, double *unused, double *out_3396684328947434456);
void car_H_29(double *state, double *unused, double *out_742289338788703383);
void car_h_28(double *state, double *unused, double *out_517084945854230916);
void car_H_28(double *state, double *unused, double *out_4340109678280827191);
void car_h_31(double *state, double *unused, double *out_5309485746790783696);
void car_H_31(double *state, double *unused, double *out_6653986385140345128);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}