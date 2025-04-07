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
void car_err_fun(double *nom_x, double *delta_x, double *out_7345504276725310336);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_1483419534327179826);
void car_H_mod_fun(double *state, double *out_5980828091480380917);
void car_f_fun(double *state, double dt, double *out_6771744444959754452);
void car_F_fun(double *state, double dt, double *out_8079006821062657786);
void car_h_25(double *state, double *unused, double *out_6013514099318955817);
void car_H_25(double *state, double *unused, double *out_8892173204326502465);
void car_h_24(double *state, double *unused, double *out_6766398854122891435);
void car_H_24(double *state, double *unused, double *out_7204356768457398274);
void car_h_30(double *state, double *unused, double *out_5662394748040244863);
void car_H_30(double *state, double *unused, double *out_5026874539255440953);
void car_h_26(double *state, double *unused, double *out_253236005693617473);
void car_H_26(double *state, double *unused, double *out_5813067550508992927);
void car_h_27(double *state, double *unused, double *out_6177405252431548545);
void car_H_27(double *state, double *unused, double *out_2852111227455016042);
void car_h_29(double *state, double *unused, double *out_8631889293023980721);
void car_H_29(double *state, double *unused, double *out_5537105883569833137);
void car_h_28(double *state, double *unused, double *out_7267665553617157426);
void car_H_28(double *state, double *unused, double *out_454706866500302563);
void car_h_31(double *state, double *unused, double *out_2626363377399586067);
void car_H_31(double *state, double *unused, double *out_5186859448275641451);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}