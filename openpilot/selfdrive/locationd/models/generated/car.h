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
void car_err_fun(double *nom_x, double *delta_x, double *out_2927168366743118455);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_633536280694991254);
void car_H_mod_fun(double *state, double *out_3756852632545933054);
void car_f_fun(double *state, double dt, double *out_1622189069467381624);
void car_F_fun(double *state, double dt, double *out_1503549483011366244);
void car_h_25(double *state, double *unused, double *out_921512697149304703);
void car_H_25(double *state, double *unused, double *out_5496703215251381325);
void car_h_24(double *state, double *unused, double *out_2538808957834133388);
void car_H_24(double *state, double *unused, double *out_3324053616245881759);
void car_h_30(double *state, double *unused, double *out_2715936831104717647);
void car_H_30(double *state, double *unused, double *out_8015036173758629952);
void car_h_26(double *state, double *unused, double *out_4838765396476033641);
void car_H_26(double *state, double *unused, double *out_8801229185012181926);
void car_h_27(double *state, double *unused, double *out_2089256498225446440);
void car_H_27(double *state, double *unused, double *out_8208113828766978447);
void car_h_29(double *state, double *unused, double *out_7717610098765563292);
void car_H_29(double *state, double *unused, double *out_8525267518073022136);
void car_h_28(double *state, double *unused, double *out_4901808601495299823);
void car_H_28(double *state, double *unused, double *out_3442868501003491562);
void car_h_31(double *state, double *unused, double *out_320094539535941149);
void car_H_31(double *state, double *unused, double *out_5873365607946353038);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}