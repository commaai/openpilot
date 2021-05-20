#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_6392157582760322357);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_3866323338818758786);
void car_H_mod_fun(double *state, double *out_6503029791175311535);
void car_f_fun(double *state, double dt, double *out_7152810423689958583);
void car_F_fun(double *state, double dt, double *out_8168263035060162837);
void car_h_25(double *state, double *unused, double *out_3988360368053177208);
void car_H_25(double *state, double *unused, double *out_3893225050488880445);
void car_h_24(double *state, double *unused, double *out_7142823418347609148);
void car_H_24(double *state, double *unused, double *out_2581754386723611109);
void car_h_30(double *state, double *unused, double *out_5320666420972523226);
void car_H_30(double *state, double *unused, double *out_5720673860460302835);
void car_h_26(double *state, double *unused, double *out_3289596398024394460);
void car_H_26(double *state, double *unused, double *out_8848925868281461898);
void car_h_27(double *state, double *unused, double *out_5572926824374950833);
void car_H_27(double *state, double *unused, double *out_7008255848296928147);
void car_h_29(double *state, double *unused, double *out_4874996376036170089);
void car_H_29(double *state, double *unused, double *out_5146115246604766660);
void car_h_28(double *state, double *unused, double *out_3302752609494473091);
void car_H_28(double *state, double *unused, double *out_8847157208749445641);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}