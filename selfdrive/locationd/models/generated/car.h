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
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_7361038110530370221);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6642685734919517258);
void car_H_mod_fun(double *state, double *out_2530205897380259158);
void car_f_fun(double *state, double dt, double *out_8104591686044368808);
void car_F_fun(double *state, double dt, double *out_2827006867786055650);
void car_h_25(double *state, double *unused, double *out_7566196993710386620);
void car_H_25(double *state, double *unused, double *out_6410871014676703632);
void car_h_24(double *state, double *unused, double *out_7879487054421198263);
void car_H_24(double *state, double *unused, double *out_8477947460230524848);
void car_h_30(double *state, double *unused, double *out_7291002931425880731);
void car_H_30(double *state, double *unused, double *out_1883174684549095434);
void car_h_26(double *state, double *unused, double *out_1114653888218345922);
void car_H_26(double *state, double *unused, double *out_2669367695802647408);
void car_h_27(double *state, double *unused, double *out_4102018324380578519);
void car_H_27(double *state, double *unused, double *out_291588627251329477);
void car_h_29(double *state, double *unused, double *out_6687687046749403214);
void car_H_29(double *state, double *unused, double *out_2393406028863487618);
void car_h_28(double *state, double *unused, double *out_860361673300714713);
void car_H_28(double *state, double *unused, double *out_2688992988206042956);
void car_h_31(double *state, double *unused, double *out_7243097551745142646);
void car_H_31(double *state, double *unused, double *out_2043159593569295932);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}