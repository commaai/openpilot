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
void car_err_fun(double *nom_x, double *delta_x, double *out_7002678419470636064);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4320393458615466509);
void car_H_mod_fun(double *state, double *out_7742013015815190770);
void car_f_fun(double *state, double dt, double *out_7561579577695376867);
void car_F_fun(double *state, double dt, double *out_3970912247205260696);
void car_h_25(double *state, double *unused, double *out_416451695291826654);
void car_H_25(double *state, double *unused, double *out_4218964377169825622);
void car_h_24(double *state, double *unused, double *out_3590841845749764283);
void car_H_24(double *state, double *unused, double *out_2046314778164326056);
void car_h_30(double *state, double *unused, double *out_6354383531058524282);
void car_H_30(double *state, double *unused, double *out_7311089355048109239);
void car_h_26(double *state, double *unused, double *out_3941861940183054722);
void car_H_26(double *state, double *unused, double *out_7523490346930626223);
void car_h_27(double *state, double *unused, double *out_5418124962952672299);
void car_H_27(double *state, double *unused, double *out_8960891406861017466);
void car_h_29(double *state, double *unused, double *out_107932473598350814);
void car_H_29(double *state, double *unused, double *out_7247528679991466433);
void car_h_28(double *state, double *unused, double *out_4972454918016319162);
void car_H_28(double *state, double *unused, double *out_2165129662921935859);
void car_h_31(double *state, double *unused, double *out_1867083087920709821);
void car_H_31(double *state, double *unused, double *out_7151104446027908741);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}