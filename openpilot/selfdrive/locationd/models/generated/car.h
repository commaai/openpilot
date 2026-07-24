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
void car_err_fun(double *nom_x, double *delta_x, double *out_7159745659714899262);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7179799985227112400);
void car_H_mod_fun(double *state, double *out_3725331759314219676);
void car_f_fun(double *state, double dt, double *out_4172624774341273126);
void car_F_fun(double *state, double dt, double *out_1397116822022408882);
void car_h_25(double *state, double *unused, double *out_297388626426175711);
void car_H_25(double *state, double *unused, double *out_5528224088483094703);
void car_h_24(double *state, double *unused, double *out_8930218040012823637);
void car_H_24(double *state, double *unused, double *out_4293580430871014843);
void car_h_30(double *state, double *unused, double *out_1699353084576632679);
void car_H_30(double *state, double *unused, double *out_1000527758355486505);
void car_h_26(double *state, double *unused, double *out_5377776018040845687);
void car_H_26(double *state, double *unused, double *out_1786720769609038479);
void car_h_27(double *state, double *unused, double *out_4704284641234669934);
void car_H_27(double *state, double *unused, double *out_3224121829539429722);
void car_h_29(double *state, double *unused, double *out_4547097972883540789);
void car_H_29(double *state, double *unused, double *out_1510759102669878689);
void car_h_28(double *state, double *unused, double *out_2083616267898044642);
void car_H_28(double *state, double *unused, double *out_3571639914399651885);
void car_h_31(double *state, double *unused, double *out_1153242766202707456);
void car_H_31(double *state, double *unused, double *out_5558870050360055131);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}