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
void car_err_fun(double *nom_x, double *delta_x, double *out_9092401817951937801);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6018369249920624325);
void car_H_mod_fun(double *state, double *out_2017416824247113709);
void car_f_fun(double *state, double dt, double *out_9091538470711422403);
void car_F_fun(double *state, double dt, double *out_6396802349455434445);
void car_h_25(double *state, double *unused, double *out_7942966278303983884);
void car_H_25(double *state, double *unused, double *out_6921108649431778549);
void car_h_24(double *state, double *unused, double *out_6744365657582572191);
void car_H_24(double *state, double *unused, double *out_3367747875231105820);
void car_h_30(double *state, double *unused, double *out_7785779609952854739);
void car_H_30(double *state, double *unused, double *out_9007302465770524440);
void car_h_26(double *state, double *unused, double *out_2784106342292009094);
void car_H_26(double *state, double *unused, double *out_3179605330557722325);
void car_h_27(double *state, double *unused, double *out_507822843875283071);
void car_H_27(double *state, double *unused, double *out_7264678296138602265);
void car_h_29(double *state, double *unused, double *out_1864654999490699737);
void car_H_29(double *state, double *unused, double *out_8497071121456132256);
void car_h_28(double *state, double *unused, double *out_435687009026379364);
void car_H_28(double *state, double *unused, double *out_4867273935183888786);
void car_h_31(double *state, double *unused, double *out_7667772216019477995);
void car_H_31(double *state, double *unused, double *out_2553397228324370849);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}