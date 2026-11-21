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
void car_err_fun(double *nom_x, double *delta_x, double *out_2996852071044234202);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5428601996215229963);
void car_H_mod_fun(double *state, double *out_6840182513517023115);
void car_f_fun(double *state, double dt, double *out_4503671108966703887);
void car_F_fun(double *state, double dt, double *out_4770600575889623320);
void car_h_25(double *state, double *unused, double *out_5459418218260901103);
void car_H_25(double *state, double *unused, double *out_1936490688332358275);
void car_h_24(double *state, double *unused, double *out_4053235228616768809);
void car_H_24(double *state, double *unused, double *out_2333135992686217770);
void car_h_30(double *state, double *unused, double *out_5302231549909771958);
void car_H_30(double *state, double *unused, double *out_4980199653159258480);
void car_h_26(double *state, double *unused, double *out_4698915665233294441);
void car_H_26(double *state, double *unused, double *out_5677994007206414499);
void car_h_27(double *state, double *unused, double *out_8561964940150418980);
void car_H_27(double *state, double *unused, double *out_2805436341358833569);
void car_h_29(double *state, double *unused, double *out_4358721697008109523);
void car_H_29(double *state, double *unused, double *out_1092073614489282536);
void car_h_28(double *state, double *unused, double *out_1479122313914905983);
void car_H_28(double *state, double *unused, double *out_3990325402580248038);
void car_h_31(double *state, double *unused, double *out_1402015568006224056);
void car_H_31(double *state, double *unused, double *out_1905844726455397847);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}