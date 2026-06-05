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
void car_err_fun(double *nom_x, double *delta_x, double *out_8735429740637832661);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7497417333367825202);
void car_H_mod_fun(double *state, double *out_3386214859166967837);
void car_f_fun(double *state, double dt, double *out_4508993452591091797);
void car_F_fun(double *state, double dt, double *out_1336565245694139133);
void car_h_25(double *state, double *unused, double *out_2296868611573953905);
void car_H_25(double *state, double *unused, double *out_1408615983760901272);
void car_h_24(double *state, double *unused, double *out_3819261768936520190);
void car_H_24(double *state, double *unused, double *out_3581265582766400838);
void car_h_30(double *state, double *unused, double *out_9067703837924304841);
void car_H_30(double *state, double *unused, double *out_1109716974746347355);
void car_h_26(double *state, double *unused, double *out_4377366968124401130);
void car_H_26(double *state, double *unused, double *out_1895909985999899329);
void car_h_27(double *state, double *unused, double *out_2704804656086891740);
void car_H_27(double *state, double *unused, double *out_1065046337054077556);
void car_h_29(double *state, double *unused, double *out_6566488132515393392);
void car_H_29(double *state, double *unused, double *out_1619948319060739539);
void car_h_28(double *state, double *unused, double *out_7771222174805234463);
void car_H_28(double *state, double *unused, double *out_3462450698008791035);
void car_h_31(double *state, double *unused, double *out_5244594601929438866);
void car_H_31(double *state, double *unused, double *out_1269701883766547853);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}