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
void car_err_fun(double *nom_x, double *delta_x, double *out_6490269132271391301);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7265692863643532060);
void car_H_mod_fun(double *state, double *out_5640827172746054554);
void car_f_fun(double *state, double dt, double *out_3709053776412774313);
void car_F_fun(double *state, double dt, double *out_3639689452659227219);
void car_h_25(double *state, double *unused, double *out_2141167549731806248);
void car_H_25(double *state, double *unused, double *out_3612728675051259825);
void car_h_24(double *state, double *unused, double *out_6454025621392139070);
void car_H_24(double *state, double *unused, double *out_1440079076045760259);
void car_h_30(double *state, double *unused, double *out_2298354218082935393);
void car_H_30(double *state, double *unused, double *out_7917325057166675036);
void car_h_26(double *state, double *unused, double *out_514197885186917990);
void car_H_26(double *state, double *unused, double *out_128774643822796399);
void car_h_27(double *state, double *unused, double *out_3304737207191273806);
void car_H_27(double *state, double *unused, double *out_1308626416107594844);
void car_h_29(double *state, double *unused, double *out_3466098019159077130);
void car_H_29(double *state, double *unused, double *out_404736310761956189);
void car_h_28(double *state, double *unused, double *out_7926923236647382120);
void car_H_28(double *state, double *unused, double *out_1558893960803370062);
void car_h_31(double *state, double *unused, double *out_6066329196351100602);
void car_H_31(double *state, double *unused, double *out_3643374636928220253);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}