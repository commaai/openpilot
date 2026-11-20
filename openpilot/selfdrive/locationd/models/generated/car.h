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
void car_err_fun(double *nom_x, double *delta_x, double *out_5852520466696422965);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_1553841940871946652);
void car_H_mod_fun(double *state, double *out_3868422467346524374);
void car_f_fun(double *state, double dt, double *out_9105879954051592652);
void car_F_fun(double *state, double dt, double *out_1694872647985395104);
void car_h_25(double *state, double *unused, double *out_4995790709594599065);
void car_H_25(double *state, double *unused, double *out_345441817357032697);
void car_h_24(double *state, double *unused, double *out_1084295506565986845);
void car_H_24(double *state, double *unused, double *out_3212483781445290439);
void car_h_30(double *state, double *unused, double *out_4720596647310093176);
void car_H_30(double *state, double *unused, double *out_6571248524134584058);
void car_h_26(double *state, double *unused, double *out_8724474158738612795);
void car_H_26(double *state, double *unused, double *out_4086945136231088921);
void car_h_27(double *state, double *unused, double *out_5939300562496261500);
void car_H_27(double *state, double *unused, double *out_4396485212334159147);
void car_h_29(double *state, double *unused, double *out_7757382698754425861);
void car_H_29(double *state, double *unused, double *out_2683122485464608114);
void car_h_28(double *state, double *unused, double *out_6108401993799440970);
void car_H_28(double *state, double *unused, double *out_2399276531604922460);
void car_h_31(double *state, double *unused, double *out_4838604041243469920);
void car_H_31(double *state, double *unused, double *out_314795855480072269);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}