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
void car_err_fun(double *nom_x, double *delta_x, double *out_8879023178729925980);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_917364609278737637);
void car_H_mod_fun(double *state, double *out_6731364818185773625);
void car_f_fun(double *state, double dt, double *out_4909689618608652368);
void car_F_fun(double *state, double dt, double *out_415125029152466339);
void car_h_25(double *state, double *unused, double *out_4795770764444923556);
void car_H_25(double *state, double *unused, double *out_2522191029611540754);
void car_h_24(double *state, double *unused, double *out_6599817667004600482);
void car_H_24(double *state, double *unused, double *out_6252701378200630140);
void car_h_30(double *state, double *unused, double *out_6887458760493917913);
void car_H_30(double *state, double *unused, double *out_2392852082468300684);
void car_h_26(double *state, double *unused, double *out_5943948907592255478);
void car_H_26(double *state, double *unused, double *out_1219312289262515470);
void car_h_27(double *state, double *unused, double *out_8679685930122430163);
void car_H_27(double *state, double *unused, double *out_218088770667875773);
void car_h_29(double *state, double *unused, double *out_3153628493571407050);
void car_H_29(double *state, double *unused, double *out_2903083426782692868);
void car_h_28(double *state, double *unused, double *out_8009781937011597724);
void car_H_28(double *state, double *unused, double *out_4866713698348019119);
void car_h_31(double *state, double *unused, double *out_5128644055090467685);
void car_H_31(double *state, double *unused, double *out_2552836991488501182);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}