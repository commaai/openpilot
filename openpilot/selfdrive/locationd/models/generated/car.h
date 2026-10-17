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
void car_err_fun(double *nom_x, double *delta_x, double *out_6382149531773064090);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4625467165850258549);
void car_H_mod_fun(double *state, double *out_1214021453202461484);
void car_f_fun(double *state, double dt, double *out_5102529219309380439);
void car_F_fun(double *state, double dt, double *out_1049561808137205989);
void car_h_25(double *state, double *unused, double *out_4688551692223537173);
void car_H_25(double *state, double *unused, double *out_1719355895402758196);
void car_h_24(double *state, double *unused, double *out_8426987597188806191);
void car_H_24(double *state, double *unused, double *out_8294927701994276297);
void car_h_30(double *state, double *unused, double *out_5865252886601491084);
void car_H_30(double *state, double *unused, double *out_8636046236894374951);
void car_h_26(double *state, double *unused, double *out_3599522015101219767);
void car_H_26(double *state, double *unused, double *out_5023881865163558797);
void car_h_27(double *state, double *unused, double *out_7725337743667869858);
void car_H_27(double *state, double *unused, double *out_6461282925093950040);
void car_h_29(double *state, double *unused, double *out_4932146013796726190);
void car_H_29(double *state, double *unused, double *out_9146277581208767135);
void car_h_28(double *state, double *unused, double *out_2499430527832544911);
void car_H_28(double *state, double *unused, double *out_4063878564139236561);
void car_h_31(double *state, double *unused, double *out_2082283534126813763);
void car_H_31(double *state, double *unused, double *out_8796031145914575449);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}