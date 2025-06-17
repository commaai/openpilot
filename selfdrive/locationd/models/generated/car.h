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
void car_err_fun(double *nom_x, double *delta_x, double *out_2868104873166063940);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8159495437129412069);
void car_H_mod_fun(double *state, double *out_8071738448770388550);
void car_f_fun(double *state, double dt, double *out_5610024755312278041);
void car_F_fun(double *state, double dt, double *out_8749456969336840526);
void car_h_25(double *state, double *unused, double *out_4495051692134795726);
void car_H_25(double *state, double *unused, double *out_692419411288880047);
void car_h_24(double *state, double *unused, double *out_2855344354432327028);
void car_H_24(double *state, double *unused, double *out_4250345010091203183);
void car_h_30(double *state, double *unused, double *out_4337865023783666581);
void car_H_30(double *state, double *unused, double *out_7609109752780496802);
void car_h_26(double *state, double *unused, double *out_6460693589154982575);
void car_H_26(double *state, double *unused, double *out_3049083907585176177);
void car_h_27(double *state, double *unused, double *out_8950019113913910245);
void car_H_27(double *state, double *unused, double *out_5434346440980071891);
void car_h_29(double *state, double *unused, double *out_1733459702974968930);
void car_H_29(double *state, double *unused, double *out_3720983714110520858);
void car_h_28(double *state, double *unused, double *out_514755787788800606);
void car_H_28(double *state, double *unused, double *out_1361415302959009716);
void car_h_31(double *state, double *unused, double *out_7180857155224404954);
void car_H_31(double *state, double *unused, double *out_723065373165840475);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}