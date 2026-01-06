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
void car_err_fun(double *nom_x, double *delta_x, double *out_7584201253892190001);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4881621785150572117);
void car_H_mod_fun(double *state, double *out_6380899889402302601);
void car_f_fun(double *state, double dt, double *out_6392402267122999460);
void car_F_fun(double *state, double dt, double *out_2526673064601828008);
void car_h_25(double *state, double *unused, double *out_4624397152353190203);
void car_H_25(double *state, double *unused, double *out_7162152359122584175);
void car_h_24(double *state, double *unused, double *out_5737550318670326813);
void car_H_24(double *state, double *unused, double *out_4984937935515434202);
void car_h_30(double *state, double *unused, double *out_901660284197832910);
void car_H_30(double *state, double *unused, double *out_4643819400615335548);
void car_h_26(double *state, double *unused, double *out_4060719713483460322);
void car_H_26(double *state, double *unused, double *out_7543088395712911217);
void car_h_27(double *state, double *unused, double *out_3557025719116557148);
void car_H_27(double *state, double *unused, double *out_2420225329431392331);
void car_h_29(double *state, double *unused, double *out_4958990177267014116);
void car_H_29(double *state, double *unused, double *out_4133588056300943364);
void car_h_28(double *state, double *unused, double *out_2953709834440079631);
void car_H_28(double *state, double *unused, double *out_9215987073370473938);
void car_h_31(double *state, double *unused, double *out_6026361610503647171);
void car_H_31(double *state, double *unused, double *out_6916880293479559741);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}