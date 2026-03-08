#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_8310779210450879699);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_7443261741520201626);
void pose_H_mod_fun(double *state, double *out_3280844175493419238);
void pose_f_fun(double *state, double dt, double *out_2175321056186301973);
void pose_F_fun(double *state, double dt, double *out_6067447866951354337);
void pose_h_4(double *state, double *unused, double *out_5162177481250063533);
void pose_H_4(double *state, double *unused, double *out_4478791657376317342);
void pose_h_10(double *state, double *unused, double *out_2981368779043647817);
void pose_H_10(double *state, double *unused, double *out_919074047538936453);
void pose_h_13(double *state, double *unused, double *out_3635540254248668032);
void pose_H_13(double *state, double *unused, double *out_645036194073793318);
void pose_h_14(double *state, double *unused, double *out_4192481788271112889);
void pose_H_14(double *state, double *unused, double *out_1396003225080945046);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}