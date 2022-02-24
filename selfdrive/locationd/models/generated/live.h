#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_8573827050636567067);
void live_err_fun(double *nom_x, double *delta_x, double *out_2292859409783966048);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_7466509707901054724);
void live_H_mod_fun(double *state, double *out_4810487012885038206);
void live_f_fun(double *state, double dt, double *out_7925576684396976167);
void live_F_fun(double *state, double dt, double *out_6553453215573577202);
void live_h_4(double *state, double *unused, double *out_5706261362768053969);
void live_H_4(double *state, double *unused, double *out_7380667893018306527);
void live_h_9(double *state, double *unused, double *out_2092307496186822377);
void live_H_9(double *state, double *unused, double *out_7139478246388715882);
void live_h_10(double *state, double *unused, double *out_105727890266340874);
void live_H_10(double *state, double *unused, double *out_2504349095668085521);
void live_h_12(double *state, double *unused, double *out_3207073958905486317);
void live_H_12(double *state, double *unused, double *out_2361211484986344732);
void live_h_31(double *state, double *unused, double *out_7167693638028319298);
void live_H_31(double *state, double *unused, double *out_4014005835645699151);
void live_h_32(double *state, double *unused, double *out_5811748684591151115);
void live_H_32(double *state, double *unused, double *out_3882841050704958682);
void live_h_13(double *state, double *unused, double *out_1221493773571756941);
void live_H_13(double *state, double *unused, double *out_6120959287865860338);
void live_h_14(double *state, double *unused, double *out_2092307496186822377);
void live_H_14(double *state, double *unused, double *out_7139478246388715882);
void live_h_33(double *state, double *unused, double *out_8598522805817106162);
void live_H_33(double *state, double *unused, double *out_863448831006841547);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}