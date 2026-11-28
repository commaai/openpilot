#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_2740203603399450496);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2944854366871732438);
void pose_H_mod_fun(double *state, double *out_1017226251035741065);
void pose_f_fun(double *state, double dt, double *out_3838696256166161148);
void pose_F_fun(double *state, double dt, double *out_6662765533021581793);
void pose_h_4(double *state, double *unused, double *out_2834632828994169778);
void pose_H_4(double *state, double *unused, double *out_4086798773631621081);
void pose_h_10(double *state, double *unused, double *out_1712914576874124643);
void pose_H_10(double *state, double *unused, double *out_1959122121965491232);
void pose_h_13(double *state, double *unused, double *out_3468100881563169718);
void pose_H_13(double *state, double *unused, double *out_6749314091761229606);
void pose_h_14(double *state, double *unused, double *out_3148208993463278653);
void pose_H_14(double *state, double *unused, double *out_1004010341336248785);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}