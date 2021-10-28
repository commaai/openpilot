#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void live_update_3(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_19(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_3945300676633952454);
void live_err_fun(double *nom_x, double *delta_x, double *out_2671642069194231640);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_3785602331772523069);
void live_H_mod_fun(double *state, double *out_5304291424146489412);
void live_f_fun(double *state, double dt, double *out_1149222669431498361);
void live_F_fun(double *state, double dt, double *out_1973587698958716554);
void live_h_3(double *state, double *unused, double *out_8904120695401502192);
void live_H_3(double *state, double *unused, double *out_3982015430621093410);
void live_h_4(double *state, double *unused, double *out_3215591188943332251);
void live_H_4(double *state, double *unused, double *out_3272653833133825581);
void live_h_9(double *state, double *unused, double *out_3365106478690958704);
void live_H_9(double *state, double *unused, double *out_5805555408070013240);
void live_h_10(double *state, double *unused, double *out_4033726368773306914);
void live_H_10(double *state, double *unused, double *out_6194162984306139645);
void live_h_12(double *state, double *unused, double *out_3257701289536419656);
void live_H_12(double *state, double *unused, double *out_6021505440201659641);
void live_h_31(double *state, double *unused, double *out_6632942395008265878);
void live_H_31(double *state, double *unused, double *out_2704119569729615329);
void live_h_32(double *state, double *unused, double *out_8850696574973992298);
void live_H_32(double *state, double *unused, double *out_7799647136716720507);
void live_h_13(double *state, double *unused, double *out_1424522614259350959);
void live_H_13(double *state, double *unused, double *out_6547590777731655305);
void live_h_14(double *state, double *unused, double *out_3365106478690958704);
void live_H_14(double *state, double *unused, double *out_5805555408070013240);
void live_h_19(double *state, double *unused, double *out_1514137303607510762);
void live_H_19(double *state, double *unused, double *out_1372875556967582059);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}