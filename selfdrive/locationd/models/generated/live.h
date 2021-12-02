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
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_3946282068900904401);
void live_err_fun(double *nom_x, double *delta_x, double *out_2465284701541298471);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_1241552013099516552);
void live_H_mod_fun(double *state, double *out_930557408510925473);
void live_f_fun(double *state, double dt, double *out_676762109596403739);
void live_F_fun(double *state, double dt, double *out_3035806162385026049);
void live_h_3(double *state, double *unused, double *out_1620223020902374011);
void live_H_3(double *state, double *unused, double *out_1247196223838839166);
void live_h_4(double *state, double *unused, double *out_2113193522270853477);
void live_H_4(double *state, double *unused, double *out_8265036806509866230);
void live_h_9(double *state, double *unused, double *out_8990785504222800374);
void live_H_9(double *state, double *unused, double *out_6964257658424747337);
void live_h_10(double *state, double *unused, double *out_3825832259270417306);
void live_H_10(double *state, double *unused, double *out_1996273469984823245);
void live_h_12(double *state, double *unused, double *out_2372630948885105719);
void live_H_12(double *state, double *unused, double *out_1291318795476646408);
void live_h_31(double *state, double *unused, double *out_1306797907447832589);
void live_H_31(double *state, double *unused, double *out_7110718391666311794);
void live_h_32(double *state, double *unused, double *out_4847799802915729958);
void live_H_32(double *state, double *unused, double *out_3991713241733039936);
void live_h_13(double *state, double *unused, double *out_4509807577828258103);
void live_H_13(double *state, double *unused, double *out_5753145722804045947);
void live_h_14(double *state, double *unused, double *out_8990785504222800374);
void live_H_14(double *state, double *unused, double *out_6964257658424747337);
void live_h_19(double *state, double *unused, double *out_5844106528649007674);
void live_H_19(double *state, double *unused, double *out_7353258701952733090);
void live_h_33(double *state, double *unused, double *out_5506844170595132976);
void live_H_33(double *state, double *unused, double *out_7007249167538208636);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}