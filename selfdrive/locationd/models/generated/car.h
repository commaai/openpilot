#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_6604114408220947675);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8451809277703177441);
void car_H_mod_fun(double *state, double *out_8861781352220683421);
void car_f_fun(double *state, double dt, double *out_8082967332842324869);
void car_F_fun(double *state, double dt, double *out_4095617844735775990);
void car_h_25(double *state, double *unused, double *out_6175323206303387376);
void car_H_25(double *state, double *unused, double *out_487413547395389457);
void car_h_24(double *state, double *unused, double *out_1658736045300990593);
void car_H_24(double *state, double *unused, double *out_5233922874361898491);
void car_h_30(double *state, double *unused, double *out_5292438858594657572);
void car_H_30(double *state, double *unused, double *out_2030919411111859170);
void car_h_26(double *state, double *unused, double *out_66531642759332625);
void car_H_26(double *state, double *unused, double *out_4228916866269445681);
void car_h_27(double *state, double *unused, double *out_2412839475501454032);
void car_H_27(double *state, double *unused, double *out_7189873189323422566);
void car_h_29(double *state, double *unused, double *out_2688033537785959921);
void car_H_29(double *state, double *unused, double *out_2541150755426251354);
void car_h_28(double *state, double *unused, double *out_3892767580075800992);
void car_H_28(double *state, double *unused, double *out_2541248261643279220);
void car_h_31(double *state, double *unused, double *out_5536559782514876574);
void car_H_31(double *state, double *unused, double *out_4855124968502797157);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}