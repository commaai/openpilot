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
void car_err_fun(double *nom_x, double *delta_x, double *out_5154743392088968873);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6484687437990115350);
void car_H_mod_fun(double *state, double *out_2724143367716012553);
void car_f_fun(double *state, double dt, double *out_4240112907484928210);
void car_F_fun(double *state, double dt, double *out_7340475373084997924);
void car_h_25(double *state, double *unused, double *out_9000275207519044529);
void car_H_25(double *state, double *unused, double *out_7801262673019679258);
void car_h_24(double *state, double *unused, double *out_1494680429397070930);
void car_H_24(double *state, double *unused, double *out_6388234424981270421);
void car_h_30(double *state, double *unused, double *out_360397730800502645);
void car_H_30(double *state, double *unused, double *out_6117785070562264160);
void car_h_26(double *state, double *unused, double *out_5845481762141335100);
void car_H_26(double *state, double *unused, double *out_6903978081815816134);
void car_h_27(double *state, double *unused, double *out_5963489156074711844);
void car_H_27(double *state, double *unused, double *out_3943021758761839249);
void car_h_29(double *state, double *unused, double *out_5505287793916150182);
void car_H_29(double *state, double *unused, double *out_6628016414876656344);
void car_h_28(double *state, double *unused, double *out_7481581649973994127);
void car_H_28(double *state, double *unused, double *out_8591646686441982595);
void car_h_31(double *state, double *unused, double *out_8725081145234538640);
void car_H_31(double *state, double *unused, double *out_6277769979582464658);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}