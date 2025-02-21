#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7351456117776073154) {
   out_7351456117776073154[0] = delta_x[0] + nom_x[0];
   out_7351456117776073154[1] = delta_x[1] + nom_x[1];
   out_7351456117776073154[2] = delta_x[2] + nom_x[2];
   out_7351456117776073154[3] = delta_x[3] + nom_x[3];
   out_7351456117776073154[4] = delta_x[4] + nom_x[4];
   out_7351456117776073154[5] = delta_x[5] + nom_x[5];
   out_7351456117776073154[6] = delta_x[6] + nom_x[6];
   out_7351456117776073154[7] = delta_x[7] + nom_x[7];
   out_7351456117776073154[8] = delta_x[8] + nom_x[8];
   out_7351456117776073154[9] = delta_x[9] + nom_x[9];
   out_7351456117776073154[10] = delta_x[10] + nom_x[10];
   out_7351456117776073154[11] = delta_x[11] + nom_x[11];
   out_7351456117776073154[12] = delta_x[12] + nom_x[12];
   out_7351456117776073154[13] = delta_x[13] + nom_x[13];
   out_7351456117776073154[14] = delta_x[14] + nom_x[14];
   out_7351456117776073154[15] = delta_x[15] + nom_x[15];
   out_7351456117776073154[16] = delta_x[16] + nom_x[16];
   out_7351456117776073154[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_9109558368636035839) {
   out_9109558368636035839[0] = -nom_x[0] + true_x[0];
   out_9109558368636035839[1] = -nom_x[1] + true_x[1];
   out_9109558368636035839[2] = -nom_x[2] + true_x[2];
   out_9109558368636035839[3] = -nom_x[3] + true_x[3];
   out_9109558368636035839[4] = -nom_x[4] + true_x[4];
   out_9109558368636035839[5] = -nom_x[5] + true_x[5];
   out_9109558368636035839[6] = -nom_x[6] + true_x[6];
   out_9109558368636035839[7] = -nom_x[7] + true_x[7];
   out_9109558368636035839[8] = -nom_x[8] + true_x[8];
   out_9109558368636035839[9] = -nom_x[9] + true_x[9];
   out_9109558368636035839[10] = -nom_x[10] + true_x[10];
   out_9109558368636035839[11] = -nom_x[11] + true_x[11];
   out_9109558368636035839[12] = -nom_x[12] + true_x[12];
   out_9109558368636035839[13] = -nom_x[13] + true_x[13];
   out_9109558368636035839[14] = -nom_x[14] + true_x[14];
   out_9109558368636035839[15] = -nom_x[15] + true_x[15];
   out_9109558368636035839[16] = -nom_x[16] + true_x[16];
   out_9109558368636035839[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_595012964091947662) {
   out_595012964091947662[0] = 1.0;
   out_595012964091947662[1] = 0;
   out_595012964091947662[2] = 0;
   out_595012964091947662[3] = 0;
   out_595012964091947662[4] = 0;
   out_595012964091947662[5] = 0;
   out_595012964091947662[6] = 0;
   out_595012964091947662[7] = 0;
   out_595012964091947662[8] = 0;
   out_595012964091947662[9] = 0;
   out_595012964091947662[10] = 0;
   out_595012964091947662[11] = 0;
   out_595012964091947662[12] = 0;
   out_595012964091947662[13] = 0;
   out_595012964091947662[14] = 0;
   out_595012964091947662[15] = 0;
   out_595012964091947662[16] = 0;
   out_595012964091947662[17] = 0;
   out_595012964091947662[18] = 0;
   out_595012964091947662[19] = 1.0;
   out_595012964091947662[20] = 0;
   out_595012964091947662[21] = 0;
   out_595012964091947662[22] = 0;
   out_595012964091947662[23] = 0;
   out_595012964091947662[24] = 0;
   out_595012964091947662[25] = 0;
   out_595012964091947662[26] = 0;
   out_595012964091947662[27] = 0;
   out_595012964091947662[28] = 0;
   out_595012964091947662[29] = 0;
   out_595012964091947662[30] = 0;
   out_595012964091947662[31] = 0;
   out_595012964091947662[32] = 0;
   out_595012964091947662[33] = 0;
   out_595012964091947662[34] = 0;
   out_595012964091947662[35] = 0;
   out_595012964091947662[36] = 0;
   out_595012964091947662[37] = 0;
   out_595012964091947662[38] = 1.0;
   out_595012964091947662[39] = 0;
   out_595012964091947662[40] = 0;
   out_595012964091947662[41] = 0;
   out_595012964091947662[42] = 0;
   out_595012964091947662[43] = 0;
   out_595012964091947662[44] = 0;
   out_595012964091947662[45] = 0;
   out_595012964091947662[46] = 0;
   out_595012964091947662[47] = 0;
   out_595012964091947662[48] = 0;
   out_595012964091947662[49] = 0;
   out_595012964091947662[50] = 0;
   out_595012964091947662[51] = 0;
   out_595012964091947662[52] = 0;
   out_595012964091947662[53] = 0;
   out_595012964091947662[54] = 0;
   out_595012964091947662[55] = 0;
   out_595012964091947662[56] = 0;
   out_595012964091947662[57] = 1.0;
   out_595012964091947662[58] = 0;
   out_595012964091947662[59] = 0;
   out_595012964091947662[60] = 0;
   out_595012964091947662[61] = 0;
   out_595012964091947662[62] = 0;
   out_595012964091947662[63] = 0;
   out_595012964091947662[64] = 0;
   out_595012964091947662[65] = 0;
   out_595012964091947662[66] = 0;
   out_595012964091947662[67] = 0;
   out_595012964091947662[68] = 0;
   out_595012964091947662[69] = 0;
   out_595012964091947662[70] = 0;
   out_595012964091947662[71] = 0;
   out_595012964091947662[72] = 0;
   out_595012964091947662[73] = 0;
   out_595012964091947662[74] = 0;
   out_595012964091947662[75] = 0;
   out_595012964091947662[76] = 1.0;
   out_595012964091947662[77] = 0;
   out_595012964091947662[78] = 0;
   out_595012964091947662[79] = 0;
   out_595012964091947662[80] = 0;
   out_595012964091947662[81] = 0;
   out_595012964091947662[82] = 0;
   out_595012964091947662[83] = 0;
   out_595012964091947662[84] = 0;
   out_595012964091947662[85] = 0;
   out_595012964091947662[86] = 0;
   out_595012964091947662[87] = 0;
   out_595012964091947662[88] = 0;
   out_595012964091947662[89] = 0;
   out_595012964091947662[90] = 0;
   out_595012964091947662[91] = 0;
   out_595012964091947662[92] = 0;
   out_595012964091947662[93] = 0;
   out_595012964091947662[94] = 0;
   out_595012964091947662[95] = 1.0;
   out_595012964091947662[96] = 0;
   out_595012964091947662[97] = 0;
   out_595012964091947662[98] = 0;
   out_595012964091947662[99] = 0;
   out_595012964091947662[100] = 0;
   out_595012964091947662[101] = 0;
   out_595012964091947662[102] = 0;
   out_595012964091947662[103] = 0;
   out_595012964091947662[104] = 0;
   out_595012964091947662[105] = 0;
   out_595012964091947662[106] = 0;
   out_595012964091947662[107] = 0;
   out_595012964091947662[108] = 0;
   out_595012964091947662[109] = 0;
   out_595012964091947662[110] = 0;
   out_595012964091947662[111] = 0;
   out_595012964091947662[112] = 0;
   out_595012964091947662[113] = 0;
   out_595012964091947662[114] = 1.0;
   out_595012964091947662[115] = 0;
   out_595012964091947662[116] = 0;
   out_595012964091947662[117] = 0;
   out_595012964091947662[118] = 0;
   out_595012964091947662[119] = 0;
   out_595012964091947662[120] = 0;
   out_595012964091947662[121] = 0;
   out_595012964091947662[122] = 0;
   out_595012964091947662[123] = 0;
   out_595012964091947662[124] = 0;
   out_595012964091947662[125] = 0;
   out_595012964091947662[126] = 0;
   out_595012964091947662[127] = 0;
   out_595012964091947662[128] = 0;
   out_595012964091947662[129] = 0;
   out_595012964091947662[130] = 0;
   out_595012964091947662[131] = 0;
   out_595012964091947662[132] = 0;
   out_595012964091947662[133] = 1.0;
   out_595012964091947662[134] = 0;
   out_595012964091947662[135] = 0;
   out_595012964091947662[136] = 0;
   out_595012964091947662[137] = 0;
   out_595012964091947662[138] = 0;
   out_595012964091947662[139] = 0;
   out_595012964091947662[140] = 0;
   out_595012964091947662[141] = 0;
   out_595012964091947662[142] = 0;
   out_595012964091947662[143] = 0;
   out_595012964091947662[144] = 0;
   out_595012964091947662[145] = 0;
   out_595012964091947662[146] = 0;
   out_595012964091947662[147] = 0;
   out_595012964091947662[148] = 0;
   out_595012964091947662[149] = 0;
   out_595012964091947662[150] = 0;
   out_595012964091947662[151] = 0;
   out_595012964091947662[152] = 1.0;
   out_595012964091947662[153] = 0;
   out_595012964091947662[154] = 0;
   out_595012964091947662[155] = 0;
   out_595012964091947662[156] = 0;
   out_595012964091947662[157] = 0;
   out_595012964091947662[158] = 0;
   out_595012964091947662[159] = 0;
   out_595012964091947662[160] = 0;
   out_595012964091947662[161] = 0;
   out_595012964091947662[162] = 0;
   out_595012964091947662[163] = 0;
   out_595012964091947662[164] = 0;
   out_595012964091947662[165] = 0;
   out_595012964091947662[166] = 0;
   out_595012964091947662[167] = 0;
   out_595012964091947662[168] = 0;
   out_595012964091947662[169] = 0;
   out_595012964091947662[170] = 0;
   out_595012964091947662[171] = 1.0;
   out_595012964091947662[172] = 0;
   out_595012964091947662[173] = 0;
   out_595012964091947662[174] = 0;
   out_595012964091947662[175] = 0;
   out_595012964091947662[176] = 0;
   out_595012964091947662[177] = 0;
   out_595012964091947662[178] = 0;
   out_595012964091947662[179] = 0;
   out_595012964091947662[180] = 0;
   out_595012964091947662[181] = 0;
   out_595012964091947662[182] = 0;
   out_595012964091947662[183] = 0;
   out_595012964091947662[184] = 0;
   out_595012964091947662[185] = 0;
   out_595012964091947662[186] = 0;
   out_595012964091947662[187] = 0;
   out_595012964091947662[188] = 0;
   out_595012964091947662[189] = 0;
   out_595012964091947662[190] = 1.0;
   out_595012964091947662[191] = 0;
   out_595012964091947662[192] = 0;
   out_595012964091947662[193] = 0;
   out_595012964091947662[194] = 0;
   out_595012964091947662[195] = 0;
   out_595012964091947662[196] = 0;
   out_595012964091947662[197] = 0;
   out_595012964091947662[198] = 0;
   out_595012964091947662[199] = 0;
   out_595012964091947662[200] = 0;
   out_595012964091947662[201] = 0;
   out_595012964091947662[202] = 0;
   out_595012964091947662[203] = 0;
   out_595012964091947662[204] = 0;
   out_595012964091947662[205] = 0;
   out_595012964091947662[206] = 0;
   out_595012964091947662[207] = 0;
   out_595012964091947662[208] = 0;
   out_595012964091947662[209] = 1.0;
   out_595012964091947662[210] = 0;
   out_595012964091947662[211] = 0;
   out_595012964091947662[212] = 0;
   out_595012964091947662[213] = 0;
   out_595012964091947662[214] = 0;
   out_595012964091947662[215] = 0;
   out_595012964091947662[216] = 0;
   out_595012964091947662[217] = 0;
   out_595012964091947662[218] = 0;
   out_595012964091947662[219] = 0;
   out_595012964091947662[220] = 0;
   out_595012964091947662[221] = 0;
   out_595012964091947662[222] = 0;
   out_595012964091947662[223] = 0;
   out_595012964091947662[224] = 0;
   out_595012964091947662[225] = 0;
   out_595012964091947662[226] = 0;
   out_595012964091947662[227] = 0;
   out_595012964091947662[228] = 1.0;
   out_595012964091947662[229] = 0;
   out_595012964091947662[230] = 0;
   out_595012964091947662[231] = 0;
   out_595012964091947662[232] = 0;
   out_595012964091947662[233] = 0;
   out_595012964091947662[234] = 0;
   out_595012964091947662[235] = 0;
   out_595012964091947662[236] = 0;
   out_595012964091947662[237] = 0;
   out_595012964091947662[238] = 0;
   out_595012964091947662[239] = 0;
   out_595012964091947662[240] = 0;
   out_595012964091947662[241] = 0;
   out_595012964091947662[242] = 0;
   out_595012964091947662[243] = 0;
   out_595012964091947662[244] = 0;
   out_595012964091947662[245] = 0;
   out_595012964091947662[246] = 0;
   out_595012964091947662[247] = 1.0;
   out_595012964091947662[248] = 0;
   out_595012964091947662[249] = 0;
   out_595012964091947662[250] = 0;
   out_595012964091947662[251] = 0;
   out_595012964091947662[252] = 0;
   out_595012964091947662[253] = 0;
   out_595012964091947662[254] = 0;
   out_595012964091947662[255] = 0;
   out_595012964091947662[256] = 0;
   out_595012964091947662[257] = 0;
   out_595012964091947662[258] = 0;
   out_595012964091947662[259] = 0;
   out_595012964091947662[260] = 0;
   out_595012964091947662[261] = 0;
   out_595012964091947662[262] = 0;
   out_595012964091947662[263] = 0;
   out_595012964091947662[264] = 0;
   out_595012964091947662[265] = 0;
   out_595012964091947662[266] = 1.0;
   out_595012964091947662[267] = 0;
   out_595012964091947662[268] = 0;
   out_595012964091947662[269] = 0;
   out_595012964091947662[270] = 0;
   out_595012964091947662[271] = 0;
   out_595012964091947662[272] = 0;
   out_595012964091947662[273] = 0;
   out_595012964091947662[274] = 0;
   out_595012964091947662[275] = 0;
   out_595012964091947662[276] = 0;
   out_595012964091947662[277] = 0;
   out_595012964091947662[278] = 0;
   out_595012964091947662[279] = 0;
   out_595012964091947662[280] = 0;
   out_595012964091947662[281] = 0;
   out_595012964091947662[282] = 0;
   out_595012964091947662[283] = 0;
   out_595012964091947662[284] = 0;
   out_595012964091947662[285] = 1.0;
   out_595012964091947662[286] = 0;
   out_595012964091947662[287] = 0;
   out_595012964091947662[288] = 0;
   out_595012964091947662[289] = 0;
   out_595012964091947662[290] = 0;
   out_595012964091947662[291] = 0;
   out_595012964091947662[292] = 0;
   out_595012964091947662[293] = 0;
   out_595012964091947662[294] = 0;
   out_595012964091947662[295] = 0;
   out_595012964091947662[296] = 0;
   out_595012964091947662[297] = 0;
   out_595012964091947662[298] = 0;
   out_595012964091947662[299] = 0;
   out_595012964091947662[300] = 0;
   out_595012964091947662[301] = 0;
   out_595012964091947662[302] = 0;
   out_595012964091947662[303] = 0;
   out_595012964091947662[304] = 1.0;
   out_595012964091947662[305] = 0;
   out_595012964091947662[306] = 0;
   out_595012964091947662[307] = 0;
   out_595012964091947662[308] = 0;
   out_595012964091947662[309] = 0;
   out_595012964091947662[310] = 0;
   out_595012964091947662[311] = 0;
   out_595012964091947662[312] = 0;
   out_595012964091947662[313] = 0;
   out_595012964091947662[314] = 0;
   out_595012964091947662[315] = 0;
   out_595012964091947662[316] = 0;
   out_595012964091947662[317] = 0;
   out_595012964091947662[318] = 0;
   out_595012964091947662[319] = 0;
   out_595012964091947662[320] = 0;
   out_595012964091947662[321] = 0;
   out_595012964091947662[322] = 0;
   out_595012964091947662[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_3792261713431509332) {
   out_3792261713431509332[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_3792261713431509332[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_3792261713431509332[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_3792261713431509332[3] = dt*state[12] + state[3];
   out_3792261713431509332[4] = dt*state[13] + state[4];
   out_3792261713431509332[5] = dt*state[14] + state[5];
   out_3792261713431509332[6] = state[6];
   out_3792261713431509332[7] = state[7];
   out_3792261713431509332[8] = state[8];
   out_3792261713431509332[9] = state[9];
   out_3792261713431509332[10] = state[10];
   out_3792261713431509332[11] = state[11];
   out_3792261713431509332[12] = state[12];
   out_3792261713431509332[13] = state[13];
   out_3792261713431509332[14] = state[14];
   out_3792261713431509332[15] = state[15];
   out_3792261713431509332[16] = state[16];
   out_3792261713431509332[17] = state[17];
}
void F_fun(double *state, double dt, double *out_6928823179361542013) {
   out_6928823179361542013[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6928823179361542013[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6928823179361542013[2] = 0;
   out_6928823179361542013[3] = 0;
   out_6928823179361542013[4] = 0;
   out_6928823179361542013[5] = 0;
   out_6928823179361542013[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6928823179361542013[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6928823179361542013[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6928823179361542013[9] = 0;
   out_6928823179361542013[10] = 0;
   out_6928823179361542013[11] = 0;
   out_6928823179361542013[12] = 0;
   out_6928823179361542013[13] = 0;
   out_6928823179361542013[14] = 0;
   out_6928823179361542013[15] = 0;
   out_6928823179361542013[16] = 0;
   out_6928823179361542013[17] = 0;
   out_6928823179361542013[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6928823179361542013[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6928823179361542013[20] = 0;
   out_6928823179361542013[21] = 0;
   out_6928823179361542013[22] = 0;
   out_6928823179361542013[23] = 0;
   out_6928823179361542013[24] = 0;
   out_6928823179361542013[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6928823179361542013[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6928823179361542013[27] = 0;
   out_6928823179361542013[28] = 0;
   out_6928823179361542013[29] = 0;
   out_6928823179361542013[30] = 0;
   out_6928823179361542013[31] = 0;
   out_6928823179361542013[32] = 0;
   out_6928823179361542013[33] = 0;
   out_6928823179361542013[34] = 0;
   out_6928823179361542013[35] = 0;
   out_6928823179361542013[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6928823179361542013[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6928823179361542013[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6928823179361542013[39] = 0;
   out_6928823179361542013[40] = 0;
   out_6928823179361542013[41] = 0;
   out_6928823179361542013[42] = 0;
   out_6928823179361542013[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6928823179361542013[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6928823179361542013[45] = 0;
   out_6928823179361542013[46] = 0;
   out_6928823179361542013[47] = 0;
   out_6928823179361542013[48] = 0;
   out_6928823179361542013[49] = 0;
   out_6928823179361542013[50] = 0;
   out_6928823179361542013[51] = 0;
   out_6928823179361542013[52] = 0;
   out_6928823179361542013[53] = 0;
   out_6928823179361542013[54] = 0;
   out_6928823179361542013[55] = 0;
   out_6928823179361542013[56] = 0;
   out_6928823179361542013[57] = 1;
   out_6928823179361542013[58] = 0;
   out_6928823179361542013[59] = 0;
   out_6928823179361542013[60] = 0;
   out_6928823179361542013[61] = 0;
   out_6928823179361542013[62] = 0;
   out_6928823179361542013[63] = 0;
   out_6928823179361542013[64] = 0;
   out_6928823179361542013[65] = 0;
   out_6928823179361542013[66] = dt;
   out_6928823179361542013[67] = 0;
   out_6928823179361542013[68] = 0;
   out_6928823179361542013[69] = 0;
   out_6928823179361542013[70] = 0;
   out_6928823179361542013[71] = 0;
   out_6928823179361542013[72] = 0;
   out_6928823179361542013[73] = 0;
   out_6928823179361542013[74] = 0;
   out_6928823179361542013[75] = 0;
   out_6928823179361542013[76] = 1;
   out_6928823179361542013[77] = 0;
   out_6928823179361542013[78] = 0;
   out_6928823179361542013[79] = 0;
   out_6928823179361542013[80] = 0;
   out_6928823179361542013[81] = 0;
   out_6928823179361542013[82] = 0;
   out_6928823179361542013[83] = 0;
   out_6928823179361542013[84] = 0;
   out_6928823179361542013[85] = dt;
   out_6928823179361542013[86] = 0;
   out_6928823179361542013[87] = 0;
   out_6928823179361542013[88] = 0;
   out_6928823179361542013[89] = 0;
   out_6928823179361542013[90] = 0;
   out_6928823179361542013[91] = 0;
   out_6928823179361542013[92] = 0;
   out_6928823179361542013[93] = 0;
   out_6928823179361542013[94] = 0;
   out_6928823179361542013[95] = 1;
   out_6928823179361542013[96] = 0;
   out_6928823179361542013[97] = 0;
   out_6928823179361542013[98] = 0;
   out_6928823179361542013[99] = 0;
   out_6928823179361542013[100] = 0;
   out_6928823179361542013[101] = 0;
   out_6928823179361542013[102] = 0;
   out_6928823179361542013[103] = 0;
   out_6928823179361542013[104] = dt;
   out_6928823179361542013[105] = 0;
   out_6928823179361542013[106] = 0;
   out_6928823179361542013[107] = 0;
   out_6928823179361542013[108] = 0;
   out_6928823179361542013[109] = 0;
   out_6928823179361542013[110] = 0;
   out_6928823179361542013[111] = 0;
   out_6928823179361542013[112] = 0;
   out_6928823179361542013[113] = 0;
   out_6928823179361542013[114] = 1;
   out_6928823179361542013[115] = 0;
   out_6928823179361542013[116] = 0;
   out_6928823179361542013[117] = 0;
   out_6928823179361542013[118] = 0;
   out_6928823179361542013[119] = 0;
   out_6928823179361542013[120] = 0;
   out_6928823179361542013[121] = 0;
   out_6928823179361542013[122] = 0;
   out_6928823179361542013[123] = 0;
   out_6928823179361542013[124] = 0;
   out_6928823179361542013[125] = 0;
   out_6928823179361542013[126] = 0;
   out_6928823179361542013[127] = 0;
   out_6928823179361542013[128] = 0;
   out_6928823179361542013[129] = 0;
   out_6928823179361542013[130] = 0;
   out_6928823179361542013[131] = 0;
   out_6928823179361542013[132] = 0;
   out_6928823179361542013[133] = 1;
   out_6928823179361542013[134] = 0;
   out_6928823179361542013[135] = 0;
   out_6928823179361542013[136] = 0;
   out_6928823179361542013[137] = 0;
   out_6928823179361542013[138] = 0;
   out_6928823179361542013[139] = 0;
   out_6928823179361542013[140] = 0;
   out_6928823179361542013[141] = 0;
   out_6928823179361542013[142] = 0;
   out_6928823179361542013[143] = 0;
   out_6928823179361542013[144] = 0;
   out_6928823179361542013[145] = 0;
   out_6928823179361542013[146] = 0;
   out_6928823179361542013[147] = 0;
   out_6928823179361542013[148] = 0;
   out_6928823179361542013[149] = 0;
   out_6928823179361542013[150] = 0;
   out_6928823179361542013[151] = 0;
   out_6928823179361542013[152] = 1;
   out_6928823179361542013[153] = 0;
   out_6928823179361542013[154] = 0;
   out_6928823179361542013[155] = 0;
   out_6928823179361542013[156] = 0;
   out_6928823179361542013[157] = 0;
   out_6928823179361542013[158] = 0;
   out_6928823179361542013[159] = 0;
   out_6928823179361542013[160] = 0;
   out_6928823179361542013[161] = 0;
   out_6928823179361542013[162] = 0;
   out_6928823179361542013[163] = 0;
   out_6928823179361542013[164] = 0;
   out_6928823179361542013[165] = 0;
   out_6928823179361542013[166] = 0;
   out_6928823179361542013[167] = 0;
   out_6928823179361542013[168] = 0;
   out_6928823179361542013[169] = 0;
   out_6928823179361542013[170] = 0;
   out_6928823179361542013[171] = 1;
   out_6928823179361542013[172] = 0;
   out_6928823179361542013[173] = 0;
   out_6928823179361542013[174] = 0;
   out_6928823179361542013[175] = 0;
   out_6928823179361542013[176] = 0;
   out_6928823179361542013[177] = 0;
   out_6928823179361542013[178] = 0;
   out_6928823179361542013[179] = 0;
   out_6928823179361542013[180] = 0;
   out_6928823179361542013[181] = 0;
   out_6928823179361542013[182] = 0;
   out_6928823179361542013[183] = 0;
   out_6928823179361542013[184] = 0;
   out_6928823179361542013[185] = 0;
   out_6928823179361542013[186] = 0;
   out_6928823179361542013[187] = 0;
   out_6928823179361542013[188] = 0;
   out_6928823179361542013[189] = 0;
   out_6928823179361542013[190] = 1;
   out_6928823179361542013[191] = 0;
   out_6928823179361542013[192] = 0;
   out_6928823179361542013[193] = 0;
   out_6928823179361542013[194] = 0;
   out_6928823179361542013[195] = 0;
   out_6928823179361542013[196] = 0;
   out_6928823179361542013[197] = 0;
   out_6928823179361542013[198] = 0;
   out_6928823179361542013[199] = 0;
   out_6928823179361542013[200] = 0;
   out_6928823179361542013[201] = 0;
   out_6928823179361542013[202] = 0;
   out_6928823179361542013[203] = 0;
   out_6928823179361542013[204] = 0;
   out_6928823179361542013[205] = 0;
   out_6928823179361542013[206] = 0;
   out_6928823179361542013[207] = 0;
   out_6928823179361542013[208] = 0;
   out_6928823179361542013[209] = 1;
   out_6928823179361542013[210] = 0;
   out_6928823179361542013[211] = 0;
   out_6928823179361542013[212] = 0;
   out_6928823179361542013[213] = 0;
   out_6928823179361542013[214] = 0;
   out_6928823179361542013[215] = 0;
   out_6928823179361542013[216] = 0;
   out_6928823179361542013[217] = 0;
   out_6928823179361542013[218] = 0;
   out_6928823179361542013[219] = 0;
   out_6928823179361542013[220] = 0;
   out_6928823179361542013[221] = 0;
   out_6928823179361542013[222] = 0;
   out_6928823179361542013[223] = 0;
   out_6928823179361542013[224] = 0;
   out_6928823179361542013[225] = 0;
   out_6928823179361542013[226] = 0;
   out_6928823179361542013[227] = 0;
   out_6928823179361542013[228] = 1;
   out_6928823179361542013[229] = 0;
   out_6928823179361542013[230] = 0;
   out_6928823179361542013[231] = 0;
   out_6928823179361542013[232] = 0;
   out_6928823179361542013[233] = 0;
   out_6928823179361542013[234] = 0;
   out_6928823179361542013[235] = 0;
   out_6928823179361542013[236] = 0;
   out_6928823179361542013[237] = 0;
   out_6928823179361542013[238] = 0;
   out_6928823179361542013[239] = 0;
   out_6928823179361542013[240] = 0;
   out_6928823179361542013[241] = 0;
   out_6928823179361542013[242] = 0;
   out_6928823179361542013[243] = 0;
   out_6928823179361542013[244] = 0;
   out_6928823179361542013[245] = 0;
   out_6928823179361542013[246] = 0;
   out_6928823179361542013[247] = 1;
   out_6928823179361542013[248] = 0;
   out_6928823179361542013[249] = 0;
   out_6928823179361542013[250] = 0;
   out_6928823179361542013[251] = 0;
   out_6928823179361542013[252] = 0;
   out_6928823179361542013[253] = 0;
   out_6928823179361542013[254] = 0;
   out_6928823179361542013[255] = 0;
   out_6928823179361542013[256] = 0;
   out_6928823179361542013[257] = 0;
   out_6928823179361542013[258] = 0;
   out_6928823179361542013[259] = 0;
   out_6928823179361542013[260] = 0;
   out_6928823179361542013[261] = 0;
   out_6928823179361542013[262] = 0;
   out_6928823179361542013[263] = 0;
   out_6928823179361542013[264] = 0;
   out_6928823179361542013[265] = 0;
   out_6928823179361542013[266] = 1;
   out_6928823179361542013[267] = 0;
   out_6928823179361542013[268] = 0;
   out_6928823179361542013[269] = 0;
   out_6928823179361542013[270] = 0;
   out_6928823179361542013[271] = 0;
   out_6928823179361542013[272] = 0;
   out_6928823179361542013[273] = 0;
   out_6928823179361542013[274] = 0;
   out_6928823179361542013[275] = 0;
   out_6928823179361542013[276] = 0;
   out_6928823179361542013[277] = 0;
   out_6928823179361542013[278] = 0;
   out_6928823179361542013[279] = 0;
   out_6928823179361542013[280] = 0;
   out_6928823179361542013[281] = 0;
   out_6928823179361542013[282] = 0;
   out_6928823179361542013[283] = 0;
   out_6928823179361542013[284] = 0;
   out_6928823179361542013[285] = 1;
   out_6928823179361542013[286] = 0;
   out_6928823179361542013[287] = 0;
   out_6928823179361542013[288] = 0;
   out_6928823179361542013[289] = 0;
   out_6928823179361542013[290] = 0;
   out_6928823179361542013[291] = 0;
   out_6928823179361542013[292] = 0;
   out_6928823179361542013[293] = 0;
   out_6928823179361542013[294] = 0;
   out_6928823179361542013[295] = 0;
   out_6928823179361542013[296] = 0;
   out_6928823179361542013[297] = 0;
   out_6928823179361542013[298] = 0;
   out_6928823179361542013[299] = 0;
   out_6928823179361542013[300] = 0;
   out_6928823179361542013[301] = 0;
   out_6928823179361542013[302] = 0;
   out_6928823179361542013[303] = 0;
   out_6928823179361542013[304] = 1;
   out_6928823179361542013[305] = 0;
   out_6928823179361542013[306] = 0;
   out_6928823179361542013[307] = 0;
   out_6928823179361542013[308] = 0;
   out_6928823179361542013[309] = 0;
   out_6928823179361542013[310] = 0;
   out_6928823179361542013[311] = 0;
   out_6928823179361542013[312] = 0;
   out_6928823179361542013[313] = 0;
   out_6928823179361542013[314] = 0;
   out_6928823179361542013[315] = 0;
   out_6928823179361542013[316] = 0;
   out_6928823179361542013[317] = 0;
   out_6928823179361542013[318] = 0;
   out_6928823179361542013[319] = 0;
   out_6928823179361542013[320] = 0;
   out_6928823179361542013[321] = 0;
   out_6928823179361542013[322] = 0;
   out_6928823179361542013[323] = 1;
}
void h_4(double *state, double *unused, double *out_7683085519451864677) {
   out_7683085519451864677[0] = state[6] + state[9];
   out_7683085519451864677[1] = state[7] + state[10];
   out_7683085519451864677[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_3149865311834297615) {
   out_3149865311834297615[0] = 0;
   out_3149865311834297615[1] = 0;
   out_3149865311834297615[2] = 0;
   out_3149865311834297615[3] = 0;
   out_3149865311834297615[4] = 0;
   out_3149865311834297615[5] = 0;
   out_3149865311834297615[6] = 1;
   out_3149865311834297615[7] = 0;
   out_3149865311834297615[8] = 0;
   out_3149865311834297615[9] = 1;
   out_3149865311834297615[10] = 0;
   out_3149865311834297615[11] = 0;
   out_3149865311834297615[12] = 0;
   out_3149865311834297615[13] = 0;
   out_3149865311834297615[14] = 0;
   out_3149865311834297615[15] = 0;
   out_3149865311834297615[16] = 0;
   out_3149865311834297615[17] = 0;
   out_3149865311834297615[18] = 0;
   out_3149865311834297615[19] = 0;
   out_3149865311834297615[20] = 0;
   out_3149865311834297615[21] = 0;
   out_3149865311834297615[22] = 0;
   out_3149865311834297615[23] = 0;
   out_3149865311834297615[24] = 0;
   out_3149865311834297615[25] = 1;
   out_3149865311834297615[26] = 0;
   out_3149865311834297615[27] = 0;
   out_3149865311834297615[28] = 1;
   out_3149865311834297615[29] = 0;
   out_3149865311834297615[30] = 0;
   out_3149865311834297615[31] = 0;
   out_3149865311834297615[32] = 0;
   out_3149865311834297615[33] = 0;
   out_3149865311834297615[34] = 0;
   out_3149865311834297615[35] = 0;
   out_3149865311834297615[36] = 0;
   out_3149865311834297615[37] = 0;
   out_3149865311834297615[38] = 0;
   out_3149865311834297615[39] = 0;
   out_3149865311834297615[40] = 0;
   out_3149865311834297615[41] = 0;
   out_3149865311834297615[42] = 0;
   out_3149865311834297615[43] = 0;
   out_3149865311834297615[44] = 1;
   out_3149865311834297615[45] = 0;
   out_3149865311834297615[46] = 0;
   out_3149865311834297615[47] = 1;
   out_3149865311834297615[48] = 0;
   out_3149865311834297615[49] = 0;
   out_3149865311834297615[50] = 0;
   out_3149865311834297615[51] = 0;
   out_3149865311834297615[52] = 0;
   out_3149865311834297615[53] = 0;
}
void h_10(double *state, double *unused, double *out_4760549729285226708) {
   out_4760549729285226708[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_4760549729285226708[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_4760549729285226708[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_748214929053840095) {
   out_748214929053840095[0] = 0;
   out_748214929053840095[1] = 9.8100000000000005*cos(state[1]);
   out_748214929053840095[2] = 0;
   out_748214929053840095[3] = 0;
   out_748214929053840095[4] = -state[8];
   out_748214929053840095[5] = state[7];
   out_748214929053840095[6] = 0;
   out_748214929053840095[7] = state[5];
   out_748214929053840095[8] = -state[4];
   out_748214929053840095[9] = 0;
   out_748214929053840095[10] = 0;
   out_748214929053840095[11] = 0;
   out_748214929053840095[12] = 1;
   out_748214929053840095[13] = 0;
   out_748214929053840095[14] = 0;
   out_748214929053840095[15] = 1;
   out_748214929053840095[16] = 0;
   out_748214929053840095[17] = 0;
   out_748214929053840095[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_748214929053840095[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_748214929053840095[20] = 0;
   out_748214929053840095[21] = state[8];
   out_748214929053840095[22] = 0;
   out_748214929053840095[23] = -state[6];
   out_748214929053840095[24] = -state[5];
   out_748214929053840095[25] = 0;
   out_748214929053840095[26] = state[3];
   out_748214929053840095[27] = 0;
   out_748214929053840095[28] = 0;
   out_748214929053840095[29] = 0;
   out_748214929053840095[30] = 0;
   out_748214929053840095[31] = 1;
   out_748214929053840095[32] = 0;
   out_748214929053840095[33] = 0;
   out_748214929053840095[34] = 1;
   out_748214929053840095[35] = 0;
   out_748214929053840095[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_748214929053840095[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_748214929053840095[38] = 0;
   out_748214929053840095[39] = -state[7];
   out_748214929053840095[40] = state[6];
   out_748214929053840095[41] = 0;
   out_748214929053840095[42] = state[4];
   out_748214929053840095[43] = -state[3];
   out_748214929053840095[44] = 0;
   out_748214929053840095[45] = 0;
   out_748214929053840095[46] = 0;
   out_748214929053840095[47] = 0;
   out_748214929053840095[48] = 0;
   out_748214929053840095[49] = 0;
   out_748214929053840095[50] = 1;
   out_748214929053840095[51] = 0;
   out_748214929053840095[52] = 0;
   out_748214929053840095[53] = 1;
}
void h_13(double *state, double *unused, double *out_6328698996174999134) {
   out_6328698996174999134[0] = state[3];
   out_6328698996174999134[1] = state[4];
   out_6328698996174999134[2] = state[5];
}
void H_13(double *state, double *unused, double *out_6362139137166630416) {
   out_6362139137166630416[0] = 0;
   out_6362139137166630416[1] = 0;
   out_6362139137166630416[2] = 0;
   out_6362139137166630416[3] = 1;
   out_6362139137166630416[4] = 0;
   out_6362139137166630416[5] = 0;
   out_6362139137166630416[6] = 0;
   out_6362139137166630416[7] = 0;
   out_6362139137166630416[8] = 0;
   out_6362139137166630416[9] = 0;
   out_6362139137166630416[10] = 0;
   out_6362139137166630416[11] = 0;
   out_6362139137166630416[12] = 0;
   out_6362139137166630416[13] = 0;
   out_6362139137166630416[14] = 0;
   out_6362139137166630416[15] = 0;
   out_6362139137166630416[16] = 0;
   out_6362139137166630416[17] = 0;
   out_6362139137166630416[18] = 0;
   out_6362139137166630416[19] = 0;
   out_6362139137166630416[20] = 0;
   out_6362139137166630416[21] = 0;
   out_6362139137166630416[22] = 1;
   out_6362139137166630416[23] = 0;
   out_6362139137166630416[24] = 0;
   out_6362139137166630416[25] = 0;
   out_6362139137166630416[26] = 0;
   out_6362139137166630416[27] = 0;
   out_6362139137166630416[28] = 0;
   out_6362139137166630416[29] = 0;
   out_6362139137166630416[30] = 0;
   out_6362139137166630416[31] = 0;
   out_6362139137166630416[32] = 0;
   out_6362139137166630416[33] = 0;
   out_6362139137166630416[34] = 0;
   out_6362139137166630416[35] = 0;
   out_6362139137166630416[36] = 0;
   out_6362139137166630416[37] = 0;
   out_6362139137166630416[38] = 0;
   out_6362139137166630416[39] = 0;
   out_6362139137166630416[40] = 0;
   out_6362139137166630416[41] = 1;
   out_6362139137166630416[42] = 0;
   out_6362139137166630416[43] = 0;
   out_6362139137166630416[44] = 0;
   out_6362139137166630416[45] = 0;
   out_6362139137166630416[46] = 0;
   out_6362139137166630416[47] = 0;
   out_6362139137166630416[48] = 0;
   out_6362139137166630416[49] = 0;
   out_6362139137166630416[50] = 0;
   out_6362139137166630416[51] = 0;
   out_6362139137166630416[52] = 0;
   out_6362139137166630416[53] = 0;
}
void h_14(double *state, double *unused, double *out_4490584305148798359) {
   out_4490584305148798359[0] = state[6];
   out_4490584305148798359[1] = state[7];
   out_4490584305148798359[2] = state[8];
}
void H_14(double *state, double *unused, double *out_7113106168173782144) {
   out_7113106168173782144[0] = 0;
   out_7113106168173782144[1] = 0;
   out_7113106168173782144[2] = 0;
   out_7113106168173782144[3] = 0;
   out_7113106168173782144[4] = 0;
   out_7113106168173782144[5] = 0;
   out_7113106168173782144[6] = 1;
   out_7113106168173782144[7] = 0;
   out_7113106168173782144[8] = 0;
   out_7113106168173782144[9] = 0;
   out_7113106168173782144[10] = 0;
   out_7113106168173782144[11] = 0;
   out_7113106168173782144[12] = 0;
   out_7113106168173782144[13] = 0;
   out_7113106168173782144[14] = 0;
   out_7113106168173782144[15] = 0;
   out_7113106168173782144[16] = 0;
   out_7113106168173782144[17] = 0;
   out_7113106168173782144[18] = 0;
   out_7113106168173782144[19] = 0;
   out_7113106168173782144[20] = 0;
   out_7113106168173782144[21] = 0;
   out_7113106168173782144[22] = 0;
   out_7113106168173782144[23] = 0;
   out_7113106168173782144[24] = 0;
   out_7113106168173782144[25] = 1;
   out_7113106168173782144[26] = 0;
   out_7113106168173782144[27] = 0;
   out_7113106168173782144[28] = 0;
   out_7113106168173782144[29] = 0;
   out_7113106168173782144[30] = 0;
   out_7113106168173782144[31] = 0;
   out_7113106168173782144[32] = 0;
   out_7113106168173782144[33] = 0;
   out_7113106168173782144[34] = 0;
   out_7113106168173782144[35] = 0;
   out_7113106168173782144[36] = 0;
   out_7113106168173782144[37] = 0;
   out_7113106168173782144[38] = 0;
   out_7113106168173782144[39] = 0;
   out_7113106168173782144[40] = 0;
   out_7113106168173782144[41] = 0;
   out_7113106168173782144[42] = 0;
   out_7113106168173782144[43] = 0;
   out_7113106168173782144[44] = 1;
   out_7113106168173782144[45] = 0;
   out_7113106168173782144[46] = 0;
   out_7113106168173782144[47] = 0;
   out_7113106168173782144[48] = 0;
   out_7113106168173782144[49] = 0;
   out_7113106168173782144[50] = 0;
   out_7113106168173782144[51] = 0;
   out_7113106168173782144[52] = 0;
   out_7113106168173782144[53] = 0;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_7351456117776073154) {
  err_fun(nom_x, delta_x, out_7351456117776073154);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_9109558368636035839) {
  inv_err_fun(nom_x, true_x, out_9109558368636035839);
}
void pose_H_mod_fun(double *state, double *out_595012964091947662) {
  H_mod_fun(state, out_595012964091947662);
}
void pose_f_fun(double *state, double dt, double *out_3792261713431509332) {
  f_fun(state,  dt, out_3792261713431509332);
}
void pose_F_fun(double *state, double dt, double *out_6928823179361542013) {
  F_fun(state,  dt, out_6928823179361542013);
}
void pose_h_4(double *state, double *unused, double *out_7683085519451864677) {
  h_4(state, unused, out_7683085519451864677);
}
void pose_H_4(double *state, double *unused, double *out_3149865311834297615) {
  H_4(state, unused, out_3149865311834297615);
}
void pose_h_10(double *state, double *unused, double *out_4760549729285226708) {
  h_10(state, unused, out_4760549729285226708);
}
void pose_H_10(double *state, double *unused, double *out_748214929053840095) {
  H_10(state, unused, out_748214929053840095);
}
void pose_h_13(double *state, double *unused, double *out_6328698996174999134) {
  h_13(state, unused, out_6328698996174999134);
}
void pose_H_13(double *state, double *unused, double *out_6362139137166630416) {
  H_13(state, unused, out_6362139137166630416);
}
void pose_h_14(double *state, double *unused, double *out_4490584305148798359) {
  h_14(state, unused, out_4490584305148798359);
}
void pose_H_14(double *state, double *unused, double *out_7113106168173782144) {
  H_14(state, unused, out_7113106168173782144);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
